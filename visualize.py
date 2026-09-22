"""
visualise.py — Visual performance evaluation for the meme caption generator.

Run this after training to judge model quality through four lenses:

    1. Loss & perplexity curves   — did training converge? is there overfitting?
    2. Temperature sweep          — how sharp / creative is the learned distribution?
    3. Character probability heatmap — has the model learned word structure?
    4. Image-conditioned generation — run the full pipeline on your own image.

Usage
-----
    # After training (text-only evaluation):
    python visualise.py

    # With your own image:
    python visualise.py --image path/to/your_meme.jpg

    # --seed is NOT needed when supplying an image: the image drives generation.
    # Use --seed only for text-only evaluation (no image), e.g. to prime the
    # heatmap and temperature sweep with a common prompt for comparison.
"""

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

import config
from data import encode, decode, prepare_all_splits
from encoder import ResNet50Encoder, encode_transform
from lstm import CharLSTM
from train import load_checkpoint
from evaluate import generate_caption, evaluate_loss

matplotlib.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":   120,
})


# =============================================================================
# 1. LOSS & PERPLEXITY CURVES
# =============================================================================

def plot_training_curves(
    history: dict[str, list[float]],
    ax: plt.Axes,
) -> None:
    """
    Plot train and validation loss over epochs on a shared axis.

    What to look for
    ----------------
    - Both curves trending down       → model is learning.
    - Val curve plateauing early      → overfitting; consider more dropout or
                                        early stopping.
    - Val curve below train           → healthy; dropout is regularising well.
    - Val curve spiking then settling → learning rate may be too high early on.

    A secondary y-axis shows perplexity (exp(loss)) — the more intuitive
    metric: perplexity of N means the model is as uncertain as if it were
    choosing uniformly among N characters at every step.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    color_train = "#185FA5"   # blue
    color_val   = "#993C1D"   # coral

    ax.plot(epochs, history["train_loss"], color=color_train,
            linewidth=2, label="train loss", marker="o", markersize=4)
    ax.plot(epochs, history["val_loss"],   color=color_val,
            linewidth=2, label="val loss",   marker="o", markersize=4, linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training & validation loss", fontweight="medium")
    ax.legend(frameon=False)

    # Secondary axis: perplexity
    ax2 = ax.twinx()
    ax2.set_ylabel("Perplexity", color="#555")
    ppl_ticks = [ax.get_yticks()[0], ax.get_yticks()[-1]]
    ax2.set_yticks([math.exp(v) for v in ppl_ticks if v > 0])
    ax2.tick_params(axis="y", labelcolor="#555")
    ax2.spines["top"].set_visible(False)

    # Annotate final values
    final_train = history["train_loss"][-1]
    final_val   = history["val_loss"][-1]
    ax.annotate(f"{final_train:.3f}", xy=(epochs[-1], final_train),
                xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=color_train)
    ax.annotate(f"{final_val:.3f}", xy=(epochs[-1], final_val),
                xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=color_val)


# =============================================================================
# 2. TEMPERATURE SWEEP
# =============================================================================

def plot_temperature_sweep(
    model:       CharLSTM,
    vocab:       dict[str, int],
    idx_to_char: dict[int, str],  # FIXED: was idx_to_word (word-level leftover)
    seed:        str,
    ax:          plt.Axes,
    image_embed: torch.Tensor | None = None,
) -> None:
    """
    Generate captions at six temperatures from conservative to wild.

    What to look for
    ----------------
    - Low temperature (0.3–0.5)  → model's most confident output. Should be
                                   grammatically correct but possibly generic.
    - Mid temperature (0.7–0.9)  → the sweet spot for meme generation.
    - High temperature (1.1–1.3) → creative but may degrade into noise.

    If even low temperature produces garbled text, the model has not yet
    learned the character-level structure of English — train longer or on
    more data.  If high temperature looks identical to low temperature, the
    model may have collapsed to a narrow distribution — try increasing dropout
    or reducing the learning rate.
    """
    temperatures = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    captions     = []

    for t in temperatures:
        cap = generate_caption(
            model, vocab, idx_to_char,  # FIXED: was idx_to_word
            seed_text=seed, temperature=t, top_k=10,
            image_embed=image_embed,
        )
        captions.append(cap)

    ax.axis("off")
    ax.set_title(f"Temperature sweep  (seed: {repr(seed)})", fontweight="medium")

    row_colors = ["#EAF3DE", "#EAF3DE", "#E6F1FB", "#E6F1FB", "#FAECE7", "#FAECE7"]
    table_data = [[f"t = {t:.1f}", cap[:80] + ("…" if len(cap) > 80 else "")]
                  for t, cap in zip(temperatures, captions)]

    tbl = ax.table(
        cellText=table_data,
        colLabels=["temperature", "generated caption"],
        cellLoc="left",
        loc="center",
        colWidths=[0.15, 0.85],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.2)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ddd")
        if row == 0:
            cell.set_facecolor("#333")
            cell.set_text_props(color="white", fontweight="medium")
        elif col == 0:
            cell.set_facecolor(row_colors[row - 1])
            cell.set_text_props(fontweight="medium")
        else:
            cell.set_facecolor(row_colors[row - 1])


# =============================================================================
# 3. CHARACTER PROBABILITY HEATMAP
# =============================================================================

@torch.no_grad()
def plot_char_heatmap(
    model:       CharLSTM,
    vocab:       dict[str, int],
    idx_to_char: dict[int, str],  # FIXED: was idx_to_word (word-level leftover)
    seed:        str,
    ax:          plt.Axes,
    n_steps:     int = 30,
    top_n:       int = 15,
) -> None:
    """
    Show the top-N character probabilities at each generation step.

    The heatmap rows are characters; columns are time steps.  Cell colour
    encodes the softmax probability at temperature=1.0 (no distortion).

    What to look for
    ----------------
    - After a space, the model should assign high probability to common
      first letters (t, a, i, w, …), not random characters.
    - The model should predict space (' ') with high confidence after
      most word-length sequences — this indicates it has learned word
      boundaries.
    - A very flat heatmap (all cells similar brightness) means the model
      is still uncertain and would benefit from more training.
    - A very spiky heatmap (one cell dominates each column) is good at
      low temperature but suggests the model may be overconfident.
    """
    model.eval()
    SOS = vocab["<sos>"]
    EOS = vocab["<eos>"]

    # Prime hidden state on the seed
    # FIXED: char-level encodes string directly — no tokenise() call needed
    current = [SOS] + encode(seed, vocab) if seed else [SOS]
    hidden  = None

    if len(current) > 1:
        seed_tensor = torch.tensor(current[:-1]).unsqueeze(0).to(config.DEVICE)
        _, hidden   = model(seed_tensor)

    # Collect top-N probs at each step
    all_probs:  list[np.ndarray] = []
    all_labels: list[list[str]]  = []
    generated   = list(seed)  # FIXED: was list(tokenize(seed)) — char-level uses list(string)

    for step in range(n_steps):
        x      = torch.tensor([[current[-1]]]).to(config.DEVICE)
        logits, hidden = model(x, hidden=hidden)
        probs  = torch.softmax(logits[:, -1, :].squeeze(0), dim=-1)

        top_probs, top_idx = torch.topk(probs, top_n)
        next_id = top_idx[0].item()
        if next_id == EOS:
            break

        top_probs = top_probs.cpu().numpy()
        top_chars = [idx_to_char.get(i.item(), "?") for i in top_idx]  # FIXED: was idx_to_word

        all_probs.append(top_probs)
        all_labels.append(top_chars)

        # Greedy decode for the column label (what the model actually picked)


        current.append(next_id)
        generated.append(idx_to_char.get(next_id, ""))  # FIXED: was idx_to_word

    n_steps_actual = len(all_probs)

    # Build matrix: rows = rank, columns = time step
    # Each column is independently normalised to [0, 1] for visual clarity
    matrix      = np.zeros((top_n, n_steps_actual))
    # FIXED: renamed word_matrix → char_matrix (char-level)
    char_matrix: list[list[str]] = [[""] * n_steps_actual for _ in range(top_n)]

    for col, (probs, chars) in enumerate(zip(all_probs, all_labels)):
        matrix[:, col]    = probs / (probs.max() + 1e-9)   # normalise column
        for row, ch in enumerate(chars):
            char_matrix[row][col] = ch if ch != " " else "·"  # FIXED: char-level display

    # FIXED: char-level — show single character per step
    col_labels = []
    for i, ch in enumerate(generated[:n_steps_actual]):
        col_labels.append(repr(ch)[1:-1] if ch != " " else "·")

    im = ax.imshow(matrix, aspect="auto", cmap="Blues",
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_title(f"Character probability heatmap  (greedy: '{(''.join(generated[:n_steps_actual]))}')",
                 fontweight="medium")
    ax.set_xlabel("Generation step (greedy character shown)")
    ax.set_ylabel("Character rank")
    ax.set_xticks(range(n_steps_actual))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=0)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"#{i+1}" for i in range(top_n)], fontsize=8)

    # Annotate each cell with (truncated) word
    for row in range(top_n):
        for col in range(n_steps_actual):
            ch = char_matrix[row][col]  # FIXED: was word_matrix
            if ch:
                brightness = matrix[row, col]
                color = "white" if brightness > 0.55 else "#333"
                ax.text(col, row, ch, ha="center", va="center",
                        fontsize=7, color=color)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02,
                 label="relative prob (col-normalised)")


# =============================================================================
# 4. IMAGE-CONDITIONED GENERATION
# =============================================================================

def load_and_encode_image(
    image_path: str,
    encoder:    ResNet50Encoder,
) -> tuple[torch.Tensor, object]:
    """
    Load a local image file, run it through the ResNet50 encoder,
    and return both the embedding vector and the PIL image for display.

    Accepts: JPEG, PNG, BMP, WEBP — anything PIL can open.
    The encode_transform (resize to 224, ImageNet normalisation) is
    applied automatically.

    Returns
    -------
    image_embed : (1, PROJECTION_DIM) tensor on config.DEVICE
    pil_image   : original PIL image for display alongside the caption
    """
    from PIL import Image

    pil_image = Image.open(image_path).convert("RGB")
    tensor    = encode_transform(pil_image).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        image_embed = encoder(tensor)   # (1, PROJECTION_DIM)

    return image_embed, pil_image


def plot_image_generation(
    model:       CharLSTM,
    vocab:       dict[str, int],
    idx_to_char: dict[int, str],  # FIXED: was idx_to_word (word-level leftover)
    image_path:  str,
    seed:        str,
    ax_img:      plt.Axes,
    ax_caps:     plt.Axes,
) -> None:
    """
    Display the input image and a set of generated captions side by side.

    Generates five captions at temperatures [0.5, 0.7, 0.9, 1.0, 1.2]
    so you can compare conservative and creative outputs for the same image.
    """
    encoder = ResNet50Encoder(
        projection_dim = config.PROJECTION_DIM,
        freeze_backbone = True,
    ).to(config.DEVICE)
    encoder.eval()

    image_embed, pil_image = load_and_encode_image(image_path, encoder)

    # Display the image
    ax_img.imshow(pil_image)
    ax_img.axis("off")
    ax_img.set_title(Path(image_path).name, fontweight="medium", fontsize=10)

    # Generate captions at a range of temperatures
    temperatures = [0.5, 0.7, 0.9, 1.0, 1.2]
    captions     = []
    for t in temperatures:
        cap = generate_caption(
            model, vocab, idx_to_char,  # FIXED: was idx_to_word
            seed_text   = seed,
            temperature = t,
            top_k       = 10,
            image_embed = image_embed.clone(),
        )
        captions.append((t, cap))

    # Display captions as a clean table
    ax_caps.axis("off")
    ax_caps.set_title("Image-conditioned captions", fontweight="medium", fontsize=10)

    y = 0.95
    for t, cap in captions:
        label = f"t={t:.1f}"
        ax_caps.text(0.0, y, label, transform=ax_caps.transAxes,
                     fontsize=9, fontweight="medium", color="#185FA5",
                     va="top")
        ax_caps.text(0.12, y, cap, transform=ax_caps.transAxes,
                     fontsize=9, color="#222", va="top",
                     wrap=True, clip_on=True)
        y -= 0.18


# =============================================================================
# 5. MAIN DASHBOARD
# =============================================================================

def run_dashboard(
    history:    dict[str, list[float]] | None = None,
    image_path: str | None = None,
    seed:       str = "",
) -> None:
    """
    Compose and display the full evaluation dashboard.

    Layout (with image)         Layout (text-only)
    ─────────────────────       ──────────────────
    [loss curves] [image]       [loss curves      ]
    [temp sweep        ]        [temp sweep       ]
    [heatmap           ]        [heatmap          ]
    """
    model, vocab, idx_to_char = load_checkpoint()  # FIXED: was idx_to_word

    has_image  = image_path is not None and Path(image_path).is_file()
    has_curves = history is not None and len(history.get("train_loss", [])) > 0

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 13) if has_image else (13, 13))
    fig.suptitle("Meme caption generator — visual evaluation",
                 fontsize=14, fontweight="medium", y=0.98)

    if has_image:
        gs = gridspec.GridSpec(3, 2, figure=fig,
                               height_ratios=[1, 0.9, 1.1],
                               hspace=0.45, wspace=0.3)
        ax_loss = fig.add_subplot(gs[0, 0])
        ax_img  = fig.add_subplot(gs[0, 1])
        ax_caps = fig.add_subplot(gs[1, :])   # image captions span full width
        ax_temp = fig.add_subplot(gs[2, :])   # temp sweep was dropped: use ax_caps
        # Re-assign: row 1 = temp sweep, row 2 = heatmap
        ax_caps_real = fig.add_subplot(gs[1, :])
        ax_temp      = fig.add_subplot(gs[2, :])

        # Cleaner: use explicit subplot_mosaic
        fig.clear()
        axd = fig.subplot_mosaic(
            [["loss", "image"],
             ["caps",  "caps"],
             ["temp",  "temp"],
             ["heat",  "heat"]],
            height_ratios=[1.1, 0.9, 0.9, 1.1],
        )
        ax_loss  = axd["loss"]
        ax_image = axd["image"]
        ax_caps  = axd["caps"]
        ax_temp  = axd["temp"]
        ax_heat  = axd["heat"]
    else:
        fig.clear()
        axd = fig.subplot_mosaic(
            [["loss"],
             ["temp"],
             ["heat"]],
            height_ratios=[1, 0.9, 1.1],
        )
        ax_loss = axd["loss"]
        ax_temp = axd["temp"]
        ax_heat = axd["heat"]
        ax_image = None
        ax_caps  = None

    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    fig.suptitle("Meme caption generator — visual evaluation",
                 fontsize=14, fontweight="medium")

    # ── 1. Loss curves ────────────────────────────────────────────────────────
    if has_curves:
        plot_training_curves(history, ax_loss)
    else:
        ax_loss.axis("off")
        ax_loss.text(0.5, 0.5,
                     "No training history.\nPass history= to run_dashboard().",
                     ha="center", va="center", transform=ax_loss.transAxes,
                     color="#999", fontsize=10)
        ax_loss.set_title("Training & validation loss", fontweight="medium")

    # ── 2. Image + captions ───────────────────────────────────────────────────
    if has_image:
        plot_image_generation(model, vocab, idx_to_char,  # FIXED: was idx_to_word
                               image_path, seed, ax_image, ax_caps)

    # ── 3. Temperature sweep ─────────────────────────────────────────────────
    plot_temperature_sweep(model, vocab, idx_to_char, seed, ax_temp)  # FIXED: was idx_to_word

    # ── 4. Character heatmap ─────────────────────────────────────────────────
    plot_char_heatmap(model, vocab, idx_to_char, seed, ax_heat)  # FIXED: was idx_to_word

    plt.savefig("evaluation_dashboard.png", dpi=150, bbox_inches="tight")
    print("Dashboard saved → evaluation_dashboard.png")
    plt.show()


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual evaluation dashboard for the meme caption generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a meme image for image-conditioned generation.")
    parser.add_argument("--seed",  type=str, default="",
                        help="Text seed for temperature sweep / heatmap (text-only). When --image is given, leave this empty — the image drives generation.")
    parser.add_argument("--history", type=str, default=None,
                        help="Path to a saved history .pt file "
                             "(saved automatically by train.py if you add "
                             "torch.save(history, 'history.pt')).")
    args = parser.parse_args()

    history = None
    if args.history and Path(args.history).is_file():
        history = torch.load(args.history, map_location="cpu")

    run_dashboard(
        history    = history,
        image_path = args.image,
        seed       = args.seed,
    )