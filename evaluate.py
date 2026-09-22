"""
evaluate.py — Evaluation and caption generation for the trained CharLSTM.

Responsibilities
----------------
- Compute held-out loss and perplexity on a validation split.
- Generate captions autoregressively with temperature + top-k sampling.
- Support both text-only and image-conditioned generation.
- Provide a qualitative inspection helper for a set of seed prompts.
"""

import math
import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from data import encode, decode, prepare_all_splits
from lstm import CharLSTM
from train import load_checkpoint


# =============================================================================
# 1. HELD-OUT LOSS & PERPLEXITY
# =============================================================================

@torch.no_grad()
def evaluate_loss(
    model:   CharLSTM,
    loader:  DataLoader,
    encoder: "ResNet50Encoder | None" = None,
) -> tuple[float, float]:
    """
    Compute average cross-entropy loss and perplexity on a data loader.

    PairedDataset yields (images, xb, yb) triples. When an encoder is
    supplied the real image embedding seeds h_0, matching training
    conditions exactly. When encoder=None h_0 defaults to zeros
    (text-only mode).

    Parameters
    ----------
    model   : trained CharLSTM
    loader  : DataLoader from PairedDataset — yields (images, xb, yb)
    encoder : ResNet50Encoder to produce real embeddings; None = text-only
    """
    from encoder import ResNet50Encoder  # local import avoids circular dep

    model.eval()
    if encoder is not None:
        encoder.eval()
    criterion  = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0

    for batch in loader:
        if len(batch) == 3:
            images, xb, yb = batch
        else:
            images, xb, yb = None, batch[0], batch[1]

        xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)

        image_embed = None
        if encoder is not None and images is not None:
            images      = images.to(config.DEVICE)
            image_embed = encoder(images)          # (B, PROJECTION_DIM)

        # image_embed seeds h_0; hidden=None so the model builds h_0 itself
        logits, _ = model(xb, image_embed=image_embed)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            yb.view(-1),
        )
        total_loss += loss.item() * xb.size(0)

    avg_loss   = total_loss / len(loader.dataset)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


# =============================================================================
# 2. CAPTION GENERATION (AUTOREGRESSIVE SAMPLING)
# =============================================================================

@torch.no_grad()
def generate_caption(
    model:       CharLSTM,
    vocab:       dict[str, int],
    idx_to_char: dict[int, str],
    seed_text:   str              = "",
    max_len:     int              = config.GEN_MAX_LEN,
    temperature: float            = config.GEN_TEMPERATURE,
    top_k:       int              = config.GEN_TOP_K,
    image_embed: torch.Tensor | None = None,
) -> str:
    """
    Autoregressively generate one meme caption character by character.

    image_embed : (1, PROJECTION_DIM) from ResNet50Encoder.
                  Passed on the first model call only to seed h_0.
                  Subsequent steps carry (h, c) forward directly.
    """
    model.eval()
    SOS = vocab["<sos>"]
    EOS = vocab["<eos>"]

    current   = [SOS] + encode(seed_text, vocab) if seed_text else [SOS]
    generated = list(seed_text)
    hidden: tuple | None = None

    # Process the seed text to build the initial hidden state.
    # image_embed is consumed here (seeds h_0); subsequent calls use
    # the returned hidden state directly.
    if len(current) > 1:
        seed_tensor = (
            torch.tensor(current[:-1], dtype=torch.long)
            .unsqueeze(0)
            .to(config.DEVICE)
        )
        _, hidden    = model(seed_tensor, image_embed=image_embed)
        image_embed  = None   # h_0 already seeded; don't re-apply

    for _ in range(max_len):
        x = torch.tensor([[current[-1]]], dtype=torch.long).to(config.DEVICE)
        # First step with no seed text: image_embed seeds h_0 here instead.
        logits, hidden = model(x, hidden=hidden, image_embed=image_embed)
        image_embed    = None   # consumed on first step; carry hidden forward

        logits = logits[:, -1, :].squeeze(0)
        logits = logits / max(temperature, 1e-6)
        top_vals, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
        probs   = torch.softmax(top_vals, dim=-1)
        next_id = top_idx[torch.multinomial(probs, 1).item()].item()
        if next_id == EOS:
            break
        current.append(next_id)
        generated.append(idx_to_char.get(next_id, ""))

    return "".join(generated)


# =============================================================================
# 3. QUALITATIVE INSPECTION
# =============================================================================

def inspect_generations(
    model:       CharLSTM,
    vocab:       dict[str, int],
    idx_to_char: dict[int, str],
    seeds:       list[str] | None = None,
    temperature: float = config.GEN_TEMPERATURE,
    top_k:       int   = config.GEN_TOP_K,
) -> None:
    """
    Print generated captions for a list of seed prompts.
    """
    if seeds is None:
        seeds = ["y u no", "when you", "that moment when", "one does not simply", ""]

    print("\n── Generated captions ──────────────────────────────────────")
    for seed in seeds:
        caption = generate_caption(
            model, vocab, idx_to_char,
            seed_text=seed,
            temperature=temperature,
            top_k=top_k,
        )
        print(f"  {repr(seed):30s}  →  {caption}")
    print("────────────────────────────────────────────────────────────\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    model, vocab, idx_to_char = load_checkpoint()

    print("Loading splits for evaluation …")
    train_loader, val_loader, test_loader, _, _ = prepare_all_splits()

    for name, loader in [("train", train_loader),
                          ("val",   val_loader),
                          ("test",  test_loader)]:
        loss, ppl = evaluate_loss(model, loader)
        print(f"  {name:5s}  loss {loss:.4f}  perplexity {ppl:.1f}")

    inspect_generations(model, vocab, idx_to_char)

    print("── Image-conditioned generation (dummy embed) ──")
    dummy_embed = torch.randn(1, config.PROJECTION_DIM).to(config.DEVICE)
    caption = generate_caption(
        model, vocab, idx_to_char,
        seed_text="",
        image_embed=dummy_embed,
    )
    print(f"  → {caption}")