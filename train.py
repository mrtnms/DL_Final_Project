"""
train.py — Training loop for the image-conditioned character-level LSTM.

Image conditioning
------------------
The ResNet50 encoder produces a (B, PROJECTION_DIM) embedding per batch.
This is passed to CharLSTM as image_embed, which the model uses to seed
h_0 (see lstm.py). The backbone remains frozen; only the projection head
and LSTM receive gradients.

Checkpointing
-------------
Two checkpoints are maintained at all times:

  CHECKPOINT_PATH        (best)    saved only when val loss improves.
                                   Use this for inference and deployment.

  CHECKPOINT_LATEST_PATH (latest)  saved at the end of every epoch.
                                   Use --resume with this path to continue
                                   an interrupted run from the last completed
                                   epoch rather than the last best epoch.
"""

import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from data import prepare_all_splits
from encoder import ResNet50Encoder
from lstm import CharLSTM

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(
    model:                  CharLSTM,
    encoder:                ResNet50Encoder,
    train_loader:           DataLoader,
    val_loader:             DataLoader,
    epochs:                 int        = config.EPOCHS,
    lr:                     float      = config.LR,
    clip:                   float      = config.GRAD_CLIP,
    checkpoint_path:        str        = config.CHECKPOINT_PATH,
    checkpoint_latest_path: str        = config.CHECKPOINT_LATEST_PATH,
    vocab:                  dict | None = None,
    idx_to_char:            dict | None = None,
    resume_from:            str | None  = None,
) -> dict[str, list[float]]:
    """
    Train CharLSTM with image embeddings from ResNet50Encoder seeding h_0.

    Parameters
    ----------
    checkpoint_path        : destination for the best-val-loss checkpoint.
    checkpoint_latest_path : destination for the end-of-every-epoch checkpoint.
    resume_from            : path to a checkpoint to resume from (typically
                             checkpoint_latest_path from a previous run).
                             Pass None to start from scratch.
    """
    trainable_params = (
        list(encoder.projection.parameters()) +
        list(model.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    criterion     = nn.CrossEntropyLoss(ignore_index=0)
    history       = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    start_epoch   = 1

    # ── Resume ────────────────────────────────────────────────────────────────
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=config.DEVICE)

        model.load_state_dict(ckpt["model_state"])
        encoder.projection.load_state_dict(ckpt["encoder_projection_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        history       = ckpt.get("history", history)

        print(
            f"  Resumed at epoch {start_epoch}/{epochs}"
            f"  |  best val loss so far: {best_val_loss:.4f}\n"
        )

    if start_epoch > epochs:
        print("Training already complete for the requested number of epochs.")
        return history

    # Backbone stays in eval mode — BatchNorm and Dropout behave correctly
    # only in eval mode when weights are frozen.
    encoder.backbone.eval()

    for epoch in range(start_epoch, epochs + 1):

        # ── Training pass ────────────────────────────────────────────────────
        model.train()
        encoder.projection.train()
        total_train_loss = 0.0

        for images, xb, yb in train_loader:
            images = images.to(config.DEVICE)
            xb     = xb.to(config.DEVICE)
            yb     = yb.to(config.DEVICE)

            with torch.no_grad():
                features = encoder.backbone(images)       # (B, 2048, 1, 1)

            image_embed = encoder.projection(features)    # (B, PROJECTION_DIM)
            logits, _   = model(xb, image_embed=image_embed)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                yb.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            total_train_loss += loss.item() * xb.size(0)

        avg_train = total_train_loss / len(train_loader.dataset)

        # ── Validation pass ──────────────────────────────────────────────────
        model.eval()
        encoder.projection.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for images, xb, yb in val_loader:
                images = images.to(config.DEVICE)
                xb     = xb.to(config.DEVICE)
                yb     = yb.to(config.DEVICE)

                features    = encoder.backbone(images)
                image_embed = encoder.projection(features)
                logits, _   = model(xb, image_embed=image_embed)

                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    yb.view(-1),
                )
                total_val_loss += loss.item() * xb.size(0)

        avg_val = total_val_loss / len(val_loader.dataset)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val

        print(
            f"epoch {epoch:02d}/{epochs}"
            f"  |  train loss {avg_train:.4f}  ppl {math.exp(avg_train):.1f}"
            f"  |  val loss {avg_val:.4f}  ppl {math.exp(avg_val):.1f}"
            + ("  best" if is_best else "")
        )

        # Always save latest so a resume picks up from this exact epoch.
        if vocab is not None:
            save_checkpoint(
                model, encoder, optimizer,
                vocab, idx_to_char,
                epoch, best_val_loss, history,
                checkpoint_latest_path,
            )

        # Save best separately so inference always has the strongest weights.
        if is_best and vocab is not None:
            save_checkpoint(
                model, encoder, optimizer,
                vocab, idx_to_char,
                epoch, best_val_loss, history,
                checkpoint_path,
            )

    return history


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(
    model:           CharLSTM,
    encoder:         ResNet50Encoder,
    optimizer:       torch.optim.Optimizer,
    vocab:           dict[str, int],
    idx_to_char:     dict[int, str],
    epoch:           int,
    best_val_loss:   float,
    history:         dict[str, list[float]],
    path:            str,
) -> None:
    """
    Save everything needed to resume training or run inference.

    Both the best and latest checkpoints share this format so either
    can be passed to --resume. The best checkpoint is also consumed by
    load_checkpoint() for inference in evaluate.py and deploy.py.
    """
    torch.save(
        {
            # ── Inference keys ───────────────────────────────────────────────
            "model_state": model.state_dict(),
            "vocab":        vocab,
            "idx_to_char":  idx_to_char,
            "config": {
                # Read directly from the model so the saved config always
                # matches the actual weights, even if config.py was changed
                # mid-run.
                "vocab_size":      model.head.out_features,
                "embed_dim":       model.embedding.embedding_dim,
                "hidden_size":     model.hidden_size,
                "num_layers":      model.num_layers,
                "dropout":         config.DROPOUT,
                "image_embed_dim": model.image_embed_dim,
            },
            # ── Resume keys ──────────────────────────────────────────────────
            "encoder_projection_state": encoder.projection.state_dict(),
            "optimizer_state":          optimizer.state_dict(),
            "epoch":                    epoch,
            "best_val_loss":            best_val_loss,
            "history":                  history,
        },
        path,
    )
    print(f"  Checkpoint saved  ->  {path}  (epoch {epoch})")


def load_checkpoint(
    path: str = config.CHECKPOINT_PATH,
) -> tuple[CharLSTM, dict[str, int], dict[int, str]]:
    """
    Load model weights for inference. Does not restore optimizer state.

    Always point this at CHECKPOINT_PATH (best) for inference and
    deployment. Use --resume with CHECKPOINT_LATEST_PATH to continue
    training.
    """
    ckpt = torch.load(path, map_location=config.DEVICE)
    cfg  = ckpt["config"]
    model = CharLSTM(
        vocab_size      = cfg["vocab_size"],
        embed_dim       = cfg["embed_dim"],
        hidden_size     = cfg["hidden_size"],
        num_layers      = cfg["num_layers"],
        dropout         = cfg["dropout"],
        image_embed_dim = cfg.get("image_embed_dim"),
    ).to(config.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Checkpoint loaded  <-  {path}")
    return model, ckpt["vocab"], ckpt["idx_to_char"]


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", type=str, default=None,
        help=(
            "Path to a checkpoint to resume training from. "
            "Typically char_lstm_checkpoint_latest.pt from a previous run."
        ),
    )
    args = parser.parse_args()

    print(f"Device: {config.DEVICE}\n")

    train_loader, val_loader, test_loader, vocab, idx_to_char = prepare_all_splits()

    encoder = ResNet50Encoder(
        projection_dim  = config.PROJECTION_DIM,
        freeze_backbone = True,
    ).to(config.DEVICE)

    model = CharLSTM(vocab_size=len(vocab)).to(config.DEVICE)
    total, trainable = model.count_parameters()
    print(f"Encoder trainable params : {sum(p.numel() for p in encoder.projection.parameters()):,}")
    print(f"LSTM total params        : {total:,}\n")

    history = train(
        model, encoder, train_loader, val_loader,
        vocab=vocab, idx_to_char=idx_to_char,
        resume_from=args.resume,
    )

    torch.save(history, "history.pt")
    print("History saved -> history.pt")
    print("Training complete. Run evaluate.py for test-set results.")