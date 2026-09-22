"""
lstm.py — Character-level LSTM model for meme caption generation.

Responsibilities
----------------
- Define CharLSTM: embedding → stacked LSTM → LayerNorm → linear head.
- Seed the LSTM hidden state h_0 with the image embedding once at the
  start of each sequence. The image sets the initial context; the LSTM
  then generates freely from that starting point.

Image conditioning — design decision
--------------------------------------
The image embedding (B, PROJECTION_DIM) is projected to
(num_layers, B, hidden_size) and used to initialise h_0. c_0 is left
as zeros. This is the standard approach from Vinyals et al. (Show and
Tell, 2015) and is appropriate here because:

  - Meme captions are short (~100 chars), so hidden state drift is
    limited — the image signal remains influential throughout.
  - The encoder produces a single global embedding, not spatial
    features. Per-step concatenation would repeat the same vector at
    every step, adding input complexity without new information.
  - Simpler input size (embed_dim only) makes the LSTM easier to train.

The image_proj layer maps PROJECTION_DIM → hidden_size * num_layers,
then reshapes to (num_layers, B, hidden_size) for direct use as h_0.
"""

import torch
from torch import nn

import config


# =============================================================================
# CHARACTER-LEVEL LSTM
# =============================================================================

class CharLSTM(nn.Module):
    """
    Character-level LSTM language model with h_0 image conditioning.

    Architecture
    ------------
    h_0 = image_proj(image_embed).reshape(num_layers, B, hidden_size)
    c_0 = zeros

    char_embed = Embedding(x)                   (B, T, embed_dim)
        ↓
    nn.LSTM(h_0, c_0) → LayerNorm → Dropout → Linear → logits (B, T, vocab_size)

    Text-only mode (no image_embed supplied): h_0 = c_0 = zeros.

    Parameters
    ----------
    vocab_size      : number of characters in the vocabulary
    embed_dim       : character embedding dimension
    hidden_size     : LSTM hidden state size
    num_layers      : number of stacked LSTM layers
    dropout         : dropout rate (between layers and before head)
    image_embed_dim : ResNet50 encoder output dimension (PROJECTION_DIM).
                      Set to None to disable image conditioning entirely.
    """

    def __init__(
        self,
        vocab_size:      int,
        embed_dim:       int        = config.EMBED_DIM,
        hidden_size:     int        = config.HIDDEN_SIZE,
        num_layers:      int        = config.NUM_LAYERS,
        dropout:         float      = config.DROPOUT,
        image_embed_dim: int | None = config.PROJECTION_DIM,
    ):
        super().__init__()

        self.hidden_size     = hidden_size
        self.num_layers      = num_layers
        self.image_embed_dim = image_embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Projects image embedding → initial hidden state h_0.
        # Output size is hidden_size * num_layers so a single linear layer
        # covers all stacked LSTM layers; reshaped in forward().
        self.image_proj: nn.Sequential | None = (
            nn.Sequential(
                nn.Linear(image_embed_dim, hidden_size * num_layers),
                nn.Tanh(),   # tanh keeps h_0 in the natural LSTM range [-1, 1]
            )
            if image_embed_dim is not None
            else None
        )

        self.lstm = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

        self.head = nn.Linear(hidden_size, vocab_size)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x:           torch.Tensor,
        hidden:      tuple[torch.Tensor, torch.Tensor] | None = None,
        image_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x           : (B, T)                 — character index sequences
        hidden      : optional (h, c) tuple  — carried across generation steps.
                      When provided, image_embed is ignored (the state was
                      already seeded on the first call).
        image_embed : optional (B, image_embed_dim) — from ResNet50Encoder.
                      Used to initialise h_0 on the first call only.
                      Ignored when hidden is already supplied.

        Returns
        -------
        logits : (B, T, vocab_size)
        hidden : updated (h, c) for the next generation step
        """
        emb = self.drop(self.embedding(x))   # (B, T, embed_dim)

        # Build initial hidden state from image embedding if no state is
        # carried in yet. On subsequent generation steps hidden is passed
        # directly and image_embed is not used.
        if hidden is None:
            h_0 = self._make_h0(image_embed, x)
            c_0 = torch.zeros_like(h_0)
            hidden = (h_0, c_0)

        out, hidden = self.lstm(emb, hidden)  # (B, T, hidden_size)

        out    = self.norm(out)
        out    = self.drop(out)
        logits = self.head(out)               # (B, T, vocab_size)

        return logits, hidden

    def _make_h0(
        self,
        image_embed: torch.Tensor | None,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build h_0 of shape (num_layers, B, hidden_size).

        If image_embed is supplied and image_proj exists, project it.
        Otherwise return zeros (text-only mode).
        """
        B = x.size(0)
        if image_embed is not None and self.image_proj is not None:
            h = self.image_proj(image_embed)                        # (B, hidden_size * num_layers)
            h = h.view(B, self.num_layers, self.hidden_size)        # (B, num_layers, hidden_size)
            h = h.permute(1, 0, 2).contiguous()                     # (num_layers, B, hidden_size)
            return h
        return torch.zeros(self.num_layers, B, self.hidden_size,
                           device=x.device, dtype=x.dtype if x.is_floating_point() else torch.float32)

    # ── Convenience ───────────────────────────────────────────────────────────

    def count_parameters(self) -> tuple[int, int]:
        """Return (total_params, trainable_params)."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    VOCAB_SIZE = 100

    model = CharLSTM(vocab_size=VOCAB_SIZE).to(config.DEVICE)
    total, trainable = model.count_parameters()
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")

    dummy_x   = torch.randint(0, VOCAB_SIZE, (8, config.SEQ_LEN)).to(config.DEVICE)
    dummy_img = torch.randn(8, config.PROJECTION_DIM).to(config.DEVICE)

    # Image-conditioned (h_0 seeded)
    logits, _ = model(dummy_x, image_embed=dummy_img)
    print(f"Image-conditioned logits : {logits.shape}")   # (8, SEQ_LEN, 100)

    # Text-only (h_0 = zeros)
    logits, _ = model(dummy_x)
    print(f"Text-only logits         : {logits.shape}")   # (8, SEQ_LEN, 100)

    # Autoregressive step — hidden carried forward, image_embed ignored
    x_step = torch.randint(0, VOCAB_SIZE, (8, 1)).to(config.DEVICE)
    logits_step, hidden = model(dummy_x, image_embed=dummy_img)
    logits_next, _      = model(x_step, hidden=hidden)
    print(f"Step logits shape        : {logits_next.shape}")  # (8, 1, 100)