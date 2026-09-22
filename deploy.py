"""
deploy.py — Deployment interface for the meme caption generator.

Responsibilities
----------------
- Load a trained checkpoint and expose a clean generate() API.
- Optionally load the ResNet50 encoder for image-conditioned generation.
- Provide a command-line interface for quick interactive use.
- Produce a batch of diverse captions by varying temperature and top-k.

This module has NO training logic.  It imports only what is needed for
inference: the model definition, the encoder, and the generation function.
"""

import argparse
import torch
from pathlib import Path

import config
from encoder import ResNet50Encoder, encode_transform
from lstm import CharLSTM
from train import load_checkpoint
from evaluate import generate_caption


# =============================================================================
# 1. GENERATOR CLASS  (clean API for downstream use)
# =============================================================================

class MemeGenerator:
    """
    High-level wrapper around CharLSTM for deployment.

    Usage
    -----
    # Text-only
    gen = MemeGenerator.from_checkpoint("char_lstm_checkpoint.pt")
    print(gen.generate("y u no"))

    # Image-conditioned (requires a meme image as a PIL Image or tensor)
    gen = MemeGenerator.from_checkpoint("char_lstm_checkpoint.pt",
                                        load_encoder=True)
    caption = gen.generate_from_image(pil_image)
    """

    def __init__(
        self,
        model:       CharLSTM,
        vocab:       dict[str, int],
        idx_to_char: dict[int, str],
        encoder:     ResNet50Encoder | None = None,
    ):
        self.model       = model
        self.vocab       = vocab
        self.idx_to_char = idx_to_char
        self.encoder     = encoder

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        path:         str  = config.CHECKPOINT_PATH,
        load_encoder: bool = False,
    ) -> "MemeGenerator":
        """
        Build a MemeGenerator from a saved checkpoint.

        Parameters
        ----------
        path         : path to the .pt checkpoint produced by train.py
        load_encoder : also load ResNet50Encoder for image conditioning
        """
        model, vocab, idx_to_char = load_checkpoint(path)

        encoder = None
        if load_encoder:
            encoder = ResNet50Encoder(
                projection_dim=config.PROJECTION_DIM,
                freeze_backbone=True,
            ).to(config.DEVICE)
            encoder.eval()
            print("ResNet50 encoder loaded (frozen backbone).")

        return cls(model, vocab, idx_to_char, encoder)

    # ── Text-only generation ─────────────────────────────────────────────────

    def generate(
        self,
        seed:        str   = "",
        temperature: float = config.GEN_TEMPERATURE,
        top_k:       int   = config.GEN_TOP_K,
        max_len:     int   = config.GEN_MAX_LEN,
    ) -> str:
        """Generate a caption from an optional text seed."""
        return generate_caption(
            self.model, self.vocab, self.idx_to_char,
            seed_text=seed,
            temperature=temperature,
            top_k=top_k,
            max_len=max_len,
        )

    # ── Image-conditioned generation ─────────────────────────────────────────

    @torch.no_grad()
    def generate_from_image(
        self,
        image,                          # PIL Image or (3, H, W) tensor
        seed:        str   = "",
        temperature: float = config.GEN_TEMPERATURE,
        top_k:       int   = config.GEN_TOP_K,
        max_len:     int   = config.GEN_MAX_LEN,
    ) -> str:
        """
        Generate a caption conditioned on a meme image.

        The image is passed through ResNet50Encoder to produce a
        (1, PROJECTION_DIM) embedding that seeds the LSTM hidden state.
        Requires the generator to have been built with load_encoder=True.
        """
        if self.encoder is None:
            raise RuntimeError(
                "No encoder loaded. Rebuild with "
                "MemeGenerator.from_checkpoint(load_encoder=True)."
            )

        # Preprocess: accept PIL Image or raw tensor
        if not isinstance(image, torch.Tensor):
            image = encode_transform(image)     # (3, 224, 224)

        if image.dim() == 3:
            image = image.unsqueeze(0)          # (1, 3, 224, 224)

        image_embed = self.encoder(image.to(config.DEVICE))   # (1, PROJECTION_DIM)

        return generate_caption(
            self.model, self.vocab, self.idx_to_char,
            seed_text=seed,
            temperature=temperature,
            top_k=top_k,
            max_len=max_len,
            image_embed=image_embed,
        )

    # ── Batch generation with diversity ──────────────────────────────────────

    def generate_batch(
        self,
        seed:         str        = "",
        n:            int        = 5,
        temperatures: list[float] | None = None,
        top_k:        int        = config.GEN_TOP_K,
    ) -> list[str]:
        """
        Generate n captions for the same seed at varying temperatures.

        Useful for picking the best caption from a diverse set, or for
        data augmentation during further fine-tuning.

        Parameters
        ----------
        seed         : optional text prompt
        n            : number of captions to generate
        temperatures : list of n temperatures; defaults to a linspace
                       from 0.5 (conservative) to 1.2 (creative)
        top_k        : top-k filtering applied at every step
        """
        if temperatures is None:
            import numpy as np
            temperatures = list(np.linspace(0.5, 1.2, n))

        return [
            self.generate(seed=seed, temperature=t, top_k=top_k)
            for t in temperatures
        ]


# =============================================================================
# 2. COMMAND-LINE INTERFACE
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate meme captions with a trained CharLSTM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, default=config.CHECKPOINT_PATH,
        help="Path to the .pt checkpoint file produced by train.py.",
    )
    parser.add_argument(
        "--seed", type=str, default="",
        help="Optional text seed to start caption generation.",
    )
    parser.add_argument(
        "--n", type=int, default=5,
        help="Number of captions to generate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=config.GEN_TEMPERATURE,
        help="Sampling temperature (higher = more random).",
    )
    parser.add_argument(
        "--top_k", type=int, default=config.GEN_TOP_K,
        help="Top-k character candidates to sample from.",
    )
    parser.add_argument(
        "--max_len", type=int, default=config.GEN_MAX_LEN,
        help="Maximum caption length in characters.",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Optional path to a meme image for image-conditioned generation.",
    )
    return parser


def main() -> None:
    args   = build_arg_parser().parse_args()
    use_img = args.image is not None and Path(args.image).is_file()

    gen = MemeGenerator.from_checkpoint(
        path=args.checkpoint,
        load_encoder=use_img,
    )

    print(f"\nDevice      : {config.DEVICE}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Seed        : {repr(args.seed)}")
    print(f"Temperature : {args.temperature}  |  top_k : {args.top_k}\n")

    if use_img:
        from PIL import Image
        pil_image = Image.open(args.image).convert("RGB")
        print(f"Image       : {args.image}\n")
        caption = gen.generate_from_image(
            pil_image,
            seed=args.seed,
            temperature=args.temperature,
            top_k=args.top_k,
            max_len=args.max_len,
        )
        print(f"Caption     : {caption}\n")
    else:
        captions = gen.generate_batch(
            seed=args.seed,
            n=args.n,
            top_k=args.top_k,
        )
        print("Generated captions:")
        for i, cap in enumerate(captions, 1):
            print(f"  [{i}] {cap}")
        print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()