"""
config.py — Central configuration for the meme caption generator pipeline.

Every other module imports from here. Change a value once; it propagates
to data loading, model construction, training, evaluation, and deployment.
"""

import torch

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 7

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ── Paths ──────────────────────────────────────────────────────────────────────
# Caption files — tab-separated: meme_name 	 score 	 caption
# The meme_name column is used to look up the corresponding image file.
CAPTION_TRAIN   = "memes900k/captions_train.txt"
CAPTION_VAL     = "memes900k/captions_val.txt"
CAPTION_TEST    = "memes900k/captions_test.txt"

# Folder containing one image per meme type.
# Image filename = meme_name + IMAGE_EXT  (e.g. "Y U No.jpg")
IMAGE_DIR       = "memes900k/images"
IMAGE_EXT       = ".jpg"   # change to ".png" if your images are PNG

# Best checkpoint — saved only when validation loss improves.
# Use this for inference and deployment.
CHECKPOINT_PATH = "char_lstm_checkpoint.pt"

# Latest checkpoint — saved at the end of every epoch regardless of val loss.
# Use this to resume an interrupted training run.
CHECKPOINT_LATEST_PATH = "char_lstm_checkpoint_latest.pt"

EMBEDDINGS_PATH = "image_embeddings.pt"

# ── Data ───────────────────────────────────────────────────────────────────────
# SEQ_LEN in characters. p99 of your dataset = 95 chars; 100 gives headroom.
SEQ_LEN    = 100
BATCH_SIZE = 16   # reduced from 64: each batch now also loads images

# ── Encoder (ResNet50) ─────────────────────────────────────────────────────────
IMAGE_SIZE      = 224
PROJECTION_DIM  = 256
FREEZE_BACKBONE = True

# ── LSTM ───────────────────────────────────────────────────────────────────────
EMBED_DIM   = 64
HIDDEN_SIZE = 256
NUM_LAYERS  = 2
DROPOUT     = 0.3

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS    = 10
LR        = 1e-3
GRAD_CLIP = 5.0

# ── Generation ─────────────────────────────────────────────────────────────────
GEN_MAX_LEN     = 100
GEN_TEMPERATURE = 0.8
GEN_TOP_K       = 10


if __name__ == "__main__":
    ckpt = torch.load("char_lstm_checkpoint_latest.pt", map_location="cpu")
    print(ckpt["config"])
