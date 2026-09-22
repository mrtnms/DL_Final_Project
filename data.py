"""
data.py — Data loading and preprocessing for the meme caption generator.

Key design decisions
---------------------
- Each caption is paired with its meme image via the meme_name in column 1
  of the TSV.  The image filename is meme_name + IMAGE_EXT inside IMAGE_DIR.
- Vocabulary is built from training captions only (no leakage).
- Character-level tokenisation: every character is a token, vocab ~100.
- PairedDataset loads (image_tensor, x, y) triples so the training loop
  receives real image embeddings instead of zero vectors.
- Images that cannot be found on disk are skipped with a warning rather
  than crashing — some meme names may contain characters invalid for
  filenames on certain operating systems.
"""

import os
import re
import numpy as np
import torch
from collections import Counter
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import config

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

# ── ImageNet normalisation required by frozen ResNet50 backbone ────────────────
encode_transform = T.Compose([
    T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])




# =============================================================================
# 0. SLUGIFY
# =============================================================================

def slugify(name: str) -> str:
    """
    Convert a meme name from the TSV into the slug format used as the
    image filename on disk.

    TSV column 1 : "Y U No"
    Image file   : "y-u-no.jpg"

    Rules applied in order:
        1. Lowercase
        2. Replace one or more whitespace characters with a single hyphen
        3. Strip any characters that are not alphanumeric or hyphens
           (handles punctuation like apostrophes, parentheses, etc.)
        4. Collapse multiple consecutive hyphens into one
        5. Strip leading/trailing hyphens
    """
    name = name.lower()
    name = re.sub(r'\s+', '-', name)
    name = re.sub(r'[^a-z0-9-]', '', name)
    name = re.sub(r'-+', '-', name)
    return name.strip('-')

# =============================================================================
# 1. RAW CAPTION LOADING
# =============================================================================

def load_captions(path: str) -> list[tuple[str, str]]:
    """
    Parse one meme caption TSV file entirely.

    Format (tab-separated):
        meme_name \t score \t top_text <sep> bottom_text

    Returns a list of (meme_name, caption_text) tuples.
    meme_name is the stem of the corresponding image filename.
    """
    pairs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            meme_name = parts[0].strip()
            text      = parts[2].replace("<sep>", " ").replace("<emp>", "").strip()
            if len(text) > 4:
                pairs.append((meme_name, text))
    print(f"  {len(pairs):>8,} captions  <-  {path}")
    return pairs


# =============================================================================
# 2. CHARACTER-LEVEL VOCABULARY
# =============================================================================

def build_char_vocab(
    train_pairs: list[tuple[str, str]],
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Build char -> index mapping from TRAINING captions only.

    Special tokens:
        <pad>  0  padding / OOV fallback
        <sos>  1  start of sequence
        <eos>  2  end of sequence
    """
    counter: Counter = Counter()
    for _, caption in train_pairs:
        counter.update(caption)

    special = ["<pad>", "<sos>", "<eos>"]
    vocab: dict[str, int] = {tok: i for i, tok in enumerate(special)}
    for ch, _ in counter.most_common():
        if ch not in vocab:
            vocab[ch] = len(vocab)

    idx_to_char: dict[int, str] = {i: ch for ch, i in vocab.items()}
    print(f"  Vocabulary : {len(vocab)} chars  "
          f"(3 special + {len(vocab) - 3} unique from training set)")
    return vocab, idx_to_char


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    """Map a string to character indices. Unknown chars -> <pad> (0)."""
    return [vocab.get(ch, 0) for ch in text]


def decode(indices: list[int], idx_to_char: dict[int, str]) -> str:
    """Map indices back to string. Skips <pad>/<sos>; truncates at <eos>."""
    chars = [idx_to_char.get(i, "") for i in indices if i not in (0, 1)]
    return "".join(chars).split("<eos>")[0]


# =============================================================================
# 3. PAIRED IMAGE-CAPTION DATASET
# =============================================================================

class PairedDataset(Dataset):
    """
    Returns (image_tensor, x, y) triples where:
        image_tensor : (3, 224, 224)  ResNet50-ready image
        x            : (seq_len,)     <sos> c1 c2 ... cN <pad>...
        y            : (seq_len,)      c1 c2 ... cN <eos> <pad>...

    Images are looked up as:
        IMAGE_DIR / slugify(meme_name) + IMAGE_EXT

    slugify() lowercases the meme name and replaces spaces with hyphens,
    matching the filename convention used on disk (e.g. "Y U No" -> "y-u-no").

    Pairs whose image file cannot be found are silently skipped during
    __init__ so the DataLoader never receives a broken sample.
    """

    def __init__(
        self,
        pairs:     list[tuple[str, str]],
        vocab:     dict[str, int],
        image_dir: str  = config.IMAGE_DIR,
        image_ext: str  = config.IMAGE_EXT,
        seq_len:   int  = config.SEQ_LEN,
        transform       = encode_transform,
    ):
        self.vocab     = vocab
        self.seq_len   = seq_len
        self.transform = transform

        PAD = vocab["<pad>"]
        SOS = vocab["<sos>"]
        EOS = vocab["<eos>"]

        self.samples: list[tuple[str, list[int], list[int]]] = []
        skipped = 0

        for meme_name, caption in pairs:
            # Convert meme name to slug format matching image filenames on disk.
            # TSV has "Y U No"; image file is "y-u-no.jpg" — slugify() bridges this.
            slug       = slugify(meme_name)
            image_path = os.path.join(image_dir, slug + image_ext)
            if not os.path.exists(image_path):
                skipped += 1
                continue

            # Encode caption
            ids     = encode(caption, vocab)[: seq_len - 1]
            x       = [SOS] + ids + [PAD] * (seq_len - len(ids) - 1)
            y       = ids + [EOS] + [PAD] * (seq_len - len(ids) - 1)

            self.samples.append((image_path, x, y))

        if skipped:
            print(f"  Warning: {skipped:,} captions skipped (image not found)")
        print(f"  {len(self.samples):,} paired samples ready")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        image_path, x, y = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return (
            image,
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


# =============================================================================
# 4. DATALOADER FACTORY
# =============================================================================

def make_loader(
    dataset:    Dataset,
    batch_size: int  = config.BATCH_SIZE,
    shuffle:    bool = True,
    num_workers: int = 2,
) -> DataLoader:
    """
    num_workers=2 enables background image loading so the GPU is not
    starved waiting for disk I/O between batches.
    """
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = config.DEVICE.type == "cuda",
    )


# =============================================================================
# 5. MAIN ENTRY POINT
# =============================================================================

def prepare_all_splits(
    train_file: str = config.CAPTION_TRAIN,
    val_file:   str = config.CAPTION_VAL,
    test_file:  str = config.CAPTION_TEST,
    image_dir:  str = config.IMAGE_DIR,
    image_ext:  str = config.IMAGE_EXT,
    seq_len:    int = config.SEQ_LEN,
    batch_size: int = config.BATCH_SIZE,
    max_train:  int | None = None,
    max_val:    int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict, dict]:
    """
    Load all three splits, build vocab from train only, return loaders.

    Each loader yields (image_tensor, x, y) batches.
    Vocab is built from training captions only — val/test chars that are
    absent from the training set fall back to <pad> (index 0).

    max_train / max_val : if set, cap the number of caption pairs used.
                          Useful for smoke-testing the full pipeline quickly.
    """
    print("Loading caption splits ...")
    train_pairs = load_captions(train_file)
    val_pairs   = load_captions(val_file)
    test_pairs  = load_captions(test_file)
    total = len(train_pairs) + len(val_pairs) + len(test_pairs)
    print(f"  Total      : {total:>8,}\n")

    # ── Optional subset for smoke testing ─────────────────────────────────────
    if max_train is not None:
        train_pairs = train_pairs[:max_train]
    if max_val is not None:
        val_pairs = val_pairs[:max_val]

    print("Building vocabulary from training captions ...")
    vocab, idx_to_char = build_char_vocab(train_pairs)
    print()

    print("Building paired datasets ...")
    train_ds = PairedDataset(train_pairs, vocab, image_dir, image_ext, seq_len)
    val_ds   = PairedDataset(val_pairs,   vocab, image_dir, image_ext, seq_len)
    test_ds  = PairedDataset(test_pairs,  vocab, image_dir, image_ext, seq_len)

    print(f"  train : {len(train_ds):>8,}")
    print(f"  val   : {len(val_ds):>8,}")
    print(f"  test  : {len(test_ds):>8,}\n")

    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader   = make_loader(val_ds,   batch_size, shuffle=False)
    test_loader  = make_loader(test_ds,  batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, vocab, idx_to_char


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab, idx_to_char = prepare_all_splits()
    images, xb, yb = next(iter(train_loader))
    print(f"Image batch : {images.shape}")
    print(f"x batch     : {xb.shape}")
    print(f"y batch     : {yb.shape}")
    print(f"Sample      : {decode(xb[0].tolist(), idx_to_char)!r}")