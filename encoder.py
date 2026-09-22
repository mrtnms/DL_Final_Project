"""
encoder.py — ResNet50 image encoder for the meme caption generator pipeline.

Responsibilities
----------------
- Define ResNet50Encoder: pretrained backbone + trainable projection head.
- Provide encode_dataset(): run a full image dataset through the encoder
  once and persist the embeddings to disk for efficient LSTM training.
- Define ImageContextProjection: small bridge module used inside the LSTM
  to seed its hidden state from an image embedding vector.

The encoder output is a (B, PROJECTION_DIM) tensor that feeds into CharLSTM
either as a pre-computed cache or in real time during inference.
"""

import torch
from torch import nn
from torchvision import transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader

import config


# ── ImageNet normalisation — required because backbone was pretrained on it ────
encode_transform = T.Compose([
    T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# =============================================================================
# 1. RESNET50 ENCODER
# =============================================================================

class ResNet50Encoder(nn.Module):
    """
    Pretrained ResNet50 image encoder.

    Architecture
    ------------
    ResNet50 backbone (frozen by default)
        → global average pool  →  (B, 2048, 1, 1)
        → Flatten              →  (B, 2048)
        → Linear(2048, PROJECTION_DIM)
        → ReLU
        → LayerNorm(PROJECTION_DIM)
        →                         (B, PROJECTION_DIM)

    The backbone is frozen by default; only the projection head is trained
    (or fine-tuned if freeze_backbone=False).  LayerNorm stabilises the
    embedding magnitude before it seeds the LSTM hidden state.

    Forward
    -------
    x   : (B, 3, 224, 224)  — ImageNet-normalised input images
    out : (B, projection_dim)

    Usage in the full pipeline
    --------------------------
    Pre-compute and cache embeddings (recommended):
        encode_dataset(encoder, image_loader, save_path="image_embeddings.pt")

    Real-time (inference only):
        z = encoder(image_batch)   # (B, projection_dim)
    """

    def __init__(
        self,
        projection_dim: int = config.PROJECTION_DIM,
        freeze_backbone: bool = config.FREEZE_BACKBONE,
    ):
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Remove the classification head; output after this is (B, 2048, 1, 1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, projection_dim),
            nn.ReLU(),
            nn.LayerNorm(projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)       # (B, 2048, 1, 1)
        return self.projection(features)  # (B, projection_dim)


# Note: ImageContextProjection removed — image conditioning is now done
# inside CharLSTM via per-step concatenation. See lstm.py.

# =============================================================================
# 2. DATASET ENCODING UTILITY
# =============================================================================

@torch.no_grad()
def encode_dataset(
    encoder: ResNet50Encoder,
    loader: DataLoader,
    save_path: str = config.EMBEDDINGS_PATH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run every image in `loader` through `encoder` once and save the result.

    Call this as a one-off preprocessing step.  At LSTM training time,
    load the .pt file directly — no need to run ResNet50 every epoch.

    Returns
    -------
    embeddings : (N, projection_dim)
    labels     : (N,)  — class indices from the dataset, or zeros if absent
    """
    encoder.eval()
    encoder.to(config.DEVICE)

    all_embeddings: list[torch.Tensor] = []
    all_labels:     list[torch.Tensor] = []

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        else:
            x = batch
            y = torch.zeros(x.size(0), dtype=torch.long)

        z = encoder(x.to(config.DEVICE))   # (B, projection_dim)
        all_embeddings.append(z.cpu())
        all_labels.append(y)

    embeddings = torch.cat(all_embeddings, dim=0)
    labels     = torch.cat(all_labels,     dim=0)

    torch.save({"embeddings": embeddings, "labels": labels}, save_path)
    print(f"Saved {embeddings.shape[0]:,} embeddings  →  {save_path}")
    return embeddings, labels


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    encoder = ResNet50Encoder().to(config.DEVICE)

    total     = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}  (projection head only)")

    dummy = torch.randn(4, 3, 224, 224).to(config.DEVICE)
    with torch.no_grad():
        z = encoder(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {z.shape}")    # (4, 256)