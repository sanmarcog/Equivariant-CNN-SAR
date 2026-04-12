"""
models/cnn_baseline.py

Standard CNN baseline for avalanche debris classification.
No rotation augmentation, no equivariance — a plain convolutional network.

Architecture:
    4 convolutional blocks (Conv2d → BatchNorm2d → ReLU → MaxPool2d)
    followed by Global Average Pooling and a linear classifier.

Input:  [B, 5, 64, 64]  (VH, VV, slope, sin_asp, cos_asp)
Output: [B, 1]          raw logit for binary classification (use BCEWithLogitsLoss)

Parameter count is configurable via `base_channels` so it can be matched
to the equivariant CNN once that architecture is finalised.

Default (base_channels=32) gives ~390k parameters — adjust to match your
equivariant model.

Usage:
    from models.cnn_baseline import CNNBaseline
    model = CNNBaseline(in_channels=5, base_channels=32)
    logits = model(x)   # x: [B, 5, 64, 64]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU → MaxPool2d."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNBaseline(nn.Module):
    """
    Standard CNN baseline — no equivariance, no augmentation.

    Args:
        in_channels:   Number of input channels (default: 5).
        base_channels: Width of the first conv block. Each subsequent block
                       doubles the channels up to 8×base_channels.
                       Tune this to match parameter count of equivariant model.
    """

    def __init__(self, in_channels: int = 5, base_channels: int = 32) -> None:
        super().__init__()

        c = base_channels
        # Input 64×64 → after 4 MaxPool2d(2) → 4×4 feature maps
        self.features = nn.Sequential(
            ConvBlock(in_channels, c,     pool=True),   # → [B, c,   32, 32]
            ConvBlock(c,           c * 2, pool=True),   # → [B, 2c,  16, 16]
            ConvBlock(c * 2,       c * 4, pool=True),   # → [B, 4c,   8,  8]
            ConvBlock(c * 4,       c * 8, pool=True),   # → [B, 8c,   4,  4]
        )

        # Global Average Pooling: [B, 8c, 4, 4] → [B, 8c]
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Linear(c * 8, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # [B, 8c, 4, 4]
        x = self.gap(x)            # [B, 8c, 1, 1]
        x = x.flatten(1)           # [B, 8c]
        return self.classifier(x)  # [B, 1]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNBaseline(in_channels=5, base_channels=32)
    x = torch.randn(4, 5, 64, 64)
    out = model(x)
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {count_parameters(model):,}")
