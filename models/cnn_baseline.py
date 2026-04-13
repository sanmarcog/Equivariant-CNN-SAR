"""
models/cnn_baseline.py

Standard CNN baseline and bi-temporal CNN baseline for avalanche debris
classification.  No rotation augmentation, no equivariance — plain
convolutional networks.

Classes:
    CNNBaseline     — single-image baseline; input [B, 5, 64, 64]
    CNNBiTemporal   — bi-temporal baseline; inputs ([B, 5, 64, 64], [B, 5, 64, 64])
                      Shared-weight encoder + feature difference.
                      Same API as D4BiTemporalCNN: forward(x_post, x_pre,
                      return_orientation=False) → (logit [B,1], None).
                      ~391K parameters, matching D4-BT.

Usage:
    from models.cnn_baseline import CNNBaseline, CNNBiTemporal
    model = CNNBaseline(in_channels=5, base_channels=32)
    logits = model(x)                           # x: [B, 5, 64, 64]

    bt = CNNBiTemporal(in_channels=5, base_channels=32)
    logit, _ = bt(x_post, x_pre)               # both [B, 5, 64, 64]
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


class CNNBiTemporal(nn.Module):
    """
    Bi-temporal plain CNN baseline — no equivariance.

    Applies a shared-weight CNNBaseline encoder to the post-event and
    pre-event patches separately, computes the feature difference, and
    classifies the difference vector.  This is the plain-CNN analogue of
    D4BiTemporalCNN: identical call signature, matched ~391K parameter count.

    Args:
        in_channels:   Number of input channels per branch (default: 5).
        base_channels: Width of the first conv block (default: 32).

    Forward:
        forward(x_post, x_pre, return_orientation=False)
            x_post, x_pre : [B, in_channels, 64, 64]
            returns        : (logit [B, 1], None)
        The `return_orientation` kwarg is accepted but ignored (no orientation
        head); it exists only to match the D4BiTemporalCNN API so that
        forward_logit() in train.py / evaluate.py can dispatch both models
        identically via the `bitemporal` attribute.
    """

    # Detected by forward_logit() in train.py / evaluate.py
    bitemporal: bool = True

    def __init__(self, in_channels: int = 5, base_channels: int = 32) -> None:
        super().__init__()

        c = base_channels
        # Shared-weight encoder — identical to CNNBaseline.features
        self.features = nn.Sequential(
            ConvBlock(in_channels, c,     pool=True),   # → [B, c,   32, 32]
            ConvBlock(c,           c * 2, pool=True),   # → [B, 2c,  16, 16]
            ConvBlock(c * 2,       c * 4, pool=True),   # → [B, 4c,   8,  8]
            ConvBlock(c * 4,       c * 8, pool=True),   # → [B, 8c,   4,  4]
        )

        # Global Average Pooling: [B, 8c, 4, 4] → [B, 8c]
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier on change feature (post − pre)
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

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single branch: [B, C, 64, 64] → [B, 8c]."""
        x = self.features(x)   # [B, 8c, 4, 4]
        x = self.gap(x)        # [B, 8c, 1, 1]
        return x.flatten(1)    # [B, 8c]

    def forward(
        self,
        x_post: torch.Tensor,
        x_pre: torch.Tensor,
        return_orientation: bool = False,
    ) -> tuple[torch.Tensor, None]:
        """
        Args:
            x_post: post-event patch  [B, in_channels, 64, 64]
            x_pre:  pre-event patch   [B, in_channels, 64, 64]
            return_orientation: ignored; kept for API compatibility with
                                D4BiTemporalCNN.

        Returns:
            (logit [B, 1], None)
        """
        feat_post = self._encode(x_post)        # [B, 8c]
        feat_pre  = self._encode(x_pre)         # [B, 8c]
        change    = feat_post - feat_pre         # [B, 8c]
        logit     = self.classifier(change)      # [B, 1]
        return logit, None


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNBaseline(in_channels=5, base_channels=32)
    x = torch.randn(4, 5, 64, 64)
    out = model(x)
    print(f"CNNBaseline output shape : {out.shape}")
    print(f"CNNBaseline parameters   : {count_parameters(model):,}")

    bt = CNNBiTemporal(in_channels=5, base_channels=32)
    x_post = torch.randn(4, 5, 64, 64)
    x_pre  = torch.randn(4, 5, 64, 64)
    logit, orientation = bt(x_post, x_pre)
    print(f"CNNBiTemporal output shape : {logit.shape}, orientation={orientation}")
    print(f"CNNBiTemporal parameters   : {count_parameters(bt):,}")
