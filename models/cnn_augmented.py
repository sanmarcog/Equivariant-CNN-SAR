"""
models/cnn_augmented.py

Augmentation baseline: identical architecture to CNNBaseline, but with heavy
random rotation augmentation applied during training.

The purpose of this baseline is to test whether data augmentation alone can
achieve what the C8 equivariant CNN achieves through its architecture. If the
equivariant model outperforms this baseline, the gain is attributable to the
structural guarantee of equivariance rather than simply seeing more rotated
examples.

Augmentations applied during training only (model.eval() disables them):
    - Random rotation: uniform sample from [0°, 360°)
    - Random horizontal flip (p=0.5)
    - Random vertical flip   (p=0.5)

These cover the full rotation group augmentations. No other augmentations
are applied so that rotation is the isolated variable.

Input:  [B, 5, 64, 64]  (VH, VV, slope, sin_asp, cos_asp)
Output: [B, 1]          raw logit (use BCEWithLogitsLoss)

Parameter count is identical to CNNBaseline with the same base_channels.

Usage:
    from models.cnn_augmented import AugmentedCNN
    model = AugmentedCNN(in_channels=5, base_channels=32)
    logits = model(x)   # augmentation applied only when model.training=True
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from models.cnn_baseline import CNNBaseline, count_parameters


class AugmentedCNN(nn.Module):
    """
    Standard CNN with heavy random rotation augmentation at training time.

    Args:
        in_channels:   Number of input channels (default: 5).
        base_channels: Width multiplier — must match CNNBaseline for fair
                       parameter comparison (default: 32).
    """

    def __init__(self, in_channels: int = 5, base_channels: int = 32) -> None:
        super().__init__()
        self.backbone = CNNBaseline(in_channels=in_channels, base_channels=base_channels)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random rotation and flips independently per sample in the batch.
        Operates on CPU or GPU tensors via torchvision functional API.
        bilinear interpolation for rotation; constant fill=0 for out-of-bounds.
        """
        augmented = []
        for sample in x:                          # sample: [C, H, W]
            # Random rotation — uniform over full circle
            angle = torch.empty(1).uniform_(0.0, 360.0).item()
            sample = TF.rotate(
                sample,
                angle=angle,
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0.0,
            )

            # Random horizontal flip
            if torch.rand(1).item() < 0.5:
                sample = TF.hflip(sample)

            # Random vertical flip
            if torch.rand(1).item() < 0.5:
                sample = TF.vflip(sample)

            augmented.append(sample)

        return torch.stack(augmented, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = self._augment(x)
        return self.backbone(x)


if __name__ == "__main__":
    model = AugmentedCNN(in_channels=5, base_channels=32)
    x = torch.randn(4, 5, 64, 64)

    model.train()
    out_train = model(x)
    print(f"Output shape (train) : {out_train.shape}")

    model.eval()
    out_eval = model(x)
    print(f"Output shape (eval)  : {out_eval.shape}")
    print(f"Parameters           : {count_parameters(model):,}")
