"""
models/resnet_baseline.py

Fine-tuned ResNet-18 baseline for avalanche debris classification.

The first convolutional layer is replaced to accept 5 input channels
(VH, VV, slope, sin_asp, cos_asp) instead of the standard 3 (RGB).
Weights for the new first layer are initialised by averaging the pretrained
RGB weights across the channel dimension and tiling — a standard approach
that preserves the pretrained feature structure as closely as possible.
All other layers are initialised from ImageNet pretrained weights and
fine-tuned end-to-end.

The final fully-connected layer is replaced with a single linear output
(binary logit).

Input:  [B, 5, 64, 64]
Output: [B, 1]          raw logit (use BCEWithLogitsLoss)

Parameter count: ~11.2M (ResNet-18 is intentionally much larger than the
CNN baselines — it represents the off-the-shelf pretrained model baseline,
not a parameter-matched comparison).

Usage:
    from models.resnet_baseline import ResNetBaseline
    model = ResNetBaseline(in_channels=5, pretrained=True)
    logits = model(x)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from models.cnn_baseline import count_parameters


class ResNetBaseline(nn.Module):
    """
    Fine-tuned ResNet-18 adapted for 5-channel SAR input.

    Args:
        in_channels: Number of input channels (default: 5).
        pretrained:  If True, load ImageNet weights and adapt first layer.
                     If False, random initialisation (for ablation).
    """

    def __init__(self, in_channels: int = 5, pretrained: bool = True) -> None:
        super().__init__()

        if pretrained:
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet18(weights=None)

        # --- Adapt first conv layer to in_channels ----------------------
        old_conv = backbone.conv1  # Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        if pretrained:
            # Average pretrained RGB weights across channel dim, tile to in_channels
            # Shape: [64, 3, 7, 7] → mean over dim=1 → [64, 1, 7, 7] → tile
            with torch.no_grad():
                mean_weight = old_conv.weight.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
                new_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        backbone.conv1 = new_conv

        # --- Replace final FC layer -------------------------------------
        in_features = backbone.fc.in_features  # 512 for ResNet-18
        backbone.fc = nn.Linear(in_features, 1)
        nn.init.xavier_normal_(backbone.fc.weight)
        nn.init.zeros_(backbone.fc.bias)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # [B, 1]


if __name__ == "__main__":
    model = ResNetBaseline(in_channels=5, pretrained=True)
    x = torch.randn(4, 5, 64, 64)
    out = model(x)
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {count_parameters(model):,}")
