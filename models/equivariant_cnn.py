"""
models/equivariant_cnn.py

Three G-equivariant CNNs for avalanche debris classification:

    C8EquivariantCNN  — cyclic group of order 8   (rot2dOnR2(N=8))
    SO2EquivariantCNN — continuous SO(2)           (rot2dOnR2(maximum_frequency=4))
    D4EquivariantCNN  — dihedral group of order 8  (flipRot2dOnR2(N=4))

Shared backbone architecture (4 equivariant conv blocks):
    Block 1 : R2Conv (trivial_in → regular), k=3, InnerBatchNorm, ELU
    Block 2 : R2Conv (regular → regular),    k=3, InnerBatchNorm, ELU, PointwiseAvgPool(2)
    Block 3 : R2Conv (regular → regular),    k=3, InnerBatchNorm, ELU, PointwiseAvgPool(2)
    Block 4 : R2Conv (regular → regular),    k=3, InnerBatchNorm, ELU, PointwiseAvgPool(2)

Two heads on the shared backbone:

    Head 1 — Invariant classification (used for training loss):
        GroupPooling → PointwiseAdaptiveAvgPool2D(1) → flatten → Linear → scalar logit

    Head 2 — Equivariant orientation readout (visualization only, NOT in loss):
        R2Conv (regular → basespace_action repr) → PointwiseAdaptiveAvgPool2D(1) → flatten
        Output: 2D vector that rotates/reflects with the input image.

        For C8 / SO(2): output is a directed 2D vector (debris flow direction).
        For D4:         output is an undirected axis — both (x,y) and (x,−y) are
                        valid due to the reflection ambiguity. Visualize both.

Input:   [B, 5, 64, 64]  float32  (VH, VV, slope, sin_asp, cos_asp)
Outputs: (logit [B,1], orientation [B,2])

Training: use logit from Head 1 with BCEWithLogitsLoss only.
          Head 2 is returned for qualitative visualization at eval time.

Parameter matching:
    All three equivariant models use `n_regular` copies of the regular (or
    band-limited regular for SO(2)) representation per layer. The value of
    `n_regular` is chosen so that parameter counts are comparable to the
    standard CNN baselines (~390k). Call count_parameters() to verify.

    For C8:  |G|=8,  regular repr dim=8  → n_regular channels in regular repr sense
    For D4:  |G|=8,  regular repr dim=8  → same
    For SO2: band-limited regular repr dim=2*L+1=9 (L=4) → slightly larger per copy

    Default n_regular=16 gives ~390k params for C8 and D4.
    SO2 uses n_regular=14 to stay near the same count.
    Verify with count_parameters() before training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import escnn.nn as enn
from escnn import gspaces


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _make_block(
    in_type: enn.FieldType,
    out_type: enn.FieldType,
    kernel_size: int = 3,
    pool: bool = False,
    norm_cls: type = enn.InnerBatchNorm,
    nonlin_fn=None,
) -> enn.SequentialModule:
    """R2Conv → BatchNorm → Nonlinearity → (optional) PointwiseAvgPool.

    norm_cls / nonlin_fn:
        Finite groups (C8, D4):
            norm_cls  = enn.InnerBatchNorm   (requires pointwise-compatible repr)
            nonlin_fn = None  →  enn.ELU (default)
        SO(2) with band-limited regular repr:
            norm_cls  = enn.IIDBatchNorm2d   (works with any repr)
            nonlin_fn = lambda ft: enn.NormNonLinearity(ft)
                        NormNonLinearity applies a scalar function to the norm of
                        each field vector — equivariant for any representation.
    """
    if nonlin_fn is None:
        nonlin_fn = lambda ft: enn.ELU(ft, inplace=True)

    layers: list[enn.EquivariantModule] = [
        enn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
        norm_cls(out_type),
        nonlin_fn(out_type),
    ]
    if pool:
        layers.append(enn.PointwiseAvgPool2D(out_type, kernel_size=2, stride=2))
    return enn.SequentialModule(*layers)


# ---------------------------------------------------------------------------
# Base class — shared logic for all three groups
# ---------------------------------------------------------------------------

class _EquivariantCNNBase(nn.Module):
    """
    Shared equivariant backbone + two-head output.
    Subclasses define self.gspace and self.feat_type_regular.
    """

    # Subclasses must set these before calling _build()
    gspace: gspaces.GSpace2D
    feat_type_regular: enn.FieldType   # regular (or bl-regular) repr, n_regular copies

    def _build(
        self,
        in_channels: int,
        norm_cls: type = enn.InnerBatchNorm,
        nonlin_fn=None,
        use_norm_pool: bool = False,
    ) -> None:
        """Wire backbone and heads. Call from subclass __init__ after setting gspace.

        use_norm_pool:
            False (default, finite groups C8/D4): GroupPooling — pools over group
                elements, requires pointwise-compatible representations.
            True (SO(2)): NormPool — computes the norm of each field vector.
                Invariant for any unitary representation since ||g·v|| = ||v||.
        """
        gs = self.gspace

        # --- Input field type: trivial repr for each input channel ----------
        feat_in = enn.FieldType(gs, [gs.trivial_repr] * in_channels)

        # --- Backbone --------------------------------------------------------
        self.block1 = _make_block(feat_in,                  self.feat_type_regular, pool=False, norm_cls=norm_cls, nonlin_fn=nonlin_fn)
        self.block2 = _make_block(self.feat_type_regular,   self.feat_type_regular, pool=True,  norm_cls=norm_cls, nonlin_fn=nonlin_fn)
        self.block3 = _make_block(self.feat_type_regular,   self.feat_type_regular, pool=True,  norm_cls=norm_cls, nonlin_fn=nonlin_fn)
        self.block4 = _make_block(self.feat_type_regular,   self.feat_type_regular, pool=True,  norm_cls=norm_cls, nonlin_fn=nonlin_fn)
        # Spatial dims: 64 → 32 → 16 → 8 after three pool-by-2 steps

        # --- Head 1: invariant classification --------------------------------
        if use_norm_pool:
            # NormPool: norm of each irrep component → invariant scalar per irrep
            # Works with any representation including band-limited regular reprs
            self.group_pool  = enn.NormPool(self.feat_type_regular)
        else:
            # GroupPooling: average over group elements → trivial repr
            # Requires pointwise-compatible representations (finite groups only)
            self.group_pool  = enn.GroupPooling(self.feat_type_regular)

        invariant_type = self.group_pool.out_type

        # Adaptive spatial pool to 1×1, then flatten, then linear → scalar
        self.spatial_pool_h1 = enn.PointwiseAdaptiveAvgPool2D(invariant_type, output_size=(1, 1))
        invariant_dim         = invariant_type.size
        self.classifier       = nn.Linear(invariant_dim, 1)

        # --- Head 2: equivariant orientation readout -------------------------
        # basespace_action: 2D repr where g acts as its 2×2 matrix on R²
        # For C8/SO2: rotation matrix. For D4: rotation + reflection matrix.
        orientation_repr = gs.basespace_action
        feat_orientation = enn.FieldType(gs, [orientation_repr])  # one 2D vector field

        self.orientation_conv = enn.R2Conv(
            self.feat_type_regular, feat_orientation,
            kernel_size=1, padding=0, bias=False,
        )
        # Note: basespace_action is a 2D irrep and does not support pointwise
        # operations, so we cannot use PointwiseAdaptiveAvgPool2D here.
        # Spatial averaging is done on the raw tensor in forward() via
        # F.adaptive_avg_pool2d, which commutes with the group action.

        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        return_orientation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x:                  [B, in_channels, 64, 64] plain tensor
            return_orientation: If True, also run Head 2 and return the 2D
                                orientation vector.  Set False during training
                                to avoid wasted compute (Head 2 is never in
                                the loss).

        Returns:
            logit:       [B, 1]        — for BCEWithLogitsLoss (Head 1)
            orientation: [B, 2] | None — equivariant 2D vector (Head 2),
                                         or None when return_orientation=False
        """
        # Wrap input as GeometricTensor
        feat_in_type = self.block1.in_type
        x_geo = enn.GeometricTensor(x, feat_in_type)

        # Backbone
        x_geo = self.block1(x_geo)
        x_geo = self.block2(x_geo)
        x_geo = self.block3(x_geo)
        x_geo = self.block4(x_geo)   # [B, feat_regular.size, 8, 8]

        # Head 1 — classification
        inv      = self.group_pool(x_geo)                    # pool over group → trivial
        inv      = self.spatial_pool_h1(inv)                 # [B, n_regular, 1, 1]
        inv_flat = inv.tensor.flatten(1)                     # [B, n_regular]
        logit    = self.classifier(inv_flat)                 # [B, 1]

        # Head 2 — orientation (skipped during training to avoid wasted compute)
        if return_orientation:
            ori    = self.orientation_conv(x_geo)            # GeometricTensor [B, 2, 8, 8]
            orient = torch.nn.functional.adaptive_avg_pool2d(
                ori.tensor, (1, 1)
            ).flatten(1)                                     # [B, 2]
        else:
            orient = None

        return logit, orient


# ---------------------------------------------------------------------------
# C8 — cyclic group of order 8
# ---------------------------------------------------------------------------

class C8EquivariantCNN(_EquivariantCNNBase):
    """
    C8-equivariant CNN. Equivariant to rotations by multiples of 45°.
    Regular representation has dimension 8 (one basis function per group element).

    Head 2 output is a directed 2D vector (debris flow direction).
    Rotating the input by 45k° rotates the output vector by 45k°.
    """

    def __init__(self, in_channels: int = 5, n_regular: int = 52) -> None:
        super().__init__()
        self.gspace           = gspaces.rot2dOnR2(N=8)
        self.feat_type_regular = enn.FieldType(
            self.gspace, [self.gspace.regular_repr] * n_regular
        )
        self._build(in_channels)


# ---------------------------------------------------------------------------
# SO(2) — continuous rotation group, band-limited to frequency 4
# ---------------------------------------------------------------------------

class SO2EquivariantCNN(_EquivariantCNNBase):
    """
    SO(2)-equivariant CNN with band-limited feature fields (max frequency=4).
    Equivariant to any continuous rotation (approximately — ELU introduces
    a small equivariance error at non-grid angles, hence unit test tolerance 1e-4).

    The band-limited regular representation has dimension 2*L+1 = 9 (L=4):
    one trivial irrep (freq 0) + four pairs of freq-k irreps (k=1..4).

    n_regular=14 gives ~390k params, close to C8/D4 at n_regular=16.

    Head 2 output is a directed 2D vector that rotates continuously with input.
    """

    MAX_FREQUENCY = 4

    def __init__(self, in_channels: int = 5, n_regular: int = 52) -> None:
        super().__init__()
        self.gspace = gspaces.rot2dOnR2(maximum_frequency=self.MAX_FREQUENCY)

        # Band-limited regular representation: sum of irreps 0..MAX_FREQUENCY
        bl_repr = self.gspace.fibergroup.bl_regular_representation(L=self.MAX_FREQUENCY)
        self.feat_type_regular = enn.FieldType(
            self.gspace, [bl_repr] * n_regular
        )
        self._build(
            in_channels,
            norm_cls=enn.IIDBatchNorm2d,
            nonlin_fn=lambda ft: enn.NormNonLinearity(ft),
            use_norm_pool=True,
        )


# ---------------------------------------------------------------------------
# D4 — dihedral group of order 8 (4 rotations × 2 reflections)
# ---------------------------------------------------------------------------

class D4EquivariantCNN(_EquivariantCNNBase):
    """
    D4-equivariant CNN. Equivariant to 90° rotations and reflections.
    Regular representation has dimension 8 (|D4|=8).

    Head 2 note — reflection ambiguity:
        The basespace_action for D4 includes reflections. A reflection maps
        (x, y) → (x, −y) (for a horizontal flip). This means the 2D vector
        output is only defined up to this reflection — both (x,y) and (x,−y)
        are geometrically valid orientations. Interpret the output as an
        *undirected axis*, not a directed vector.

        Visualize both (x,y) and (x,−y) per patch to make this ambiguity
        visible. See visualize_d4_orientation() below.
    """

    def __init__(self, in_channels: int = 5, n_regular: int = 52) -> None:
        super().__init__()
        self.gspace            = gspaces.flipRot2dOnR2(N=4)
        self.feat_type_regular = enn.FieldType(
            self.gspace, [self.gspace.regular_repr] * n_regular
        )
        self._build(in_channels)


# ---------------------------------------------------------------------------
# D4 orientation visualization
# ---------------------------------------------------------------------------

def visualize_d4_orientation(
    orientations: torch.Tensor,
    patch_indices: list[int] | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot both (x, y) and (x, −y) orientation vectors for D4 patches to make
    the reflection ambiguity visible.

    Args:
        orientations:  [B, 2] tensor of Head 2 outputs from D4EquivariantCNN.
        patch_indices: Which batch indices to plot. Defaults to all.
        save_path:     If given, save figure to this path instead of showing.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib is required for visualization: pip install matplotlib")

    ori = orientations.detach().cpu().numpy()   # [B, 2]
    B   = ori.shape[0]

    if patch_indices is None:
        patch_indices = list(range(B))

    n = len(patch_indices)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)

    for col, idx in enumerate(patch_indices):
        ax = axes[0, col]
        x, y = ori[idx, 0], ori[idx, 1]

        # Both valid orientations under D4 reflection ambiguity
        ax.annotate("", xy=(x, y),   xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="steelblue", lw=2),
                    label="(x, y)")
        ax.annotate("", xy=(x, -y),  xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="tomato", lw=2,
                                   linestyle="dashed"),
                    label="(x, −y)")

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"Patch {idx}\n(x,y) vs (x,−y)", fontsize=8)
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color="steelblue", lw=2, label="(x, y)"),
                plt.Line2D([0], [0], color="tomato",    lw=2, linestyle="dashed", label="(x, −y)"),
            ],
            fontsize=7, loc="lower right",
        )

    fig.suptitle("D4 orientation readout — reflection ambiguity", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    x = torch.randn(2, 5, 64, 64)

    for ModelClass, name in [
        (C8EquivariantCNN,  "C8"),
        (SO2EquivariantCNN, "SO(2)"),
        (D4EquivariantCNN,  "D4"),
    ]:
        model = ModelClass(in_channels=5)
        model.eval()
        with torch.no_grad():
            logit, orient = model(x)
        params = count_parameters(model)
        print(f"{name:6s}  logit={tuple(logit.shape)}  orient={tuple(orient.shape)}  params={params:,}")
