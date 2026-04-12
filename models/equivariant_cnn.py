"""
models/equivariant_cnn.py

Five G-equivariant CNNs for avalanche debris classification:

    C8EquivariantCNN      — cyclic group of order 8   (rot2dOnR2(N=8))
    SO2EquivariantCNN     — continuous SO(2)           (rot2dOnR2(maximum_frequency=4))
    D4EquivariantCNN      — dihedral group of order 8  (flipRot2dOnR2(N=4))
    O2EquivariantCNN      — continuous O(2)            (flipRot2dOnR2(N=-1, maximum_frequency=8))
    D4BiTemporalCNN       — D4 with shared-weight bi-temporal encoder (pre + post SAR)

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
    For O2:  band-limited regular repr dim=1+1+2*L=18 (L=8) → 18-dimensional per copy

    All four equivariant models target ~390K parameters by matching total effective
    feature channels (n_regular × repr_dim) ≈ 468 across architectures:
        C8/D4 n_regular=52, dim=8  → 52×8=416
        SO2   n_regular=52, dim=9  → 52×9=468
        O2    n_regular=26, dim=18 → 26×18=468
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
# O(2) — continuous orthogonal group (rotations + reflections), band-limited to L=8
# ---------------------------------------------------------------------------

class O2EquivariantCNN(_EquivariantCNNBase):
    """
    O(2)-equivariant CNN: continuous rotations AND reflections, band-limited at L=8.

    Design rationale — O(2) vs D4 as a controlled comparison:
        D4 is the discrete dihedral group: exactly 4 rotations + 4 reflections.
        O(2) is the continuous orthogonal group: all rotations + all reflections.
        Both encode reflection symmetry (physically motivated by the approximate
        bilateral symmetry of avalanche runouts perpendicular to the fall line).

        maximum_frequency=8 matches the frequency content of C8, which decomposes
        into 8 one-dimensional irreps at angular frequencies k=0,...,7. O(2) at L=8
        covers the same angular frequency range while (a) making the rotational
        symmetry continuous and (b) adding continuous reflection symmetry. This
        makes O(2) vs D4 a controlled test of continuous vs discrete dihedral
        symmetry at matched frequency content, independent of bandwidth effects.

    Band-limited regular representation at L=8:
        trivial (freq 0, det=+1): dim 1
        sign   (freq 0, det=-1): dim 1
        ρ_k for k=1,...,8: dim 2 each → 16
        Total: 18

    n_regular=26 keeps total effective feature channels (n_regular × dim) ≈ 468,
    matching SO2 (n_regular=52, dim=9) and C8/D4 (n_regular=52, dim=8/8).

    Uses NormNonLinearity + IIDBatchNorm2d (same as SO2) — required for continuous
    groups where InnerBatchNorm and ELU do not guarantee equivariance.

    Head 2 note — same reflection ambiguity as D4:
        The basespace_action for O(2) maps reflections to (x,y) → (x,−y).
        Interpret the 2D orientation output as an undirected axis, not a directed
        vector. See visualize_d4_orientation() (applies equally here).
    """

    MAX_FREQUENCY = 8

    def __init__(self, in_channels: int = 5, n_regular: int = 26) -> None:
        super().__init__()
        # N=-1: continuous rotations (SO(2) subgroup); maximum_frequency sets band limit.
        # flipRot2dOnR2 adds reflection generators to make the full O(2) group.
        self.gspace = gspaces.flipRot2dOnR2(N=-1, maximum_frequency=self.MAX_FREQUENCY)

        # Pre-instantiate O(2) irreps up to 2*MAX_FREQUENCY.
        # Clebsch-Gordan decompositions of tensor products of irreps at frequency ≤ L
        # can produce irreps at frequencies up to 2L. escnn will raise
        # InsufficientIrrepsException if these are not already instantiated when
        # building the equivariant kernel basis.
        #
        # O(2) irrep parameterization in escnn: (j, k)
        #   irrep(0, 0): trivial (1D, det=+1, rot=trivial)
        #   irrep(1, 0): sign   (1D, det=-1, rot=trivial)
        #   irrep(1, k) for k≥1: 2D irreps at rotational frequency k
        #   irrep(0, k) for k≥1: INVALID — only one 2D irrep per frequency k≥1.
        fg = self.gspace.fibergroup
        fg.irrep(0, 0)   # trivial
        for k in range(2 * self.MAX_FREQUENCY + 1):
            fg.irrep(1, k)   # sign (k=0) and all 2D irreps (k≥1)

        # Band-limited regular repr: trivial + sign + ρ_k (k=1..L), total dim=18 at L=8.
        bl_repr = fg.bl_regular_representation(L=self.MAX_FREQUENCY)
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
# D4 bi-temporal — shared-weight encoder for pre + post SAR patches
# ---------------------------------------------------------------------------

class D4BiTemporalCNN(_EquivariantCNNBase):
    """
    D4-equivariant bi-temporal CNN for change detection.

    A shared-weight D4-equivariant encoder is applied independently to both
    the pre-event and post-event SAR patches. The feature difference
    (post − pre) forms the change representation passed to the classification head.

    Equivariance of the change signal:
        Let f be the D4-equivariant encoder and g ∈ D4. Then:
            f(g·x_post) − f(g·x_pre)
          = g·f(x_post) − g·f(x_pre)   (D4-equivariance of f, applied twice)
          = g·(f(x_post) − f(x_pre))   (linearity of the group action on feature vectors)
        So the change feature is D4-equivariant under simultaneous rotation/reflection
        of both patches — the correct symmetry for a scene observed from a fixed sensor
        at an unknown orientation.

    Inputs (two separate 5-channel tensors):
        x_post: [B, 5, 64, 64]  (VH_post, VV_post, slope, sin_asp, cos_asp)
        x_pre:  [B, 5, 64, 64]  (VH_pre,  VV_pre,  slope, sin_asp, cos_asp)

    Outputs: (logit [B, 1], orientation [B, 2] | None)

    Parameter count: identical to D4EquivariantCNN (~391K) because the encoder
    weights are shared between the two branches — one set of weights, two forward passes.

    Use AvalancheDataset(bitemporal=True) to obtain (post_5ch, pre_5ch) pairs.
    """

    bitemporal: bool = True   # flag read by train/evaluate to handle tuple batches

    def __init__(self, in_channels: int = 5, n_regular: int = 52) -> None:
        super().__init__()
        self.gspace            = gspaces.flipRot2dOnR2(N=4)
        self.feat_type_regular = enn.FieldType(
            self.gspace, [self.gspace.regular_repr] * n_regular
        )
        self._build(in_channels)

    def forward(
        self,
        x_post: torch.Tensor,
        x_pre:  torch.Tensor,
        return_orientation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x_post:             [B, 5, 64, 64] post-event patch
            x_pre:              [B, 5, 64, 64] pre-event patch
            return_orientation: If True, also compute the equivariant orientation head.

        Returns:
            logit:       [B, 1]        — for BCEWithLogitsLoss
            orientation: [B, 2] | None
        """
        feat_in_type = self.block1.in_type

        def _encode(x: torch.Tensor) -> enn.GeometricTensor:
            """Run the shared backbone on one patch."""
            g = enn.GeometricTensor(x, feat_in_type)
            g = self.block1(g)
            g = self.block2(g)
            g = self.block3(g)
            return self.block4(g)

        feat_post = _encode(x_post)   # [B, feat_regular.size, 8, 8]
        feat_pre  = _encode(x_pre)    # [B, feat_regular.size, 8, 8]

        # Equivariant change feature: post − pre
        # By linearity: g·(feat_post − feat_pre) = g·feat_post − g·feat_pre
        change_tensor = feat_post.tensor - feat_pre.tensor
        change_geo    = enn.GeometricTensor(change_tensor, self.feat_type_regular)

        # Head 1 — classification
        inv      = self.group_pool(change_geo)
        inv      = self.spatial_pool_h1(inv)
        inv_flat = inv.tensor.flatten(1)
        logit    = self.classifier(inv_flat)

        # Head 2 — orientation (change-flow direction)
        if return_orientation:
            ori    = self.orientation_conv(change_geo)
            orient = torch.nn.functional.adaptive_avg_pool2d(
                ori.tensor, (1, 1)
            ).flatten(1)
        else:
            orient = None

        return logit, orient


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
        (O2EquivariantCNN,  "O(2)"),
    ]:
        model = ModelClass(in_channels=5)
        model.eval()
        with torch.no_grad():
            logit, orient = model(x, return_orientation=True)
        params = count_parameters(model)
        print(f"{name:6s}  logit={tuple(logit.shape)}  orient={tuple(orient.shape)}  params={params:,}")

    # Bi-temporal model takes (post, pre) pair
    bt_model = D4BiTemporalCNN(in_channels=5)
    bt_model.eval()
    with torch.no_grad():
        logit, orient = bt_model(x, x, return_orientation=True)
    params = count_parameters(bt_model)
    print(f"D4-BT  logit={tuple(logit.shape)}  orient={tuple(orient.shape)}  params={params:,}")
