"""
dataset.py

PyTorch Dataset for AvalCD avalanche debris classification.

Each sample is a [5, 64, 64] float32 tensor with channels:
    0  VH_post   — post-event VH backscatter (σ⁰ dB)
    1  VV_post   — post-event VV backscatter (σ⁰ dB)
    2  slope     — terrain slope (degrees)
    3  sin_asp   — sin(aspect), circular encoding
    4  cos_asp   — cos(aspect), circular encoding

Aspect is encoded as sin/cos (2 channels) because it is a circular variable:
0° and 360° are the same compass direction. Raw degree values would make
north-facing slopes appear maximally different on either side of 0°/360°.

Label: 1 = avalanche debris present in patch, 0 = clean snowpack.

Class imbalance (~1:8) is handled outside this class via
WeightedRandomSampler. Use get_sample_weights(dataset) to obtain per-sample
weights for the sampler.

Normalization statistics (channel-wise mean and std) are computed from the
training split and stored in data/splits/norm_stats.json. Pass
compute_stats=True on the training dataset to compute and save them; pass
stats_path to all datasets to apply the same normalization.

Goal 2 note (pre+post):
    When pre+post channels are added, set use_pre=True. This appends
    VH_pre and VV_pre as channels 5 and 6, giving a [7, 64, 64] tensor.
    The normalization stats file will include pre-channel stats.

Bi-temporal mode (for D4BiTemporalCNN):
    Set bitemporal=True. __getitem__ returns ((post_5ch, pre_5ch), label)
    where each patch is [5, 64, 64] with channels [VH, VV, slope, sin_asp, cos_asp].
    Terrain channels are shared (same physical location). Use a separate 7-channel
    norm_stats_bitemporal.json (computed automatically on first run).

Usage:
    from data_pipeline.dataset import AvalancheDataset, get_sample_weights

    train_ds = AvalancheDataset(
        split_csv="data/splits/train.csv",
        compute_stats=True,
        stats_path="data/splits/norm_stats.json",
    )
    val_ds = AvalancheDataset(
        split_csv="data/splits/val.csv",
        stats_path="data/splits/norm_stats.json",
    )

    sampler = WeightedRandomSampler(
        weights=get_sample_weights(train_ds),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# SAR dB clip range from arXiv:2502.18157
SAR_MIN_DB = -25.0
SAR_MAX_DB = -5.0


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _compute_channel_stats(
    records: list[dict],
    use_pre: bool,
) -> dict[str, list[float]]:
    """
    Compute per-channel mean and std over all training patches.
    Returns a dict with keys 'mean' and 'std', each a list of C floats.

    Runs in a single pass using Welford's online algorithm to avoid
    loading all patches into memory simultaneously.
    """
    log.info("Computing normalization statistics over %d patches...", len(records))

    # Welford accumulators: one per channel
    n_channels = 7 if use_pre else 5
    count = np.zeros(n_channels, dtype=np.float64)
    mean  = np.zeros(n_channels, dtype=np.float64)
    M2    = np.zeros(n_channels, dtype=np.float64)

    for record in records:
        patch = _load_patch(record, use_pre=use_pre, normalize=False)
        # patch: [C, H, W] tensor → flatten spatial dims
        arr = patch.numpy().reshape(n_channels, -1)  # [C, H*W]
        for c in range(n_channels):
            pixels = arr[c]
            for x in pixels:
                count[c] += 1
                delta = x - mean[c]
                mean[c] += delta / count[c]
                M2[c] += delta * (x - mean[c])

    std = np.sqrt(M2 / count)
    # Guard against zero std (constant channels, e.g. flat terrain)
    std = np.where(std < 1e-6, 1.0, std)

    return {
        "mean": mean.tolist(),
        "std":  std.tolist(),
        "channels": (
            ["VH_post", "VV_post", "slope", "sin_asp", "cos_asp"]
            if not use_pre else
            ["VH_post", "VV_post", "slope", "sin_asp", "cos_asp", "VH_pre", "VV_pre"]
        ),
    }


# ---------------------------------------------------------------------------
# Patch loading
# ---------------------------------------------------------------------------

def _load_patch(
    record: dict,
    use_pre: bool,
    normalize: bool,
    stats: dict | None = None,
) -> torch.Tensor:
    """
    Load one patch folder into a float32 tensor of shape [C, 64, 64].

    Channel order:
        0  VH_post
        1  VV_post
        2  slope
        3  sin(aspect)
        4  cos(aspect)
        [5  VH_pre]   (only if use_pre=True)
        [6  VV_pre]   (only if use_pre=True)
    """
    patch_dir = Path(record["patch_dir"])

    def read_band(tif_name: str, band: int = 1) -> np.ndarray:
        with rasterio.open(patch_dir / tif_name) as src:
            return src.read(band).astype(np.float32)

    # Post-event SAR — band 1=VH, band 2=VV (patchify.py writes VH first)
    vh_post = read_band("post.tif", band=1)
    vv_post = read_band("post.tif", band=2)

    # Clip SAR to valid dB range before any normalization
    vh_post = np.clip(vh_post, SAR_MIN_DB, SAR_MAX_DB)
    vv_post = np.clip(vv_post, SAR_MIN_DB, SAR_MAX_DB)

    # Terrain
    slope = read_band("slope.tif")

    # Aspect — circular encoding
    asp_deg = read_band("aspect.tif")
    asp_rad = np.deg2rad(asp_deg)
    sin_asp = np.sin(asp_rad)
    cos_asp = np.cos(asp_rad)

    channels = [vh_post, vv_post, slope, sin_asp, cos_asp]

    # Pre-event SAR (Goal 2)
    if use_pre:
        vh_pre = np.clip(read_band("pre.tif", band=1), SAR_MIN_DB, SAR_MAX_DB)
        vv_pre = np.clip(read_band("pre.tif", band=2), SAR_MIN_DB, SAR_MAX_DB)
        channels.extend([vh_pre, vv_pre])

    # Replace any remaining NaNs/infs with 0 before stacking
    for i, ch in enumerate(channels):
        channels[i] = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)

    patch = torch.from_numpy(np.stack(channels, axis=0))  # [C, H, W]

    if normalize and stats is not None:
        mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
        std  = torch.tensor(stats["std"],  dtype=torch.float32).view(-1, 1, 1)
        patch = (patch - mean) / std

    return patch


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AvalancheDataset(Dataset):
    """
    PyTorch Dataset for AvalCD patch-based avalanche classification.

    Args:
        split_csv:      Path to one of train.csv / val.csv / test_ood.csv.
        stats_path:     Path to norm_stats.json. If None, no normalization.
        compute_stats:  If True, compute and save stats to stats_path.
                        Only set this on the training dataset.
        use_pre:        If True, include pre-event VH/VV channels as channels 5–6.
        bitemporal:     If True, return ((post_5ch, pre_5ch), label) per sample
                        instead of (patch_7ch, label). Implies use_pre=True.
                        Use with D4BiTemporalCNN.
    """

    def __init__(
        self,
        split_csv: str | Path,
        stats_path: str | Path | None = None,
        compute_stats: bool = False,
        use_pre: bool = False,
        bitemporal: bool = False,
    ) -> None:
        self.split_csv  = Path(split_csv)
        self.stats_path = Path(stats_path) if stats_path else None
        self.bitemporal = bitemporal
        self.use_pre    = use_pre or bitemporal  # bitemporal requires pre channels

        # Load manifest rows
        with open(self.split_csv, newline="") as f:
            self.records = list(csv.DictReader(f))

        if not self.records:
            raise ValueError(f"Split CSV is empty: {self.split_csv}")

        self.labels = [int(r["label"]) for r in self.records]

        # Normalization stats
        self.stats: dict | None = None

        if compute_stats:
            if self.stats_path is None:
                raise ValueError("stats_path must be set when compute_stats=True")
            self.stats = _compute_channel_stats(self.records, use_pre=self.use_pre)
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.stats_path, "w") as f:
                json.dump(self.stats, f, indent=2)
            log.info("Normalization stats saved to %s", self.stats_path)

        elif self.stats_path is not None and self.stats_path.exists():
            with open(self.stats_path) as f:
                self.stats = json.load(f)
            log.info("Loaded normalization stats from %s", self.stats_path)

        else:
            log.warning(
                "No normalization stats available for %s. "
                "Run with compute_stats=True on the training split first.",
                self.split_csv.name,
            )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        patch = _load_patch(
            record,
            use_pre=self.use_pre,
            normalize=self.stats is not None,
            stats=self.stats,
        )
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.bitemporal:
            # patch is [7, H, W]: [VH_post, VV_post, slope, sin_asp, cos_asp, VH_pre, VV_pre]
            # Split into two 5-channel inputs for the shared encoder:
            #   post_5ch: [VH_post, VV_post, slope, sin_asp, cos_asp]  (channels 0-4)
            #   pre_5ch:  [VH_pre,  VV_pre,  slope, sin_asp, cos_asp]  (channels 5,6 + 2,3,4)
            # Terrain channels (slope, sin_asp, cos_asp) are shared — same physical location.
            post_5ch = patch[0:5]                             # [5, H, W]
            pre_5ch  = torch.cat([patch[5:7], patch[2:5]])   # [5, H, W]
            return (post_5ch, pre_5ch), label

        return patch, label

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_channels(self) -> int:
        if self.bitemporal:
            return 5   # per-branch input channels
        return 7 if self.use_pre else 5

    @property
    def n_positive(self) -> int:
        return sum(self.labels)

    @property
    def n_negative(self) -> int:
        return len(self.labels) - self.n_positive

    def class_counts(self) -> dict[str, int]:
        return {"positive": self.n_positive, "negative": self.n_negative}


# ---------------------------------------------------------------------------
# Weighted sampler helper
# ---------------------------------------------------------------------------

def get_sample_weights(dataset: AvalancheDataset) -> torch.Tensor:
    """
    Return a per-sample weight tensor for use with WeightedRandomSampler.
    Positive and negative samples receive equal total weight, so each
    training epoch sees a balanced 1:1 class ratio regardless of dataset
    imbalance — consistent with arXiv:2603.22658.

    Example:
        sampler = WeightedRandomSampler(
            weights=get_sample_weights(train_ds),
            num_samples=len(train_ds),
            replacement=True,
        )
    """
    n_pos = dataset.n_positive
    n_neg = dataset.n_negative

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Dataset has no {'positive' if n_pos == 0 else 'negative'} samples."
        )

    weight_pos = 1.0 / n_pos
    weight_neg = 1.0 / n_neg

    weights = torch.tensor(
        [weight_pos if label == 1 else weight_neg for label in dataset.labels],
        dtype=torch.float64,
    )
    return weights
