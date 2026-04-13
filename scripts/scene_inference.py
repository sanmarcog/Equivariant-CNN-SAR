"""
scripts/scene_inference.py

Full-scene sliding-window inference for the Tromsø OOD test scene.

Tiles the full SAR scene into 64×64 patches with configurable overlap,
runs inference with a trained model, and stitches predictions back into a
full-scene probability map saved as a GeoTIFF.

For bi-temporal models (d4_bitemporal) the pre-event SAR bands are also
loaded and passed as the second branch input.

Channel layout (matches training dataset.py):
    Post branch: [VH_post, VV_post, slope, sin_asp, cos_asp]
    Pre branch:  [VH_pre,  VV_pre,  slope, sin_asp, cos_asp]

Usage:
    python scripts/scene_inference.py \\
        --model d4_bitemporal \\
        --data-fraction 0.5 \\
        --scene-dir data/raw/Tromso_20241220 \\
        --checkpoint-dir checkpoints \\
        --stats-path data/splits/norm_stats_bitemporal.json \\
        --output results/scene/tromso_d4bt_frac0p5_prob.tif

    python scripts/scene_inference.py --model d4 --data-fraction 1.0 \\
        --stats-path data/splits/norm_stats.json \\
        --output results/scene/tromso_d4_frac1p0_prob.tif
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import rasterio
    from rasterio.transform import from_bounds
except ImportError:
    print("rasterio is required. Install it or run inside the Apptainer container.")
    sys.exit(1)

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.cnn_baseline import CNNBaseline
from models.cnn_augmented import AugmentedCNN
from models.resnet_baseline import ResNetBaseline
from models.equivariant_cnn import (
    C8EquivariantCNN, SO2EquivariantCNN, D4EquivariantCNN,
    O2EquivariantCNN, D4BiTemporalCNN,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Must match training
SAR_MIN_DB = -25.0
SAR_MAX_DB = -5.0
PATCH_SIZE  = 64


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(name: str, in_channels: int = 5) -> nn.Module:
    if name == "c8":          return C8EquivariantCNN(in_channels=in_channels)
    if name == "so2":         return SO2EquivariantCNN(in_channels=in_channels)
    if name == "d4":          return D4EquivariantCNN(in_channels=in_channels)
    if name == "o2":          return O2EquivariantCNN(in_channels=in_channels)
    if name == "d4_bitemporal": return D4BiTemporalCNN(in_channels=in_channels)
    if name == "cnn":         return CNNBaseline(in_channels=in_channels)
    if name == "aug":         return AugmentedCNN(in_channels=in_channels)
    if name == "resnet":      return ResNetBaseline(in_channels=in_channels, pretrained=False)
    raise ValueError(f"Unknown model: {name}")


def forward_logit(model: nn.Module, x_post: torch.Tensor,
                  x_pre: torch.Tensor | None) -> torch.Tensor:
    """Return scalar logit tensor [B, 1]."""
    base = model.module if isinstance(model, nn.DataParallel) else model
    if getattr(base, "bitemporal", False):
        logit, _ = model(x_post, x_pre, return_orientation=False)
        return logit
    if hasattr(base, "orientation_conv"):
        logit, _ = model(x_post, return_orientation=False)
        return logit
    return model(x_post)


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------

def load_scene_bands(scene_dir: Path, is_bitemporal: bool) -> dict[str, np.ndarray]:
    """
    Load all required bands as float32 arrays of shape [H, W].

    Returns dict with keys: vh_post, vv_post, slope, sin_asp, cos_asp,
    and optionally vh_pre, vv_pre (if is_bitemporal).
    Also returns 'profile' (rasterio meta) and 'nodata_mask' [H, W] bool.
    """
    def read(path: Path, band: int = 1) -> tuple[np.ndarray, dict]:
        with rasterio.open(path) as src:
            arr = src.read(band).astype(np.float32)
            profile = src.profile
        return arr, profile

    def find(pattern: str) -> Path:
        matches = sorted(scene_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No file matching {pattern} in {scene_dir}")
        return matches[0]

    post_vh_path = find("*postVH*")
    post_vv_path = find("*postVV*")
    slp_path     = find("*SLP*")
    asp_path     = find("*ASP*")

    log.info("Loading post-event SAR: %s", post_vh_path.name)
    vh_post, profile = read(post_vh_path)
    vv_post, _       = read(post_vv_path)
    slope,   _       = read(slp_path)
    asp_deg, _       = read(asp_path)

    # Clip SAR to training range
    vh_post = np.clip(vh_post, SAR_MIN_DB, SAR_MAX_DB)
    vv_post = np.clip(vv_post, SAR_MIN_DB, SAR_MAX_DB)

    # Aspect: raw degrees → circular encoding (matches dataset.py)
    asp_rad = np.deg2rad(asp_deg)
    sin_asp = np.sin(asp_rad)
    cos_asp = np.cos(asp_rad)

    # NaN mask: any pixel invalid in any band → mask out
    nodata_mask = (
        np.isnan(vh_post) | np.isnan(vv_post) | np.isnan(slope) | np.isnan(asp_deg)
    )

    # Fill NaNs with 0 (will be masked in output)
    def fill(arr: np.ndarray) -> np.ndarray:
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    bands = {
        "vh_post":  fill(vh_post),
        "vv_post":  fill(vv_post),
        "slope":    fill(slope),
        "sin_asp":  fill(sin_asp),
        "cos_asp":  fill(cos_asp),
        "nodata_mask": nodata_mask,
        "profile":  profile,
    }

    if is_bitemporal:
        pre_vh_path = find("*preVH*")
        pre_vv_path = find("*preVV*")
        log.info("Loading pre-event SAR: %s", pre_vh_path.name)
        vh_pre, _ = read(pre_vh_path)
        vv_pre, _ = read(pre_vv_path)
        vh_pre = np.clip(vh_pre, SAR_MIN_DB, SAR_MAX_DB)
        vv_pre = np.clip(vv_pre, SAR_MIN_DB, SAR_MAX_DB)
        nodata_mask |= (np.isnan(vh_pre) | np.isnan(vv_pre))
        bands["vh_pre"] = fill(vh_pre)
        bands["vv_pre"] = fill(vv_pre)
        bands["nodata_mask"] = nodata_mask

    return bands


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def load_norm_stats(stats_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) arrays shaped [C]."""
    with open(stats_path) as f:
        stats = json.load(f)
    return np.array(stats["mean"], dtype=np.float32), np.array(stats["std"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------

def extract_patch_tensor(
    bands: dict,
    row: int,
    col: int,
    H: int,
    W: int,
    is_bitemporal: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Extract a [5, P, P] float32 patch (post) and optionally pre from global bands.

    Handles boundary by padding with edge reflection.
    Returns (post_patch, pre_patch_or_None) as numpy arrays [5, P, P].
    """
    P = PATCH_SIZE

    def crop(arr: np.ndarray) -> np.ndarray:
        """Crop [H, W] with reflection padding for out-of-bounds."""
        r0, r1 = row, row + P
        c0, c1 = col, col + P

        pad_top    = max(0, -r0)
        pad_bottom = max(0, r1 - H)
        pad_left   = max(0, -c0)
        pad_right  = max(0, c1 - W)

        r0c, r1c = max(r0, 0), min(r1, H)
        c0c, c1c = max(c0, 0), min(c1, W)

        patch = arr[r0c:r1c, c0c:c1c]
        if pad_top or pad_bottom or pad_left or pad_right:
            patch = np.pad(
                patch,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="reflect",
            )
        return patch

    post_patch = np.stack([
        crop(bands["vh_post"]),
        crop(bands["vv_post"]),
        crop(bands["slope"]),
        crop(bands["sin_asp"]),
        crop(bands["cos_asp"]),
    ], axis=0).astype(np.float32)  # [5, P, P]

    pre_patch = None
    if is_bitemporal:
        pre_patch = np.stack([
            crop(bands["vh_pre"]),
            crop(bands["vv_pre"]),
            crop(bands["slope"]),
            crop(bands["sin_asp"]),
            crop(bands["cos_asp"]),
        ], axis=0).astype(np.float32)

    return post_patch, pre_patch


def run_scene_inference(
    model: nn.Module,
    bands: dict,
    mean: np.ndarray,
    std: np.ndarray,
    is_bitemporal: bool,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Sliding-window inference over the full scene.

    Returns probability map [H, W] as float32, range [0, 1].
    Overlapping patches are averaged (linear blending).
    """
    H, W = bands["vh_post"].shape
    P = PATCH_SIZE

    # Accumulator and count map for averaging overlapping predictions
    prob_sum  = np.zeros((H, W), dtype=np.float64)
    count_map = np.zeros((H, W), dtype=np.float64)

    # Generate tile grid
    row_starts = list(range(0, H - P + 1, stride))
    if row_starts[-1] + P < H:
        row_starts.append(H - P)
    col_starts = list(range(0, W - P + 1, stride))
    if col_starts[-1] + P < W:
        col_starts.append(W - P)

    tiles = [(r, c) for r in row_starts for c in col_starts]
    n_tiles = len(tiles)
    log.info("Scene %d×%d, patch %d, stride %d → %d tiles", H, W, P, stride, n_tiles)

    model.eval()
    # Normalisation tensors
    mean_t = torch.tensor(mean, dtype=torch.float32).to(device).view(-1, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32).to(device).view(-1, 1, 1)

    # Pre + post split for bitemporal normalisation
    # Norm stats channels: [VH_post, VV_post, slope, sin_asp, cos_asp, VH_pre, VV_pre]
    if is_bitemporal:
        mean_post = mean_t[:5]   # first 5 channels
        std_post  = std_t[:5]
        mean_pre  = torch.cat([mean_t[5:7], mean_t[2:5]])  # [VH_pre, VV_pre, slope, sin_asp, cos_asp]
        std_pre   = torch.cat([std_t[5:7],  std_t[2:5]])
    else:
        mean_post = mean_t
        std_post  = std_t

    processed = 0
    with torch.no_grad():
        # Collect batch
        batch_rows, batch_cols = [], []
        batch_post_list, batch_pre_list = [], []

        def flush_batch():
            nonlocal processed
            if not batch_post_list:
                return

            post_batch = torch.stack(batch_post_list, dim=0).to(device)  # [B, 5, P, P]
            post_batch = (post_batch - mean_post) / std_post

            if is_bitemporal:
                pre_batch = torch.stack(batch_pre_list, dim=0).to(device)
                pre_batch = (pre_batch - mean_pre) / std_pre
                logits = forward_logit(model, post_batch, pre_batch)
            else:
                logits = forward_logit(model, post_batch, None)

            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # [B]

            for i, (r, c) in enumerate(zip(batch_rows, batch_cols)):
                r0, r1 = max(r, 0), min(r + P, H)
                c0, c1 = max(c, 0), min(c + P, W)
                # Contribution from this tile to the valid region
                tr0 = r0 - r   # offset within patch
                tc0 = c0 - c
                tr1 = tr0 + (r1 - r0)
                tc1 = tc0 + (c1 - c0)
                prob_sum[r0:r1, c0:c1]  += probs[i]
                count_map[r0:r1, c0:c1] += 1.0

            processed += len(batch_rows)
            if processed % max(1000, n_tiles // 10) < len(batch_rows):
                log.info("  %d / %d tiles (%.0f%%)", processed, n_tiles,
                         100 * processed / n_tiles)

            batch_rows.clear()
            batch_cols.clear()
            batch_post_list.clear()
            batch_pre_list.clear()

        for (r, c) in tiles:
            post_np, pre_np = extract_patch_tensor(bands, r, c, H, W, is_bitemporal)
            batch_post_list.append(torch.from_numpy(post_np))
            if is_bitemporal:
                batch_pre_list.append(torch.from_numpy(pre_np))
            batch_rows.append(r)
            batch_cols.append(c)

            if len(batch_rows) >= batch_size:
                flush_batch()

        flush_batch()

    # Average overlapping predictions
    valid = count_map > 0
    prob_map = np.zeros((H, W), dtype=np.float32)
    prob_map[valid] = (prob_sum[valid] / count_map[valid]).astype(np.float32)

    # Zero out nodata pixels
    prob_map[bands["nodata_mask"]] = 0.0

    log.info("Scene inference complete. Prob map range: [%.4f, %.4f]",
             prob_map.min(), prob_map.max())
    return prob_map


# ---------------------------------------------------------------------------
# Save GeoTIFF
# ---------------------------------------------------------------------------

def save_geotiff(prob_map: np.ndarray, profile: dict, out_path: Path) -> None:
    """Save probability map as single-band float32 GeoTIFF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update({
        "count":   1,
        "dtype":   "float32",
        "compress": "lzw",
        "driver":  "GTiff",
    })
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(prob_map, 1)
    log.info("Probability map saved to %s", out_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-scene sliding-window inference.")
    p.add_argument("--model", required=True,
                   choices=["c8", "so2", "d4", "o2", "d4_bitemporal", "cnn", "aug", "resnet"])
    p.add_argument("--data-fraction", type=float, default=1.0,
                   choices=[0.1, 0.25, 0.5, 1.0])
    p.add_argument("--scene-dir", default="data/raw/Tromso_20241220",
                   help="Directory containing scene GeoTIFFs.")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--stats-path", default="data/splits/norm_stats_bitemporal.json",
                   help="Normalization stats JSON (7-ch for bitemporal, 5-ch otherwise).")
    p.add_argument("--output", default=None,
                   help="Output GeoTIFF path. Default: results/scene/<run_name>_prob.tif")
    p.add_argument("--stride", type=int, default=32,
                   help="Sliding window stride in pixels (default: 32 = 50%% overlap).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    frac_str  = str(args.data_fraction).replace(".", "p")
    run_name  = f"{args.model}_frac{frac_str}"
    ckpt_path = Path(args.checkpoint_dir) / run_name / "best.pt"

    if args.output is None:
        out_path = Path("results/scene") / f"{run_name}_prob.tif"
    else:
        out_path = Path(args.output)

    device = torch.device(args.device)

    # Load model
    log.info("Building model: %s", args.model)
    model = build_model(args.model)
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    log.info("Loaded checkpoint: epoch %d  best_val_auc=%.4f",
             ckpt.get("epoch", -1), ckpt.get("best_val_auc", float("nan")))

    # Load norm stats
    stats_path = Path(args.stats_path)
    if not stats_path.exists():
        log.error("Norm stats not found: %s", stats_path)
        sys.exit(1)
    mean, std = load_norm_stats(stats_path)
    log.info("Loaded norm stats from %s (%d channels)", stats_path, len(mean))

    # Load scene
    is_bitemporal = (args.model == "d4_bitemporal")
    scene_dir = Path(args.scene_dir)
    log.info("Loading scene from %s", scene_dir)
    bands = load_scene_bands(scene_dir, is_bitemporal)
    H, W = bands["vh_post"].shape
    log.info("Scene size: %d × %d  nodata pixels: %d",
             H, W, bands["nodata_mask"].sum())

    # Run inference
    prob_map = run_scene_inference(
        model, bands, mean, std, is_bitemporal,
        stride=args.stride,
        batch_size=args.batch_size,
        device=device,
    )

    # Save
    save_geotiff(prob_map, bands["profile"], out_path)
    log.info("Done. Output: %s", out_path)


if __name__ == "__main__":
    main()
