"""
scripts/rotation_sensitivity.py

Diagnose rotation invariance in practice for trained models.

For each model checkpoint, the script:
  1. Loads 200 random patches from test_ood.csv
  2. Rotates each patch by 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
     using bilinear interpolation
  3. Runs inference on every (patch, angle) pair on CPU
  4. Reports:
     - Mean variance in predicted probability across the 8 angles per patch,
       averaged over all patches (0.0 = perfectly invariant)
     - AUC at each rotation angle separately (identical AUC = perfectly invariant)
  5. Saves a plot: AUC vs rotation angle, one line per model

A model that is perfectly rotation-invariant will have zero variance across
angles and identical AUC at every angle.  If SO(2) achieves very low variance
but lower overall AUC, it is over-constrained (invariant but not discriminative).
If SO(2) has HIGH variance, it is not achieving the equivariance it promises.

Usage:
    # Single model
    python scripts/rotation_sensitivity.py --models c8 \\
        --checkpoint-dir /gscratch/scrubbed/sanmarco/equivariant-sar/checkpoints \\
        --test-csv       /gscratch/scrubbed/sanmarco/equivariant-sar/data/splits/test_ood.csv \\
        --stats-path     /gscratch/scrubbed/sanmarco/equivariant-sar/data/splits/norm_stats.json

    # All available models (skips missing checkpoints)
    python scripts/rotation_sensitivity.py --models c8 so2 d4 cnn aug resnet \\
        --checkpoint-dir /gscratch/scrubbed/sanmarco/equivariant-sar/checkpoints \\
        --test-csv       /gscratch/scrubbed/sanmarco/equivariant-sar/data/splits/test_ood.csv \\
        --stats-path     /gscratch/scrubbed/sanmarco/equivariant-sar/data/splits/norm_stats.json \\
        --data-fraction  1.0

All inference runs on CPU — safe to run on the login node.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/rotation_sensitivity.py` from the project root
# or from inside the scripts/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from models.cnn_baseline import CNNBaseline
from models.cnn_augmented import AugmentedCNN
from models.resnet_baseline import ResNetBaseline
from models.equivariant_cnn import C8EquivariantCNN, SO2EquivariantCNN, D4EquivariantCNN


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
N_SAMPLES = 200

MODEL_META = {
    "c8":     {"label": "C8 equivariant",   "color": "#2196F3", "marker": "o", "ls": "-"},
    "so2":    {"label": "SO(2) equivariant", "color": "#4CAF50", "marker": "s", "ls": "-"},
    "d4":     {"label": "D4 equivariant",    "color": "#9C27B0", "marker": "^", "ls": "-"},
    "cnn":    {"label": "CNN baseline",      "color": "#F44336", "marker": "D", "ls": "--"},
    "aug":    {"label": "CNN + aug",         "color": "#FF9800", "marker": "v", "ls": "--"},
    "resnet": {"label": "ResNet-18",         "color": "#607D8B", "marker": "P", "ls": "-."},
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(name: str) -> nn.Module:
    if name == "c8":
        return C8EquivariantCNN()
    if name == "so2":
        return SO2EquivariantCNN()
    if name == "d4":
        return D4EquivariantCNN()
    if name == "cnn":
        return CNNBaseline()
    if name == "aug":
        return AugmentedCNN()
    if name == "resnet":
        return ResNetBaseline()
    raise ValueError(f"Unknown model: {name}")


def forward_logit(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def load_checkpoint(model_name: str, fraction: float, ckpt_dir: Path) -> nn.Module | None:
    frac_str = str(fraction).replace(".", "p")
    ckpt_path = ckpt_dir / f"{model_name}_frac{frac_str}" / "best.pt"
    if not ckpt_path.exists():
        print(f"  [skip] checkpoint not found: {ckpt_path}")
        return None
    model = build_model(model_name)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def rotate_batch(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """
    Rotate a [B, C, H, W] tensor by angle_deg using bilinear interpolation.
    Angle is counter-clockwise in degrees.  Corners are filled with zeros
    (border pixels are near zero after normalization for most channels).
    """
    if angle_deg == 0:
        return x
    angle_rad = torch.tensor(angle_deg * np.pi / 180.0)
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    # Affine matrix for rotation (2×3, batch size 1 broadcast)
    theta = torch.tensor(
        [[cos_a, -sin_a, 0.0],
         [sin_a,  cos_a, 0.0]],
        dtype=torch.float32,
    ).unsqueeze(0).expand(x.size(0), -1, -1)  # [B, 2, 3]
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse_model(
    model: nn.Module,
    patches: torch.Tensor,   # [N, C, H, W]
    labels: np.ndarray,      # [N]
    batch_size: int = 16,
) -> dict:
    """
    Run inference for each angle on all patches, in mini-batches.
    Returns:
        angle_auc   : {angle: auc}
        mean_var    : mean variance across angles per patch, averaged over patches
        probs       : np.ndarray [N, 8]  predicted probabilities
    """
    n = patches.size(0)
    n_angles = len(ANGLES)
    probs = np.zeros((n, n_angles), dtype=np.float32)

    with torch.no_grad():
        for ai, angle in enumerate(ANGLES):
            preds = []
            for start in range(0, n, batch_size):
                batch   = patches[start : start + batch_size]
                rotated = rotate_batch(batch, angle)
                logits  = forward_logit(model, rotated)
                p       = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                preds.append(p)
            probs[:, ai] = np.concatenate(preds)

    angle_auc = {}
    for ai, angle in enumerate(ANGLES):
        p_col = probs[:, ai]
        if len(np.unique(labels)) > 1:
            angle_auc[angle] = roc_auc_score(labels, p_col)
        else:
            angle_auc[angle] = float("nan")

    # Variance across angles for each patch, then mean over patches
    per_patch_var = probs.var(axis=1)          # [N]
    mean_var      = float(per_patch_var.mean())

    return {"angle_auc": angle_auc, "mean_var": mean_var, "probs": probs}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results: dict[str, dict]) -> None:
    angle_w = 6
    col_w   = 9

    print()
    print("=" * 78)
    print("Rotation Sensitivity Analysis")
    print("=" * 78)

    # Header
    header = f"  {'Model':<22}  {'MeanVar':>9}"
    for a in ANGLES:
        header += f"  {f'AUC@{a}°':>{col_w}}"
    print(header)
    print("  " + "-" * 74)

    for model_name, res in results.items():
        label = MODEL_META[model_name]["label"]
        row   = f"  {label:<22}  {res['mean_var']:>9.5f}"
        for a in ANGLES:
            auc = res["angle_auc"].get(a, float("nan"))
            if np.isnan(auc):
                row += f"  {'—':>{col_w}}"
            else:
                row += f"  {auc:>{col_w}.4f}"
        print(row)

    print("=" * 78)
    print()
    print("MeanVar: mean variance in predicted probability across 8 rotation angles,")
    print("         averaged over all test patches. 0.0 = perfectly rotation-invariant.")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_auc_vs_angle(
    results: dict[str, dict],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.array(ANGLES)
    for model_name, res in results.items():
        meta = MODEL_META[model_name]
        y    = np.array([res["angle_auc"].get(a, float("nan")) for a in ANGLES])
        ax.plot(
            x, y,
            color=meta["color"],
            linestyle=meta["ls"],
            linewidth=1.8,
            marker=meta["marker"],
            markersize=7,
            label=f"{meta['label']}  (var={res['mean_var']:.4f})",
        )

    ax.set_xlabel("Rotation angle (degrees)", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("AUC vs. rotation angle — rotation sensitivity analysis", fontsize=13)
    ax.set_xticks(ANGLES)
    ax.set_xticklabels([f"{a}°" for a in ANGLES])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(True, linestyle="--", alpha=0.4)

    all_aucs = [
        v for res in results.values()
        for v in res["angle_auc"].values()
        if not np.isnan(v)
    ]
    if all_aucs:
        ax.set_ylim(max(0.0, min(all_aucs) - 0.05), min(1.0, max(all_aucs) + 0.05))

    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rotation sensitivity analysis.")
    p.add_argument("--models", nargs="+", default=list(MODEL_META),
                   help="Models to evaluate (default: all six).")
    p.add_argument("--data-fraction", type=float, default=1.0,
                   help="Which checkpoint fraction to load (default: 1.0).")
    p.add_argument("--checkpoint-dir",  # Hyak default; override for local use
                   default="/gscratch/scrubbed/sanmarco/equivariant-sar/checkpoints")
    p.add_argument("--test-csv",        # Hyak default; override for local use
                   default="/gscratch/scrubbed/sanmarco/equivariant-sar/data/splits/test_ood.csv")
    p.add_argument("--stats-path",      # Hyak default; override for local use
                   default="/gscratch/scrubbed/sanmarco/equivariant-sar/data/splits/norm_stats.json")
    p.add_argument("--n-samples", type=int, default=N_SAMPLES,
                   help="Number of test patches to sample (default: 200).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-batch-size", type=int, default=32,
                   help="Max patches per inference batch (default: 32). "
                        "Reduce if SO2 / large models OOM on the login node.")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: results/rotation_sensitivity.png).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    out_path = Path(args.out) if args.out else \
        Path("/gscratch/scrubbed/sanmarco/equivariant-sar/results/rotation_sensitivity.png")

    # ---- Load dataset ----
    print(f"Loading test dataset from {args.test_csv}")
    with open(args.test_csv, newline="") as f:
        records = list(csv.DictReader(f))

    all_labels = [int(r["label"]) for r in records]

    # Sample N patches, ensuring both classes are represented
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    pos_idx = [i for i, l in enumerate(all_labels) if l == 1]
    neg_idx = [i for i, l in enumerate(all_labels) if l == 0]
    n_pos = min(args.n_samples // 2, len(pos_idx))
    n_neg = min(args.n_samples - n_pos, len(neg_idx))
    sampled_idx = random.sample(pos_idx, n_pos) + random.sample(neg_idx, n_neg)
    random.shuffle(sampled_idx)
    print(f"Sampled {len(sampled_idx)} patches ({n_pos} positive, {n_neg} negative)")

    # Load normalization stats
    with open(args.stats_path) as f:
        stats = json.load(f)
    norm_mean = np.array(stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
    norm_std  = np.array(stats["std"],  dtype=np.float32).reshape(-1, 1, 1)

    SAR_MIN, SAR_MAX = -25.0, -5.0

    patches_list, labels_list = [], []
    for i in sampled_idx:
        r = records[i]
        patch_dir = Path(r["patch_dir"])

        def read_tif(name: str, band: int = 0) -> np.ndarray:
            # Use tifffile — already available in the pytorch container
            import tifffile
            img = tifffile.imread(str(patch_dir / name))
            if img.ndim == 3:          # (bands, H, W) or (H, W, bands)
                if img.shape[0] <= 8:  # (bands, H, W)
                    return img[band].astype(np.float32)
                else:                  # (H, W, bands)
                    return img[:, :, band].astype(np.float32)
            return img.astype(np.float32)

        vh  = np.clip(read_tif("post.tif", 0), SAR_MIN, SAR_MAX)
        vv  = np.clip(read_tif("post.tif", 1), SAR_MIN, SAR_MAX)
        slp = read_tif("slope.tif", 0)
        asp = read_tif("aspect.tif", 0)
        asp_rad = np.deg2rad(asp)
        sin_asp = np.sin(asp_rad)
        cos_asp = np.cos(asp_rad)

        patch = np.stack([vh, vv, slp, sin_asp, cos_asp], axis=0)  # [5, H, W]
        patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
        patch = (patch - norm_mean) / norm_std

        patches_list.append(torch.from_numpy(patch))
        labels_list.append(int(r["label"]))

    patches = torch.stack(patches_list)           # [N, 5, H, W]
    labels  = np.array(labels_list, dtype=np.int32)

    # ---- Analyse each model ----
    results: dict[str, dict] = {}
    for model_name in args.models:
        if model_name not in MODEL_META:
            print(f"Unknown model '{model_name}' — skipping.")
            continue
        print(f"\nAnalysing {model_name} (frac={args.data_fraction}) ...")
        model = load_checkpoint(model_name, args.data_fraction, ckpt_dir)
        if model is None:
            continue
        res = analyse_model(model, patches, labels, batch_size=args.max_batch_size)
        results[model_name] = res
        print(f"  mean_var = {res['mean_var']:.5f}")
        for a, auc in res["angle_auc"].items():
            print(f"  AUC @ {a:>3}° = {auc:.4f}")

    if not results:
        print("No models loaded — nothing to report.")
        return

    print_results(results)
    plot_auc_vs_angle(results, out_path)

    # Save JSON summary alongside the plot
    json_path = out_path.with_suffix(".json")
    summary = {
        m: {
            "mean_var": res["mean_var"],
            "angle_auc": {str(k): v for k, v in res["angle_auc"].items()},
        }
        for m, res in results.items()
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
