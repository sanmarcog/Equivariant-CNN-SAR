"""
calibrate.py

Post-hoc temperature scaling calibration for all trained models.

Temperature scaling fits a single scalar T > 0 on the validation set by
minimising NLL (binary cross-entropy):

    calibrated_logit = logit / T
    calibrated_prob  = sigmoid(logit / T)

T > 1 → model is overconfident (probabilities pushed toward 0.5)
T < 1 → model is underconfident (probabilities pushed toward 0 or 1)
T = 1 → no change

The temperature is fit on val logits saved by evaluate.py.  It is then
applied to both the val and test_ood logits.  AUC-ROC is unaffected by
calibration (it is rank-invariant), but calibration improves reliability
diagrams and expected calibration error (ECE).

Inputs (written by evaluate.py):
    results/<run_name>/scores_val.npz
    results/<run_name>/scores_test_ood.npz

Outputs:
    results/<run_name>/calibration.json   — T, ECE before/after, NLL before/after
    results/<run_name>/figures/reliability_<split>.png

Aggregate summary:
    python calibrate.py --summary

Usage:
    python calibrate.py --model c8 --data-fraction 1.0
    python calibrate.py --summary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_NAMES = ("c8", "so2", "d4", "o2", "d4_bitemporal", "cnn_bitemporal", "cnn", "aug", "resnet")


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def _nll(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Binary cross-entropy loss at temperature T (scalar, numpy)."""
    scaled = torch.tensor(logits / T, dtype=torch.float32)
    y      = torch.tensor(labels,     dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(scaled, y).item()


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    T_bounds: tuple[float, float] = (0.05, 50.0),
) -> float:
    """
    Find T that minimises NLL on (logits, labels) via scalar Brent search.
    Returns the optimal temperature T > 0.
    """
    result = minimize_scalar(
        lambda T: _nll(T, logits, labels),
        bounds=T_bounds,
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x)


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    ECE: weighted mean absolute difference between mean predicted probability
    and observed fraction of positives within equally spaced bins.
    """
    bin_edges  = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx    = np.digitize(probs, bin_edges[1:-1])   # 0 … n_bins-1
    ece        = 0.0
    n          = len(probs)

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        mean_conf = probs[mask].mean()
        mean_acc  = labels[mask].mean()
        ece += (mask.sum() / n) * abs(mean_conf - mean_acc)

    return float(ece)


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def _reliability_diagram(
    out_path: Path,
    probs_uncal: np.ndarray,
    probs_cal:   np.ndarray,
    labels:      np.ndarray,
    split_name:  str,
    n_bins:      int = 10,
) -> None:
    if not _MATPLOTLIB:
        return

    bin_edges  = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def _bin_stats(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean predicted prob and observed fraction per bin."""
        mean_conf = np.full(n_bins, np.nan)
        frac_pos  = np.full(n_bins, np.nan)
        idx = np.digitize(probs, bin_edges[1:-1])
        for b in range(n_bins):
            mask = idx == b
            if mask.sum() == 0:
                continue
            mean_conf[b] = probs[mask].mean()
            frac_pos[b]  = labels[mask].mean()
        return mean_conf, frac_pos

    conf_u, frac_u = _bin_stats(probs_uncal)
    conf_c, frac_c = _bin_stats(probs_cal)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, conf, frac, title in [
        (axes[0], conf_u, frac_u, "Uncalibrated"),
        (axes[1], conf_c, frac_c, "Calibrated"),
    ]:
        valid = ~np.isnan(conf)
        ax.bar(bin_centers[valid], frac[valid], width=1.0 / n_bins,
               alpha=0.7, color="steelblue", label="Fraction positive")
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="Perfect calibration")
        ax.scatter(conf[valid], frac[valid], color="tomato", zorder=5, s=30)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(f"{title} — {split_name}")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Fraction of positives")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single-run calibration
# ---------------------------------------------------------------------------

def calibrate_run(args: argparse.Namespace) -> None:
    frac_str = str(args.data_fraction).replace(".", "p")
    run_name = f"{args.model}_frac{frac_str}"
    run_dir  = Path(args.results_dir) / run_name
    fig_dir  = run_dir / "figures"

    val_scores_path = run_dir / "scores_val.npz"
    if not val_scores_path.exists():
        log.error(
            "Val scores not found: %s\n"
            "Run `python evaluate.py --model %s --data-fraction %s` first.",
            val_scores_path, args.model, args.data_fraction,
        )
        sys.exit(1)

    # --- Load val scores ---
    val_data = np.load(val_scores_path)
    val_logits = val_data["logits"].astype(np.float64)
    val_labels = val_data["labels"].astype(np.float64)

    # --- Fit temperature ---
    T = fit_temperature(val_logits, val_labels)
    log.info("Fitted temperature T = %.4f", T)

    # --- Evaluate calibration on each available split ---
    split_files = {
        "val":      run_dir / "scores_val.npz",
        "test_ood": run_dir / "scores_test_ood.npz",
    }

    cal_results: dict[str, dict] = {}

    for split_name, scores_path in split_files.items():
        if not scores_path.exists():
            log.warning("Scores not found, skipping %s: %s", split_name, scores_path)
            continue

        data   = np.load(scores_path)
        logits = data["logits"].astype(np.float64)
        labels = data["labels"].astype(np.float64)

        probs_uncal = 1.0 / (1.0 + np.exp(-logits))
        probs_cal   = 1.0 / (1.0 + np.exp(-logits / T))

        nll_before = _nll(1.0, logits, labels)
        nll_after  = _nll(T,   logits, labels)
        ece_before = expected_calibration_error(probs_uncal, labels)
        ece_after  = expected_calibration_error(probs_cal,   labels)

        log.info(
            "  %s: NLL %.4f → %.4f  |  ECE %.4f → %.4f",
            split_name, nll_before, nll_after, ece_before, ece_after,
        )

        cal_results[split_name] = {
            "nll_before": nll_before,
            "nll_after":  nll_after,
            "ece_before": ece_before,
            "ece_after":  ece_after,
        }

        # Save calibrated scores
        np.savez_compressed(
            run_dir / f"scores_{split_name}_calibrated.npz",
            logits_uncal=logits,
            logits_cal=(logits / T),
            probs_uncal=probs_uncal,
            probs_cal=probs_cal,
            labels=labels,
        )

        _reliability_diagram(
            fig_dir / f"reliability_{split_name}.png",
            probs_uncal, probs_cal, labels, split_name,
        )

    # --- Save calibration results ---
    result = {
        "model":         args.model,
        "data_fraction": args.data_fraction,
        "run_name":      run_name,
        "temperature":   T,
        "splits":        cal_results,
    }
    cal_path = run_dir / "calibration.json"
    with open(cal_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Calibration results saved to %s", cal_path)


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def print_summary(results_dir: str) -> None:
    root  = Path(results_dir)
    paths = sorted(root.glob("*/calibration.json"))

    if not paths:
        print(f"No calibration.json files found under {root}")
        sys.exit(0)

    rows: list[dict] = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        for split, m in d.get("splits", {}).items():
            rows.append({
                "model":    d["model"],
                "fraction": d["data_fraction"],
                "split":    split,
                "T":        d["temperature"],
                "nll_before": m["nll_before"],
                "nll_after":  m["nll_after"],
                "ece_before": m["ece_before"],
                "ece_after":  m["ece_after"],
            })

    header = (
        f"| {'Model':<8} | {'Frac':>5} | {'Split':<9} "
        f"| {'T':>6} "
        f"| {'NLL↑':>7} | {'NLL↓':>7} "
        f"| {'ECE↑':>6} | {'ECE↓':>6} |"
    )
    sep = "|" + "|".join(["-" * (w + 2) for w in [8, 5, 9, 6, 7, 7, 6, 6]]) + "|"
    print(header)
    print(sep)
    for r in rows:
        print(
            f"| {r['model']:<8} | {r['fraction']:>5.2f} | {r['split']:<9} "
            f"| {r['T']:>6.3f} "
            f"| {r['nll_before']:>7.4f} | {r['nll_after']:>7.4f} "
            f"| {r['ece_before']:>6.4f} | {r['ece_after']:>6.4f} |"
        )


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temperature scaling calibration.")
    p.add_argument("--model", choices=MODEL_NAMES,
                   help="Model to calibrate (required unless --summary).")
    p.add_argument("--data-fraction", type=float, default=1.0,
                   choices=[0.1, 0.25, 0.5, 1.0])
    p.add_argument("--results-dir", default="results")
    p.add_argument("--summary", action="store_true",
                   help="Print aggregate calibration table for all runs and exit.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if args.summary:
        print_summary(args.results_dir)
        sys.exit(0)

    if args.model is None:
        print("Error: --model is required unless --summary is set.")
        sys.exit(1)

    try:
        calibrate_run(args)
    except Exception:
        log.exception("Calibration failed.")
        sys.exit(1)
