"""
scripts/threshold_analysis.py

Threshold analysis for all completed evaluation runs.

For each model/fraction with a scores_*.npz file:
  - Plots the full precision-recall curve
  - Reports F1 and F2 at fixed thresholds: 0.3, 0.4, 0.5
  - Reports F1 and F2 at threshold-optimal points: optimal-F1, optimal-F2
  - Compares to the "at_optimal" threshold stored in metrics.json
    (which is Youden's J = TPR + TNR - 1, not F1 or F2 optimal)

Output:
  results/threshold_analysis/summary.csv          — all thresholds, all models
  results/threshold_analysis/pr_curves.png        — precision-recall grid
  results/threshold_analysis/threshold_table.txt  — human-readable table

Usage:
    python scripts/threshold_analysis.py
    python scripts/threshold_analysis.py --results-dir /path/to/results
    python scripts/threshold_analysis.py --split test_ood  # default
    python scripts/threshold_analysis.py --split val
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def f_beta(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    """F-beta score from precision/recall arrays. Handles zeros."""
    b2 = beta ** 2
    denom = b2 * precision + recall
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(denom > 0, (1 + b2) * precision * recall / denom, 0.0)
    return score


def metrics_at_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    f1_val = f_beta(np.array([prec]), np.array([rec]), beta=1.0)[0]
    f2_val = f_beta(np.array([prec]), np.array([rec]), beta=2.0)[0]

    return {
        "threshold": threshold,
        "precision": prec,
        "recall":    rec,
        "specificity": spec,
        "f1": f1_val,
        "f2": f2_val,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def pr_curve(
    probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precision, recall, thresholds) sorted by recall ascending."""
    thresholds = np.unique(np.concatenate([probs, [0.0, 1.0]]))
    thresholds = np.sort(thresholds)[::-1]   # descending: high thr → low recall

    precisions, recalls = [], []
    for thr in thresholds:
        m = metrics_at_threshold(probs, labels, thr)
        precisions.append(m["precision"])
        recalls.append(m["recall"])

    return np.array(precisions), np.array(recalls), thresholds


def optimal_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    metric: str = "f2",
) -> tuple[float, float]:
    """
    Return (best_threshold, best_score) maximising the given metric over all
    unique probability thresholds.
    """
    thresholds = np.unique(probs)
    best_thr, best_score = 0.5, 0.0
    for thr in thresholds:
        m = metrics_at_threshold(probs, labels, thr)
        score = m[metric]
        if score > best_score:
            best_score = score
            best_thr   = thr
    return best_thr, best_score


# ---------------------------------------------------------------------------
# Discover runs
# ---------------------------------------------------------------------------

def discover_runs(results_dir: Path, split: str) -> list[dict]:
    """
    Scan results_dir for completed runs that have scores_{split}.npz and
    metrics.json.
    """
    runs = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        scores_path  = run_dir / f"scores_{split}.npz"
        metrics_path = run_dir / "metrics.json"

        if not scores_path.exists() or not metrics_path.exists():
            continue

        # Parse model and fraction from directory name: {model}_frac{frac}
        name = run_dir.name
        if "_frac" not in name:
            continue
        parts = name.rsplit("_frac", maxsplit=1)
        if len(parts) != 2:
            continue
        model    = parts[0]
        frac_str = parts[1]
        try:
            fraction = float(frac_str.replace("p", "."))
        except ValueError:
            continue

        runs.append({
            "model":        model,
            "fraction":     fraction,
            "run_dir":      run_dir,
            "scores_path":  scores_path,
            "metrics_path": metrics_path,
        })
    return runs


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

FIXED_THRESHOLDS = [0.3, 0.4, 0.5]


def analyse_run(run: dict) -> dict | None:
    """Load scores and compute threshold metrics. Returns result dict."""
    data = np.load(run["scores_path"])
    logits = data["logits"].astype(np.float64)
    labels = data["labels"].astype(np.float64)
    probs  = sigmoid(logits)

    if labels.sum() == 0 or (labels == 0).sum() == 0:
        return None   # degenerate split

    with open(run["metrics_path"]) as f:
        metrics_json = json.load(f)

    # Youden's J threshold stored in metrics.json
    splits_data = metrics_json.get("splits", {})
    youden_thr  = float("nan")
    stored_auc  = float("nan")
    for split_name in ("test_ood", "val"):
        if split_name in splits_data:
            youden_thr = splits_data[split_name].get(
                "at_optimal", {}
            ).get("threshold", float("nan"))
            stored_auc = splits_data[split_name].get("auc_roc", float("nan"))
            break

    # Optimal F1 and F2 thresholds
    opt_f1_thr, opt_f1 = optimal_threshold(probs, labels, "f1")
    opt_f2_thr, opt_f2 = optimal_threshold(probs, labels, "f2")

    result = {
        "model":        run["model"],
        "fraction":     run["fraction"],
        "auc_roc":      stored_auc,
        "youden_thr":   youden_thr,
        "opt_f1_thr":   opt_f1_thr,
        "opt_f2_thr":   opt_f2_thr,
        "opt_f1":       opt_f1,
        "opt_f2":       opt_f2,
        "probs":        probs,
        "labels":       labels,
        "at_thresholds": {},
    }

    for thr in FIXED_THRESHOLDS + [youden_thr, opt_f1_thr, opt_f2_thr]:
        if np.isnan(thr):
            continue
        result["at_thresholds"][round(float(thr), 4)] = metrics_at_threshold(
            probs, labels, thr
        )

    return result


# ---------------------------------------------------------------------------
# Output: summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(results: list[dict], out_path: Path) -> None:
    fieldnames = [
        "model", "fraction", "auc_roc",
        "youden_thr",
        "f1_at_05", "f2_at_05",
        "f1_at_04", "f2_at_04",
        "f1_at_03", "f2_at_03",
        "opt_f1_thr", "f1_at_opt_f1",
        "opt_f2_thr", "f2_at_opt_f2",
        "gap_f2_05_vs_opt",    # f2@opt_f2 − f2@0.5
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            def _get(thr, key):
                m = r["at_thresholds"].get(round(thr, 4))
                return m[key] if m else float("nan")

            row = {
                "model":    r["model"],
                "fraction": r["fraction"],
                "auc_roc":  r["auc_roc"],
                "youden_thr":      r["youden_thr"],
                "f1_at_05":        _get(0.5,  "f1"),
                "f2_at_05":        _get(0.5,  "f2"),
                "f1_at_04":        _get(0.4,  "f1"),
                "f2_at_04":        _get(0.4,  "f2"),
                "f1_at_03":        _get(0.3,  "f1"),
                "f2_at_03":        _get(0.3,  "f2"),
                "opt_f1_thr":      r["opt_f1_thr"],
                "f1_at_opt_f1":    r["opt_f1"],
                "opt_f2_thr":      r["opt_f2_thr"],
                "f2_at_opt_f2":    r["opt_f2"],
                "gap_f2_05_vs_opt": r["opt_f2"] - _get(0.5, "f2"),
            }
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items()})


# ---------------------------------------------------------------------------
# Output: human-readable table
# ---------------------------------------------------------------------------

def write_table(results: list[dict], out_path: Path) -> None:
    lines = []
    header = (
        f"{'Model':<14} {'Frac':>5} | {'AUC':>6} "
        f"| {'F1@0.5':>7} {'F2@0.5':>7} "
        f"| {'F2@opt':>7} {'thr':>6} "
        f"| {'Gain':>6}"
    )
    sep = "-" * len(header)
    lines += [header, sep]

    def _f(v, w=7):
        return f"{v:{w}.4f}" if (isinstance(v, float) and not np.isnan(v)) else f"{'—':>{w}}"

    for r in sorted(results, key=lambda x: (x["model"], x["fraction"])):
        def _get(thr, key):
            m = r["at_thresholds"].get(round(thr, 4))
            return m[key] if m else float("nan")

        f2_05  = _get(0.5, "f2")
        f2_opt = r["opt_f2"]
        gain   = f2_opt - f2_05 if not np.isnan(f2_05) else float("nan")

        lines.append(
            f"{r['model']:<14} {r['fraction']:>5.2f} | {_f(r['auc_roc'],6)} "
            f"| {_f(_get(0.5,'f1'))} {_f(f2_05)} "
            f"| {_f(f2_opt)} {_f(r['opt_f2_thr'],6)} "
            f"| {_f(gain,6)}"
        )

    lines.append(sep)
    lines.append("Gain = F2@opt_F2 − F2@0.5  (positive = lower threshold helps)")

    text = "\n".join(lines)
    out_path.write_text(text)
    print(text)


# ---------------------------------------------------------------------------
# Output: PR curve plots
# ---------------------------------------------------------------------------

def plot_pr_curves(results: list[dict], out_path: Path) -> None:
    models = sorted({r["model"] for r in results})
    n_models = len(models)
    if n_models == 0:
        return

    ncols = min(4, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    cmap = plt.cm.viridis_r
    fracs = [0.1, 0.25, 0.5, 1.0]

    for col, model in enumerate(models):
        row = col // ncols
        c   = col % ncols
        ax  = axes[row][c]

        model_runs = [r for r in results if r["model"] == model]

        for i, r in enumerate(sorted(model_runs, key=lambda x: x["fraction"])):
            prec, rec, _ = pr_curve(r["probs"], r["labels"])
            color = cmap(i / max(len(fracs) - 1, 1))
            ax.plot(rec, prec, color=color,
                    label=f"{r['fraction']:.0%}", lw=1.5, alpha=0.85)

            # Mark threshold points
            for thr, marker, ms in [(0.5, "o", 6), (r["opt_f2_thr"], "^", 7)]:
                m = r["at_thresholds"].get(round(thr, 4))
                if m:
                    ax.scatter(m["recall"], m["precision"],
                               color=color, marker=marker, s=ms**2, zorder=5)

        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Recall", fontsize=8)
        ax.set_ylabel("Precision", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, title="Data frac", loc="upper right")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for col in range(len(models), nrows * ncols):
        row = col // ncols
        c   = col % ncols
        axes[row][c].set_visible(False)

    fig.suptitle(
        "Precision-Recall curves  (○ = thr 0.5, △ = optimal-F2 threshold)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PR curves saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Threshold analysis across all runs.")
    p.add_argument("--results-dir", default="results",
                   help="Root results directory (default: results/).")
    p.add_argument("--split", default="test_ood",
                   choices=["test_ood", "val"],
                   help="Which split to analyse (default: test_ood).")
    return p.parse_args()


def main() -> None:
    args        = parse_args()
    results_dir = Path(args.results_dir)
    out_dir     = results_dir / "threshold_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(results_dir, split=args.split)
    if not runs:
        print(f"No completed runs found in {results_dir} for split={args.split}")
        return

    print(f"Found {len(runs)} run(s) with {args.split} scores. Analysing …\n")

    results = []
    for run in runs:
        r = analyse_run(run)
        if r is not None:
            results.append(r)
        else:
            print(f"  [skip] {run['model']} frac={run['fraction']}: degenerate split")

    if not results:
        print("No valid results to analyse.")
        return

    write_summary_csv(results, out_dir / "summary.csv")
    write_table(results, out_dir / "threshold_table.txt")
    plot_pr_curves(results, out_dir / "pr_curves.png")

    print(f"\nOutputs written to {out_dir}/")


if __name__ == "__main__":
    main()
