"""
scripts/plot_data_efficiency.py

Read completed results from the results directory, extract test-set AUC for
each model/fraction combination, and plot data-efficiency curves.

Usage:
    python scripts/plot_data_efficiency.py
    python scripts/plot_data_efficiency.py --results-dir /path/to/results
    python scripts/plot_data_efficiency.py --metric f1   # plot F1@opt instead
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("/gscratch/scrubbed/sanmarco/equivariant-sar/results")  # Hyak default; override with --results-dir

FRACTIONS   = [0.1, 0.25, 0.5, 1.0]
FRAC_LABELS = ["10%", "25%", "50%", "100%"]

# Display order and labels
MODEL_META = {
    "d4_bitemporal": {"label": "D4-BT (bi-temporal)", "color": "#E91E63", "marker": "*", "ls": "-",  "lw": 2.5},
    "d4":     {"label": "D4 equivariant",       "color": "#9C27B0", "marker": "^", "ls": "-"},
    "c8":     {"label": "C8 equivariant",       "color": "#2196F3", "marker": "o", "ls": "-"},
    "so2":    {"label": "SO(2) equivariant",    "color": "#4CAF50", "marker": "s", "ls": "-"},
    "resnet": {"label": "ResNet-18",            "color": "#607D8B", "marker": "P", "ls": "-."},
    "cnn":    {"label": "CNN baseline",         "color": "#F44336", "marker": "D", "ls": "--"},
    "aug":    {"label": "CNN + aug",            "color": "#FF9800", "marker": "v", "ls": "--"},
}

METRIC_KEYS = {
    "auc": ("auc_roc",   "AUC-ROC"),
    "f1":  ("f1",        "F1 @ optimal threshold"),
    "f2":  ("f2",        "F2 @ optimal threshold"),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def frac_to_str(f: float) -> str:
    """0.1 → '0p1', 1.0 → '1p0'"""
    return str(f).replace(".", "p")


def load_value(results_dir: Path, model: str, fraction: float, metric: str) -> float:
    """Return metric value or NaN if missing."""
    run_dir = results_dir / f"{model}_frac{frac_to_str(fraction)}"
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return float("nan")

    with open(metrics_path) as f:
        d = json.load(f)

    splits = d.get("splits", {})
    # Prefer test_ood, fall back to val
    for split in ("test_ood", "val"):
        if split not in splits:
            continue
        s = splits[split]
        top_level_key, _ = METRIC_KEYS[metric]
        if metric == "auc":
            v = s.get(top_level_key, float("nan"))
        else:
            v = s.get("at_optimal", {}).get(top_level_key, float("nan"))
        return float(v)

    return float("nan")


def load_split_used(results_dir: Path, model: str, fraction: float) -> str:
    run_dir = results_dir / f"{model}_frac{frac_to_str(fraction)}" / "metrics.json"
    if not run_dir.exists():
        return "—"
    with open(run_dir) as f:
        d = json.load(f)
    splits = d.get("splits", {})
    for split in ("test_ood", "val"):
        if split in splits:
            return split
    return "—"


def build_table(results_dir: Path, metric: str) -> dict[str, list[float]]:
    """Return {model: [v_10, v_25, v_50, v_100]} with NaN for missing."""
    return {
        model: [load_value(results_dir, model, f, metric) for f in FRACTIONS]
        for model in MODEL_META
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(table: dict[str, list[float]], metric_label: str) -> None:
    col_w = 9
    header = f"  {'Model':<20}" + "".join(f"{lbl:>{col_w}}" for lbl in FRAC_LABELS)
    print()
    print("=" * (20 + col_w * len(FRACTIONS) + 4))
    print(f"  {metric_label}")
    print("=" * (20 + col_w * len(FRACTIONS) + 4))
    print(header)
    print("  " + "-" * (18 + col_w * len(FRACTIONS)))
    for model, meta in MODEL_META.items():
        vals = table[model]
        row = f"  {meta['label']:<20}"
        for v in vals:
            if np.isnan(v):
                row += f"{'—':>{col_w}}"
            else:
                row += f"{v:>{col_w}.4f}"
        print(row)
    print("=" * (20 + col_w * len(FRACTIONS) + 4))
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_data_efficiency(
    table: dict[str, list[float]],
    metric_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.array([10, 25, 50, 100])  # percent

    any_plotted = False
    for model, meta in MODEL_META.items():
        vals = np.array(table[model], dtype=float)
        if np.all(np.isnan(vals)):
            continue  # nothing to plot yet

        # Solid segments between available points; gaps stay blank naturally
        ax.plot(
            x, vals,
            color=meta["color"],
            linestyle=meta["ls"],
            linewidth=meta.get("lw", 1.8),
            marker=meta["marker"],
            markersize=8 if meta.get("lw", 1.8) > 2 else 7,
            label=meta["label"],
            clip_on=False,
            zorder=5 if meta.get("lw", 1.8) > 2 else 3,
        )
        any_plotted = True

    if not any_plotted:
        print("No completed results found — nothing to plot.")
        return

    ax.set_xlabel("Training set fraction", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title("Data-efficiency curves — SAR avalanche detection", fontsize=13)

    ax.set_xticks(x)
    ax.set_xticklabels(FRAC_LABELS)
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    # Y-axis: start just below the minimum non-NaN value
    all_vals = [v for vals in table.values() for v in vals if not np.isnan(v)]
    if all_vals:
        ymin = max(0.0, min(all_vals) - 0.05)
        ymax = min(1.0, max(all_vals) + 0.05)
        ax.set_ylim(ymin, ymax)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle="--", alpha=0.4)
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
    p = argparse.ArgumentParser(description="Plot data-efficiency curves from results.")
    p.add_argument("--results-dir", default=str(RESULTS_DIR),
                   help="Root results directory.")
    p.add_argument("--metric", choices=list(METRIC_KEYS), default="auc",
                   help="Metric to plot (default: auc).")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: <results-dir>/data_efficiency.png).")
    return p.parse_args()


def main() -> None:
    args        = parse_args()
    results_dir = Path(args.results_dir)
    out_path    = Path(args.out) if args.out else results_dir / "figures" / "data_efficiency_auc.png"
    metric      = args.metric
    _, metric_label = METRIC_KEYS[metric]

    table = build_table(results_dir, metric)
    print_summary(table, metric_label)
    plot_data_efficiency(table, metric_label, out_path)


if __name__ == "__main__":
    main()
