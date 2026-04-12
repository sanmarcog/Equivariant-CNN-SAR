"""
evaluate.py

Evaluation script for all 6 avalanche debris classification models.

Loads the best checkpoint for a given model + data-fraction combination and
evaluates on the validation set and the OOD test set (Tromsø).

Metrics reported (at threshold 0.5 AND at the Youden-optimal threshold):
    AUC-ROC, average precision (AUPRC)
    F1, precision, recall, specificity (TNR), NPV
    Balanced accuracy, MCC, Brier score

Per-event / per-region breakdown:
    AUC-ROC computed separately for each event / region in the split.
    (Events with < 20 samples are skipped — AUC undefined.)

Figures saved to results/<run_name>/figures/:
    roc_<split>.png              — ROC curve
    pr_<split>.png               — Precision-Recall curve
    scores_<split>.png           — Score histogram (neg vs pos)
    scores_cdf_<split>.png       — CDF of predicted probabilities
    confusion_<split>.png        — Confusion matrices at 0.5 and optimal threshold
    auc_by_event_<split>.png     — Bar chart: AUC-ROC per event

Output files:
    results/<run_name>/metrics.json        — all metrics per split
    results/<run_name>/scores_<split>.npz  — raw (logits, labels) for calibrate.py

Aggregate summary:
    python evaluate.py --summary
    Prints a Markdown table across all completed runs.

Usage:
    python evaluate.py --model c8 --data-fraction 1.0
    python evaluate.py --model resnet --data-fraction 0.5
    python evaluate.py --summary
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
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False

from data_pipeline.dataset import AvalancheDataset
from models.cnn_baseline import CNNBaseline, count_parameters
from models.cnn_augmented import AugmentedCNN
from models.resnet_baseline import ResNetBaseline
from models.equivariant_cnn import C8EquivariantCNN, SO2EquivariantCNN, D4EquivariantCNN, O2EquivariantCNN, D4BiTemporalCNN


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_NAMES = ("c8", "so2", "d4", "o2", "d4_bitemporal", "cnn", "aug", "resnet")
MIN_SAMPLES_PER_EVENT = 20   # minimum to compute per-event AUC


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(name: str, in_channels: int = 5) -> nn.Module:
    if name == "c8":
        return C8EquivariantCNN(in_channels=in_channels)
    if name == "so2":
        return SO2EquivariantCNN(in_channels=in_channels)
    if name == "d4":
        return D4EquivariantCNN(in_channels=in_channels)
    if name == "o2":
        return O2EquivariantCNN(in_channels=in_channels)
    if name == "d4_bitemporal":
        return D4BiTemporalCNN(in_channels=in_channels)
    if name == "cnn":
        return CNNBaseline(in_channels=in_channels)
    if name == "aug":
        return AugmentedCNN(in_channels=in_channels)
    if name == "resnet":
        return ResNetBaseline(in_channels=in_channels, pretrained=False)
    raise ValueError(f"Unknown model: {name}")


def forward_logit(model: nn.Module, x) -> torch.Tensor:
    """Return scalar logit [B, 1]. x may be a tensor or (post, pre) tuple."""
    base = model.module if isinstance(model, nn.DataParallel) else model
    if getattr(base, "bitemporal", False):
        x_post, x_pre = x
        logit, _ = model(x_post, x_pre, return_orientation=False)
        return logit
    if hasattr(base, "orientation_conv"):
        logit, _ = model(x, return_orientation=False)
        return logit
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits [N], labels [N]) as float64 numpy arrays."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        x, y = batch
        if isinstance(x, (tuple, list)):
            x = tuple(t.to(device, non_blocking=True) for t in x)
        else:
            x = x.to(device, non_blocking=True)
        logit = forward_logit(model, x).squeeze(1).cpu().numpy()
        all_logits.append(logit)
        all_labels.append(y.numpy())
    return (
        np.concatenate(all_logits).astype(np.float64),
        np.concatenate(all_labels).astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics_at_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0,
    )
    _, _, f2, _ = precision_recall_fscore_support(
        labels, preds, average="binary", beta=2.0, zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    bal_acc     = balanced_accuracy_score(labels, preds)
    mcc         = float(matthews_corrcoef(labels, preds))
    return {
        "threshold":   threshold,
        "f1":          float(f1),
        "f2":          float(f2),   # recall-weighted; missing an avalanche costs more than a false alarm
        "precision":   float(prec),
        "recall":      float(rec),
        "specificity": float(specificity),
        "npv":         float(npv),
        "balanced_accuracy": float(bal_acc),
        "mcc":         mcc,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    probs = 1.0 / (1.0 + np.exp(-logits))

    auc_roc  = float(roc_auc_score(labels, probs))
    avg_prec = float(average_precision_score(labels, probs))
    brier    = float(np.mean((probs - labels) ** 2))

    # Optimal threshold: maximise Youden's J = TPR + TNR - 1
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores  = tpr + (1 - fpr) - 1
    best_idx  = int(np.argmax(j_scores))
    opt_thr   = float(thresholds[best_idx])

    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)

    return {
        "auc_roc":       auc_roc,
        "avg_precision": avg_prec,
        "brier_score":   brier,
        "n_samples":     int(len(labels)),
        "n_positive":    n_pos,
        "n_negative":    n_neg,
        "prevalence":    n_pos / len(labels),
        "at_0.5":        _metrics_at_threshold(probs, labels, 0.5),
        "at_optimal":    _metrics_at_threshold(probs, labels, opt_thr),
    }


# ---------------------------------------------------------------------------
# Per-event / per-region breakdown
# ---------------------------------------------------------------------------

def per_group_auc(
    logits: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    min_samples: int = MIN_SAMPLES_PER_EVENT,
) -> dict[str, float | str]:
    """
    Compute AUC-ROC per unique value in `groups` (e.g. event or region).
    Groups with fewer than `min_samples` or only one class are skipped.
    """
    result: dict[str, float | str] = {}
    probs = 1.0 / (1.0 + np.exp(-logits))
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() < min_samples:
            result[g] = "skip (too few samples)"
            continue
        if len(np.unique(labels[mask])) < 2:
            result[g] = "skip (single class)"
            continue
        result[g] = float(roc_auc_score(labels[mask], probs[mask]))
    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_figures(
    out_dir: Path,
    split_name: str,
    logits: np.ndarray,
    labels: np.ndarray,
    metrics: dict,
    event_auc: dict[str, float | str] | None = None,
    region_auc: dict[str, float | str] | None = None,
) -> None:
    if not _MATPLOTLIB:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    probs    = 1.0 / (1.0 + np.exp(-logits))
    opt_thr  = metrics["at_optimal"]["threshold"]
    prev     = metrics["prevalence"]

    # ---- ROC curve ----
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=1.5, label=f"AUC = {metrics['auc_roc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {split_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"roc_{split_name}.png", dpi=150)
    plt.close(fig)

    # ---- PR curve ----
    prec_c, rec_c, _ = precision_recall_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec_c, prec_c, lw=1.5, label=f"AP = {metrics['avg_precision']:.3f}")
    ax.axhline(prev, color="k", linestyle="--", lw=0.8,
               label=f"Baseline (prev={prev:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {split_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"pr_{split_name}.png", dpi=150)
    plt.close(fig)

    # ---- Score histogram ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(probs[labels == 0], bins=50, alpha=0.6, label="Negative (clean)",
            density=True, color="steelblue")
    ax.hist(probs[labels == 1], bins=50, alpha=0.6, label="Positive (debris)",
            density=True, color="tomato")
    ax.axvline(0.5,     color="k",      linestyle="--", lw=0.8, label="Threshold 0.5")
    ax.axvline(opt_thr, color="orange", linestyle="--", lw=0.8,
               label=f"Youden optimal ({opt_thr:.2f})")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Score distribution — {split_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"scores_{split_name}.png", dpi=150)
    plt.close(fig)

    # ---- Score CDF ----
    sorted_probs = np.sort(probs)
    cdf = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sorted_probs[labels.astype(bool)[np.argsort(probs)] == 0],
            cdf[labels.astype(bool)[np.argsort(probs)] == 0],
            color="steelblue", label="Negative")
    ax.plot(sorted_probs[labels.astype(bool)[np.argsort(probs)] == 1],
            cdf[labels.astype(bool)[np.argsort(probs)] == 1],
            color="tomato", label="Positive")
    ax.axvline(0.5,     color="k",      linestyle="--", lw=0.8)
    ax.axvline(opt_thr, color="orange", linestyle="--", lw=0.8)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(f"Score CDF — {split_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"scores_cdf_{split_name}.png", dpi=150)
    plt.close(fig)

    # ---- Confusion matrices (side by side: threshold 0.5 and optimal) ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, thr_key, thr_val in [
        (axes[0], "at_0.5",    0.5),
        (axes[1], "at_optimal", opt_thr),
    ]:
        m = metrics[thr_key]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        # Normalize rows to show rates
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Neg", "Pred Pos"])
        ax.set_yticklabels(["True Neg", "True Pos"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i,
                        f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.6 else "black",
                        fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(
            f"thr={thr_val:.2f}  F1={m['f1']:.3f}  "
            f"MCC={m['mcc']:.3f}\n"
            f"Prec={m['precision']:.3f}  Rec={m['recall']:.3f}  "
            f"Spec={m['specificity']:.3f}"
        )
    fig.suptitle(f"Confusion matrices — {split_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{split_name}.png", dpi=150)
    plt.close(fig)

    # ---- Per-event AUC bar chart ----
    for group_name, group_auc in [("event", event_auc), ("region", region_auc)]:
        if group_auc is None:
            continue
        numeric = {k: v for k, v in group_auc.items() if isinstance(v, float)}
        if not numeric:
            continue
        keys = list(numeric.keys())
        vals = [numeric[k] for k in keys]
        fig, ax = plt.subplots(figsize=(max(5, len(keys) * 0.9 + 1), 4))
        colors = ["tomato" if v < 0.7 else "steelblue" for v in vals]
        ax.bar(keys, vals, color=colors, edgecolor="white")
        ax.axhline(0.5, color="k", linestyle="--", lw=0.8, label="Random")
        ax.set_ylim(0, 1)
        ax.set_ylabel("AUC-ROC")
        ax.set_title(f"AUC-ROC per {group_name} — {split_name}")
        ax.tick_params(axis="x", rotation=20)
        for x, y in enumerate(vals):
            ax.text(x, y + 0.02, f"{y:.3f}", ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"auc_by_{group_name}_{split_name}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Single-run evaluation
# ---------------------------------------------------------------------------

def evaluate_run(args: argparse.Namespace) -> None:
    frac_str = str(args.data_fraction).replace(".", "p")
    run_name = f"{args.model}_frac{frac_str}"

    ckpt_path = Path(args.checkpoint_dir) / run_name / "best.pt"
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    out_dir = Path(args.results_dir) / run_name
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Building model: %s", args.model)
    model = build_model(args.model)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    log.info("Loaded checkpoint: epoch %d  best_val_auc=%.4f",
             ckpt["epoch"] + 1, ckpt["best_auc"])
    log.info("Parameters: %s", f"{count_parameters(model):,}")

    splits = {
        "val":      args.val_csv,
        "test_ood": args.test_csv,
    }

    all_metrics: dict[str, dict] = {}

    for split_name, csv_path in splits.items():
        if not Path(csv_path).exists():
            log.warning("CSV not found, skipping %s: %s", split_name, csv_path)
            continue

        is_bitemporal = (args.model == "d4_bitemporal")
        stats = args.bitemporal_stats_path if is_bitemporal else args.stats_path
        ds = AvalancheDataset(split_csv=csv_path, stats_path=stats, bitemporal=is_bitemporal)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        log.info("Running inference on %s (%d samples) …", split_name, len(ds))
        logits, labels = run_inference(model, loader, device)

        metrics = compute_metrics(logits, labels)
        all_metrics[split_name] = metrics

        m05  = metrics["at_0.5"]
        mopt = metrics["at_optimal"]
        log.info(
            "  %s | AUC=%.4f  AP=%.4f  Brier=%.4f\n"
            "          thr=0.50 → F1=%.4f  F2=%.4f  Prec=%.4f  Rec=%.4f  "
            "Spec=%.4f  BalAcc=%.4f  MCC=%.4f\n"
            "          thr=%.3f → F1=%.4f  F2=%.4f  Prec=%.4f  Rec=%.4f  "
            "Spec=%.4f  BalAcc=%.4f  MCC=%.4f",
            split_name,
            metrics["auc_roc"], metrics["avg_precision"], metrics["brier_score"],
            m05["f1"],  m05["f2"],  m05["precision"],  m05["recall"],
            m05["specificity"],  m05["balanced_accuracy"],  m05["mcc"],
            mopt["threshold"],
            mopt["f1"], mopt["f2"], mopt["precision"], mopt["recall"],
            mopt["specificity"], mopt["balanced_accuracy"], mopt["mcc"],
        )

        # Per-event / per-region AUC
        events  = np.array([r["event"]  for r in ds.records])
        regions = np.array([r["region"] for r in ds.records])

        ev_auc  = per_group_auc(logits, labels, events)
        reg_auc = per_group_auc(logits, labels, regions)

        log.info("  Per-event AUC:  %s",
                 {k: f"{v:.3f}" if isinstance(v, float) else v
                  for k, v in ev_auc.items()})
        log.info("  Per-region AUC: %s",
                 {k: f"{v:.3f}" if isinstance(v, float) else v
                  for k, v in reg_auc.items()})

        metrics["per_event_auc"]  = ev_auc
        metrics["per_region_auc"] = reg_auc

        # Save raw logits + labels (needed by calibrate.py)
        np.savez_compressed(
            out_dir / f"scores_{split_name}.npz",
            logits=logits,
            labels=labels,
        )

        _save_figures(fig_dir, split_name, logits, labels, metrics,
                      event_auc=ev_auc, region_auc=reg_auc)

    result = {
        "model":         args.model,
        "data_fraction": args.data_fraction,
        "run_name":      run_name,
        "checkpoint":    str(ckpt_path),
        "splits":        all_metrics,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Results saved to %s", metrics_path)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results_dir: str) -> None:
    root  = Path(results_dir)
    paths = sorted(root.glob("*/metrics.json"))

    if not paths:
        print(f"No metrics.json files found under {root}")
        sys.exit(0)

    rows: list[dict] = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        for split, m in d.get("splits", {}).items():
            m05  = m.get("at_0.5", {})
            mopt = m.get("at_optimal", {})
            rows.append({
                "model":    d["model"],
                "fraction": d["data_fraction"],
                "split":    split,
                "auc_roc":  m["auc_roc"],
                "ap":       m["avg_precision"],
                "brier":    m["brier_score"],
                "f1_05":    m05.get("f1",  float("nan")),
                "f2_05":    m05.get("f2",  float("nan")),
                "f1_opt":   mopt.get("f1", float("nan")),
                "f2_opt":   mopt.get("f2", float("nan")),
                "bal_acc":  mopt.get("balanced_accuracy", float("nan")),
                "mcc":      mopt.get("mcc", float("nan")),
            })

    header = (
        f"| {'Model':<8} | {'Frac':>5} | {'Split':<9} "
        f"| {'AUC-ROC':>7} | {'AUPRC':>6} | {'Brier':>6} "
        f"| {'F1@0.5':>7} | {'F2@0.5':>7} "
        f"| {'F1@opt':>7} | {'F2@opt':>7} "
        f"| {'BalAcc':>7} | {'MCC':>6} |"
    )
    sep = "|" + "|".join(
        ["-" * (w + 2) for w in [8, 5, 9, 7, 6, 6, 7, 7, 7, 7, 7, 6]]
    ) + "|"
    print(header)
    print(sep)
    for r in rows:
        print(
            f"| {r['model']:<8} | {r['fraction']:>5.2f} | {r['split']:<9} "
            f"| {r['auc_roc']:>7.4f} | {r['ap']:>6.4f} | {r['brier']:>6.4f} "
            f"| {r['f1_05']:>7.4f} | {r['f2_05']:>7.4f} "
            f"| {r['f1_opt']:>7.4f} | {r['f2_opt']:>7.4f} "
            f"| {r['bal_acc']:>7.4f} | {r['mcc']:>6.4f} |"
        )


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained models.")
    p.add_argument("--model", choices=MODEL_NAMES,
                   help="Which model to evaluate (required unless --summary).")
    p.add_argument("--data-fraction", type=float, default=1.0,
                   choices=[0.1, 0.25, 0.5, 1.0])
    p.add_argument("--val-csv",    default="data/splits/val.csv")
    p.add_argument("--test-csv",   default="data/splits/test_ood.csv")
    p.add_argument("--stats-path", default="data/splits/norm_stats.json")
    p.add_argument("--bitemporal-stats-path", default="data/splits/norm_stats_bitemporal.json",
                   help="7-channel norm stats for d4_bitemporal.")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--results-dir",    default="results")
    p.add_argument("--batch-size",  type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--summary", action="store_true",
                   help="Print aggregate Markdown table of all completed runs.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.summary:
        print_summary(args.results_dir)
        sys.exit(0)
    if args.model is None:
        print("Error: --model is required unless --summary is set.")
        sys.exit(1)
    try:
        evaluate_run(args)
    except Exception:
        log.exception("Evaluation failed.")
        sys.exit(1)
