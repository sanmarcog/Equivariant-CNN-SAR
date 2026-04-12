"""
train.py

Training loop for all 6 avalanche debris classification models.

Models:
    c8    — C8EquivariantCNN
    so2   — SO2EquivariantCNN
    d4    — D4EquivariantCNN
    cnn   — CNNBaseline
    aug   — AugmentedCNN
    resnet — ResNetBaseline

Training protocol:
    - Loss: BCEWithLogitsLoss with pos_weight (wp=3.0 by default)
    - Sampler: WeightedRandomSampler for class balance
    - Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
    - Scheduler: CosineAnnealingLR (T_max=max_epochs)
    - Early stopping: patience=10 on validation AUC-ROC
    - Checkpoint: saved every epoch (for SLURM preemption recovery)
    - W&B logging: loss, AUC per epoch, parameter count, GPU memory

Data-efficiency:
    Pass --data-fraction 0.1 / 0.25 / 0.5 / 1.0 to train on a subset.
    Subsampling is stratified by label and applied before the sampler.

SLURM preemption:
    If a checkpoint exists at --checkpoint-dir/<run_name>/last.pt, training
    resumes from that epoch automatically.

Usage:
    python train.py --model c8 --data-fraction 1.0
    python train.py --model resnet --data-fraction 0.5 --epochs 50
    python train.py --resume --model c8 --data-fraction 1.0

Exit codes:
    0 — training completed (or already done)
    1 — error
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

# Optional W&B — gracefully disabled if not installed / not logged in
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from data_pipeline.dataset import AvalancheDataset, get_sample_weights
from models.cnn_baseline import CNNBaseline, count_parameters
from models.cnn_augmented import AugmentedCNN
from models.resnet_baseline import ResNetBaseline
from models.equivariant_cnn import C8EquivariantCNN, SO2EquivariantCNN, D4EquivariantCNN


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

MODEL_NAMES = ("c8", "so2", "d4", "cnn", "aug", "resnet")

def build_model(name: str, in_channels: int = 5) -> nn.Module:
    if name == "c8":
        return C8EquivariantCNN(in_channels=in_channels)
    if name == "so2":
        return SO2EquivariantCNN(in_channels=in_channels)
    if name == "d4":
        return D4EquivariantCNN(in_channels=in_channels)
    if name == "cnn":
        return CNNBaseline(in_channels=in_channels)
    if name == "aug":
        return AugmentedCNN(in_channels=in_channels)
    if name == "resnet":
        return ResNetBaseline(in_channels=in_channels, pretrained=True)
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Forward pass — handles equivariant models (2-tuple output) and plain models
# ---------------------------------------------------------------------------

def forward_logit(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return scalar logit [B, 1] regardless of model type."""
    base = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(base, "orientation_conv"):
        # Equivariant model — explicitly skip orientation head (not in loss,
        # running it during training wastes compute every batch)
        logit, _ = model(x, return_orientation=False)
        return logit
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    return out


# ---------------------------------------------------------------------------
# Data-efficiency subsetting (stratified)
# ---------------------------------------------------------------------------

def stratified_subset(dataset: AvalancheDataset, fraction: float) -> AvalancheDataset | Subset:
    """Return a stratified subset keeping `fraction` of each class."""
    if fraction >= 1.0:
        return dataset

    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    rng = np.random.default_rng(42)
    n_pos = max(1, int(len(pos_idx) * fraction))
    n_neg = max(1, int(len(neg_idx) * fraction))

    chosen_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    indices = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(indices)

    return Subset(dataset, indices.tolist())


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_auc: float,
    patience_counter: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_auc": best_auc,
            "patience_counter": patience_counter,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["best_auc"], ckpt["patience_counter"]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_logits, all_labels = [], []

    non_blocking = device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=non_blocking)
        logit = forward_logit(model, x).squeeze(1).cpu()
        all_logits.append(logit)
        all_labels.append(y.float())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    auc   = float(roc_auc_score(labels, probs))

    # BCE loss (no pos_weight for evaluation)
    loss_fn = nn.BCEWithLogitsLoss()
    val_loss = loss_fn(
        torch.from_numpy(logits),
        torch.from_numpy(labels),
    ).item()

    return {"auc": auc, "loss": val_loss}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        log.info("Using %d GPU(s): %s", n_gpus, [torch.cuda.get_device_name(i) for i in range(n_gpus)])
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        n_gpus = 0
        log.info("Using Apple MPS backend.")
    else:
        device = torch.device("cpu")
        n_gpus = 0
        log.warning("No GPU found, training on CPU (slow).")

    # --- Run name for checkpoints and W&B ---
    frac_str = str(args.data_fraction).replace(".", "p")
    run_name = f"{args.model}_frac{frac_str}"

    ckpt_dir  = Path(args.checkpoint_dir) / run_name
    ckpt_last = ckpt_dir / "last.pt"
    ckpt_best = ckpt_dir / "best.pt"

    # --- Datasets ---
    log.info("Loading datasets …")
    train_full = AvalancheDataset(
        split_csv=args.train_csv,
        compute_stats=(not Path(args.stats_path).exists()),
        stats_path=args.stats_path,
    )
    val_ds = AvalancheDataset(
        split_csv=args.val_csv,
        stats_path=args.stats_path,
    )

    train_ds = stratified_subset(train_full, args.data_fraction)
    log.info("Train patches (after subsetting): %d  |  Val patches: %d",
             len(train_ds), len(val_ds))

    # --- Sampler (WeightedRandomSampler for class balance) ---
    # get_sample_weights works on the full dataset; re-index if Subset
    if isinstance(train_ds, Subset):
        full_weights = get_sample_weights(train_full)
        sample_weights = [full_weights[i] for i in train_ds.indices]
    else:
        sample_weights = get_sample_weights(train_ds)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    # --- Model ---
    log.info("Building model: %s", args.model)
    model = build_model(args.model)
    n_params = count_parameters(model)
    log.info("Parameters: %s", f"{n_params:,}")

    if n_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # --- Loss (pos_weight on positive class for imbalanced data) ---
    pos_weight = torch.tensor([args.pos_weight], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-2,
    )

    # --- Resume from checkpoint ---
    start_epoch    = 0
    best_auc       = 0.0
    patience_count = 0

    if ckpt_last.exists():
        log.info("Resuming from %s", ckpt_last)
        start_epoch, best_auc, patience_count = load_checkpoint(
            ckpt_last, model if n_gpus <= 1 else model.module,
            optimizer, scheduler,
        )
        start_epoch += 1  # next epoch
        log.info("Resumed at epoch %d  best_auc=%.4f", start_epoch, best_auc)

    if start_epoch >= args.epochs:
        log.info("Training already completed (%d epochs). Exiting.", args.epochs)
        return

    # --- W&B ---
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model":         args.model,
                "data_fraction": args.data_fraction,
                "batch_size":    args.batch_size,
                "epochs":        args.epochs,
                "lr":            args.lr,
                "weight_decay":  args.weight_decay,
                "pos_weight":    args.pos_weight,
                "early_stop_patience": args.patience,
                "n_params":      n_params,
                "n_train":       len(train_ds),
                "n_val":         len(val_ds),
                "n_gpus":        n_gpus,
            },
            resume="allow",
            id=run_name,  # deterministic run ID for SLURM resume
        )

    # --- Training ---
    log.info("Starting training: %s  epochs=%d  start=%d",
             run_name, args.epochs, start_epoch)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches  = 0

        non_blocking = device.type == "cuda"
        for x, y in train_loader:
            x = x.to(device, non_blocking=non_blocking)
            y = y.float().to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            logit = forward_logit(model, x).squeeze(1)
            loss  = loss_fn(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        train_loss = epoch_loss / n_batches
        val_metrics = evaluate(model if n_gpus <= 1 else model.module, val_loader, device)
        val_auc  = val_metrics["auc"]
        val_loss = val_metrics["loss"]
        elapsed  = time.time() - t0

        # GPU memory
        gpu_mb = (torch.cuda.memory_reserved(0) / 1e6) if device.type == "cuda" else 0.0

        log.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_auc=%.4f  "
            "lr=%.2e  %.1fs  GPU=%.0fMB",
            epoch + 1, args.epochs, train_loss, val_loss, val_auc,
            scheduler.get_last_lr()[0], elapsed, gpu_mb,
        )

        if use_wandb:
            wandb.log({
                "epoch":       epoch + 1,
                "train_loss":  train_loss,
                "val_loss":    val_loss,
                "val_auc":     val_auc,
                "lr":          scheduler.get_last_lr()[0],
                "gpu_mem_mb":  gpu_mb,
            })

        # --- Checkpoint every epoch (SLURM preemption safety) ---
        _model_state = model.module if n_gpus > 1 else model
        save_checkpoint(
            ckpt_last, epoch, _model_state,
            optimizer, scheduler, best_auc, patience_count,
        )

        # --- Early stopping ---
        if val_auc > best_auc:
            best_auc       = val_auc
            patience_count = 0
            save_checkpoint(
                ckpt_best, epoch, _model_state,
                optimizer, scheduler, best_auc, patience_count,
            )
            log.info("  ↑ New best val_auc=%.4f  checkpoint saved.", best_auc)
        else:
            patience_count += 1
            log.info("  No improvement (%d/%d).", patience_count, args.patience)
            if patience_count >= args.patience:
                log.info("Early stopping triggered.")
                break

        # Log best_val_auc after early-stopping block so it reflects the
        # updated best_auc for this epoch (not the value before the update).
        if use_wandb:
            wandb.log({"best_val_auc": best_auc, "epoch": epoch + 1})

    log.info("Training finished. Best val_auc=%.4f", best_auc)

    # Save final metrics JSON alongside checkpoints
    metrics_path = ckpt_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"best_val_auc": best_auc, "model": args.model,
                   "data_fraction": args.data_fraction}, f, indent=2)
    log.info("Metrics saved to %s", metrics_path)

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train avalanche debris classifier.")
    p.add_argument("--model", required=True, choices=MODEL_NAMES,
                   help="Which model to train.")
    p.add_argument("--data-fraction", type=float, default=1.0,
                   choices=[0.1, 0.25, 0.5, 1.0],
                   help="Fraction of training data to use (data-efficiency experiment).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pos-weight", type=float, default=3.0,
                   help="Positive class weight in BCEWithLogitsLoss (wp in the paper).")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (epochs without val AUC improvement).")
    p.add_argument("--train-csv",  default="data/splits/train.csv")
    p.add_argument("--val-csv",    default="data/splits/val.csv")
    p.add_argument("--stats-path", default="data/splits/norm_stats.json")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable W&B logging (useful for quick local runs).")
    p.add_argument("--wandb-project", default="equivariant-sar",
                   help="W&B project name.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    try:
        train(args)
    except Exception:
        log.exception("Training failed.")
        sys.exit(1)
