"""
scripts/run_eval_all.py

Scan the checkpoint directory for completed training runs, run evaluate.py
and calibrate.py for each one, and print a summary table.

Skips any run that already has results/calibration JSON files so it is safe
to re-run after partial completion.

Usage (on Hyak login node — wraps each call in apptainer exec automatically):
    python scripts/run_eval_all.py

Usage with explicit paths (useful on any cluster or locally):
    python scripts/run_eval_all.py \\
        --project-dir /path/to/equivariant-sar \\
        --sif         /path/to/pytorch_24.12-py3.sif \\
        --venv        /path/to/venv

Usage (inside the Apptainer container, or locally without apptainer):
    python scripts/run_eval_all.py --no-apptainer

Dry run (print what would run, execute nothing):
    python scripts/run_eval_all.py --dry-run

Note: if the script detects it is already running inside an Apptainer/Singularity
container (/.singularity.d or /singularity present), it automatically skips the
apptainer wrapper and calls evaluate.py / calibrate.py directly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def _inside_container() -> bool:
    """Return True when running inside an Apptainer / Singularity container."""
    return os.path.exists("/.singularity.d") or os.path.exists("/singularity")


# ---------------------------------------------------------------------------
# Default Hyak paths — overridden at runtime via --project-dir / --sif / --venv
# ---------------------------------------------------------------------------
_DEFAULT_PROJECT = Path("/gscratch/scrubbed/sanmarco/equivariant-sar")
_DEFAULT_SIF     = Path("/gscratch/scrubbed/sanmarco/pytorch_24.12-py3.sif")
_DEFAULT_VENV    = Path("/gscratch/scrubbed/sanmarco/venv")

VALID_MODELS    = {"c8", "so2", "d4", "o2", "d4_bitemporal", "cnn_bitemporal", "cnn", "aug", "resnet"}
FRACTION_MAP    = {"0p1": 0.1, "0p25": 0.25, "0p5": 0.5, "1p0": 1.0}


# ---------------------------------------------------------------------------
# Parse a checkpoint directory name → (model, fraction)
# e.g. "c8_frac1p0" → ("c8", 1.0)
# ---------------------------------------------------------------------------

def parse_run_name(name: str) -> tuple[str, float] | None:
    m = re.fullmatch(r"([a-z0-9_]+)_frac(\w+)", name)
    if m is None:
        return None
    model    = m.group(1)
    frac_str = m.group(2)
    if model not in VALID_MODELS:
        return None
    if frac_str not in FRACTION_MAP:
        return None
    return model, FRACTION_MAP[frac_str]


# ---------------------------------------------------------------------------
# Discover completed checkpoints
# ---------------------------------------------------------------------------

def find_completed_runs(ckpt_dir: Path) -> list[tuple[str, float, Path]]:
    """Return list of (model, fraction, best_pt_path) for all completed runs."""
    runs = []
    if not ckpt_dir.exists():
        return runs
    for d in sorted(ckpt_dir.iterdir()):
        if not d.is_dir():
            continue
        best_pt = d / "best.pt"
        if not best_pt.exists():
            continue
        parsed = parse_run_name(d.name)
        if parsed is None:
            print(f"  [skip] unrecognised directory name: {d.name}")
            continue
        model, fraction = parsed
        runs.append((model, fraction, best_pt))
    return runs


# ---------------------------------------------------------------------------
# Check whether a run already has results
# ---------------------------------------------------------------------------

def is_evaluated(model: str, fraction: float, results_dir: Path) -> bool:
    frac_str = str(fraction).replace(".", "p")
    run_dir  = results_dir / f"{model}_frac{frac_str}"
    return (run_dir / "metrics.json").exists()


def is_calibrated(model: str, fraction: float, results_dir: Path) -> bool:
    frac_str = str(fraction).replace(".", "p")
    run_dir  = results_dir / f"{model}_frac{frac_str}"
    return (run_dir / "calibration.json").exists()


# ---------------------------------------------------------------------------
# Build the shell command — optionally wrapped in apptainer exec
# ---------------------------------------------------------------------------

def make_cmd(python_args: str, use_apptainer: bool, project: Path, sif: Path, venv: Path) -> str:
    if use_apptainer:
        inner = (
            f"source {venv}/bin/activate && "
            f"cd {project} && "
            f"python {python_args}"
        )
        return (
            f"apptainer exec --nv --bind /gscratch "
            f"{sif} "
            f"/bin/bash -c '{inner}'"
        )
    # Already inside the container (or local run): use the same interpreter
    # that is running this script so the venv / PATH doesn't matter.
    return f"{sys.executable} {python_args}"


def run(cmd: str, dry_run: bool) -> int:
    print(f"\n  $ {cmd}")
    if dry_run:
        print("  [dry-run — not executed]")
        return 0
    result = subprocess.run(cmd, shell=True, executable="/bin/bash")
    return result.returncode


# ---------------------------------------------------------------------------
# Read metrics from completed result files
# ---------------------------------------------------------------------------

def read_metrics(model: str, fraction: float, results_dir: Path) -> dict:
    frac_str = str(fraction).replace(".", "p")
    run_dir  = results_dir / f"{model}_frac{frac_str}"

    out = {"model": model, "fraction": fraction}

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            d = json.load(f)
        for split in ("test_ood", "val"):
            if split in d.get("splits", {}):
                s = d["splits"][split]
                out["auc"]  = s.get("auc_roc",      float("nan"))
                out["f1"]   = s.get("at_optimal", {}).get("f1",  float("nan"))
                out["f2"]   = s.get("at_optimal", {}).get("f2",  float("nan"))
                out["split_used"] = split
                break

    cal_path = run_dir / "calibration.json"
    if cal_path.exists():
        with open(cal_path) as f:
            d = json.load(f)
        out["T"] = d.get("temperature", float("nan"))
        for split in ("test_ood", "val"):
            if split in d.get("splits", {}):
                out["ece_before"] = d["splits"][split].get("ece_before", float("nan"))
                out["ece_after"]  = d["splits"][split].get("ece_after",  float("nan"))
                break

    return out


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict]) -> None:
    if not rows:
        print("\nNo completed runs to summarise.")
        return

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    header = (
        f"  {'Model':<8}  {'Frac':>5}  {'Split':<9}"
        f"  {'AUC':>6}  {'F1@opt':>7}  {'F2@opt':>7}"
        f"  {'T':>6}  {'ECE↑':>6}  {'ECE↓':>6}"
    )
    print(header)
    print("  " + "-" * 74)
    for r in sorted(rows, key=lambda x: (x["model"], x["fraction"])):
        def _f(v, fmt=".4f"):
            return f"{v:{fmt}}" if isinstance(v, float) and v == v else "   —  "
        print(
            f"  {r['model']:<8}  {r['fraction']:>5.2f}  "
            f"{r.get('split_used', '—'):<9}"
            f"  {_f(r.get('auc',float('nan'))):>6}"
            f"  {_f(r.get('f1', float('nan'))):>7}"
            f"  {_f(r.get('f2', float('nan'))):>7}"
            f"  {_f(r.get('T',  float('nan'))):>6}"
            f"  {_f(r.get('ece_before', float('nan'))):>6}"
            f"  {_f(r.get('ece_after',  float('nan'))):>6}"
        )
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate all completed training runs.")
    p.add_argument("--project-dir", default=str(_DEFAULT_PROJECT),
                   help="Root project directory (contains checkpoints/, results/, data/).")
    p.add_argument("--sif", default=str(_DEFAULT_SIF),
                   help="Path to the Apptainer/Singularity .sif container image.")
    p.add_argument("--venv", default=str(_DEFAULT_VENV),
                   help="Path to the Python venv to activate inside the container.")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Root checkpoint directory (default: <project-dir>/checkpoints).")
    p.add_argument("--results-dir", default=None,
                   help="Root results directory (default: <project-dir>/results).")
    p.add_argument("--no-apptainer", action="store_true",
                   help="Run python directly (no apptainer wrapper — for local use).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing them.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--batch-size",  type=int, default=256)
    return p.parse_args()


def main() -> None:
    args        = parse_args()
    project_dir = Path(args.project_dir)
    sif         = Path(args.sif)
    venv        = Path(args.venv)
    data_dir    = project_dir / "data"
    ckpt_dir    = Path(args.checkpoint_dir) if args.checkpoint_dir else project_dir / "checkpoints"
    results_dir = Path(args.results_dir)    if args.results_dir    else project_dir / "results"

    if args.no_apptainer:
        apptainer = False
    elif _inside_container():
        print("Detected Apptainer/Singularity container — skipping apptainer wrapper.")
        apptainer = False
    else:
        apptainer = True

    runs = find_completed_runs(ckpt_dir)

    if not runs:
        print(f"No completed checkpoints found under {ckpt_dir}")
        sys.exit(0)

    print(f"Found {len(runs)} completed checkpoint(s):")
    for model, fraction, path in runs:
        print(f"  {model}  frac={fraction}  ({path})")

    errors = []

    for model, fraction, _ in runs:
        frac_str = str(fraction).replace(".", "p")
        tag      = f"{model}_frac{frac_str}"

        # ---- evaluate.py ----
        if is_evaluated(model, fraction, results_dir):
            print(f"\n[skip evaluate] {tag} — metrics.json already exists")
        else:
            print(f"\n[evaluate] {tag}")
            bt_flag = (
                f" --bitemporal-stats-path {data_dir}/splits/norm_stats_bitemporal.json"
                if model in ("d4_bitemporal", "cnn_bitemporal") else ""
            )
            eval_args = (
                f"evaluate.py "
                f"--model {model} "
                f"--data-fraction {fraction} "
                f"--val-csv      {data_dir}/splits/val.csv "
                f"--test-csv     {data_dir}/splits/test_ood.csv "
                f"--stats-path   {data_dir}/splits/norm_stats.json"
                f"{bt_flag} "
                f"--checkpoint-dir {ckpt_dir} "
                f"--results-dir    {results_dir} "
                f"--batch-size   {args.batch_size} "
                f"--num-workers  {args.num_workers}"
            )
            rc = run(make_cmd(eval_args, apptainer, project_dir, sif, venv), args.dry_run)
            if rc != 0:
                print(f"  ERROR: evaluate.py failed for {tag} (exit {rc})")
                errors.append(f"evaluate {tag}")
                continue   # don't attempt calibrate if evaluate failed

        # ---- calibrate.py ----
        if is_calibrated(model, fraction, results_dir):
            print(f"[skip calibrate] {tag} — calibration.json already exists")
        else:
            print(f"[calibrate] {tag}")
            cal_args = (
                f"calibrate.py "
                f"--model {model} "
                f"--data-fraction {fraction} "
                f"--results-dir {results_dir}"
            )
            rc = run(make_cmd(cal_args, apptainer, project_dir, sif, venv), args.dry_run)
            if rc != 0:
                print(f"  ERROR: calibrate.py failed for {tag} (exit {rc})")
                errors.append(f"calibrate {tag}")

    # ---- summary ----
    summary_rows = [
        read_metrics(model, fraction, results_dir)
        for model, fraction, _ in runs
    ]
    print_summary(summary_rows)

    if errors:
        print(f"\n{len(errors)} run(s) failed: {errors}")
        sys.exit(1)


if __name__ == "__main__":
    main()
