"""
build_manifest.py

Walk the patch directories produced by AvalCD's patchify.py and write a
single patches_manifest.csv that records, for every patch:

    patch_dir   : relative path to the patch folder (from repo root)
    label       : 1 = avalanche debris present, 0 = clean snowpack
    region      : Livigno | Nuuk | Pish | Tromso
    event       : full event name, e.g. Livigno_20240403
    patch_id    : integer id assigned by patchify.py

The manifest is the single source of truth consumed by split.py and
the PyTorch Dataset class. It contains no pixel data — just bookkeeping.

Usage:
    python data_pipeline/build_manifest.py \\
        --patches-dir data/raw/patches/64 \\
        --output      data/patches_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# Region is everything before the date suffix (8 digits).
# e.g. "Livigno_20240403" → "Livigno", "Nuuk_20160413" → "Nuuk"
def _region_from_event(event: str) -> str:
    parts = event.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 8:
        return parts[0]
    return event  # fallback: return as-is


def _read_label(mask_path: Path) -> int:
    """Return 1 if the mask contains any avalanche pixel, else 0."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return int(np.max(mask) >= 1)


def build_manifest(patches_dir: Path, output_path: Path, repo_root: Path) -> None:
    """Walk *patches_dir* and write a manifest CSV to *output_path*."""
    event_dirs = sorted(d for d in patches_dir.iterdir() if d.is_dir())
    if not event_dirs:
        log.error("No event directories found in %s", patches_dir)
        sys.exit(1)

    log.info("Found %d event(s): %s", len(event_dirs), [d.name for d in event_dirs])

    fieldnames = ["patch_dir", "label", "region", "event", "patch_id"]
    records: list[dict] = []

    for event_dir in event_dirs:
        event = event_dir.name
        region = _region_from_event(event)
        patch_subdirs = sorted(event_dir.iterdir(), key=lambda p: int(p.name))

        n_pos = n_neg = 0
        for patch_subdir in tqdm(patch_subdirs, desc=event, unit="patch"):
            mask_path = patch_subdir / "mask.tif"
            if not mask_path.exists():
                log.warning("No mask.tif in %s — skipping.", patch_subdir)
                continue

            label = _read_label(mask_path)
            rel_path = patch_subdir.relative_to(repo_root)

            records.append(dict(
                patch_dir=str(rel_path),
                label=label,
                region=region,
                event=event,
                patch_id=int(patch_subdir.name),
            ))

            if label == 1:
                n_pos += 1
            else:
                n_neg += 1

        log.info("%s — positive: %d  negative: %d", event, n_pos, n_neg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    total_pos = sum(r["label"] == 1 for r in records)
    total_neg = sum(r["label"] == 0 for r in records)
    log.info("Manifest written: %s", output_path)
    log.info("Total — positive: %d  negative: %d  total: %d", total_pos, total_neg, len(records))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build patches_manifest.csv from AvalCD patch directories."
    )
    p.add_argument(
        "--patches-dir", required=True, type=Path,
        help="Path to the patch size directory, e.g. data/raw/patches/64",
    )
    p.add_argument(
        "--output", default=Path("data/patches_manifest.csv"), type=Path,
        help="Output CSV path (default: data/patches_manifest.csv)",
    )
    p.add_argument(
        "--repo-root", default=Path("."), type=Path,
        help="Repo root for computing relative patch paths (default: .)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    patches_dir = args.patches_dir.resolve()
    output_path = args.output.resolve()
    repo_root = args.repo_root.resolve()

    if not patches_dir.exists():
        log.error("--patches-dir not found: %s", patches_dir)
        sys.exit(1)

    build_manifest(patches_dir, output_path, repo_root)


if __name__ == "__main__":
    main()

