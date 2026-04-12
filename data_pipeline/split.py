"""
split.py

Assign each patch in patches_manifest.csv to a split and write four CSVs:

    data/splits/train.csv
    data/splits/val.csv
    data/splits/test_id.csv   (in-distribution: held-out val event)
    data/splits/test_ood.csv  (out-of-distribution: Tromsø, never seen in training)

Split design (consistent with arXiv:2603.22658):
    - Test OOD  : Tromsø_20241220  (geographic hold-out, never touches training)
    - Val       : Livigno_20250318 (one full event held out from training regions)
    - Train     : all remaining events
                  (Livigno_20240403, Livigno_20250129,
                   Nuuk_20160413, Nuuk_20210411, Pish_20230221)

Note: splits are by event, not by patch. Random patch-level splits would
cause data leakage because overlapping sliding-window patches from the same
event are nearly identical.

Usage:
    python data_pipeline/split.py \\
        --manifest data/patches_manifest.csv \\
        --output-dir data/splits
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Split assignment — edit here if events change
# ---------------------------------------------------------------------------

TEST_OOD_EVENTS = {"Tromso_20241220"}

VAL_EVENTS = {"Livigno_20250318"}

# Train is everything not in the above two sets


def assign_split(event: str) -> str:
    if event in TEST_OOD_EVENTS:
        return "test_ood"
    if event in VAL_EVENTS:
        return "val"
    return "train"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_splits(manifest_path: Path, output_dir: Path) -> None:
    # Read manifest
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        log.error("Manifest is empty: %s", manifest_path)
        sys.exit(1)

    # Assign splits
    for row in rows:
        row["split"] = assign_split(row["event"])

    # Group by split
    splits: dict[str, list[dict]] = {
        "train": [], "val": [], "test_ood": []
    }
    for row in rows:
        splits[row["split"]].append(row)

    # Sanity check — every event must be assigned
    assigned_events = {r["event"] for r in rows}
    all_events = set(r["event"] for r in rows)
    unassigned = all_events - assigned_events
    if unassigned:
        log.error("Unassigned events: %s", unassigned)
        sys.exit(1)

    # Verify expected events are present
    for event in TEST_OOD_EVENTS | VAL_EVENTS:
        if event not in all_events:
            log.warning("Expected event %r not found in manifest.", event)

    # Write split CSVs
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    split_files = {
        "train":    output_dir / "train.csv",
        "val":      output_dir / "val.csv",
        "test_ood": output_dir / "test_ood.csv",
    }

    for split_name, split_rows in splits.items():
        out_path = split_files[split_name]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)

        n_pos = sum(1 for r in split_rows if int(r["label"]) == 1)
        n_neg = sum(1 for r in split_rows if int(r["label"]) == 0)
        events = sorted({r["event"] for r in split_rows})
        log.info(
            "%-10s  %5d patches  (pos: %4d  neg: %5d)  events: %s",
            split_name, len(split_rows), n_pos, n_neg, events,
        )

    log.info("Splits written to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split patches_manifest.csv into train/val/test_ood by event."
    )
    p.add_argument(
        "--manifest", default=Path("data/patches_manifest.csv"), type=Path,
        help="Path to patches_manifest.csv (default: data/patches_manifest.csv)",
    )
    p.add_argument(
        "--output-dir", default=Path("data/splits"), type=Path,
        help="Directory to write split CSVs (default: data/splits)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()

    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    build_splits(manifest_path, output_dir)


if __name__ == "__main__":
    main()
