"""
extract_patches.py

Extract 64×64 pixel patches from terrain-corrected Sentinel-1 GeoTIFFs
(output of preprocess_snap.py) centered on labeled avalanche debris polygons,
plus matched negative (clean snowpack) patches sampled randomly from the
same scene.

Inputs:
  --raster-dir   Directory of *_processed.tif files (VV=band1, VH=band2)
  --labels       GeoJSON or shapefile of debris polygons (any CRS; reprojected
                 to match each raster automatically)
  --output-dir   Where to write .npy patches and patches_manifest.csv

Output layout:
  output-dir/
    patches/
      <scene_stem>_r<row>_c<col>_pos.npy   # shape [2, 64, 64], float32
      <scene_stem>_r<row>_c<col>_neg.npy
    patches_manifest.csv   # appended across all scenes

Manifest columns:
  filename, label (1=debris / 0=clean), scene, row, col, centroid_x, centroid_y

Usage:
    python extract_patches.py \\
        --raster-dir /data/processed \\
        --labels     /data/labels/avalanche_debris.geojson \\
        --output-dir /data/patches \\
        --neg-ratio  1 \\
        --seed       42
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from shapely.geometry import box

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

PATCH_SIZE = 64
HALF = PATCH_SIZE // 2  # 32

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _pixel_to_geo(transform, row: int, col: int) -> tuple[float, float]:
    """Return (x, y) map coordinates of the centre of pixel (row, col)."""
    x = transform.c + col * transform.a + 0.5 * transform.a
    y = transform.f + row * transform.e + 0.5 * transform.e
    return x, y


def _geo_to_pixel(transform, x: float, y: float) -> tuple[int, int]:
    """Return (row, col) for map coordinate (x, y) — nearest pixel centre."""
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    return row, col


def _patch_in_bounds(row: int, col: int, height: int, width: int) -> bool:
    return (
        row - HALF >= 0
        and col - HALF >= 0
        and row + HALF <= height
        and col + HALF <= width
    )


def _extract(data: np.ndarray, row: int, col: int) -> np.ndarray:
    """Extract [2, 64, 64] patch centred at (row, col). Assumes in-bounds."""
    return data[:, row - HALF : row + HALF, col - HALF : col + HALF].copy()


def _has_nodata(patch: np.ndarray) -> bool:
    """Return True if any pixel is NaN or non-finite."""
    return not np.all(np.isfinite(patch))


# ---------------------------------------------------------------------------
# Core per-scene logic
# ---------------------------------------------------------------------------

def process_scene(
    raster_path: Path,
    labels_gdf: gpd.GeoDataFrame,
    output_dir: Path,
    neg_ratio: int,
    rng: np.random.Generator,
    max_neg_attempts: int = 5000,
) -> list[dict]:
    """
    Extract positive and negative patches from one scene.
    Returns a list of manifest row dicts.
    """
    patch_dir = output_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    scene_stem = raster_path.stem

    with rasterio.open(raster_path) as src:
        if src.count < 2:
            log.warning("%s: expected 2 bands (VV, VH), got %d — skipping.", scene_stem, src.count)
            return records

        raster_crs: CRS = src.crs
        transform = src.transform
        height, width = src.height, src.width

        # Reproject labels to raster CRS for this scene
        scene_labels = labels_gdf.to_crs(raster_crs)

        # Clip to raster extent (discard polygons outside this scene)
        scene_box = box(*src.bounds)
        scene_labels = scene_labels[scene_labels.geometry.intersects(scene_box)].copy()

        if scene_labels.empty:
            log.info("%s: no overlapping debris polygons — skipping.", scene_stem)
            return records

        log.info("%s: %d debris polygon(s) found.", scene_stem, len(scene_labels))

        # Read full raster into memory as float32 [2, H, W]
        data = src.read([1, 2]).astype(np.float32)

    # ------------------------------------------------------------------
    # Positive patches — one per polygon centroid
    # ------------------------------------------------------------------
    positive_pixels: list[tuple[int, int]] = []  # for negative exclusion

    for _, poly_row in scene_labels.iterrows():
        centroid = poly_row.geometry.centroid
        row, col = _geo_to_pixel(transform, centroid.x, centroid.y)

        if not _patch_in_bounds(row, col, height, width):
            log.debug("%s: centroid (%d,%d) too close to edge — skipped.", scene_stem, row, col)
            continue

        patch = _extract(data, row, col)

        if _has_nodata(patch):
            log.debug("%s: positive patch (%d,%d) contains nodata — skipped.", scene_stem, row, col)
            continue

        cx, cy = _pixel_to_geo(transform, row, col)
        fname = f"{scene_stem}_r{row}_c{col}_pos.npy"
        np.save(patch_dir / fname, patch)
        positive_pixels.append((row, col))
        records.append(dict(
            filename=fname, label=1, scene=scene_stem,
            row=row, col=col, centroid_x=cx, centroid_y=cy,
        ))

    n_pos = len(positive_pixels)
    if n_pos == 0:
        log.warning("%s: no valid positive patches extracted.", scene_stem)
        return records

    log.info("%s: extracted %d positive patch(es).", scene_stem, n_pos)

    # ------------------------------------------------------------------
    # Negative patches — random sampling, no overlap with any debris polygon
    # ------------------------------------------------------------------
    # Build a set of occupied (row, col) centres for fast exclusion.
    # A candidate is rejected if its 64×64 window overlaps any debris polygon.
    # We use a pixel-level bounding-box check first (cheap), then shapely
    # intersection (only when bbox overlaps).

    n_neg_target = n_pos * neg_ratio
    neg_collected = 0
    attempts = 0

    # Pre-union all debris geometries for overlap checks
    debris_union = scene_labels.geometry.union_all()

    while neg_collected < n_neg_target and attempts < max_neg_attempts:
        attempts += 1
        # Sample a random pixel centre that allows a full patch
        row = int(rng.integers(HALF, height - HALF))
        col = int(rng.integers(HALF, width - HALF))

        # Convert patch corners to map coords for overlap test
        x0, y0 = _pixel_to_geo(transform, row - HALF, col - HALF)
        x1, y1 = _pixel_to_geo(transform, row + HALF, col + HALF)
        patch_box = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

        if patch_box.intersects(debris_union):
            continue

        patch = _extract(data, row, col)
        if _has_nodata(patch):
            continue

        cx, cy = _pixel_to_geo(transform, row, col)
        fname = f"{scene_stem}_r{row}_c{col}_neg.npy"
        np.save(patch_dir / fname, patch)
        neg_collected += 1
        records.append(dict(
            filename=fname, label=0, scene=scene_stem,
            row=row, col=col, centroid_x=cx, centroid_y=cy,
        ))

    if neg_collected < n_neg_target:
        log.warning(
            "%s: only %d/%d negative patches collected after %d attempts.",
            scene_stem, neg_collected, n_neg_target, max_neg_attempts,
        )
    else:
        log.info("%s: extracted %d negative patch(es).", scene_stem, neg_collected)

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract 64×64 patches from Sentinel-1 GeoTIFFs around debris polygons."
    )
    p.add_argument("--raster-dir", required=True, type=Path,
                   help="Directory containing *_processed.tif files.")
    p.add_argument("--labels", required=True, type=Path,
                   help="GeoJSON or shapefile of avalanche debris polygons.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Root output directory for patches and manifest.")
    p.add_argument("--neg-ratio", default=1, type=int,
                   help="Negative patches per positive patch (default: 1).")
    p.add_argument("--seed", default=42, type=int,
                   help="Random seed for reproducible negative sampling.")
    p.add_argument("--max-neg-attempts", default=5000, type=int,
                   help="Max random draws per scene when sampling negatives.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    raster_dir: Path = args.raster_dir.resolve()
    labels_path: Path = args.labels.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not raster_dir.is_dir():
        log.error("--raster-dir not found: %s", raster_dir)
        sys.exit(1)
    if not labels_path.exists():
        log.error("--labels not found: %s", labels_path)
        sys.exit(1)

    raster_files = sorted(raster_dir.glob("*_processed.tif"))
    if not raster_files:
        log.error("No *_processed.tif files found in %s", raster_dir)
        sys.exit(1)

    log.info("Loading labels from %s", labels_path)
    labels_gdf = gpd.read_file(labels_path)
    log.info("Label CRS: %s  |  %d polygon(s)", labels_gdf.crs, len(labels_gdf))

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    manifest_path = output_dir / "patches_manifest.csv"
    fieldnames = ["filename", "label", "scene", "row", "col", "centroid_x", "centroid_y"]

    all_records: list[dict] = []
    for raster_path in raster_files:
        log.info("Processing scene: %s", raster_path.name)
        records = process_scene(
            raster_path=raster_path,
            labels_gdf=labels_gdf,
            output_dir=output_dir,
            neg_ratio=args.neg_ratio,
            rng=rng,
            max_neg_attempts=args.max_neg_attempts,
        )
        all_records.extend(records)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    n_pos = sum(1 for r in all_records if r["label"] == 1)
    n_neg = sum(1 for r in all_records if r["label"] == 0)
    log.info("Manifest written: %s", manifest_path)
    log.info("Total patches — positive: %d  negative: %d", n_pos, n_neg)


if __name__ == "__main__":
    main()
