"""
scripts/polygon_eval.py

Polygon-level evaluation of a full-scene probability map against reference
avalanche polygons.

Pipeline:
  1. Load predicted probability map (GeoTIFF from scene_inference.py)
  2. Threshold at a given value → binary prediction raster
  3. Extract connected components → predicted polygons (via rasterio/shapely)
  4. Load reference polygons from GT GeoPackage
  5. Match predicted vs reference using IoU ≥ threshold (default 0.1,
     matching Gattimgatti et al. 2026)
  6. Compute and report precision, recall, F1, F2 at polygon level

Two evaluation modes
--------------------
1. IoU-based polygon matching (--mode iou, default, same as Gattimgatti 2026):
   - Threshold the probability map → binary raster
   - Extract connected components → predicted polygons
   - Match predicted vs reference using IoU ≥ iou_threshold
   - NOTE: this mode is most meaningful for pixel-level segmentation models.
     Patch classifiers produce coarse blobs (≥ patch-size) that systematically
     over-predict spatial extent; IoU scores are very low against small GT
     polygons regardless of correct detection.

2. Hit-rate (--mode hitrate):
   - For each reference polygon, extract max predicted probability within its
     boundary. A polygon is "detected" if max_prob ≥ threshold.
   - Appropriate for patch classifiers: measures whether the model assigns
     high probability anywhere inside each reference polygon.
   - Only detections (no false positives from blob extent) — reported as
     hit_rate = TP / n_ref.

Usage:
    python scripts/polygon_eval.py \\
        --prob-map results/scene/d4_bitemporal_frac0p5_prob.tif \\
        --gt-path  data/raw/Tromso_20241220/Tromso_20241220_GT.gpkg \\
        --threshold 0.861 \\
        --output   results/scene/d4_bitemporal_frac0p5_polygon_eval.json

    # Sweep thresholds to find F2-optimal
    python scripts/polygon_eval.py \\
        --prob-map results/scene/d4_bitemporal_frac0p5_prob.tif \\
        --gt-path  data/raw/Tromso_20241220/Tromso_20241220_GT.gpkg \\
        --sweep-thresholds

    # Hit-rate mode (appropriate for patch classifiers)
    python scripts/polygon_eval.py \\
        --prob-map results/scene/d4_bitemporal_frac0p5_prob.tif \\
        --gt-path  data/raw/Tromso_20241220/Tromso_20241220_GT.gpkg \\
        --mode hitrate --sweep-thresholds
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import numpy as np

try:
    import rasterio
    import rasterio.features
    from rasterio.crs import CRS
    import geopandas as gpd
    from shapely.geometry import shape, mapping
    from shapely.validation import make_valid
except ImportError as e:
    print(f"Missing dependency: {e}. Run inside the Apptainer container.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Minimum predicted blob area (pixels) to count as a polygon — filters noise
MIN_PRED_AREA_PX = 9   # ~750 m² at 9.23 m res; Gattimgatti use no explicit filter


# ---------------------------------------------------------------------------
# Load and threshold probability map
# ---------------------------------------------------------------------------

def load_prob_map(prob_path: Path) -> tuple[np.ndarray, dict]:
    """Return (prob_array [H, W] float32, rasterio profile)."""
    with rasterio.open(prob_path) as src:
        prob = src.read(1).astype(np.float32)
        profile = src.profile
    log.info("Loaded prob map %s  shape=%s  range=[%.4f, %.4f]",
             prob_path.name, prob.shape, float(prob.min()), float(prob.max()))
    return prob, profile


def threshold_to_binary(prob: np.ndarray, threshold: float) -> np.ndarray:
    """Return uint8 binary mask (1=predicted avalanche)."""
    return (prob >= threshold).astype(np.uint8)


# ---------------------------------------------------------------------------
# Extract connected components → predicted polygons
# ---------------------------------------------------------------------------

def extract_predicted_polygons(
    binary: np.ndarray,
    profile: dict,
    min_area_px: int = MIN_PRED_AREA_PX,
) -> gpd.GeoDataFrame:
    """
    Vectorise connected components of the binary prediction mask.

    Returns a GeoDataFrame with columns: geometry, area_px.
    """
    transform = profile["transform"]
    crs       = profile["crs"]

    shapes = list(rasterio.features.shapes(
        binary, mask=binary, transform=transform
    ))

    if not shapes:
        log.warning("No predicted polygons found at this threshold.")
        return gpd.GeoDataFrame(columns=["geometry", "area_px"], crs=crs)

    geoms, areas = [], []
    for geom_dict, value in shapes:
        if value != 1:
            continue
        geom = make_valid(shape(geom_dict))
        area_px = geom.area / (abs(transform.a) * abs(transform.e))
        if area_px < min_area_px:
            continue
        geoms.append(geom)
        areas.append(area_px)

    gdf = gpd.GeoDataFrame({"geometry": geoms, "area_px": areas}, crs=crs)
    log.info("Extracted %d predicted polygons (min_area_px=%d)", len(gdf), min_area_px)
    return gdf


# ---------------------------------------------------------------------------
# Load reference polygons
# ---------------------------------------------------------------------------

def load_reference_polygons(gt_path: Path, pred_crs) -> gpd.GeoDataFrame:
    """Load GT GeoPackage and reproject to match prediction CRS if needed."""
    gdf = gpd.read_file(gt_path)
    gdf["geometry"] = gdf["geometry"].apply(make_valid)
    if gdf.crs != pred_crs:
        log.info("Reprojecting GT from %s to %s", gdf.crs, pred_crs)
        gdf = gdf.to_crs(pred_crs)
    log.info("Loaded %d reference polygons from %s", len(gdf), gt_path.name)
    return gdf


# ---------------------------------------------------------------------------
# IoU matching
# ---------------------------------------------------------------------------

def compute_iou(geom_a, geom_b) -> float:
    """Intersection-over-union between two shapely geometries."""
    try:
        inter = geom_a.intersection(geom_b).area
        if inter == 0.0:
            return 0.0
        union = geom_a.union(geom_b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def match_polygons(
    pred_gdf: gpd.GeoDataFrame,
    ref_gdf: gpd.GeoDataFrame,
    iou_threshold: float = 0.1,
) -> dict:
    """
    Greedy IoU matching between predicted and reference polygons.

    Returns dict with:
        tp_indices_pred  — indices into pred_gdf that are TPs
        tp_indices_ref   — indices into ref_gdf that are TPs
        fp_indices       — predicted polygon indices that are FPs
        fn_indices       — reference polygon indices that are FNs
        iou_matrix       — [n_pred, n_ref] IoU values
        match_iou        — IoU for each TP pair {pred_idx: iou}
    """
    n_pred = len(pred_gdf)
    n_ref  = len(ref_gdf)

    if n_pred == 0:
        return {
            "tp_indices_pred": [], "tp_indices_ref": [],
            "fp_indices": [], "fn_indices": list(range(n_ref)),
            "iou_matrix": np.zeros((0, n_ref)),
            "match_iou": {},
        }

    if n_ref == 0:
        return {
            "tp_indices_pred": [], "tp_indices_ref": [],
            "fp_indices": list(range(n_pred)), "fn_indices": [],
            "iou_matrix": np.zeros((n_pred, 0)),
            "match_iou": {},
        }

    # Build IoU matrix using spatial index for efficiency
    log.info("Computing IoU matrix (%d pred × %d ref)…", n_pred, n_ref)
    iou_matrix = np.zeros((n_pred, n_ref), dtype=np.float32)

    ref_sindex = ref_gdf.sindex

    for i, pred_geom in enumerate(pred_gdf.geometry):
        # Candidate refs via bounding box
        candidates = list(ref_sindex.intersection(pred_geom.bounds))
        for j in candidates:
            iou_matrix[i, j] = compute_iou(pred_geom, ref_gdf.geometry.iloc[j])

    # Greedy matching: highest IoU first
    matched_pred = set()
    matched_ref  = set()
    tp_pred, tp_ref, match_iou = [], [], {}

    # Get all (iou, i, j) pairs above threshold, sorted descending
    above = np.argwhere(iou_matrix >= iou_threshold)
    if len(above) > 0:
        scores = iou_matrix[above[:, 0], above[:, 1]]
        order  = np.argsort(-scores)
        for k in order:
            i, j = above[k]
            if i in matched_pred or j in matched_ref:
                continue
            matched_pred.add(i)
            matched_ref.add(j)
            tp_pred.append(int(i))
            tp_ref.append(int(j))
            match_iou[int(i)] = float(iou_matrix[i, j])

    fp_indices = [i for i in range(n_pred) if i not in matched_pred]
    fn_indices = [j for j in range(n_ref)  if j not in matched_ref]

    return {
        "tp_indices_pred": tp_pred,
        "tp_indices_ref":  tp_ref,
        "fp_indices":      fp_indices,
        "fn_indices":      fn_indices,
        "iou_matrix":      iou_matrix,
        "match_iou":       match_iou,
    }


# ---------------------------------------------------------------------------
# Compute F-scores
# ---------------------------------------------------------------------------

def f_beta(precision: float, recall: float, beta: float) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def polygon_metrics(match: dict, n_ref: int) -> dict:
    tp = len(match["tp_indices_ref"])
    fp = len(match["fp_indices"])
    fn = len(match["fn_indices"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = f_beta(precision, recall, beta=1.0)
    f2 = f_beta(precision, recall, beta=2.0)
    hit_rate = tp / n_ref if n_ref > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "f2":        f2,
        "hit_rate":  hit_rate,
        "n_pred":    tp + fp,
        "n_ref":     n_ref,
    }


# ---------------------------------------------------------------------------
# Main evaluation at a single threshold
# ---------------------------------------------------------------------------

def evaluate_at_threshold(
    prob: np.ndarray,
    profile: dict,
    ref_gdf: gpd.GeoDataFrame,
    threshold: float,
    iou_threshold: float,
    min_area_px: int,
) -> dict:
    binary   = threshold_to_binary(prob, threshold)
    pred_gdf = extract_predicted_polygons(binary, profile, min_area_px)
    match    = match_polygons(pred_gdf, ref_gdf, iou_threshold)
    metrics  = polygon_metrics(match, n_ref=len(ref_gdf))
    metrics["threshold"]     = threshold
    metrics["iou_threshold"] = iou_threshold
    metrics["mean_match_iou"] = (
        float(np.mean(list(match["match_iou"].values())))
        if match["match_iou"] else 0.0
    )
    return metrics


# ---------------------------------------------------------------------------
# Argument parsing + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polygon-level avalanche detection evaluation.")
    p.add_argument("--prob-map", required=True,
                   help="Predicted probability GeoTIFF from scene_inference.py.")
    p.add_argument("--gt-path", required=True,
                   help="Reference polygon GeoPackage (.gpkg).")
    p.add_argument("--threshold", type=float, default=0.861,
                   help="Decision threshold (default: F2-optimal for d4_bitemporal_frac0p5).")
    p.add_argument("--iou-threshold", type=float, default=0.1,
                   help="IoU threshold for matching (default: 0.1, same as Gattimgatti 2026).")
    p.add_argument("--min-area-px", type=int, default=MIN_PRED_AREA_PX,
                   help="Minimum predicted blob area in pixels.")
    p.add_argument("--mode", choices=["iou", "hitrate"], default="iou",
                   help="Evaluation mode: 'iou' (polygon matching) or 'hitrate' "
                        "(max prob within GT polygon). Default: iou.")
    p.add_argument("--sweep-thresholds", action="store_true",
                   help="Sweep thresholds 0.1–0.95 and report F2-optimal.")
    p.add_argument("--output", default=None,
                   help="Save results JSON to this path.")
    return p.parse_args()


def hitrate_eval(
    prob_path: Path,
    ref_gdf: gpd.GeoDataFrame,
    thresholds: list[float],
) -> list[dict]:
    """
    For each reference polygon, extract max predicted prob within its boundary.
    A polygon is 'detected' (TP) if max_prob >= threshold.
    Returns list of per-threshold result dicts.
    """
    import rasterio.mask

    log.info("Computing max prob within each of %d reference polygons…", len(ref_gdf))
    max_probs = []
    for geom in ref_gdf.geometry:
        try:
            with rasterio.open(prob_path) as src:
                masked, _ = rasterio.mask.mask(
                    src, [mapping(geom)], crop=True, nodata=0.0
                )
            vals = masked[0].flatten()
            vals = vals[vals > 0]
            max_probs.append(float(vals.max()) if len(vals) > 0 else 0.0)
        except Exception:
            max_probs.append(0.0)

    max_probs = np.array(max_probs)
    n_ref = len(ref_gdf)

    rows = []
    for thr in thresholds:
        tp = int((max_probs >= thr).sum())
        fn = n_ref - tp
        hit_rate = tp / n_ref if n_ref > 0 else 0.0
        rows.append({
            "threshold": thr,
            "tp": tp, "fn": fn, "n_ref": n_ref,
            "hit_rate": hit_rate,
        })
        log.info("  thr=%.3f  TP=%d FN=%d  hit_rate=%.3f", thr, tp, fn, hit_rate)

    return rows


def main() -> None:
    args = parse_args()

    prob_path = Path(args.prob_map)
    gt_path   = Path(args.gt_path)

    prob, profile = load_prob_map(prob_path)
    ref_gdf = load_reference_polygons(gt_path, profile["crs"])

    if args.mode == "hitrate":
        if args.sweep_thresholds:
            thresholds = [round(t, 3) for t in np.arange(0.05, 1.0, 0.05).tolist()]
        else:
            thresholds = [args.threshold]

        rows = hitrate_eval(prob_path, ref_gdf, thresholds)

        best = max(rows, key=lambda r: r["hit_rate"])
        print("\n=== Hit-rate evaluation (max prob within GT polygon) ===")
        print("%-8s  %-6s  %-6s  %-8s" % ("thr", "TP", "FN", "hit_rate"))
        print("-" * 36)
        for r in rows:
            print("%-8.3f  %-6d  %-6d  %-8.4f" % (
                r["threshold"], r["tp"], r["fn"], r["hit_rate"]))
        print("\nBest hit_rate: %.4f at threshold %.3f" % (best["hit_rate"], best["threshold"]))

        result = {"mode": "hitrate", "rows": rows, "best": best}
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            log.info("Results saved to %s", out_path)
        return

    if args.sweep_thresholds:
        thresholds = [round(t, 3) for t in np.arange(0.05, 1.0, 0.05).tolist()]
        rows = []
        log.info("Sweeping %d thresholds…", len(thresholds))
        for thr in thresholds:
            m = evaluate_at_threshold(
                prob, profile, ref_gdf, thr,
                args.iou_threshold, args.min_area_px,
            )
            rows.append(m)
            log.info(
                "  thr=%.3f  TP=%d FP=%d FN=%d  P=%.3f R=%.3f F1=%.3f F2=%.3f",
                thr, m["tp"], m["fp"], m["fn"],
                m["precision"], m["recall"], m["f1"], m["f2"],
            )

        best_f2 = max(rows, key=lambda r: r["f2"])
        best_f1 = max(rows, key=lambda r: r["f1"])
        print("\n=== Sweep results ===")
        print(f"Best F2: {best_f2['f2']:.4f}  at threshold {best_f2['threshold']:.3f}"
              f"  (P={best_f2['precision']:.3f} R={best_f2['recall']:.3f}"
              f"  TP={best_f2['tp']} FP={best_f2['fp']} FN={best_f2['fn']})")
        print(f"Best F1: {best_f1['f1']:.4f}  at threshold {best_f1['threshold']:.3f}"
              f"  (P={best_f1['precision']:.3f} R={best_f1['recall']:.3f}"
              f"  TP={best_f1['tp']} FP={best_f1['fp']} FN={best_f1['fn']})")

        result = {"sweep": rows, "best_f2": best_f2, "best_f1": best_f1}
    else:
        m = evaluate_at_threshold(
            prob, profile, ref_gdf, args.threshold,
            args.iou_threshold, args.min_area_px,
        )
        print("\n=== Polygon-level evaluation ===")
        print(f"Threshold:        {m['threshold']:.4f}")
        print(f"IoU threshold:    {m['iou_threshold']:.2f}")
        print(f"Reference polys:  {m['n_ref']}")
        print(f"Predicted polys:  {m['n_pred']}")
        print(f"TP / FP / FN:     {m['tp']} / {m['fp']} / {m['fn']}")
        print(f"Precision:        {m['precision']:.4f}")
        print(f"Recall (hit rate):{m['recall']:.4f}")
        print(f"F1:               {m['f1']:.4f}")
        print(f"F2:               {m['f2']:.4f}")
        print(f"Mean match IoU:   {m['mean_match_iou']:.4f}")
        result = m

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove non-serialisable iou_matrix before saving
        if "sweep" in result:
            for row in result["sweep"]:
                row.pop("iou_matrix", None)
            result["best_f2"].pop("iou_matrix", None)
            result["best_f1"].pop("iou_matrix", None)
        else:
            result.pop("iou_matrix", None)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
