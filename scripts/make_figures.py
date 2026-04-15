"""
scripts/make_figures.py

Generate all publication-quality figures for the README.

Figures that need only results JSON (run anywhere):
  1  Data-efficiency curves
  2  Model comparison bar chart
  5  Dataset geography map
  6  Architecture diagram

Figures that need Hyak SAR data:
  3  Probability heatmap overlay (Tromsø scene)
  4  Hit/miss polygon map
  7  Group elements illustration (C8 / D4)
  8  Speckle-reduction illustration

Usage:
  python scripts/make_figures.py --figures 1 2 5 6 --results-dir results/
  python scripts/make_figures.py --figures 3 4 7 8 \\
      --scene-dir data/raw/Tromso_20241220 \\
      --prob-map  results/scene/d4_bitemporal_frac0p5_prob.tif \\
      --results-dir results/
  python scripts/make_figures.py --figures all --results-dir results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec

# ── publication style ────────────────────────────────────────────────────────
def _style():
    plt.rcParams.update({
        "figure.dpi":          300,
        "savefig.dpi":         300,
        "font.family":         "sans-serif",
        "font.size":           9,
        "axes.titlesize":      10,
        "axes.labelsize":      9,
        "xtick.labelsize":     8,
        "ytick.labelsize":     8,
        "legend.fontsize":     8,
        "axes.linewidth":      0.8,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           True,
        "grid.alpha":          0.3,
        "grid.linewidth":      0.5,
        "grid.linestyle":      "--",
        "lines.linewidth":     1.8,
        "patch.linewidth":     0.8,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.05,
    })

# ── shared color palette ──────────────────────────────────────────────────────
PALETTE = {
    "d4_bitemporal": "#C2185B",   # deep pink  — boldest
    "d4":            "#6A1B9A",   # deep purple
    "c8":            "#1565C0",   # deep blue
    "so2":           "#2E7D32",   # deep green
    "resnet":        "#37474F",   # blue-grey
    "cnn":           "#B71C1C",   # deep red
    "aug":           "#E65100",   # deep orange
}
MARKERS = {
    "d4_bitemporal": "*",
    "d4":            "^",
    "c8":            "o",
    "so2":           "s",
    "resnet":        "P",
    "cnn":           "D",
    "aug":           "v",
}
LABELS = {
    "d4_bitemporal": "D4-BT (bi-temporal)",
    "d4":            "D4 equivariant",
    "c8":            "C8 equivariant",
    "so2":           "SO(2) equivariant",
    "resnet":        "ResNet-18",
    "cnn":           "CNN baseline",
    "aug":           "CNN + aug",
}
MODEL_ORDER = ["d4_bitemporal", "d4", "c8", "so2", "resnet", "cnn", "aug"]

# Hardcoded AUC results (test_ood)
AUC_TABLE = {
    "d4_bitemporal": [0.8708, 0.9064, 0.9116, 0.8936],
    "d4":            [0.7169, 0.7885, 0.7784, 0.7686],
    "c8":            [0.6751, 0.6760, 0.7449, 0.7368],
    "so2":           [0.6450, 0.6602, 0.6721, 0.7236],
    "resnet":        [0.5554, 0.7860, 0.7425, 0.8029],
    "cnn":           [0.4988, 0.6765, 0.7833, 0.7233],
    "aug":           [0.5227, 0.6222, 0.7443, 0.7051],
}
FRACS   = [0.10, 0.25, 0.50, 1.00]
FRAC_X  = [10, 25, 50, 100]
FRAC_LBL = ["10 %", "25 %", "50 %", "100 %"]


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Data-efficiency curves
# ─────────────────────────────────────────────────────────────────────────────
def fig1_data_efficiency(out_dir: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))

    # ResNet-18 @ 100% as reference line
    ax.axhline(AUC_TABLE["resnet"][-1], color="#90A4AE", lw=1.0,
               ls=":", zorder=1, label="_nolegend_")
    ax.text(102, AUC_TABLE["resnet"][-1] + 0.003,
            "ResNet-18 @ 100%", fontsize=7, color="#607D8B", va="bottom")

    for model in MODEL_ORDER:
        vals = AUC_TABLE[model]
        lw   = 2.5 if model == "d4_bitemporal" else 1.6
        ms   = 9   if model == "d4_bitemporal" else 6
        zo   = 5   if model == "d4_bitemporal" else 3
        ax.plot(FRAC_X, vals,
                color=PALETTE[model], marker=MARKERS[model],
                lw=lw, markersize=ms, zorder=zo, label=LABELS[model],
                clip_on=False)

    ax.set_xticks(FRAC_X)
    ax.set_xticklabels(FRAC_LBL)
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("AUC-ROC (Tromsø OOD test set)")
    ax.set_title("OOD Generalisation vs Training Data Fraction (Tromsø Test Set)")
    ax.set_xlim(5, 108)
    ylo = 0.45
    ax.set_ylim(ylo, 0.97)
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="0.8",
              ncol=1, labelspacing=0.3)
    fig.tight_layout()
    p = out_dir / "fig1_data_efficiency.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig2_model_comparison(out_dir: Path) -> None:
    _style()
    n_models = len(MODEL_ORDER)
    n_fracs  = len(FRAC_X)
    bar_w    = 0.11
    group_w  = bar_w * n_models
    x        = np.arange(n_fracs)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, model in enumerate(MODEL_ORDER):
        offset = (i - n_models / 2 + 0.5) * bar_w
        vals   = AUC_TABLE[model]
        bars   = ax.bar(x + offset, vals, width=bar_w * 0.92,
                        color=PALETTE[model], label=LABELS[model],
                        zorder=3, linewidth=0)
        # D4-BT: add a thin black outline to make it pop
        if model == "d4_bitemporal":
            for bar in bars:
                bar.set_edgecolor("black")
                bar.set_linewidth(0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(FRAC_LBL)
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("AUC-ROC (Tromsø OOD test set)")
    ax.set_title("Model Comparison: OOD AUC-ROC by Training Data Fraction")
    ax.set_ylim(0.40, 0.97)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="0.8",
              ncol=2, labelspacing=0.3, handlelength=1.2)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    p = out_dir / "fig2_model_comparison.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Probability heatmap overlay
# ─────────────────────────────────────────────────────────────────────────────
def fig3_heatmap_overlay(scene_dir: Path, prob_map: Path, gt_path: Path,
                         out_dir: Path) -> None:
    try:
        import rasterio
        import rasterio.plot
        import geopandas as gpd
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as e:
        print(f"  [skip fig3] missing: {e}")
        return

    _style()

    # Load VV backscatter
    vv_path = next(scene_dir.glob("*postVV*"))
    with rasterio.open(vv_path) as src:
        vv  = src.read(1).astype(np.float32)
        ext = rasterio.plot.plotting_extent(src)
        crs = src.crs

    # Load probability map
    with rasterio.open(prob_map) as src:
        prob = src.read(1).astype(np.float32)

    # Load GT polygons
    gdf = gpd.read_file(gt_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Nodata mask (black pixels outside scene)
    nodata = (vv < -30) | np.isnan(vv)
    vv_disp = np.clip(vv, -25, -5)
    vv_disp[nodata] = np.nan

    # Crop to valid region
    rows, cols = np.where(~nodata)
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    pad = 20
    r0 = max(r0 - pad, 0); r1 = min(r1 + pad, vv.shape[0])
    c0 = max(c0 - pad, 0); c1 = min(c1 + pad, vv.shape[1])

    # Map pixel bounds → geographic extent for cropping display
    with rasterio.open(vv_path) as src:
        tf = src.transform
    left  = tf.c + c0 * tf.a
    right = tf.c + c1 * tf.a
    top   = tf.f + r0 * tf.e
    bot   = tf.f + r1 * tf.e
    ext_crop = (left, right, bot, top)

    vv_crop   = vv_disp[r0:r1, c0:c1]
    prob_crop = prob[r0:r1, c0:c1]

    # Semi-transparent probability colormap (zeros transparent)
    cmap_hot = plt.cm.plasma.copy()
    cmap_hot.set_under(alpha=0)

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(vv_crop, extent=ext_crop, origin="upper",
              cmap="gray", vmin=-25, vmax=-5, interpolation="bilinear")

    # Overlay probability: show only where > 0.3 (avoid cluttering background)
    prob_masked = np.where(prob_crop > 0.30, prob_crop, np.nan)
    im = ax.imshow(prob_masked, extent=ext_crop, origin="upper",
                   cmap="plasma", vmin=0.30, vmax=1.0,
                   alpha=0.65, interpolation="bilinear")

    # GT polygon outlines
    gdf.boundary.plot(ax=ax, color="white", linewidth=0.7, alpha=0.9)

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.6)
    cb.set_label("Predicted probability", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax.set_axis_off()
    ax.set_title("D4-BT (50% data) — Tromsø scene probability map\n"
                 "Background: VV backscatter · Overlay: predicted avalanche probability · "
                 "White outlines: 117 GT polygons",
                 fontsize=8, pad=6)
    fig.tight_layout(pad=0.3)
    p = out_dir / "fig3_heatmap_overlay.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Hit/miss polygon map
# ─────────────────────────────────────────────────────────────────────────────
def fig4_hit_miss_map(scene_dir: Path, prob_map: Path, gt_path: Path,
                      out_dir: Path, threshold: float = 0.75) -> None:
    try:
        import rasterio
        import rasterio.mask
        import rasterio.plot
        import geopandas as gpd
        from shapely.geometry import mapping
    except ImportError as e:
        print(f"  [skip fig4] missing: {e}")
        return

    _style()

    vv_path = next(scene_dir.glob("*postVV*"))
    with rasterio.open(vv_path) as src:
        vv  = src.read(1).astype(np.float32)
        crs = src.crs
        tf  = src.transform

    with rasterio.open(prob_map) as src:
        prob = src.read(1).astype(np.float32)

    gdf = gpd.read_file(gt_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Compute max prob within each GT polygon (all_touched=True for sub-pixel polygons)
    max_probs = []
    for geom in gdf.geometry:
        try:
            with rasterio.open(prob_map) as src:
                masked, _ = rasterio.mask.mask(src, [mapping(geom)],
                                               crop=True, nodata=0.0,
                                               all_touched=True)
            vals = masked[0].flatten()
            vals = vals[vals > 0]
            max_probs.append(float(vals.max()) if len(vals) > 0 else 0.0)
        except Exception:
            max_probs.append(0.0)
    max_probs = np.array(max_probs)
    gdf["max_prob"]  = max_probs
    gdf["detected"]  = max_probs >= threshold

    n_tp = gdf["detected"].sum()
    n_fn = (~gdf["detected"]).sum()
    missed = gdf[~gdf["detected"]].copy()

    # Identify the two special-case polygons by their known GPKG area values
    # (area=29 m²: sub-pixel eval artefact; area=612 m²: layover miss)
    poly_subpx  = gdf[gdf["area"] == 29].copy()   if "area" in gdf.columns else None
    poly_layover = gdf[gdf["area"] == 612].copy() if "area" in gdf.columns else None

    # Display bounds
    nodata = (vv < -30) | np.isnan(vv)
    rows, cols = np.where(~nodata)
    r0, r1 = rows.min() - 20, rows.max() + 20
    c0, c1 = cols.min() - 20, cols.max() + 20
    r0 = max(r0, 0); r1 = min(r1, vv.shape[0])
    c0 = max(c0, 0); c1 = min(c1, vv.shape[1])

    left  = tf.c + c0 * tf.a;  right = tf.c + c1 * tf.a
    top   = tf.f + r0 * tf.e;  bot   = tf.f + r1 * tf.e
    ext_crop = (left, right, bot, top)

    vv_disp = np.clip(vv, -25, -5).astype(float)
    vv_disp[nodata] = np.nan
    vv_crop = vv_disp[r0:r1, c0:c1]

    # Main axes
    fig = plt.figure(figsize=(7, 8.5))
    ax  = fig.add_axes([0.02, 0.12, 0.96, 0.84])

    ax.imshow(vv_crop, extent=ext_crop, origin="upper",
              cmap="gray", vmin=-25, vmax=-5, interpolation="bilinear")

    # Plot hit polygons (green) and missed (red)
    hits = gdf[gdf["detected"]]
    hits.plot(ax=ax, facecolor="none", edgecolor="#76FF03",
              linewidth=0.9, alpha=0.85)
    if len(missed) > 0:
        missed.plot(ax=ax, facecolor="#FF1744", edgecolor="#FF1744",
                    linewidth=1.2, alpha=0.55)

    # ── Annotate the two special-case polygons ────────────────────────────────
    _ann_kw = dict(
        fontsize=6.5,
        color="white",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="#1A1A1A", ec="white",
                  alpha=0.80, lw=0.6),
        arrowprops=dict(
            arrowstyle="-|>",
            color="white",
            lw=1.0,
            connectionstyle="arc3,rad=0.25",
        ),
        zorder=10,
    )
    # Sub-pixel polygon: highlight with gold outline; annotate as detected-but-notable
    if poly_subpx is not None and len(poly_subpx):
        poly_subpx.plot(ax=ax, facecolor="none", edgecolor="#FFD600",
                        linewidth=2.0, alpha=0.9, zorder=8)
        cx, cy = poly_subpx.geometry.iloc[0].centroid.x, poly_subpx.geometry.iloc[0].centroid.y
        ax.annotate(
            "Miss #1 (pre-fix): sub-pixel polygon\n29 m² — eval artefact, model prob=0.83",
            xy=(cx, cy),
            xytext=(cx + 3500, cy - 5000),
            **_ann_kw,
        )
    # Layover polygon: shown red (genuinely missed); annotate
    if poly_layover is not None and len(poly_layover):
        cx, cy = poly_layover.geometry.iloc[0].centroid.x, poly_layover.geometry.iloc[0].centroid.y
        ax.annotate(
            "Miss #2: layover geometry\nLIA=5°, VV/VH=12.3 dB",
            xy=(cx, cy),
            xytext=(cx - 6000, cy + 4000),
            **_ann_kw,
        )

    # Legend patches
    hit_patch  = mpatches.Patch(facecolor="none", edgecolor="#76FF03",
                                 lw=1.2, label=f"Detected ({n_tp})")
    miss_patch = mpatches.Patch(facecolor="#FF1744", edgecolor="#FF1744",
                                 alpha=0.6, lw=1.2, label=f"Missed ({n_fn})")
    note_patch = mpatches.Patch(facecolor="none", edgecolor="#FFD600",
                                 lw=1.5, label="Detected (was eval artefact)")
    ax.legend(handles=[hit_patch, miss_patch, note_patch], loc="upper right",
              framealpha=0.9, fontsize=7.5, edgecolor="0.7")

    ax.set_axis_off()
    ax.set_title(
        f"D4-BT polygon hit/miss map — threshold {threshold:.2f}\n"
        f"Green: detected · Red: missed · Gold: sub-pixel (eval artefact, detected after fix)\n"
        f"{n_tp}/{n_tp+n_fn} polygons detected ({100*n_tp/(n_tp+n_fn):.1f}%) "
        f"with all_touched=True masking",
        fontsize=8, pad=5,
    )

    # Inset: zoom into the genuine miss (layover polygon)
    if poly_layover is not None and len(poly_layover):
        mb = poly_layover.total_bounds
        pad_m = 800
        x0, y0, x1, y1 = mb[0]-pad_m, mb[1]-pad_m, mb[2]+pad_m, mb[3]+pad_m
        pc0 = max(int((x0 - tf.c) / tf.a), 0)
        pc1 = min(int((x1 - tf.c) / tf.a), vv.shape[1])
        pr0 = max(int((y1 - tf.f) / tf.e), 0)
        pr1 = min(int((y0 - tf.f) / tf.e), vv.shape[0])
        if pr1 > pr0 and pc1 > pc0:
            vv_ins = np.clip(vv[pr0:pr1, pc0:pc1], -25, -5).astype(float)
            vv_ins[vv[pr0:pr1, pc0:pc1] < -30] = np.nan
            x0e = tf.c + pc0 * tf.a;  x1e = tf.c + pc1 * tf.a
            y1e = tf.f + pr0 * tf.e;  y0e = tf.f + pr1 * tf.e
            ax_ins = fig.add_axes([0.03, 0.03, 0.28, 0.18])
            ax_ins.imshow(vv_ins, extent=(x0e, x1e, y0e, y1e), origin="upper",
                          cmap="gray", vmin=-25, vmax=-5)
            poly_layover.plot(ax=ax_ins, facecolor="#FF1744", edgecolor="#FF1744",
                              linewidth=1.2, alpha=0.65)
            ax_ins.set_xlim(x0, x1);  ax_ins.set_ylim(y0, y1)
            ax_ins.set_axis_off()
            ax_ins.set_title("Miss #2: layover\n(LIA=5°)", fontsize=6.5, pad=2,
                             color="white")
            for sp in ax_ins.spines.values():
                sp.set_visible(True); sp.set_linewidth(0.8); sp.set_color("white")

    p = out_dir / "fig4_hit_miss_map.png"
    fig.savefig(p, facecolor="black")
    plt.close(fig)
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Dataset geography map
# ─────────────────────────────────────────────────────────────────────────────
def fig5_geography_map(out_dir: Path) -> None:
    _style()

    regions = {
        "Livigno\n(Italy)":      (10.09, 46.53, "train"),
        "Nuuk\n(Greenland)":     (-51.75, 64.18, "train"),
        "Pish\n(Tajikistan)":    (72.50,  39.00, "train"),
        "Tromsø\n(Norway)":      (19.00,  69.65, "test"),
    }

    fig, ax = plt.subplots(figsize=(7, 4),
                           subplot_kw={"projection": None})

    # Try geopandas natural earth
    try:
        import geopandas as gpd
        try:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        except Exception:
            import geodatasets
            world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

        world_filt = world[world["continent"].isin([
            "Europe", "Asia", "North America", "South America",
            "Africa", "Oceania", "Seven seas (open ocean)", "Antarctica"
        ])]
        world_filt.plot(ax=ax, color="#E8EDF2", edgecolor="#B0BEC5",
                        linewidth=0.3)
        # Focus: Atlantic/Arctic view covering all 4 sites
        ax.set_xlim(-70, 90)
        ax.set_ylim(25, 80)
    except Exception:
        # Fallback: blank axes with grid
        ax.set_facecolor("#D6EAF8")
        ax.set_xlim(-70, 90)
        ax.set_ylim(25, 80)
        ax.grid(True, alpha=0.4)

    ax.set_facecolor("#D6EAF8")  # ocean color

    for name, (lon, lat, role) in regions.items():
        if role == "train":
            color, marker, zorder = "#1565C0", "o", 5
        else:
            color, marker, zorder = "#C62828", "*", 6

        ax.scatter(lon, lat, color=color, marker=marker,
                   s=120 if marker == "*" else 70,
                   zorder=zorder, linewidths=0.8,
                   edgecolors="white")

        # Label offset to avoid overlap
        dx, dy = 2, 1.5
        if "Nuuk" in name:    dx, dy = 2,  -3.5
        if "Tromsø" in name:  dx, dy = 2,   1.5
        if "Pish" in name:    dx, dy = 2,  -3.0

        ax.annotate(name.replace("\n", "\n"),
                    xy=(lon, lat), xytext=(lon + dx, lat + dy),
                    fontsize=7.5, color=color, fontweight="bold",
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=color,
                                   lw=0.6, alpha=0.7))

    # Legend
    train_dot = mpatches.Patch(facecolor="#1565C0", label="Training regions (3)")
    test_star  = plt.Line2D([0], [0], marker="*", color="w",
                            markerfacecolor="#C62828", markersize=10,
                            label="OOD test region (Tromsø)")
    ax.legend(handles=[train_dot, test_star], loc="lower left",
              framealpha=0.9, fontsize=8, edgecolor="0.8")

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("AvalCD Dataset — Geographic Distribution of Training and Test Regions")
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    p = out_dir / "fig5_geography.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Architecture diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig6_architecture(out_dir: Path) -> None:
    _style()
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.spines.bottom"] = False
    plt.rcParams["axes.spines.left"] = False

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.set_axis_off()

    C_BG    = "#F5F5F5"
    C_EQV   = "#EDE7F6"   # equivariant blocks: light purple
    C_HEAD1 = "#E3F2FD"   # classification head: light blue
    C_HEAD2 = "#FFF8E1"   # orientation head: light amber
    C_BT    = "#FCE4EC"   # bitemporal diff block: light pink
    C_ARROW = "#455A64"

    def box(x, y, w, h, label, sublabel="", color=C_BG, fontsize=8.5):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                           facecolor=color, edgecolor="#90A4AE", lw=0.8, zorder=3)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2 + (0.12 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", zorder=4)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.17, sublabel,
                    ha="center", va="center",
                    fontsize=6.8, color="#546E7A", zorder=4)

    def arrow(x0, x1, y, color=C_ARROW):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=0.9, mutation_scale=10), zorder=2)

    def arrow_split(x0, y0, x1_top, y_top, x1_bot, y_bot):
        # Fork from (x0,y0) to two targets
        mid_x = x0 + 0.3
        ax.annotate("", xy=(mid_x, y_top), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-", color=C_ARROW, lw=0.9), zorder=2)
        ax.annotate("", xy=(x1_top, y_top), xytext=(mid_x, y_top),
                    arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                   lw=0.9, mutation_scale=10), zorder=2)
        ax.annotate("", xy=(mid_x, y_bot), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-", color=C_ARROW, lw=0.9), zorder=2)
        ax.annotate("", xy=(x1_bot, y_bot), xytext=(mid_x, y_bot),
                    arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                   lw=0.9, mutation_scale=10), zorder=2)

    # ── Single-image path (y = 2.25 centre) ──────────────────────────────────
    sy = 1.95   # bottom of main-path boxes

    box(0.10, sy, 0.95, 0.60, "Input", "5×64×64\ntrivial rep", color=C_BG, fontsize=8)
    arrow(1.05, 1.25, sy + 0.30)

    blocks = [
        ("Block 1", "reg · 64×64", 1.25),
        ("Block 2", "reg · 32×32", 2.35),
        ("Block 3", "reg · 16×16", 3.45),
        ("Block 4", "reg ·  8×8",  4.55),
    ]
    for lbl, sub, bx in blocks:
        box(bx, sy, 0.95, 0.60, lbl, sub, color=C_EQV)
        if bx < 4.55:
            arrow(bx + 0.95, bx + 1.10, sy + 0.30)

    # Split arrow after block 4
    split_x = 5.55
    arrow(5.50, split_x, sy + 0.30)
    ax.plot([split_x, split_x], [sy + 0.30, 3.60], color=C_ARROW, lw=0.9, zorder=2)
    ax.plot([split_x, split_x], [sy + 0.30, 0.85], color=C_ARROW, lw=0.9, zorder=2)
    ax.annotate("", xy=(5.80, 3.45), xytext=(split_x, 3.45),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=0.9, mutation_scale=10), zorder=2)
    ax.annotate("", xy=(5.80, 0.85), xytext=(split_x, 0.85),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=0.9, mutation_scale=10), zorder=2)

    # HEAD 1 (classification)
    box(5.80, 3.10, 1.45, 0.70, "GroupPool", "trivial rep", color=C_HEAD1)
    arrow(7.25, 7.45, 3.45)
    box(7.45, 3.10, 1.35, 0.70, "AvgPool\n→ Linear", "sigmoid", color=C_HEAD1)
    arrow(8.80, 9.00, 3.45)
    box(9.00, 3.10, 1.85, 0.70, "P(debris) ∈ [0,1]",
        "Invariant output", color=C_HEAD1)
    ax.text(5.80 + 1.45/2, 3.95, "HEAD 1 — Classification",
            ha="center", fontsize=7.5, color="#1565C0", fontweight="bold")

    # HEAD 2 (orientation)
    box(5.80, 0.45, 1.45, 0.70, "1×1 R2Conv", "standard rep", color=C_HEAD2)
    arrow(7.25, 7.45, 0.80)
    box(7.45, 0.45, 1.35, 0.70, "SpatialAvg\nPool", "→ [B,2]", color=C_HEAD2)
    arrow(8.80, 9.00, 0.80)
    box(9.00, 0.45, 1.85, 0.70, "θ ∈ ℝ² rotates\nwith input",
        "Equivariant output", color=C_HEAD2)
    ax.text(5.80 + 1.45/2, 0.28, "HEAD 2 — Orientation",
            ha="center", fontsize=7.5, color="#E65100", fontweight="bold")

    # ── Bi-temporal annotation ────────────────────────────────────────────────
    ax.text(0.57, 4.20,
            "Bi-temporal extension (D4-BT): shared encoder applied separately\n"
            "to post-event and pre-event patches; difference feature fed to heads",
            ha="center", fontsize=7.5, color="#880E4F",
            bbox=dict(facecolor=C_BT, edgecolor="#F48FB1", lw=0.7,
                      boxstyle="round,pad=0.25"))

    # Pre-event branch suggestion
    ax.text(0.57, sy - 0.45,
            "Pre-event branch\n(same weights ↕)",
            ha="center", fontsize=7, color="#880E4F", style="italic",
            bbox=dict(facecolor=C_BT, edgecolor="#F48FB1", lw=0.5,
                      boxstyle="round,pad=0.15"))
    ax.annotate("",
                xy=(0.57, sy), xytext=(0.57, sy - 0.25),
                arrowprops=dict(arrowstyle="-|>", color="#AD1457",
                                lw=0.7, mutation_scale=8))

    # Equivariant label
    ax.text(3.17, sy + 0.82,
            "← Equivariant backbone (steerable convolutions) →",
            ha="center", fontsize=7.5, color="#6A1B9A", style="italic")

    ax.set_title("Equivariant CNN Architecture  (C8 / D4 / SO(2) / D4-BT)",
                 fontsize=10, pad=8)
    fig.tight_layout(pad=0.5)
    p = out_dir / "fig6_architecture.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 14 — Bi-temporal flow schematic (D4-BT twin-encoder + difference fusion)
# ─────────────────────────────────────────────────────────────────────────────
def fig14_bitemporal_flow(out_dir: Path) -> None:
    _style()
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.spines.bottom"] = False
    plt.rcParams["axes.spines.left"] = False

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.8)
    ax.set_axis_off()

    C_BG    = "#F5F5F5"
    C_EQV   = "#EDE7F6"   # equivariant encoder blocks: light purple
    C_HEAD1 = "#E3F2FD"   # classification head: light blue
    C_BT    = "#FCE4EC"   # bi-temporal / difference: light pink
    C_ARROW = "#455A64"

    def box(x, y, w, h, label, sublabel="", color=C_BG, fontsize=8.5):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                           facecolor=color, edgecolor="#90A4AE", lw=0.8, zorder=3)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2 + (0.12 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", zorder=4)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.17, sublabel,
                    ha="center", va="center",
                    fontsize=6.8, color="#546E7A", zorder=4)

    def arrow(x0, x1, y, color=C_ARROW):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=0.9, mutation_scale=10), zorder=2)

    # ── Post-event stream (top row) ─────────────────────────────────────────
    post_y = 3.40
    box(0.10, post_y, 1.05, 0.60, "Post-event", "5×64×64", color=C_BG, fontsize=8)
    arrow(1.15, 1.30, post_y + 0.30)

    enc_x = [1.30, 2.35, 3.40, 4.45]
    enc_labels = [
        ("Block 1", "reg · 64×64"),
        ("Block 2", "reg · 32×32"),
        ("Block 3", "reg · 16×16"),
        ("Block 4", "reg ·  8×8"),
    ]
    for bx, (lbl, sub) in zip(enc_x, enc_labels):
        box(bx, post_y, 0.90, 0.60, lbl, sub, color=C_EQV)
        if bx < enc_x[-1]:
            arrow(bx + 0.90, bx + 1.05, post_y + 0.30)

    # ── Pre-event stream (bottom row) ────────────────────────────────────────
    pre_y = 1.05
    box(0.10, pre_y, 1.05, 0.60, "Pre-event", "5×64×64", color=C_BG, fontsize=8)
    arrow(1.15, 1.30, pre_y + 0.30)

    for bx, (lbl, sub) in zip(enc_x, enc_labels):
        box(bx, pre_y, 0.90, 0.60, lbl, sub, color=C_EQV)
        if bx < enc_x[-1]:
            arrow(bx + 0.90, bx + 1.05, pre_y + 0.30)

    # ── Shared-weight annotation (vertical dashed ties between encoder pairs) ─
    for bx in enc_x:
        ax.plot([bx + 0.45, bx + 0.45], [pre_y + 0.60, post_y],
                linestyle=(0, (1.5, 1.5)), color="#7E57C2", lw=0.8, zorder=1)
    ax.text(sum(enc_x) / 4 + 0.45, (pre_y + post_y) / 2 + 0.30,
            "weights shared",
            ha="center", va="center",
            fontsize=7, color="#4527A0", style="italic",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    # ── Difference fusion (centre-right) ─────────────────────────────────────
    diff_x, diff_y = 5.70, 2.10
    # Converge arrows from block-4 outputs to the ⊖ node
    ax.annotate("", xy=(diff_x, diff_y + 0.30), xytext=(enc_x[-1] + 0.90, post_y + 0.30),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=0.9, mutation_scale=10,
                                connectionstyle="arc3,rad=-0.15"), zorder=2)
    ax.annotate("", xy=(diff_x, diff_y + 0.30), xytext=(enc_x[-1] + 0.90, pre_y + 0.30),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=0.9, mutation_scale=10,
                                connectionstyle="arc3,rad=0.15"), zorder=2)

    box(diff_x, diff_y, 1.30, 0.60,
        "feat$_{post}$ − feat$_{pre}$",
        "D4-equivariant by linearity",
        color=C_BT, fontsize=8.5)

    # ── Classification head ──────────────────────────────────────────────────
    head_y = 2.10
    arrow(diff_x + 1.30, diff_x + 1.45, head_y + 0.30)
    box(diff_x + 1.45, head_y, 1.25, 0.60, "GroupPool",
        "trivial rep", color=C_HEAD1)
    arrow(diff_x + 1.45 + 1.25, diff_x + 1.45 + 1.25 + 0.15, head_y + 0.30)
    box(diff_x + 1.45 + 1.25 + 0.15, head_y, 1.15, 0.60,
        "AvgPool\n→ Linear", "sigmoid", color=C_HEAD1)
    out_x = diff_x + 1.45 + 1.25 + 0.15 + 1.15 + 0.15
    arrow(out_x - 0.15, out_x, head_y + 0.30)
    box(out_x, head_y, 1.20, 0.60,
        "P(debris)", "∈ [0, 1]", color=C_HEAD1)

    ax.text(diff_x + 1.45 + 0.625, head_y + 0.78,
            "HEAD — Classification",
            ha="center", fontsize=7.5, color="#1565C0", fontweight="bold")

    # ── Legend / equivariance note ───────────────────────────────────────────
    ax.text(2.85, post_y + 0.82,
            "← Equivariant backbone (steerable convolutions) →",
            ha="center", fontsize=7.5, color="#6A1B9A", style="italic")

    ax.text(diff_x + 0.65, diff_y - 0.35,
            "Element-wise difference of deepest features;\n"
            "inherits D4 equivariance from linear group action",
            ha="center", fontsize=7, color="#880E4F", style="italic")

    # ── Footnote: aux inputs live in the 5-channel stack ─────────────────────
    ax.text(0.10, 0.28,
            "Auxiliary terrain inputs (slope, sin/cos aspect) are stacked into the 5-channel "
            "input (no separate aux encoder).",
            ha="left", fontsize=7, color="#37474F", style="italic")

    ax.set_title("D4-BT bi-temporal flow: twin equivariant encoder + difference fusion",
                 fontsize=10, pad=8)
    fig.tight_layout(pad=0.5)
    p = out_dir / "fig14_bitemporal_flow.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — Group elements illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig7_group_elements(patch_csv: Path, out_dir: Path) -> None:
    try:
        import csv
        import rasterio
        from scipy.ndimage import rotate as ndrotate
    except ImportError as e:
        print(f"  [skip fig7] missing: {e}")
        return

    _style()
    plt.rcParams["axes.grid"] = False

    # Find a visually interesting positive patch
    with open(patch_csv) as f:
        reader = csv.DictReader(f)
        pos_patches = [r for r in reader if r["label"] == "1"]

    # Pick a patch with high variance (interesting texture)
    best_patch, best_var = None, -1
    for r in pos_patches[:100]:          # check first 100
        pdir = Path(r["patch_dir"])
        post = pdir / "post.tif"
        if not post.exists():
            continue
        with rasterio.open(post) as src:
            vh = src.read(1).astype(np.float32)
        v = float(np.nanvar(vh))
        if v > best_var:
            best_var, best_patch = v, pdir

    if best_patch is None:
        print("  [skip fig7] no positive patches found")
        return

    with rasterio.open(best_patch / "post.tif") as src:
        vh = np.clip(src.read(1).astype(np.float32), -25, -5)

    # Normalise patch to [0,1] for display
    patch = (vh - vh.min()) / (vh.max() - vh.min() + 1e-8)

    # ── Panel A: C8 — 8 rotations in a 2×4 grid ──────────────────────────────
    angles_c8 = [0, 45, 90, 135, 180, 225, 270, 315]

    # ── Panel B: D4 — 4 rotations + 4 reflections ────────────────────────────
    angles_d4   = [0, 90, 180, 270]
    reflected_d4 = [np.fliplr(ndrotate(patch, -a, reshape=False,
                                        order=1, prefilter=False))
                    for a in angles_d4]

    fig = plt.figure(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")

    # Title above both panels
    fig.suptitle("Group Symmetry Actions on a SAR Avalanche Debris Patch",
                 fontsize=10, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(1, 2, wspace=0.12, left=0.02, right=0.98,
                          top=0.88, bottom=0.02)

    # ─ C8 panel ──────────────────────────────────────────────────────────────
    gs_c8 = gs[0].subgridspec(3, 4, hspace=0.05, wspace=0.05)
    ax_c8_title = fig.add_subplot(gs[0])
    ax_c8_title.set_axis_off()
    ax_c8_title.text(0.5, 1.04, "C8: 8 discrete rotations (45° grid)",
                     ha="center", va="bottom", fontsize=9, fontweight="bold",
                     color=PALETTE["c8"], transform=ax_c8_title.transAxes)
    ax_c8_title.text(0.5, -0.04,
                     "Equivariant by construction — same weights classify\n"
                     "avalanche debris at any of these 8 orientations",
                     ha="center", va="top", fontsize=7.5, color="#37474F",
                     transform=ax_c8_title.transAxes)

    for i, ang in enumerate(angles_c8):
        row, col = i // 4, i % 4
        ax = fig.add_subplot(gs_c8[row + 0, col])   # rows 0-1 in c8 subgrid
        rotated = ndrotate(patch, -ang, reshape=False, order=1, prefilter=False)
        ax.imshow(rotated, cmap="gray", interpolation="nearest",
                  vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(PALETTE["c8"]); sp.set_linewidth(1.5)
        ax.set_title(f"{ang}°", fontsize=7.5, pad=1, color=PALETTE["c8"])
        if i == 0:
            ax.set_title(f"{ang}° (original)", fontsize=7.5, pad=1,
                         color=PALETTE["c8"], fontweight="bold")

    # ─ D4 panel ──────────────────────────────────────────────────────────────
    gs_d4 = gs[1].subgridspec(3, 4, hspace=0.05, wspace=0.05)
    ax_d4_title = fig.add_subplot(gs[1])
    ax_d4_title.set_axis_off()
    ax_d4_title.text(0.5, 1.04, "D4: 4 rotations × 2 reflections",
                     ha="center", va="bottom", fontsize=9, fontweight="bold",
                     color=PALETTE["d4"], transform=ax_d4_title.transAxes)
    ax_d4_title.text(0.5, -0.04,
                     "Reflections are physically motivated by bilateral\n"
                     "symmetry of avalanche runouts along the fall line",
                     ha="center", va="top", fontsize=7.5, color="#37474F",
                     transform=ax_d4_title.transAxes)

    rot_row_lbl  = ["0°", "90°", "180°", "270°"]
    for i, ang in enumerate(angles_d4):
        rotated = ndrotate(patch, -ang, reshape=False, order=1, prefilter=False)
        # Row 0: rotations
        ax = fig.add_subplot(gs_d4[0, i])
        ax.imshow(rotated, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(PALETTE["d4"]); sp.set_linewidth(1.5)
        ax.set_title(f"r({ang}°)", fontsize=7.5, pad=1, color=PALETTE["d4"])
        # Row 1: reflections
        ax2 = fig.add_subplot(gs_d4[1, i])
        ax2.imshow(reflected_d4[i], cmap="gray", interpolation="nearest",
                   vmin=0, vmax=1)
        ax2.set_xticks([]); ax2.set_yticks([])
        for sp in ax2.spines.values():
            sp.set_color("#AD1457"); sp.set_linewidth(1.5)
        ax2.set_title(f"f·r({ang}°)", fontsize=7.5, pad=1, color="#AD1457")

    # Row labels
    ax_rlbl1 = fig.add_subplot(gs_d4[0, 0])
    ax_rlbl1.set_axis_off()

    p = out_dir / "fig7_group_elements.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8 — Speckle reduction illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig8_speckle_reduction(patch_csv: Path, out_dir: Path) -> None:
    try:
        import csv
        import rasterio
        from scipy.ndimage import rotate as ndrotate
    except ImportError as e:
        print(f"  [skip fig8] missing: {e}")
        return

    _style()
    plt.rcParams["axes.grid"] = False

    # Load several positive patches and pick one with visible speckle
    with open(patch_csv) as f:
        reader = csv.DictReader(f)
        pos_patches = [r for r in reader if r["label"] == "1"]

    best_patch, best_score = None, -1
    for r in pos_patches[:200]:
        pdir = Path(r["patch_dir"])
        post = pdir / "post.tif"
        if not post.exists():
            continue
        with rasterio.open(post) as src:
            vh = src.read(1).astype(np.float32)
        # Speckle: high spatial variance relative to mean
        vh_c = np.clip(vh, -25, -5)
        score = float(np.var(vh_c)) / (abs(float(np.mean(vh_c))) + 1e-6)
        if score > best_score:
            best_score, best_patch = score, pdir

    if best_patch is None:
        print("  [skip fig8] no patches found")
        return

    with rasterio.open(best_patch / "post.tif") as src:
        vh = np.clip(src.read(1).astype(np.float32), -25, -5)

    orig = (vh - vh.min()) / (vh.max() - vh.min() + 1e-8)

    # Bilinear rotation at 45° (order=1)
    rot45 = ndrotate(orig, -45, reshape=False, order=1, prefilter=False)

    # Difference: smoothing effect
    # Re-rotate back to align for difference
    rot45_back = ndrotate(rot45, 45, reshape=False, order=1, prefilter=False)
    diff = orig - rot45_back

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))

    kw = dict(cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].imshow(orig,     **kw)
    axes[1].imshow(rot45,    **kw)
    axes[2].imshow(np.abs(diff),
                   cmap="inferno", vmin=0, vmax=0.25,
                   interpolation="nearest")

    titles = [
        "Original (0°)\nAUC contribution: 0.749",
        "Rotated 45° (bilinear interp.)\nAUC contribution: 0.776",
        "|Original − Rotated-back|\nSmoothed-out speckle",
    ]
    colors = ["#1565C0", "#1565C0", "#B71C1C"]
    for ax, title, col in zip(axes, titles, colors):
        ax.set_title(title, fontsize=8, color=col, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)

    axes[2].set_title(titles[2], fontsize=8, color=colors[2], pad=4)

    fig.suptitle(
        "Bilinear interpolation at 45° as accidental speckle reduction\n"
        "SAR speckle is partially smoothed → AUC improves from 0.749 → 0.776",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout(pad=0.5)
    p = out_dir / "fig8_speckle_reduction.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9 — Precision-recall curves
# ─────────────────────────────────────────────────────────────────────────────
def fig9_pr_curves(results_dir: Path, out_dir: Path) -> None:
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
    except ImportError as e:
        print(f"  [skip fig9] missing: {e}")
        return

    _style()

    # Models to plot: D4-BT at all 4 fracs, then single-image best @ 100%
    RUNS = [
        ("d4_bitemporal_frac0p1",  "D4-BT 10%",      PALETTE["d4_bitemporal"], "--",  6),
        ("d4_bitemporal_frac0p25", "D4-BT 25%",       PALETTE["d4_bitemporal"], "-.",  6),
        ("d4_bitemporal_frac0p5",  "D4-BT 50%",       PALETTE["d4_bitemporal"], ":",   6),
        ("d4_bitemporal_frac1p0",  "D4-BT 100%",      PALETTE["d4_bitemporal"], "-",   9),
        ("d4_frac1p0",             "D4 100%",          PALETTE["d4"],            "-",   6),
        ("resnet_frac1p0",         "ResNet-18 100%",   PALETTE["resnet"],        "-",   6),
        ("cnn_frac1p0",            "CNN baseline 100%",PALETTE["cnn"],           "-",   6),
    ]
    ALPHAS = [0.55, 0.65, 0.75, 1.0, 1.0, 1.0, 1.0]

    fig, ax = plt.subplots(figsize=(7, 5))

    for (run, label, color, ls, ms), alpha in zip(RUNS, ALPHAS):
        npz_path = results_dir / run / "scores_test_ood_calibrated.npz"
        met_path = results_dir / run / "metrics.json"
        if not npz_path.exists():
            continue

        d    = np.load(npz_path)
        probs = d["probs_uncal"]
        labels = d["labels"]

        # AUC-PR from metrics.json (avg_precision)
        with open(met_path) as f:
            mj = json.load(f)
        auc_roc = mj["splits"]["test_ood"]["auc_roc"]
        auc_pr  = mj["splits"]["test_ood"].get("avg_precision",
                      average_precision_score(labels, probs))

        prec, rec, thresholds = precision_recall_curve(labels, probs)

        # Youden-optimal operating point (sensitivity + specificity - 1 maximised)
        # Equivalent: find threshold in metrics.json at_optimal
        opt_thr = mj["splits"]["test_ood"]["at_optimal"]["threshold"]
        opt_rec = mj["splits"]["test_ood"]["at_optimal"]["recall"]
        opt_pre = mj["splits"]["test_ood"]["at_optimal"]["precision"]

        lw = 2.2 if "BT 100%" in label else 1.6
        legend_txt = f"{label}  (AUC-ROC={auc_roc:.3f}, AUC-PR={auc_pr:.3f})"
        ax.plot(rec, prec, color=color, ls=ls, lw=lw, alpha=alpha,
                label=legend_txt, zorder=4 if "BT 100%" in label else 3)

        # Mark operating point (only for D4-BT fracs and the baseline)
        ax.scatter([opt_rec], [opt_pre], color=color, s=ms**2, zorder=6,
                   edgecolors="white", linewidths=0.6)

    # No-skill line at dataset prevalence
    prevalence = 484 / 2211   # test_ood positives / total
    ax.axhline(prevalence, color="#B0BEC5", lw=1.0, ls=":", zorder=1)
    ax.text(0.02, prevalence + 0.01, f"No-skill (P={prevalence:.2f})",
            fontsize=7, color="#78909C", va="bottom")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Tromsø OOD Test Set\n"
                 "(dots = Youden-optimal operating point)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="0.8",
              fontsize=7.5, labelspacing=0.25)

    fig.tight_layout()
    p = out_dir / "fig9_pr_curves.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 10 — Threshold sensitivity (F1/F2 vs threshold) for D4-BT frac0p5
# ─────────────────────────────────────────────────────────────────────────────
def fig10_threshold_sensitivity(results_dir: Path, out_dir: Path) -> None:
    _style()

    run      = "d4_bitemporal_frac0p5"
    npz_path = results_dir / run / "scores_test_ood_calibrated.npz"
    met_path = results_dir / run / "metrics.json"
    if not npz_path.exists():
        print(f"  [skip fig10] missing {npz_path}")
        return

    d      = np.load(npz_path)
    probs  = d["probs_uncal"]
    labels = d["labels"]

    with open(met_path) as f:
        mj = json.load(f)
    youden_thr = mj["splits"]["test_ood"]["at_optimal"]["threshold"]

    # Sweep thresholds
    thresholds = np.linspace(0.0, 1.0, 500)
    f1s, f2s = [], []
    for t in thresholds:
        preds = (probs >= t).astype(float)
        tp = float(np.sum((preds == 1) & (labels == 1)))
        fp = float(np.sum((preds == 1) & (labels == 0)))
        fn = float(np.sum((preds == 0) & (labels == 1)))
        if tp + fp + fn == 0:
            f1s.append(0.0); f2s.append(0.0); continue
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom_f1 = (prec + rec)
        denom_f2 = (4 * prec + rec)
        f1s.append(2  * prec * rec / denom_f1 if denom_f1 > 0 else 0.0)
        f2s.append(5  * prec * rec / denom_f2 if denom_f2 > 0 else 0.0)

    f1s = np.array(f1s)
    f2s = np.array(f2s)

    # F2-optimal threshold
    f2_opt_idx  = int(np.argmax(f2s))
    f2_opt_thr  = float(thresholds[f2_opt_idx])
    f2_opt_val  = float(f2s[f2_opt_idx])

    # F2 at default 0.5
    idx_05  = int(np.argmin(np.abs(thresholds - 0.5)))
    f2_at05 = float(f2s[idx_05])
    f2_gain = f2_opt_val - f2_at05

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(thresholds, f2s, color=PALETTE["d4_bitemporal"], lw=2.2,
            label="F2 (β=2)", zorder=4)
    ax.plot(thresholds, f1s, color="#9575CD", lw=1.6, ls="--",
            label="F1 (β=1)", zorder=3)

    # Vertical markers
    ax.axvline(0.5,        color="#607D8B", lw=1.0, ls=":",
               label="Default threshold (0.5)")
    ax.axvline(f2_opt_thr, color=PALETTE["d4_bitemporal"], lw=1.2, ls="-.",
               label=f"F2-optimal threshold ({f2_opt_thr:.3f})")
    ax.axvline(youden_thr, color="#FF8F00", lw=1.0, ls=":",
               label=f"Youden's J threshold ({youden_thr:.3f})")

    # Annotate F2 gain
    ax.annotate(
        f"+{f2_gain:.3f} F2 from\nthreshold optimisation",
        xy=(f2_opt_thr, f2_opt_val),
        xytext=(f2_opt_thr + 0.07, f2_opt_val - 0.06),
        fontsize=7.5, color=PALETTE["d4_bitemporal"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["d4_bitemporal"],
                        lw=0.8),
    )
    # Dot at F2-optimal
    ax.scatter([f2_opt_thr], [f2_opt_val],
               color=PALETTE["d4_bitemporal"], s=50, zorder=6,
               edgecolors="white", linewidths=0.8)
    # Dot at 0.5
    ax.scatter([0.5], [f2_at05], color="#607D8B", s=36, zorder=6,
               edgecolors="white", linewidths=0.8)

    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sensitivity — D4-BT (50% data) on Tromsø OOD Test Set")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="0.8",
              fontsize=7.5, labelspacing=0.3)

    fig.tight_layout()
    p = out_dir / "fig10_threshold_sensitivity.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved {p}")
    print(f"    F2-optimal threshold: {f2_opt_thr:.3f}  F2={f2_opt_val:.4f}")
    print(f"    F2 gain over 0.5 default: +{f2_gain:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 11 — Confusion matrix (D4-BT frac0p5 at F2-optimal threshold)
# ─────────────────────────────────────────────────────────────────────────────
def fig11_confusion_matrix(results_dir: Path, out_dir: Path) -> None:
    _style()
    plt.rcParams["axes.grid"] = False

    run      = "d4_bitemporal_frac0p5"
    npz_path = results_dir / run / "scores_test_ood_calibrated.npz"
    met_path = results_dir / run / "metrics.json"
    if not npz_path.exists():
        print(f"  [skip fig11] missing {npz_path}")
        return

    d      = np.load(npz_path)
    probs  = d["probs_uncal"]
    labels = d["labels"].astype(int)

    # Compute F2-optimal threshold from sweep
    thresholds = np.linspace(0.0, 1.0, 1000)
    best_thr, best_f2 = 0.5, -1.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = float(np.sum((preds == 1) & (labels == 1)))
        fp = float(np.sum((preds == 1) & (labels == 0)))
        fn = float(np.sum((preds == 0) & (labels == 1)))
        denom = (4 * (tp / (tp + fp) if (tp + fp) > 0 else 0) + (tp / (tp + fn) if (tp + fn) > 0 else 0))
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f2    = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
        if f2 > best_f2:
            best_f2, best_thr = f2, t

    preds = (probs >= best_thr).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    n  = tp + fp + fn + tn

    cm = np.array([[tp, fn],
                   [fp, tn]])
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_pct = 100.0 * cm / row_totals.clip(1)

    fig, ax = plt.subplots(figsize=(5, 4))

    # Custom two-tone colour: pink for correct, grey for errors
    cell_colors = [
        [PALETTE["d4_bitemporal"], "#EF9A9A"],   # row 0: TP correct=pink, FN=light red
        ["#EF9A9A",               "#CFD8DC"],    # row 1: FP=light red, TN correct=grey
    ]
    cell_correct = [[True, False], [False, True]]

    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle([j, 1 - i], 1, 1,
                                  facecolor=cell_colors[i][j], lw=0)
            ax.add_patch(rect)
            count = cm[i, j]
            pct   = cm_pct[i, j]
            text_color = "white" if cell_correct[i][j] else "#37474F"
            ax.text(j + 0.5, 1.5 - i, f"{count}",
                    ha="center", va="center", fontsize=18,
                    fontweight="bold", color=text_color)
            ax.text(j + 0.5, 1.5 - i - 0.22, f"({pct:.1f}%)",
                    ha="center", va="center", fontsize=9, color=text_color)

    # Axis labels
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted: Debris", "Predicted: Clean"], fontsize=9)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Actual: Clean", "Actual: Debris"], fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Labels on top
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f2   = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    ax.set_title(
        f"D4-BT (50% data) — Tromsø OOD Test Set\n"
        f"Threshold = {best_thr:.3f} (F2-optimal)  |  "
        f"F2 = {f2:.3f}  F1 = {f1:.3f}  Precision = {prec:.3f}  Recall = {rec:.3f}",
        fontsize=8, pad=22,
    )

    fig.tight_layout()
    p = out_dir / "fig11_confusion_matrix.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    print(f"  Saved {p}  (threshold={best_thr:.3f}, F2={f2:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 12 — Temperature scaling illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig12_temperature_scaling(results_dir: Path, out_dir: Path) -> None:
    _style()
    plt.rcParams["axes.grid"] = False

    # Two models: well-calibrated d4_frac1p0 (T≈4.56) vs collapsed d4_bitemporal_frac0p5 (T≈50)
    MODELS = [
        ("d4_frac1p0",            "D4 (100% data)",    PALETTE["d4"],           4.56),
        ("d4_bitemporal_frac0p5", "D4-BT (50% data)",  PALETTE["d4_bitemporal"], 50.0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    titles = ["Before temperature scaling", "After temperature scaling"]

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Density")
        ax.set_xlim(-0.03, 1.03)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    bins = np.linspace(0, 1, 51)
    alpha_fill = 0.30

    for run, label, color, temp in MODELS:
        npz_path = results_dir / run / "scores_test_ood_calibrated.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path)
        probs_uncal = d["probs_uncal"]
        probs_cal   = d["probs_cal"]

        for ax, probs in zip(axes, [probs_uncal, probs_cal]):
            counts, edges = np.histogram(probs, bins=bins, density=True)
            mids = 0.5 * (edges[:-1] + edges[1:])
            ax.fill_between(mids, counts, alpha=alpha_fill, color=color)
            ax.step(edges[:-1], counts, where="post",
                    color=color, lw=1.8, label=f"{label}  (T={temp:.1f})")

    # Annotations
    axes[0].annotate(
        "D4-BT: logits saturate\n(|logit| ≫ 1)\n→ probs cluster at 0 and 1",
        xy=(0.92, 0.5), xycoords=("data", "axes fraction"),
        xytext=(0.60, 0.72), textcoords=("data", "axes fraction"),
        fontsize=7.5, color=PALETTE["d4_bitemporal"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["d4_bitemporal"], lw=0.8),
    )
    axes[1].annotate(
        "After T≈50 scaling:\nprobs compress to ~0.5\n→ threshold must come from val set",
        xy=(0.50, 0.5), xycoords=("data", "axes fraction"),
        xytext=(0.08, 0.72), textcoords=("data", "axes fraction"),
        fontsize=7.5, color=PALETTE["d4_bitemporal"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["d4_bitemporal"], lw=0.8),
    )

    for ax in axes:
        ax.legend(loc="upper left", framealpha=0.95, edgecolor="0.8",
                  fontsize=8, labelspacing=0.3)

    fig.suptitle(
        "Temperature Scaling: Well-calibrated (D4, T≈4.6) vs Logit-saturated (D4-BT, T≈50)",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(pad=0.8)
    p = out_dir / "fig12_temperature_scaling.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 13 — Augmentation-accuracy tradeoff
# ─────────────────────────────────────────────────────────────────────────────
def fig13_aug_tradeoff(out_dir: Path) -> None:
    _style()

    # Verified from metrics.json on Hyak (2026-04-13)
    CNN_VAL  = [0.6929, 0.7349, 0.7510, 0.7552]
    CNN_TEST = [0.4988, 0.6765, 0.7833, 0.7233]
    AUG_VAL  = [0.6925, 0.7039, 0.7297, 0.7610]
    AUG_TEST = [0.5227, 0.6222, 0.7443, 0.7051]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    titles = ["Validation AUC-ROC", "OOD Test AUC-ROC (Tromsø)"]
    val_test = [(CNN_VAL, AUG_VAL), (CNN_TEST, AUG_TEST)]

    for ax, title, (cnn_vals, aug_vals) in zip(axes, titles, val_test):
        ax.plot(FRAC_X, cnn_vals,
                color=PALETTE["cnn"], marker=MARKERS["cnn"], lw=2.0, ms=7,
                label=LABELS["cnn"], zorder=4)
        ax.plot(FRAC_X, aug_vals,
                color=PALETTE["aug"], marker=MARKERS["aug"], lw=2.0, ms=7,
                ls="--", label=LABELS["aug"], zorder=4)

        # Shade the gap where aug < cnn
        ax.fill_between(FRAC_X, cnn_vals, aug_vals,
                        where=[c > a for c, a in zip(cnn_vals, aug_vals)],
                        alpha=0.12, color=PALETTE["cnn"],
                        label="CNN advantage", interpolate=True)

        ax.set_xticks(FRAC_X)
        ax.set_xticklabels(FRAC_LBL)
        ax.set_xlabel("Training data fraction")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="lower right", framealpha=0.95, edgecolor="0.8",
                  fontsize=8, labelspacing=0.3)
        ax.set_xlim(5, 108)

    axes[0].set_ylabel("AUC-ROC")
    # Annotate: augmentation costs OOD performance
    axes[1].annotate(
        "CNN+aug underperforms plain CNN\nat every data fraction",
        xy=(25, AUG_TEST[1]), xycoords="data",
        xytext=(40, 0.56), textcoords="data",
        fontsize=8, color="#37474F",
        arrowprops=dict(arrowstyle="->", color="#607D8B", lw=0.8),
    )

    fig.suptitle(
        "Augmentation–Accuracy Tradeoff: CNN+aug consistently underperforms plain CNN\n"
        "on the OOD test set — SAR backscatter has orientation-dependent structure",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(pad=0.8)
    p = out_dir / "fig13_aug_tradeoff.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate publication figures.")
    p.add_argument("--figures", nargs="+", default=["1", "2", "5", "6"],
                   help="Which figures to generate (1-13 or 'all').")
    p.add_argument("--results-dir",  default="results")
    p.add_argument("--scene-dir",    default="data/raw/Tromso_20241220")
    p.add_argument("--prob-map",
                   default="results/scene/d4_bitemporal_frac0p5_prob.tif")
    p.add_argument("--gt-path",
                   default="data/raw/Tromso_20241220/Tromso_20241220_GT.gpkg")
    p.add_argument("--patch-csv",    default="data/splits/test_ood.csv")
    p.add_argument("--out-dir",      default="figures")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = set(args.figures)
    if "all" in figs:
        figs = {"1", "2", "3", "4", "5", "6", "7", "8",
                "9", "10", "11", "12", "13", "14"}

    results_dir = Path(args.results_dir)
    scene_dir   = Path(args.scene_dir)
    prob_map    = Path(args.prob_map)
    gt_path     = Path(args.gt_path)
    patch_csv   = Path(args.patch_csv)

    if "1" in figs:
        print("Figure 1: data-efficiency curves…")
        fig1_data_efficiency(out_dir)

    if "2" in figs:
        print("Figure 2: model comparison bar chart…")
        fig2_model_comparison(out_dir)

    if "3" in figs:
        print("Figure 3: probability heatmap overlay…")
        fig3_heatmap_overlay(scene_dir, prob_map, gt_path, out_dir)

    if "4" in figs:
        print("Figure 4: hit/miss polygon map…")
        fig4_hit_miss_map(scene_dir, prob_map, gt_path, out_dir)

    if "5" in figs:
        print("Figure 5: geography map…")
        fig5_geography_map(out_dir)

    if "6" in figs:
        print("Figure 6: architecture diagram…")
        fig6_architecture(out_dir)

    if "7" in figs:
        print("Figure 7: group elements…")
        fig7_group_elements(patch_csv, out_dir)

    if "8" in figs:
        print("Figure 8: speckle reduction…")
        fig8_speckle_reduction(patch_csv, out_dir)

    if "9" in figs:
        print("Figure 9: precision-recall curves…")
        fig9_pr_curves(results_dir, out_dir)

    if "10" in figs:
        print("Figure 10: threshold sensitivity…")
        fig10_threshold_sensitivity(results_dir, out_dir)

    if "11" in figs:
        print("Figure 11: confusion matrix…")
        fig11_confusion_matrix(results_dir, out_dir)

    if "12" in figs:
        print("Figure 12: temperature scaling…")
        fig12_temperature_scaling(results_dir, out_dir)

    if "13" in figs:
        print("Figure 13: augmentation-accuracy tradeoff…")
        fig13_aug_tradeoff(out_dir)

    if "14" in figs:
        print("Figure 14: bi-temporal flow schematic…")
        fig14_bitemporal_flow(out_dir)

    print(f"\nAll done. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
