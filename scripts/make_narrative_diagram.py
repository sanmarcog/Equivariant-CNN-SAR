"""
scripts/make_narrative_diagram.py

Render the 20-minute Phase 1 narrative as a single readable slide-ready PNG.
Four acts: Setup → Experimental arc → Problems/fixes → Value.

Output: figures/fig_narrative.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


C_BG      = "#FAFAFA"
C_SETUP   = "#E8EAF6"   # indigo tint
C_EXP     = "#EDE7F6"   # equivariant purple
C_FINDING = "#FCE4EC"   # pink
C_PROB    = "#FFEBEE"   # red tint
C_FIX     = "#E8F5E9"   # green tint
C_VALUE   = "#E3F2FD"   # blue tint
C_ARROW   = "#455A64"
C_BORDER  = "#90A4AE"


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.grid": False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.bottom": False,
        "axes.spines.left":   False,
    })


def box(ax, x, y, w, h, title, body="", color=C_BG,
        title_fs=10, body_fs=8.5, title_color="#263238", body_color="#37474F"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                       facecolor=color, edgecolor=C_BORDER, lw=0.8, zorder=3)
    ax.add_patch(b)
    if body:
        ax.text(x + w / 2, y + h - 0.28, title,
                ha="center", va="top",
                fontsize=title_fs, fontweight="bold",
                color=title_color, zorder=4)
        ax.text(x + w / 2, y + h - 0.62, body,
                ha="center", va="top",
                fontsize=body_fs, color=body_color, zorder=4,
                wrap=True)
    else:
        ax.text(x + w / 2, y + h / 2, title,
                ha="center", va="center",
                fontsize=title_fs, fontweight="bold",
                color=title_color, zorder=4)


def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.1, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=12), zorder=2)


def act_header(ax, x, y, w, text, color):
    ax.text(x + w / 2, y, text,
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            color="white",
            bbox=dict(facecolor=color, edgecolor="none",
                      boxstyle="round,pad=0.45"),
            zorder=5)


def main() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(16, 20))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 20)
    ax.set_axis_off()

    # ═════════════════════════════════════════════════════════════════════
    # ACT 1 — SETUP  (top strip, y ≈ 16.8 .. 19.5)
    # ═════════════════════════════════════════════════════════════════════
    y_act1 = 17.0
    act_header(ax, 0.5, 19.3, 15.0,
               "①  Setup & hypothesis   (≈4 min)", "#3F51B5")
    box(ax, 0.7, y_act1, 4.7, 2.0,
        "Problem",
        "SAR avalanche interpretation is\nmanual, slow, and unsuited to\nlarge-area monitoring.",
        color=C_SETUP)
    box(ax, 5.7, y_act1, 4.7, 2.0,
        "Hypothesis",
        "Debris has rotational + reflective\nsymmetries → equivariant CNNs\nshould need less labelled data\nthan standard CNNs.",
        color=C_SETUP)
    box(ax, 10.7, y_act1, 4.7, 2.0,
        "Dataset — AvalCD (Gatti et al. 2026)",
        "Train/Val: Livigno + Nuuk + Pish\nTest OOD: Tromsø\n(never seen in training)",
        color=C_SETUP)
    arrow(ax, 5.4, y_act1 + 1.0, 5.7, y_act1 + 1.0)
    arrow(ax, 10.4, y_act1 + 1.0, 10.7, y_act1 + 1.0)

    # Big arrow Act 1 → Act 2
    arrow(ax, 8.0, 16.7, 8.0, 16.0, lw=2.0)

    # ═════════════════════════════════════════════════════════════════════
    # ACT 2 — EXPERIMENTAL ARC  (y ≈ 10.2 .. 15.9)
    # ═════════════════════════════════════════════════════════════════════
    act_header(ax, 0.5, 15.7, 15.0,
               "②  Experimental arc   (≈9 min)", "#6A1B9A")

    # Row 1: baselines
    box(ax, 0.7, 13.7, 4.7, 1.7,
        "Train matched-parameter baselines",
        "~391K params · same schedule\nscripts: train.py · run_eval_all.py",
        color=C_EXP)
    box(ax, 5.7, 13.7, 4.7, 1.7,
        "Single-image comparison",
        "plain CNN · CNN+aug\nC8 · D4 · SO(2)",
        color=C_EXP)
    box(ax, 10.7, 13.7, 4.7, 1.7,
        "Finding 1 — equivariance helps",
        "D4 > C8 > SO(2) > CNN+aug\nCNN+aug trails plain CNN\n(augmentation injects speckle noise)",
        color=C_FINDING)
    arrow(ax, 5.4, 14.55, 5.7, 14.55)
    arrow(ax, 10.4, 14.55, 10.7, 14.55)

    # Down-arrow between rows
    arrow(ax, 13.05, 13.6, 13.05, 13.0, lw=1.4)
    arrow(ax, 13.05, 13.0, 10.4, 13.0, lw=1.4, style="-")
    arrow(ax, 10.4, 13.0, 10.4, 12.7, lw=1.4)

    # Row 2: extension & headline finding
    box(ax, 0.7, 10.6, 4.7, 2.0,
        "Extend architecture → D4-BT",
        "Shared encoder on pre & post\npatches + difference feature.\nEquivariance preserved by linearity.",
        color=C_EXP)
    box(ax, 5.7, 10.6, 4.7, 2.0,
        "★  Finding 2 — data efficiency",
        "D4-BT @ 10% data  (AUC 0.871)\nbeats every single-image model @ 100%.\nscript: plot_data_efficiency.py",
        color=C_FINDING, title_color="#AD1457")
    box(ax, 10.7, 10.6, 4.7, 2.0,
        "Full-scene inference on Tromsø",
        "Sliding 64×64 window, 50% overlap\n→ full probability raster\nscript: scene_inference.py",
        color=C_EXP)
    arrow(ax, 10.4, 12.7, 5.4, 12.7, lw=1.4, style="-")
    arrow(ax, 5.4, 12.7, 5.4, 12.6, lw=1.4)
    arrow(ax, 5.7, 11.6, 5.4, 11.6)
    # correct: left→middle
    # (simpler: drop the up arrows and just show left-to-right)
    arrow(ax, 5.4, 11.6, 5.7, 11.6, lw=1.4)
    arrow(ax, 10.4, 11.6, 10.7, 11.6, lw=1.4)

    # Big arrow Act 2 → Act 3
    arrow(ax, 8.0, 10.4, 8.0, 9.7, lw=2.0)

    # ═════════════════════════════════════════════════════════════════════
    # ACT 3 — PROBLEMS & FIXES  (y ≈ 3.4 .. 9.5)
    # ═════════════════════════════════════════════════════════════════════
    act_header(ax, 0.5, 9.4, 15.0,
               "③  Problems encountered & how we solved them   (≈4 min)",
               "#C62828")

    prob_x = 0.7
    fix_x  = 8.6
    prob_w = 7.2
    fix_w  = 6.7

    rows = [
        (7.3, "▲  How do we compare to Gatti?",
         "They report pixel-level F1 on\nthe full scene. We're a patch\nclassifier — metric mismatch.",
         "✓  Introduce polygon hit rate",
         "For each GT polygon, ask: did\nprob > t fire anywhere inside?\n99.1% hit rate + explicit caveats.\nscript: polygon_eval.py"),
        (5.6, "▲  Logit collapse",
         "Temperature scaling fits T ≈ 50.\nRaw probabilities saturate near\n0 and 1 → unusable for deployment.",
         "✓  Threshold calibration on val",
         "F2-optimal t = 0.862\n→ +0.065 F2 over default 0.5.\nscript: threshold_analysis.py"),
        (3.9, "▲  One 'genuine' miss (polygon 42)",
         "Confident model, real deposit,\nbut the model saw no signal. Bug?",
         "✓  Physical diagnosis",
         "LIA = 5° → SAR layover geometry.\nNot a model failure — a fundamental\nSAR acquisition limit."),
        (2.2, "▲  Rotation-robustness claim\nneeds empirical check",
         "Is equivariance actually preserved\nend-to-end on real SAR patches?",
         "✓  Rotate-and-test",
         "Predictions stable across\n8 × 45° rotations.\nscript: rotation_sensitivity.py"),
    ]
    for y, pt, pb, ft, fb in rows:
        box(ax, prob_x, y, prob_w, 1.5, pt, pb, color=C_PROB,
            title_color="#B71C1C")
        box(ax, fix_x, y, fix_w, 1.5, ft, fb, color=C_FIX,
            title_color="#1B5E20")
        arrow(ax, prob_x + prob_w, y + 0.75, fix_x, y + 0.75, lw=1.4)

    # Big arrow Act 3 → Act 4
    arrow(ax, 8.0, 2.1, 8.0, 1.5, lw=2.0)

    # ═════════════════════════════════════════════════════════════════════
    # ACT 4 — VALUE  (bottom strip, y ≈ -0.5 .. 1.3)
    # ═════════════════════════════════════════════════════════════════════
    act_header(ax, 0.5, 1.5, 15.0,
               "④  Value this repo adds   (≈3 min)", "#1565C0")

    vw = 2.9
    vh = 1.4
    vy = 0.05
    gap = 0.06

    values = [
        ("1st — First of its kind",
         "Equivariant CNNs applied to\nSAR avalanche detection."),
        ("10×  Data efficiency",
         "Deployable in regions with\nfew labelled avalanches."),
        ("A|B  Clean ablation",
         "Equivariance (+0.18) and\nbi-temporal (+0.13) are\nindependent and additive."),
        ("↻  Reproducible pipeline",
         "13 publication figures\nfrom one script:\nmake_figures.py"),
        ("→  Launchpad for Phase 2",
         "Pixel segmentation with same\nD4-BT backbone → direct\nF1/F2 comparison with Gatti."),
    ]
    for i, (t, b) in enumerate(values):
        vx = 0.7 + i * (vw + gap)
        box(ax, vx, vy, vw, vh, t, b, color=C_VALUE,
            title_color="#0D47A1", title_fs=9.5, body_fs=8.0)

    # ═════════════════════════════════════════════════════════════════════
    # Main title
    # ═════════════════════════════════════════════════════════════════════
    ax.text(8.0, 19.85,
            "Equivariant CNNs for SAR Avalanche Detection — Phase 1 Narrative",
            ha="center", fontsize=14, fontweight="bold", color="#1A237E")

    fig.tight_layout(pad=0.3)
    out = Path("figures/fig_narrative.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
