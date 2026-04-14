# Evaluation — Phase 2

**Primary metric: IoU-based polygon F1/F2 matching Gattimgatti 2026's protocol exactly.**

---

## Polygon matching protocol

1. Threshold predicted pixel mask at 0.5 → binary mask
2. Extract connected components → predicted polygons
3. For each GT polygon, find predicted polygon with highest IoU
4. **Match** if IoU ≥ threshold (see below); **miss** otherwise
5. Predicted polygons with no GT match = false positives

This is an object-level metric, not pixel-level. One GT polygon = one detection event.

> ⚠ OPEN: exact IoU threshold used by Gattimgatti. Their paper (arXiv:2603.22658) must be checked for the precise value. Typical values are 0.1, 0.25, or 0.5. This affects comparability directly — do not assume.

---

## F1 and F2 scores

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1 = 2 × P × R / (P + R)          — equal weight
F2 = 5 × P × R / (4P + R)         — recall weighted 2× more than precision
```

F2 is the primary reporting metric for hazard applications: missing a real avalanche
(FN) is more costly than a false alarm (FP).

**Gattimgatti 2026 targets**: F1=0.806, F2=0.841 on Tromsø held-out test set.

---

## Per-size-class breakdown

Report F1/F2 separately for D1, D2, D3, D4 on Tromsø GT.
This is the key diagnostic for the "D2 detection advantage" question.

---

## Secondary metrics

| Metric | Purpose |
|--------|---------|
| Pixel-level IoU | Standard segmentation diagnostic |
| Pixel-level precision/recall | Complement to polygon matching |
| Area estimation MAE (m²) | Size estimation experiment |
| D-scale classification accuracy | Supplementary snowpack fusion experiment |

---

## Statistical validity

- 3 seeds per configuration → report mean ± std
- Single-scene test set (Tromsø, 117 polygons) limits statistical power — report confidence intervals

> ⚠ OPEN: bootstrap confidence intervals vs analytical. Bootstrap preferred given non-normal distribution of polygon IoUs.

---

## What we are NOT reporting

- AUC-ROC — Phase 1 metric, not comparable to Gattimgatti
- Patch hit-rate — non-standard, Phase 1 only; described as such in Phase 1 README
