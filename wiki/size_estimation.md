# Size Estimation — Phase 2

**Deposit area from segmentation mask → D-scale proxy. Supplementary experiment: SeNorge snowpack fusion improves D-scale classification on Tromsø.**

---

## D-scale reference

EAWS destructive potential scale. No official cubic-metre thresholds; empirical area ranges from literature:

| Class | Approx. area range | Tromsø GT count |
|-------|--------------------|----------------|
| D1 | ~29–437 m² | 5 |
| D2 | ~612–4,978 m² | 25 |
| D3 | ~5,000–50,000 m² | 71 |
| D4 | >50,000 m² | 16 |

> ⚠ OPEN: confirm area thresholds from Tromsø GT data. Fit empirical decision boundaries from the 117 polygons rather than using literature values, since literature ranges are approximate and dataset-specific.

---

## Deposit area pipeline

```
Predicted pixel mask (64×64, stride=16)
  → stitch patches → full-scene binary mask
  → connected components (scipy.ndimage or skimage)
  → georeferenced polygons (rasterio + shapely)
  → polygon.area × (10m)² = area_m²
  → log-linear classifier → D-scale {1,2,3,4}
```

- Pixel size: 10m × 10m = 100 m² per pixel
- Minimum detectable area: ~1 pixel (100 m²) → D1 range, but below SAR detection floor
- Practical floor: ~400 m² (4 pixels) → lower D2 range

---

## SAR detection floor analysis (from Phase 1 research)

- D2 deposits: +4.1 dB VH mean backscatter change (above speckle noise)
- D1 deposits: below speckle floor at 10m GRD — undetectable regardless of algorithm
- Fix for D2: 75% inference overlap (stride=16) concentrates signal; reduces spatial dilution

> ✓ DECIDED: D1 polygons will be reported separately in per-class results and likely excluded from size classifier training (n=5, likely below detection floor).

---

## SeNorge snowpack fusion (supplementary)

**Hypothesis**: SAR deposit area + fresh snow load (ΔSD₇₂) improves D-scale classification vs area alone,
because a small deposit on deep fresh snow represents a larger event than the same area on settled snowpack.

### Features for fusion model

| Feature | Source | Physical meaning |
|---------|--------|-----------------|
| log(area_m²) | Segmentation mask | Deposit size proxy |
| ΔSD₇₂ (m) | SeNorge 1km | Fresh snow load at deposit location |
| bulk density (kg/m³) | SeNorge 1km | Snow type (new vs settled) |

### Model

Simple log-linear or random forest classifier — not deep learning.
Input: 3 features per polygon. Output: D-scale class {2, 3, 4} (D1 excluded).
Train/test: leave-one-out or stratified split on the 112 Tromsø polygons with D-scale labels.

### Why this is novel

No published paper has combined SAR-derived deposit geometry with operational snowpack model output for avalanche size estimation. Gap confirmed via Phase 1 literature search.

### Caveats

- Single scene (Tromsø_20241220) — cannot generalize
- SeNorge at 1km vs Sentinel-1 at 10m — spatial mismatch
- Proof of concept only; explicitly labeled as such in paper

---

## What this does NOT do

- Does not predict D-scale from SAR backscatter amplitude alone
- Does not use CROCUS snowpack model (French; less relevant for Norway)
- Does not attempt volume estimation (would require DEM differencing)
