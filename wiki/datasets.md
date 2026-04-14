# Datasets — Phase 2

**Same AvalCD geographic split as Phase 1 + pixel-level GT rasters for segmentation + Tromsø D-scale annotations + SeNorge snowpack variables.**

---

## AvalCD — primary dataset

- Source: Zenodo (1.1 GB)
- 4 regions: Livigno (Italy), Nuuk (Greenland), Pish (Tajikistan), Tromsø (Norway)
- 7 acquisition events
- Phase 2 uses **pixel-level GT rasters** (not just patch labels) — all scenes have binary debris masks

### Geographic split (identical to Phase 1 and Gattimgatti 2026)

| Split | Scenes | Notes |
|-------|--------|-------|
| Train | Livigno_20240403, Livigno_20250129, Nuuk_20160413, Nuuk_20210411, Pish_20230221 | |
| Val | Livigno_20250318 | |
| Test OOD | Tromsø_20241220 | **never seen in training; only scene with D-scale labels** |

---

## Tromsø GT — size annotations

- 117 debris polygons with D-scale labels and area measurements
- Distribution: 5 × D1, 25 × D2, 71 × D3, 16 × D4
- `size` column: int {1, 2, 3, 4}
- `area` column: float, m²
- Used for: D-scale proxy calibration, size estimation experiment, D2 detection analysis

> ⚠ OPEN: D1 (n=5) is likely below Sentinel-1 detection floor. May need to exclude from size classifier or treat as "undetectable" class.

---

## SeNorge — snowpack variables (Tromsø supplementary)

- Norwegian national 1km daily snow model
- Variables of interest:
  - `ΔSD₇₂`: 72-hour snow depth change (m) — proxy for fresh snow load
  - bulk density (kg/m³)
- Date: 2024-12-19 to 2024-12-20 (day before and day of Tromsø acquisition)
- Access: thredds.met.no (NetCDF, open access)
- Spatial resolution: 1 km — must be resampled/interpolated to Sentinel-1 10m grid
- Used only for: snowpack fusion experiment on Tromsø (supplementary goal)

> ⚠ OPEN: SeNorge download not yet done. Need to identify exact variable names in the NetCDF schema and confirm December 2024 availability.

---

## Input channels (unchanged from Phase 1)

`[VH_post, VV_post, slope, sin(aspect), cos(aspect)]` = 5 channels, 64×64 patches

- For bi-temporal models: additionally `[VH_pre, VV_pre]` → 7 channels total (3 DEM-derived shared)
- VH/VV clipped to [−25, −5] dB per arXiv:2502.18157

---

## Normalization

- Norm stats JSON from Phase 1: `data/splits/norm_stats_bitemporal.json`
- Stats computed on train split only; applied at inference

---

## What's new in Phase 2

- Pixel-level GT rasters (already in AvalCD; Phase 1 didn't use them)
- SeNorge variables (new download)
- No new SAR scenes added
