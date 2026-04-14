# Baselines — Phase 2

**Primary comparison: Gattimgatti et al. 2026 (arXiv:2603.22658).**

---

## Gattimgatti et al. 2026

| Property | Value |
|----------|-------|
| arXiv | 2603.22658 |
| Task | Pixel-level SAR avalanche segmentation |
| Dataset | AvalCD (same) |
| Geographic split | Same — Tromsø held out as OOD test |
| Train polygons | 112 (vs our 117 — they may use a slightly different AvalCD version) |
| Architecture | Not equivariant; details TBD — need to read paper |
| Parameters | ~2.39M |
| Patch size | 128×128 |
| Metric | IoU-based polygon F1/F2 |
| **F1 (Tromsø)** | **0.806** |
| **F2 (Tromsø)** | **0.841** |

> ⚠ OPEN: Gattimgatti architecture details not yet extracted. Need to ingest paper to fill in: backbone, decoder, loss function, inference stride, IoU matching threshold.

### Why this comparison is valid
- Identical geographic split (same hold-out logic, Tromsø never in training)
- Same dataset source (AvalCD)
- Same OOD test scene (Tromsø_20241220)

### Known differences to document
1. 112 vs 117 GT polygons — likely different AvalCD version; affects TP/FP counts
2. 128×128 vs 64×64 patch size — we argue smaller patches give D2 detection advantage
3. 2.39M vs ~391K parameters — our primary efficiency claim
4. Equivariant vs standard convolutions — our primary architectural claim

---

## Internal Phase 1 baselines (carried forward for context)

See [phase1_results.md](phase1_results.md) for full numbers.

| Model | Best AUC (OOD Tromsø) | Params |
|-------|----------------------|--------|
| D4-BT (bi-temporal equivariant) | 0.912 @ 50% data | ~391K |
| CNN-BT (bi-temporal plain CNN) | 0.789 @ 50% data | ~391K |
| D4 single-image | 0.814 @ 100% data | ~391K |
| ResNet-18 | 0.823 @ 100% data | 11.2M |

These are patch-level AUC numbers — not directly comparable to Phase 2 polygon F1/F2.
The D4-BT backbone is the starting point for Phase 2.

---

## Other relevant papers

### Bianchi & Grahn 2025 — arXiv:2502.18157
*Monitoring Snow Avalanches from SAR Data with Deep Learning.*

| Property | Value |
|----------|-------|
| Task | SAR avalanche segmentation benchmark |
| Architectures tested | 10+, including FPN+Xception (best) |
| Key finding | Rotation TTA at inference improves all models |
| Relevance to Phase 2 | Defines input representation used here: VH/VV channels, dB clipping [−25, −5], DEM-derived slope/aspect. Equivariant architecture removes the need for TTA by construction. |

> ⚠ OPEN: check whether Bianchi & Grahn use the same AvalCD dataset or a different SAR corpus. If same, their FPN+Xception numbers are an additional comparison point.

---

### Bianchi et al. 2021 — arXiv:1910.05411
*Snow Avalanche Segmentation in SAR Images With FCNNs.* IEEE JSTARS.

| Property | Value |
|----------|-------|
| Task | SAR avalanche segmentation (earlier work) |
| Relevance to Phase 2 | Foundational paper for FCNNs on SAR avalanche data; establishes the pixel-level segmentation framing this project builds on. |

> ⚠ OPEN: not yet ingested. Read to check metrics, dataset, and architectural choices for related work section.

---

### Weiler & Cesa 2019 — arXiv:1911.08251
*General E(2)-Equivariant Steerable CNNs.* NeurIPS 2019.

| Property | Value |
|----------|-------|
| Task | Theoretical framework + image classification |
| Relevance to Phase 2 | Theoretical foundation for all equivariant models used here. D4, C8, SO(2) groups implemented via escnn which implements this framework. |

---

### Cesa et al. 2022 — ICLR 2022
*A Program to Build E(N)-Equivariant Steerable CNNs.* ([escnn](https://github.com/QUVA-Lab/escnn))

| Property | Value |
|----------|-------|
| Task | Software / framework paper |
| Relevance to Phase 2 | The escnn library is the implementation backbone for all equivariant layers including the decoder R2Conv layers. |

---

### Han et al. 2021 — arXiv:2103.07733
*ReDet: A Rotation-Equivariant Detector for Aerial Object Detection.* CVPR 2021.

| Property | Value |
|----------|-------|
| Task | Rotation-equivariant object detection in aerial imagery |
| Relevance to Phase 2 | Closest prior work applying equivariant CNNs to aerial/satellite imagery. Supports the novelty claim that equivariant CNNs have not been applied to SAR avalanche detection before this project. |

> ⚠ OPEN: audit flag noted thin related work coverage. Continue adding papers from ISPRS/TGRS/Remote Sensing as encountered during Phase 2.
