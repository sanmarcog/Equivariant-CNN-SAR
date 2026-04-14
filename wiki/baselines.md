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

> ⚠ OPEN: audit flag noted thin related work (5 papers for 7-year window). Ingest additional papers from ISPRS/TGRS/Remote Sensing when encountered.
