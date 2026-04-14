# Open Questions — Phase 2

**The 3 key scientific questions Phase 2 answers, plus unresolved sub-questions.**

---

## Q1: Can equivariant segmentation match or beat Gattimgatti 2026 with 6× fewer parameters?

**Operationalization**: F1 ≥ 0.806 and/or F2 ≥ 0.841 on Tromsø OOD, with ≤391K parameters.

Sub-questions:
- Does the D4 symmetry group still provide benefit in a segmentation setting (vs classification)?
- Does freezing the Phase 1 backbone hurt performance relative to fine-tuning end-to-end?
- Does label smoothing (ε=0.05) adequately fix the T≈50 logit collapse?

---

## Q2: Does 64×64 patching at 75% overlap give measurable D2 detection advantage vs 128×128?

**Operationalization**: Compare D2-class recall (F2_D2) against Gattimgatti's equivalent metric if reported.

Sub-questions:
- At stride=16, how many patches cover a typical D2 polygon (area ~600–5000 m²)?
- Does higher overlap increase FP rate for non-deposit background?
- Does speckle filtering (Lee 5×5) change the tradeoff?

> ⚠ OPEN: Gattimgatti 2026 may not break down results by D-scale. If they don't, this question can only be answered internally (our model at 64×64 vs 128×128).

---

## Q3: Does SeNorge snowpack fusion improve D-scale classification beyond area alone?

**Operationalization**: D-scale accuracy (area only) vs D-scale accuracy (area + ΔSD₇₂ + density) on Tromsø GT.

Sub-questions:
- Is ΔSD₇₂ available at daily resolution for 2024-12-19/20 from thredds.met.no?
- Does spatial interpolation from 1km to polygon centroid introduce enough error to wash out the signal?
- Can 112 labeled polygons provide sufficient statistical power for a 3-class classifier?

---

## Additional unresolved decisions

See `> ⚠ OPEN` callouts in:
- [architecture.md](architecture.md): freeze vs fine-tune, skip connections, speckle filter timing, Dice vs BCE
- [evaluation.md](evaluation.md): exact IoU threshold used by Gattimgatti, bootstrap vs analytical CIs
- [datasets.md](datasets.md): D1 exclusion from size classifier, SeNorge variable names
- [size_estimation.md](size_estimation.md): empirical area thresholds for D-scale boundaries
- [baselines.md](baselines.md): Gattimgatti architecture details (need to read paper)
