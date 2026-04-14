# Architecture — Phase 2

**D4-BT backbone (frozen or fine-tuned) + equivariant segmentation decoder + differentiable area regression head.**

---

## Backbone: D4BiTemporalCNN (Phase 1)

- Shared-weight D4-equivariant encoder applied to post and pre patches separately
- Change feature: `GroupPooling(feat_post − feat_pre)` → global avg pool → [B, 256]
- ~391K parameters total
- Phase 1 checkpoint: best AUC=0.912 at 50% data fraction (Tromsø OOD)

> ⚠ OPEN: freeze backbone vs fine-tune end-to-end. Freezing is safer given single training scene for Tromsø; fine-tuning risks catastrophic forgetting. Decision deferred to ablation.

---

## Segmentation decoder

4-stage upsampling path mirroring the encoder:

```
Encoder features [B, 256, 4, 4]
  → R2Conv equivariant (regular → regular, 128ch) + BN + ELU + ConvTranspose2d ×2  → [B, 128, 8, 8]
  → R2Conv equivariant (128 → 64ch)  + BN + ELU + ConvTranspose2d ×2              → [B, 64, 16, 16]
  → R2Conv equivariant (64 → 32ch)   + BN + ELU + ConvTranspose2d ×2              → [B, 32, 32, 32]
  → R2Conv equivariant (32 → 16ch)   + BN + ELU + ConvTranspose2d ×2              → [B, 16, 64, 64]
  → Conv2d 1×1 (trivial repr → scalar)                                             → [B, 1, 64, 64]  (logit mask)
```

> ⚠ OPEN: skip connections from encoder to decoder (U-Net style). Would improve small-deposit recall but increases parameter count and couples backbone to decoder more tightly.

---

## Area regression head

- Input: sigmoid of logit mask, thresholded at 0.5 → binary mask
- `area_pixels = mask.sum(dim=[1,2,3])`
- `area_m2 = area_pixels × (pixel_size_m)²`  where pixel_size_m = 10.0 (Sentinel-1 GRD)
- D-scale proxy: log-linear mapping from area_m2 → {D1, D2, D3, D4} using Tromsø GT thresholds
- **Differentiable path**: use soft mask (pre-threshold) for training; hard mask for reporting

> ✓ DECIDED: area head is supplementary, not in the main segmentation loss. Only used for size estimation experiment on Tromsø.

---

## Loss function

```
L = L_seg + λ_area × L_area
```

- `L_seg`: Binary cross-entropy on pixel mask (with positive pixel weight to handle class imbalance)
- `L_area`: L1 loss on log(area_m2) vs log(GT_area_m2), only for Tromsø samples that have D-scale labels
- `λ_area`: tuned on val set; default 0.1

> ⚠ OPEN: Dice loss vs BCE for L_seg. Dice is better for class-imbalanced segmentation (most pixels are background). Decision deferred to initial experiments.

---

## Inference

- Patch size: 64×64 (unchanged from Phase 1)
- Stride: 16 (75% overlap) — improves D2 detection; see [open_questions.md](open_questions.md) Q2
- Patch logits stitched into full-scene mask via average pooling in overlap regions
- Final scene mask thresholded at 0.5 → binary → connected components → polygon extraction

---

## Preprocessing additions (vs Phase 1)

- **Speckle filter**: Lee or Refined Lee 5×5 applied to VH and VV channels before patching
- **Label smoothing**: replace hard {0,1} GT labels with {ε, 1−ε}, ε=0.05 — fixes T≈50 logit collapse seen in Phase 1 temperature scaling

> ⚠ OPEN: apply speckle filter before or after patching? Before is physically correct; after is faster. Probably before.
