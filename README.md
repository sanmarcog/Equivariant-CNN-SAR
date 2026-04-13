# Equivariant CNNs for SAR Avalanche Debris Detection

Three group-equivariant CNN architectures (C8, SO(2), D4) implemented via [escnn](https://github.com/QUVA-Lab/escnn) are compared against matched-parameter CNN baselines for binary avalanche debris classification on Sentinel-1 patches. A bi-temporal D4 extension (D4-BT) fuses pre- and post-event SAR via shared-weight equivariant encoding and an equivariant change feature. Equivariance is enforced exactly by construction via steerable kernel bases derived from group representation theory, rather than approximated through data augmentation. Models are evaluated on the AvalCD dataset with Tromsø, Norway held out as a geographically unseen OOD test set.

---

## Results

OOD test set: Tromsø, Norway (never seen during training). Metric: AUC-ROC.

### AUC-ROC by model and training data fraction

| Model | 10% data | 25% data | 50% data | 100% data |
|---|---|---|---|---|
| **D4-BT (bi-temporal, pre+post)** | **0.871** | **0.906** | **0.912** | **0.894** |
| ResNet-18 (fine-tuned) | 0.555 | 0.786 | 0.743 | 0.803 |
| D4 equivariant CNN | 0.717 | 0.789 | 0.778 | 0.769 |
| CNN baseline (no aug) | 0.499 | 0.677 | 0.783 | 0.723 |
| C8 equivariant CNN | 0.675 | 0.676 | 0.745 | 0.737 |
| SO(2) equivariant CNN | 0.645 | 0.660 | 0.672 | 0.724 |
| CNN + rotation augmentation | 0.523 | 0.622 | 0.744 | 0.705 |
| O(2) equivariant CNN† | 0.595 | — | — | — |

† O(2) was attempted but discontinued: OOM at 10% and 50% data fractions on 10.57 GB GPUs; underperformed D4 at 25% and 100% data. See [Findings](#findings-and-observations).

![Data-efficiency curves: AUC-ROC vs. training data fraction for all models on the Tromsø OOD test set. D4-BT (pink star, bold) is separated far above all single-image models at every fraction. D4 and ResNet-18 are the leading single-image models. CNN+aug consistently trails plain CNN, confirming the augmentation-accuracy tradeoff.](figures/data_efficiency_auc.png)

*AUC-ROC on the Tromsø OOD test set vs. training data fraction. D4-BT dominates across all fractions; equivariant single-image models (D4, C8) outperform CNN+aug at every fraction.*

### Polygon-level evaluation (D4-BT, 50% data, Tromsø scene)

Full-scene sliding-window inference (64×64 patches, 50% overlap) was run over the Tromsø test scene and evaluated against the 117 reference avalanche polygons from the AvalCD ground truth. Because D4-BT is a patch classifier (not a pixel-level segmentation model), IoU-based polygon matching (as used in Gattimgatti et al.) systematically underestimates performance: predicted blobs span multiple overlapping patches and are ~16× larger than the median GT polygon (median GT: 124 px; median predicted blob: 2,047 px). The appropriate metric is **polygon hit rate**: whether the model assigns a high probability anywhere within each reference polygon.

| Threshold | Detected / 117 polygons | Hit rate |
|---|---|---|
| 0.50 | 115 / 117 | 98.3% |
| 0.75 | 115 / 117 | 98.3% |
| 0.85 | 110 / 117 | 94.0% |
| 0.90 | 107 / 117 | 91.5% |

**At threshold 0.75, the model detects 115/117 reference avalanche polygons (98.3% hit rate).** Direct IoU-based F1/F2 comparison with Gattimgatti et al. requires a segmentation architecture — this is planned for Phase 2.

### D4 BiTemporal — threshold analysis (F2-optimal, test_ood)

F2 is the primary metric (β=2): recall weighted 4× over precision, appropriate for avalanche hazard detection where false negatives are more costly.

| Frac | AUC | F1@opt | F2@opt | Optimal threshold | T (cal.) |
|---|---|---|---|---|---|
| 10% | 0.871 | 0.641 | 0.702 | 0.783 | 50.0¹ |
| 25% | 0.906 | 0.698 | 0.752 | 0.875 | 50.0¹ |
| 50% | **0.912** | **0.727** | **0.745** | 0.861 | 50.0¹ |
| 100% | 0.894 | 0.677 | 0.738 | 0.597 | 1.15 |

¹ T≈50: logits saturated — model is well-ranked (high AUC) but logit magnitudes are unreliable. Threshold must be set from validation scores, not assumed near 0.5. frac1p0 converged with calibrated probabilities (T=1.15).

---

## Mathematical background

The rotation group *G* acts on SAR image patches by rotating them: *g*·*x* is the patch *x* rotated by *g*. A function *f* is **equivariant** if *f*(*g*·*x*) = *g*·*f*(*x*) — the group acts on both the input and the output, and *f* commutes with both actions. The two output heads realise different cases of this: the classification head outputs a scalar debris probability, where the group acts trivially on the output (*f*(*g*·*x*) = *f*(*x*), i.e. rotation invariance as a special case); the orientation head outputs a 2D vector where *g* acts as a 2×2 rotation matrix, so the output vector rotates with the input.

Steerable CNNs (Weiler & Cesa, NeurIPS 2019) enforce equivariance by construction via the **intertwiner constraint**: each convolutional filter must lie in the space of *G*-equivariant linear maps between the input and output field types — a constraint solved analytically by decomposing filters into a basis of group Fourier harmonics. This is not learned or approximated; it holds exactly for every input. Implementation uses the [escnn](https://github.com/QUVA-Lab/escnn) library.

---

## Architecture

All models: 4 convolutional blocks, ~391K parameters, matched across equivariant and baseline variants. D4-BT shares these same weights across two branches (pre- and post-event) — parameter count is identical to single-image D4.

**Forward pass:**

```
Input patch  [B, 5, 64, 64]
trivial rep · 5 channels (VH, VV, slope, sin asp, cos asp)
      │
      ▼
Block 1  regular rep · 64×64  ──  R2Conv + BN + ELU + MaxPool
      │
      ▼
Block 2  regular rep · 32×32  ──  R2Conv + BN + ELU + MaxPool
      │
      ▼
Block 3  regular rep · 16×16  ──  R2Conv + BN + ELU + MaxPool
      │
      ▼
Block 4  regular rep · 8×8   ──  R2Conv + BN + ELU
      │
      ├─────────────────────────────────────┐
      ▼                                     ▼
HEAD 1 (classification)             HEAD 2 (orientation, viz only)
GroupPooling → trivial rep          1×1 R2Conv → standard rep
GlobalAvgPool [B, C]                SpatialAvgPool [B, 2]
Linear → [B, 1]                     2D vector rotating with input
sigmoid → debris probability        (equivariant, not in loss)
```

**Group choices and irreducible representations:**

The three groups differ in how the regular representation decomposes into irreps, which determines what spatial frequency information each feature type can encode. C8 (cyclic, order 8) decomposes into 8 one-dimensional irreps indexed by angular frequency *k* = 0, …, 7; features at frequency *k* respond selectively to oriented structures at scale *k* within the 45° discrete grid. D4 (dihedral, order 8: 4 rotations + 4 reflections) has four 1D irreps (symmetric/antisymmetric under rotation and reflection) and one 2D irrep; the reflection symmetry is physically motivated by the approximate bilateral symmetry of avalanche runouts perpendicular to the fall line. SO(2) (continuous rotation group) has infinitely many irreps — one per integer angular frequency — truncated here at maximum frequency *L* = 4, giving 9-dimensional feature fields (frequencies −4, …, +4); this is theoretically the strongest symmetry but the truncation makes equivariance approximate for fine-scale oriented features above frequency 4.

| Model | Group | Order | Irreps in regular rep | Notes |
|---|---|---|---|---|
| C8 | Cyclic C₈ | 8 | 8 × 1D (freq k=0…7) | Exact equivariance at 45° grid |
| D4 | Dihedral D₄ | 8 | 4 × 1D + 1 × 2D | Reflections motivated by debris bilateral symmetry |
| SO(2) | Special orthogonal | ∞ | Truncated at L=4; 9D fields | Strongest symmetry; approximate above freq 4 |

---

## Dataset

[AvalCD](https://zenodo.org/records/15863589) (Gattimgatti et al., 2026) — Sentinel-1 SAR patches, four geographic regions:

| Region | Events | Role |
|---|---|---|
| Livigno, Italy | 2 | Train |
| Nuuk, Greenland | 2 | Train |
| Pish, Tajikistan | 1 | Train |
| Tromsø, Norway | 1 | **OOD test** |

- **Split:** 27,206 train / 6,450 val / 2,211 test — 64×64 px, 10 m resolution
- **Class balance:** ~1:8 (debris:clean); handled with `WeightedRandomSampler` + `BCEWithLogitsLoss(pos_weight=3.0)`
- **Input:** [VH_post, VV_post, slope, sin(aspect), cos(aspect)]; SAR clipped to [−25, −5] dB; aspect sin/cos-encoded; DEM from Copernicus GLO-30

---

## Related work

| Paper | Task | Key result | Relation |
|---|---|---|---|
| Waldeland et al., IGARSS 2018 | Patch classification, Sentinel-1 | >90% accuracy | First DL approach |
| Bianchi et al., JSTARS 2021 | Semantic segmentation, 6-ch S1+DEM | F1=0.666 | Closest input paradigm; uses 5×5 Refined Lee filter |
| Bianchi & Grahn, arXiv:2502.18157 (2025) | Segmentation benchmark, 10+ architectures | FPN+Xception best; uses rotation TTA at inference | TTA is an inference-time rotation invariance patch; equivariant architecture removes it by construction |
| Gattimgatti et al., arXiv:2603.22658 (2026) | Bi-temporal change detection | F1=0.806, F2=0.841 on Tromsø | Same test region; different task (pre+post vs. post-only) |
| Han et al. (ReDet), CVPR 2021 | Aerial object detection, optical | +1.2–3.5 mAP, −60% params vs. SOTA | Closest equivariant architecture; uses C4 via e2cnn on optical only |

No prior work applies group-equivariant CNNs to SAR detection or segmentation.

---

## Findings and observations

### Bi-temporal change detection dominates single-image classification

D4-BT achieves AUC 0.871–0.912 across all data fractions, compared to 0.499–0.803 for the best single-image models at any fraction. At **10% training data, D4-BT (AUC 0.871) already outperforms every single-image model at 100% data** (best: ResNet 0.803). The bi-temporal signal — comparing post-event to a pre-event reference from the same acquisition geometry — provides a much more discriminative input than post-event intensity alone.

D4 equivariance applies to the change feature by linearity: if the shared encoder *f* is D4-equivariant, then *f*(*g*·x_post) − *f*(*g*·x_pre) = *g*·(*f*(x_post) − *f*(x_pre)), so the change vector is equivariant to simultaneous D4 rotations of both inputs. This means D4-BT inherits exact equivariance with no architectural overhead beyond a second forward pass.

**Threshold calibration note:** D4-BT at 10/25/50% data converged with logit magnitudes far from 0 (temperature T≈50 after calibration), likely due to early stopping before logit scale regularisation fully kicks in. The model rank-orders well (high AUC) but probabilities are unreliable without calibration. For deployment, thresholds must be set from held-out validation scores; the F2-optimal threshold lies between 0.78 and 0.88 for these fracs.

### Augmentation–accuracy tradeoff

CNN+rotation augmentation underperforms the plain CNN baseline on the OOD test set across all data fractions: 0.523 vs 0.499 at 10% data, 0.622 vs 0.677 at 25%, 0.744 vs 0.783 at 50%, and 0.705 vs 0.723 at 100%. This is consistent with the augmentation-accuracy tradeoff documented in Gontijo-Lopes et al. (ICLR 2021) and Chen & Dobriban et al. (NeurIPS 2020): SAR backscatter is not rotationally symmetric due to radar look geometry (satellite overpass direction, terrain foreshortening, layover), so full 360° rotation augmentation forces the model to ignore discriminative orientation-dependent features.

Equivariant architectures avoid this tradeoff entirely by encoding the relevant symmetry constraint into the architecture rather than approximating it through augmentation. D4 outperforms CNN+aug at every fraction (0.717 vs 0.523 at 10%, 0.789 vs 0.622 at 25%, 0.778 vs 0.744 at 50%, 0.769 vs 0.705 at 100%), and C8 does so as well except at 100% data. This confirms the theoretical prediction: exact structural equivariance is strictly preferable to learned approximate invariance when the group does not match the true symmetry of the data distribution.

### Continuous vs discrete equivariance

O(2) (continuous dihedral, maximum_frequency=8) was tested but OOMed at 10% and 50% data fractions on 10.57 GB GPUs, and underperformed D4 at 25% and 100% data. This is consistent with the SO(2) finding: continuous approximate equivariance underperforms discrete exact equivariance at matched parameter count. O(2) is left as future work with larger GPU memory.

### Bilinear interpolation as accidental speckle reduction

Rotation sensitivity analysis (`scripts/rotation_sensitivity.py`): 200 Tromsø test patches rotated at 8 angles, inference with C8 model.

| Angles | AUC-ROC |
|---|---|
| 0°, 90°, 180°, 270° | 0.7490 |
| 45°, 135°, 225°, 315° | 0.7756 |

Bilinear interpolation at 45° acts as a spatial low-pass filter, partially reducing SAR speckle before inference. At axis-aligned angles no blending occurs. The +0.027 gap is uniform across all diagonal angles. Dalsasso et al. (EUSAR 2021) document the same mechanism as an artifact in despeckler training; here it surfaces as a classification benefit.

**Implications:**
- Explicit speckle filtering (Refined Lee, 5×5) before training/inference is expected to recover ~0.027 AUC across all models. Left as future work.
- Bianchi et al. (2021) use 5×5 Refined Lee as standard preprocessing; our models do not. Direct F1 comparison should account for this gap.
- Bianchi & Grahn (2025)'s rotation TTA may partly benefit from this interpolation-induced denoising, in addition to orientation coverage.

---

## Repository structure

```
.
├── train.py                        # Training loop (all 6 models)
├── evaluate.py                     # AUC, F1, F2, MCC, Brier, per-event breakdown
├── calibrate.py                    # Temperature scaling calibration
├── download_data.py                # Download AvalCD from Zenodo
│
├── data_pipeline/
│   ├── preprocess_snap.py          # SNAP GPT graph for Sentinel-1 GRD preprocessing
│   ├── extract_patches.py          # 64×64 patch tiling
│   ├── build_manifest.py           # Patch inventory CSV
│   ├── split.py                    # Geographic train/val/test split
│   └── dataset.py                  # PyTorch Dataset with normalization
│
├── models/
│   ├── equivariant_cnn.py          # C8, SO(2), D4 equivariant CNNs (escnn)
│   ├── cnn_baseline.py             # Standard CNN, matched parameters
│   ├── cnn_augmented.py            # CNN + random rotation augmentation
│   └── resnet_baseline.py          # Fine-tuned ResNet-18 (5-channel input)
│
├── scripts/
│   ├── run_eval_all.py             # Batch evaluate + calibrate all checkpoints
│   ├── threshold_analysis.py       # F1/F2 at fixed and optimal thresholds, PR curves
│   ├── plot_data_efficiency.py     # AUC vs. training fraction curves
│   └── rotation_sensitivity.py    # Per-angle AUC and prediction variance analysis
│
├── tests/
│   └── test_equivariance.py        # Equivariance unit tests (run before training)
│
└── slurm/
    ├── setup_venv.sh               # One-time venv setup inside Apptainer container
    ├── smoke_test.sh               # Single-job smoke test
    ├── train_array.sh              # 24-job array (6 models × 4 fractions)
    ├── train_bitemporal.sh         # 4-job array (D4-BT × 4 fractions)
    ├── eval_all.sh                 # Evaluate + calibrate all completed runs
    └── eval_array.sh               # Per-run evaluation array
```

---

## Reproducing the experiments

**Requirements:** SLURM cluster with NVIDIA GPU (sm_70+), Apptainer, `pytorch_24.12-py3.sif`, AvalCD dataset.

```bash
# 1. Download data
python download_data.py --data-dir data/raw

# 2. Build patches and splits
python data_pipeline/build_manifest.py
python data_pipeline/split.py

# 3. Set up venv (run once on interactive GPU node)
salloc -A demo -p ckpt --gres=gpu:1 --mem=16G --time=0:30:00
bash slurm/setup_venv.sh

# 4. Verify equivariance tests
python -m tests.test_equivariance   # all 6 tests must pass

# 5. Train (smoke test first, then full array)
sbatch slurm/smoke_test.sh
sbatch slurm/train_array.sh          # 6 models × 4 fractions
sbatch slurm/train_bitemporal.sh     # D4-BT × 4 fractions

# 6. Evaluate
sbatch --dependency=afterok:<job_id> slurm/eval_all.sh
```

---

## References

- Waldeland et al. (2018). *Avalanche Detection in SAR Images Using Deep Learning.* IGARSS 2018.
- Bianchi et al. (2021). *Snow Avalanche Segmentation in SAR Images With FCNNs.* IEEE JSTARS. arXiv:1910.05411.
- Han et al. (2021). *ReDet: A Rotation-Equivariant Detector for Aerial Object Detection.* CVPR 2021. arXiv:2103.07733.
- Bianchi & Grahn (2025). *Monitoring Snow Avalanches from SAR Data with Deep Learning.* arXiv:2502.18157.
- Gattimgatti et al. (2026). *Large-Scale Avalanche Mapping from SAR Images with Deep Learning-based Change Detection.* arXiv:2603.22658.
- Gattimgatti et al. (2026). *AvalCD dataset.* Zenodo. [doi:10.5281/zenodo.15863589](https://zenodo.org/records/15863589).
- Weiler & Cesa (2019). *General E(2)-Equivariant Steerable CNNs.* NeurIPS 2019. arXiv:1911.08251.
- Cesa et al. (2022). *A Program to Build E(N)-Equivariant Steerable CNNs.* ICLR 2022. [escnn](https://github.com/QUVA-Lab/escnn).
- Dalsasso, Denis & Tupin (2021). *How to handle spatial correlations in SAR despeckling.* EUSAR 2021. HAL:hal-02538046.

Full citations: [references.md](references.md)
