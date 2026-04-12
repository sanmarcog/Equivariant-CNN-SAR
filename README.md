# Equivariant CNNs for SAR Avalanche Debris Detection

**Work in progress — training in progress on Hyak. Results will be updated as runs complete.**

Avalanche debris leaves a distinctive backscatter signature in Sentinel-1 radar imagery, but it can appear at any orientation depending on slope aspect and satellite viewing angle. This project asks whether baking rotation symmetry directly into the network architecture — so the model responds identically to a debris patch regardless of orientation, by construction — outperforms a standard CNN that learns orientation invariance from data augmentation alone. Three equivariant architectures (C8, SO(2), D4), derived from group representation theory, are compared against matched-parameter CNN baselines on the AvalCD dataset (35,000 labeled Sentinel-1 patches from four geographic regions), with Tromsø, Norway held out as an unseen test region. Preliminary results show the C8 model reaching AUC 0.780 at 100% training data, with full F1/F2 comparisons against CNN baselines and Bianchi et al. (2021) pending.

---

## Results (preliminary)

Training is running on the University of Washington Hyak cluster. Results below are best validation AUC-ROC from completed runs only. OOD test set (Tromsø) results pending calibration.

| Model | 10% data | 25% data | 50% data | 100% data |
|---|---|---|---|---|
| C8 equivariant CNN | 0.703 | 0.713 | 0.745 | 0.780 |
| SO(2) equivariant CNN | 0.606 | — | — | — |
| D4 equivariant CNN | — | — | — | — |
| CNN baseline (no aug) | — | — | — | — |
| CNN + rotation augmentation | — | — | — | — |
| ResNet-18 (fine-tuned) | — | — | — | — |

All — entries are pending. This table will be updated as runs complete.

---

## Project overview

Avalanche debris detection from satellite SAR imagery is operationally valuable for avalanche forecasting agencies and road maintenance organizations. Sentinel-1 C-band SAR captures backscatter changes over snow-covered terrain that are visible even through clouds and in darkness, making it the primary satellite tool for post-event avalanche mapping.

The scientific question this project addresses is whether *geometric structure in the architecture* — specifically, rotational equivariance — provides a measurable advantage over learning rotation invariance from data augmentation alone. SAR avalanche debris has no preferred orientation: runouts deposit at aspect-dependent angles across mountain terrain, and the satellite viewing geometry varies by acquisition. An equivariant model handles all orientations by construction rather than by memorizing training examples at each angle.

The motivation for the Washington Cascades is operational: while this project trains and evaluates on the AvalCD European/Arctic dataset, the eventual goal is a model useful to forecasters in the Pacific Northwest. A small held-out Cascades test set (in progress, pending collaboration with WSDOT and Crystal Mountain Ski Patrol) will provide a geographically independent operational evaluation.

---

## Mathematical background

A function *f* mapping images to feature representations is **equivariant** to a group *G* if transforming the input by any group element *g* produces a predictably transformed output: *f*(*g* · *x*) = *g* · *f*(*x*). For rotations, this means the network's internal feature maps rotate with the input rather than treating each orientation as a new pattern to learn. Standard CNNs are translation-equivariant by construction (shared weights across spatial positions), but achieve rotation invariance only approximately, through data augmentation. Steerable CNNs — implemented here via the [escnn](https://github.com/QUVA-Lab/escnn) library — extend equivariance to arbitrary subgroups of the Euclidean group *E*(2) by constraining convolutional filters to lie in spaces of *G*-steerable kernels. The filter weights are decomposed into a basis of harmonics (Fourier modes on the group), and the constraint that *f*(*g* · *x*) = *g* · *f*(*x*) is enforced exactly by construction — not learned. This matters for SAR avalanche detection because the guarantee holds for *any* input, including aspect angles and slope orientations not seen during training, which is precisely the regime of geographic generalization tested here.

---

## Architecture

All three equivariant models share a backbone of four convolutional blocks, two output heads, and approximately 391,000 parameters — matched to the CNN baselines for fair comparison.

**Backbone field type progression**

Each layer operates on *geometric tensors* — feature maps where each channel transforms according to a group representation rather than as a scalar. The progression through the backbone is:

- **Input**: trivial representation — each of the 5 input channels (VH, VV, slope, sin(aspect), cos(aspect)) is a scalar that does not transform under rotations. The network "knows" it is receiving scalar fields.
- **Hidden layers**: regular representation — each feature carries a full copy of the group's action. For C8 (8 rotations), this means 8 coupled channels that rotate into each other when the input rotates. A convolution in this space is constrained to be equivariant by the steerable kernel basis. The spatial resolution halves every block (64→32→16→8 pixels).
- **Head 1 output**: trivial representation — GroupPooling averages over group elements, collapsing the equivariant features into rotation-invariant scalars. A global average pool and linear layer produce a single logit for binary classification. This head is used in the training loss.

**Head 2 — orientation readout (visualization only)**

A 1×1 equivariant convolution maps the final backbone features to the *basespace action* representation — a 2D vector field where the group acts as its standard rotation (and reflection, for D4) matrix. After spatial averaging this produces a single [B, 2] vector per image. Because this head is equivariant, the output vector rotates with the input: if the input image is rotated by θ, the output vector rotates by exactly θ. This provides an unsupervised readout of the dominant orientation in each patch — for example, the aspect direction of the strongest debris feature — without any orientation labels. It is not included in the training loss.

For D4, the output is an *undirected axis*: the reflection symmetry means both (x, y) and (x, −y) are valid orientations. Both are plotted to make the ambiguity explicit.

**Group choices**

| Model | Group | Order | Elements | Equivariant to |
|---|---|---|---|---|
| C8 | Cyclic C₈ | 8 | Rotations 0°, 45°, ..., 315° | 8 discrete rotation angles |
| SO(2) | Special orthogonal SO(2) | ∞ | All rotations | Any continuous rotation |
| D4 | Dihedral D₄ | 8 | 4 rotations × 2 reflections | 90° rotations + horizontal/vertical flip |

SO(2) uses band-limited feature fields (maximum frequency L=4, dimension 9 per field) with `NormNonLinearity` and `NormPool` in place of the pointwise ELU and GroupPooling used by the finite groups.

---

## Dataset

[AvalCD](https://zenodo.org/records/15863589) (Gattimgatti et al., 2026) is a Sentinel-1 SAR patch dataset of avalanche debris from four geographic regions:

| Region | Events | Role |
|---|---|---|
| Livigno, Italy | 2 | Train |
| Nuuk, Greenland | 2 | Train |
| Pish, Tajikistan | 1 | Train |
| Tromsø, Norway | 1 | **OOD test (never seen in training)** |

**Split:** 27,206 train / 6,450 val / 2,211 test patches at 64×64 pixels, 10 m resolution. Class imbalance ~1:8 (debris:clean), handled with `WeightedRandomSampler` and `BCEWithLogitsLoss(pos_weight=3.0)`.

**Input channels:** [VH_post, VV_post, slope, sin(aspect), cos(aspect)] — 5 channels. SAR backscatter clipped to [−25, −5] dB. Aspect encoded as (sin, cos) to handle the circular 0°/360° boundary. Slope and aspect are derived from the Copernicus DEM GLO-30 at 10 m.

The Tromsø test set is geographically isolated from all training data and represents operational generalization to an unseen region — consistent with the evaluation protocol in Gattimgatti et al. (arXiv:2603.22658).

---

## Reproducing the experiments

### Requirements

- Hyak access (UW) or equivalent SLURM cluster with NVIDIA GPU
- Apptainer with NVIDIA PyTorch container `pytorch_24.12-py3.sif`
- AvalCD dataset downloaded from Zenodo

### 1. Download data

```bash
python download_data.py --data-dir data/raw
```

### 2. Build patches and splits

```bash
python data_pipeline/build_manifest.py
python data_pipeline/split.py
```

### 3. Set up environment on Hyak

```bash
# Run once on an interactive GPU node
salloc -A demo -p ckpt --gres=gpu:1 --mem=16G --time=0:30:00
bash slurm/setup_venv.sh
```

### 4. Verify equivariance tests pass

```bash
apptainer exec --nv --bind /gscratch /gscratch/scrubbed/sanmarco/pytorch_24.12-py3.sif \
  /bin/bash -c 'source /gscratch/scrubbed/sanmarco/venv/bin/activate && \
                cd /gscratch/scrubbed/sanmarco/equivariant-sar && \
                python -m tests.test_equivariance'
```

All 6 tests (3 backbone + 3 orientation head) must pass before submitting training jobs.

### 5. Submit training

```bash
# Smoke test first (C8, 10% data, 2 epochs)
sbatch slurm/smoke_test.sh

# Full 24-job array (6 models × 4 data fractions) after smoke test passes
sbatch slurm/train_array.sh
```

### 6. Evaluate and calibrate

```bash
# Submit after training array completes
sbatch --dependency=afterok:<train_job_id> slurm/eval_array.sh

# Pull results to local machine
rsync -avz sanmarco@klone.hyak.uw.edu:/gscratch/scrubbed/sanmarco/equivariant-sar/results/ \
  /Users/sanmarco/Documents/GitHub/Equivariant-CNN-SAR/results/
```

### 7. Summary tables

```bash
python evaluate.py --summary    # AUC, F1, F2, MCC, Brier across all runs
python calibrate.py --summary   # Temperature scaling results
```

---

## Repository structure

```
.
├── train.py                        # Training loop (all 6 models)
├── evaluate.py                     # Evaluation: AUC, F1, F2, MCC, Brier, per-event breakdown
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
├── tests/
│   └── test_equivariance.py        # Equivariance unit tests (must pass before training)
│
└── slurm/
    ├── setup_venv.sh               # One-time venv setup inside Apptainer container
    ├── smoke_test.sh               # Single-job smoke test (run before full array)
    ├── train_array.sh              # 24-job training array (6 models × 4 data fractions)
    └── eval_array.sh               # Evaluation + calibration array
```

---

## Related work and positioning

### DL avalanche detection in SAR — a brief history

Deep learning for SAR avalanche detection has progressed steadily from patch classifiers to full-image segmentation. Waldeland et al. (IGARSS 2018) introduced the first CNN-based approach on Sentinel-1, achieving >90% patch accuracy using a VGG-19 variant. Sinha et al. (NeurIPS Climate Change AI 2019) extended this to the French Alps. Bianchi et al. (IEEE JSTARS 2021) made the leap to semantic segmentation with a fully convolutional encoder-decoder on 6-channel Sentinel-1 + topographic input — the same input paradigm used here — reaching F1=66.6% vs. 38.1% for the prior signal-processing baseline. The operational baseline from Eckerstorfer et al. (Remote Sensing 2019) that Bianchi et al. beat is a change-detection algorithm with 67% probability of detection and 46% false alarm rate.

The current DL benchmark for the segmentation task is Bianchi & Grahn (arXiv:2502.18157, 2025), which benchmarks over ten architectures (U-Net, FPN, DeepLabV3+, SegFormer, and others) with ResNet, ConvNeXt, and Swin backbones. Their best model — FPN with an Xception backbone — compensates for rotation sensitivity by applying **test-time augmentation: rotating and flipping each image at inference and averaging predictions**. This workaround acknowledges that standard CNNs treat each orientation as a distinct pattern; it adds inference cost and is not guaranteed to cover all orientations. An equivariant architecture removes this workaround by construction: the model is rotation-equivariant for all inputs, including orientations never seen during training.

### Gattimgatti et al. (2026) — the AvalCD paper

The dataset used here, AvalCD, was released alongside Gattimgatti et al. (arXiv:2603.22658), which reports the best published results on this data to our knowledge: **F1 = 0.806, F2 = 0.841** on Alpine ecoregions using bi-temporal change detection with a deep learning backbone.

**The task formulations are different, and the numbers are not directly comparable.** Gattimgatti et al. use *pre-event and post-event* SAR image pairs; the model sees the change signal directly. This project uses *post-event only*. The difference matters: the pre/post intensity ratio suppresses most non-avalanche backscatter variation (snow compaction, wind slab, vegetation), leaving a much cleaner signal. Single-image classification is a genuinely harder problem. We are not claiming to beat their numbers — we are asking a different question: how much of the available signal can a model extract from a single post-event acquisition, and does architectural equivariance help.

That said, single-image detection is operationally relevant in two scenarios: when pre-event imagery is unavailable or degraded (cloud cover during the pre-event pass, large temporal gap, SAR mode change), and as a component in systems where acquiring co-registered pre-event imagery is logistically difficult for near-real-time applications.

**Tromsø is also their held-out test set.** Gattimgatti et al. use the exact same geographic split: Tromsø excluded from training entirely, evaluated independently. Their F1=0.806 and F2=0.841 are reported on Tromsø. This means our Tromsø results will be directly comparable to theirs on the same test region, with the task formulation (single-image vs. bi-temporal) as the stated difference. This is a cleaner and more honest comparison than "novel evaluation geography" — it is the same benchmark, different approach.

### Equivariant CNNs in remote sensing

The closest architectural analogue is ReDet (Han et al., CVPR 2021), which applies rotation-equivariant CNNs — implemented with the e2cnn library, the predecessor to escnn — to aerial object detection on optical imagery. ReDet uses cyclic group C4 and achieves both higher accuracy (+1.2–3.5 mAP on DOTA) and 60% fewer parameters than the prior SOTA. To our knowledge, no prior work applies group-equivariant CNNs to SAR imagery for detection or segmentation — only one TGRS 2024 paper (RSENet) uses equivariant features for optical-to-SAR *image matching*, which is a different task. The avalanche detection sub-field specifically has never used equivariant architectures; the TTA workaround in Bianchi & Grahn (2025) is the current state of the art for rotation handling.

### What this project adds

- **First application of group-equivariant neural networks to SAR avalanche debris detection**, closing the gap that TTA currently patches with inference-time heuristics.
- **Groups C8, SO(2), and D4** — extending beyond ReDet's C4 to higher-order discrete and continuous symmetry groups, with controlled comparison against matched-parameter baselines.
- **Data-efficiency curves** (10/25/50/100% of training data) testing whether equivariant models require less labelled data — a practically important question given the cost of generating labelled SAR avalanche datasets.
- **Equivariant orientation readout head** providing an unsupervised estimate of debris flow direction per patch, with no orientation labels required.

**Honest caveats about our current results:** the numbers in the results table above are validation AUC-ROC, not F1 or F2, and not on the held-out Tromsø test set. A meaningful comparison to Gattimgatti et al. requires our full test-set F1/F2 results after calibration, which are pending. This section will be updated when those results are available.

### Figures

*Figures will be added here once all training runs complete. Planned plots:*
- *Data-efficiency curves: val AUC vs. training set fraction for all 6 models*
- *ROC and PR curves on Tromsø OOD test set*
- *Calibration reliability diagrams (before/after temperature scaling)*
- *Per-event AUC breakdown across geographic regions*
- *Orientation readout visualizations: C8/SO(2) arrows and D4 axis ambiguity*

---

## Findings and observations

### Bilinear interpolation as accidental speckle reduction

A rotation sensitivity experiment (see `scripts/rotation_sensitivity.py`) rotated 200 test patches from the Tromsø OOD set by 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315° and ran inference with the C8 model at each angle. The AUC-ROC was consistently higher at diagonal angles than at axis-aligned angles:

| Angles | AUC-ROC |
|---|---|
| 0°, 90°, 180°, 270° | 0.7490 |
| 45°, 135°, 225°, 315° | 0.7756 |

The +0.027 difference is uniform across all four diagonal angles — a binary split that mirrors the structure of the interpolation, not random noise.

**Interpretation.** When a 64×64 patch is rotated by 45° using bilinear interpolation, each output pixel is a weighted average of its four nearest neighbours in the original grid. Bilinear interpolation is a spatial low-pass filter (its frequency response is a 2D tent function); this is well-established in signal processing. For SAR imagery, spatial averaging reduces speckle variance — this is the same mechanism as multilooking, which averages N independent intensity observations to suppress the negative-exponential speckle distribution (Goodman 1976; Oliver & Quegan 1998). At 0°/90° rotations the pixel grid aligns exactly and no blending occurs, leaving speckle intact. At 45° the diagonal offset maximises the blending across neighbours, producing a mild denoising effect. Dalsasso et al. (EUSAR 2021) document that any non-nearest-neighbour resampling of SAR images introduces spatial correlation between adjacent pixels — the same physical mechanism, framed there as a problem for despecklers rather than a classification benefit.

**Important caveat.** Interpolation-induced averaging is *not* equivalent to true multilooking: true multilooking averages statistically independent samples of the same ground cell; bilinear interpolation averages spatially adjacent, partially overlapping samples. The variance reduction is real but the statistical improvement in ENL is weaker than the same averaging applied to independent looks.

**Actionable implication.** The 0.027 AUC gap between 45° and 0° rotations suggests that SAR speckle is a meaningful noise floor on classification performance with the current preprocessing. Applying an explicit speckle filter (Lee, Frost, or Refined Lee) before inference — or training on despeckled inputs — could recover a comparable improvement across all six models, not just C8. This is a preprocessing ablation worth running once all training is complete.

**Implication for comparison with prior work.** Bianchi et al. (2021) apply a 5×5 Refined Lee speckle filter as standard preprocessing before training and inference. Our models are trained and evaluated on unfiltered patches. Based on the rotation sensitivity analysis showing ~0.027 AUC improvement from incidental speckle reduction via bilinear interpolation, explicit speckle filtering is expected to recover a similar improvement across all models. Direct numerical comparison with Bianchi et al. (2021)'s F1=0.666 should account for this preprocessing difference. Adding a Lee filter preprocessing step is left as future work.

**Possible connection to Bianchi & Grahn (2025) TTA.** Bianchi & Grahn (2025) apply multi-transformation test-time augmentation at inference, noting it reduces border and checkerboard artifacts. One plausible contributor to TTA's benefit in that setting — beyond orientation coverage — is exactly this interpolation-induced denoising: each rotated version of a patch is mildly despeckled before prediction, and averaging predictions across transformations further smooths classifier variance. This is speculative; Bianchi & Grahn do not ablate TTA or attribute its benefit to speckle reduction, and the general TTA literature attributes improvement to approximate rotation invariance (Shanmugam et al. 2021) rather than to input denoising. The two effects are not mutually exclusive and no quantitative separation exists in the literature. We note it here as a hypothesis worth testing, not an established finding.

---

## References

- Waldeland et al. (2018). *Avalanche Detection in SAR Images Using Deep Learning.* IGARSS 2018.
- Sinha et al. (2019). *Can Avalanche Deposits be Effectively Detected by Deep Learning on Sentinel-1 SAR Images?* NeurIPS Climate Change AI Workshop. hal-02278230.
- Eckerstorfer et al. (2019). *Near-Real Time Automatic Snow Avalanche Activity Monitoring System Using Sentinel-1 SAR Data in Norway.* Remote Sensing 11(23):2863.
- Bianchi et al. (2021). *Snow Avalanche Segmentation in SAR Images With Fully Convolutional Neural Networks.* IEEE JSTARS. arXiv:1910.05411.
- Han et al. (2021). *ReDet: A Rotation-Equivariant Detector for Aerial Object Detection.* CVPR 2021. arXiv:2103.07733.
- Bianchi & Grahn (2025). *Monitoring Snow Avalanches from SAR Data with Deep Learning.* arXiv:2502.18157.
- Gattimgatti et al. (2026). *Large-Scale Avalanche Mapping from SAR Images with Deep Learning-based Change Detection.* arXiv:2603.22658.
- Gattimgatti et al. (2026). *AvalCD dataset.* Zenodo. [doi:10.5281/zenodo.15863589](https://zenodo.org/records/15863589).
- Weiler & Cesa (2019). *General E(2)-Equivariant Steerable CNNs.* NeurIPS 2019. arXiv:1911.08251.
- Cesa et al. (2022). *A Program to Build E(N)-Equivariant Steerable CNNs.* ICLR 2022. [github.com/QUVA-Lab/escnn](https://github.com/QUVA-Lab/escnn).
- Goodman (1976). *Some fundamental properties of speckle.* JOSA 66(11):1145–1150.
- Oliver & Quegan (1998). *Understanding Synthetic Aperture Radar Images.* Artech House.
- Dalsasso, Denis & Tupin (2021). *How to handle spatial correlations in SAR despeckling? Resampling strategies and deep learning approaches.* EUSAR 2021. HAL:hal-02538046.

Full citations with links: [references.md](references.md)
