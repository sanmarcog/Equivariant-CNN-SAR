# SLURM Scripts

## First time on Hyak

1. Clone the repo and transfer data:
   ```bash
   # On your local machine
   rsync -avP data/splits/  <netid>@klone.hyak.uw.edu:~/equivariant-sar/data/splits/
   rsync -avP data/raw/patches/  <netid>@klone.hyak.uw.edu:~/equivariant-sar/data/raw/patches/
   ```

2. Set up the environment (login node, run once):
   ```bash
   bash slurm/setup_hyak.sh
   ```

3. Edit the two `#SBATCH --account=ACCOUNT` lines in `train_array.sh` and `eval_array.sh`
   to match your Hyak allocation. Check available accounts with:
   ```bash
   hyakalloc
   ```

4. Confirm equivariance tests pass:
   ```bash
   conda activate sar-equivariant
   python -m tests.test_equivariance
   ```

## Submitting jobs

```bash
# Submit all 24 training runs
sbatch slurm/train_array.sh
# Returns something like: Submitted batch job 12345

# Submit evaluation after all training jobs finish
sbatch --dependency=afterok:12345 slurm/eval_array.sh
```

## Job layout

| Task ID | Model  | Fraction |
|---------|--------|----------|
| 0       | c8     | 0.10     |
| 1       | c8     | 0.25     |
| 2       | c8     | 0.50     |
| 3       | c8     | 1.00     |
| 4       | so2    | 0.10     |
| 5       | so2    | 0.25     |
| ...     | ...    | ...      |
| 20      | resnet | 0.10     |
| 21      | resnet | 0.25     |
| 22      | resnet | 0.50     |
| 23      | resnet | 1.00     |

## Re-running a single job

```bash
sbatch --array=3 slurm/train_array.sh    # c8 at 100% data only
sbatch --array=3 slurm/eval_array.sh
```

## Monitoring

```bash
squeue -u $USER
tail -f logs/train_3.log
```

## After all jobs finish

```bash
# Print full results table
python evaluate.py --summary

# Print calibration table
python calibrate.py --summary
```

## Checkpoints and results layout

```
checkpoints/
  c8_frac1p0/
    last.pt     ← saved every epoch (preemption recovery)
    best.pt     ← best val AUC

results/
  c8_frac1p0/
    metrics.json
    calibration.json
    scores_val.npz
    scores_test_ood.npz
    scores_val_calibrated.npz
    scores_test_ood_calibrated.npz
    figures/
      roc_val.png
      pr_val.png
      scores_val.png
      scores_cdf_val.png
      confusion_val.png
      auc_by_event_val.png
      auc_by_region_val.png
      reliability_val.png
      ... (same for test_ood)
```
