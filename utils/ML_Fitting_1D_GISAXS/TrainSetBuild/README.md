# TrainSetBuild

Generate synthetic 1D GISAXS/SAXS Yoneda-cut training sets using the existing `utils.fitting` physical model. The default output format is TFRecord for TensorFlow training.

Quick dataset:

```bash
conda activate tf
python TrainSetBuild/build_dataset.py \
  --output_dir /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_QUICK \
  --num_samples 256 \
  --samples_per_shard 128 \
  --seed 1 \
  --format tfrecord \
  --quick_test \
  --overwrite
```

Inspect:

```bash
python TrainSetBuild/inspect_dataset.py \
  --dataset_dir /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_QUICK
```

Full dataset:

```bash
python TrainSetBuild/build_dataset.py \
  --output_dir /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS \
  --num_samples 100000 \
  --samples_per_shard 1000 \
  --seed 42 \
  --max_points 1000 \
  --format tfrecord
```

Default noise is now mild: `poisson_scale=10-200` and `rel_noise=0.001-0.02`.
Use `--poisson_scale_min/max` and `--rel_noise_min/max` to tune it for a specific dataset.
Noisy points at or below `--drop_noisy_floor` are dropped together with their q values; default is `1e-20`.

Parameter sampling is q-conditioned by default: 70% observable, 20% edge, and 10% out-of-window.
Use `--no-q_conditioned_sampling` to restore global sampling, or tune with
`--visible_fraction`, `--edge_fraction`, and `--out_of_window_fraction`.

By default, each synthetic noisy curve also gets short detector-gap style intensity drops:
1-3 local regions, 1-10 points per region, dropped to 1%-70% intensity, capped at 5% of the curve points.
Disable with `--gap_drop_prob 0` or change the cap with `--gap_drop_max_fraction`.

Submit full build:

```bash
sbatch TrainSetBuild/build_trainset_pscpu.sbatch 100000 1000 42
```

Submit full build with four parallel Slurm array tasks on `pscpu` (7 hour limit):

```bash
sbatch --array=0-3 TrainSetBuild/build_trainset_pscpu.sbatch 100000 1000 42 4
```

Submit the same build on `allcpu` (24 hour limit, requeue enabled). The job cleans stale temporary shard files at startup and on termination:

```bash
sbatch --array=0-3 TrainSetBuild/build_trainset_allcpu.sbatch 100000 1000 42 4
```

Equivalent manual four-job submission:

```bash
sbatch TrainSetBuild/build_trainset_pscpu.sbatch 100000 1000 42 4 0
sbatch TrainSetBuild/build_trainset_pscpu.sbatch 100000 1000 42 4 1
sbatch TrainSetBuild/build_trainset_pscpu.sbatch 100000 1000 42 4 2
sbatch TrainSetBuild/build_trainset_pscpu.sbatch 100000 1000 42 4 3
```

Build curriculum-style datasets with fixed component-count ranges:

```bash
# K = 1 only
sbatch --array=0-3 TrainSetBuild/build_trainset_pscpu.sbatch \
  100000 1000 42 4 \
  /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1 \
  1

# K = 1 or 2, uniform probability
sbatch --array=0-3 TrainSetBuild/build_trainset_pscpu.sbatch \
  100000 1000 43 4 \
  /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1K2 \
  1,2

# K = 3 or 4, uniform probability
sbatch --array=0-3 TrainSetBuild/build_trainset_pscpu.sbatch \
  100000 1000 44 4 \
  /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K3K4 \
  3,4

# K = 1 or 2, custom probabilities
sbatch --array=0-3 TrainSetBuild/build_trainset_pscpu.sbatch \
  100000 1000 45 4 \
  /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1K2_p70p30 \
  1,2 \
  0.7,0.3
```
