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

## Optional D and hard spacing constraints

Every active component may have no structure-factor period (`D=sigma_D=0`), with
probability 0.25 by default. Change this with `--d_absent_probability`.

Datasets mix three D-spacing rules by default:

- `free`: no relational lower bound on a present D;
- `max_diameter`: D is greater than the largest active component exclusion size;
- `mean_diameter`: D is greater than the arithmetic mean active exclusion size.

Sphere and vertical-cylinder exclusion size is `2*R`. For a randomly oriented
cylinder it is the conservative circumscribed-sphere diameter
`sqrt((2*R)^2 + h^2)`. A factor of 1.001 makes the generated inequality strict.
If that lower bound reaches the schema maximum `D=500`, the component is stored
without D rather than violating the constraint.

Choose a single rule or custom mixture when building:

```bash
python TrainSetBuild/build_dataset.py \
  --output_dir /path/to/dataset \
  --num_samples 100000 \
  --d_spacing_rules max_diameter,mean_diameter \
  --d_spacing_rule_probs 0.5,0.5 \
  --d_absent_probability 0.25
```

The Slurm wrapper accepts the same settings as positional arguments after
`K_PROBS`: `D_SPACING_RULES`, `D_SPACING_RULE_PROBS`, and
`D_ABSENT_PROBABILITY`. For an array job, for example:

```bash
sbatch --array=0-3 TrainSetBuild/build_trainset_pscpu.sbatch \
  100000 1000 42 4 /path/to/dataset 1,2 0.5,0.5 \
  max_diameter,mean_diameter 0.5,0.5 0.25
```

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
