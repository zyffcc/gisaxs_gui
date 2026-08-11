# Training

Train a TensorFlow 2.15 slot model for 1D GISAXS/SAXS fitting. `Training/train.py` automatically uses TFRecord shards when `train/*.tfrecord` exists, with NPZ kept only as a debug fallback.

Quick training:

```bash
conda activate tf
python Training/train.py \
  --dataset_dir /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_QUICK \
  --model_dir /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS_QUICK \
  --epochs 2 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --quick_test
```

Full training:

```bash
sbatch Training/training.sbatch 50 64 0.0001
```

Training resumes automatically from the latest checkpoint in `MODEL_DIR/checkpoints`.
Progress is printed every `--log_interval` steps and saved every `--save_interval` steps;
the defaults are both 10. Each save updates `model.keras`, `history.json`,
`step_history.json`, `step_history.csv`, and `loss_curve.png`.

Train a curriculum dataset after its build job finishes:

```bash
# K = 1 model, after build array 22984596 finishes
sbatch --dependency=afterany:22984596 Training/training.sbatch \
  50 64 0.0001 \
  /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1 \
  /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS_K1 \
  --log_interval 10 \
  --save_interval 10

# K = 3/4 model, after build array 22984619 finishes
sbatch --dependency=afterany:22984619 Training/training.sbatch \
  50 64 0.0001 \
  /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K3K4 \
  /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS_K3K4 \
  --log_interval 10 \
  --save_interval 10
```

Predict TOP 20:

```bash
python Training/predict_topk.py \
  --model_dir /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS_QUICK \
  --input_csv /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_QUICK/inspection/example_curve.csv \
  --output_dir /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS_QUICK/prediction_example \
  --num_samples 200 \
  --top_k 20
```

The current model has an explicit `D present` head and a relational spacing
loss. Candidate generation then enforces the selected spacing rule as a hard
constraint. Because this adds model inputs and an output head, train into a new
model directory; checkpoints from the older architecture are not shape-compatible.

Inference constraints can combine the existing fixed parameter ranges with D
presence and spacing constraints:

```json
{
  "mode": "free",
  "parameter_ranges": {
    "slot_0": {"R": [5.0, 30.0]}
  },
  "d_constraint": {
    "presence": "optional",
    "spacing_rule": "max_diameter",
    "slot_presence": {"slot_0": "required"}
  }
}
```

`presence` and each `slot_presence` accept `optional`, `absent`, or `required`.
`spacing_rule` accepts `free`, `max_diameter`, or `mean_diameter`. If a required
D cannot satisfy both the relational lower bound and a supplied D upper range,
that posterior candidate is rejected.

TOP-K output is ranked by parameter modes rather than by unique type strings.
Consequently, the same combination (for example `sphere+cylinder`) may appear
several times with distinct parameter distributions. `--parameter_mode_radius`
controls mode separation in normalized parameter space (default 0.10).

When fits are comparably good, a parsimony prior favors fewer components. Its
log penalty is `--complexity_penalty * (K - 1)` (default 1.0). This affects the
ranking probability, while `best_log_rmse`, `best_chi2_weighted`, and the other
fit metrics remain unmodified so that a simpler but visibly worse fit is not
presented as a better physical fit.

## Differentiable physical reconstruction loss

Training can add a TensorFlow implementation of the same sphere, random-cylinder,
vertical-cylinder, structure-factor, resolution, background, and scale equations
used by dataset generation. The full quadrature is evaluated on strided q points
and only a small number of curves per replica, so the forward model remains
accurate without applying its full cost to every item in a batch.

```bash
python Training/train.py \
  --dataset_dir /path/to/dataset \
  --model_dir /path/to/new_model \
  --reconstruction_loss_weight 0.05 \
  --reconstruction_start_epoch 6 \
  --reconstruction_ramp_epochs 5 \
  --reconstruction_q_stride 16 \
  --reconstruction_samples_per_batch 2
```

The weight is zero before `reconstruction_start_epoch`, then ramps linearly to
the requested value. Reconstruction is compared against synthetic `I_clean` in
log intensity with a Huber loss, avoiding detector-gap/noise artifacts. Training
also includes an explicit component-count loss and reports
`component_count_accuracy` for distinguishing K=1,2,3,4.

Monitor Slurm state, live step progress, checkpoints, validation-loss trend,
K accuracy, type accuracy, plateau/degradation, and overfitting warnings with:

```bash
python Training/monitor_training.py \
  --job_id JOB_ID \
  --model_dir /path/to/model
```

Add `--json` for machine-readable output. During training,
`training_status.json` is refreshed at each logging interval. Checkpoints,
`history.json`, `step_history.csv`, `loss_curve.png`, and TensorBoard logs are
updated at save points.
