# 1D GISAXS K=1 Debug Pipeline Audit

Audit date: 2026-06-05

Scope: audit only. No training or prediction code was changed for this report. A diagnostic JSON was written to `Training/mean_candidate_debug.json`.

## Executive Summary

The inspection example has a valid oracle label, and the physics chain is now closed:

| Check | logRMSE |
|---|---:|
| `I_clean` vs true-label forward on full inspection curve | `3.32e-08` |
| `I_noisy` / example `I` vs true-label forward on full inspection curve | `0.216` |
| `I` used by `predict_topk.load_curve()` vs true-label forward after current filtering | `0.0584` |

The bad `predict_topk.py` result is not caused by the forward model or the inactive-param oracle bug anymore. The current strongest evidence points to two issues:

1. **Model prediction is poor for this example.** On the exact points used by `predict_topk`, the deterministic mean candidate has `logRMSE = 2.50`. A 200-sample posterior probe with `sampling_std=0.005` found best sampled `logRMSE = 1.19`, still bad.
2. **Prediction preprocessing is not identical to training preprocessing.** `predict_topk.load_curve()` currently removes low-intensity points using `I > percentile(I_positive, 0.5) * 5`. For the inspection example this changes the curve from 285 points, `q_max=0.536`, to 139 points, `q_max=0.334`. Training did not use this percentile filter.

The decomposition says both heads are off, with the parameter head worse:

| Forward combination, compared to `predict_topk` input `I` | logRMSE |
|---|---:|
| A. model params + model global | `2.50` |
| B. model params + true global | `1.83` |
| C. true params + model global | `0.647` |
| D. true params + true global | `0.0584` |

## 1. Dataset / Inspection Example

Files:

- `/data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1/inspection/example_curve.csv`
- `/data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1/inspection/example_curve_with_clean.csv`
- `/data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1/inspection/example_ground_truth.json`

The example does have an oracle label.

True component:

```text
type = cylinder
weight = 1.0
R = 97.19374084472656
sigma_R = 51.84303665161133
h = 15.80517578125
sigma_h = 5.200435161590576
D = 30.593164443969727
sigma_D = 16.75853157043457
```

True global:

```text
BG = 0.26741090416908264
sigma_Res = 0.07061117142438889
nu_Res = 5.1258544921875
int_Res = 0.0
k = 827.541259765625
```

Oracle closure:

```text
I_clean vs true-label forward logRMSE = 3.32e-08
I_noisy/example I vs true-label forward logRMSE = 0.216
```

The example is from the K=1 train split:

```text
source shard = train/shard_000075.tfrecord
sample_index_in_loaded_records = 7
sampling_mode = observable
```

Therefore the inspection example is from the same dataset family, but note that `predict_topk.load_curve()` currently transforms the visible data distribution by removing low-intensity points.

## 2. Preprocessing Consistency

| Path | Curve used for input | Sigma used | q transform | I transform | global features | point mask / padding | q/downsample behavior |
|---|---|---|---|---|---|---|---|
| TrainSetBuild dataset generation | `I_noisy` after noise, missing q windows, gap drops, noisy-floor deletion | generated `sigma` from noise model | `schema.normalize_logq(q)` | `log(I_noisy)`, then median/IQR normalize | `[q_min_norm, q_max_norm, N/MAX_POINTS, I_offset, I_scale]` from `I_noisy` | pad to `schema.MAX_POINTS=1000` | q grid sampled linearly; no percentile intensity filter |
| Training/data_loader.py | precomputed TFRecord `x` | precomputed TFRecord `x[:,2]` | already in `x[:,0]` | already in `x[:,1]` | stored `global_features` | TFRecord shape `(1000, ...)` | no runtime q filtering |
| overfit_debug.py | intentionally uses `I_clean` | artificial `noise_frac * I_clean` | same `preprocess_curve()` | same formula, but on `I_clean` | same formula, but on `I_clean` | pad to `debug_max_points`; often 256 in debug | optional uniform downsample by `np.linspace` |
| predict_topk.py | user input `I` from CSV/txt | CSV sigma if present, else `0.05 * I` | same `preprocess_curve()` | same formula, but on provided input `I` | same formula, but on provided input `I` | pad to `schema.MAX_POINTS=1000` | applies q range, optional outlier filter, and currently low-intensity percentile floor in `load_curve()` |

Important mismatch:

```text
inspection raw curve: 285 points, q=[0.0539, 0.5359], I=[3.62, 85.76]
predict_topk loaded: 139 points, q=[0.0539, 0.3340], I=[18.81, 85.76]
```

This is caused by:

```python
floor = percentile(positive, 0.5)
keep = keep & (I > floor * 5.0)
```

That filter was useful for clipped zero-count tails, but here it removes real low-intensity high-q data. It is not used during training.

## 3. Model Input Shape Consistency

Model directory:

`/data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS_K1`

Model config:

```json
{
  "max_points": 1000,
  "max_slots": 4,
  "num_types": 4,
  "learning_rate": 0.0001,
  "batch_size": 64,
  "epochs": 50
}
```

Loaded model input shapes:

```text
x: (None, 1000, 3)
point_mask: (None, 1000)
global_features: (None, 5)
type_allowed: (None, 4, 4)
param_low_norm/high_norm/range_mask: (None, 4, 4, 6)
force_exist: (None, 4)
global_low/high/range_mask: (None, 5)
```

Conclusion:

- The current K1 model is a 1000-point model.
- `predict_topk.py` also pads to `schema.MAX_POINTS=1000`.
- There is no 256-vs-1000 mismatch for the tested K1 model path.
- `overfit_debug.py` can build 256-point models when `--debug_max_points 256`, but those debug models are local debug artifacts, not the current `/data/dust/.../ML_1D_Fitting_GISAXS_K1` model.

## 4. Inactive Parameter Mask Chain

Current status:

| File | Status |
|---|---|
| `TrainSetBuild/schema.py` | Has `apply_type_param_mask()`, `denormalize_params_with_mask()`, `effective_param_mask()`, and optional-zero global denorm |
| `TrainSetBuild/physics_adapter.py` | Assumes incoming physical params are already correct; no normalization logic here |
| `Training/predict_topk.py` | `sample_candidate()` and `mean_candidate()` now apply type mask after denormalization |
| `Training/overfit_debug.py` | Label oracle uses masked denorm; prediction curve uses masked denorm |
| `Training/losses.py` | Param loss uses `slot_param_mask`; no physical denorm here |
| `Training/data_loader.py` / `tfrecord_io.py` | Old TFRecord optional `D/sigma_D=0` masks are corrected on read |
| `Training/train.py` | Calls data loader and losses; no physical denorm |

Remaining risk:

- `schema.type_param_mask(type_id)` is a type-level mask. It does not represent optional “structure factor absent” for a supported type. For example, sphere/cylinder/vertical cylinder can physically have `D=0, sigma_D=0`, but type-level mask alone marks `D/sigma_D` active for supported types.
- For labels this is handled using `effective_param_mask(type_id, stored_phys)`.
- For predictions there is no explicit “D active/inactive” head, so predicted candidates for a type with supported `D/sigma_D` cannot cleanly express “no structure factor” except by outputting values near the lower bound. This may matter on samples where true `D=0`.

This residual risk is not the root cause of the tested inspection example, because its true cylinder has active `D/sigma_D`.

## 5. Weight Logic

K=1 true labels:

```text
true slot_weight = [1.0, 0.0, 0.0, 0.0]
```

Current `predict_topk.py` logic:

- `sample_candidate()` computes weight softmax only over active sampled slots.
- `mean_candidate()` also computes weight softmax only over active slots.
- For exactly one active slot, the candidate weight is exactly `1.0`.

For the tested mean candidate:

```text
active slots = 1
mean candidate weight = 1.0
```

Therefore this K=1 example is not failing because active-only weight is less than 1.

Potential caveat:

- Training loss uses softmax over all 4 slots against target `[1, 0, 0, 0]`.
- Prediction uses active-only softmax. For K=1 this is actually helpful and forces candidate weight to 1 if only one slot is active.
- If multiple slots are sampled without `--exact_nonempty 1`, weights are distributed among active sampled slots. For strict K=1 inference, use `--exact_nonempty 1` or force/constraint only one active slot.

## 6. Params vs Global Failure Decomposition

On the same inspection example after current `predict_topk` loading/filtering:

| Forward | logRMSE vs input `I` | logRMSE vs interpolated `I_clean` |
|---|---:|---:|
| A. model params + model global | `2.498` | `2.501` |
| B. model params + true global | `1.835` | `1.838` |
| C. true params + model global | `0.647` | `0.646` |
| D. true params + true global | `0.0584` | `3.33e-08` |

Interpretation:

- The forward model and true label are good.
- The model global head is not accurate for this example: true params + model global gives `0.65` logRMSE.
- The model parameter head/type uncertainty is worse: model params + true global still gives `1.84` logRMSE.
- The mean candidate predicts the correct broad type family (`cylinder`) but wrong enough parameters/global to fail the physics verification.

Mean predicted component:

```text
type = cylinder
weight = 1.0
R = 73.57
sigma_R = 45.66
h = 116.46
sigma_h = 42.33
D = 20.62
sigma_D = 3.06
```

True component:

```text
type = cylinder
weight = 1.0
R = 97.19
sigma_R = 51.84
h = 15.81
sigma_h = 5.20
D = 30.59
sigma_D = 16.76
```

The biggest visible parameter errors are `h`, `sigma_h`, and `sigma_D`.

## 7. Mean Candidate vs Sampling Candidate

Diagnostic file:

`Training/mean_candidate_debug.json`

Mean candidate:

```text
source = mean
logRMSE = 2.498
weighted_log_chi2 = 4547
relative_rmse = 0.898
```

200-sample posterior probe with `sampling_std=0.005`:

```text
n_valid = 198
best sampled logRMSE = 1.188
median sampled logRMSE = 2.178
```

In the 10-sample smoke run through `predict_topk.py`, TOP results had `best_source = sample`, not mean. That means sampling can improve over the deterministic mean, but the posterior neighborhood is still far from the true solution.

`sampling_std=0.005` is not obviously too large; even a very narrow sampling run cannot reach the oracle. `--use_predicted_logstd` may be risky if logstd heads are broad, but it was not used in this audit.

## 8. Training Status

Model directory logs exist:

- `history.json`
- `step_history.json`
- `step_history.csv`
- TensorBoard event files in `logs/`
- `loss_curve.png`

Latest epoch-level history entry:

```text
epoch = 76
train total_loss = -5.983
train exist_loss = 0.0143
train type_loss = 0.0663
train param_loss = -2.262
train weight_loss = 0.00308
train global_loss = -1.541
train slot_type_accuracy = 0.983
train nonempty_type_accuracy = 0.941

val total_loss = -5.505
val exist_loss = 0.0257
val type_loss = 0.0844
val param_loss = -2.047
val weight_loss = 0.00467
val global_loss = -1.523
val slot_type_accuracy = 0.981
val nonempty_type_accuracy = 0.936
```

Latest step history:

```text
global_step = 107900
epoch = 77
step = 968
train total_loss = -5.990
slot_type_accuracy = 0.965
nonempty_type_accuracy = 0.875
```

Important limitation:

- Formal training logs report NLL-style losses and type accuracies.
- They do not report direct physical-curve logRMSE, direct normalized param MAE, or global MAE on validation.
- Therefore “loss looks good” does not prove the model can produce physics-verified curves.

Overfit debug status:

```text
Training/overfit_debug_simple_loss final epoch 100:
type_accuracy = 1.0
param_mae = 0.0457
global_mae = 0.1417
oracle_curve_from_true_label_log_rmse ~= 5.7e-07
pred_params_true_global_curve_logRMSE for sample_000 ~= 1.03
pred_params_pred_global_curve_logRMSE for sample_000 ~= 1.34
```

Even in simple K=1 debug, global and parameter heads are not yet memorizing perfectly. That supports the conclusion that model/training dynamics are still a major bottleneck.

## 9. Minimal Reproduction Commands

### A. Oracle Check

```bash
/data/dust/user/zhaiyufe/conda/envs/tf/bin/python -u Training/debug_sample_fields.py \
  --train_dir /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1/train \
  --sample_index 0 \
  --seed 42
```

Batch physics-chain check:

```bash
/data/dust/user/zhaiyufe/conda/envs/tf/bin/python -u Training/validate_dataset_physics_chain.py \
  --train_dir /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1/train \
  --max_samples 32 \
  --logrmse_threshold 1e-5
```

### B. Overfit Debug

```bash
/data/dust/user/zhaiyufe/conda/envs/tf/bin/python -u Training/overfit_debug.py \
  --train_dir /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1/train \
  --num_samples 64 \
  --epochs 100 \
  --batch_size 8 \
  --eval_batch_size 1 \
  --print_every 5 \
  --noise_frac 0.01 \
  --debug_max_points 256 \
  --simple_k1_loss \
  --plot_n 16 \
  --skip_model_save \
  --use_true_global_for_plot \
  --output_dir /home/zhaiyufe/PycharmProjects/ML_Fitting_1D_GISAXS/Training/overfit_debug_simple_loss
```

### C. predict_topk Debug

```bash
python Training/predict_topk.py \
  --model_dir /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS_K1 \
  --input_csv /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1/inspection/example_curve.csv \
  --output_dir /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS/prediction_logrmse \
  --num_samples 5000 \
  --top_k 20 \
  --score_mode unweighted_log \
  --sampling_std 0.005 \
  --include_mean_candidate \
  --q_min 0.001 \
  --q_max 2.0 \
  --progress_interval 100 \
  --allow_unsafe_lambda
```

For strict K=1 debug, also try:

```bash
  --exact_nonempty 1
```

## 10. Final Conclusion

Most likely cause of the current bad result:

1. **The model is not predicting accurate physical parameters/global for this K=1 inspection example.** The true-label forward closes, but model mean candidate is far away.
2. **`predict_topk.py` has a preprocessing mismatch due to low-intensity percentile filtering.** This should be made optional or disabled by default for synthetic inspection examples.
3. **The inactive param mask bug is no longer the main blocker for this example.** It was real, but oracle closure is now good.

Is it model failure or predict_topk failure?

- The evidence is mixed, but weighted toward **model/training quality** plus one **predict preprocessing mismatch**.
- It is not primarily a `physics_adapter.py` forward bug.
- It is not primarily a shape mismatch.
- It is not a K=1 weight-softmax issue for the deterministic mean candidate, because active-only K=1 weight is 1.

Recommended next modification:

1. In `Training/predict_topk.py`, make the low-intensity percentile filter optional, e.g. `--drop_low_intensity_floor`, default off. Keep zero/negative filtering only.
2. Add validation-time physics metrics to `Training/train.py`: mean-candidate curve logRMSE on a small fixed validation subset with stored true labels, plus direct param/global MAE.
3. For K=1 prediction, use `--exact_nonempty 1` during debug so posterior sampling does not occasionally create extra active slots.
4. Investigate model/training after the preprocessing change: simple K=1 overfit should reach much lower `param_mae/global_mae` before trusting TOP-K physics verification.

Do not modify next:

- `TrainSetBuild/physics_adapter.py`: true label forward already reproduces `I_clean`.
- Core permutation loss structure in `Training/losses.py`: param loss already uses `slot_param_mask`.
- Dataset physics generation solely to fix this example: the inspection sample itself is valid and oracle-closed.
