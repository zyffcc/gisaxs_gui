"""Posterior sampling plus physics verification for TOP-K component candidates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    NUMPY_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    np = None
    NUMPY_IMPORT_ERROR = exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ML_FITTING_ROOT = Path(__file__).resolve().parent / "ML_Fitting_1D_GISAXS"
if ML_FITTING_ROOT.exists():
    sys.path.insert(0, str(ML_FITTING_ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_AVAILABLE = True
except ImportError:
    plt = None
    PLOT_AVAILABLE = False

try:
    from TrainSetBuild import constraints, schema
    from TrainSetBuild.physics_adapter import component_array_to_dict, evaluate_clean, global_array_to_dict
    from TrainSetBuild.sampling import pad_2d, preprocess_curve
    DEPENDENCY_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    constraints = schema = component_array_to_dict = evaluate_clean = global_array_to_dict = None
    pad_2d = preprocess_curve = SlotQueryBase = None
    DEPENDENCY_IMPORT_ERROR = exc

try:
    from Training.model import SlotQueryBase
    TRAINING_MODEL_IMPORT_ERROR = None
except Exception as exc:
    SlotQueryBase = None
    TRAINING_MODEL_IMPORT_ERROR = exc
from utils.ai_fitting_models import load_tensorflow_model_compatible


def _install_standalone_backend():
    """Install minimal inference helpers when the training package is absent."""
    global constraints, schema, component_array_to_dict, global_array_to_dict
    global pad_2d, preprocess_curve, SlotQueryBase, evaluate_clean

    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        tf = None
    globals()["tf"] = tf
    globals()["TYPE_MASK_LOGIT"] = -1.0e9
    globals()["FORCE_EXIST_LOGIT"] = 20.0
    globals()["FORCE_EMPTY_LOGIT"] = -20.0

    class StandaloneSchema:
        MAX_POINTS = 1000
        MAX_SLOTS = 4
        NUM_TYPES = 4
        TYPE_EMPTY = 0
        TYPE_SPHERE = 1
        TYPE_CYLINDER = 2
        TYPE_VERTICAL_CYLINDER = 3
        TYPE_CYLINDER_VERTICAL = 3
        TYPE_NAMES = ["empty", "sphere", "cylinder", "vertical_cylinder"]
        PARAM_NAMES = ["R", "sigma_R", "h", "sigma_h", "D", "sigma_D"]
        GLOBAL_PARAM_NAMES = ["background", "sigma_res", "nu_res", "int_res", "k"]
        P_MAX = 6
        G_MAX = 5
        PARAM_NORM_RANGES = {
            "R": (0.05, 120.0),
            "sigma_R": (0.0, 1.0),
            "h": (0.05, 300.0),
            "sigma_h": (0.0, 1.0),
            "D": (0.0, 400.0),
            "sigma_D": (0.0, 1.0),
        }
        GLOBAL_NORM_RANGES = {
            "background": (0.0, 1.0),
            "sigma_res": (0.0, 1.0),
            "nu_res": (0.0, 20.0),
            "int_res": (0.0, 1.0),
            "k": (0.0, 10.0),
        }

        @classmethod
        def normalize_value(cls, value, value_range):
            lo, hi = value_range
            return float(np.clip((float(value) - lo) / max(hi - lo, 1e-12), 0.0, 1.0))

        @classmethod
        def denormalize_value(cls, value, value_range):
            lo, hi = value_range
            return float(lo + np.clip(float(value), 0.0, 1.0) * (hi - lo))

        @classmethod
        def type_param_mask(cls, tid):
            mask = np.zeros(cls.P_MAX, dtype=np.float64)
            if int(tid) == cls.TYPE_CYLINDER:
                mask[:] = 1.0
            elif int(tid) in (cls.TYPE_SPHERE, cls.TYPE_VERTICAL_CYLINDER):
                mask[[0, 1, 4, 5]] = 1.0
            return mask

        @classmethod
        def denormalize_params(cls, params_norm, tid):
            params_norm = np.asarray(params_norm, dtype=np.float64)
            return np.asarray(
                [
                    cls.denormalize_value(params_norm[i], cls.PARAM_NORM_RANGES[name])
                    for i, name in enumerate(cls.PARAM_NAMES)
                ],
                dtype=np.float64,
            )

        @classmethod
        def apply_type_param_mask(cls, params_phys, tid):
            return np.asarray(params_phys, dtype=np.float64) * cls.type_param_mask(tid)

        @classmethod
        def normalize_global(cls, global_phys):
            global_phys = np.asarray(global_phys, dtype=np.float64)
            return np.asarray(
                [
                    cls.normalize_value(global_phys[i], cls.GLOBAL_NORM_RANGES[name])
                    for i, name in enumerate(cls.GLOBAL_PARAM_NAMES)
                ],
                dtype=np.float64,
            )

        @classmethod
        def denormalize_global_with_optional_zero(cls, global_norm):
            global_norm = np.asarray(global_norm, dtype=np.float64)
            return np.asarray(
                [
                    cls.denormalize_value(global_norm[i], cls.GLOBAL_NORM_RANGES[name])
                    for i, name in enumerate(cls.GLOBAL_PARAM_NAMES)
                ],
                dtype=np.float64,
            )

    class StandaloneConstraints:
        @staticmethod
        def from_json_dict(config=None):
            s = StandaloneSchema
            return {
                "type_allowed": np.ones((s.MAX_SLOTS, s.NUM_TYPES), dtype=np.float32),
                "param_low_norm": np.zeros((s.MAX_SLOTS, s.NUM_TYPES, s.P_MAX), dtype=np.float32),
                "param_high_norm": np.ones((s.MAX_SLOTS, s.NUM_TYPES, s.P_MAX), dtype=np.float32),
                "param_range_mask": np.zeros((s.MAX_SLOTS, s.NUM_TYPES, s.P_MAX), dtype=np.float32),
                "force_exist": np.full((s.MAX_SLOTS,), -1.0, dtype=np.float32),
                "global_low_norm": np.zeros((s.G_MAX,), dtype=np.float32),
                "global_high_norm": np.ones((s.G_MAX,), dtype=np.float32),
                "global_range_mask": np.zeros((s.G_MAX,), dtype=np.float32),
            }

    def standalone_pad_2d(x, rows, cols):
        out = np.zeros((int(rows), int(cols)), dtype=np.float32)
        arr = np.asarray(x, dtype=np.float32)
        r = min(out.shape[0], arr.shape[0])
        c = min(out.shape[1], arr.shape[1])
        out[:r, :c] = arr[:r, :c]
        return out

    def standalone_preprocess_curve(q, I, sigma_arr):
        q = np.asarray(q, dtype=np.float64)
        I = np.asarray(I, dtype=np.float64)
        sigma_arr = np.asarray(sigma_arr, dtype=np.float64)
        log_q = np.log10(np.maximum(q, 1e-30))
        log_i = np.log10(np.maximum(I, 1e-30))
        q_span = max(float(log_q.max() - log_q.min()), 1e-12)
        q_norm = (log_q - float(log_q.min())) / q_span
        i_center = float(np.median(log_i))
        i_scale = float(np.percentile(log_i, 95) - np.percentile(log_i, 5))
        i_scale = max(i_scale, 1e-12)
        i_norm = (log_i - i_center) / i_scale
        sigma_rel = np.clip(sigma_arr / np.maximum(I, 1e-30), 1e-6, 1e6)
        sigma_feat = np.log10(sigma_rel)
        x = np.stack([q_norm, i_norm, sigma_feat], axis=1).astype(np.float32)
        global_features = np.asarray(
            [log_q.min(), log_q.max(), log_i.min(), log_i.max(), len(q) / StandaloneSchema.MAX_POINTS],
            dtype=np.float32,
        )
        return x, global_features

    def standalone_component_array_to_dict(tid, params_phys, weight):
        tid = int(tid)
        return {
            "type_id": tid,
            "type_name": StandaloneSchema.TYPE_NAMES[tid] if 0 <= tid < len(StandaloneSchema.TYPE_NAMES) else str(tid),
            "params_phys": np.asarray(params_phys, dtype=np.float64),
            "weight": float(weight),
        }

    def standalone_global_array_to_dict(global_phys):
        return {
            name: float(np.asarray(global_phys, dtype=np.float64)[i])
            for i, name in enumerate(StandaloneSchema.GLOBAL_PARAM_NAMES)
        }

    try:
        keras_layer_base = tf.keras.layers.Layer if tf is not None else None
    except Exception:
        keras_layer_base = None

    if keras_layer_base is not None:
        class StandaloneSlotQueryBase(keras_layer_base):
            def __init__(self, max_slots=4, dim=128, **kwargs):
                super().__init__(**kwargs)
                self.max_slots = int(max_slots)
                self.dim = int(dim)

            def build(self, input_shape):
                self.query = self.add_weight(
                    name="query",
                    shape=(self.max_slots, self.dim),
                    initializer="glorot_uniform",
                    trainable=True,
                )
                super().build(input_shape)

            def call(self, inputs):
                batch = tf.shape(inputs)[0]
                return tf.tile(tf.expand_dims(self.query, 0), [batch, 1, 1])

            def get_config(self):
                cfg = super().get_config()
                cfg.update({"max_slots": self.max_slots, "dim": self.dim})
                return cfg
    else:
        StandaloneSlotQueryBase = None

    schema = StandaloneSchema
    constraints = StandaloneConstraints
    pad_2d = standalone_pad_2d
    preprocess_curve = standalone_preprocess_curve
    component_array_to_dict = standalone_component_array_to_dict
    global_array_to_dict = standalone_global_array_to_dict
    SlotQueryBase = StandaloneSlotQueryBase
    evaluate_clean = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--constraints_json")
    p.add_argument("--q_min", type=float)
    p.add_argument("--q_max", type=float)
    p.add_argument("--drop_low_intensity_floor", action="store_true")
    p.add_argument("--low_intensity_floor_percentile", type=float, default=0.5)
    p.add_argument("--low_intensity_floor_factor", type=float, default=5.0)
    p.add_argument("--drop_outliers", action="store_true")
    p.add_argument("--outlier_sigma", type=float, default=6.0)
    p.add_argument("--outlier_max_run", type=int, default=10, help="Maximum consecutive outlier points to treat as a local bad gap.")
    p.add_argument("--chi2_threshold", type=float, default=2.0)
    p.add_argument("--log_rmse_threshold", type=float, default=0.25)
    p.add_argument("--score_mode", choices=["unweighted_log", "weighted_log", "robust_log"], default="unweighted_log")
    p.add_argument("--robust_loss", choices=["huber", "cauchy"], default="huber")
    p.add_argument("--robust_f_scale", type=float, default=0.3)
    p.add_argument("--sampling_std", type=float, default=0.03, help="Default normalized posterior sampling std.")
    p.add_argument("--use_predicted_logstd", action="store_true", help="Use model log-std heads instead of --sampling_std.")
    p.add_argument("--include_mean_candidate", action="store_true", help="Verify and rank the deterministic posterior mean candidate.")
    p.add_argument(
        "--refine_top_n",
        type=int,
        default=0,
        help="Run scipy least_squares log-residual refinement on the first N ranked candidates.",
    )
    p.add_argument(
        "--refine_max_nfev",
        type=int,
        default=80,
        help="Maximum scipy least_squares function evaluations per refined candidate.",
    )
    p.add_argument(
        "--refine_progress_interval",
        type=int,
        default=10,
        help="Print refine progress every N residual evaluations; 0 disables per-candidate progress.",
    )
    p.add_argument("--refine_ftol", type=float, default=1e-8, help="SciPy least_squares ftol; <=0 disables.")
    p.add_argument("--refine_xtol", type=float, default=1e-8, help="SciPy least_squares xtol; <=0 disables.")
    p.add_argument("--refine_gtol", type=float, default=1e-8, help="SciPy least_squares gtol; <=0 disables.")
    p.add_argument(
        "--refine_target_logrmse",
        type=float,
        default=0.0,
        help="Stop refinement early when best logRMSE reaches this target; <=0 disables.",
    )
    p.add_argument(
        "--refine_stall_patience",
        type=int,
        default=80,
        help="Stop refinement after this many residual calls without clear logRMSE improvement; <=0 disables.",
    )
    p.add_argument(
        "--refine_stall_tol",
        type=float,
        default=1e-4,
        help="Minimum logRMSE improvement needed to reset refine stall patience.",
    )
    p.add_argument(
        "--exact_nonempty",
        type=int,
        default=None,
        help="Require exactly K non-empty components per sampled candidate (e.g., 1 for single-component fits).",
    )
    p.add_argument("--progress_interval", type=int, default=100, help="Print sampling progress every N posterior samples; 0 disables.")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--allow_unsafe_lambda",
        action="store_true",
        help="Allow Keras Lambda layer deserialization (safe_mode=False) for trusted models.",
    )
    return p.parse_args()


def split_data_line(line: str):
    line = line.strip()
    if "\t" in line:
        return [part.strip() for part in line.split("\t") if part.strip()]
    if "," in line:
        return [part.strip() for part in line.split(",") if part.strip()]
    return line.split()


def token_is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def normalize_col_name(name: str) -> str:
    return name.strip().lower().lstrip("#").replace("(", "").replace(")", "").replace("[", "").replace("]", "")


def find_column(names, aliases, default_idx):
    normalized = [normalize_col_name(n) for n in names]
    for alias in aliases:
        alias_norm = normalize_col_name(alias)
        for idx, name in enumerate(normalized):
            if name == alias_norm or alias_norm in name:
                return idx
    return default_idx


def load_numeric_table(path: Path):
    data_rows = []
    header = None
    comment_header = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                candidate = stripped.lstrip("#").strip()
                if candidate:
                    tokens = split_data_line(candidate)
                    if len(tokens) >= 2 and not all(token_is_float(t) for t in tokens[:2]):
                        comment_header = tokens
                continue
            tokens = split_data_line(stripped)
            if len(tokens) < 2:
                continue
            if all(token_is_float(t) for t in tokens[: min(3, len(tokens))]):
                data_rows.append([float(t) for t in tokens])
            elif header is None:
                header = tokens

    if not data_rows:
        raise ValueError(f"No numeric q/I rows found in {path}")
    width = min(len(row) for row in data_rows)
    if width < 2:
        raise ValueError(f"Need at least two numeric columns q and I in {path}")
    arr = np.asarray([row[:width] for row in data_rows], dtype=np.float64)
    names = header or comment_header
    if names is not None and len(names) < width:
        names = None
    return arr, names


def load_curve(
    path: Path,
    drop_low_intensity_floor=False,
    low_intensity_floor_percentile=0.5,
    low_intensity_floor_factor=5.0,
):
    arr, names = load_numeric_table(path)
    original_n = int(arr.shape[0])
    if names:
        q_idx = find_column(names, ["q", "q_nm", "q_nm^-1", "q_1/nm", "x"], 0)
        i_idx = find_column(names, ["I", "intensity", "counts", "y"], 1)
        sigma_idx = find_column(names, ["sigma", "err", "error", "uncertainty", "dI"], -1)
    else:
        q_idx, i_idx, sigma_idx = 0, 1, 2 if arr.shape[1] >= 3 else -1
    if q_idx >= arr.shape[1] or i_idx >= arr.shape[1]:
        raise ValueError(f"Could not resolve q/I columns in {path}; names={names}")
    q = np.asarray(arr[:, q_idx], dtype=np.float64)
    I = np.asarray(arr[:, i_idx], dtype=np.float64)
    if sigma_idx >= 0 and sigma_idx < arr.shape[1] and sigma_idx not in (q_idx, i_idx):
        sigma_arr = np.asarray(arr[:, sigma_idx], dtype=np.float64)
    else:
        sigma_arr = np.maximum(0.05 * np.maximum(I, 1e-30), 1e-30)
    order = np.argsort(q)
    q, I, sigma_arr = q[order], I[order], sigma_arr[order]
    keep = np.isfinite(q) & np.isfinite(I) & np.isfinite(sigma_arr) & (q > 0) & (I > 0) & (sigma_arr > 0)
    finite_positive_n = int(np.sum(keep))
    floor_removed = 0
    floor = None
    if drop_low_intensity_floor:
        positive = I[keep]
        if positive.size > 0:
            floor = float(np.percentile(positive, float(low_intensity_floor_percentile)))
            if np.isfinite(floor) and floor > 0:
                before = int(np.sum(keep))
                keep = keep & (I > floor * float(low_intensity_floor_factor))
                floor_removed = before - int(np.sum(keep))
                if floor_removed:
                    print(
                        f"Low-intensity floor removed {floor_removed} points "
                        f"(percentile={low_intensity_floor_percentile}, factor={low_intensity_floor_factor}, floor={floor:.4g}).",
                        flush=True,
                    )
    if keep.sum() < 16:
        raise ValueError("Input curve has too few finite positive points.")
    debug = {
        "original_n_points": original_n,
        "after_finite_positive_n_points": finite_positive_n,
        "drop_low_intensity_floor": bool(drop_low_intensity_floor),
        "low_intensity_floor_percentile": float(low_intensity_floor_percentile),
        "low_intensity_floor_factor": float(low_intensity_floor_factor),
        "low_intensity_floor_value": None if floor is None else float(floor),
        "low_intensity_floor_removed_n_points": int(floor_removed),
        "after_low_intensity_floor_n_points": int(np.sum(keep)),
    }
    return q[keep], I[keep], sigma_arr[keep], debug


def load_model(model_dir: Path, allow_unsafe_lambda: bool = False):
    custom_objects = {
        "TYPE_MASK_LOGIT": -1.0e9,
        "FORCE_EXIST_LOGIT": 20.0,
        "FORCE_EMPTY_LOGIT": -20.0,
    }
    if SlotQueryBase is not None:
        custom_objects["SlotQueryBase"] = SlotQueryBase
    elif TRAINING_MODEL_IMPORT_ERROR is not None:
        print(
            "Training.model could not be imported; model loading will use SavedModel signature fallback if needed. "
            f"Reason: {type(TRAINING_MODEL_IMPORT_ERROR).__name__}: {TRAINING_MODEL_IMPORT_ERROR}",
            flush=True,
        )
    model, _artifact = load_tensorflow_model_compatible(
        model_dir,
        custom_objects=custom_objects,
        allow_unsafe_lambda=allow_unsafe_lambda,
    )
    return model


def make_input(q, I, sigma_arr, cons):
    x, global_features = preprocess_curve(q, I, sigma_arr)
    n = min(len(q), schema.MAX_POINTS)
    mask = np.zeros(schema.MAX_POINTS, dtype=bool)
    mask[:n] = True
    batch = {
        "x": pad_2d(x[:n], schema.MAX_POINTS, 3)[None, ...],
        "point_mask": mask[None, ...],
        "global_features": global_features[None, ...],
    }
    for key, val in cons.items():
        batch[key] = val[None, ...]
    return batch


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    expv = np.exp(logits)
    return expv / np.maximum(expv.sum(), 1e-300)


def sigmoid_stable(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-np.minimum(x[pos], 80.0)))
    exp_x = np.exp(np.maximum(x[~pos], -80.0))
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def downsample_curve(q, I, sigma_arr):
    if len(q) <= schema.MAX_POINTS:
        return q, I, sigma_arr
    idx = np.linspace(0, len(q) - 1, schema.MAX_POINTS).astype(int)
    return q[idx], I[idx], sigma_arr[idx]


def apply_q_range(q, I, sigma_arr, q_min=None, q_max=None):
    keep = np.ones_like(q, dtype=bool)
    if q_min is not None:
        keep &= q >= float(q_min)
    if q_max is not None:
        keep &= q <= float(q_max)
    removed = int(np.sum(~keep))
    if np.sum(keep) < 16:
        raise ValueError(f"q range leaves too few points: kept {np.sum(keep)}, removed {removed}")
    if removed:
        print(f"q range mask removed {removed} points; kept {np.sum(keep)}", flush=True)
    return q[keep], I[keep], sigma_arr[keep]


def rolling_median(values, window_size):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        return values.copy()
    window_size = int(window_size)
    window_size = max(3, min(window_size, values.size if values.size % 2 == 1 else values.size - 1))
    if window_size % 2 == 0:
        window_size -= 1
    pad = window_size // 2
    padded = np.pad(values, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_size)
    return np.median(windows, axis=1)


def short_true_runs(mask, max_run=2):
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    idx = 0
    while idx < mask.size:
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < mask.size and mask[idx]:
            idx += 1
        if idx - start <= max_run:
            out[start:idx] = True
    return out


def drop_log_outliers(q, I, sigma_arr, outlier_sigma=6.0, window_size=21, max_run=10):
    if len(q) < 16:
        return q, I, sigma_arr, np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    window_size = min(31, max(11, int(window_size)))
    if window_size % 2 == 0:
        window_size += 1
    log_i = np.log(np.maximum(I, 1e-30))
    local_median = rolling_median(log_i, window_size)
    residual = log_i - local_median
    center = np.median(residual)
    mad = np.median(np.abs(residual - center))
    robust_sigma = 1.4826 * mad
    if not np.isfinite(robust_sigma) or robust_sigma <= 1e-12:
        print("Outlier filtering skipped: MAD is too small.", flush=True)
        return q, I, sigma_arr, np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    raw_outlier = np.abs(residual) > float(outlier_sigma) * robust_sigma
    isolated_outlier = short_true_runs(raw_outlier, max_run=max_run)
    keep = ~isolated_outlier
    if np.sum(keep) < 16:
        print("Outlier filtering skipped: it would leave too few points.", flush=True)
        return q, I, sigma_arr, np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    removed = int(np.sum(isolated_outlier))
    broad = int(np.sum(raw_outlier & ~isolated_outlier))
    print(
        f"Outlier filtering removed {removed} isolated points "
        f"(raw outliers={int(np.sum(raw_outlier))}, broad-run kept={broad}, "
        f"window={window_size}, max_run={max_run}, MAD={mad:.4g}).",
        flush=True,
    )
    return q[keep], I[keep], sigma_arr[keep], q[isolated_outlier], I[isolated_outlier]


def effective_range(low, high, mask):
    low_eff = np.where(mask > 0.0, low, 0.0)
    high_eff = np.where(mask > 0.0, high, 1.0)
    return low_eff, high_eff


def sample_candidate(
    pred,
    cons,
    rng,
    sampling_std=0.03,
    use_predicted_logstd=False,
    exact_nonempty=None,
    return_reason=False,
):
    exist_prob = sigmoid_stable(pred["exist_logit"][0])
    type_logits = pred["type_logits"][0]
    param_mu = pred["param_mu_norm"][0]
    param_logstd = np.clip(pred["param_logstd_raw"][0], -5.0, 1.0)
    weight_logits = pred["weight_logit"][0]
    global_mu = pred["global_mu_norm"][0]
    global_logstd = np.clip(pred["global_logstd_raw"][0], -5.0, 1.0)

    active = []
    for j in range(schema.MAX_SLOTS):
        force = float(cons["force_exist"][j])
        if force == 1.0:
            exists = True
        elif force == 0.0:
            exists = False
        else:
            exists = bool(rng.random() < exist_prob[j])
        if not exists:
            continue
        probs = softmax(type_logits[j])
        probs[schema.TYPE_EMPTY] = 0.0
        probs = probs / np.maximum(probs.sum(), 1e-300)
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            continue
        tid = int(rng.choice(np.arange(schema.NUM_TYPES), p=probs))
        if tid == schema.TYPE_EMPTY:
            continue
        mu = param_mu[j, tid]
        std = np.exp(param_logstd[j, tid]) if use_predicted_logstd else np.full_like(mu, sampling_std)
        low_eff, high_eff = effective_range(
            cons["param_low_norm"][j, tid],
            cons["param_high_norm"][j, tid],
            cons["param_range_mask"][j, tid],
        )
        params_norm = np.clip(rng.normal(mu, std), low_eff, high_eff)
        params_phys = schema.denormalize_params(params_norm, tid)
        params_phys = schema.apply_type_param_mask(params_phys, tid)
        active.append({"slot": j, "type_id": tid, "params_phys": params_phys})

    if not active:
        return (None, "empty") if return_reason else None
    if exact_nonempty is not None:
        target = int(exact_nonempty)
        if len(active) != target:
            return (None, "exact_nonempty") if return_reason else None
    # Weight softmax is evaluated only for slots that sampled a non-empty component.
    active_logits = np.array([weight_logits[a["slot"]] for a in active], dtype=np.float64)
    weights = softmax(active_logits)
    components = [
        component_array_to_dict(a["type_id"], a["params_phys"], float(weights[i]))
        for i, a in enumerate(active)
    ]
    global_std = np.exp(global_logstd) if use_predicted_logstd else np.full_like(global_mu, sampling_std)
    global_low_eff, global_high_eff = effective_range(cons["global_low_norm"], cons["global_high_norm"], cons["global_range_mask"])
    global_norm = np.clip(rng.normal(global_mu, global_std), global_low_eff, global_high_eff)
    global_phys = schema.denormalize_global_with_optional_zero(global_norm)
    candidate = (components, global_phys)
    return (candidate, None) if return_reason else candidate


def mean_candidate(pred, cons, exact_nonempty=None, return_reason=False):
    exist_prob = sigmoid_stable(pred["exist_logit"][0])
    type_logits = pred["type_logits"][0]
    param_mu = pred["param_mu_norm"][0]
    weight_logits = pred["weight_logit"][0]
    global_mu = pred["global_mu_norm"][0]

    active = []
    for j in range(schema.MAX_SLOTS):
        force = float(cons["force_exist"][j])
        if force == 1.0:
            exists = True
        elif force == 0.0:
            exists = False
        else:
            exists = bool(exist_prob[j] >= 0.5)
        if not exists:
            continue

        scores = np.asarray(type_logits[j], dtype=np.float64).copy()
        scores[schema.TYPE_EMPTY] = -np.inf
        tid = int(np.argmax(scores))
        if tid == schema.TYPE_EMPTY or not np.isfinite(scores[tid]):
            continue

        params_norm = np.asarray(param_mu[j, tid], dtype=np.float64)
        params_phys = schema.denormalize_params(params_norm, tid)
        params_phys = schema.apply_type_param_mask(params_phys, tid)
        active.append({"slot": j, "type_id": tid, "params_phys": params_phys})

    if not active:
        return (None, "empty") if return_reason else None
    if exact_nonempty is not None:
        target = int(exact_nonempty)
        if len(active) != target:
            return (None, "exact_nonempty") if return_reason else None

    active_logits = np.array([weight_logits[a["slot"]] for a in active], dtype=np.float64)
    weights = softmax(active_logits)
    components = [
        component_array_to_dict(a["type_id"], a["params_phys"], float(weights[i]))
        for i, a in enumerate(active)
    ]
    global_phys = schema.denormalize_global_with_optional_zero(global_mu)
    candidate = (components, global_phys)
    return (candidate, None) if return_reason else candidate


def component_params_json(comp):
    tid = int(comp["type_id"])
    p = np.asarray(comp["params_phys"], dtype=float)
    names = ["R", "sigma_R", "D", "sigma_D"] if tid != schema.TYPE_CYLINDER else schema.PARAM_NAMES
    idxs = [0, 1, 4, 5] if tid != schema.TYPE_CYLINDER else list(range(schema.P_MAX))
    return {name: float(p[i]) for name, i in zip(names, idxs)}


def combination_key(components):
    return "+".join(sorted([c["type_name"] for c in components]))


def sorted_components(components):
    return sorted(components, key=lambda c: (int(c["type_id"]), float(c["weight"])))


def quantile_summary(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "p16": float(np.percentile(arr, 16)),
        "p84": float(np.percentile(arr, 84)),
    }


def posterior_stats(items):
    if not items:
        return {}
    best_sorted = sorted_components(items[0]["components"])
    comp_stats = []
    for comp_idx, comp in enumerate(best_sorted):
        tid = int(comp["type_id"])
        names = ["R", "sigma_R", "D", "sigma_D"] if tid != schema.TYPE_CYLINDER else schema.PARAM_NAMES
        idxs = [0, 1, 4, 5] if tid != schema.TYPE_CYLINDER else list(range(schema.P_MAX))
        same_position = [sorted_components(item["components"])[comp_idx] for item in items if len(item["components"]) > comp_idx]
        stats = {
            "type": comp["type_name"],
            "weight": quantile_summary([c["weight"] for c in same_position]),
            "params": {},
        }
        for name, pidx in zip(names, idxs):
            stats["params"][name] = quantile_summary([np.asarray(c["params_phys"])[pidx] for c in same_position])
        comp_stats.append(stats)
    global_stats = {
        name: quantile_summary([item["global_phys"][i] for item in items])
        for i, name in enumerate(schema.GLOBAL_PARAM_NAMES)
    }
    return {"components": comp_stats, "global_params": global_stats}


def robust_log_score(log_residual, robust_loss="huber", f_scale=0.3):
    f_scale = max(float(f_scale), 1e-12)
    r = np.asarray(log_residual, dtype=np.float64)
    if robust_loss == "cauchy":
        return float(np.mean(np.log1p((r / f_scale) ** 2)))
    if robust_loss == "huber":
        abs_r = np.abs(r)
        quadratic = abs_r <= f_scale
        loss = np.where(quadratic, 0.5 * r**2, f_scale * (abs_r - 0.5 * f_scale))
        return float(np.mean(loss))
    raise ValueError(f"Unsupported robust_loss: {robust_loss}")


def fit_metrics(I_fit, I_exp, sigma_log, robust_loss="huber", robust_f_scale=0.3, eps=1e-30):
    log_i_exp = np.log10(np.maximum(I_exp, eps))
    log_i_fit = np.log10(np.maximum(I_fit, eps))
    log_residual = log_i_fit - log_i_exp
    linear_residual = I_fit - I_exp
    log_rmse = float(np.sqrt(np.mean(log_residual**2)))
    weighted_log_chi2 = float(np.mean((log_residual / sigma_log) ** 2))
    relative_rmse = float(np.sqrt(np.mean((linear_residual / np.maximum(I_exp, eps)) ** 2)))
    linear_rmse = float(np.sqrt(np.mean(linear_residual**2)))
    robust_log = robust_log_score(log_residual, robust_loss=robust_loss, f_scale=robust_f_scale)
    return {
        "log_rmse": log_rmse,
        "weighted_log_chi2": weighted_log_chi2,
        "robust_log": robust_log,
        "relative_rmse": relative_rmse,
        "linear_rmse": linear_rmse,
        "log_residual": log_residual.astype(np.float32),
        "linear_residual": linear_residual.astype(np.float32),
    }


def score_from_metrics(metrics, score_mode):
    if score_mode == "unweighted_log":
        return float(metrics["log_rmse"])
    if score_mode == "weighted_log":
        return float(metrics["weighted_log_chi2"])
    if score_mode == "robust_log":
        return float(metrics["robust_log"])
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def score_weight(score):
    if not np.isfinite(score):
        return 0.0
    return float(np.exp(-0.5 * np.clip(score, 0.0, 1e6)))


def components_json(components):
    return [
        {
            "type": c["type_name"],
            "type_id": int(c["type_id"]),
            "weight": float(c["weight"]),
            "params": component_params_json(c),
        }
        for c in components
    ]


def candidate_refine_setup(item):
    components = item["components"]
    global_phys = np.asarray(item["global_phys"], dtype=np.float64)
    x0 = []
    lower = []
    upper = []
    comp_specs = []

    for comp in components:
        tid = int(comp["type_id"])
        params_phys = np.asarray(comp["params_phys"], dtype=np.float64)
        param_indices = [i for i, enabled in enumerate(schema.type_param_mask(tid)) if enabled > 0.5]
        comp_specs.append({"type_id": tid, "param_indices": param_indices})
        for pidx in param_indices:
            name = schema.PARAM_NAMES[pidx]
            x0.append(schema.normalize_value(float(params_phys[pidx]), schema.PARAM_NORM_RANGES[name]))
            lower.append(0.0)
            upper.append(1.0)

    global_norm = schema.normalize_global(global_phys)
    global_start = len(x0)
    x0.extend([float(v) for v in global_norm])
    lower.extend([0.0] * schema.G_MAX)
    upper.extend([1.0] * schema.G_MAX)

    weight_start = len(x0)
    if len(components) > 1:
        for comp in components:
            x0.append(float(np.log(max(float(comp.get("weight", 0.0)), 1e-12))))
            lower.append(-20.0)
            upper.append(20.0)

    setup = {
        "components": components,
        "comp_specs": comp_specs,
        "global_start": global_start,
        "weight_start": weight_start,
    }
    return np.asarray(x0, dtype=np.float64), np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64), setup


def unpack_refined_candidate(x, setup):
    x = np.asarray(x, dtype=np.float64)
    components = []
    cursor = 0
    for spec in setup["comp_specs"]:
        tid = int(spec["type_id"])
        params_norm = np.zeros(schema.P_MAX, dtype=np.float64)
        for pidx in spec["param_indices"]:
            params_norm[pidx] = x[cursor]
            cursor += 1
        params_phys = schema.denormalize_params(params_norm, tid)
        params_phys = schema.apply_type_param_mask(params_phys, tid)
        components.append({"type_id": tid, "params_phys": params_phys})

    global_start = int(setup["global_start"])
    global_norm = x[global_start : global_start + schema.G_MAX]
    global_phys = schema.denormalize_global_with_optional_zero(global_norm)

    if len(components) == 1:
        weights = np.array([1.0], dtype=np.float64)
    else:
        weight_start = int(setup["weight_start"])
        weights = softmax(x[weight_start : weight_start + len(components)])

    component_dicts = [
        component_array_to_dict(comp["type_id"], comp["params_phys"], float(weights[i]))
        for i, comp in enumerate(components)
    ]
    return component_dicts, global_phys


def refine_candidate(
    item,
    q_eval,
    I_eval,
    sigma_log,
    score_mode,
    robust_loss,
    robust_f_scale,
    max_nfev=200,
    progress_interval=10,
    progress_label="",
    refine_target_logrmse=0.0,
    refine_stall_patience=80,
    refine_stall_tol=1e-4,
    refine_ftol=1e-8,
    refine_xtol=1e-8,
    refine_gtol=1e-8,
):
    class EarlyStopRefine(Exception):
        pass

    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        return item, {
            "success": False,
            "message": f"scipy is not available: {exc}",
            "nfev": 0,
            "residual_calls": 0,
            "early_stop_reason": None,
            "best_log_rmse_seen": float(item.get("log_rmse", np.nan)),
        }

    x0, lower, upper, setup = candidate_refine_setup(item)
    log_i_exp = np.log10(np.maximum(I_eval, 1e-30))
    progress = {
        "calls": 0,
        "best_log_rmse": float(item.get("log_rmse", np.inf)),
        "best_x": np.asarray(x0, dtype=np.float64).copy(),
        "last_improve_call": 0,
        "early_stop_reason": None,
    }
    refine_stall_tol = float(refine_stall_tol)

    def residual_fn(x):
        progress["calls"] += 1
        components, global_phys = unpack_refined_candidate(x, setup)
        try:
            i_fit = evaluate_clean(q_eval, components, global_array_to_dict(global_phys))
        except Exception:
            residual = np.full_like(log_i_exp, 1e6, dtype=np.float64)
        else:
            if np.all(np.isfinite(i_fit)):
                residual = np.log10(np.maximum(i_fit, 1e-30)) - log_i_exp
            else:
                residual = np.full_like(log_i_exp, 1e6, dtype=np.float64)
        log_rmse = float(np.sqrt(np.mean(residual**2)))
        if log_rmse < progress["best_log_rmse"] - refine_stall_tol:
            progress["best_log_rmse"] = log_rmse
            progress["best_x"] = np.asarray(x, dtype=np.float64).copy()
            progress["last_improve_call"] = progress["calls"]
        if progress_interval > 0 and (
            progress["calls"] == 1 or progress["calls"] % int(progress_interval) == 0
        ):
            print(
                f"  refine{progress_label} eval={progress['calls']:04d} "
                f"current_logRMSE={log_rmse:.5g} best_logRMSE={progress['best_log_rmse']:.5g}",
                flush=True,
            )
        if refine_target_logrmse > 0.0 and progress["best_log_rmse"] <= float(refine_target_logrmse):
            progress["early_stop_reason"] = f"target_logrmse reached: {progress['best_log_rmse']:.6g}"
            raise EarlyStopRefine()
        if refine_stall_patience > 0 and progress["calls"] - progress["last_improve_call"] >= int(refine_stall_patience):
            progress["early_stop_reason"] = (
                f"stalled for {int(refine_stall_patience)} residual calls; "
                f"best_logRMSE={progress['best_log_rmse']:.6g}"
            )
            raise EarlyStopRefine()
        return residual

    try:
        try:
            result = least_squares(
                residual_fn,
                x0,
                bounds=(lower, upper),
                max_nfev=int(max_nfev),
                x_scale="jac",
                ftol=float(refine_ftol) if float(refine_ftol) > 0 else None,
                xtol=float(refine_xtol) if float(refine_xtol) > 0 else None,
                gtol=float(refine_gtol) if float(refine_gtol) > 0 else None,
            )
            x_final = result.x
            success = bool(result.success)
            message = str(result.message)
            nfev = int(result.nfev)
        except EarlyStopRefine:
            x_final = progress["best_x"]
            success = True
            message = str(progress["early_stop_reason"])
            nfev = -1

        components, global_phys = unpack_refined_candidate(x_final, setup)
        I_fit = evaluate_clean(q_eval, components, global_array_to_dict(global_phys))
        if not np.all(np.isfinite(I_fit)):
            raise ValueError("refined forward curve contains non-finite values")
        metrics = fit_metrics(
            I_fit,
            I_eval,
            sigma_log,
            robust_loss=robust_loss,
            robust_f_scale=robust_f_scale,
        )
        refined = dict(item)
        refined.update(
            {
                "source": f"{item.get('source', 'sample')}+refined",
                "score": score_from_metrics(metrics, score_mode),
                "log_rmse": metrics["log_rmse"],
                "weighted_log_chi2": metrics["weighted_log_chi2"],
                "robust_log": metrics["robust_log"],
                "relative_rmse": metrics["relative_rmse"],
                "linear_rmse": metrics["linear_rmse"],
                "log_residual": metrics["log_residual"],
                "linear_residual": metrics["linear_residual"],
                "components": components,
                "global_phys": global_phys,
                "I_fit": I_fit.astype(np.float32),
            }
        )
        return refined, {
            "success": success,
            "message": message,
            "nfev": nfev,
            "residual_calls": int(progress["calls"]),
            "early_stop_reason": progress["early_stop_reason"],
            "best_log_rmse_seen": float(progress["best_log_rmse"]),
            "initial_score": float(item["score"]),
            "final_score": float(refined["score"]),
            "initial_log_rmse": float(item["log_rmse"]),
            "final_log_rmse": float(refined["log_rmse"]),
        }
    except Exception as exc:
        return item, {
            "success": False,
            "message": str(exc),
            "nfev": 0,
            "residual_calls": int(progress["calls"]),
            "early_stop_reason": progress["early_stop_reason"],
            "best_log_rmse_seen": float(progress["best_log_rmse"]),
            "initial_score": float(item.get("score", np.nan)),
            "final_score": float(item.get("score", np.nan)),
            "initial_log_rmse": float(item.get("log_rmse", np.nan)),
            "final_log_rmse": float(item.get("log_rmse", np.nan)),
        }


def write_proposal_only_outputs(
    out_dir,
    pred,
    cons,
    rng,
    args,
    q_eval,
    I_eval,
    sigma_eval,
):
    """Write neural proposal candidates when physics verification code is unavailable."""
    groups = defaultdict(list)
    rejected_empty = 0
    rejected_exact_nonempty = 0

    def add_candidate(candidate, source):
        components, global_phys = candidate
        item = {
            "source": source,
            "components": components,
            "global_phys": global_phys,
        }
        groups[combination_key(components)].append(item)

    if args.include_mean_candidate:
        candidate, reason = mean_candidate(pred, cons, exact_nonempty=args.exact_nonempty, return_reason=True)
        if candidate is None:
            if reason == "exact_nonempty":
                rejected_exact_nonempty += 1
            else:
                rejected_empty += 1
            print(f"Mean candidate skipped in proposal-only mode: reason={reason or 'unknown'}", flush=True)
        else:
            add_candidate(candidate, "mean_proposal")

    for sample_idx in range(1, args.num_samples + 1):
        candidate, reason = sample_candidate(
            pred,
            cons,
            rng,
            sampling_std=args.sampling_std,
            use_predicted_logstd=args.use_predicted_logstd,
            exact_nonempty=args.exact_nonempty,
            return_reason=True,
        )
        if candidate is None:
            if reason == "exact_nonempty":
                rejected_exact_nonempty += 1
            else:
                rejected_empty += 1
            continue
        add_candidate(candidate, "sample_proposal")
        if args.progress_interval > 0 and sample_idx % args.progress_interval == 0:
            print(f"Progress {sample_idx}/{args.num_samples}: proposal_groups={len(groups)}", flush=True)

    if not groups:
        print("WARNING: no neural proposal candidates were produced.", flush=True)
        return

    total = sum(len(items) for items in groups.values()) or 1
    ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[: args.top_k]
    json_rows = []
    for rank, (key, items) in enumerate(ranked, start=1):
        best = items[0]
        json_rows.append(
            {
                "rank": rank,
                "combination": key,
                "mode": "proposal_only",
                "note": "Training/physics modules are unavailable; this row is a neural model proposal, not a physics-verified fit.",
                "score_mode": "proposal_frequency",
                "best_source": best["source"],
                "score_weighted_probability": len(items) / total,
                "posterior_frequency": len(items) / total,
                "chi2_weighted_probability": 0.0,
                "fit_percent": 0.0,
                "fit_percent_by_log_rmse": 0.0,
                "fit_percent_by_chi2": 0.0,
                "best_score": float("nan"),
                "best_log_rmse": float("nan"),
                "best_chi2_weighted": float("nan"),
                "best_robust_log": float("nan"),
                "best_relative_rmse": float("nan"),
                "best_linear_rmse": float("nan"),
                "rejected_empty": int(rejected_empty),
                "rejected_exact_nonempty": int(rejected_exact_nonempty),
                "components": [
                    {
                        "type": c["type_name"],
                        "weight": float(c["weight"]),
                        "params": component_params_json(c),
                    }
                    for c in best["components"]
                ],
                "global_params": {
                    name: float(best["global_phys"][i]) for i, name in enumerate(schema.GLOBAL_PARAM_NAMES)
                },
                "posterior_parameter_stats": posterior_stats(
                    [
                        {
                            "components": item["components"],
                            "global_phys": item["global_phys"],
                        }
                        for item in items
                    ]
                ),
            }
        )

    with (out_dir / "top20_candidates.json").open("w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2)
    with (out_dir / "top20_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "combination",
                "mode",
                "best_source",
                "score_weighted_probability",
                "posterior_frequency",
                "best_log_rmse",
                "best_chi2_weighted",
                "rejected_empty",
                "rejected_exact_nonempty",
                "note",
            ],
        )
        writer.writeheader()
        for row in json_rows:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})
    np.savez_compressed(out_dir / "input_curve_used.npz", q_eval=q_eval, I_eval=I_eval, sigma_eval=sigma_eval)
    print(
        f"Wrote {len(json_rows)} proposal-only candidates to {out_dir}. "
        "Physics verification/refinement was skipped because TrainSetBuild/Training are unavailable.",
        flush=True,
    )


def main():
    args = parse_args()
    if NUMPY_IMPORT_ERROR is not None:
        raise RuntimeError("numpy is required for TOP-K prediction.") from NUMPY_IMPORT_ERROR
    if DEPENDENCY_IMPORT_ERROR is not None:
        _install_standalone_backend()
        print(
            "TrainSetBuild/Training modules are unavailable; running standalone neural proposal mode. "
            "Physics verification and refinement will be skipped.",
            flush=True,
        )
    if args.exact_nonempty is not None and args.exact_nonempty < 1:
        raise ValueError("--exact_nonempty must be >= 1")
    if args.refine_top_n < 0:
        raise ValueError("--refine_top_n must be >= 0")
    if args.refine_max_nfev < 1:
        raise ValueError("--refine_max_nfev must be >= 1")
    if args.refine_progress_interval < 0:
        raise ValueError("--refine_progress_interval must be >= 0")
    if args.refine_stall_patience < 0:
        raise ValueError("--refine_stall_patience must be >= 0")
    if args.refine_stall_tol < 0:
        raise ValueError("--refine_stall_tol must be >= 0")
    rng = np.random.default_rng(args.seed)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading input curve: {args.input_csv}", flush=True)
    q, I, sigma_arr, input_debug = load_curve(
        Path(args.input_csv),
        drop_low_intensity_floor=args.drop_low_intensity_floor,
        low_intensity_floor_percentile=args.low_intensity_floor_percentile,
        low_intensity_floor_factor=args.low_intensity_floor_factor,
    )
    input_debug["after_load_n_points"] = int(len(q))
    before_q_range_n = int(len(q))
    q, I, sigma_arr = apply_q_range(q, I, sigma_arr, q_min=args.q_min, q_max=args.q_max)
    input_debug["q_min_arg"] = None if args.q_min is None else float(args.q_min)
    input_debug["q_max_arg"] = None if args.q_max is None else float(args.q_max)
    input_debug["q_range_removed_n_points"] = int(before_q_range_n - len(q))
    input_debug["after_q_range_n_points"] = int(len(q))
    q_outlier = np.array([], dtype=np.float64)
    I_outlier = np.array([], dtype=np.float64)
    before_outlier_n = int(len(q))
    if args.drop_outliers:
        q, I, sigma_arr, q_outlier, I_outlier = drop_log_outliers(
            q,
            I,
            sigma_arr,
            outlier_sigma=args.outlier_sigma,
            window_size=21,
            max_run=args.outlier_max_run,
        )
    input_debug["drop_outliers"] = bool(args.drop_outliers)
    input_debug["outlier_removed_n_points"] = int(before_outlier_n - len(q))
    input_debug["after_outlier_filter_n_points"] = int(len(q))
    q_eval, I_eval, sigma_eval = downsample_curve(q, I, sigma_arr)
    input_debug["after_downsample_n_points"] = int(len(q_eval))
    input_debug["final_q_min"] = float(q_eval.min())
    input_debug["final_q_max"] = float(q_eval.max())
    input_debug["final_I_min"] = float(I_eval.min())
    input_debug["final_I_max"] = float(I_eval.max())
    with (out_dir / "input_curve_debug.json").open("w", encoding="utf-8") as f:
        json.dump(input_debug, f, indent=2)
    np.savez_compressed(out_dir / "input_curve_used.npz", q_eval=q_eval, I_eval=I_eval, sigma_eval=sigma_eval)
    print(
        f"Curve ready: {len(q)} valid points, using {len(q_eval)} points; "
        f"q=[{q_eval.min():.4g}, {q_eval.max():.4g}], I=[{I_eval.min():.4g}, {I_eval.max():.4g}]",
        flush=True,
    )

    cons_config = None
    if args.constraints_json:
        with Path(args.constraints_json).open("r", encoding="utf-8") as f:
            cons_config = json.load(f)
        print(f"Loaded constraints: {args.constraints_json}", flush=True)
    cons = constraints.from_json_dict(cons_config)

    print(f"Loading model from: {model_dir}", flush=True)
    model = load_model(model_dir, allow_unsafe_lambda=args.allow_unsafe_lambda)
    print("Running neural network proposal pass...", flush=True)
    pred = model(make_input(q_eval, I_eval, sigma_eval, cons), training=False)
    pred = {k: v.numpy() for k, v in pred.items()}
    exist_prob = sigmoid_stable(pred["exist_logit"][0])
    type_prob = np.stack([softmax(pred["type_logits"][0, j]) for j in range(schema.MAX_SLOTS)], axis=0)
    print(
        "Model proposal summary: "
        f"exist_prob={np.array2string(exist_prob, precision=3)}, "
        f"top_types={[schema.TYPE_NAMES[int(np.argmax(type_prob[j]))] for j in range(schema.MAX_SLOTS)]}",
        flush=True,
    )

    if evaluate_clean is None:
        write_proposal_only_outputs(out_dir, pred, cons, rng, args, q_eval, I_eval, sigma_eval)
        return

    groups = defaultdict(list)
    curve_bank = []
    sigma_log = np.maximum(
        sigma_eval / (np.maximum(I_eval, 1e-30) * np.log(10.0)),
        1e-3,
    )

    print(
        f"Sampling {args.num_samples} posterior candidates and verifying with physics forward model...",
        flush=True,
    )
    rejected_empty = 0
    rejected_exact_nonempty = 0
    rejected_forward = 0
    rejected_nonfinite = 0
    kept = 0
    mean_candidate_kept = False
    mean_candidate_item = None
    mean_candidate_curve = None

    def verify_candidate(candidate, source):
        nonlocal kept, rejected_forward, rejected_nonfinite, mean_candidate_kept, mean_candidate_item, mean_candidate_curve
        components, global_phys = candidate
        try:
            I_fit = evaluate_clean(q_eval, components, global_array_to_dict(global_phys))
        except Exception:
            rejected_forward += 1
            return False
        if not np.all(np.isfinite(I_fit)):
            rejected_nonfinite += 1
            return False
        metrics = fit_metrics(
            I_fit,
            I_eval,
            sigma_log,
            robust_loss=args.robust_loss,
            robust_f_scale=args.robust_f_scale,
        )
        score = score_from_metrics(metrics, args.score_mode)
        item = {
            "source": source,
            "score": score,
            "log_rmse": metrics["log_rmse"],
            "weighted_log_chi2": metrics["weighted_log_chi2"],
            "robust_log": metrics["robust_log"],
            "relative_rmse": metrics["relative_rmse"],
            "linear_rmse": metrics["linear_rmse"],
            "log_residual": metrics["log_residual"],
            "linear_residual": metrics["linear_residual"],
            "components": components,
            "global_phys": global_phys,
            "I_fit": I_fit.astype(np.float32),
        }
        groups[combination_key(components)].append(item)
        kept += 1
        if source == "mean":
            mean_candidate_kept = True
            mean_candidate_item = item
            mean_candidate_curve = I_fit.astype(np.float32)
        return True

    if args.include_mean_candidate:
        candidate, reason = mean_candidate(pred, cons, exact_nonempty=args.exact_nonempty, return_reason=True)
        if candidate is None:
            if reason == "exact_nonempty":
                rejected_exact_nonempty += 1
            else:
                rejected_empty += 1
            print(
                f"Mean candidate skipped: reason={reason or 'unknown'}, "
                f"exact_nonempty={args.exact_nonempty}.",
                flush=True,
            )
        else:
            ok = verify_candidate(candidate, source="mean")
            status = "kept" if ok else "rejected"
            print(f"Mean candidate {status}; kept={kept}, groups={len(groups)}", flush=True)

    if mean_candidate_item is not None:
        mean_debug = {
            "source": "mean",
            "exact_nonempty": args.exact_nonempty,
            "components": components_json(mean_candidate_item["components"]),
            "global_params": {
                name: float(mean_candidate_item["global_phys"][i])
                for i, name in enumerate(schema.GLOBAL_PARAM_NAMES)
            },
            "log_rmse": float(mean_candidate_item["log_rmse"]),
            "weighted_log_chi2": float(mean_candidate_item["weighted_log_chi2"]),
            "weighted_chi2": float(mean_candidate_item["weighted_log_chi2"]),
            "robust_log": float(mean_candidate_item["robust_log"]),
            "relative_rmse": float(mean_candidate_item["relative_rmse"]),
            "linear_rmse": float(mean_candidate_item["linear_rmse"]),
            "q": q_eval.astype(float).tolist(),
            "I_exp": I_eval.astype(float).tolist(),
            "I_fit": mean_candidate_curve.astype(float).tolist(),
        }
        with (out_dir / "mean_candidate_debug.json").open("w", encoding="utf-8") as f:
            json.dump(mean_debug, f, indent=2)
        np.savez_compressed(
            out_dir / "mean_candidate_curve.npz",
            q=q_eval,
            I_exp=I_eval,
            sigma=sigma_eval,
            I_fit=mean_candidate_curve,
        )

    for sample_idx in range(1, args.num_samples + 1):
        candidate, reason = sample_candidate(
            pred,
            cons,
            rng,
            sampling_std=args.sampling_std,
            use_predicted_logstd=args.use_predicted_logstd,
            exact_nonempty=args.exact_nonempty,
            return_reason=True,
        )
        if candidate is None:
            if reason == "exact_nonempty":
                rejected_exact_nonempty += 1
            else:
                rejected_empty += 1
            if args.progress_interval > 0 and sample_idx % args.progress_interval == 0:
                print(
                    f"Progress {sample_idx}/{args.num_samples}: kept={kept}, "
                    f"groups={len(groups)}, empty={rejected_empty}, exact_nonempty={rejected_exact_nonempty}, "
                    f"forward_fail={rejected_forward}, nonfinite={rejected_nonfinite}",
                    flush=True,
                )
            continue
        if not verify_candidate(candidate, source="sample"):
            if args.progress_interval > 0 and sample_idx % args.progress_interval == 0:
                print(
                    f"Progress {sample_idx}/{args.num_samples}: kept={kept}, "
                    f"groups={len(groups)}, empty={rejected_empty}, exact_nonempty={rejected_exact_nonempty}, "
                    f"forward_fail={rejected_forward}, nonfinite={rejected_nonfinite}",
                    flush=True,
                )
            continue
        if args.progress_interval > 0 and (sample_idx % args.progress_interval == 0 or sample_idx == args.num_samples):
            best_so_far = min((it["score"] for items in groups.values() for it in items), default=float("nan"))
            print(
                f"Progress {sample_idx}/{args.num_samples}: kept={kept}, groups={len(groups)}, "
                f"best_{args.score_mode}={best_so_far:.4g}, empty={rejected_empty}, "
                f"exact_nonempty={rejected_exact_nonempty}, "
                f"forward_fail={rejected_forward}, nonfinite={rejected_nonfinite}",
                flush=True,
            )

    if not groups:
        print("WARNING: all posterior samples failed. No candidates written.")
        return

    print(
        "Rejection summary: "
        f"rejected_empty={rejected_empty}, "
        f"rejected_exact_nonempty={rejected_exact_nonempty}, "
        f"rejected_forward={rejected_forward}, "
        f"rejected_nonfinite={rejected_nonfinite}",
        flush=True,
    )
    print(
        f"Aggregating {kept} valid candidates into TOP {args.top_k} combinations "
        f"(mean_candidate_kept={mean_candidate_kept})...",
        flush=True,
    )
    total_kept = sum(len(v) for v in groups.values())
    ranked = []
    for key, items in groups.items():
        scores = np.array([it["score"] for it in items], dtype=np.float64)
        log_rmses = np.array([it["log_rmse"] for it in items], dtype=np.float64)
        weighted_chi2s = np.array([it["weighted_log_chi2"] for it in items], dtype=np.float64)
        score_weights = np.array([score_weight(s) for s in scores], dtype=np.float64)
        chi2_weights = np.exp(-0.5 * np.clip(weighted_chi2s, 0, 1e6))
        best = items[int(np.argmin(scores))]
        ranked.append(
            {
                "combination": key,
                "posterior_frequency": len(items) / total_kept,
                "score_weighted_probability": float(score_weights.sum()),
                "chi2_weighted_probability": float(chi2_weights.sum()),
                "fit_percent_by_log_rmse": float(np.mean(log_rmses < args.log_rmse_threshold)),
                "fit_percent_by_chi2": float(np.mean(weighted_chi2s < args.chi2_threshold)),
                "fit_percent": float(np.mean(log_rmses < args.log_rmse_threshold)),
                "best_score": float(np.min(scores)),
                "best_log_rmse": float(best["log_rmse"]),
                "best_chi2_weighted": float(best["weighted_log_chi2"]),
                "best_robust_log": float(best["robust_log"]),
                "best_relative_rmse": float(best["relative_rmse"]),
                "best_linear_rmse": float(best["linear_rmse"]),
                "best_source": str(best.get("source", "sample")),
                "best": best,
                "count": len(items),
            }
        )
    score_norm = sum(r["score_weighted_probability"] for r in ranked) or 1.0
    chi2_norm = sum(r["chi2_weighted_probability"] for r in ranked) or 1.0
    for r in ranked:
        r["score_weighted_probability"] /= score_norm
        r["chi2_weighted_probability"] /= chi2_norm
    ranked.sort(key=lambda r: (-r["score_weighted_probability"], r["best_score"]))
    ranked = ranked[: args.top_k]

    if args.refine_top_n > 0:
        n_refine = min(int(args.refine_top_n), len(ranked))
        print(f"Refining top {n_refine} candidates with scipy least_squares log residuals...", flush=True)
        print(
            "Note: refine eval=... counts residual calls, not scipy nfev. "
            "With numerical Jacobian, residual_calls can be roughly nfev * (n_variables + 1).",
            flush=True,
        )
        for idx, r in enumerate(ranked[:n_refine], start=1):
            unrefined = r["best"]
            n_components = len(unrefined["components"])
            type_names = [c["type_name"] for c in unrefined["components"]]
            n_refine_vars = sum(int(np.sum(schema.type_param_mask(int(c["type_id"])))) for c in unrefined["components"])
            n_refine_vars += schema.G_MAX
            if n_components > 1:
                n_refine_vars += n_components
            print(
                f"Refine #{idx}/{n_refine} start: combination={r['combination']}, "
                f"types={type_names}, components={n_components}, variables={n_refine_vars}, "
                f"initial_logRMSE={unrefined['log_rmse']:.5g}, initial_score={unrefined['score']:.5g}",
                flush=True,
            )
            refined, refine_info = refine_candidate(
                unrefined,
                q_eval,
                I_eval,
                sigma_log,
                score_mode=args.score_mode,
                robust_loss=args.robust_loss,
                robust_f_scale=args.robust_f_scale,
                max_nfev=args.refine_max_nfev,
                progress_interval=args.refine_progress_interval,
                progress_label=f" #{idx}/{n_refine}",
                refine_target_logrmse=args.refine_target_logrmse,
                refine_stall_patience=args.refine_stall_patience,
                refine_stall_tol=args.refine_stall_tol,
                refine_ftol=args.refine_ftol,
                refine_xtol=args.refine_xtol,
                refine_gtol=args.refine_gtol,
            )
            r["refine_info"] = refine_info
            r["unrefined_best_score"] = float(unrefined["score"])
            r["unrefined_best_log_rmse"] = float(unrefined["log_rmse"])
            r["unrefined_best_chi2_weighted"] = float(unrefined["weighted_log_chi2"])
            r["unrefined_best_source"] = str(unrefined.get("source", "sample"))
            r["best"] = refined
            r["best_score"] = float(refined["score"])
            r["best_log_rmse"] = float(refined["log_rmse"])
            r["best_chi2_weighted"] = float(refined["weighted_log_chi2"])
            r["best_robust_log"] = float(refined["robust_log"])
            r["best_relative_rmse"] = float(refined["relative_rmse"])
            r["best_linear_rmse"] = float(refined["linear_rmse"])
            r["best_source"] = str(refined.get("source", "sample"))
            print(
                f"Refine #{idx} {r['combination']}: "
                f"logRMSE {refine_info.get('initial_log_rmse', np.nan):.4g} -> "
                f"{refine_info.get('final_log_rmse', np.nan):.4g}, "
                f"success={refine_info.get('success', False)}, "
                f"nfev={refine_info.get('nfev', 0)}, "
                f"residual_calls={refine_info.get('residual_calls', 0)}, "
                f"early_stop={refine_info.get('early_stop_reason')}",
                flush=True,
            )

    json_rows = []
    residual_bank = []
    linear_residual_bank = []
    for rank, r in enumerate(ranked, start=1):
        best = r["best"]
        refine_info = r.get("refine_info")
        curve_bank.append(best["I_fit"])
        residual_bank.append(best["log_residual"])
        linear_residual_bank.append(best["linear_residual"])
        json_rows.append(
            {
                "rank": rank,
                "combination": r["combination"],
                "score_mode": args.score_mode,
                "robust_loss": args.robust_loss,
                "robust_f_scale": args.robust_f_scale,
                "q_min": args.q_min,
                "q_max": args.q_max,
                "drop_outliers": bool(args.drop_outliers),
                "n_removed_outliers": int(len(q_outlier)),
                "include_mean_candidate": bool(args.include_mean_candidate),
                "exact_nonempty": args.exact_nonempty,
                "refine_top_n": int(args.refine_top_n),
                "refine_max_nfev": int(args.refine_max_nfev),
                "refine_target_logrmse": float(args.refine_target_logrmse),
                "refine_stall_patience": int(args.refine_stall_patience),
                "refine_stall_tol": float(args.refine_stall_tol),
                "refine_attempted": refine_info is not None,
                "refine_success": bool(refine_info.get("success", False)) if refine_info else False,
                "refine_nfev": int(refine_info.get("nfev", 0)) if refine_info else 0,
                "refine_residual_calls": int(refine_info.get("residual_calls", 0)) if refine_info else 0,
                "refine_early_stop_reason": refine_info.get("early_stop_reason") if refine_info else None,
                "refine_best_log_rmse_seen": float(refine_info.get("best_log_rmse_seen", np.nan)) if refine_info else None,
                "refine_message": str(refine_info.get("message", "")) if refine_info else "",
                "rejected_empty": int(rejected_empty),
                "rejected_exact_nonempty": int(rejected_exact_nonempty),
                "rejected_forward": int(rejected_forward),
                "rejected_nonfinite": int(rejected_nonfinite),
                "best_source": r["best_source"],
                "unrefined_best_source": r.get("unrefined_best_source", r["best_source"]),
                "score_weighted_probability": r["score_weighted_probability"],
                "posterior_frequency": r["posterior_frequency"],
                "chi2_weighted_probability": r["chi2_weighted_probability"],
                "fit_percent": r["fit_percent"],
                "fit_percent_by_log_rmse": r["fit_percent_by_log_rmse"],
                "fit_percent_by_chi2": r["fit_percent_by_chi2"],
                "best_score": r["best_score"],
                "best_log_rmse": r["best_log_rmse"],
                "best_chi2_weighted": r["best_chi2_weighted"],
                "best_robust_log": r["best_robust_log"],
                "best_relative_rmse": r["best_relative_rmse"],
                "best_linear_rmse": r["best_linear_rmse"],
                "unrefined_best_score": r.get("unrefined_best_score", r["best_score"]),
                "unrefined_best_log_rmse": r.get("unrefined_best_log_rmse", r["best_log_rmse"]),
                "unrefined_best_chi2_weighted": r.get("unrefined_best_chi2_weighted", r["best_chi2_weighted"]),
                "components": [
                    {
                        "type": c["type_name"],
                        "weight": float(c["weight"]),
                        "params": component_params_json(c),
                    }
                    for c in best["components"]
                ],
                "global_params": {
                    name: float(best["global_phys"][i]) for i, name in enumerate(schema.GLOBAL_PARAM_NAMES)
                },
                "posterior_parameter_stats": posterior_stats(groups[r["combination"]]),
            }
        )

    with (out_dir / "top20_candidates.json").open("w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2)
    with (out_dir / "top20_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "combination",
                "score_mode",
                "robust_loss",
                "robust_f_scale",
                "q_min",
                "q_max",
                "drop_outliers",
                "n_removed_outliers",
                "include_mean_candidate",
                "exact_nonempty",
                "refine_top_n",
                "refine_max_nfev",
                "refine_target_logrmse",
                "refine_stall_patience",
                "refine_stall_tol",
                "refine_attempted",
                "refine_success",
                "refine_nfev",
                "refine_residual_calls",
                "refine_early_stop_reason",
                "refine_best_log_rmse_seen",
                "refine_message",
                "rejected_empty",
                "rejected_exact_nonempty",
                "rejected_forward",
                "rejected_nonfinite",
                "best_source",
                "unrefined_best_source",
                "score_weighted_probability",
                "posterior_frequency",
                "chi2_weighted_probability",
                "fit_percent_by_log_rmse",
                "fit_percent_by_chi2",
                "best_score",
                "best_log_rmse",
                "best_chi2_weighted",
                "best_robust_log",
                "best_relative_rmse",
                "best_linear_rmse",
                "unrefined_best_score",
                "unrefined_best_log_rmse",
                "unrefined_best_chi2_weighted",
            ],
        )
        writer.writeheader()
        for row in json_rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    np.savez_compressed(
        out_dir / "best_fit_curves.npz",
        q=q_eval,
        I_exp=I_eval,
        sigma=sigma_eval,
        I_fit=np.stack(curve_bank),
        q_removed_outliers=q_outlier,
        I_removed_outliers=I_outlier,
    )
    np.savez_compressed(
        out_dir / "residuals_top5.npz",
        q=q_eval,
        log_residual=np.stack(residual_bank[:5]),
        linear_residual=np.stack(linear_residual_bank[:5]),
    )

    if PLOT_AVAILABLE:
        plt.figure(figsize=(8, 6))
        plt.loglog(q_eval, I_eval, "k.", ms=3, label="input")
        for i, row in enumerate(json_rows[:5]):
            plt.loglog(
                q_eval,
                curve_bank[i],
                lw=1.0,
                label=f"#{row['rank']} {row['combination']} logRMSE={row['best_log_rmse']:.3g} chi2={row['best_chi2_weighted']:.3g}",
            )
        plt.xlabel("q / nm^-1")
        plt.ylabel("I")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "plot_top5.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.loglog(q_eval, I_eval, "k.", ms=3, label="valid input")
        if len(q_outlier) > 0:
            plt.loglog(q_outlier, I_outlier, "rx", ms=5, label="removed outliers")
        for i, row in enumerate(json_rows[:5]):
            plt.loglog(
                q_eval,
                curve_bank[i],
                lw=1.0,
                label=f"#{row['rank']} {row['combination']} score={row['best_score']:.3g}",
            )
        plt.xlabel("q / nm^-1")
        plt.ylabel("I")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "plot_top5_with_mask.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 6))
        for i, row in enumerate(json_rows[:5]):
            plt.semilogx(q_eval, residual_bank[i], lw=1.0, label=f"#{row['rank']} {row['combination']}")
        plt.axhline(0.0, color="k", lw=0.8)
        plt.xlabel("q / nm^-1")
        plt.ylabel("log(I_fit) - log(I_exp)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "plot_residuals_top5.png", dpi=180)
        plt.close()
    else:
        print("matplotlib is not available; skipped PNG plot outputs.", flush=True)
    print(f"Wrote {len(json_rows)} candidates to {out_dir}")


if __name__ == "__main__":
    main()

'''
conda run -n tf python Training/predict_topk.py \
  --model_dir /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS \
  --input_csv /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS/inspection/example_curve.csv \
  --output_dir /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS/prediction_logrmse \
  --num_samples 200 \
  --top_k 20 \
  --score_mode unweighted_log \
  --sampling_std 0.03 \
  --progress_interval 100

python Training/predict_topk.py \
  --model_dir /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS_K1 \
  --input_csv /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1/inspection/example_curve.csv \
  --output_dir /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS/prediction_k1_debug \
  --num_samples 5000 \
  --top_k 20 \
  --score_mode unweighted_log \
  --sampling_std 0.005 \
  --include_mean_candidate \
  --exact_nonempty 1 \
  --refine_top_n 5 \
  --refine_max_nfev 80 \
  --refine_progress_interval 20 \
  --refine_stall_patience 80 \
  --refine_stall_tol 1e-4 \
  --refine_target_logrmse 0.08 \
  --q_min 0.001 \
  --q_max 2.0 \
  --progress_interval 100 \
  --allow_unsafe_lambda

# Quick refinement sanity check:
#   --refine_top_n 3 \
#   --refine_max_nfev 40 \
#   --refine_stall_patience 40 \
#   --refine_stall_tol 1e-4

python Training/predict_topk.py \
  --model_dir /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS_K3K4 \
  --input_csv /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K3K4/inspection/example_curve.csv \
  --output_dir /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS/prediction_k3k4_debug \
  --num_samples 500 \
  --top_k 20 \
  --score_mode unweighted_log \
  --sampling_std 0.005 \
  --include_mean_candidate \
  --refine_top_n 5 \
  --refine_max_nfev 80 \
  --refine_progress_interval 20 \
  --refine_stall_patience 80 \
  --refine_stall_tol 1e-4 \
  --refine_target_logrmse 0.08 \
  --q_min 0.001 \
  --q_max 2.0 \
  --progress_interval 50 \
  --allow_unsafe_lambda
'''
