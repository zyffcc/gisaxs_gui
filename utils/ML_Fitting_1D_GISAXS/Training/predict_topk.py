"""Posterior sampling plus physics verification for TOP-K component candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TrainSetBuild import constraints, schema
from TrainSetBuild.physics_adapter import component_array_to_dict, evaluate_clean, global_array_to_dict
from TrainSetBuild.sampling import D_HARD_CORE_MARGIN, d_spacing_threshold, pad_2d, preprocess_curve
from Training.model import SlotQueryBase


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--constraints_json")
    p.add_argument(
        "--initial_candidates_json",
        action="append",
        default=[],
        help=(
            "Optional previous top-candidate JSON used as verified warm starts. "
            "May be supplied more than once; previous solutions are retained only if they still pass the current forward model."
        ),
    )
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
    p.add_argument(
        "--score_mode",
        choices=["unweighted_log", "weighted_log", "robust_log", "hybrid_log_relative"],
        default="unweighted_log",
    )
    p.add_argument(
        "--rank_mode",
        choices=["physics", "posterior"],
        default="physics",
        help=(
            "Final TOP-K ordering. 'physics' ranks by the verified/refined forward-model score; "
            "'posterior' preserves posterior-frequency and complexity-prior ordering."
        ),
    )
    p.add_argument(
        "--complexity_penalty",
        type=float,
        default=1.0,
        help="Log-prior penalty per component beyond K=1; larger values prefer simpler fits more strongly.",
    )
    p.add_argument(
        "--fit_equivalence_tolerance",
        type=float,
        default=0.01,
        help=(
            "Absolute logRMSE interval treated as physically equivalent during final physics ranking. "
            "Within one interval, fewer components rank first; 0 disables this parsimony tie-break."
        ),
    )
    p.add_argument(
        "--score_equivalence_tolerance",
        type=float,
        default=0.0,
        help=(
            "Equivalence interval in the selected --score_mode units for final physics ranking. "
            "When positive this takes precedence over --fit_equivalence_tolerance; fewer components rank first within a band."
        ),
    )
    p.add_argument(
        "--parameter_mode_radius",
        type=float,
        default=0.10,
        help="RMS radius in normalized parameter space used to split one type combination into distinct TOP-K modes.",
    )
    p.add_argument("--robust_loss", choices=["huber", "cauchy"], default="huber")
    p.add_argument("--robust_f_scale", type=float, default=0.3)
    p.add_argument("--sampling_std", type=float, default=0.03, help="Default normalized posterior sampling std.")
    p.add_argument(
        "--sampling_scales",
        default="1.0",
        help="Comma-separated multipliers cycled across posterior samples, e.g. 0.5,1,2,4.",
    )
    p.add_argument("--use_predicted_logstd", action="store_true", help="Use model log-std heads instead of --sampling_std.")
    p.add_argument("--include_mean_candidate", action="store_true", help="Verify and rank the deterministic posterior mean candidate.")
    p.add_argument(
        "--refine_top_n",
        type=int,
        default=0,
        help="Refine the N parameter modes with the smallest verified forward-model score.",
    )
    p.add_argument(
        "--refine_best_per_k",
        action="store_true",
        help="Also refine the best verified mode for every represented K, preventing one K from monopolizing refinement.",
    )
    p.add_argument(
        "--refine_q_stride",
        type=int,
        default=1,
        help="Use every Nth q point during optimization, then evaluate/rank the refined result on the full curve.",
    )
    p.add_argument(
        "--refine_max_nfev",
        type=int,
        default=80,
        help="Maximum scipy least_squares function evaluations per refined candidate.",
    )
    p.add_argument("--refine_ftol", type=float, default=1e-8, help="least_squares function tolerance.")
    p.add_argument("--refine_xtol", type=float, default=1e-8, help="least_squares step tolerance.")
    p.add_argument("--refine_gtol", type=float, default=1e-8, help="least_squares gradient tolerance.")
    p.add_argument(
        "--refine_progress_interval",
        type=int,
        default=10,
        help="Print refine progress every N residual evaluations; 0 disables per-candidate progress.",
    )
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
    # A comment such as ``# q (1/A) I err`` tokenizes to four labels for a
    # three-column table because the unit is separated from q.  Treat any
    # width mismatch as an ambiguous header instead of silently mapping I to
    # the uncertainty column.
    if names is not None and len(names) != width:
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


def _validate_model_contract(model_dir: Path, model, artifact: Path):
    manifest_path = model_dir / "manifest.json"
    config_path = model_dir / "model_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    expected_points = int(manifest.get("max_points", config.get("max_points", schema.MAX_POINTS)))
    expected_slots = int(manifest.get("max_slots", config.get("max_slots", schema.MAX_SLOTS)))
    expected_types = int(manifest.get("num_types", config.get("num_types", schema.NUM_TYPES)))
    actual = (schema.MAX_POINTS, schema.MAX_SLOTS, schema.NUM_TYPES)
    declared = (expected_points, expected_slots, expected_types)
    if declared != actual:
        raise RuntimeError(
            f"Model/schema mismatch: model declares max_points/max_slots/num_types={declared}, GUI inference expects {actual}."
        )
    expected_hash = str(manifest.get("sha256", "")).lower()
    if expected_hash and artifact.is_file():
        digest = hashlib.sha256()
        with artifact.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest().lower() != expected_hash:
            raise RuntimeError(f"Model checksum mismatch for {artifact}")
    input_names = {tensor.name.split(":", 1)[0].split("/")[-1] for tensor in model.inputs}
    missing_inputs = set(manifest.get("required_inputs", ())) - input_names
    output_names = set(getattr(model, "output_names", ()) or ())
    if isinstance(getattr(model, "output", None), dict):
        output_names.update(model.output)
    missing_outputs = set(manifest.get("required_outputs", ())) - output_names
    if missing_inputs or missing_outputs:
        raise RuntimeError(
            "Model architecture is incompatible: "
            f"missing inputs={sorted(missing_inputs)}, missing outputs={sorted(missing_outputs)}"
        )


def load_model(model_dir: Path, allow_unsafe_lambda: bool = False):
    model_dir = Path(model_dir)
    if model_dir.is_file():
        model_dir = model_dir.parent
    if model_dir.name == "saved_model" and (model_dir / "saved_model.pb").is_file():
        model_dir = model_dir.parent
    custom_objects = {"SlotQueryBase": SlotQueryBase}
    errors = []
    for candidate in [model_dir / "model.keras", model_dir / "saved_model"]:
        if candidate.exists():
            try:
                model = tf.keras.models.load_model(
                    candidate,
                    custom_objects=custom_objects,
                    compile=False,
                    safe_mode=not allow_unsafe_lambda,
                )
                _validate_model_contract(model_dir, model, candidate)
                print(f"Loaded and validated model artifact: {candidate}", flush=True)
                return model
            except ValueError as exc:
                message = str(exc)
                if "Lambda layer" in message and not allow_unsafe_lambda:
                    raise ValueError(
                        "Model contains Lambda layers and Keras safe deserialization blocked loading. "
                        "If you trust this model source, rerun with --allow_unsafe_lambda."
                    ) from exc
                errors.append((candidate, exc))
                print(f"WARNING: failed to load {candidate}: {type(exc).__name__}: {exc}", flush=True)
            except Exception as exc:
                errors.append((candidate, exc))
                print(f"WARNING: failed to load {candidate}: {type(exc).__name__}: {exc}", flush=True)
    if errors:
        detail = "\n".join([f"- {path}: {type(exc).__name__}: {exc}" for path, exc in errors])
        raise RuntimeError(f"No loadable model artifact found in {model_dir}. Tried:\n{detail}")
    raise FileNotFoundError(f"No saved_model or model.keras found in {model_dir}")


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


def enforce_d_constraints(active, pred, cons, rng=None, d_hard_core_margin=D_HARD_CORE_MARGIN):
    """Choose optional D presence and hard-project active D above the requested spacing bound."""
    d_logits = pred.get("d_present_logit")
    for item in active:
        slot = item["slot"]
        absent_allowed, present_allowed = np.asarray(cons["d_allowed"][slot], dtype=float)
        if absent_allowed < 0.5:
            present = True
        elif present_allowed < 0.5:
            present = False
        elif d_logits is None:
            present = bool(item["params_phys"][4] > 0.0)
        else:
            probability = float(sigmoid_stable(d_logits[0, slot]))
            present = probability >= 0.5 if rng is None else bool(rng.random() < probability)
        item["d_present"] = present
        if not present:
            item["params_phys"][4:6] = 0.0

    types = np.array([item["type_id"] for item in active], dtype=np.int32)
    params = np.stack([item["params_phys"] for item in active], axis=0)
    exists = np.ones(len(active), dtype=np.float32)
    rule_id = int(np.argmax(cons["d_spacing_rule"]))
    threshold = d_spacing_threshold(types, params, exists, rule_id)
    required_d = threshold * float(d_hard_core_margin)
    for item in active:
        if not item["d_present"]:
            continue
        slot, tid = item["slot"], item["type_id"]
        _, high_eff = effective_range(
            cons["param_low_norm"][slot, tid],
            cons["param_high_norm"][slot, tid],
            cons["param_range_mask"][slot, tid],
        )
        d_high = schema.denormalize_value(float(high_eff[4]), schema.PARAM_NORM_RANGES["D"])
        if required_d >= d_high:
            return False
        if item["params_phys"][4] <= required_d:
            item["params_phys"][4] = np.nextafter(required_d, np.inf)
    return True


def enforce_size_distribution_constraints(params_phys, type_id):
    """Project widths into the physical domain used by training.

    Vertical-cylinder sigma_R is fractional.  All other active widths are
    absolute and may not exceed 90% of their corresponding mean.  The trained
    no-D representation is exactly D=sigma_D=0.
    """
    params = np.asarray(params_phys, dtype=np.float64).copy()
    tiny = np.finfo(np.float32).eps
    if int(type_id) == schema.TYPE_VERTICAL_CYLINDER:
        params[1] = np.clip(params[1], tiny, 0.9)
    else:
        params[1] = np.clip(params[1], tiny, 0.9 * max(params[0], tiny))
    if int(type_id) == schema.TYPE_CYLINDER:
        params[3] = np.clip(params[3], tiny, 0.9 * max(params[2], tiny))
    if params[4] <= 0.0:
        params[4:6] = 0.0
    else:
        params[5] = np.clip(params[5], tiny, 0.9 * params[4])
    return params


def sample_candidate(
    pred,
    cons,
    rng,
    sampling_std=0.03,
    use_predicted_logstd=False,
    exact_nonempty=None,
    return_reason=False,
    d_hard_core_margin=D_HARD_CORE_MARGIN,
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
        params_phys = enforce_size_distribution_constraints(params_phys, tid)
        active.append({"slot": j, "type_id": tid, "params_phys": params_phys})

    if not active:
        return (None, "empty") if return_reason else None
    if exact_nonempty is not None:
        target = int(exact_nonempty)
        if len(active) != target:
            return (None, "exact_nonempty") if return_reason else None
    if not enforce_d_constraints(active, pred, cons, rng=rng, d_hard_core_margin=d_hard_core_margin):
        return (None, "d_spacing") if return_reason else None
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


def mean_candidate(
    pred,
    cons,
    exact_nonempty=None,
    return_reason=False,
    d_hard_core_margin=D_HARD_CORE_MARGIN,
):
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
        params_phys = enforce_size_distribution_constraints(params_phys, tid)
        active.append({"slot": j, "type_id": tid, "params_phys": params_phys})

    if not active:
        return (None, "empty") if return_reason else None
    if exact_nonempty is not None:
        target = int(exact_nonempty)
        if len(active) != target:
            return (None, "exact_nonempty") if return_reason else None

    if not enforce_d_constraints(active, pred, cons, rng=None, d_hard_core_margin=d_hard_core_margin):
        return (None, "d_spacing") if return_reason else None
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
    return sorted(
        components,
        key=lambda c: (int(c["type_id"]), float(np.asarray(c["params_phys"])[0]), float(c["weight"])),
    )


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


def candidate_mode_vector(item):
    """Fixed-scale vector for clustering parameter modes within one type combination."""
    values = []
    comps = sorted(
        item["components"],
        key=lambda c: (int(c["type_id"]), float(np.asarray(c["params_phys"])[0]), float(c["weight"])),
    )
    for comp in comps:
        tid = int(comp["type_id"])
        params = np.asarray(comp["params_phys"], dtype=np.float64)
        effective_mask = schema.effective_param_mask(tid, params)
        for pidx, name in enumerate(schema.PARAM_NAMES):
            values.append(
                schema.normalize_value(float(params[pidx]), schema.PARAM_NORM_RANGES[name])
                if effective_mask[pidx] > 0.5
                else 0.0
            )
        values.append(float(effective_mask[4] > 0.5))
        values.append(float(comp["weight"]))
    values.extend(schema.normalize_global(np.asarray(item["global_phys"], dtype=np.float64)).tolist())
    return np.asarray(values, dtype=np.float64)


def cluster_parameter_modes(items, radius=0.10):
    """Greedy score-ordered clustering; each returned mode may enter TOP-K separately."""
    radius = max(float(radius), 1e-6)
    modes = []
    for item in sorted(items, key=lambda it: it["score"]):
        vector = candidate_mode_vector(item)
        best_mode = None
        best_distance = np.inf
        for mode in modes:
            distance = float(np.sqrt(np.mean(np.square(vector - mode["centroid"]))))
            if distance <= radius and distance < best_distance:
                best_mode = mode
                best_distance = distance
        if best_mode is None:
            modes.append({"centroid": vector.copy(), "items": [item]})
        else:
            best_mode["items"].append(item)
            n = len(best_mode["items"])
            best_mode["centroid"] += (vector - best_mode["centroid"]) / float(n)
    return [mode["items"] for mode in modes]


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
    log_i_exp = np.log(np.maximum(I_exp, eps))
    log_i_fit = np.log(np.maximum(I_fit, eps))
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
    if score_mode == "hybrid_log_relative":
        # Retain scale-balanced log fitting while penalizing narrow overshoots
        # that touch few q points but are unacceptable in linear intensity.
        return float(metrics["log_rmse"] + 0.25 * np.log1p(metrics["relative_rmse"]))
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def score_weight(score, n_components=1, complexity_penalty=0.0):
    if not np.isfinite(score):
        return 0.0
    log_weight = -0.5 * np.clip(score, 0.0, 1e6) - float(complexity_penalty) * max(int(n_components) - 1, 0)
    return float(np.exp(np.clip(log_weight, -745.0, 0.0)))


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


def candidate_from_json_row(row):
    """Reconstruct one saved candidate without trusting its stored scores/curves."""
    components = []
    for saved in row.get("components", []):
        type_name = str(saved["type"])
        if type_name not in schema.NAME_TO_TYPE:
            raise ValueError(f"Unknown saved component type: {type_name}")
        tid = int(schema.NAME_TO_TYPE[type_name])
        params_phys = np.zeros(schema.P_MAX, dtype=np.float64)
        for name, value in saved.get("params", {}).items():
            if name not in schema.PARAM_NAMES:
                continue
            params_phys[schema.PARAM_NAMES.index(name)] = float(value)
        params_phys = schema.apply_type_param_mask(params_phys, tid)
        params_phys = enforce_size_distribution_constraints(params_phys, tid)
        components.append(component_array_to_dict(tid, params_phys, float(saved.get("weight", 1.0))))
    if not components:
        raise ValueError("Saved candidate has no components")
    global_saved = row.get("global_params", {})
    global_phys = np.asarray(
        [float(global_saved[name]) for name in schema.GLOBAL_PARAM_NAMES],
        dtype=np.float64,
    )
    return components, global_phys


def candidate_refine_setup(item, cons=None, d_hard_core_margin=D_HARD_CORE_MARGIN):
    components = item["components"]
    global_phys = np.asarray(item["global_phys"], dtype=np.float64)
    x0 = []
    lower = []
    upper = []
    comp_specs = []

    for comp_index, comp in enumerate(components):
        tid = int(comp["type_id"])
        params_phys = np.asarray(comp["params_phys"], dtype=np.float64)
        param_indices = [i for i, enabled in enumerate(schema.effective_param_mask(tid, params_phys)) if enabled > 0.5]
        comp_specs.append({"type_id": tid, "param_indices": param_indices})
        if cons is None:
            param_low = np.zeros(schema.P_MAX, dtype=np.float64)
            param_high = np.ones(schema.P_MAX, dtype=np.float64)
        else:
            slot = min(comp_index, schema.MAX_SLOTS - 1)
            param_low, param_high = effective_range(
                cons["param_low_norm"][slot, tid],
                cons["param_high_norm"][slot, tid],
                cons["param_range_mask"][slot, tid],
            )
        for pidx in param_indices:
            name = schema.PARAM_NAMES[pidx]
            low = float(param_low[pidx])
            high = float(param_high[pidx])
            value = schema.normalize_value(float(params_phys[pidx]), schema.PARAM_NORM_RANGES[name])
            x0.append(float(np.clip(value, low, high)))
            lower.append(low)
            upper.append(high)

    global_norm = schema.normalize_global(global_phys)
    global_start = len(x0)
    if cons is None:
        global_low = np.zeros(schema.G_MAX, dtype=np.float64)
        global_high = np.ones(schema.G_MAX, dtype=np.float64)
    else:
        global_low, global_high = effective_range(
            cons["global_low_norm"], cons["global_high_norm"], cons["global_range_mask"]
        )
    x0.extend(np.clip(global_norm, global_low, global_high).astype(float).tolist())
    lower.extend(np.asarray(global_low, dtype=float).tolist())
    upper.extend(np.asarray(global_high, dtype=float).tolist())

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
        "d_rule_id": schema.D_RULE_FREE if cons is None else int(np.argmax(cons["d_spacing_rule"])),
        "d_hard_core_margin": float(d_hard_core_margin),
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
        effective_mask = np.zeros(schema.P_MAX, dtype=np.float32)
        effective_mask[spec["param_indices"]] = 1.0
        params_phys = schema.apply_param_mask(params_phys, effective_mask)
        params_phys = enforce_size_distribution_constraints(params_phys, tid)
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
    rule_id = int(setup.get("d_rule_id", schema.D_RULE_FREE))
    if rule_id != schema.D_RULE_FREE:
        types = np.array([c["type_id"] for c in components], dtype=np.int32)
        params = np.stack([c["params_phys"] for c in components], axis=0)
        threshold = d_spacing_threshold(types, params, np.ones(len(components)), rule_id)
        required_d = threshold * float(setup.get("d_hard_core_margin", D_HARD_CORE_MARGIN))
        for comp, comp_dict in zip(components, component_dicts):
            if comp["params_phys"][4] > 0.0:
                if required_d >= schema.PARAM_RANGES["D"].high:
                    raise ValueError("D spacing constraint is infeasible during refinement")
                comp_dict["params_phys"][4] = max(
                    float(comp_dict["params_phys"][4]),
                    float(np.nextafter(required_d, np.inf)),
                )
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
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    progress_interval=10,
    progress_label="",
    refine_target_logrmse=0.0,
    refine_stall_patience=80,
    refine_stall_tol=1e-4,
    cons=None,
    optimization_q_stride=1,
    d_hard_core_margin=D_HARD_CORE_MARGIN,
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

    x0, lower, upper, setup = candidate_refine_setup(
        item,
        cons=cons,
        d_hard_core_margin=d_hard_core_margin,
    )
    optimization_q_stride = max(int(optimization_q_stride), 1)
    q_opt = q_eval[::optimization_q_stride]
    I_opt = I_eval[::optimization_q_stride]
    log_i_exp = np.log(np.maximum(I_opt, 1e-30))
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
            i_fit = evaluate_clean(q_opt, components, global_array_to_dict(global_phys))
        except Exception:
            residual = np.full_like(log_i_exp, 1e6, dtype=np.float64)
        else:
            if np.all(np.isfinite(i_fit)):
                residual = np.log(np.maximum(i_fit, 1e-30)) - log_i_exp
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
                ftol=float(ftol),
                xtol=float(xtol),
                gtol=float(gtol),
                x_scale="jac",
                loss=robust_loss,
                f_scale=max(float(robust_f_scale), 1e-12),
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


def main():
    args = parse_args()
    try:
        sampling_scales = tuple(float(item.strip()) for item in str(args.sampling_scales).split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--sampling_scales must be a comma-separated list of positive numbers") from exc
    if not sampling_scales or any(not np.isfinite(scale) or scale <= 0 for scale in sampling_scales):
        raise ValueError("--sampling_scales must contain at least one positive finite number")
    if args.exact_nonempty is not None and args.exact_nonempty < 1:
        raise ValueError("--exact_nonempty must be >= 1")
    if args.refine_top_n < 0:
        raise ValueError("--refine_top_n must be >= 0")
    if args.refine_max_nfev < 1:
        raise ValueError("--refine_max_nfev must be >= 1")
    if args.refine_ftol <= 0 or args.refine_xtol <= 0 or args.refine_gtol <= 0:
        raise ValueError("--refine_ftol, --refine_xtol, and --refine_gtol must be > 0")
    if args.refine_progress_interval < 0:
        raise ValueError("--refine_progress_interval must be >= 0")
    if args.refine_stall_patience < 0:
        raise ValueError("--refine_stall_patience must be >= 0")
    if args.refine_stall_tol < 0:
        raise ValueError("--refine_stall_tol must be >= 0")
    if args.refine_q_stride < 1:
        raise ValueError("--refine_q_stride must be >= 1")
    if args.fit_equivalence_tolerance < 0.0:
        raise ValueError("--fit_equivalence_tolerance must be >= 0")
    if args.score_equivalence_tolerance < 0.0:
        raise ValueError("--score_equivalence_tolerance must be >= 0")
    if args.complexity_penalty < 0:
        raise ValueError("--complexity_penalty must be >= 0")
    if args.parameter_mode_radius <= 0:
        raise ValueError("--parameter_mode_radius must be > 0")
    rng = np.random.default_rng(args.seed)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = model_dir / "dataset_metadata.json"
    try:
        model_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    except Exception:
        model_metadata = {}
    cylinder_revision = ((model_metadata.get("forward_model") or {}).get("random_cylinder_radial_amplitude"))
    if cylinder_revision != "2*cylindrical_J1(x)/x":
        print(
            "WARNING: this checkpoint predates the exact random-cylinder J1 forward-model revision. "
            "Physics verification now uses the corrected equation, but retraining is recommended before "
            "interpreting random-cylinder proposal probabilities quantitatively.",
            flush=True,
        )

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
    d_hard_core_margin = float(
        ((cons_config or {}).get("d_constraint") or {}).get("margin", D_HARD_CORE_MARGIN)
    )
    if d_hard_core_margin <= 1.0:
        raise ValueError("d_constraint.margin must be strictly greater than 1.0")

    print(f"Loading model from: {model_dir}", flush=True)
    model = load_model(model_dir, allow_unsafe_lambda=args.allow_unsafe_lambda)
    print("Running neural network proposal pass...", flush=True)
    proposal_input = make_input(q_eval, I_eval, sigma_eval, cons)
    model_input_names = {tensor.name.split(":")[0] for tensor in model.inputs}
    proposal_input = {key: value for key, value in proposal_input.items() if key in model_input_names}
    pred = model(proposal_input, training=False)
    pred = {k: v.numpy() for k, v in pred.items()}
    exist_prob = sigmoid_stable(pred["exist_logit"][0])
    type_prob = np.stack([softmax(pred["type_logits"][0, j]) for j in range(schema.MAX_SLOTS)], axis=0)
    print(
        "Model proposal summary: "
        f"exist_prob={np.array2string(exist_prob, precision=3)}, "
        f"top_types={[schema.TYPE_NAMES[int(np.argmax(type_prob[j]))] for j in range(schema.MAX_SLOTS)]}",
        flush=True,
    )

    groups = defaultdict(list)
    curve_bank = []
    sigma_log = np.maximum(sigma_eval / np.maximum(I_eval, 1e-30), 1e-3)

    print(
        f"Sampling {args.num_samples} posterior candidates with normalized std scales "
        f"{sampling_scales} and verifying with physics forward model...",
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

    warm_start_loaded = 0
    warm_start_rejected = 0
    for initial_path_text in args.initial_candidates_json:
        initial_path = Path(initial_path_text)
        with initial_path.open("r", encoding="utf-8") as f:
            saved_rows = json.load(f)
        if not isinstance(saved_rows, list):
            raise ValueError(f"Initial candidates must be a JSON list: {initial_path}")
        for saved_rank, saved_row in enumerate(saved_rows, start=1):
            try:
                candidate = candidate_from_json_row(saved_row)
                accepted = verify_candidate(candidate, source=f"warm_start:{initial_path.name}:rank{saved_rank}")
            except Exception as exc:
                warm_start_rejected += 1
                print(f"Warm-start rank {saved_rank} rejected from {initial_path}: {exc}", flush=True)
                continue
            if accepted:
                warm_start_loaded += 1
            else:
                warm_start_rejected += 1
    if args.initial_candidates_json:
        print(
            f"Warm starts verified: loaded={warm_start_loaded}, rejected={warm_start_rejected}, "
            f"files={len(args.initial_candidates_json)}",
            flush=True,
        )

    if args.include_mean_candidate:
        candidate, reason = mean_candidate(
            pred,
            cons,
            exact_nonempty=args.exact_nonempty,
            return_reason=True,
            d_hard_core_margin=d_hard_core_margin,
        )
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
        effective_sampling_std = args.sampling_std * sampling_scales[(sample_idx - 1) % len(sampling_scales)]
        candidate, reason = sample_candidate(
            pred,
            cons,
            rng,
            sampling_std=effective_sampling_std,
            use_predicted_logstd=args.use_predicted_logstd,
            exact_nonempty=args.exact_nonempty,
            return_reason=True,
            d_hard_core_margin=d_hard_core_margin,
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
        f"Aggregating {kept} valid candidates into TOP {args.top_k} parameter modes "
        f"(mean_candidate_kept={mean_candidate_kept})...",
        flush=True,
    )
    total_kept = sum(len(v) for v in groups.values())
    ranked = []
    for key, items in groups.items():
        parameter_modes = cluster_parameter_modes(items, radius=args.parameter_mode_radius)
        for mode_index, mode_items in enumerate(parameter_modes, start=1):
            scores = np.array([it["score"] for it in mode_items], dtype=np.float64)
            log_rmses = np.array([it["log_rmse"] for it in mode_items], dtype=np.float64)
            weighted_chi2s = np.array([it["weighted_log_chi2"] for it in mode_items], dtype=np.float64)
            n_components = len(mode_items[0]["components"])
            score_weights = np.array(
                [score_weight(s, n_components, args.complexity_penalty) for s in scores], dtype=np.float64
            )
            complexity_prior = float(np.exp(-args.complexity_penalty * max(n_components - 1, 0)))
            chi2_weights = np.exp(-0.5 * np.clip(weighted_chi2s, 0, 1e6)) * complexity_prior
            # The representative passed to physical refinement must be the
            # curve with the smallest directly interpretable forward-model
            # logRMSE.  Robust/posterior scores still contribute probability,
            # but selecting the representative by a different metric can hide
            # a much better starting point inside the same parameter mode.
            best = mode_items[int(np.argmin(log_rmses))]
            ranked.append(
                {
                    "combination": key,
                    "parameter_mode": mode_index,
                    "n_components": n_components,
                    "complexity_log_prior": -args.complexity_penalty * max(n_components - 1, 0),
                    "posterior_frequency": len(mode_items) / total_kept,
                    "score_weighted_probability": float(score_weights.sum()),
                    "chi2_weighted_probability": float(chi2_weights.sum()),
                    "fit_percent_by_log_rmse": float(np.mean(log_rmses < args.log_rmse_threshold)),
                    "fit_percent_by_chi2": float(np.mean(weighted_chi2s < args.chi2_threshold)),
                    "fit_percent": float(np.mean(log_rmses < args.log_rmse_threshold)),
                    "best_score": float(best["score"]),
                    "best_log_rmse": float(best["log_rmse"]),
                    "best_chi2_weighted": float(best["weighted_log_chi2"]),
                    "best_robust_log": float(best["robust_log"]),
                    "best_relative_rmse": float(best["relative_rmse"]),
                    "best_linear_rmse": float(best["linear_rmse"]),
                    "best_source": str(best.get("source", "sample")),
                    "best": best,
                    "items": mode_items,
                    "count": len(mode_items),
                }
            )
    score_norm = sum(r["score_weighted_probability"] for r in ranked) or 1.0
    chi2_norm = sum(r["chi2_weighted_probability"] for r in ranked) or 1.0
    for r in ranked:
        r["score_weighted_probability"] /= score_norm
        r["chi2_weighted_probability"] /= chi2_norm
    posterior_order = sorted(ranked, key=lambda r: (-r["score_weighted_probability"], r["best_score"]))
    for posterior_rank, row in enumerate(posterior_order, start=1):
        row["posterior_rank"] = posterior_rank

    # Refinement selection is deliberately based on verified physics error,
    # not posterior frequency or the parsimony prior.  Those are useful model
    # preferences, but must not prevent a better-fitting basin from being
    # optimized.
    physics_order = sorted(ranked, key=lambda r: (r["best_score"], r["best_log_rmse"], r["n_components"]))
    refine_targets = list(physics_order[: min(int(args.refine_top_n), len(physics_order))])
    if args.refine_best_per_k:
        for k in sorted({int(row["n_components"]) for row in physics_order}):
            best_for_k = next(row for row in physics_order if int(row["n_components"]) == k)
            if all(best_for_k is not existing for existing in refine_targets):
                refine_targets.append(best_for_k)

    if refine_targets:
        n_refine = len(refine_targets)
        print(
            f"Refining {n_refine} physics-selected modes "
            f"(top_n={args.refine_top_n}, best_per_k={args.refine_best_per_k}, "
            f"q_stride={args.refine_q_stride})...",
            flush=True,
        )
        print(
            "Note: refine eval=... counts residual calls, not scipy nfev. "
            "With numerical Jacobian, residual_calls can be roughly nfev * (n_variables + 1).",
            flush=True,
        )
        for idx, r in enumerate(refine_targets, start=1):
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
                ftol=args.refine_ftol,
                xtol=args.refine_xtol,
                gtol=args.refine_gtol,
                progress_interval=args.refine_progress_interval,
                progress_label=f" #{idx}/{n_refine}",
                refine_target_logrmse=args.refine_target_logrmse,
                refine_stall_patience=args.refine_stall_patience,
                refine_stall_tol=args.refine_stall_tol,
                cons=cons,
                optimization_q_stride=args.refine_q_stride,
                d_hard_core_margin=d_hard_core_margin,
            )
            r["refine_info"] = refine_info
            r["unrefined_best_score"] = float(unrefined["score"])
            r["unrefined_best_log_rmse"] = float(unrefined["log_rmse"])
            r["unrefined_best_chi2_weighted"] = float(unrefined["weighted_log_chi2"])
            r["unrefined_best_source"] = str(unrefined.get("source", "sample"))
            refine_accepted = float(refined["score"]) <= float(unrefined["score"])
            refine_info["accepted_by_score"] = bool(refine_accepted)
            selected = refined if refine_accepted else unrefined
            r["best"] = selected
            r["best_score"] = float(selected["score"])
            r["best_log_rmse"] = float(selected["log_rmse"])
            r["best_chi2_weighted"] = float(selected["weighted_log_chi2"])
            r["best_robust_log"] = float(selected["robust_log"])
            r["best_relative_rmse"] = float(selected["relative_rmse"])
            r["best_linear_rmse"] = float(selected["linear_rmse"])
            r["best_source"] = str(selected.get("source", "sample"))
            print(
                f"Refine #{idx} {r['combination']}: "
                f"logRMSE {refine_info.get('initial_log_rmse', np.nan):.4g} -> "
                f"{refine_info.get('final_log_rmse', np.nan):.4g}, "
                f"success={refine_info.get('success', False)}, "
                f"accepted_by_score={refine_accepted}, "
                f"nfev={refine_info.get('nfev', 0)}, "
                f"residual_calls={refine_info.get('residual_calls', 0)}, "
                f"early_stop={refine_info.get('early_stop_reason')}",
                flush=True,
            )

    # Re-rank only after every selected local optimization has been evaluated
    # on the full q grid.  Keep posterior/prior rank as a separate diagnostic.
    use_selected_score = args.score_mode != "unweighted_log" or float(args.score_equivalence_tolerance) > 0.0
    if use_selected_score:
        best_physics_value = min(float(r["best_score"]) for r in ranked)
        equivalence_tolerance = float(args.score_equivalence_tolerance)
    else:
        best_physics_value = min(float(r["best_log_rmse"]) for r in ranked)
        equivalence_tolerance = float(args.fit_equivalence_tolerance)

    def final_physics_key(row):
        physics_value = float(row["best_score"] if use_selected_score else row["best_log_rmse"])
        if equivalence_tolerance > 0.0:
            fit_band = int(np.floor(max(physics_value - best_physics_value, 0.0) / equivalence_tolerance + 1e-12))
            return fit_band, int(row["n_components"]), physics_value, float(row["best_log_rmse"])
        return physics_value, int(row["n_components"]), float(row["best_log_rmse"])

    physics_order = sorted(ranked, key=final_physics_key)
    for physics_rank, row in enumerate(physics_order, start=1):
        row["physics_rank"] = physics_rank
    if args.rank_mode == "physics":
        ranked = physics_order[: args.top_k]
    else:
        ranked = posterior_order[: args.top_k]

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
                "rank_mode": args.rank_mode,
                "physics_rank": int(r["physics_rank"]),
                "posterior_rank": int(r["posterior_rank"]),
                "combination": r["combination"],
                "parameter_mode": int(r["parameter_mode"]),
                "parameter_mode_radius": float(args.parameter_mode_radius),
                "sampling_std": float(args.sampling_std),
                "sampling_scales": list(sampling_scales),
                "n_components": int(r["n_components"]),
                "complexity_penalty": float(args.complexity_penalty),
                "fit_equivalence_tolerance": float(args.fit_equivalence_tolerance),
                "score_equivalence_tolerance": float(args.score_equivalence_tolerance),
                "complexity_log_prior": float(r["complexity_log_prior"]),
                "score_mode": args.score_mode,
                "robust_loss": args.robust_loss,
                "robust_f_scale": args.robust_f_scale,
                "q_min": args.q_min,
                "q_max": args.q_max,
                "drop_outliers": bool(args.drop_outliers),
                "n_removed_outliers": int(len(q_outlier)),
                "include_mean_candidate": bool(args.include_mean_candidate),
                "initial_candidates_json": [str(p) for p in args.initial_candidates_json],
                "warm_start_loaded": int(warm_start_loaded),
                "warm_start_rejected": int(warm_start_rejected),
                "exact_nonempty": args.exact_nonempty,
                "refine_top_n": int(args.refine_top_n),
                "refine_best_per_k": bool(args.refine_best_per_k),
                "refine_q_stride": int(args.refine_q_stride),
                "refine_max_nfev": int(args.refine_max_nfev),
                "refine_target_logrmse": float(args.refine_target_logrmse),
                "refine_stall_patience": int(args.refine_stall_patience),
                "refine_stall_tol": float(args.refine_stall_tol),
                "refine_attempted": refine_info is not None,
                "refine_success": bool(refine_info.get("success", False)) if refine_info else False,
                "refine_accepted": bool(refine_info.get("accepted_by_score", False)) if refine_info else False,
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
                "posterior_parameter_stats": posterior_stats(r["items"]),
            }
        )

    with (out_dir / "top20_candidates.json").open("w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2)
    with (out_dir / "top20_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "rank_mode",
                "physics_rank",
                "posterior_rank",
                "combination",
                "parameter_mode",
                "parameter_mode_radius",
                "n_components",
                "complexity_penalty",
                "fit_equivalence_tolerance",
                "score_equivalence_tolerance",
                "complexity_log_prior",
                "score_mode",
                "robust_loss",
                "robust_f_scale",
                "q_min",
                "q_max",
                "drop_outliers",
                "n_removed_outliers",
                "include_mean_candidate",
                "initial_candidates_json",
                "warm_start_loaded",
                "warm_start_rejected",
                "exact_nonempty",
                "refine_top_n",
                "refine_best_per_k",
                "refine_q_stride",
                "refine_max_nfev",
                "refine_target_logrmse",
                "refine_stall_patience",
                "refine_stall_tol",
                "refine_attempted",
                "refine_success",
                "refine_accepted",
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
