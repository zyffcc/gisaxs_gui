"""Focused helpers for TOP-K prediction: prediction refinement."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TrainSetBuild import schema
from TrainSetBuild.physics_adapter import (
    component_array_to_dict,
    evaluate_clean,
    global_array_to_dict,
)
from TrainSetBuild.sampling import D_HARD_CORE_MARGIN, d_spacing_threshold

from Training.prediction_candidates import enforce_size_distribution_constraints
from Training.prediction_preprocessing import effective_range, softmax
from Training.prediction_scoring import fit_metrics, score_from_metrics


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
        if refine_target_logrmse > 0.0 and progress["best_log_rmse"] <= float(
            refine_target_logrmse
        ):
            progress["early_stop_reason"] = (
                f"target_logrmse reached: {progress['best_log_rmse']:.6g}"
            )
            raise EarlyStopRefine()
        if refine_stall_patience > 0 and progress["calls"] - progress["last_improve_call"] >= int(
            refine_stall_patience
        ):
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
        param_indices = [
            i
            for i, enabled in enumerate(schema.effective_param_mask(tid, params_phys))
            if enabled > 0.5
        ]
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
    return (
        np.asarray(x0, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        setup,
    )
