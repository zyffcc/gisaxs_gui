"""Focused helpers for TOP-K prediction: prediction scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TrainSetBuild import schema
from TrainSetBuild.physics_adapter import component_array_to_dict

from Training.prediction_candidates import (
    component_params_json,
    enforce_size_distribution_constraints,
)


def score_weight(score, n_components=1, complexity_penalty=0.0):
    if not np.isfinite(score):
        return 0.0
    log_weight = -0.5 * np.clip(score, 0.0, 1e6) - float(complexity_penalty) * max(
        int(n_components) - 1, 0
    )
    return float(np.exp(np.clip(log_weight, -745.0, 0.0)))


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
        components.append(
            component_array_to_dict(tid, params_phys, float(saved.get("weight", 1.0)))
        )
    if not components:
        raise ValueError("Saved candidate has no components")
    global_saved = row.get("global_params", {})
    global_phys = np.asarray(
        [float(global_saved[name]) for name in schema.GLOBAL_PARAM_NAMES],
        dtype=np.float64,
    )
    return components, global_phys


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
