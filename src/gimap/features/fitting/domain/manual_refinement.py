"""Manual Auto Refine 的无 UI least-squares 实现。"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares

from .constraints import clamp_to_open_bounds
from .scoring import log_rmse, log_residuals


def run_manual_refinement(
    setup,
    selected,
    options,
    progress_callback=None,
    stop_callback=None,
):
    model_func = setup["model_func"]
    q_model = np.asarray(setup["q_model"], dtype=float)
    observed = np.asarray(setup["y"], dtype=float)
    initial_params = np.array([float(item["value"]) for item in setup["params"]], dtype=float)
    variable_indices = [int(item["index"]) for item, _lower, _upper in selected]
    lower = np.array([float(bound) for _item, bound, _upper in selected], dtype=float)
    upper = np.array([float(bound) for _item, _lower, bound in selected], dtype=float)
    x0 = clamp_to_open_bounds(initial_params[variable_indices], lower, upper)

    calls_per_nfev = max(1, int(x0.size)) + 1
    progress = {
        "best": np.inf,
        "best_x": x0.copy(),
        "calls": 0,
        "last_report_nfev": -1,
        "last_report_time": 0.0,
    }
    target = float(options.get("target_logrmse", 0.0) or 0.0)
    progress_interval = max(1, int(options.get("progress_interval", 5) or 5))
    show_interval = int(options.get("show_interval", 0) or 0)
    max_nfev = int(options.get("max_nfev", 120))
    min_progress_seconds = max(
        0.3,
        float(options.get("min_progress_seconds", 0.5) or 0.5),
    )

    def build_params(values):
        params = initial_params.copy()
        params[variable_indices] = values
        return params

    def score(params):
        predicted = np.asarray(model_func(q_model, *params), dtype=float)
        if predicted.shape != observed.shape:
            predicted = predicted[: observed.shape[0]]
        if not np.all(np.isfinite(predicted)) or predicted.shape != observed.shape:
            return np.inf, predicted
        return log_rmse(observed, predicted), predicted

    initial_log_rmse, _ = score(initial_params)
    if progress_callback:
        progress_callback(
            _progress_payload(
                initial_params,
                variable_indices,
                initial_log_rmse,
                initial_log_rmse,
                0,
                0,
                max_nfev,
                show_interval,
                "started",
                False,
            )
        )

    def residuals(values):
        if stop_callback and stop_callback():
            raise RuntimeError("__AUTO_REFINE_STOP_REQUESTED__")
        params = build_params(values)
        predicted = np.asarray(model_func(q_model, *params), dtype=float)
        if predicted.shape != observed.shape or not np.all(np.isfinite(predicted)):
            return np.full_like(observed, 1e6, dtype=float)
        residual = log_residuals(observed, predicted)
        current = float(np.sqrt(np.mean(residual * residual)))
        progress["calls"] += 1
        nfev_estimate = max(1, int(np.ceil(progress["calls"] / calls_per_nfev)))
        if current < progress["best"]:
            progress["best"] = current
            progress["best_x"] = np.array(values, dtype=float, copy=True)
        now = time.perf_counter()
        interval_due = (
            nfev_estimate != progress["last_report_nfev"]
            and nfev_estimate % progress_interval == 0
            and now - float(progress["last_report_time"]) >= min_progress_seconds
        )
        if progress_callback and (progress["calls"] == 1 or interval_due):
            progress["last_report_nfev"] = nfev_estimate
            progress["last_report_time"] = now
            progress_callback(
                _progress_payload(
                    build_params(progress["best_x"]),
                    variable_indices,
                    initial_log_rmse,
                    float(progress["best"]),
                    nfev_estimate,
                    int(progress["calls"]),
                    max_nfev,
                    show_interval,
                    "running",
                    False,
                    current=current,
                )
            )
        if target > 0 and current <= target:
            raise RuntimeError("__AUTO_REFINE_TARGET_REACHED__")
        return residual

    stopped = False
    try:
        result = least_squares(
            residuals,
            x0,
            bounds=(lower, upper),
            max_nfev=max_nfev,
            ftol=options.get("ftol"),
            xtol=options.get("xtol"),
            gtol=options.get("gtol"),
        )
        final_x = result.x
        message = str(result.message)
        nfev = int(result.nfev)
    except RuntimeError as exc:
        text = str(exc)
        if "__AUTO_REFINE_TARGET_REACHED__" not in text and "__AUTO_REFINE_STOP_REQUESTED__" not in text:
            raise
        final_x = np.array(progress.get("best_x", x0), dtype=float, copy=True)
        stopped = "__AUTO_REFINE_STOP_REQUESTED__" in text
        message = "Stopped by user." if stopped else "Stopped after reaching target logRMSE."
        nfev = max(1, int(np.ceil(int(progress.get("calls", 0)) / calls_per_nfev)))

    final_params = build_params(final_x)
    final_log_rmse, _ = score(final_params)
    payload = _progress_payload(
        final_params,
        variable_indices,
        initial_log_rmse,
        final_log_rmse,
        nfev,
        int(progress.get("calls", nfev)),
        max_nfev,
        show_interval,
        message,
        stopped,
    )
    if progress_callback:
        progress_callback(payload)
    return payload


def _progress_payload(
    params,
    selected_indices,
    initial,
    final,
    nfev,
    calls,
    max_nfev,
    show_interval,
    message,
    stopped,
    *,
    current=None,
):
    return {
        "params": np.asarray(params, dtype=float),
        "selected_indices": [int(index) for index in selected_indices],
        "initial_log_rmse": float(initial),
        "final_log_rmse": float(final),
        "best_log_rmse": float(final),
        "current_log_rmse": float(final if current is None else current),
        "nfev": int(nfev),
        "nfev_est": int(nfev),
        "calls": int(calls),
        "max_nfev": int(max_nfev),
        "show_interval": int(show_interval),
        "message": str(message),
        "stopped": bool(stopped),
    }
