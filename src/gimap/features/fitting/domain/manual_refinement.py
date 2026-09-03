"""Framework-neutral local refinement and bounded global parameter search."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import differential_evolution, least_squares, lsq_linear
from scipy.stats import qmc

from .constraints import clamp_to_open_bounds, parameter_base_name
from .scattering_model import mixed_model_components
from .scoring import log_residuals

_STOP_REQUESTED = "__AUTO_REFINE_STOP_REQUESTED__"
_TARGET_REACHED = "__AUTO_REFINE_TARGET_REACHED__"


@dataclass(frozen=True)
class _ParameterTransform:
    lower: np.ndarray
    upper: np.ndarray
    logarithmic: np.ndarray

    @classmethod
    def create(cls, lower, upper, *, logarithmic_wide_ranges: bool):
        lower_array = np.asarray(lower, dtype=float)
        upper_array = np.asarray(upper, dtype=float)
        logarithmic = np.zeros(lower_array.shape, dtype=bool)
        if logarithmic_wide_ranges:
            positive = lower_array > 0
            logarithmic[positive] = (upper_array[positive] / lower_array[positive]) >= 100.0
        return cls(lower_array, upper_array, logarithmic)

    def decode(self, normalized) -> np.ndarray:
        values = np.asarray(normalized, dtype=float)
        physical = self.lower + values * (self.upper - self.lower)
        if np.any(self.logarithmic):
            lo = np.log(self.lower[self.logarithmic])
            hi = np.log(self.upper[self.logarithmic])
            physical[self.logarithmic] = np.exp(lo + values[self.logarithmic] * (hi - lo))
        return physical

    def encode(self, physical) -> np.ndarray:
        values = np.asarray(physical, dtype=float)
        normalized = (values - self.lower) / (self.upper - self.lower)
        if np.any(self.logarithmic):
            lo = np.log(self.lower[self.logarithmic])
            hi = np.log(self.upper[self.logarithmic])
            normalized[self.logarithmic] = (np.log(values[self.logarithmic]) - lo) / (hi - lo)
        return clamp_to_open_bounds(
            normalized,
            np.zeros_like(normalized),
            np.ones_like(normalized),
            epsilon=1e-12,
        )


def run_manual_refinement(
    setup,
    selected,
    options,
    progress_callback=None,
    stop_callback=None,
):
    """Optimize selected parameters using local or global-then-local search.

    Least-squares runs in a unit hypercube, so its step tolerance is independent
    of the physical scales of the selected parameters.
    """

    if not selected:
        raise ValueError("Select at least one parameter to refine")
    mode = str(options.get("mode", "local")).strip().lower()
    if mode not in {"local", "global"}:
        raise ValueError(f"Unsupported refinement mode: {mode}")

    model_func = setup["model_func"]
    q_model = np.asarray(setup["q_model"], dtype=float)
    observed = np.asarray(setup["y"], dtype=float)
    initial_params = np.array([float(item["value"]) for item in setup["params"]], dtype=float)
    variable_indices = [int(item["index"]) for item, _lower, _upper in selected]
    lower = np.array([float(bound) for _item, bound, _upper in selected], dtype=float)
    upper = np.array([float(bound) for _item, _lower, bound in selected], dtype=float)
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(upper <= lower):
        raise ValueError("Every selected parameter needs finite bounds with max greater than min")

    initial_values = initial_params[variable_indices]
    if np.any(initial_values < lower) or np.any(initial_values > upper):
        raise ValueError("Every selected parameter range must include its current value")
    physical_start = clamp_to_open_bounds(initial_values, lower, upper)
    transform = _ParameterTransform.create(
        lower,
        upper,
        logarithmic_wide_ranges=mode == "global",
    )
    normalized_start = transform.encode(physical_start)

    max_nfev = max(1, int(options.get("max_nfev", 120)))
    global_samples = max(1, int(options.get("global_samples", 16384))) if mode == "global" else 0
    global_starts = max(1, int(options.get("global_starts", 3))) if mode == "global" else 1
    global_starts = min(global_starts, global_samples)
    progress_interval = max(1, int(options.get("progress_interval", 5) or 5))
    show_interval = int(options.get("show_interval", 0) or 0)
    target = float(options.get("target_logrmse", 0.0) or 0.0)
    min_progress_seconds = max(0.1, float(options.get("min_progress_seconds", 0.5) or 0.5))
    calls_per_nfev = normalized_start.size + 1
    total_work = max_nfev if mode == "local" else global_samples + global_starts * max_nfev

    def build_params(normalized):
        params = initial_params.copy()
        params[variable_indices] = transform.decode(normalized)
        return params

    def residual_for_params(params):
        try:
            with np.errstate(all="ignore"):
                predicted = np.asarray(model_func(q_model, *params), dtype=float)
        except Exception:
            return None
        if predicted.shape != observed.shape or not np.all(np.isfinite(predicted)):
            return None
        return log_residuals(observed, predicted)

    def score_params(params):
        residual = residual_for_params(params)
        return np.inf if residual is None else float(np.sqrt(np.mean(residual * residual)))

    initial_log_rmse = score_params(initial_params)
    state = {
        "best": initial_log_rmse,
        # Keep the exact physical starting point.  ``least_squares`` needs an
        # open-bound start, but returning that nudged point could otherwise be
        # microscopically worse when a current value sits exactly on a bound.
        "best_params": initial_params.copy(),
        "calls": 0,
        "completed_nfev": 0,
        "local_nfev_est": 0,
        "last_report_step": -1,
        "last_report_time": 0.0,
        "phase": "local_refine" if mode == "local" else "global_search",
        "work_done": 0,
        "local_start": 0,
    }

    def update_best(params, score):
        if np.isfinite(score) and score < float(state["best"]):
            state["best"] = float(score)
            state["best_params"] = np.asarray(params, dtype=float).copy()

    def emit_progress(current, *, force=False):
        if progress_callback is None:
            return
        step = int(state["work_done"])
        now = time.perf_counter()
        interval_due = (
            step != int(state["last_report_step"])
            and step % progress_interval == 0
            and now - float(state["last_report_time"]) >= min_progress_seconds
        )
        if not (force or int(state["calls"]) == 1 or interval_due):
            return
        state["last_report_step"] = step
        state["last_report_time"] = now
        progress_callback(
            _progress_payload(
                state["best_params"],
                variable_indices,
                initial_log_rmse,
                float(state["best"]),
                int(state["completed_nfev"]) + int(state["local_nfev_est"]),
                int(state["calls"]),
                max_nfev,
                show_interval,
                "running",
                False,
                current=current,
                mode=mode,
                phase=str(state["phase"]),
                work_done=step,
                work_total=total_work,
                global_samples=global_samples,
                global_starts=global_starts,
                local_start=int(state["local_start"]),
            )
        )

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
                mode=mode,
                phase=str(state["phase"]),
                work_done=0,
                work_total=total_work,
                global_samples=global_samples,
                global_starts=global_starts,
            )
        )

    stopped = False
    message = ""
    try:
        starts = [normalized_start]
        if mode == "global":
            amplitude_positions = [
                position
                for position, (item, _lo, _hi) in enumerate(selected)
                if parameter_base_name(item["name"]) in {"Int", "BG", "int_Res"}
            ]
            exploration_positions = [
                position
                for position in range(normalized_start.size)
                if position not in amplitude_positions
            ]
            candidate_cache = {}

            def evaluate_global_candidate(candidate):
                _raise_if_stopped(stop_callback)
                normalized = normalized_start.copy()
                if exploration_positions:
                    normalized[exploration_positions] = candidate
                params, score, model_calls = _profile_linear_amplitudes(
                    setup,
                    build_params(normalized),
                    selected,
                    amplitude_positions,
                    observed,
                    q_model,
                    model_func,
                )
                normalized = transform.encode(params[variable_indices])
                state["calls"] = int(state["calls"]) + int(model_calls)
                state["work_done"] = min(
                    global_samples,
                    int(state["work_done"]) + 1,
                )
                update_best(params, score)
                emit_progress(score)
                key = np.asarray(candidate, dtype=float).tobytes()
                candidate_cache[key] = (score, normalized)
                if target > 0 and score <= target:
                    raise RuntimeError(_TARGET_REACHED)
                return score

            scored = [(initial_log_rmse, normalized_start.copy())]
            if exploration_positions and global_samples >= 5:
                dimension = len(exploration_positions)
                population_size = min(global_samples, max(5, 6 * dimension))
                initial_population = qmc.LatinHypercube(
                    d=dimension,
                    seed=int(options.get("random_seed", 1729)),
                ).random(population_size)
                initial_population[0] = normalized_start[exploration_positions]
                generations = max(0, global_samples // population_size - 1)
                evolution = differential_evolution(
                    evaluate_global_candidate,
                    [(0.0, 1.0)] * dimension,
                    init=initial_population,
                    maxiter=generations,
                    seed=int(options.get("random_seed", 1729)),
                    polish=False,
                    tol=0.0,
                    atol=0.0,
                    updating="immediate",
                )
                for population_index in np.argsort(evolution.population_energies):
                    candidate = evolution.population[population_index]
                    key = np.asarray(candidate, dtype=float).tobytes()
                    cached = candidate_cache.get(key)
                    if cached is None:
                        evaluate_global_candidate(candidate)
                        cached = candidate_cache[key]
                    scored.append(cached)
            else:
                candidates = (
                    _sobol_candidates(
                        len(exploration_positions),
                        global_samples,
                        int(options.get("random_seed", 1729)),
                    )
                    if exploration_positions
                    else np.empty((1, 0), dtype=float)
                )
                for candidate in candidates:
                    evaluate_global_candidate(candidate)
                    scored.append(candidate_cache[np.asarray(candidate, dtype=float).tobytes()])
            state["work_done"] = global_samples
            scored.sort(key=lambda item: float(item[0]))
            starts = [candidate for _score, candidate in scored[:global_starts]]

        last_local_message = ""
        for start_number, start in enumerate(starts, start=1):
            _raise_if_stopped(stop_callback)
            state["phase"] = "local_refine"
            state["local_start"] = start_number
            local_calls = 0

            def residuals(normalized):
                nonlocal local_calls
                _raise_if_stopped(stop_callback)
                residual = residual_for_params(build_params(normalized))
                if residual is None:
                    residual = np.full_like(observed, 1e6, dtype=float)
                current = float(np.sqrt(np.mean(residual * residual)))
                local_calls += 1
                state["calls"] = int(state["calls"]) + 1
                state["local_nfev_est"] = max(1, int(np.ceil(local_calls / calls_per_nfev)))
                local_offset = (
                    0 if mode == "local" else global_samples + (start_number - 1) * max_nfev
                )
                state["work_done"] = local_offset + int(state["local_nfev_est"])
                update_best(build_params(normalized), current)
                emit_progress(current)
                if target > 0 and current <= target:
                    raise RuntimeError(_TARGET_REACHED)
                return residual

            result = least_squares(
                residuals,
                clamp_to_open_bounds(
                    start,
                    np.zeros_like(start),
                    np.ones_like(start),
                    epsilon=1e-12,
                ),
                bounds=(np.zeros_like(start), np.ones_like(start)),
                max_nfev=max_nfev,
                ftol=options.get("ftol"),
                xtol=options.get("xtol"),
                gtol=options.get("gtol"),
                x_scale="jac" if mode == "global" else 1.0,
            )
            result_score = score_params(build_params(result.x))
            update_best(build_params(result.x), result_score)
            state["completed_nfev"] = int(state["completed_nfev"]) + int(result.nfev)
            state["local_nfev_est"] = 0
            last_local_message = str(result.message)
        message = (
            f"Differential evolution plus {len(starts)} local start(s) completed. "
            f"Last local termination: {last_local_message}"
            if mode == "global"
            else last_local_message
        )
    except RuntimeError as exc:
        marker = str(exc)
        if marker not in {_TARGET_REACHED, _STOP_REQUESTED}:
            raise
        stopped = marker == _STOP_REQUESTED
        message = "Stopped by user." if stopped else "Stopped after reaching target logRMSE."

    final_params = np.asarray(state["best_params"], dtype=float).copy()
    final_log_rmse = score_params(final_params)
    final_nfev = int(state["completed_nfev"]) + int(state["local_nfev_est"])
    payload = _progress_payload(
        final_params,
        variable_indices,
        initial_log_rmse,
        final_log_rmse,
        final_nfev,
        int(state["calls"]),
        max_nfev,
        show_interval,
        message,
        stopped,
        mode=mode,
        phase="finished",
        work_done=int(state["work_done"]),
        work_total=total_work,
        global_samples=global_samples,
        global_starts=global_starts,
        local_start=int(state["local_start"]),
    )
    if progress_callback:
        progress_callback(payload)
    return payload


def _profile_linear_amplitudes(
    setup,
    params,
    selected,
    amplitude_positions,
    observed,
    q_model,
    model_func,
):
    """Eliminate selected linear amplitudes for one nonlinear candidate."""

    params = np.asarray(params, dtype=float).copy()
    if not amplitude_positions:
        return params, _score_model(model_func, q_model, observed, params), 1

    amplitude_indices = [int(selected[position][0]["index"]) for position in amplitude_positions]
    lower = np.asarray([selected[position][1] for position in amplitude_positions], dtype=float)
    upper = np.asarray([selected[position][2] for position in amplitude_positions], dtype=float)
    try:
        base, basis, model_calls = _linear_amplitude_basis(
            setup,
            params,
            amplitude_indices,
            q_model,
            model_func,
        )
        coefficients = None
        weights = 1.0 / np.maximum(observed, 1e-30)
        for _iteration in range(3):
            result = lsq_linear(
                basis * weights[:, np.newaxis],
                (observed - base) * weights,
                bounds=(lower, upper),
                method="bvls",
            )
            coefficients = np.asarray(result.x, dtype=float)
            predicted = base + basis @ coefficients
            weights = 1.0 / np.maximum(predicted, 1e-30)
        if coefficients is None or np.any(~np.isfinite(coefficients)):
            raise ValueError("Linear amplitude profiling returned invalid values")
        params[amplitude_indices] = coefficients
        predicted = base + basis @ coefficients
        if np.any(~np.isfinite(predicted)):
            raise ValueError("Linear amplitude profiling returned an invalid curve")
        score = float(np.sqrt(np.mean(log_residuals(observed, predicted) ** 2)))
        return params, score, model_calls
    except Exception:
        return params, _score_model(model_func, q_model, observed, params), 1


def _linear_amplitude_basis(setup, params, amplitude_indices, q_model, model_func):
    """Return the affine base and columns for selected amplitude parameters."""

    names = [str(item.get("name", "")) for item in setup.get("params", ())]
    shapes = list(setup.get("shapes", ()))
    if shapes and len(names) == len(params):
        try:
            probe = np.asarray(params, dtype=float).copy()
            for index in amplitude_indices:
                probe[index] = 0.0 if parameter_base_name(names[index]) == "BG" else 1.0
            components = mixed_model_components(shapes, q_model, probe)
            base = np.zeros_like(q_model, dtype=float)
            curves = {}
            particle_cursor = 0
            for index, name in enumerate(names):
                base_name = parameter_base_name(name)
                if base_name == "Int":
                    curve = np.asarray(
                        components["particles"][particle_cursor]["I"],
                        dtype=float,
                    )
                    particle_cursor += 1
                    if index in amplitude_indices:
                        curves[index] = curve
                    else:
                        base += curve
                elif base_name == "int_Res":
                    curve = np.asarray(components["resolution"], dtype=float)
                    if index in amplitude_indices:
                        curves[index] = curve
                    else:
                        base += curve
                elif base_name == "BG":
                    curve = np.ones_like(q_model, dtype=float)
                    if index in amplitude_indices:
                        curves[index] = curve
                    else:
                        base += np.asarray(components["BG_total"], dtype=float)
            return base, np.column_stack([curves[index] for index in amplitude_indices]), 1
        except (IndexError, KeyError, TypeError, ValueError):
            pass

    probe = np.asarray(params, dtype=float).copy()
    probe[amplitude_indices] = 0.0
    base = np.asarray(model_func(q_model, *probe), dtype=float)
    columns = []
    for index in amplitude_indices:
        unit = probe.copy()
        unit[index] = 1.0
        columns.append(np.asarray(model_func(q_model, *unit), dtype=float) - base)
    return base, np.column_stack(columns), len(amplitude_indices) + 1


def _score_model(model_func, q_model, observed, params) -> float:
    try:
        with np.errstate(all="ignore"):
            predicted = np.asarray(model_func(q_model, *params), dtype=float)
        if predicted.shape != observed.shape or not np.all(np.isfinite(predicted)):
            return np.inf
        residual = log_residuals(observed, predicted)
        return float(np.sqrt(np.mean(residual * residual)))
    except Exception:
        return np.inf


def _sobol_candidates(dimension: int, count: int, seed: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, dimension), dtype=float)
    power = int(np.ceil(np.log2(count)))
    return qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(power)[:count]


def _raise_if_stopped(stop_callback) -> None:
    if stop_callback and stop_callback():
        raise RuntimeError(_STOP_REQUESTED)


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
    mode="local",
    phase="local_refine",
    work_done=0,
    work_total=0,
    global_samples=0,
    global_starts=1,
    local_start=0,
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
        "mode": str(mode),
        "phase": str(phase),
        "work_done": int(work_done),
        "work_total": int(work_total),
        "global_samples": int(global_samples),
        "global_starts": int(global_starts),
        "local_start": int(local_start),
    }
