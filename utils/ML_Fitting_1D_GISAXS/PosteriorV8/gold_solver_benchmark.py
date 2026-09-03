"""Slurm-friendly sequential-vs-joint exact gold benchmark.

Both solvers receive the same simulated curve, hard branch, codec seed and
log-residual metric.  The resulting JSON is an audit artifact, not an
acceptance decision and not part of production inference.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from unittest.mock import patch

import numpy as np
import scipy

from . import amplitude_polish as amplitude_polish_module
from . import joint_exact_optimization as joint_module
from . import profiled_refinement as refinement_module
from .amplitude_polish import AMPLITUDE_POLISH_VERSION
from .branch_codec import BRANCH_CODEC_VERSION, ResolutionBounds
from .contract import (
    RESOLUTION_NU_DOMAIN,
    RESOLUTION_SIGMA_DOMAIN,
    TOPOLOGIES,
    full_component_bounds,
)
from .evaluation import (
    RAW_LOG_RMSE_METRIC,
    STANDARDIZED_LOG_RMSE_METRIC,
    natural_log_rmse,
)
from .joint_exact_optimization import JOINT_EXACT_OPTIMIZATION_VERSION
from .profiled_forward import evaluate_profiled_forward
from .profiled_refinement import ProfiledRefinementResult
from .proposal_sampling import generate_profiled_branch_seeds
from .simulation import NoiseProvenance, sample_identifiable_recipe, simulate_recipe


GOLD_BENCHMARK_SCHEMA = "gisaxs.posterior_v8.sequential_joint_gold_benchmark/v2"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    module_root = Path(__file__).resolve().parent
    repository = Path(__file__).resolve().parents[3]
    sources = {
        name: module_root / name
        for name in (
            "gold_solver_benchmark.py",
            "joint_exact_optimization.py",
            "joint_gui_amplitude_coordinates.py",
            "gui_amplitude_constraints.py",
            "amplitude_polish.py",
            "profiled_refinement.py",
            "profiled_forward.py",
            "proposal_sampling.py",
            "simulation.py",
            "branch_codec.py",
            "contract.py",
        )
    }
    sources.update(
        {
            "src/gimap/features/fitting/domain/scattering_model.py": repository
            / "src/gimap/features/fitting/domain/scattering_model.py",
            "src/gimap/features/fitting/domain/physical_constraints.py": repository
            / "src/gimap/features/fitting/domain/physical_constraints.py",
        }
    )
    return {name: _file_sha256(path) for name, path in sorted(sources.items())}


@dataclass(frozen=True, kw_only=True)
class GoldBenchmarkConfig:
    curves: int = 1
    component_schedule: tuple[int, ...] = (2,)
    noise_mode: str = "clean"
    points: int = 64
    seed: int = 20260902
    sequential_max_nfev: int = 20
    amplitude_polish_max_nfev: int = 20
    joint_max_exact_evaluations: int = 64

    def __post_init__(self) -> None:
        for name in (
            "curves",
            "points",
            "sequential_max_nfev",
            "amplitude_polish_max_nfev",
            "joint_max_exact_evaluations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 64 <= self.points <= 1000:
            raise ValueError("points must be between 64 and 1000")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        schedule = tuple(self.component_schedule)
        if not schedule or any(
            isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4
            for value in schedule
        ):
            raise ValueError("component_schedule must contain only K values in 1..4")
        if self.noise_mode not in {"clean", "default"}:
            raise ValueError("noise_mode must be clean or default")
        object.__setattr__(self, "component_schedule", schedule)


def _topology_ids(component_count: int) -> tuple[int, ...]:
    return tuple(
        topology_id
        for topology_id, topology in enumerate(TOPOLOGIES)
        if len(topology) == component_count
    )


def _curve_digest(q, intensity, sigma_log) -> str:
    digest = hashlib.sha256()
    for value in (q, intensity, sigma_log):
        if value is None:
            digest.update(b"none")
        else:
            array = np.ascontiguousarray(value, dtype="<f8")
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _comparison_input_id(curve_sha256, branch_seed, metric_name):
    encoded = json.dumps(
        {
            "curve_sha256": curve_sha256,
            # Include the complete hard-branch request, not only its unit
            # coordinates: identical coordinates have different physical
            # meaning under different topology or user-bound codecs.
            "branch_seed": asdict(branch_seed),
            "metric_name": metric_name,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metric_values(exact, observed, sigma_log):
    raw = natural_log_rmse(exact, observed)
    standardized = (
        None if sigma_log is None else natural_log_rmse(exact, observed, sigma_log=sigma_log)
    )
    return float(raw), None if standardized is None else float(standardized)


def _amplitude_bounds_payload(bounds):
    return {
        "lower": [float(value) for value in bounds.lower],
        "upper": [None if np.isposinf(value) else float(value) for value in bounds.upper],
        "null_upper_means_unbounded": True,
    }


def _run_sequential(q, intensity, sigma_log, metric_name, kwargs, config):
    exact_calls = 0

    def counted_exact(q_values, profile):
        nonlocal exact_calls
        exact_calls += 1
        return evaluate_profiled_forward(q_values, profile)

    started = time.perf_counter()
    with (
        patch.object(refinement_module, "evaluate_profiled_forward", counted_exact),
        patch.object(amplitude_polish_module, "evaluate_profiled_forward", counted_exact),
    ):
        refined: ProfiledRefinementResult = refinement_module.refine_profiled_branch(
            q,
            intensity,
            sigma_log=sigma_log,
            max_nfev=config.sequential_max_nfev,
            **kwargs,
        )
        polished = amplitude_polish_module.polish_profiled_amplitudes(
            q,
            intensity,
            refined.final_profile,
            sigma_log=sigma_log,
            metric_name=metric_name,
            max_nfev=config.amplitude_polish_max_nfev,
        )
    wall = time.perf_counter() - started
    raw, standardized = _metric_values(polished.exact_intensity, intensity, sigma_log)
    initial_metric = (
        refined.initial_log_rmse
        if metric_name == RAW_LOG_RMSE_METRIC
        else refined.initial_weighted_log_rmse
    )
    return {
        "returned": True,
        "optimizer_success": bool(refined.success and polished.success),
        "wall_seconds": wall,
        "exact_forward_calls": exact_calls,
        "initial_metric": float(initial_metric),
        "final_metric": raw if metric_name == RAW_LOG_RMSE_METRIC else standardized,
        "final_raw_log_rmse": raw,
        "final_standardized_log_rmse": standardized,
        "bounds_satisfied": bool(refined.bounds_satisfied and polished.bounds_satisfied),
        "amplitude_bounds": _amplitude_bounds_payload(polished.coefficient_bounds),
        "best_seen": {
            "geometry_refiner_retains_best_seen": True,
            "amplitude_returned_source": polished.returned_source,
            "amplitude_best_residual_call": polished.best_residual_call,
        },
        "refinement": {
            "nfev": refined.nfev,
            "njev": refined.njev,
            "residual_calls": refined.residual_calls,
            "status": refined.status,
            "message": refined.message,
        },
        "amplitude_polish": {
            "nfev": polished.nfev,
            "njev": polished.njev,
            "residual_calls": polished.residual_calls,
            "status": polished.status,
            "message": polished.message,
        },
    }


def _run_joint(q, intensity, sigma_log, metric_name, kwargs, config):
    started = time.perf_counter()
    result = joint_module.optimize_joint_exact_branch(
        q,
        intensity,
        sigma_log=sigma_log,
        metric_name=metric_name,
        max_exact_evaluations=config.joint_max_exact_evaluations,
        **kwargs,
    )
    wall = time.perf_counter() - started
    return {
        "returned": True,
        "optimizer_success": result.success,
        "wall_seconds": wall,
        "exact_forward_calls": result.exact_forward_calls,
        "max_exact_evaluations": result.max_exact_evaluations,
        "budget_exhausted": result.budget_exhausted,
        "objective_requests": result.objective_requests,
        "cache_hits": result.cache_hits,
        "initial_metric": result.initial_metric,
        "final_metric": result.final_metric,
        "final_raw_log_rmse": result.final_raw_log_rmse,
        "final_standardized_log_rmse": result.final_standardized_log_rmse,
        "bounds_satisfied": result.bounds_satisfied,
        "best_seen": {
            "returned_source": result.returned_source,
            "best_exact_call": result.best_exact_call,
        },
        "effective_amplitude_bounds": _amplitude_bounds_payload(result.effective_amplitude_bounds),
        "amplitude_scales": result.amplitude_scales,
        "optimizer": {
            "nfev": result.optimizer_nfev,
            "njev": result.optimizer_njev,
            "status": result.status,
            "message": result.message,
        },
    }


def _method_failure(started: float, exc: Exception) -> dict[str, object]:
    return {
        "returned": False,
        "optimizer_success": False,
        "wall_seconds": time.perf_counter() - started,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _case(config: GoldBenchmarkConfig, curve_index: int) -> dict[str, object]:
    component_count = config.component_schedule[curve_index % len(config.component_schedule)]
    choices = _topology_ids(component_count)
    topology_id = choices[(curve_index // len(config.component_schedule)) % len(choices)]
    recipe_seed = int(
        np.random.SeedSequence([config.seed, curve_index, 0x60AD]).generate_state(
            1, dtype=np.uint32
        )[0]
    )
    clean_noise = NoiseProvenance(
        poisson_count_scale=None,
        relative_sigma=0.0,
        sigma_floor_fraction=1.0e-12,
    )
    recipe = sample_identifiable_recipe(
        recipe_seed,
        topology_id=topology_id,
        max_points=config.points,
        noise=clean_noise if config.noise_mode == "clean" else None,
    )
    simulated = simulate_recipe(recipe)
    sigma_log = None if config.noise_mode == "clean" else simulated.sigma / simulated.intensity
    metric_name = RAW_LOG_RMSE_METRIC if sigma_log is None else STANDARDIZED_LOG_RMSE_METRIC
    topology = TOPOLOGIES[topology_id]
    component_bounds = tuple(
        full_component_bounds(shape, d_policy="optional") for shape in topology
    )
    resolution_bounds = (
        None
        if recipe.resolution is None
        else ResolutionBounds(RESOLUTION_SIGMA_DOMAIN, RESOLUTION_NU_DOMAIN)
    )
    branch_seed = generate_profiled_branch_seeds(
        topology,
        component_bounds,
        tuple(component.D is not None for component in recipe.components),
        resolution_bounds=resolution_bounds,
        seed=config.seed + 7919 * curve_index,
        count=1,
    )[0]
    kwargs = branch_seed.refinement_kwargs()

    sequential_started = time.perf_counter()
    try:
        sequential = _run_sequential(
            simulated.q,
            simulated.intensity,
            sigma_log,
            metric_name,
            kwargs,
            config,
        )
    except (FloatingPointError, RuntimeError, TypeError, ValueError) as exc:
        sequential = _method_failure(sequential_started, exc)
    joint_started = time.perf_counter()
    try:
        joint = _run_joint(
            simulated.q,
            simulated.intensity,
            sigma_log,
            metric_name,
            kwargs,
            config,
        )
    except (FloatingPointError, RuntimeError, TypeError, ValueError) as exc:
        joint = _method_failure(joint_started, exc)

    comparison = None
    if sequential["returned"] and joint["returned"]:
        comparison = {
            "initial_metric_absolute_delta": abs(
                float(sequential["initial_metric"]) - float(joint["initial_metric"])
            ),
            "joint_minus_sequential_final_metric": (
                float(joint["final_metric"]) - float(sequential["final_metric"])
            ),
            "joint_minus_sequential_exact_forward_calls": (
                int(joint["exact_forward_calls"]) - int(sequential["exact_forward_calls"])
            ),
            "joint_minus_sequential_wall_seconds": (
                float(joint["wall_seconds"]) - float(sequential["wall_seconds"])
            ),
        }
    curve_sha256 = _curve_digest(simulated.q, simulated.intensity, sigma_log)
    return {
        "case_index": curve_index,
        "comparison_input_id": _comparison_input_id(curve_sha256, branch_seed, metric_name),
        "curve": {
            "curve_sha256": curve_sha256,
            "topology_id": topology_id,
            "topology": topology,
            "component_count": component_count,
            "points": int(simulated.q.size),
            "noise_mode": config.noise_mode,
            "metric_name": metric_name,
            "recipe_seed": recipe_seed,
        },
        "branch": {
            "codec_version": branch_seed.codec_version,
            "d_present": branch_seed.d_present,
            "resolution_present": branch_seed.resolution_seed is not None,
            "component_bounds": [asdict(value) for value in component_bounds],
            "resolution_bounds": (None if resolution_bounds is None else asdict(resolution_bounds)),
            "seed_components": [asdict(value) for value in branch_seed.seed_components],
            "resolution_seed": (
                None if branch_seed.resolution_seed is None else asdict(branch_seed.resolution_seed)
            ),
            "active_unit_coordinates": [
                value
                for value, active in zip(branch_seed.unit_cube, branch_seed.active_mask)
                if active
            ],
        },
        "sequential": sequential,
        "joint": joint,
        "comparison": comparison,
    }


def _mean(values):
    values = tuple(float(value) for value in values)
    return None if not values else float(np.mean(values))


def run_gold_benchmark(
    config: GoldBenchmarkConfig,
    output: str | Path,
) -> dict[str, object]:
    """Run a serial smoke/formal audit and atomically create one JSON file."""

    if not isinstance(config, GoldBenchmarkConfig):
        raise TypeError("config must be a GoldBenchmarkConfig")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cases = [_case(config, index) for index in range(config.curves)]
    paired = [item for item in cases if item["comparison"] is not None]
    source_sha256 = _source_hashes()
    payload = {
        "schema": GOLD_BENCHMARK_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "versions": {
            "branch_codec": BRANCH_CODEC_VERSION,
            "amplitude_polish": AMPLITUDE_POLISH_VERSION,
            "joint_exact": JOINT_EXACT_OPTIMIZATION_VERSION,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "source_sha256": source_sha256,
        "source_sha256_aggregate": hashlib.sha256(
            json.dumps(
                source_sha256,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "summary": {
            "case_count": len(cases),
            "paired_return_count": len(paired),
            "sequential_return_count": sum(item["sequential"]["returned"] for item in cases),
            "joint_return_count": sum(item["joint"]["returned"] for item in cases),
            "mean_sequential_final_metric": _mean(
                item["sequential"]["final_metric"] for item in paired
            ),
            "mean_joint_final_metric": _mean(item["joint"]["final_metric"] for item in paired),
            "mean_sequential_exact_forward_calls": _mean(
                item["sequential"]["exact_forward_calls"] for item in paired
            ),
            "mean_joint_exact_forward_calls": _mean(
                item["joint"]["exact_forward_calls"] for item in paired
            ),
        },
        "cases": cases,
        "wall_seconds": time.perf_counter() - started,
    }
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(destination)
    return payload


def _schedule(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "component schedule must be comma-separated integers"
        ) from exc
    if not result or any(not 1 <= item <= 4 for item in result):
        raise argparse.ArgumentTypeError("component schedule values must lie in 1..4")
    return result


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=os.environ.get("POSTERIOR_V8_GOLD_OUTPUT"),
    )
    parser.add_argument("--curves", type=int, default=_env_int("POSTERIOR_V8_GOLD_CURVES", 1))
    parser.add_argument(
        "--component-schedule",
        type=_schedule,
        default=_schedule(os.environ.get("POSTERIOR_V8_GOLD_K_SCHEDULE", "2")),
    )
    parser.add_argument(
        "--noise-mode",
        choices=("clean", "default"),
        default=os.environ.get("POSTERIOR_V8_GOLD_NOISE_MODE", "clean"),
    )
    parser.add_argument("--points", type=int, default=_env_int("POSTERIOR_V8_GOLD_POINTS", 64))
    parser.add_argument("--seed", type=int, default=_env_int("POSTERIOR_V8_GOLD_SEED", 20260902))
    parser.add_argument(
        "--sequential-max-nfev",
        type=int,
        default=_env_int("POSTERIOR_V8_GOLD_SEQUENTIAL_MAX_NFEV", 20),
    )
    parser.add_argument(
        "--amplitude-polish-max-nfev",
        type=int,
        default=_env_int("POSTERIOR_V8_GOLD_AMPLITUDE_MAX_NFEV", 20),
    )
    parser.add_argument(
        "--joint-max-exact-evaluations",
        type=int,
        default=_env_int("POSTERIOR_V8_GOLD_JOINT_MAX_EXACT", 64),
    )
    args = parser.parse_args(argv)
    if args.output is None:
        parser.error("--output or POSTERIOR_V8_GOLD_OUTPUT is required")
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    config = GoldBenchmarkConfig(
        curves=args.curves,
        component_schedule=args.component_schedule,
        noise_mode=args.noise_mode,
        points=args.points,
        seed=args.seed,
        sequential_max_nfev=args.sequential_max_nfev,
        amplitude_polish_max_nfev=args.amplitude_polish_max_nfev,
        joint_max_exact_evaluations=args.joint_max_exact_evaluations,
    )
    payload = run_gold_benchmark(config, args.output)
    print(json.dumps(payload["summary"], allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GOLD_BENCHMARK_SCHEMA",
    "GoldBenchmarkConfig",
    "run_gold_benchmark",
]
