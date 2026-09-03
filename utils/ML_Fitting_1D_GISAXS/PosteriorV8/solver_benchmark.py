"""Slurm-friendly exact-solver upper-bound benchmark for Posterior V8."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

from src.gimap.features.fitting.domain.physical_constraints import ConstraintSet

from .amplitude_polish import polish_profiled_amplitudes
from .contract import (
    RESOLUTION_NU_DOMAIN,
    RESOLUTION_SIGMA_DOMAIN,
    TOPOLOGIES,
    full_component_bounds,
    gui_component_to_latent,
    latent_component_to_gui,
)
from .branch_codec import ResolutionBounds
from .evaluation import (
    EVALUATION_AUDIT_SCHEMA,
    REFERENCE_MATCHING_VERSION,
    REFERENCE_SET_DISTANCE_VERSION,
    CandidateInput,
    EvaluationThresholds,
    LinearSolutionSnapshot,
    ObservedCurve,
    RAW_LOG_RMSE_METRIC,
    ReferenceMode,
    STANDARDIZED_LOG_RMSE_METRIC,
    evaluate_candidates,
    natural_log_rmse,
)
from .profiled_refinement import refine_profiled_branch
from .proposal_sampling import PROPOSAL_SAMPLING_VERSION, generate_profiled_branch_seeds
from .simulation import (
    SIMULATION_VERSION,
    NoiseProvenance,
    sample_identifiable_recipe,
    simulate_recipe,
)


BENCHMARK_SCHEMA = "gisaxs.posterior_v8.solver_benchmark/v5"
K1_TOPOLOGY_IDS = (0, 1, 2)
MIN_EFFECTIVE_PARTICLE_WEIGHT = 0.01
MIN_EFFECTIVE_RESOLUTION_RATIO = 1.0e-4
DEFAULT_THRESHOLDS = EvaluationThresholds(
    raw_exact_log_rmse_max=0.05,
    standardized_exact_log_rmse_max=1.5,
    parameter_mode_distance_max=0.03,
    raw_curve_equivalence_log_rmse_max=0.01,
    reference_mode_distance_max=0.05,
)


@dataclass(frozen=True, kw_only=True)
class BenchmarkConfig:
    mode: str
    noise_mode: str
    curves: int
    starts_per_branch: int
    max_nfev: int
    points: int
    seed: int
    workers: int
    amplitude_polish_max_nfev: int = 80

    def __post_init__(self) -> None:
        if self.mode not in {"oracle_k1", "search_k1"}:
            raise ValueError("mode must be oracle_k1 or search_k1")
        if self.noise_mode not in {"clean", "default"}:
            raise ValueError("noise_mode must be clean or default")
        for name in (
            "curves",
            "starts_per_branch",
            "max_nfev",
            "workers",
            "amplitude_polish_max_nfev",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.points, bool) or int(self.points) != self.points:
            raise ValueError("points must be an integer")
        if not 64 <= int(self.points) <= 1000:
            raise ValueError("points must be between 64 and 1000")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolution_bounds() -> ResolutionBounds:
    return ResolutionBounds(RESOLUTION_SIGMA_DOMAIN, RESOLUTION_NU_DOMAIN)


def _physics_violations(components) -> list[dict[str, object]]:
    payload = []
    for latent in components:
        value = latent_component_to_gui(latent)
        payload.append(
            {
                "type": value.shape,
                "params": {
                    "R": value.R,
                    "sigma_R": value.sigma_R,
                    "h": value.h,
                    "sigma_h": value.sigma_h,
                    "D": 0.0 if value.D is None else value.D,
                    "sigma_D": 0.0 if value.sigma_D is None else value.sigma_D,
                },
            }
        )
    return [
        asdict(value)
        for value in ConstraintSet.defaults().validate_components(payload)
    ]


def _visibility_violations(profile) -> list[str]:
    violations = []
    if any(
        weight < MIN_EFFECTIVE_PARTICLE_WEIGHT
        for weight in profile.effective_component_fractions
    ):
        violations.append("particle_effective_weight_below_threshold")
    if (
        profile.resolution is not None
        and profile.effective_resolution_ratio < MIN_EFFECTIVE_RESOLUTION_RATIO
    ):
        violations.append("resolution_effective_amplitude_below_threshold")
    return violations


def _branches(mode: str, recipe):
    if mode == "oracle_k1":
        topology = TOPOLOGIES[recipe.topology_id]
        flags = tuple(component.D is not None for component in recipe.components)
        yield topology, flags, recipe.resolution is not None
        return
    for topology_id in K1_TOPOLOGY_IDS:
        topology = TOPOLOGIES[topology_id]
        for d_present in (False, True):
            for resolution_present in (False, True):
                yield topology, (d_present,), resolution_present


def _recipe_payload(recipe) -> dict[str, object]:
    result = asdict(recipe)
    result["components"] = [asdict(value) for value in recipe.components]
    result["resolution"] = None if recipe.resolution is None else asdict(recipe.resolution)
    return result


def _curve_task(config: BenchmarkConfig, curve_index: int) -> dict[str, object]:
    started = time.monotonic()
    topology_id = K1_TOPOLOGY_IDS[curve_index % len(K1_TOPOLOGY_IDS)]
    recipe_seed = int(
        np.random.SeedSequence([config.seed, curve_index, 0xB38]).generate_state(
            1, dtype=np.uint32
        )[0]
    )
    recipe = sample_identifiable_recipe(
        recipe_seed,
        topology_id=topology_id,
        max_points=config.points,
        noise=(
            NoiseProvenance(
                poisson_count_scale=None,
                relative_sigma=0.0,
                sigma_floor_fraction=1.0e-12,
            )
            if config.noise_mode == "clean"
            else None
        ),
    )
    simulated = simulate_recipe(recipe)
    sigma_log = (
        None
        if config.noise_mode == "clean"
        else simulated.sigma / simulated.intensity
    )
    observed = ObservedCurve(
        curve_id=f"synthetic_k1_{curve_index:05d}",
        source_kind="synthetic",
        q=simulated.q,
        intensity=simulated.intensity,
        sigma_log=sigma_log,
    )
    truth = ReferenceMode(
        reference_id="generating_mode",
        topology_id=recipe.topology_id,
        components=tuple(gui_component_to_latent(value) for value in recipe.components),
        resolution=recipe.resolution,
        linear_solution=LinearSolutionSnapshot(
            background=recipe.background,
            particle_amplitudes=recipe.effective_amplitudes,
            resolution_amplitude=recipe.resolution_effective_amplitude,
        ),
    )

    branch_specs = tuple(_branches(config.mode, recipe))
    branch_seed_groups = []
    failures = []
    for branch_index, (topology, d_present, resolution_present) in enumerate(branch_specs):
        bounds = tuple(full_component_bounds(shape, d_policy="optional") for shape in topology)
        try:
            seeds = generate_profiled_branch_seeds(
                topology,
                bounds,
                d_present,
                resolution_bounds=_resolution_bounds() if resolution_present else None,
                seed=config.seed + curve_index * 1009 + branch_index,
                count=config.starts_per_branch,
            )
        except (TypeError, ValueError) as exc:
            failures.append(
                {
                    "stage": "seed_generation",
                    "branch_index": branch_index,
                    "topology": topology,
                    "d_present": d_present,
                    "resolution_present": resolution_present,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        branch_seed_groups.append((branch_index, seeds))

    expected_attempts = len(branch_specs) * config.starts_per_branch
    candidates = []
    attempt_records = []
    proposal_rank = 0
    # Round-robin gives every branch one attempt before any branch gets a
    # second attempt.  Missing/failed attempts retain their schedule rank.
    for sequence_index in range(config.starts_per_branch):
        for branch_index, seeds in branch_seed_groups:
            seed = seeds[sequence_index]
            attempt_rank = sequence_index * len(branch_specs) + branch_index + 1
            call_started = time.monotonic()
            try:
                result = refine_profiled_branch(
                    simulated.q,
                    simulated.intensity,
                    sigma_log=sigma_log,
                    max_nfev=config.max_nfev,
                    ftol=1.0e-7,
                    xtol=1.0e-7,
                    gtol=1.0e-7,
                    **seed.refinement_kwargs(),
                )
            except (FloatingPointError, RuntimeError, TypeError, ValueError) as exc:
                failure = {
                    "stage": "refinement",
                    "attempt_rank": attempt_rank,
                    "branch_index": branch_index,
                    "sequence_index": sequence_index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                attempt_records.append(
                    {
                        **failure,
                        "status": "exception",
                        "elapsed_seconds": time.monotonic() - call_started,
                    }
                )
                continue
            try:
                polished = polish_profiled_amplitudes(
                    simulated.q,
                    simulated.intensity,
                    result.final_profile,
                    sigma_log=sigma_log,
                    metric_name=(
                        RAW_LOG_RMSE_METRIC
                        if sigma_log is None
                        else STANDARDIZED_LOG_RMSE_METRIC
                    ),
                    max_nfev=config.amplitude_polish_max_nfev,
                )
                final_profile = polished.final_profile
                final_exact_intensity = polished.exact_intensity
            except (FloatingPointError, RuntimeError, TypeError, ValueError) as exc:
                polished = None
                final_profile = result.final_profile
                final_exact_intensity = result.exact_forward_intensity
                failures.append(
                    {
                        "stage": "amplitude_polish",
                        "attempt_rank": attempt_rank,
                        "branch_index": branch_index,
                        "sequence_index": sequence_index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            proposal_rank += 1
            candidate_id = f"candidate_{proposal_rank:05d}"
            physics_violations = _physics_violations(result.final_latent_components)
            visibility_violations = _visibility_violations(final_profile)
            physics_pass = not physics_violations and not visibility_violations
            candidates.append(
                CandidateInput(
                    candidate_id=candidate_id,
                    proposal_rank=proposal_rank,
                    topology_id=seed.topology_id,
                    components=result.final_latent_components,
                    resolution=result.final_resolution,
                    linear_solution=LinearSolutionSnapshot.from_profiled_forward(
                        final_profile
                    ),
                    exact_intensity=final_exact_intensity,
                    bounds_pass=result.bounds_satisfied,
                    physics_pass=physics_pass,
                )
            )
            attempt_records.append(
                {
                    "attempt_rank": attempt_rank,
                    "candidate_id": candidate_id,
                    "branch_index": branch_index,
                    "sequence_index": sequence_index,
                    "status": "returned",
                    "nfev": result.nfev,
                    "residual_calls": result.residual_calls,
                    "optimizer_converged": result.success,
                    "initial_raw_log_rmse": result.initial_log_rmse,
                    "final_raw_log_rmse": result.final_log_rmse,
                    "initial_standardized_log_rmse": (
                        None
                        if sigma_log is None
                        else result.initial_weighted_log_rmse
                    ),
                    "final_standardized_log_rmse": (
                        None
                        if sigma_log is None
                        else result.final_weighted_log_rmse
                    ),
                    "amplitude_polish_applied": polished is not None,
                    "amplitude_polish_success": (
                        None if polished is None else polished.success
                    ),
                    "amplitude_polish_nfev": (
                        0 if polished is None else polished.nfev
                    ),
                    "amplitude_polish_residual_calls": (
                        0 if polished is None else polished.residual_calls
                    ),
                    "amplitude_polish_returned_source": (
                        None if polished is None else polished.returned_source
                    ),
                    "post_polish_raw_log_rmse": natural_log_rmse(
                        final_exact_intensity, simulated.intensity
                    ),
                    "post_polish_standardized_log_rmse": (
                        None
                        if sigma_log is None
                        else natural_log_rmse(
                            final_exact_intensity,
                            simulated.intensity,
                            sigma_log=sigma_log,
                        )
                    ),
                    "raw_log_rmse_improvement": (
                        result.final_log_rmse
                        - natural_log_rmse(final_exact_intensity, simulated.intensity)
                    ),
                    "standardized_log_rmse_improvement": (
                        None
                        if sigma_log is None
                        else result.final_weighted_log_rmse
                        - natural_log_rmse(
                            final_exact_intensity,
                            simulated.intensity,
                            sigma_log=sigma_log,
                        )
                    ),
                    "bounds_pass": result.bounds_satisfied,
                    "physics_violations": physics_violations,
                    "visibility_violations": visibility_violations,
                    "elapsed_seconds": time.monotonic() - call_started,
                }
            )

    if not candidates:
        raise RuntimeError(f"curve {curve_index} produced no solver candidate: {failures[:3]}")
    candidate_cutoffs = sorted({1, len(candidates)})
    report = evaluate_candidates(
        observed,
        candidates,
        thresholds=DEFAULT_THRESHOLDS,
        best_of_n=candidate_cutoffs,
        reference_modes=(truth,),
    )
    audit = report.to_audit_dict()
    assessments = {item["candidate_id"]: item for item in audit["candidates"]}
    for record in attempt_records:
        candidate_id = record.get("candidate_id")
        if candidate_id is not None:
            assessment = assessments[candidate_id]
            record.update(
                {
                    "primary_gate_error": assessment["primary_gate_error"],
                    "raw_exact_log_rmse": assessment["raw_exact_log_rmse"],
                    "standardized_exact_log_rmse": assessment[
                        "standardized_exact_log_rmse"
                    ],
                    "accepted": assessment["accepted"],
                }
            )

    if config.mode == "search_k1":
        rounds = (1, 2, 4, 8, 16)
        attempt_cutoffs = sorted(
            {
                len(branch_specs) * min(rounds_count, config.starts_per_branch)
                for rounds_count in rounds
            }
        )
    else:
        attempt_cutoffs = sorted(
            {1, min(8, expected_attempts), expected_attempts}
        )
    attempt_best = []
    for cutoff in attempt_cutoffs:
        eligible = [
            record
            for record in attempt_records
            if record["attempt_rank"] <= cutoff and record.get("accepted", False)
        ]
        best = min(eligible, key=lambda item: item["primary_gate_error"], default=None)
        attempt_best.append(
            {
                "attempts": cutoff,
                "accepted_found": best is not None,
                "best_candidate_id": None if best is None else best["candidate_id"],
                "best_primary_gate_error": (
                    None if best is None else best["primary_gate_error"]
                ),
                "best_raw_exact_log_rmse": (
                    None if best is None else best["raw_exact_log_rmse"]
                ),
            }
        )

    accepted = [item for item in audit["candidates"] if item["accepted"]]
    best_accepted = min(
        accepted, key=lambda item: item["primary_gate_error"], default=None
    )
    raw_noise = natural_log_rmse(simulated.clean_intensity, simulated.intensity)
    standardized_noise = (
        None
        if sigma_log is None
        else natural_log_rmse(
            simulated.clean_intensity,
            simulated.intensity,
            sigma_log=sigma_log,
        )
    )
    truth_primary = raw_noise if standardized_noise is None else standardized_noise
    truth_threshold = (
        DEFAULT_THRESHOLDS.raw_exact_log_rmse_max
        if standardized_noise is None
        else DEFAULT_THRESHOLDS.standardized_exact_log_rmse_max
    )
    returned = [item for item in attempt_records if item["status"] == "returned"]
    return {
        "curve_index": curve_index,
        "recipe": _recipe_payload(recipe),
        "truth_noise_baseline": {
            "raw_log_rmse": raw_noise,
            "standardized_log_rmse": standardized_noise,
            "primary_gate_error": truth_primary,
            "primary_gate_threshold": truth_threshold,
            "truth_gate_pass": truth_primary <= truth_threshold,
        },
        "evaluation": audit,
        "attempt_best": attempt_best,
        "attempt_records": attempt_records,
        "failures": failures,
        "search_accounting": {
            "expected_branches": len(branch_specs),
            "generated_branches": len(branch_seed_groups),
            "expected_attempts": expected_attempts,
            "attempted_refinements": len(attempt_records),
            "returned_candidates": len(returned),
            "optimizer_converged": sum(
                bool(item.get("optimizer_converged")) for item in returned
            ),
            "amplitude_polish_attempted": len(returned),
            "amplitude_polish_completed": sum(
                bool(item.get("amplitude_polish_applied")) for item in returned
            ),
            "amplitude_polish_optimizer_converged": sum(
                bool(item.get("amplitude_polish_success")) for item in returned
            ),
            "amplitude_polish_nfev": sum(
                int(item.get("amplitude_polish_nfev", 0)) for item in returned
            ),
            "amplitude_polish_residual_calls": sum(
                int(item.get("amplitude_polish_residual_calls", 0))
                for item in returned
            ),
            "accepted_candidates": len(accepted),
            "schedule_complete": (
                len(branch_seed_groups) == len(branch_specs)
                and len(attempt_records) == expected_attempts
            ),
        },
        "best_accepted": (
            None
            if best_accepted is None
            else {
                "candidate_id": best_accepted["candidate_id"],
                "primary_gate_error": best_accepted["primary_gate_error"],
                "raw_exact_log_rmse": best_accepted["raw_exact_log_rmse"],
                "standardized_exact_log_rmse": best_accepted[
                    "standardized_exact_log_rmse"
                ],
            }
        ),
        "unconstrained_best_raw_log_rmse": min(
            item["raw_exact_log_rmse"] for item in audit["candidates"]
        ),
        "elapsed_seconds": time.monotonic() - started,
    }


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_hashes() -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    repository = package_dir.parents[2]
    sources = {
        name: package_dir / name
        for name in (
            "amplitude_polish.py",
            "branch_codec.py",
            "contract.py",
            "evaluation.py",
            "profiled_forward.py",
            "profiled_refinement.py",
            "proposal_sampling.py",
            "simulation.py",
            "solver_benchmark.py",
        )
    }
    sources.update(
        {
            "authoritative/scattering_model.py": repository
            / "src/gimap/features/fitting/domain/scattering_model.py",
            "authoritative/physical_constraints.py": repository
            / "src/gimap/features/fitting/domain/physical_constraints.py",
        }
    )
    return {name: _sha256(path) for name, path in sources.items()}


def _runtime_environment(config: BenchmarkConfig) -> dict[str, object]:
    thread_names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "multiprocessing_start_method": mp.get_start_method(allow_none=True),
        "workers": config.workers,
        "thread_environment": {name: os.environ.get(name) for name in thread_names},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def _without_runtime(value):
    if isinstance(value, dict):
        return {
            key: _without_runtime(item)
            for key, item in value.items()
            if key not in {"elapsed_seconds", "slurm_job_id"}
        }
    if isinstance(value, list):
        return [_without_runtime(item) for item in value]
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _distribution(values) -> dict[str, float] | None:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return None
    return {
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def run_benchmark(config: BenchmarkConfig, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite benchmark output: {output_dir}")
    source_files = _source_hashes()
    environment = _runtime_environment(config)
    output_dir.mkdir(parents=True)
    (output_dir / "curves").mkdir()
    _atomic_json(
        output_dir / "run_status.json",
        {
            "schema": BENCHMARK_SCHEMA,
            "state": "RUNNING",
            "config": asdict(config),
            "source_files": source_files,
            "environment": environment,
        },
    )
    started = time.monotonic()
    try:
        if config.workers == 1:
            results = [_curve_task(config, index) for index in range(config.curves)]
        else:
            with ProcessPoolExecutor(max_workers=config.workers) as pool:
                results = list(
                    pool.map(_curve_task, [config] * config.curves, range(config.curves))
                )
        if _source_hashes() != source_files:
            raise RuntimeError("benchmark source changed while workers were running")
    except BaseException as exc:
        _atomic_json(
            output_dir / "run_status.json",
            {
                "schema": BENCHMARK_SCHEMA,
                "state": "FAILED",
                "config": asdict(config),
                "error": f"{type(exc).__name__}: {exc}",
                "source_files": source_files,
                "environment": environment,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise
    results.sort(key=lambda item: item["curve_index"])
    for result in results:
        _atomic_json(
            output_dir / "curves" / f"curve_{result['curve_index']:05d}.json",
            result,
        )

    accepted_best = [item["best_accepted"] for item in results]
    successful_best = [item for item in accepted_best if item is not None]
    recalls = [item["evaluation"]["mode_recall"] for item in results]
    accounting_fields = (
        "expected_branches",
        "generated_branches",
        "expected_attempts",
        "attempted_refinements",
        "returned_candidates",
        "optimizer_converged",
        "amplitude_polish_attempted",
        "amplitude_polish_completed",
        "amplitude_polish_optimizer_converged",
        "amplitude_polish_nfev",
        "amplitude_polish_residual_calls",
        "accepted_candidates",
    )
    attempt_cutoffs = sorted(
        {entry["attempts"] for result in results for entry in result["attempt_best"]}
    )
    attempt_coverage = []
    for cutoff in attempt_cutoffs:
        entries = [
            next(
                (entry for entry in result["attempt_best"] if entry["attempts"] == cutoff),
                None,
            )
            for result in results
        ]
        available = [entry for entry in entries if entry is not None]
        errors = [
            entry["best_primary_gate_error"]
            for entry in available
            if entry["accepted_found"]
        ]
        attempt_coverage.append(
            {
                "attempts": cutoff,
                "eligible_curves": len(available),
                "accepted_curve_fraction": (
                    None
                    if not available
                    else sum(entry["accepted_found"] for entry in available)
                    / len(available)
                ),
                "accepted_primary_error_conditional": _distribution(errors),
            }
        )

    topology_metrics = {}
    for topology_id in K1_TOPOLOGY_IDS:
        subset = [
            item for item in results if item["recipe"]["topology_id"] == topology_id
        ]
        if not subset:
            continue
        subset_best = [item["best_accepted"] for item in subset]
        topology_metrics[str(topology_id)] = {
            "topology": TOPOLOGIES[topology_id],
            "curve_count": len(subset),
            "accepted_curve_fraction": sum(item is not None for item in subset_best)
            / len(subset),
            "generating_nonlinear_mode_recovery": float(
                np.mean([item["evaluation"]["mode_recall"] for item in subset])
            ),
        }

    failure_counts = {}
    for result in results:
        for failure in result["failures"]:
            key = failure["stage"]
            failure_counts[key] = failure_counts.get(key, 0) + 1
    returned_attempts = [
        attempt
        for result in results
        for attempt in result["attempt_records"]
        if attempt["status"] == "returned"
    ]
    completed_polishes = [
        attempt
        for attempt in returned_attempts
        if attempt["amplitude_polish_applied"]
    ]
    scientific_payload = {
        "schema": BENCHMARK_SCHEMA,
        "config": {key: value for key, value in asdict(config).items() if key != "workers"},
        "thresholds": asdict(DEFAULT_THRESHOLDS),
        "source_files": source_files,
        "results": _without_runtime(results),
    }
    summary = {
        "schema": BENCHMARK_SCHEMA,
        "evaluation_contract": {
            "audit_schema": EVALUATION_AUDIT_SCHEMA,
            "reference_matching_version": REFERENCE_MATCHING_VERSION,
            "reference_set_distance_version": REFERENCE_SET_DISTANCE_VERSION,
        },
        "config": asdict(config),
        "thresholds": asdict(DEFAULT_THRESHOLDS),
        "simulation_version": SIMULATION_VERSION,
        "proposal_sampling_version": PROPOSAL_SAMPLING_VERSION,
        "environment": environment,
        "source_files": source_files,
        "scientific_payload_sha256": _canonical_sha256(scientific_payload),
        "curve_count": len(results),
        "headline": {
            "accepted_curve_fraction": len(successful_best) / len(results),
            "accepted_primary_error_conditional": _distribution(
                item["primary_gate_error"] for item in successful_best
            ),
            "accepted_raw_log_rmse_conditional": _distribution(
                item["raw_exact_log_rmse"] for item in successful_best
            ),
            "all_truth_noise_baselines_pass": all(
                item["truth_noise_baseline"]["truth_gate_pass"] for item in results
            ),
        },
        "generating_nonlinear_mode_recovery": float(np.mean(recalls)),
        "generating_mode_scope": results[0]["evaluation"]["parameter_distance_scope"],
        "attempt_budget": attempt_coverage,
        "search_accounting_totals": {
            name: int(sum(item["search_accounting"][name] for item in results))
            for name in accounting_fields
        },
        "complete_schedule_curve_fraction": float(
            np.mean([item["search_accounting"]["schedule_complete"] for item in results])
        ),
        "failure_counts_by_stage": failure_counts,
        "optimizer_nonconverged_count": int(
            sum(
                item["search_accounting"]["returned_candidates"]
                - item["search_accounting"]["optimizer_converged"]
                for item in results
            )
        ),
        "amplitude_polish": {
            "attempted": len(returned_attempts),
            "completed": len(completed_polishes),
            "completed_fraction": (
                None
                if not returned_attempts
                else len(completed_polishes) / len(returned_attempts)
            ),
            "optimizer_converged_fraction": (
                None
                if not completed_polishes
                else sum(
                    bool(item["amplitude_polish_success"])
                    for item in completed_polishes
                )
                / len(completed_polishes)
            ),
            "raw_log_rmse_improvement": _distribution(
                item["raw_log_rmse_improvement"] for item in completed_polishes
            ),
            "standardized_log_rmse_improvement": _distribution(
                item["standardized_log_rmse_improvement"]
                for item in completed_polishes
                if item["standardized_log_rmse_improvement"] is not None
            ),
            "nfev": _distribution(
                item["amplitude_polish_nfev"] for item in completed_polishes
            ),
            "residual_calls": _distribution(
                item["amplitude_polish_residual_calls"]
                for item in completed_polishes
            ),
        },
        "unconstrained_best_raw_log_rmse_diagnostic": _distribution(
            item["unconstrained_best_raw_log_rmse"] for item in results
        ),
        "per_generating_topology": topology_metrics,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output_dir / "summary.json", summary)
    final_state = (
        "COMPLETE"
        if summary["complete_schedule_curve_fraction"] == 1.0
        else "COMPLETE_INCOMPLETE_SEARCH"
    )
    _atomic_json(
        output_dir / "run_status.json",
        {
            "schema": BENCHMARK_SCHEMA,
            "state": final_state,
            "summary_file": "summary.json",
            "scientific_payload_sha256": summary["scientific_payload_sha256"],
            "elapsed_seconds": summary["elapsed_seconds"],
        },
    )
    return summary


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("oracle_k1", "search_k1"), required=True)
    parser.add_argument("--noise-mode", choices=("clean", "default"), default="default")
    parser.add_argument("--curves", type=int, default=12)
    parser.add_argument("--starts-per-branch", type=int, default=16)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--amplitude-polish-max-nfev", type=int, default=80)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    config = BenchmarkConfig(
        mode=args.mode,
        noise_mode=args.noise_mode,
        curves=args.curves,
        starts_per_branch=args.starts_per_branch,
        max_nfev=args.max_nfev,
        points=args.points,
        seed=args.seed,
        workers=args.workers,
        amplitude_polish_max_nfev=args.amplitude_polish_max_nfev,
    )
    summary = run_benchmark(config, args.output_dir)
    print(json.dumps(summary, allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
