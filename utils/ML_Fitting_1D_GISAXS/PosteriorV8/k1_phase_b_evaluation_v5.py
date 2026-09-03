"""Scientific parent replay, proposal sampling, and metrics for K1 Phase-B."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import Callable, Mapping, Sequence

import numpy as np

from .candidate_batch_v5 import build_v5_candidate_context_batch
from .candidate_proposals_v5 import sample_v5_local_proposals
from .candidate_refinement_contract_v5 import EXACT_FORWARD_PHASES
from .candidate_refinement_v5 import run_v5_exact_refinement
from .clean_recipe_forward_v5 import evaluate_v5_clean_recipe_forward
from .evaluation import ObservedCurve, natural_log_rmse
from .grouped_artifact_v5 import canonical_json
from .grouped_dataset_v5 import (
    V5GroupedDataset,
    clean_array,
    observation_array,
)
from .k1_phase_b_contract_v5 import (
    EXPECTED_GATE_KEYS,
    PROPOSAL_COUNT,
    SINGLE_DRAW_INDEX,
    V5_EXACT_FORWARD_BUDGET_UNIT,
    V5_K1_PHASE_B_VERSION,
    V5K1PhaseBGateConfig,
    V5K1PhaseBParentRecord,
    digest,
    live_gate_contract,
    strict_json_object,
)
from .model_v5_contract import MODEL_V5_INPUT_KEYS
from .observation_v5 import build_v5_observation_data_view
from .simulation import GridProvenance
from .synthetic_recipe_v5 import sample_v5_clean_recipe


def _quantile(values: Sequence[float], q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError("quantile input must be a non-empty finite vector")
    return float(np.quantile(array, q, method="linear"))


def _record_reasons(
    record: V5K1PhaseBParentRecord,
    *,
    raw_compatibility_threshold: float,
) -> list[str]:
    reasons: list[str] = []
    for name in (
        "clean_parent_sha256",
        "candidate_query_sha256",
        "geometry_query_sha256",
        "amplitude_query_sha256",
        "model_output_sha256",
    ):
        try:
            digest(getattr(record, name), name)
        except ValueError as exc:
            reasons.append(str(exc))
    if record.proposal_count != PROPOSAL_COUNT:
        reasons.append("proposal_count_is_not_32")
    if record.single_draw_index != SINGLE_DRAW_INDEX:
        reasons.append("single_draw_is_not_stochastic_draw_index_1")
    if record.varying_dimension_count < 1:
        reasons.append("clean_parent_has_no_varying_dimension")
    if record.topology_id < 0 or record.pattern_id < 0:
        reasons.append("branch_identity_is_invalid")
    if not np.all(
        np.isfinite((record.single_draw_local_rms, record.best_of_32_local_rms))
    ):
        reasons.append("local_metric_is_nan_or_infinite")
    if record.best_of_32_local_rms > record.single_draw_local_rms + 1.0e-15:
        reasons.append("best_of_32_is_not_nested_with_single_draw")
    if record.exact_best_raw_log_rmse is None or not np.isfinite(
        float(record.exact_best_raw_log_rmse)
    ):
        reasons.append("exact_metric_missing_nan_or_infinite")
    else:
        expected_compatible = bool(
            float(record.exact_best_raw_log_rmse) < raw_compatibility_threshold
        )
        if record.exact_compatible is not expected_compatible:
            reasons.append("exact_compatibility_flag_disagrees_with_frozen_raw_threshold")
    if record.refinement_status != "exact_candidates_ready_for_verification":
        reasons.append("exact_refinement_status_is_partial_or_empty")
    if record.all_input_seeds_processed is not True:
        reasons.append("not_all_exact_seeds_were_processed")
    if record.configured_per_candidate_limit < 1 or record.configured_total_limit < (
        PROPOSAL_COUNT * record.configured_per_candidate_limit
    ):
        reasons.append("configured_exact_budget_cannot_cover_all_32_seed_limits")
    if record.exact_calls_used + record.exact_calls_remaining != record.configured_total_limit:
        reasons.append("exact_call_ledger_total_does_not_reconcile")
    phase_names = tuple(name for name, _ in record.exact_calls_by_phase)
    phase_counts = dict(record.exact_calls_by_phase)
    if (
        phase_names != EXACT_FORWARD_PHASES
        or len(phase_counts) != len(record.exact_calls_by_phase)
        or any(
            isinstance(count, bool) or int(count) != count or int(count) < 0
            for _, count in record.exact_calls_by_phase
        )
        or sum(count for _, count in record.exact_calls_by_phase)
        != record.exact_calls_used
    ):
        reasons.append("exact_call_phase_ledger_does_not_reconcile")
    if record.attempts_recorded != PROPOSAL_COUNT:
        reasons.append("exact_attempt_ledger_is_partial")
    if record.refinement_successes != PROPOSAL_COUNT:
        reasons.append("one_or_more_exact_attempts_are_unverified")
    failures = (
        record.validation_failures
        + record.refinement_failures
        + record.per_candidate_budget_exhausted_attempts
        + record.total_budget_exhausted_before_seed_attempts
    )
    if record.refinement_successes + failures != record.attempts_recorded:
        reasons.append("exact_attempt_outcome_counts_do_not_reconcile")
    if failures:
        reasons.append("exact_attempt_failures_or_budget_exhaustion_present")
    for name in (
        "bounds_compliant_attempts",
        "physics_compliant_attempts",
        "amplitude_compliant_attempts",
    ):
        if getattr(record, name) != PROPOSAL_COUNT:
            reasons.append(f"{name}_is_not_32")
    return reasons


def assess_v5_k1_phase_b_records(
    records: Sequence[V5K1PhaseBParentRecord],
    *,
    gate_contract: Mapping[str, object] | None = None,
    engineering_subset: bool = False,
) -> dict[str, object]:
    """Macro-average clean parents and evaluate every frozen gate fail closed."""

    if type(engineering_subset) is not bool:
        raise TypeError("engineering_subset must be a bool")
    contract = live_gate_contract() if gate_contract is None else dict(gate_contract)
    gates = contract.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(EXPECTED_GATE_KEYS):
        raise ValueError("K1 Phase-B assessment requires exactly every frozen gate")
    selected = tuple(records)
    if not all(isinstance(value, V5K1PhaseBParentRecord) for value in selected):
        raise TypeError("records must contain V5K1PhaseBParentRecord values")
    raw_threshold = float(gates[EXPECTED_GATE_KEYS[3]])
    integrity_reasons: list[str] = []
    if len(selected) != int(contract.get("recipes", -1)):
        integrity_reasons.append("clean_parent_count_does_not_match_frozen_protocol")
    if engineering_subset:
        integrity_reasons.append("engineering_parent_subset_cannot_pass_the_formal_gate")
    if len({value.clean_parent_sha256 for value in selected}) != len(selected):
        integrity_reasons.append("clean_parent_records_are_not_unique")
    per_parent_reasons = {
        value.clean_parent_sha256: _record_reasons(
            value, raw_compatibility_threshold=raw_threshold
        )
        for value in selected
    }
    if any(per_parent_reasons.values()):
        integrity_reasons.append("one_or_more_clean_parent_records_are_partial_or_invalid")

    finite_local = bool(
        selected
        and all(
            np.isfinite(value.single_draw_local_rms)
            and np.isfinite(value.best_of_32_local_rms)
            for value in selected
        )
    )
    finite_exact = bool(
        selected
        and all(
            value.exact_best_raw_log_rmse is not None
            and np.isfinite(float(value.exact_best_raw_log_rmse))
            for value in selected
        )
    )
    metrics: dict[str, float | None] = {
        "branch_conditioned_local_mdn_single_draw_local_rms_median": (
            float(np.median([value.single_draw_local_rms for value in selected]))
            if finite_local
            else None
        ),
        "branch_conditioned_local_mdn_best_of_32_local_rms_median": (
            float(np.median([value.best_of_32_local_rms for value in selected]))
            if finite_local
            else None
        ),
        "branch_conditioned_local_mdn_best_of_32_local_rms_p90": (
            _quantile([value.best_of_32_local_rms for value in selected], 0.9)
            if finite_local
            else None
        ),
        "exact_post_refine_raw_log_rmse_p90": (
            _quantile(
                [float(value.exact_best_raw_log_rmse) for value in selected], 0.9
            )
            if finite_exact
            else None
        ),
        "exact_post_refine_compatible_rate": (
            float(np.mean([value.exact_compatible for value in selected]))
            if finite_exact
            else None
        ),
    }
    metric_for_gate = {
        "branch_conditioned_local_mdn_single_draw_local_rms_median_lt": metrics[
            "branch_conditioned_local_mdn_single_draw_local_rms_median"
        ],
        "branch_conditioned_local_mdn_best_of_32_local_rms_median_lt": metrics[
            "branch_conditioned_local_mdn_best_of_32_local_rms_median"
        ],
        "branch_conditioned_local_mdn_best_of_32_local_rms_p90_lt": metrics[
            "branch_conditioned_local_mdn_best_of_32_local_rms_p90"
        ],
        "exact_post_refine_raw_log_rmse_p90_lt": metrics[
            "exact_post_refine_raw_log_rmse_p90"
        ],
        "exact_post_refine_compatible_rate_gte": metrics[
            "exact_post_refine_compatible_rate"
        ],
    }
    decisions = {}
    integrity_ok = not integrity_reasons
    for name in EXPECTED_GATE_KEYS:
        observed = metric_for_gate[name]
        threshold = float(gates[name])
        operator = "lt" if name.endswith("_lt") else "gte"
        passed = bool(
            integrity_ok
            and observed is not None
            and np.isfinite(observed)
            and (observed < threshold if operator == "lt" else observed >= threshold)
        )
        decisions[name] = {
            "metric": name.removesuffix("_lt").removesuffix("_gte"),
            "observed": observed,
            "operator": operator,
            "threshold": threshold,
            "passed": passed,
        }

    attempts = sum(value.attempts_recorded for value in selected)
    aggregate_phases = Counter()
    for value in selected:
        aggregate_phases.update(dict(value.exact_calls_by_phase))
    compliance = {
        "denominator_all_32_attempts_per_clean_parent": attempts,
        "bounds_compliance_fraction": (
            sum(value.bounds_compliant_attempts for value in selected) / attempts
            if attempts
            else None
        ),
        "physics_compliance_fraction": (
            sum(value.physics_compliant_attempts for value in selected) / attempts
            if attempts
            else None
        ),
        "amplitude_range_compliance_fraction": (
            sum(value.amplitude_compliant_attempts for value in selected) / attempts
            if attempts
            else None
        ),
    }
    ledger = {
        "budget_unit": V5_EXACT_FORWARD_BUDGET_UNIT,
        "clean_parent_count": len(selected),
        "input_seed_count": len(selected) * PROPOSAL_COUNT,
        "attempts_recorded": attempts,
        "refinement_successes": sum(value.refinement_successes for value in selected),
        "validation_failures": sum(value.validation_failures for value in selected),
        "refinement_failures": sum(value.refinement_failures for value in selected),
        "per_candidate_budget_exhausted_attempts": sum(
            value.per_candidate_budget_exhausted_attempts for value in selected
        ),
        "total_budget_exhausted_before_seed_attempts": sum(
            value.total_budget_exhausted_before_seed_attempts for value in selected
        ),
        "calls_used": sum(value.exact_calls_used for value in selected),
        "calls_remaining": sum(value.exact_calls_remaining for value in selected),
        "calls_by_phase": [
            [phase, int(aggregate_phases[phase])] for phase in EXACT_FORWARD_PHASES
        ],
    }
    single_branch_passed = integrity_ok and all(
        value["passed"] for value in decisions.values()
    )
    return {
        "statistical_unit": "independent_clean_parent_macro_average",
        "quantile_method": "numpy_linear",
        "metrics": metrics,
        "gate_decisions": decisions,
        "integrity_passed": integrity_ok,
        "integrity_failure_reasons": integrity_reasons,
        "per_parent_failure_reasons": per_parent_reasons,
        "bounds_and_physics_compliance": compliance,
        "exact_forward_ledger": ledger,
        "single_branch_phase_b_gate_passed": single_branch_passed,
        "complete_k1_proposal_exact_gate_passed": False,
        "full_k1_all_legal_branches_gate_status": (
            "pending_fail_closed_requires_balanced_12_branch_cohort"
        ),
    }


def parent_seed(seed: int, clean_parent_sha256: str) -> int:
    value = sha256(
        f"{V5_K1_PHASE_B_VERSION}\0{seed}\0{clean_parent_sha256}".encode("ascii")
    ).digest()
    return int.from_bytes(value[:8], "big")


def _model_output_sha256(outputs: Mapping[str, np.ndarray]) -> str:
    result = sha256()
    for name in sorted(outputs):
        array = np.ascontiguousarray(outputs[name])
        result.update(name.encode("ascii"))
        result.update(b"\0")
        result.update(array.dtype.str.encode("ascii"))
        result.update(canonical_json(list(array.shape)).encode("ascii"))
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def _same_array(actual: object, expected: np.ndarray) -> bool:
    value = np.asarray(actual)
    return (
        value.dtype == expected.dtype
        and value.shape == expected.shape
        and np.ascontiguousarray(value).tobytes(order="C")
        == expected.tobytes(order="C")
    )


def replay_parent_contexts(
    dataset: V5GroupedDataset, *, parent_limit: int | None = None
):
    """Replay every checked clean parent and its sole bound generating branch."""

    if dataset.observation_count != dataset.recipe_count:
        raise ValueError("K1 Phase-B requires exactly one observation per clean parent")
    inputs, labels = dataset.joined_numpy(include_unverified=False)
    observed_join, candidate_join = dataset.join_indices(include_unverified=False)
    if inputs["x"].shape[0] != dataset.recipe_count:
        raise ValueError("K1 Phase-B joined rows are not one-to-one with clean parents")
    arrays = dataset.arrays
    observation_recipe = arrays[observation_array("recipe_index")]
    candidate_recipe = arrays["candidate_context__recipe_index"]
    selected_count = dataset.recipe_count if parent_limit is None else int(parent_limit)
    if not 1 <= selected_count <= dataset.recipe_count:
        raise ValueError("parent_limit must select between one and all clean parents")
    contexts = []
    for recipe_index in range(selected_count):
        observation_indices = np.flatnonzero(observation_recipe == recipe_index)
        candidate_indices = np.flatnonzero(candidate_recipe == recipe_index)
        if observation_indices.size != 1 or candidate_indices.size != 1:
            raise ValueError("K1 Phase-B requires one observation and one branch per parent")
        observation_index, candidate_index = int(observation_indices[0]), int(candidate_indices[0])
        encoded = str(arrays[clean_array("recipe_canonical_json")][recipe_index])
        payload = strict_json_object(encoded, "clean recipe")
        topology = tuple(str(value) for value in payload.get("query", {}).get("topology", ()))
        recipe = sample_v5_clean_recipe(
            topology,
            recipe_seed=int(payload["recipe_seed"]),
            amplitude_range_regime=str(payload["amplitude_range_regime"]),
            pattern_id=int(payload["branch_pattern_id"]),
            grid=GridProvenance(**payload["grid"]),
        )
        expected_sha = str(arrays[clean_array("recipe_sha256")][recipe_index])
        if recipe.canonical_json != encoded or recipe.sha256 != expected_sha:
            raise ValueError("clean parent no longer replays byte-for-byte")
        view = build_v5_observation_data_view(
            recipe,
            int(arrays[observation_array("view_index")][observation_index]),
            split_id=str(arrays[observation_array("split_id")][observation_index]),
        )
        if str(arrays[observation_array("audit_json")][observation_index]) != canonical_json(
            view.audit_payload()
        ):
            raise ValueError("observation view audit no longer replays")
        batch = build_v5_candidate_context_batch(
            view.preprocessed,
            view.uncertainty,
            recipe.query,
            recipe.amplitude_query,
            pattern_ids=(recipe.target.pattern_id,),
        )
        matches = np.flatnonzero(
            (observed_join == observation_index) & (candidate_join == candidate_index)
        )
        if matches.size != 1:
            raise ValueError("clean parent does not map to exactly one joined model row")
        joined_index = int(matches[0])
        for name in MODEL_V5_INPUT_KEYS:
            expected = np.asarray(inputs[name][joined_index : joined_index + 1])
            if not _same_array(batch.model_inputs[name], expected):
                raise ValueError(f"replayed parent model input drift detected: {name}")
        stored_target = np.asarray(labels["target_local"][joined_index])
        target = np.asarray(stored_target, dtype=np.float64)
        varying = np.asarray(labels["varying_dimension_mask"][joined_index], dtype=bool)
        if not np.array_equal(
            stored_target,
            np.asarray(recipe.target.local_target_unit, dtype=stored_target.dtype),
        ):
            raise ValueError("replayed clean target disagrees with the checked dataset")
        if not np.array_equal(
            varying,
            np.asarray(batch.branch_conditions[0].varying_dimension_mask, dtype=bool),
        ):
            raise ValueError("replayed varying mask disagrees with the checked dataset")
        q = recipe.grid.values()
        curve = ObservedCurve(
            curve_id=f"k1-phase-b-clean:{recipe.sha256}",
            source_kind="synthetic",
            q=q,
            intensity=evaluate_v5_clean_recipe_forward(recipe, q),
            sigma_log=None,
        )
        contexts.append((recipe, batch, target, varying, curve))
    return tuple(contexts)


def as_numpy_outputs(raw: Mapping[str, object]) -> dict[str, np.ndarray]:
    if not isinstance(raw, Mapping):
        raise TypeError("V5 model call must return a mapping")
    return {
        str(name): np.asarray(value.numpy() if hasattr(value, "numpy") else value)
        for name, value in raw.items()
    }


def build_parent_record(
    *,
    recipe,
    batch,
    target: np.ndarray,
    varying: np.ndarray,
    curve: ObservedCurve,
    model_outputs: Mapping[str, np.ndarray],
    frozen_parent_seed: int,
    raw_compatibility_threshold: float,
    config: V5K1PhaseBGateConfig,
    proposal_sampler: Callable = sample_v5_local_proposals,
    refiner: Callable = run_v5_exact_refinement,
) -> V5K1PhaseBParentRecord:
    """Sample/refine one parent; injectable boundaries are only for focused tests."""

    proposals = proposal_sampler(
        batch,
        model_outputs,
        mixture_limit=1,
        stochastic_draws_per_mixture=PROPOSAL_COUNT,
        seed=frozen_parent_seed,
        include_mixture_medians=False,
    )
    if (
        len(proposals) != PROPOSAL_COUNT
        or tuple(value.draw_index for value in proposals) != tuple(range(1, PROPOSAL_COUNT + 1))
        or any(value.source != "stochastic_draw" for value in proposals)
    ):
        raise RuntimeError("proposal sampler did not return frozen stochastic draws 1..32")
    varying_count = int(np.count_nonzero(varying))
    if varying_count < 1:
        raise ValueError("K1 Phase-B parent has no varying local coordinate")
    local_rms = np.asarray(
        [
            np.sqrt(
                np.mean(
                    np.square(
                        np.asarray(value.local_unit, dtype=np.float64)[varying]
                        - target[varying]
                    )
                )
            )
            for value in proposals
        ],
        dtype=np.float64,
    )
    refinement = refiner(
        batch,
        curve,
        proposals,
        per_candidate_forward_evaluation_limit=config.per_candidate_forward_evaluation_limit,
        forward_evaluation_limit=config.per_parent_forward_evaluation_limit,
    )
    raw_values = [
        natural_log_rmse(candidate.exact_intensity, curve.intensity)
        for candidate in refinement.candidates
    ]
    exact_best = min(raw_values) if raw_values else None
    attempts, ledger = refinement.attempts, refinement.ledger
    bounds = sum(
        item.prerequisite_audit is not None
        and item.prerequisite_audit.geometry_bounds_satisfied
        for item in attempts
    )
    physics = sum(
        item.prerequisite_audit is not None and item.prerequisite_audit.physics_satisfied
        for item in attempts
    )
    amplitude = sum(
        item.prerequisite_audit is not None
        and item.prerequisite_audit.initial_amplitude_audit.all_constraints_satisfied
        and item.prerequisite_audit.final_amplitude_audit.all_constraints_satisfied
        for item in attempts
    )
    return V5K1PhaseBParentRecord(
        clean_parent_sha256=recipe.sha256,
        candidate_query_sha256=batch.query_sha256,
        geometry_query_sha256=batch.query.sha256,
        amplitude_query_sha256=batch.amplitude_query.sha256,
        topology_id=recipe.query.topology_id,
        pattern_id=recipe.target.pattern_id,
        frozen_seed=frozen_parent_seed,
        varying_dimension_count=varying_count,
        proposal_count=len(proposals),
        single_draw_index=proposals[0].draw_index,
        single_draw_local_rms=float(local_rms[0]),
        best_of_32_local_rms=float(np.min(local_rms)),
        exact_best_raw_log_rmse=None if exact_best is None else float(exact_best),
        exact_compatible=bool(
            exact_best is not None and exact_best < raw_compatibility_threshold
        ),
        refinement_status=refinement.status,
        all_input_seeds_processed=refinement.all_input_seeds_processed,
        configured_total_limit=ledger.configured_total_limit,
        configured_per_candidate_limit=ledger.configured_per_candidate_limit,
        exact_calls_used=ledger.calls_used,
        exact_calls_remaining=ledger.calls_remaining,
        exact_calls_by_phase=ledger.calls_by_phase,
        attempts_recorded=ledger.attempts_recorded,
        refinement_successes=ledger.refinement_successes,
        validation_failures=ledger.validation_failures,
        refinement_failures=ledger.refinement_failures,
        per_candidate_budget_exhausted_attempts=ledger.per_candidate_budget_exhausted_attempts,
        total_budget_exhausted_before_seed_attempts=(
            ledger.total_budget_exhausted_before_seed_attempts
        ),
        bounds_compliant_attempts=int(bounds),
        physics_compliant_attempts=int(physics),
        amplitude_compliant_attempts=int(amplitude),
        model_output_sha256=_model_output_sha256(model_outputs),
    )


__all__ = [
    "as_numpy_outputs",
    "assess_v5_k1_phase_b_records",
    "build_parent_record",
    "parent_seed",
    "replay_parent_contexts",
]
