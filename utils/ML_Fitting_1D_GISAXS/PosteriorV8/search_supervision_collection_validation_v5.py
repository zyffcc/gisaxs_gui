"""Fail-closed checks performed before collecting V5 search supervision."""

from __future__ import annotations

import numpy as np

from .grouped_amplitude_join_v5 import amplitude_query_from_json
from .grouped_dataset_v5 import (
    V5_GROUPED_SOLUTION_STAGE,
    V5GroupedDataset,
    candidate_context_array,
    candidate_input,
    clean_array,
    observation_array,
    observation_input,
)
from .search_supervision_contract_v5 import (
    V5ExactSearchObservation,
    V5FrozenBranchSearchResult,
    V5FrozenSearchTask,
)
from .search_supervision_sidecar_v5 import (
    BRANCH_MODEL_INPUT_KEYS,
    OBSERVATION_MODEL_INPUT_KEYS,
    V5_SEARCH_SIDECAR_ALLOWED_SPLITS,
)
from .universal_query_v5 import V5UniversalCandidateContext


def validate_v5_search_sidecar_parent(dataset: V5GroupedDataset) -> str:
    if dataset.manifest["stage"] != V5_GROUPED_SOLUTION_STAGE:
        raise ValueError("search sidecar requires a solution-stage grouped parent")
    if dataset.manifest["build_policy"]["generating_candidate_only"]:
        raise ValueError("search sidecar requires all source-topology feasible branches")
    clean_splits = set(dataset.arrays[clean_array("split_id")].tolist())
    if len(clean_splits) != 1 or not clean_splits <= set(V5_SEARCH_SIDECAR_ALLOWED_SPLITS):
        raise ValueError("search-supervision sidecars are model-development-split only")
    split_id = str(next(iter(clean_splits)))
    if set(dataset.arrays[observation_array("split_id")].tolist()) != {split_id}:
        raise ValueError("parent observations crossed their clean-parent split")
    return split_id


def validate_v5_universal_context_parent_join(
    dataset: V5GroupedDataset,
    observation_index: int,
    context: V5UniversalCandidateContext,
    exact_observation: V5ExactSearchObservation,
) -> tuple[int, np.ndarray]:
    if observation_index >= dataset.observation_count:
        raise ValueError("universal-search spec references a missing parent observation")
    arrays = dataset.arrays
    recipe_index = int(arrays[observation_array("recipe_index")][observation_index])
    if not isinstance(exact_observation, V5ExactSearchObservation):
        raise TypeError("exact_observation must be V5ExactSearchObservation")
    view = exact_observation.observation_view
    observation_id = str(arrays[observation_array("observation_id")][observation_index])
    if (
        exact_observation.observed_curve.curve_id != observation_id
        or exact_observation.parent_observation_audit_json
        != arrays[observation_array("audit_json")][observation_index]
        or view.clean_recipe_sha256
        != arrays[clean_array("recipe_sha256")][recipe_index]
        or view.view_index != int(arrays[observation_array("view_index")][observation_index])
        or view.split_id != arrays[observation_array("split_id")][observation_index]
        or view.acquisition_policy_id
        != arrays[observation_array("acquisition_policy_id")][observation_index]
    ):
        raise ValueError(
            "exact-search curve/provenance disagrees with its parent observation"
        )
    for name in OBSERVATION_MODEL_INPUT_KEYS:
        expected = arrays[observation_input(name)][observation_index]
        exact_input = view.encoder_curve_inputs()[name]
        if not np.array_equal(exact_input, expected):
            raise ValueError(
                "exact-search PreprocessedCurve disagrees with parent observation tensors"
            )
        for batch in context.batches:
            if not np.array_equal(batch.model_inputs[name][0], expected):
                raise ValueError("universal query observation disagrees with its grouped parent")

    geometry_sha = str(arrays[clean_array("geometry_query_sha256")][recipe_index])
    amplitude_sha = str(arrays[clean_array("amplitude_query_sha256")][recipe_index])
    matches = np.asarray(
        [
            index
            for index, entry in enumerate(context.topology_queries)
            if entry.geometry.sha256 == geometry_sha and entry.amplitude.sha256 == amplitude_sha
        ],
        dtype=np.int32,
    )
    if matches.size != 1:
        raise ValueError("universal query must contain exactly one source grouped query")
    topology_batch_index = int(matches[0])
    batch = context.batches[topology_batch_index]
    candidates = np.flatnonzero(
        arrays[candidate_context_array("recipe_index")] == recipe_index
    )
    base_patterns = arrays[candidate_input("branch_pattern_id")][candidates].reshape(-1)
    if tuple(int(value) for value in base_patterns) != batch.pattern_ids:
        raise ValueError("source grouped candidate catalog is partial or reordered")
    amplitude_query = amplitude_query_from_json(
        str(arrays[clean_array("amplitude_query_canonical_json")][recipe_index]),
        amplitude_sha,
    )
    expected_amplitude = np.asarray(
        amplitude_query.model_embedding(
            float(arrays[observation_array("intensity_reference")][observation_index])
        ),
        dtype=np.float32,
    )
    for local_index, candidate_index in enumerate(candidates):
        for name in BRANCH_MODEL_INPUT_KEYS:
            expected = (
                expected_amplitude
                if name == "amplitude_bounds_embedding"
                else arrays[candidate_input(name)][candidate_index]
            )
            if not np.array_equal(batch.model_inputs[name][local_index], expected):
                raise ValueError("source grouped query/branch model context disagrees")
        branch = context.branches[
            context.batch_slices[topology_batch_index].start + local_index
        ]
        if (
            str(arrays[candidate_context_array("geometry_query_sha256")][candidate_index])
            != geometry_sha
            or str(arrays[candidate_context_array("amplitude_query_sha256")][candidate_index])
            != amplitude_sha
            or str(
                arrays[candidate_context_array("amplitude_constraint_sha256")][candidate_index]
            )
            != branch.amplitude_constraint_sha256
        ):
            raise ValueError("source clean/query/branch/context join is inconsistent")
    return recipe_index, candidates


def validate_v5_frozen_search_result(
    task: V5FrozenSearchTask,
    result: V5FrozenBranchSearchResult,
) -> None:
    if not isinstance(result, V5FrozenBranchSearchResult):
        raise TypeError("search runner must return V5FrozenBranchSearchResult")
    branch = task.branch
    if (
        result.universal_query_sha256 != task.universal_context.audit_sha256
        or result.exact_curve_sha256 != task.exact_curve_sha256
        or result.global_branch_key != branch.global_key
        or result.context_sha256 != branch.context_sha256
    ):
        raise ValueError("search runner returned a result for a different query/branch/context")
    protocol = task.protocol
    if result.exact_forward_calls_used > protocol.exact_forward_call_budget:
        raise ValueError("search runner exceeded the frozen exact-forward budget")
    if (
        result.outcome != "unverified"
        and result.exact_forward_calls_used != protocol.exact_forward_call_budget
    ):
        raise ValueError(
            "completed frozen-search outcome did not consume the equal per-branch budget"
        )
    if (
        result.outcome == "no_compatible_found_within_frozen_search_budget"
        and result.termination_reason == "exact_forward_budget_exhausted_without_compatible"
        and result.exact_forward_calls_used != protocol.exact_forward_call_budget
    ):
        raise ValueError("budget-exhausted negative did not consume the frozen budget")
    for representative in result.representatives:
        if (
            representative.metric_value > task.selected_threshold_value
            or not representative.bounds_passed
            or not representative.physics_passed
        ):
            raise ValueError("positive representative fails the frozen exact gate")


__all__ = [
    "validate_v5_frozen_search_result",
    "validate_v5_search_sidecar_parent",
    "validate_v5_universal_context_parent_join",
]
