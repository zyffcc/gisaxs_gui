"""Immutable cross-topology frozen-search supervision sidecars for V5.1.

The solution-stage grouped artifact remains unchanged.  This sidecar is
bound to its exact bytes and adds one observation-specific universal branch
catalog plus independently executed search outcomes.  An injectable runner
keeps schema work separate from the still-evolving exact-search executor.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .candidate_batch_v5 import V5_USER_QUERY_DIGEST_VERSION
from .candidate_supervision_v5 import (
    CANDIDATE_SUPERVISION_TENSOR_KEYS,
    SEARCH_OUTCOME_CODE,
    V5CandidateSupervision,
    candidate_supervision_v5_contract_payload,
    stack_candidate_supervision_v5,
)
from .grouped_artifact_v5 import (
    V5_CHECKED_ARRAY_ARTIFACT_SCHEMA,
    V5_CHECKED_ARRAY_ARTIFACT_VERSION,
    V5ArtifactReceipt,
    array_manifest,
    array_sha256,
    canonical_json,
    read_checked_array_artifact,
    validate_array_manifest,
    write_checked_array_artifact,
)
from .grouped_dataset_v5 import (
    V5GroupedDataset,
    clean_array,
    observation_array,
    observation_input,
    read_v5_grouped_dataset,
)
from .model_v5_contract import MODEL_V5_INPUT_KEYS, model_v5_contract_payload
from .search_supervision_contract_v5 import (
    V5ExactSearchObservation,
    V5FrozenBranchSearchRunner,
    V5FrozenExactSearchProtocol,
    V5FrozenSearchTask,
    V5UniversalSearchSpec,
    unverified_v5_search_result,
)
from .search_supervision_evidence_v5 import (
    V5_SEARCH_RECORD_SCHEMA,
    build_v5_candidate_supervision,
    build_v5_search_record,
)
from .universal_training_contract_v5 import (
    V5_CROSS_TOPOLOGY_VERIFIED_SEARCH_STAGE,
    universal_training_contract_v5_payload,
)

V5_SEARCH_SIDECAR_SCHEMA = "gisaxs.posterior_v8.cross_topology_search_sidecar/v5"
V5_SEARCH_SIDECAR_VERSION = (
    "posterior_v8_exact_f64_reference_full_scientific_source_closure_overlay_v6"
)
V5_SEARCH_SIDECAR_ALLOWED_SPLITS = ("train", "tuning_validation")

OBSERVATION_MODEL_INPUT_KEYS = (
    "x",
    "point_mask",
    "global_features",
    "uncertainty_provenance",
)
BRANCH_MODEL_INPUT_KEYS = tuple(
    value for value in MODEL_V5_INPUT_KEYS if value not in OBSERVATION_MODEL_INPUT_KEYS
)
QUERY_FIELDS = (
    "clean_group_id",
    "recipe_id",
    "observation_id",
    "parent_recipe_index",
    "parent_observation_index",
    "split_id",
    "universal_query_sha256",
    "universal_query_audit_json",
    "observation_inputs_sha256",
    "intensity_reference",
    "intensity_reference_sha256",
    "parent_observation_audit_sha256",
    "exact_observation_audit_json",
    "exact_observation_audit_sha256",
    "exact_curve_id",
    "exact_curve_source_kind",
    "exact_curve_sha256",
    "exact_curve_valid_count",
    "exact_curve_q",
    "exact_curve_intensity",
    "exact_curve_sigma_log",
    "exact_curve_point_mask",
    "acceptance_sigma_log_available",
    "acceptance_sigma_source_id",
    "preprocessed_valid_arrays_sha256",
    "selected_metric_name",
    "selected_threshold_name",
    "selected_threshold_value",
    "selected_threshold_source_id",
    "selected_threshold_binding_json",
    "selected_threshold_binding_sha256",
    "full_training_label_permitted",
    "query_catalog_artifact_id",
    "query_catalog_artifact_sha256",
    "topology_query_catalog_json",
    "topology_query_catalog_sha256",
    "selected_topology_ids_json",
    "catalog_sha256",
    "branch_count",
)
BRANCH_FIELDS = (
    "query_index",
    "global_branch_key",
    "topology_id",
    "pattern_id",
    "declared_topology_query_sha256",
    "candidate_query_sha256",
    "geometry_query_sha256",
    "amplitude_query_sha256",
    "context_sha256",
    "amplitude_constraint_sha256",
    "amplitude_constraint_json",
    "frozen_exact_forward_call_budget",
    "exact_forward_calls_used",
    "runner_completed",
    "runner_termination_reason",
    "executor_artifact_id",
    "executor_artifact_sha256",
    "executor_artifact_relative_path",
    "executor_artifact_schema",
    "executor_artifact_version",
    "executor_source_bundle_sha256",
    "task_audit_sha256",
    "search_record_json",
    "search_record_sha256",
    "compatible_representatives_json",
    "supervision_audit_json",
)


def query_array(name: str) -> str:
    return f"query__{name}"


def branch_array(name: str) -> str:
    return f"branch__{name}"


def branch_input(name: str) -> str:
    return f"branch__input__{name}"


def branch_label(name: str) -> str:
    return f"branch__label__{name}"


def _digest_text(*values: str) -> str:
    digest = sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _candidate_query_sha256(geometry_sha256: str, amplitude_sha256: str) -> str:
    return _digest_text(
        V5_USER_QUERY_DIGEST_VERSION,
        geometry_sha256,
        amplitude_sha256,
    )


def _catalog_sha256(branches: Sequence[Mapping[str, object]]) -> str:
    return sha256(canonical_json(list(branches)).encode("utf-8")).hexdigest()


def topology_query_catalog_json(context) -> str:
    """Freeze every caller-supplied geometry/amplitude range without inference."""

    payload = [
        {
            "topology_id": entry.topology_id,
            "topology_query_sha256": entry.sha256,
            "geometry_query_sha256": entry.geometry.sha256,
            "geometry_query_canonical_json": entry.geometry.canonical_json,
            "amplitude_query_sha256": entry.amplitude.sha256,
            "amplitude_query_canonical_json": entry.amplitude.canonical_json,
        }
        for entry in context.topology_queries
    ]
    return canonical_json(payload)


def _padded_exact_curve(
    exact: V5ExactSearchObservation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    view = exact.observation_view
    curve = exact.observed_curve
    point_mask = np.asarray(view.preprocessed.point_mask, dtype=np.bool_)
    q = np.zeros(point_mask.shape, dtype=np.float64)
    intensity = np.zeros(point_mask.shape, dtype=np.float64)
    sigma_log = np.zeros(point_mask.shape, dtype=np.float64)
    q[point_mask] = curve.q
    intensity[point_mask] = curve.intensity
    if curve.sigma_log is not None:
        sigma_log[point_mask] = curve.sigma_log
    return q, intensity, sigma_log, point_mask


def observation_inputs_sha256(values: Mapping[str, np.ndarray]) -> str:
    digest = sha256()
    for name in OBSERVATION_MODEL_INPUT_KEYS:
        value = np.ascontiguousarray(values[name])
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _source_sha256() -> dict[str, str]:
    posterior_root = Path(__file__).resolve().parent
    repository_root = posterior_root.parents[2]
    sources = sorted(posterior_root.glob("*.py"))
    sources.extend(
        (
            repository_root / "src/gimap/features/fitting/domain/scattering_model.py",
            repository_root / "src/gimap/features/fitting/domain/physical_constraints.py",
        )
    )
    return {
        path.relative_to(repository_root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sources
    }


def _strict_json(encoded: str, name: str) -> dict[str, object]:
    try:
        value = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict) or canonical_json(value) != encoded:
        raise ValueError(f"{name} must be one canonical JSON object")
    return value


def _contract_bundle() -> dict[str, object]:
    return {
        "model": model_v5_contract_payload(),
        "candidate_supervision": candidate_supervision_v5_contract_payload(),
        "universal_training": universal_training_contract_v5_payload(),
    }


def _manifest(
    *,
    sidecar_id: str,
    parent: V5GroupedDataset,
    parent_receipt: V5ArtifactReceipt,
    protocol: V5FrozenExactSearchProtocol,
    split_id: str,
    arrays: Mapping[str, np.ndarray],
    counts: Mapping[str, int],
    executor_evidence_index_complete: bool,
) -> dict[str, object]:
    bundle = _contract_bundle()
    core = {
        "container_schema": V5_CHECKED_ARRAY_ARTIFACT_SCHEMA,
        "container_version": V5_CHECKED_ARRAY_ARTIFACT_VERSION,
        "sidecar_schema": V5_SEARCH_SIDECAR_SCHEMA,
        "sidecar_version": V5_SEARCH_SIDECAR_VERSION,
        "sidecar_id": sidecar_id,
        "stage": V5_CROSS_TOPOLOGY_VERIFIED_SEARCH_STAGE,
        "split_id": split_id,
        "parent_grouped_artifact": {
            "dataset_id": parent.manifest["dataset_id"],
            "dataset_schema": parent.manifest["dataset_schema"],
            "dataset_version": parent.manifest["dataset_version"],
            "stage": parent.manifest["stage"],
            "artifact_sha256": parent_receipt.artifact_sha256,
            "manifest_sha256": parent_receipt.manifest_sha256,
        },
        "protocol": protocol.audit_payload(),
        "protocol_sha256": protocol.sha256,
        "contract_bundle": bundle,
        "contract_bundle_sha256": sha256(
            canonical_json(bundle).encode("utf-8")
        ).hexdigest(),
        "build_policy": {
            "model_development_splits_only": True,
            "one_universal_query_per_parent_observation": True,
            "all_parent_observations_covered": True,
            "all_selected_topology_feasible_branches_materialized": True,
            "completed_negative_requires_frozen_search_completion": True,
            "unverified_is_not_negative": True,
            "neural_scores_used_to_create_labels": False,
            "parent_solution_stage_mutated": False,
            "alternative_topology_queries_are_explicit_caller_supplied": True,
            "unknown_alternative_ranges_are_never_inferred": True,
            "exact_float64_intensity_reference_is_persisted_per_query": True,
            "exact_curve_is_preprocessed_valid_physical_curve": True,
            "encoder_sigma_proxy_is_acceptance_evidence": False,
            "task_bound_executor_evidence_receipt_required_for_full_training": True,
        },
        "executor_evidence_index_complete": executor_evidence_index_complete,
        "full_training_labels_permitted": (
            protocol.protocol_tier == "paper_full_calibrated"
        ),
        "counts": dict(counts),
        "table_fields": {
            "query": list(QUERY_FIELDS),
            "branch": list(BRANCH_FIELDS),
            "branch_inputs": list(BRANCH_MODEL_INPUT_KEYS),
            "branch_labels": list(CANDIDATE_SUPERVISION_TENSOR_KEYS),
        },
        "source_sha256": _source_sha256(),
        "arrays": array_manifest(arrays),
    }
    return {**core, "manifest_sha256": sha256(canonical_json(core).encode()).hexdigest()}


@dataclass(frozen=True)
class V5SearchSupervisionSidecar:
    manifest: Mapping[str, object]
    arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        from .search_supervision_sidecar_validation_v5 import (
            validate_v5_search_supervision_sidecar,
        )

        manifest = deepcopy(dict(self.manifest))
        arrays = validate_array_manifest(self.arrays, manifest.get("arrays", {}))
        validate_v5_search_supervision_sidecar(manifest, arrays)
        object.__setattr__(self, "manifest", MappingProxyType(manifest))
        object.__setattr__(self, "arrays", arrays)

    @property
    def query_count(self) -> int:
        return int(self.manifest["counts"]["universal_queries"])

    @property
    def branch_count(self) -> int:
        return int(self.manifest["counts"]["branch_outcomes"])


def collect_v5_search_supervision_sidecar(
    parent_dataset_path: str | os.PathLike[str],
    specs: Sequence[V5UniversalSearchSpec],
    *,
    sidecar_id: str,
    protocol: V5FrozenExactSearchProtocol,
    runner: V5FrozenBranchSearchRunner,
) -> V5SearchSupervisionSidecar:
    """Run every catalog task; exceptions abort before any artifact is published."""

    from .search_supervision_collection_validation_v5 import (
        validate_v5_frozen_search_result,
        validate_v5_search_sidecar_parent,
        validate_v5_universal_context_parent_join,
    )

    if not isinstance(sidecar_id, str) or not sidecar_id.strip():
        raise ValueError("sidecar_id must be non-empty")
    if not isinstance(protocol, V5FrozenExactSearchProtocol):
        raise TypeError("protocol must be V5FrozenExactSearchProtocol")
    parent, receipt = read_v5_grouped_dataset(parent_dataset_path)
    split_id = validate_v5_search_sidecar_parent(parent)
    items = tuple(specs)
    if not items or not all(isinstance(value, V5UniversalSearchSpec) for value in items):
        raise ValueError("specs must contain V5UniversalSearchSpec values")
    indices = tuple(value.parent_observation_index for value in items)
    if len(set(indices)) != len(indices) or set(indices) != set(range(parent.observation_count)):
        raise ValueError("search sidecar must cover every parent observation exactly once")
    items = tuple(sorted(items, key=lambda value: value.parent_observation_index))

    query_values: dict[str, list[object]] = {}
    branch_values: dict[str, list[object]] = {}
    supervisions: list[V5CandidateSupervision] = []
    recipe_indices: list[int] = []
    outcome_counts = {name: 0 for name in SEARCH_OUTCOME_CODE}
    evidence_index_complete = True
    clean_query_catalogs: dict[str, tuple[str, str, str]] = {}
    for query_index, spec in enumerate(items):
        context = spec.context
        recipe_index, _ = validate_v5_universal_context_parent_join(
            parent,
            spec.parent_observation_index,
            context,
            spec.exact_observation,
        )
        clean_group = str(parent.arrays[clean_array("clean_group_id")][recipe_index])
        recipe_id = str(parent.arrays[clean_array("recipe_id")][recipe_index])
        observation_id = str(
            parent.arrays[observation_array("observation_id")][spec.parent_observation_index]
        )
        intensity_reference = float(
            parent.arrays[observation_array("intensity_reference")][
                spec.parent_observation_index
            ]
        )
        if not np.isfinite(intensity_reference) or intensity_reference <= 0.0:
            raise ValueError("parent intensity reference must be finite and positive")
        intensity_reference_digest = array_sha256(
            query_array("intensity_reference"),
            np.asarray(intensity_reference, dtype=np.float64),
        )
        exact = spec.exact_observation
        topology_catalog_json = topology_query_catalog_json(context)
        topology_catalog_sha = sha256(topology_catalog_json.encode("utf-8")).hexdigest()
        catalog_binding = (
            spec.query_catalog_artifact_id,
            spec.query_catalog_artifact_sha256,
            topology_catalog_sha,
        )
        previous_binding = clean_query_catalogs.setdefault(clean_group, catalog_binding)
        if previous_binding != catalog_binding:
            raise ValueError(
                "all observation views of one clean group require one frozen topology query catalog"
            )
        exact_q, exact_intensity, exact_sigma_log, exact_mask = _padded_exact_curve(
            exact
        )
        has_acceptance_sigma = exact.observed_curve.sigma_log is not None
        tasks = tuple(
            V5FrozenSearchTask(
                query_index=query_index,
                clean_group_id=clean_group,
                recipe_id=recipe_id,
                observation_id=observation_id,
                universal_context=context,
                exact_observation=exact,
                query_catalog_artifact_id=spec.query_catalog_artifact_id,
                query_catalog_artifact_sha256=spec.query_catalog_artifact_sha256,
                branch_index=branch_index,
                protocol=protocol,
                calibrated_threshold=spec.calibrated_threshold,
            )
            for branch_index in range(context.branch_count)
        )
        threshold_task = tasks[0]
        selected_metric = threshold_task.selected_metric_name
        selected_threshold_name = threshold_task.selected_threshold_name
        selected_threshold_value = threshold_task.selected_threshold_value
        selected_threshold_source_id = threshold_task.selected_threshold_source_id
        threshold_binding = {
            "protocol_tier": protocol.protocol_tier,
            "selected_metric_name": selected_metric,
            "selected_threshold_name": selected_threshold_name,
            "selected_threshold_value": selected_threshold_value,
            "selected_threshold_source_id": selected_threshold_source_id,
            "calibrated_threshold": (
                None
                if spec.calibrated_threshold is None
                else spec.calibrated_threshold.audit_payload()
            ),
            "calibrated_threshold_sha256": (
                None
                if spec.calibrated_threshold is None
                else spec.calibrated_threshold.sha256
            ),
            "full_training_label_permitted": (
                threshold_task.full_training_label_permitted
            ),
        }
        threshold_binding_json = canonical_json(threshold_binding)
        catalog_rows = [
            {
                "global_branch_key": branch.global_key.wire_key,
                "topology_query_sha256": branch.topology_query_sha256,
                "context_sha256": branch.context_sha256,
                "amplitude_constraint_sha256": branch.amplitude_constraint_sha256,
            }
            for branch in context.branches
        ]
        query_row = {
            "clean_group_id": clean_group,
            "recipe_id": recipe_id,
            "observation_id": observation_id,
            "parent_recipe_index": recipe_index,
            "parent_observation_index": spec.parent_observation_index,
            "split_id": split_id,
            "universal_query_sha256": context.audit_sha256,
            "universal_query_audit_json": context.audit_json,
            "observation_inputs_sha256": observation_inputs_sha256(
                {
                    name: parent.arrays[observation_input(name)][
                        spec.parent_observation_index
                    ]
                    for name in OBSERVATION_MODEL_INPUT_KEYS
                }
            ),
            "intensity_reference": intensity_reference,
            "intensity_reference_sha256": intensity_reference_digest,
            "parent_observation_audit_sha256": exact.parent_observation_audit_sha256,
            "exact_observation_audit_json": exact.audit_json,
            "exact_observation_audit_sha256": exact.audit_sha256,
            "exact_curve_id": exact.observed_curve.curve_id,
            "exact_curve_source_kind": exact.observed_curve.source_kind,
            "exact_curve_sha256": exact.curve_sha256,
            "exact_curve_valid_count": int(exact.observed_curve.q.size),
            "exact_curve_q": exact_q,
            "exact_curve_intensity": exact_intensity,
            "exact_curve_sigma_log": exact_sigma_log,
            "exact_curve_point_mask": exact_mask,
            "acceptance_sigma_log_available": has_acceptance_sigma,
            "acceptance_sigma_source_id": exact.acceptance_sigma_source_id,
            "preprocessed_valid_arrays_sha256": exact.preprocessed_valid_arrays_sha256,
            "selected_metric_name": selected_metric,
            "selected_threshold_name": selected_threshold_name,
            "selected_threshold_value": selected_threshold_value,
            "selected_threshold_source_id": selected_threshold_source_id,
            "selected_threshold_binding_json": threshold_binding_json,
            "selected_threshold_binding_sha256": sha256(
                threshold_binding_json.encode("utf-8")
            ).hexdigest(),
            "full_training_label_permitted": (
                threshold_task.full_training_label_permitted
            ),
            "query_catalog_artifact_id": spec.query_catalog_artifact_id,
            "query_catalog_artifact_sha256": spec.query_catalog_artifact_sha256,
            "topology_query_catalog_json": topology_catalog_json,
            "topology_query_catalog_sha256": topology_catalog_sha,
            "selected_topology_ids_json": canonical_json(list(context.topology_ids)),
            "catalog_sha256": _catalog_sha256(catalog_rows),
            "branch_count": context.branch_count,
        }
        for name, value in query_row.items():
            query_values.setdefault(query_array(name), []).append(value)

        combined_inputs = context.for_model()
        for branch_index, (branch, task) in enumerate(zip(context.branches, tasks)):
            result = runner(task)
            validate_v5_frozen_search_result(task, result)
            evidence_binding = None
            binding_provider = getattr(runner, "evidence_binding", None)
            if result.completed and callable(binding_provider):
                evidence_binding = dict(binding_provider(task, result))
            if result.completed and evidence_binding is None:
                evidence_index_complete = False
            search_id, search_json, search_sha = build_v5_search_record(task, result)
            supervision = build_v5_candidate_supervision(
                task, result, search_id, search_sha
            )
            topology_entry = context.topology_queries[branch.topology_batch_index]
            values = {
                "query_index": query_index,
                "global_branch_key": branch.global_key.wire_key,
                "topology_id": branch.global_key.topology_id,
                "pattern_id": branch.global_key.pattern_id,
                "declared_topology_query_sha256": topology_entry.sha256,
                "candidate_query_sha256": branch.topology_query_sha256,
                "geometry_query_sha256": topology_entry.geometry.sha256,
                "amplitude_query_sha256": topology_entry.amplitude.sha256,
                "context_sha256": branch.context_sha256,
                "amplitude_constraint_sha256": branch.amplitude_constraint_sha256,
                "amplitude_constraint_json": canonical_json(
                    branch.amplitude_constraint.to_audit_dict()
                ),
                "frozen_exact_forward_call_budget": protocol.exact_forward_call_budget,
                "exact_forward_calls_used": result.exact_forward_calls_used,
                "runner_completed": result.completed,
                "runner_termination_reason": result.termination_reason,
                "executor_artifact_id": result.executor_artifact_id or "",
                "executor_artifact_sha256": result.executor_artifact_sha256 or "",
                "executor_artifact_relative_path": (
                    "" if evidence_binding is None else evidence_binding["relative_path"]
                ),
                "executor_artifact_schema": (
                    "" if evidence_binding is None else evidence_binding["artifact_schema"]
                ),
                "executor_artifact_version": (
                    "" if evidence_binding is None else evidence_binding["artifact_version"]
                ),
                "executor_source_bundle_sha256": (
                    ""
                    if evidence_binding is None
                    else evidence_binding["source_bundle_sha256"]
                ),
                "task_audit_sha256": task.audit_sha256,
                "search_record_json": search_json,
                "search_record_sha256": search_sha,
                "compatible_representatives_json": canonical_json(
                    [value.audit_payload() for value in result.representatives]
                ),
                "supervision_audit_json": canonical_json(supervision.audit_payload()),
            }
            for name, value in values.items():
                branch_values.setdefault(branch_array(name), []).append(value)
            for name in BRANCH_MODEL_INPUT_KEYS:
                branch_values.setdefault(branch_input(name), []).append(
                    combined_inputs[name][branch_index]
                )
            supervisions.append(supervision)
            recipe_indices.append(recipe_index)
            outcome_counts[result.outcome] += 1

    arrays = {
        **{
            name: np.asarray(
                values,
                dtype=(
                    np.bool_
                    if name
                    in {
                        query_array("exact_curve_point_mask"),
                        query_array("acceptance_sigma_log_available"),
                        query_array("full_training_label_permitted"),
                    }
                    else np.float64
                    if name
                    in {
                        query_array("exact_curve_q"),
                        query_array("exact_curve_intensity"),
                        query_array("exact_curve_sigma_log"),
                        query_array("selected_threshold_value"),
                        query_array("intensity_reference"),
                    }
                    else np.int32
                    if name
                    in {
                        query_array("parent_recipe_index"),
                        query_array("parent_observation_index"),
                        query_array("branch_count"),
                        query_array("exact_curve_valid_count"),
                    }
                    else np.str_
                ),
            )
            for name, values in query_values.items()
        },
        **{
            name: np.asarray(
                values,
                dtype=(
                    np.bool_
                    if name == branch_array("runner_completed")
                    else np.int32
                    if name
                    in {
                        branch_array("query_index"),
                        branch_array("topology_id"),
                        branch_array("pattern_id"),
                        branch_array("frozen_exact_forward_call_budget"),
                        branch_array("exact_forward_calls_used"),
                    }
                    else np.str_
                ),
            )
            for name, values in branch_values.items()
            if not name.startswith("branch__input__")
        },
        **{
            name: np.asarray(
                values,
                dtype=(
                    np.int32
                    if name in {branch_input("branch_topology_id"), branch_input("branch_pattern_id")}
                    else np.float32
                ),
            )
            for name, values in branch_values.items()
            if name.startswith("branch__input__")
        },
    }
    labels = stack_candidate_supervision_v5(
        supervisions, clean_recipe_indices=recipe_indices
    )
    arrays.update({branch_label(name): value for name, value in labels.items()})
    counts = {
        "universal_queries": len(items),
        "branch_outcomes": len(supervisions),
        "compatible_found": outcome_counts["compatible_found"],
        "completed_negative": outcome_counts[
            "no_compatible_found_within_frozen_search_budget"
        ],
        "unverified": outcome_counts["unverified"],
        "exact_forward_call_budget_total": len(supervisions)
        * protocol.exact_forward_call_budget,
        "exact_forward_calls_used_total": int(
            np.sum(arrays[branch_array("exact_forward_calls_used")])
        ),
    }
    manifest = _manifest(
        sidecar_id=sidecar_id.strip(),
        parent=parent,
        parent_receipt=receipt,
        protocol=protocol,
        split_id=split_id,
        arrays=arrays,
        counts=counts,
        executor_evidence_index_complete=evidence_index_complete,
    )
    return V5SearchSupervisionSidecar(manifest, arrays)


def write_v5_search_supervision_sidecar(
    sidecar: V5SearchSupervisionSidecar,
    path: str | os.PathLike[str],
) -> V5ArtifactReceipt:
    if not isinstance(sidecar, V5SearchSupervisionSidecar):
        raise TypeError("sidecar must be V5SearchSupervisionSidecar")
    return write_checked_array_artifact(path, manifest=sidecar.manifest, arrays=sidecar.arrays)


def read_v5_search_supervision_sidecar(
    path: str | os.PathLike[str],
) -> tuple[V5SearchSupervisionSidecar, V5ArtifactReceipt]:
    manifest, arrays, receipt = read_checked_array_artifact(path)
    return V5SearchSupervisionSidecar(manifest, arrays), receipt


__all__ = [
    "BRANCH_FIELDS",
    "BRANCH_MODEL_INPUT_KEYS",
    "OBSERVATION_MODEL_INPUT_KEYS",
    "QUERY_FIELDS",
    "V5FrozenSearchTask",
    "V5SearchSupervisionSidecar",
    "V5UniversalSearchSpec",
    "V5_SEARCH_RECORD_SCHEMA",
    "V5_SEARCH_SIDECAR_SCHEMA",
    "V5_SEARCH_SIDECAR_ALLOWED_SPLITS",
    "V5_SEARCH_SIDECAR_VERSION",
    "branch_array",
    "branch_input",
    "branch_label",
    "collect_v5_search_supervision_sidecar",
    "observation_inputs_sha256",
    "query_array",
    "read_v5_search_supervision_sidecar",
    "unverified_v5_search_result",
    "write_v5_search_supervision_sidecar",
]
