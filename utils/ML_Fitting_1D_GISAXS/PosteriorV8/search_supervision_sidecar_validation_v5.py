"""Semantic validator for immutable V5.1 frozen-search sidecars."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Mapping

import numpy as np

from .candidate_supervision_v5 import (
    CANDIDATE_SUPERVISION_V5_SCHEMA,
    CANDIDATE_SUPERVISION_V5_VERSION,
    CANDIDATE_SUPERVISION_TENSOR_KEYS,
    NEGATIVE_TERMINATION_REASONS,
    POSITIVE_TERMINATION_REASON,
    SEARCH_OUTCOME_CODE,
    candidate_supervision_v5_contract_payload,
)
from .candidate_batch_v5 import V5_USER_QUERY_DIGEST_VERSION
from .bounds_query_v5 import bounds_query_from_json, branch_condition
from .calibrated_search_threshold_v5 import V5CalibratedObservationThreshold
from .compatibility_calibration import (
    CompatibilityStratum,
    acquisition_policy_payload,
)
from .evaluation import ObservedCurve
from .exact_search_executor_v5 import (
    V5_EXACT_SEARCH_EXECUTOR_SCHEMA,
    V5_EXACT_SEARCH_EXECUTOR_VERSION,
)
from .grouped_artifact_v5 import (
    V5_CHECKED_ARRAY_ARTIFACT_SCHEMA,
    V5_CHECKED_ARRAY_ARTIFACT_VERSION,
    array_sha256,
    canonical_json,
)
from .grouped_amplitude_join_v5 import amplitude_query_from_json
from .model_v5_contract import model_v5_contract_payload
from .search_supervision_contract_v5 import (
    V5_EXACT_SEARCH_OBSERVATION_SCHEMA,
    V5_EXACT_SEARCH_OBSERVATION_VERSION,
    V5CompatibleRepresentativeReference,
    V5FrozenExactSearchProtocol,
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
    observed_curve_sha256,
)
from .search_supervision_evidence_v5 import V5_SEARCH_RECORD_SCHEMA
from .search_supervision_sidecar_v5 import (
    BRANCH_FIELDS,
    BRANCH_MODEL_INPUT_KEYS,
    QUERY_FIELDS,
    V5_SEARCH_SIDECAR_ALLOWED_SPLITS,
    V5_SEARCH_SIDECAR_SCHEMA,
    V5_SEARCH_SIDECAR_VERSION,
    branch_array,
    branch_input,
    branch_label,
    query_array,
)
from .universal_query_contract_v5 import (
    V5GlobalBranchKey,
    V5TopologyQuery,
    amplitude_constraint_sha256,
    branch_context_sha256,
)
from .universal_query_v5 import V5_UNIVERSAL_QUERY_SCHEMA, V5_UNIVERSAL_QUERY_VERSION
from .universal_training_contract_v5 import (
    V5_CROSS_TOPOLOGY_VERIFIED_SEARCH_STAGE,
    universal_training_contract_v5_payload,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_FIELDS = {
    "container_schema",
    "container_version",
    "sidecar_schema",
    "sidecar_version",
    "sidecar_id",
    "stage",
    "split_id",
    "parent_grouped_artifact",
    "protocol",
    "protocol_sha256",
    "contract_bundle",
    "contract_bundle_sha256",
    "build_policy",
    "executor_evidence_index_complete",
    "full_training_labels_permitted",
    "counts",
    "table_fields",
    "source_sha256",
    "arrays",
    "manifest_sha256",
}
_POLICY = {
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
}


def _digest_text(*values: str) -> str:
    digest = sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _strict_json(encoded: object, name: str):
    if not isinstance(encoded, str):
        raise ValueError(f"{name} must be canonical JSON")
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if canonical_json(value) != encoded:
        raise ValueError(f"{name} must use canonical JSON")
    return value


def _contract_bundle() -> dict[str, object]:
    return {
        "model": model_v5_contract_payload(),
        "candidate_supervision": candidate_supervision_v5_contract_payload(),
        "universal_training": universal_training_contract_v5_payload(),
    }


def _validate_manifest(manifest: Mapping[str, object]) -> V5FrozenExactSearchProtocol:
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("search-sidecar manifest fields are incomplete or unsupported")
    if (
        manifest["container_schema"] != V5_CHECKED_ARRAY_ARTIFACT_SCHEMA
        or manifest["container_version"] != V5_CHECKED_ARRAY_ARTIFACT_VERSION
        or manifest["sidecar_schema"] != V5_SEARCH_SIDECAR_SCHEMA
        or manifest["sidecar_version"] != V5_SEARCH_SIDECAR_VERSION
    ):
        raise ValueError("unsupported search-sidecar identity")
    if not isinstance(manifest["sidecar_id"], str) or not manifest["sidecar_id"].strip():
        raise ValueError("search-sidecar ID must be non-empty")
    if (
        manifest["stage"] != V5_CROSS_TOPOLOGY_VERIFIED_SEARCH_STAGE
        or manifest["split_id"] not in V5_SEARCH_SIDECAR_ALLOWED_SPLITS
    ):
        raise ValueError("search sidecar escaped its model-development verified-search stage")
    parent = manifest["parent_grouped_artifact"]
    if not isinstance(parent, Mapping) or set(parent) != {
        "dataset_id",
        "dataset_schema",
        "dataset_version",
        "stage",
        "artifact_sha256",
        "manifest_sha256",
    }:
        raise ValueError("search-sidecar parent binding is incomplete")
    for name in ("dataset_id", "dataset_schema", "dataset_version", "stage"):
        if not isinstance(parent[name], str) or not parent[name].strip():
            raise ValueError("search-sidecar parent identity is empty")
    _digest(parent["artifact_sha256"], "parent artifact SHA-256")
    _digest(parent["manifest_sha256"], "parent manifest SHA-256")
    if not isinstance(manifest["protocol"], Mapping):
        raise ValueError("search-sidecar protocol must be a mapping")
    protocol = V5FrozenExactSearchProtocol(**dict(manifest["protocol"]))
    if manifest["protocol_sha256"] != protocol.sha256:
        raise ValueError("search-sidecar protocol SHA-256 does not reproduce")
    evidence_complete = manifest["executor_evidence_index_complete"]
    training_permitted = manifest["full_training_labels_permitted"]
    if type(evidence_complete) is not bool or type(training_permitted) is not bool:
        raise ValueError("search-sidecar evidence/training flags must be boolean")
    if training_permitted is not (
        protocol.protocol_tier == V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED
    ):
        raise ValueError("search-sidecar training flag disagrees with its protocol tier")
    bundle = _contract_bundle()
    bundle_sha = sha256(canonical_json(bundle).encode("utf-8")).hexdigest()
    if manifest["contract_bundle"] != bundle or manifest["contract_bundle_sha256"] != bundle_sha:
        raise ValueError("search-sidecar contract bundle is incompatible")
    if manifest["build_policy"] != _POLICY:
        raise ValueError("search-sidecar build policy is incompatible")
    expected_tables = {
        "query": list(QUERY_FIELDS),
        "branch": list(BRANCH_FIELDS),
        "branch_inputs": list(BRANCH_MODEL_INPUT_KEYS),
        "branch_labels": list(CANDIDATE_SUPERVISION_TENSOR_KEYS),
    }
    if manifest["table_fields"] != expected_tables:
        raise ValueError("search-sidecar table inventory is incompatible")
    sources = manifest["source_sha256"]
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("search-sidecar source hashes are missing")
    for name, digest in sources.items():
        if not isinstance(name, str) or not name:
            raise ValueError("search-sidecar source hash name is empty")
        _digest(digest, f"source_sha256[{name}]")
    core = dict(manifest)
    supplied = _digest(core.pop("manifest_sha256"), "manifest_sha256")
    if supplied != sha256(canonical_json(core).encode("utf-8")).hexdigest():
        raise ValueError("search-sidecar manifest SHA-256 does not reproduce")
    return protocol


def validate_v5_search_supervision_sidecar(
    manifest: Mapping[str, object], arrays: Mapping[str, np.ndarray]
) -> None:
    protocol = _validate_manifest(manifest)
    expected_names = {
        *(query_array(name) for name in QUERY_FIELDS),
        *(branch_array(name) for name in BRANCH_FIELDS),
        *(branch_input(name) for name in BRANCH_MODEL_INPUT_KEYS),
        *(branch_label(name) for name in CANDIDATE_SUPERVISION_TENSOR_KEYS),
    }
    if set(arrays) != expected_names:
        raise ValueError("search-sidecar arrays are incomplete or unsupported")
    _validate_tables(manifest, arrays, protocol)


def _validate_tables(manifest, arrays, protocol) -> None:
    counts = manifest["counts"]
    expected_count_fields = {
        "universal_queries",
        "branch_outcomes",
        "compatible_found",
        "completed_negative",
        "unverified",
        "exact_forward_call_budget_total",
        "exact_forward_calls_used_total",
    }
    if not isinstance(counts, Mapping) or set(counts) != expected_count_fields:
        raise ValueError("search-sidecar counts are incomplete")
    if any(isinstance(value, (bool, np.bool_)) or int(value) != value for value in counts.values()):
        raise ValueError("search-sidecar counts must be integers")
    query_count, branch_count = int(counts["universal_queries"]), int(
        counts["branch_outcomes"]
    )
    if min(query_count, branch_count) < 1:
        raise ValueError("search sidecar cannot be empty")
    if any(arrays[query_array(name)].shape[0] != query_count for name in QUERY_FIELDS):
        raise ValueError("search-sidecar query table length mismatch")
    branch_names = (
        *(branch_array(name) for name in BRANCH_FIELDS),
        *(branch_input(name) for name in BRANCH_MODEL_INPUT_KEYS),
        *(branch_label(name) for name in CANDIDATE_SUPERVISION_TENSOR_KEYS),
    )
    if any(arrays[name].shape[0] != branch_count for name in branch_names):
        raise ValueError("search-sidecar branch table length mismatch")
    parent_observations = arrays[query_array("parent_observation_index")]
    if (
        not np.array_equal(parent_observations, np.arange(query_count, dtype=np.int32))
        or len(set(arrays[query_array("observation_id")].tolist())) != query_count
    ):
        raise ValueError("search sidecar has missing or duplicate parent observations")
    if set(arrays[query_array("split_id")].tolist()) != {manifest["split_id"]}:
        raise ValueError("search-sidecar query rows crossed their development split")
    query_indices = arrays[branch_array("query_index")]
    if np.any(query_indices < 0) or np.any(query_indices >= query_count):
        raise ValueError("search-sidecar branch/query foreign key is invalid")
    seen = set()
    clean_catalogs = {}
    for query_index in range(query_count):
        rows = np.flatnonzero(query_indices == query_index)
        if rows.size != int(arrays[query_array("branch_count")][query_index]):
            raise ValueError("search-sidecar universal catalog is partial")
        catalog_binding = _validate_query_observation(
            arrays, query_index, protocol
        )
        clean_group = str(arrays[query_array("clean_group_id")][query_index])
        previous = clean_catalogs.setdefault(clean_group, catalog_binding)
        if previous != catalog_binding:
            raise ValueError(
                "observation views of one clean group use different frozen topology catalogs"
            )
        _validate_query_catalog(arrays, query_index, rows, protocol, seen)
    _validate_executor_evidence_index(manifest, arrays)
    _validate_counts(counts, arrays, protocol, branch_count)


def _validate_executor_evidence_index(manifest, arrays) -> None:
    complete = manifest["executor_evidence_index_complete"]
    indexed_rows = 0
    completed_rows = 0
    source_bundles: set[str] = set()
    for row in range(int(manifest["counts"]["branch_outcomes"])):
        completed = bool(arrays[branch_array("runner_completed")][row])
        completed_rows += int(completed)
        values = {
            name: str(arrays[branch_array(name)][row])
            for name in (
                "executor_artifact_relative_path",
                "executor_artifact_schema",
                "executor_artifact_version",
                "executor_source_bundle_sha256",
            )
        }
        if not completed:
            if any(values.values()):
                raise ValueError("unverified branch carries a completed executor index")
            continue
        if not complete:
            if any(values.values()):
                raise ValueError("executor evidence index is only partially populated")
            continue
        relative = PurePosixPath(values["executor_artifact_relative_path"])
        if (
            not values["executor_artifact_relative_path"]
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix != ".gvd5"
            or values["executor_artifact_schema"]
            != V5_EXACT_SEARCH_EXECUTOR_SCHEMA
            or values["executor_artifact_version"]
            != V5_EXACT_SEARCH_EXECUTOR_VERSION
        ):
            raise ValueError("executor evidence index path/schema/version is incompatible")
        source_bundles.add(
            _digest(
                values["executor_source_bundle_sha256"],
                "executor source bundle SHA-256",
            )
        )
        indexed_rows += 1
    if complete and indexed_rows != completed_rows:
        raise ValueError("complete executor evidence index omitted a completed branch")
    if indexed_rows and len(source_bundles) != 1:
        raise ValueError("executor evidence index mixes scientific source bundles")
    expected_training = (
        manifest["protocol"]["protocol_tier"]
        == V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED
    )
    if manifest["full_training_labels_permitted"] is not expected_training:
        raise ValueError("sidecar full-training flag escaped its protocol tier")
    flags = arrays[query_array("full_training_label_permitted")]
    if not np.all(flags == expected_training):
        raise ValueError("query training flags disagree with sidecar protocol tier")


def _validate_query_observation(arrays, query_index, protocol):
    q = np.asarray(arrays[query_array("exact_curve_q")][query_index])
    intensity = np.asarray(
        arrays[query_array("exact_curve_intensity")][query_index]
    )
    sigma_log = np.asarray(
        arrays[query_array("exact_curve_sigma_log")][query_index]
    )
    mask = np.asarray(
        arrays[query_array("exact_curve_point_mask")][query_index]
    )
    valid_count = int(arrays[query_array("exact_curve_valid_count")][query_index])
    if (
        q.ndim != 1
        or intensity.shape != q.shape
        or sigma_log.shape != q.shape
        or mask.shape != q.shape
        or mask.dtype.kind != "b"
        or np.count_nonzero(mask) != valid_count
        or valid_count < 1
        or np.any(q[~mask] != 0.0)
        or np.any(intensity[~mask] != 0.0)
        or np.any(sigma_log[~mask] != 0.0)
    ):
        raise ValueError("exact-search curve arrays are malformed or partially masked")
    has_sigma = bool(
        arrays[query_array("acceptance_sigma_log_available")][query_index]
    )
    curve = ObservedCurve(
        curve_id=str(arrays[query_array("exact_curve_id")][query_index]),
        source_kind=str(
            arrays[query_array("exact_curve_source_kind")][query_index]
        ),
        q=q[mask],
        intensity=intensity[mask],
        sigma_log=sigma_log[mask] if has_sigma else None,
    )
    curve_sha = observed_curve_sha256(curve)
    if curve_sha != arrays[query_array("exact_curve_sha256")][query_index]:
        raise ValueError("exact-search physical curve SHA-256 does not reproduce")
    source = str(
        arrays[query_array("acceptance_sigma_source_id")][query_index]
    )
    if (
        (has_sigma and not source.endswith("|acceptance_sigma_log"))
        or (
            not has_sigma
            and not source.endswith("|acceptance_sigma_absent_use_raw_metric")
        )
    ):
        raise ValueError("exact-search acceptance sigma provenance is incompatible")
    binding_json = str(
        arrays[query_array("selected_threshold_binding_json")][query_index]
    )
    binding = _strict_json(binding_json, "selected threshold binding")
    if sha256(binding_json.encode("utf-8")).hexdigest() != arrays[
        query_array("selected_threshold_binding_sha256")
    ][query_index]:
        raise ValueError("selected threshold binding SHA-256 does not reproduce")
    formal = protocol.protocol_tier == V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED
    calibrated_payload = binding.get("calibrated_threshold")
    if formal:
        if not has_sigma:
            raise ValueError("paper/full calibrated sidecar cannot label missing sigma")
        calibrated = V5CalibratedObservationThreshold.from_payload(
            calibrated_payload
        )
        suffix = "|acceptance_sigma_log"
        policy_id = source[: -len(suffix)]
        policy = acquisition_policy_payload(policy_id)
        expected_stratum = CompatibilityStratum(
            point_count=int(policy["grid"]["design_point_count"]),
            noise_id=str(policy["sigma"]["noise_id"]),
            q_window_id=str(policy["grid"]["q_window_id"]),
        )
        if (
            calibrated.calibration_identity != protocol.calibration_identity
            or calibrated.stratum != expected_stratum
            or calibrated.acquisition_policy_id_sha256
            != sha256(policy_id.encode("utf-8")).hexdigest()
        ):
            raise ValueError(
                "selected threshold escaped its calibration artifact or acquisition stratum"
            )
        expected_metric = calibrated.metric_name
        expected_threshold_name = calibrated.threshold_name
        expected_threshold = calibrated.threshold_value
        expected_source = calibrated.threshold_source_id
        expected_calibrated_sha = calibrated.sha256
        expected_calibrated_payload = calibrated.audit_payload()
    else:
        if calibrated_payload is not None:
            raise ValueError("engineering-pilot sidecar cannot carry calibrated threshold")
        expected_metric = (
            protocol.metric_name
            if has_sigma
            else protocol.missing_acceptance_sigma_metric_name
        )
        expected_threshold_name = (
            protocol.threshold_name
            if has_sigma
            else protocol.missing_acceptance_sigma_threshold_name
        )
        expected_threshold = (
            protocol.threshold_value
            if has_sigma
            else protocol.missing_acceptance_sigma_threshold_value
        )
        expected_source = protocol.threshold_source_id
        expected_calibrated_sha = None
        expected_calibrated_payload = None
    expected_binding = {
        "protocol_tier": protocol.protocol_tier,
        "selected_metric_name": expected_metric,
        "selected_threshold_name": expected_threshold_name,
        "selected_threshold_value": expected_threshold,
        "selected_threshold_source_id": expected_source,
        "calibrated_threshold": expected_calibrated_payload,
        "calibrated_threshold_sha256": expected_calibrated_sha,
        "full_training_label_permitted": formal,
    }
    if (
        binding_json != canonical_json(expected_binding)
        or arrays[query_array("selected_metric_name")][query_index] != expected_metric
        or arrays[query_array("selected_threshold_name")][query_index]
        != expected_threshold_name
        or arrays[query_array("selected_threshold_source_id")][query_index]
        != expected_source
        or bool(
            arrays[query_array("full_training_label_permitted")][query_index]
        )
        is not formal
        or not np.isclose(
            arrays[query_array("selected_threshold_value")][query_index],
            expected_threshold,
            rtol=0.0,
            atol=np.finfo(np.float64).eps,
        )
    ):
        raise ValueError("exact-search metric selection disagrees with sigma evidence")
    parent_audit_sha = _digest(
        str(arrays[query_array("parent_observation_audit_sha256")][query_index]),
        "parent observation audit SHA-256",
    )
    exact_audit_json = str(
        arrays[query_array("exact_observation_audit_json")][query_index]
    )
    exact_audit = _strict_json(exact_audit_json, "exact observation audit")
    expected_exact_audit = {
        "schema_version": V5_EXACT_SEARCH_OBSERVATION_SCHEMA,
        "version": V5_EXACT_SEARCH_OBSERVATION_VERSION,
        "exact_curve_sha256": curve_sha,
        "exact_curve_id": curve.curve_id,
        "exact_curve_source_kind": curve.source_kind,
        "exact_curve_point_count": valid_count,
        "acceptance_sigma_log_available": has_sigma,
        "acceptance_sigma_source_id": source,
        "parent_observation_audit_sha256": parent_audit_sha,
        "preprocessed_valid_arrays_sha256": str(
            arrays[query_array("preprocessed_valid_arrays_sha256")][query_index]
        ),
    }
    if (
        exact_audit != expected_exact_audit
        or sha256(exact_audit_json.encode("utf-8")).hexdigest()
        != arrays[query_array("exact_observation_audit_sha256")][query_index]
    ):
        raise ValueError("exact-search observation audit does not reproduce")
    catalog_json = str(
        arrays[query_array("topology_query_catalog_json")][query_index]
    )
    _strict_json(catalog_json, "topology query catalog")
    catalog_sha = sha256(catalog_json.encode("utf-8")).hexdigest()
    if catalog_sha != arrays[query_array("topology_query_catalog_sha256")][query_index]:
        raise ValueError("topology query catalog SHA-256 does not reproduce")
    artifact_id = str(
        arrays[query_array("query_catalog_artifact_id")][query_index]
    )
    if not artifact_id.strip():
        raise ValueError("topology query catalog artifact ID is empty")
    artifact_sha = _digest(
        str(arrays[query_array("query_catalog_artifact_sha256")][query_index]),
        "topology query catalog artifact SHA-256",
    )
    return artifact_id, artifact_sha, catalog_sha


def _validate_query_catalog(arrays, query_index, rows, protocol, seen) -> None:
    encoded = str(arrays[query_array("universal_query_audit_json")][query_index])
    audit = _strict_json(encoded, "universal_query_audit_json")
    query_sha = sha256(encoded.encode("utf-8")).hexdigest()
    if (
        audit.get("schema_version") != V5_UNIVERSAL_QUERY_SCHEMA
        or audit.get("version") != V5_UNIVERSAL_QUERY_VERSION
        or query_sha != arrays[query_array("universal_query_sha256")][query_index]
        or audit.get("branch_count") != rows.size
    ):
        raise ValueError("universal-query audit identity or branch count is incompatible")
    selected_ids = _strict_json(
        canonical_json({"ids": audit.get("selected_topology_ids")}), "selected topology IDs"
    )["ids"]
    if canonical_json(selected_ids) != arrays[query_array("selected_topology_ids_json")][query_index]:
        raise ValueError("selected topology IDs disagree with universal-query audit")
    expected = audit.get("branches")
    topology_values = audit.get("topology_queries")
    if (
        not isinstance(expected, list)
        or len(expected) != rows.size
        or not isinstance(topology_values, list)
    ):
        raise ValueError("universal-query audit has a partial branch catalog")
    topology_queries = _validate_topology_query_catalog(
        arrays,
        query_index,
        topology_values,
        selected_ids,
    )
    intensity_reference = _persisted_intensity_reference(arrays, query_index)
    reconstructed = _reconstruct_branch_catalog(
        arrays,
        rows,
        topology_queries,
        topology_values,
        intensity_reference,
    )
    if audit.get("global_row_slices") != [
        value["global_row_slice"] for value in topology_values
    ]:
        raise ValueError("universal-query topology row slices do not reproduce")
    catalog_rows = []
    for row, expected_branch, reconstructed_branch in zip(rows, expected, reconstructed):
        catalog_rows.append(
            _validate_branch_identity(
                arrays,
                row,
                query_index,
                query_sha,
                expected_branch,
                reconstructed_branch,
                seen,
            )
        )
        _validate_search_row(arrays, row, query_index, protocol)
    expected_catalog_sha = sha256(canonical_json(catalog_rows).encode("utf-8")).hexdigest()
    if expected_catalog_sha != arrays[query_array("catalog_sha256")][query_index]:
        raise ValueError("universal-query catalog SHA-256 does not reproduce")


def _validate_topology_query_catalog(
    arrays,
    query_index,
    topology_values,
    selected_ids,
) -> tuple[V5TopologyQuery, ...]:
    catalog = _strict_json(
        str(arrays[query_array("topology_query_catalog_json")][query_index]),
        "topology query catalog",
    )
    if (
        not isinstance(catalog, list)
        or len(catalog) != len(topology_values)
        or [value.get("topology_id") for value in catalog] != selected_ids
    ):
        raise ValueError("topology query catalog is missing, reordered, or partial")
    result = []
    for entry, audit in zip(catalog, topology_values):
        if not isinstance(entry, Mapping) or set(entry) != {
            "topology_id",
            "topology_query_sha256",
            "geometry_query_sha256",
            "geometry_query_canonical_json",
            "amplitude_query_sha256",
            "amplitude_query_canonical_json",
        }:
            raise ValueError("topology query catalog entry is incomplete")
        geometry_json = entry["geometry_query_canonical_json"]
        amplitude_json = entry["amplitude_query_canonical_json"]
        geometry_sha = str(entry["geometry_query_sha256"])
        amplitude_sha = str(entry["amplitude_query_sha256"])
        geometry = bounds_query_from_json(geometry_json, geometry_sha)
        amplitude = amplitude_query_from_json(amplitude_json, amplitude_sha)
        topology_query = V5TopologyQuery(geometry, amplitude)
        candidate_sha = _digest_text(
            V5_USER_QUERY_DIGEST_VERSION,
            geometry_sha,
            amplitude_sha,
        )
        if (
            entry["topology_id"] != topology_query.topology_id
            or entry["topology_query_sha256"] != topology_query.sha256
            or not isinstance(audit, Mapping)
            or {
                name: audit.get(name) for name in topology_query.audit_payload()
            }
            != topology_query.audit_payload()
            or topology_query.sha256 != audit.get("topology_query_sha256")
            or candidate_sha != audit.get("candidate_query_sha256")
        ):
            raise ValueError(
                "explicit topology geometry/amplitude ranges disagree with universal audit"
            )
        _digest(audit.get("candidate_batch_audit_sha256"), "candidate batch audit SHA-256")
        result.append(topology_query)
    return tuple(result)


def _same_array_bytes(actual: object, expected: np.ndarray) -> bool:
    value = np.asarray(actual)
    return (
        value.dtype == expected.dtype
        and value.shape == expected.shape
        and np.ascontiguousarray(value).tobytes(order="C") == expected.tobytes(order="C")
    )


def _persisted_intensity_reference(arrays, query_index: int) -> float:
    references = np.asarray(arrays[query_array("intensity_reference")])
    if references.ndim != 1 or references.dtype != np.dtype(np.float64):
        raise ValueError("intensity reference must use one exact float64 scalar per query")
    reference = float(references[query_index])
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("intensity reference must be finite and positive")
    expected_digest = array_sha256(
        query_array("intensity_reference"),
        np.asarray(reference, dtype=np.float64),
    )
    if (
        str(arrays[query_array("intensity_reference_sha256")][query_index])
        != expected_digest
    ):
        raise ValueError("intensity reference SHA-256 does not reproduce")
    return reference


def _reconstruct_amplitude_embeddings(
    arrays,
    rows,
    topology_queries,
    topology_values,
    intensity_reference,
) -> dict[int, np.ndarray]:
    expected_by_topology = {}
    row_offset = 0
    for topology_query, audit in zip(topology_queries, topology_values):
        count = len(topology_query.feasible_wire_pattern_ids)
        topology_rows = rows[row_offset : row_offset + count]
        row_offset += count
        if not topology_rows.size:
            raise ValueError("selected topology has no materialized feasible branch")
        actual = np.asarray(
            arrays[branch_input("amplitude_bounds_embedding")][topology_rows[0]]
        )
        if actual.shape != (21,) or actual.dtype != np.dtype(np.float32):
            raise ValueError("amplitude bounds embedding has the wrong byte contract")
        if any(
            not _same_array_bytes(
                arrays[branch_input("amplitude_bounds_embedding")][row], actual
            )
            for row in topology_rows
        ):
            raise ValueError("one topology repeats inconsistent amplitude bounds embeddings")
        expected = np.asarray(
            topology_query.amplitude.model_embedding(intensity_reference),
            dtype=np.float32,
        )
        if not _same_array_bytes(actual, expected):
            raise ValueError(
                "amplitude bounds embedding does not reproduce from persisted "
                "intensity reference and canonical query"
            )
        expected_by_topology[topology_query.topology_id] = expected
        if audit.get("global_row_slice") != [
            int(topology_rows[0] - rows[0]),
            int(topology_rows[-1] - rows[0] + 1),
        ]:
            raise ValueError("topology branch slice disagrees with reconstructed query")
    return expected_by_topology


def _reconstruct_branch_catalog(
    arrays,
    rows,
    topology_queries,
    topology_values,
    intensity_reference,
):
    amplitude_embeddings = _reconstruct_amplitude_embeddings(
        arrays,
        rows,
        topology_queries,
        topology_values,
        intensity_reference,
    )
    reconstructed = []
    global_index = 0
    for topology_batch_index, (topology_query, audit) in enumerate(
        zip(topology_queries, topology_values)
    ):
        topology_branches = []
        amplitude_polytopes = []
        for branch_batch_index, pattern_id in enumerate(
            topology_query.feasible_wire_pattern_ids
        ):
            condition = branch_condition(topology_query.geometry, pattern_id)
            constraint = topology_query.amplitude.constraint_for_branch(
                resolution_present=condition.resolution_present
            )
            constraint_payload = constraint.to_audit_dict()
            constraint_sha = amplitude_constraint_sha256(constraint)
            key = V5GlobalBranchKey(topology_query.topology_id, pattern_id)
            candidate_sha = _digest_text(
                V5_USER_QUERY_DIGEST_VERSION,
                topology_query.geometry.sha256,
                topology_query.amplitude.sha256,
            )
            context_sha = branch_context_sha256(key, candidate_sha, constraint_sha)
            branch_audit = {
                "global_index": global_index,
                "topology_batch_index": topology_batch_index,
                "branch_batch_index": branch_batch_index,
                "global_branch_key": key.wire_key,
                "topology_query_sha256": candidate_sha,
                "context_sha256": context_sha,
                "amplitude_constraint_sha256": constraint_sha,
            }
            expected_inputs = {
                "branch_topology_id": np.asarray([key.topology_id], dtype=np.int32),
                "branch_pattern_id": np.asarray([key.pattern_id], dtype=np.int32),
                "geometry_bounds_embedding": np.asarray(
                    condition.bounds_embedding, dtype=np.float32
                ),
                "amplitude_bounds_embedding": amplitude_embeddings[key.topology_id],
                "available_dimension_mask": np.asarray(
                    condition.available_dimension_mask, dtype=np.float32
                ),
                "active_dimension_mask": np.asarray(
                    condition.active_dimension_mask, dtype=np.float32
                ),
                "varying_dimension_mask": np.asarray(
                    condition.varying_dimension_mask, dtype=np.float32
                ),
            }
            reconstructed.append(
                {
                    "key": key,
                    "topology_query": topology_query,
                    "candidate_query_sha256": candidate_sha,
                    "constraint_json": canonical_json(constraint_payload),
                    "constraint_sha256": constraint_sha,
                    "context_sha256": context_sha,
                    "audit": branch_audit,
                    "inputs": expected_inputs,
                }
            )
            topology_branches.append(key.wire_key)
            amplitude_polytopes.append(
                {
                    "global_branch_key": key.wire_key,
                    "sha256": constraint_sha,
                    "constraint": constraint_payload,
                }
            )
            global_index += 1
        expected_topology_audit = {
            **topology_query.audit_payload(),
            "topology_query_sha256": topology_query.sha256,
            "candidate_query_sha256": reconstructed[-1]["candidate_query_sha256"],
            "candidate_batch_audit_sha256": audit.get("candidate_batch_audit_sha256"),
            "global_row_slice": audit.get("global_row_slice"),
            "global_branch_keys": topology_branches,
            "all_feasible_wire_branches_enumerated": True,
            "amplitude_polytopes": amplitude_polytopes,
        }
        if audit != expected_topology_audit:
            raise ValueError("topology universal context does not reproduce canonical queries")
    if len(reconstructed) != len(rows):
        raise ValueError("reconstructed feasible branch catalog has the wrong size")
    return tuple(reconstructed)


def _validate_branch_identity(
    arrays, row, query_index, query_sha, expected, reconstructed, seen
):
    global_key = str(arrays[branch_array("global_branch_key")][row])
    key = V5GlobalBranchKey(
        int(arrays[branch_array("topology_id")][row]),
        int(arrays[branch_array("pattern_id")][row]),
    )
    context_hash = str(arrays[branch_array("context_sha256")][row])
    identity = (
        str(arrays[query_array("clean_group_id")][query_index]),
        query_sha,
        global_key,
        context_hash,
    )
    if identity in seen:
        raise ValueError("duplicate clean/query/branch/context row")
    seen.add(identity)
    if (
        global_key != key.wire_key
        or expected != reconstructed["audit"]
        or key != reconstructed["key"]
    ):
        raise ValueError("global branch identity disagrees with universal audit")
    topology_query = reconstructed["topology_query"]
    candidate_query = str(arrays[branch_array("candidate_query_sha256")][row])
    amplitude_hash = str(arrays[branch_array("amplitude_constraint_sha256")][row])
    amplitude_json = str(arrays[branch_array("amplitude_constraint_json")][row])
    if (
        amplitude_json != reconstructed["constraint_json"]
        or amplitude_hash != reconstructed["constraint_sha256"]
    ):
        raise ValueError("amplitude constraint JSON/SHA-256 does not reproduce canonical query")
    if (
        reconstructed["candidate_query_sha256"] != candidate_query
        or reconstructed["context_sha256"] != context_hash
        or topology_query.sha256
        != arrays[branch_array("declared_topology_query_sha256")][row]
        or topology_query.geometry.sha256
        != arrays[branch_array("geometry_query_sha256")][row]
        or topology_query.amplitude.sha256
        != arrays[branch_array("amplitude_query_sha256")][row]
        or branch_context_sha256(key, candidate_query, amplitude_hash) != context_hash
    ):
        raise ValueError("query/branch/context hashes do not reproduce")
    for name, expected_input in reconstructed["inputs"].items():
        if not _same_array_bytes(arrays[branch_input(name)][row], expected_input):
            raise ValueError(f"branch model input {name} does not reproduce canonical query")
    for mask_name in ("active_dimension_mask", "varying_dimension_mask"):
        expected_mask = reconstructed["inputs"][mask_name]
        if not _same_array_bytes(arrays[branch_label(mask_name)][row], expected_mask):
            raise ValueError("branch supervision masks disagree with canonical query")
    return {
        "global_branch_key": global_key,
        "topology_query_sha256": candidate_query,
        "context_sha256": context_hash,
        "amplitude_constraint_sha256": amplitude_hash,
    }


def _validate_search_row(arrays, row, query_index, protocol) -> None:
    encoded = str(arrays[branch_array("search_record_json")][row])
    record = _strict_json(encoded, "search_record_json")
    record_sha = sha256(encoded.encode("utf-8")).hexdigest()
    outcome_code = int(arrays[branch_label("search_outcome_code")][row])
    outcome_names = {value: name for name, value in SEARCH_OUTCOME_CODE.items()}
    completed = bool(arrays[branch_array("runner_completed")][row])
    used = int(arrays[branch_array("exact_forward_calls_used")][row])
    representatives_payload = _strict_json(
        canonical_json(
            {
                "values": json.loads(
                    str(arrays[branch_array("compatible_representatives_json")][row])
                )
            }
        ),
        "compatible representatives",
    )["values"]
    references = tuple(
        V5CompatibleRepresentativeReference(**value) for value in representatives_payload
    )
    threshold_binding = _strict_json(
        str(arrays[query_array("selected_threshold_binding_json")][query_index]),
        "selected threshold binding",
    )
    task_payload = {
        "query_index": query_index,
        "clean_group_id": str(arrays[query_array("clean_group_id")][query_index]),
        "recipe_id": str(arrays[query_array("recipe_id")][query_index]),
        "observation_id": str(arrays[query_array("observation_id")][query_index]),
        "exact_observation_audit_sha256": str(
            arrays[query_array("exact_observation_audit_sha256")][query_index]
        ),
        "query_catalog_artifact_sha256": str(
            arrays[query_array("query_catalog_artifact_sha256")][query_index]
        ),
        "universal_query_sha256": str(
            arrays[query_array("universal_query_sha256")][query_index]
        ),
        "global_branch_key": str(arrays[branch_array("global_branch_key")][row]),
        "context_sha256": str(arrays[branch_array("context_sha256")][row]),
        "protocol_sha256": protocol.sha256,
        "calibrated_threshold_sha256": threshold_binding[
            "calibrated_threshold_sha256"
        ],
    }
    task_audit_sha = sha256(
        canonical_json(task_payload).encode("utf-8")
    ).hexdigest()
    if arrays[branch_array("task_audit_sha256")][row] != task_audit_sha:
        raise ValueError(
            "sidecar branch task/protocol audit SHA-256 does not reproduce"
        )
    expected_artifact_id = "v5-search-" + _digest_text(
        str(arrays[query_array("clean_group_id")][query_index]),
        str(arrays[query_array("observation_id")][query_index]),
        str(arrays[query_array("exact_curve_sha256")][query_index]),
        str(arrays[query_array("universal_query_sha256")][query_index]),
        str(arrays[query_array("query_catalog_artifact_sha256")][query_index]),
        str(arrays[branch_array("global_branch_key")][row]),
        str(arrays[branch_array("context_sha256")][row]),
        protocol.sha256,
        task_audit_sha,
    )
    expected_record = {
        "schema": V5_SEARCH_RECORD_SCHEMA,
        "artifact_id": expected_artifact_id,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "protocol_tier": protocol.protocol_tier,
        "task_audit_sha256": task_audit_sha,
        "calibrated_threshold_sha256": threshold_binding[
            "calibrated_threshold_sha256"
        ],
        "full_training_label_permitted": bool(
            arrays[query_array("full_training_label_permitted")][query_index]
        ),
        "clean_group_id": str(arrays[query_array("clean_group_id")][query_index]),
        "recipe_id": str(arrays[query_array("recipe_id")][query_index]),
        "observation_id": str(arrays[query_array("observation_id")][query_index]),
        "exact_curve_sha256": str(
            arrays[query_array("exact_curve_sha256")][query_index]
        ),
        "exact_observation_audit_sha256": str(
            arrays[query_array("exact_observation_audit_sha256")][query_index]
        ),
        "acceptance_sigma_log_available": bool(
            arrays[query_array("acceptance_sigma_log_available")][query_index]
        ),
        "acceptance_sigma_source_id": str(
            arrays[query_array("acceptance_sigma_source_id")][query_index]
        ),
        "exact_metric_name": str(
            arrays[query_array("selected_metric_name")][query_index]
        ),
        "exact_threshold_name": str(
            arrays[query_array("selected_threshold_name")][query_index]
        ),
        "exact_threshold_value": float(
            arrays[query_array("selected_threshold_value")][query_index]
        ),
        "exact_threshold_source_id": str(
            arrays[query_array("selected_threshold_source_id")][query_index]
        ),
        "query_catalog_artifact_id": str(
            arrays[query_array("query_catalog_artifact_id")][query_index]
        ),
        "query_catalog_artifact_sha256": str(
            arrays[query_array("query_catalog_artifact_sha256")][query_index]
        ),
        "universal_query_sha256": str(
            arrays[query_array("universal_query_sha256")][query_index]
        ),
        "global_branch_key": str(arrays[branch_array("global_branch_key")][row]),
        "context_sha256": str(arrays[branch_array("context_sha256")][row]),
        "outcome": outcome_names.get(outcome_code),
        "completed": completed,
        "exact_forward_call_budget": protocol.exact_forward_call_budget,
        "exact_forward_calls_used": used,
        "termination_reason": str(arrays[branch_array("runner_termination_reason")][row]),
        "executor_artifact_id": (
            str(arrays[branch_array("executor_artifact_id")][row]) or None
        ),
        "executor_artifact_sha256": (
            str(arrays[branch_array("executor_artifact_sha256")][row]) or None
        ),
        "compatible_representatives": [value.audit_payload() for value in references],
        "negative_is_no_solution_certificate": False,
    }
    if record != expected_record or record_sha != arrays[branch_array("search_record_sha256")][row]:
        raise ValueError("branch search record identity does not reproduce")
    if used > protocol.exact_forward_call_budget or int(
        arrays[branch_array("frozen_exact_forward_call_budget")][row]
    ) != protocol.exact_forward_call_budget:
        raise ValueError("branch search record budget/result fields disagree")
    _validate_label_evidence(
        arrays,
        row,
        query_index,
        protocol,
        outcome_code,
        completed,
        used,
        references,
        expected_artifact_id,
        record_sha,
    )
    _validate_supervision_audit(
        arrays,
        row,
        query_index,
        outcome_names.get(outcome_code),
    )


def _validate_supervision_audit(arrays, row, query_index, outcome) -> None:
    encoded = str(arrays[branch_array("supervision_audit_json")][row])
    audit = _strict_json(encoded, "candidate supervision audit")
    supplied_sha = audit.get("audit_sha256")
    payload_without_sha = dict(audit)
    payload_without_sha.pop("audit_sha256", None)
    expected_sha = sha256(canonical_json(payload_without_sha).encode("utf-8")).hexdigest()
    completed = bool(arrays[branch_label("search_completed")][row])
    has_exact = bool(arrays[branch_label("exact_artifact_id")][row])
    has_target = bool(arrays[branch_label("has_local_target")][row])
    search = None
    if completed:
        search = {
            "search_artifact_id": str(
                arrays[branch_label("search_artifact_id")][row]
            ),
            "search_artifact_sha256": str(
                arrays[branch_label("search_artifact_sha256")][row]
            ),
            "protocol_id": str(arrays[branch_label("search_protocol_id")][row]),
            "protocol_sha256": str(
                arrays[branch_label("search_protocol_sha256")][row]
            ),
            "evaluator_version": str(
                arrays[branch_label("search_evaluator_version")][row]
            ),
            "metric_name": str(arrays[branch_label("search_metric_name")][row]),
            "threshold_name": str(
                arrays[branch_label("search_threshold_name")][row]
            ),
            "threshold_value": float(
                arrays[branch_label("search_threshold_value")][row]
            ),
            "threshold_source_id": str(
                arrays[branch_label("search_threshold_source_id")][row]
            ),
            "exact_forward_call_budget": int(
                arrays[branch_label("search_exact_forward_call_budget")][row]
            ),
            "exact_forward_calls_used": int(
                arrays[branch_label("search_exact_forward_calls_used")][row]
            ),
            "termination_reason": str(
                arrays[branch_label("search_termination_reason")][row]
            ),
            "completed": True,
            "compatible_representative_count": int(
                arrays[
                    branch_label("search_compatible_representative_count")
                ][row]
            ),
        }
        audited_search = audit.get("search_provenance")
        if not isinstance(audited_search, Mapping) or not np.isclose(
            audited_search.get("threshold_value", np.nan),
            search["threshold_value"],
        ):
            raise ValueError("candidate supervision audit threshold disagrees")
        search["threshold_value"] = audited_search["threshold_value"]
    exact = None
    if has_exact:
        exact = {
            "artifact_id": str(arrays[branch_label("exact_artifact_id")][row]),
            "artifact_sha256": str(
                arrays[branch_label("exact_artifact_sha256")][row]
            ),
            "metric_value": float(
                arrays[branch_label("exact_metric_value")][row]
            ),
            "bounds_passed": bool(
                arrays[branch_label("exact_bounds_passed")][row]
            ),
            "physics_passed": bool(
                arrays[branch_label("exact_physics_passed")][row]
            ),
        }
        audited_exact = audit.get("exact_compatible")
        if not isinstance(audited_exact, Mapping) or not np.isclose(
            audited_exact.get("metric_value", np.nan),
            exact["metric_value"],
        ):
            raise ValueError("candidate supervision audit metric disagrees")
        exact["metric_value"] = audited_exact["metric_value"]
    target = None
    if has_target:
        audited_target = audit.get("target_local")
        if not isinstance(audited_target, list) or not np.allclose(
            arrays[branch_label("target_local")][row],
            audited_target,
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError("candidate supervision audit local target disagrees")
        target = audited_target
    expected = {
        "schema_version": CANDIDATE_SUPERVISION_V5_SCHEMA,
        "version": CANDIDATE_SUPERVISION_V5_VERSION,
        "clean_recipe_id": str(
            arrays[query_array("recipe_id")][query_index]
        ),
        "candidate_id": (
            f"{arrays[query_array('observation_id')][query_index]}:"
            f"{arrays[branch_array('global_branch_key')][row]}:"
            f"context-{arrays[branch_array('context_sha256')][row]}"
        ),
        "outcome": outcome,
        "outcome_code": int(arrays[branch_label("search_outcome_code")][row]),
        "search_provenance": search,
        "exact_compatible": exact,
        "target_local": target,
        "bce_eligible": outcome != "unverified",
        "local_mdn_eligible": outcome == "compatible_found" and has_target,
        "generating_candidate_match": None,
        "generating_mismatch_is_automatic_negative": False,
        "audit_sha256": expected_sha,
    }
    if audit != expected or supplied_sha != expected_sha:
        raise ValueError("candidate supervision audit does not reproduce")


def _validate_label_evidence(
    arrays,
    row,
    query_index,
    protocol,
    outcome_code,
    completed,
    used,
    references,
    expected_artifact_id,
    record_sha,
) -> None:
    unverified = outcome_code == SEARCH_OUTCOME_CODE["unverified"]
    if unverified:
        empty_strings = (
            "exact_artifact_id",
            "exact_artifact_sha256",
            "search_artifact_id",
            "search_artifact_sha256",
            "search_protocol_id",
            "search_protocol_sha256",
            "search_evaluator_version",
            "search_metric_name",
            "search_threshold_name",
            "search_threshold_source_id",
            "search_termination_reason",
        )
        if (
            completed
            or references
            or arrays[branch_label("search_completed")][row]
            or arrays[branch_label("has_local_target")][row]
            or any(arrays[branch_label(name)][row] != "" for name in empty_strings)
            or float(arrays[branch_label("exact_metric_value")][row]) != 0.0
            or arrays[branch_label("exact_bounds_passed")][row]
            or arrays[branch_label("exact_physics_passed")][row]
            or float(arrays[branch_label("search_threshold_value")][row]) != 0.0
            or int(
                arrays[branch_label("search_exact_forward_call_budget")][row]
            )
            != 0
            or int(arrays[branch_label("search_exact_forward_calls_used")][row])
            != 0
            or int(
                arrays[
                    branch_label("search_compatible_representative_count")
                ][row]
            )
            != 0
        ):
            raise ValueError("unverified branch became completed supervision")
        return
    if not completed or not arrays[branch_label("search_completed")][row]:
        raise ValueError("verified branch lacks completed frozen-search evidence")
    if used != protocol.exact_forward_call_budget:
        raise ValueError(
            "verified branch did not consume the equal frozen per-branch budget"
        )
    if (
        not str(arrays[branch_array("executor_artifact_id")][row])
        or _SHA256.fullmatch(
            str(arrays[branch_array("executor_artifact_sha256")][row])
        )
        is None
    ):
        raise ValueError("verified branch lacks immutable executor evidence")
    expected = {
        "search_protocol_sha256": protocol.sha256,
        "search_protocol_id": protocol.protocol_id,
        "search_evaluator_version": protocol.evaluator_version,
        "search_metric_name": arrays[query_array("selected_metric_name")][query_index],
        "search_threshold_name": arrays[
            query_array("selected_threshold_name")
        ][query_index],
        "search_threshold_source_id": arrays[
            query_array("selected_threshold_source_id")
        ][query_index],
    }
    if any(arrays[branch_label(name)][row] != value for name, value in expected.items()):
        raise ValueError("verified supervision disagrees with frozen protocol")
    if (
        int(arrays[branch_label("search_exact_forward_call_budget")][row])
        != protocol.exact_forward_call_budget
        or int(arrays[branch_label("search_exact_forward_calls_used")][row]) != used
        or arrays[branch_label("search_artifact_sha256")][row]
        != record_sha
        or arrays[branch_label("search_artifact_id")][row]
        != expected_artifact_id
        or arrays[branch_label("search_termination_reason")][row]
        != arrays[branch_array("runner_termination_reason")][row]
        or not np.isclose(
            arrays[branch_label("search_threshold_value")][row],
            arrays[query_array("selected_threshold_value")][query_index],
            rtol=0.0,
            atol=np.finfo(np.float32).eps,
        )
    ):
        raise ValueError("verified supervision budget/threshold disagrees with protocol")
    positive = outcome_code == SEARCH_OUTCOME_CODE["compatible_found"]
    if positive:
        if (
            str(arrays[branch_label("search_termination_reason")][row])
            != POSITIVE_TERMINATION_REASON
        ):
            raise ValueError("positive search row has a noncanonical termination reason")
        if not references or int(
            arrays[branch_label("search_compatible_representative_count")][row]
        ) != len(references):
            raise ValueError("positive search row has no complete representative references")
        threshold = float(
            arrays[query_array("selected_threshold_value")][query_index]
        )
        if any(
            value.metric_value > threshold
            or not value.bounds_passed
            or not value.physics_passed
            for value in references
        ):
            raise ValueError("positive representative set contains an incompatible member")
        fixed = np.logical_not(
            np.asarray(
                arrays[branch_label("varying_dimension_mask")][row],
                dtype=np.bool_,
            )
        )
        if any(
            reference.target_local is not None
            and np.any(np.asarray(reference.target_local)[fixed] != 0.5)
            for reference in references
        ):
            raise ValueError("positive representative target moves a fixed local axis")
        first = references[0]
        if (
            arrays[branch_label("exact_artifact_id")][row] != first.artifact_id
            or arrays[branch_label("exact_artifact_sha256")][row] != first.artifact_sha256
            or bool(arrays[branch_label("exact_bounds_passed")][row])
            != first.bounds_passed
            or bool(arrays[branch_label("exact_physics_passed")][row])
            != first.physics_passed
            or not np.isclose(
                arrays[branch_label("exact_metric_value")][row], first.metric_value
            )
        ):
            raise ValueError("positive label disagrees with its exact representative")
        if first.target_local is None:
            if arrays[branch_label("has_local_target")][row]:
                raise ValueError("positive label invented a local target")
        elif not np.allclose(
            arrays[branch_label("target_local")][row], first.target_local, rtol=0.0, atol=1e-7
        ):
            raise ValueError("positive local target disagrees with representative")
        return
    negative = SEARCH_OUTCOME_CODE["no_compatible_found_within_frozen_search_budget"]
    if outcome_code != negative:
        raise ValueError("search-sidecar outcome code is unsupported")
    reason = str(arrays[branch_label("search_termination_reason")][row])
    if (
        references
        or int(arrays[branch_label("search_compatible_representative_count")][row])
        or arrays[branch_label("has_local_target")][row]
        or arrays[branch_label("exact_artifact_id")][row] != ""
        or arrays[branch_label("exact_artifact_sha256")][row] != ""
        or float(arrays[branch_label("exact_metric_value")][row]) != 0.0
        or arrays[branch_label("exact_bounds_passed")][row]
        or arrays[branch_label("exact_physics_passed")][row]
        or reason not in NEGATIVE_TERMINATION_REASONS
        or (
            reason == "exact_forward_budget_exhausted_without_compatible"
            and used != protocol.exact_forward_call_budget
        )
    ):
        raise ValueError("completed negative is not a complete frozen-search negative")


def _validate_counts(counts, arrays, protocol, branch_count) -> None:
    outcomes = arrays[branch_label("search_outcome_code")]
    expected = {
        "compatible_found": int(
            np.count_nonzero(outcomes == SEARCH_OUTCOME_CODE["compatible_found"])
        ),
        "completed_negative": int(
            np.count_nonzero(
                outcomes
                == SEARCH_OUTCOME_CODE[
                    "no_compatible_found_within_frozen_search_budget"
                ]
            )
        ),
        "unverified": int(np.count_nonzero(outcomes == SEARCH_OUTCOME_CODE["unverified"])),
    }
    if any(int(counts[name]) != value for name, value in expected.items()):
        raise ValueError("search-sidecar outcome counts do not reproduce")
    if int(counts["exact_forward_call_budget_total"]) != (
        branch_count * protocol.exact_forward_call_budget
    ):
        raise ValueError("search-sidecar total exact-forward budget does not reproduce")
    if int(counts["exact_forward_calls_used_total"]) != int(
        np.sum(arrays[branch_array("exact_forward_calls_used")])
    ):
        raise ValueError("search-sidecar exact-forward call count does not reproduce")


__all__ = ["validate_v5_search_supervision_sidecar"]
