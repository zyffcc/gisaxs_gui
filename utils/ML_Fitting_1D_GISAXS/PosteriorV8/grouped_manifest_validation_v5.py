"""Fail-closed manifest semantics for V5.1 grouped candidate shards."""

from __future__ import annotations

from hashlib import sha256
import re
from typing import Mapping, Sequence

from .candidate_supervision_v5 import CANDIDATE_SUPERVISION_TENSOR_KEYS
from .clean_recipe_forward_v5 import V5_CLEAN_RECIPE_PROTOCOL_VERSION
from .grouped_artifact_v5 import (
    V5_CHECKED_ARRAY_ARTIFACT_SCHEMA,
    V5_CHECKED_ARRAY_ARTIFACT_VERSION,
    canonical_json,
)
from .grouped_known_truth_oracle_v5 import (
    V5_ORACLE_PROTOCOL_PAYLOAD,
    V5_ORACLE_PROTOCOL_SHA256,
)
from .model_v5_contract import MODEL_V5_INPUT_KEYS


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_FIELDS = {
    "container_schema",
    "container_version",
    "dataset_schema",
    "dataset_version",
    "dataset_id",
    "stage",
    "build_policy",
    "counts",
    "table_fields",
    "model_input_keys",
    "candidate_label_keys",
    "contract_bundle",
    "contract_bundle_sha256",
    "clean_recipe_identity",
    "source_sha256",
    "oracle_protocol",
    "shard_metadata",
    "split_semantics",
    "arrays",
    "manifest_sha256",
}
_SPLIT_SEMANTICS = {
    "statistical_unit": "independent_clean_physical_recipe",
    "inheritance": "all_observation_views_and_candidates_follow_clean_parent",
    "sobol_parent_fields_are_empty_together_or_complete_together": True,
}


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def validate_grouped_manifest_semantics(
    manifest: Mapping[str, object],
    *,
    dataset_schema: str,
    dataset_version: str,
    stage: str,
    clean_fields: Sequence[str],
    observation_fields: Sequence[str],
    candidate_context_fields: Sequence[str],
    candidate_label_extra_fields: Sequence[str],
    contract_bundle: Mapping[str, object],
) -> None:
    """Reject self-consistent manifests that alter any frozen scientific meaning."""

    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("grouped manifest fields are incomplete or unsupported")
    if (
        manifest.get("container_schema") != V5_CHECKED_ARRAY_ARTIFACT_SCHEMA
        or manifest.get("container_version") != V5_CHECKED_ARRAY_ARTIFACT_VERSION
    ):
        raise ValueError("unsupported grouped checked-array container identity")
    if manifest.get("dataset_schema") != dataset_schema:
        raise ValueError("unsupported V5 grouped-dataset schema")
    if manifest.get("dataset_version") != dataset_version:
        raise ValueError("unsupported V5 grouped-dataset version")
    if not isinstance(manifest.get("dataset_id"), str) or not manifest["dataset_id"].strip():
        raise ValueError("grouped dataset_id must be non-empty")
    if manifest.get("stage") != stage:
        raise ValueError("grouped dataset stage is incompatible")

    policy = manifest.get("build_policy")
    if not isinstance(policy, Mapping) or type(policy.get("generating_candidate_only")) is not bool:
        raise ValueError("grouped build policy is incomplete or unsupported")
    expected_policy = {
        "generating_candidate_only": policy["generating_candidate_only"],
        "non_generating_feasible_branch_outcome": "unverified",
        "unsearched_branch_is_negative": False,
        "curve_storage": "once_per_observation_view",
        "candidate_context_storage": "once_per_clean_recipe_branch",
    }
    if dict(policy) != expected_policy:
        raise ValueError("grouped build policy altered frozen solution-stage semantics")

    expected_tables = {
        "clean": list(clean_fields),
        "observation": list(observation_fields),
        "candidate_context": list(candidate_context_fields),
        "candidate_label": [
            *CANDIDATE_SUPERVISION_TENSOR_KEYS,
            *candidate_label_extra_fields,
        ],
    }
    if manifest.get("table_fields") != expected_tables:
        raise ValueError("grouped table-field inventory is incompatible")
    if tuple(manifest.get("model_input_keys", ())) != MODEL_V5_INPUT_KEYS:
        raise ValueError("grouped model-input keys are incompatible")
    if tuple(manifest.get("candidate_label_keys", ())) != CANDIDATE_SUPERVISION_TENSOR_KEYS:
        raise ValueError("grouped candidate-label keys are incompatible")

    expected_bundle = dict(contract_bundle)
    if manifest.get("contract_bundle") != expected_bundle:
        raise ValueError("grouped dataset uses incompatible V5.1 contracts")
    bundle_hash = sha256(canonical_json(expected_bundle).encode("utf-8")).hexdigest()
    if manifest.get("contract_bundle_sha256") != bundle_hash:
        raise ValueError("grouped contract-bundle SHA-256 does not reproduce")

    identity = manifest.get("clean_recipe_identity")
    if not isinstance(identity, Mapping) or set(identity) != {
        "protocol_version",
        "schema_version",
        "generator_version",
    }:
        raise ValueError("clean recipe identity is incomplete or unsupported")
    if identity["protocol_version"] != V5_CLEAN_RECIPE_PROTOCOL_VERSION:
        raise ValueError("grouped dataset uses an incompatible clean-recipe protocol")
    if any(
        not isinstance(identity[name], str) or not identity[name].strip()
        for name in ("schema_version", "generator_version")
    ):
        raise ValueError("clean recipe schema/generator identity must be non-empty")

    if manifest.get("oracle_protocol") != {
        "payload": V5_ORACLE_PROTOCOL_PAYLOAD,
        "sha256": V5_ORACLE_PROTOCOL_SHA256,
    }:
        raise ValueError("grouped oracle protocol altered frozen one-call semantics")
    if manifest.get("split_semantics") != _SPLIT_SEMANTICS:
        raise ValueError("grouped split inheritance semantics are incompatible")

    sources = manifest.get("source_sha256")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("source_sha256 must be a non-empty mapping")
    for name, digest in sources.items():
        if not isinstance(name, str) or not name:
            raise ValueError("source hash names must be non-empty")
        _digest(digest, f"source_sha256[{name}]")

    core = dict(manifest)
    supplied_hash = _digest(core.pop("manifest_sha256"), "manifest_sha256")
    if sha256(canonical_json(core).encode("utf-8")).hexdigest() != supplied_hash:
        raise ValueError("grouped manifest SHA-256 does not reproduce")


__all__ = ["validate_grouped_manifest_semantics"]
