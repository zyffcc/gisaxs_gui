"""Cross-table scientific provenance checks for V5.1 grouped artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Mapping

import numpy as np

from .branch_catalog import decode_branch_pattern
from .grouped_amplitude_join_v5 import amplitude_query_from_json
from .grouped_artifact_v5 import array_sha256, canonical_json


def _object(encoded: str, name: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate {name} field {key!r}")
            result[key] = value
        return result

    try:
        result = json.loads(encoded, object_pairs_hook=reject_duplicates)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict JSON") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{name} must contain one object")
    return result


def _hashed_json(encoded: str, digest: str, name: str) -> dict[str, object]:
    result = _object(encoded, name)
    if sha256(encoded.encode("utf-8")).hexdigest() != digest:
        raise ValueError(f"{name} JSON/hash does not reproduce")
    return result


def _protocol(manifest: Mapping[str, object]) -> tuple[dict[str, object], str]:
    protocol = manifest.get("oracle_protocol")
    if not isinstance(protocol, Mapping) or set(protocol) != {"payload", "sha256"}:
        raise ValueError("oracle protocol manifest is incomplete")
    payload, digest = protocol["payload"], protocol["sha256"]
    if not isinstance(payload, Mapping) or not isinstance(digest, str):
        raise ValueError("oracle protocol manifest is invalid")
    if sha256(canonical_json(payload).encode("utf-8")).hexdigest() != digest:
        raise ValueError("oracle protocol SHA-256 does not reproduce")
    if (
        payload.get("exact_forward_call_budget") != 1
        or payload.get("negative_labels_produced") is not False
        or payload.get("candidate_source") != "frozen_generating_parameter_vector"
    ):
        raise ValueError("oracle protocol is not the frozen one-call solution-stage protocol")
    return dict(payload), digest


def _clean_recipe_identity(payload: Mapping[str, object]) -> tuple[object, object]:
    """Read the common identity from seeded or direct-Sobol canonical JSON."""

    if "schema_version" in payload or "generator_version" in payload:
        return payload.get("schema_version"), payload.get("generator_version")
    return payload.get("schema"), payload.get("version")


def _amplitude_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
    """Select the generating composition from either supported recipe envelope."""

    amplitude = payload.get("amplitude_composition")
    if amplitude is None:
        physics = payload.get("physics")
        if isinstance(physics, Mapping):
            amplitude = physics.get("amplitude_composition")
    if not isinstance(amplitude, Mapping):
        raise ValueError("clean recipe has no generating amplitude composition")
    return amplitude


def validate_grouped_provenance(
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, object],
    *,
    positive: np.ndarray,
) -> None:
    """Bind candidates, exact proofs, amplitude polytopes, and clean parents."""

    protocol, protocol_hash = _protocol(manifest)
    clean_ids = arrays["clean__recipe_id"]
    clean_recipe_hashes = arrays["clean__recipe_sha256"]
    clean_recipes = tuple(
        _hashed_json(str(encoded), str(digest), "clean recipe")
        for encoded, digest in zip(arrays["clean__recipe_canonical_json"], clean_recipe_hashes)
    )
    identity = manifest["clean_recipe_identity"]
    expected_identity = (identity["schema_version"], identity["generator_version"])
    if any(_clean_recipe_identity(value) != expected_identity for value in clean_recipes):
        raise ValueError("clean recipe JSON escaped its shard schema/generator identity")
    amplitude_queries = tuple(
        amplitude_query_from_json(str(encoded), str(digest))
        for encoded, digest in zip(
            arrays["clean__amplitude_query_canonical_json"],
            arrays["clean__amplitude_query_sha256"],
        )
    )
    candidate_recipe = arrays["candidate_context__recipe_index"]
    candidate_ids = arrays["candidate_context__candidate_id"]
    patterns = arrays["candidate_context__input__branch_pattern_id"].reshape(-1)
    outcomes = arrays["candidate_label__search_outcome_code"]
    exact_json = arrays["candidate_label__oracle_exact_artifact_json"]
    search_json = arrays["candidate_label__oracle_search_artifact_json"]
    supervision_json = arrays["candidate_label__supervision_audit_json"]

    for index, recipe_index in enumerate(candidate_recipe):
        recipe_index = int(recipe_index)
        clean = clean_recipes[recipe_index]
        candidate_id = str(candidate_ids[index])
        if (
            arrays["candidate_context__geometry_query_sha256"][index]
            != arrays["clean__geometry_query_sha256"][recipe_index]
        ):
            raise ValueError("candidate geometry query escaped its clean parent")
        if (
            arrays["candidate_context__amplitude_query_sha256"][index]
            != arrays["clean__amplitude_query_sha256"][recipe_index]
        ):
            raise ValueError("candidate amplitude query escaped its clean parent")

        _, resolution_present = decode_branch_pattern(int(patterns[index]))
        constraint = amplitude_queries[recipe_index].constraint_for_branch(
            resolution_present=resolution_present
        )
        encoded_constraint = str(arrays["candidate_context__amplitude_constraint_json"][index])
        supplied_constraint = _hashed_json(
            encoded_constraint,
            str(arrays["candidate_context__amplitude_constraint_sha256"][index]),
            "amplitude constraint",
        )
        if supplied_constraint != constraint.to_audit_dict():
            raise ValueError("candidate amplitude constraint does not match its physical query")

        supervision = _object(str(supervision_json[index]), "candidate supervision")
        supervision_hash = supervision.pop("audit_sha256", None)
        if sha256(canonical_json(supervision).encode("utf-8")).hexdigest() != supervision_hash:
            raise ValueError("candidate supervision audit SHA-256 does not reproduce")
        if (
            supervision.get("clean_recipe_id") != clean_ids[recipe_index]
            or supervision.get("candidate_id") != candidate_id
            or supervision.get("outcome_code") != int(outcomes[index])
        ):
            raise ValueError("candidate supervision audit escaped its table row")

        if not positive[index]:
            if exact_json[index] != "" or search_json[index] != "":
                raise ValueError("unverified candidate cannot carry oracle evidence")
            continue
        exact = _hashed_json(
            str(exact_json[index]),
            str(arrays["candidate_label__exact_artifact_sha256"][index]),
            "oracle exact artifact",
        )
        search = _hashed_json(
            str(search_json[index]),
            str(arrays["candidate_label__search_artifact_sha256"][index]),
            "oracle search artifact",
        )
        target = arrays["clean__target_local"][recipe_index]
        expected = {
            "recipe_sha256": clean_recipe_hashes[recipe_index],
            "geometry_query_sha256": arrays["clean__geometry_query_sha256"][recipe_index],
            "amplitude_query_sha256": arrays["clean__amplitude_query_sha256"][recipe_index],
            "branch_pattern_id": int(patterns[index]),
            "target_local_sha256": array_sha256("oracle_target_local", target),
            "protocol_sha256": protocol_hash,
            "metric_name": protocol["metric_name"],
            "metric_value": 0.0,
            "exact_forward_calls_used": 1,
        }
        if any(exact.get(name) != value for name, value in expected.items()):
            raise ValueError("oracle exact proof escaped its known truth or protocol")
        if (
            search.get("candidate_id") != candidate_id
            or search.get("protocol_sha256") != protocol_hash
            or search.get("exact_artifact_sha256")
            != arrays["candidate_label__exact_artifact_sha256"][index]
            or search.get("exact_forward_calls_used") != 1
            or search.get("completed") is not True
        ):
            raise ValueError("oracle search proof escaped its candidate or one-call protocol")
        amplitude = _amplitude_payload(clean)
        coefficients = [
            amplitude["background"],
            *amplitude["particle_amplitudes"],
        ]
        if resolution_present:
            coefficients.append(amplitude["resolution_amplitude"])
        if not constraint.contains(coefficients, k=amplitude["k"], atol=2.0e-9):
            raise ValueError("generating amplitude truth escaped the candidate polytope")


__all__ = ["validate_grouped_provenance"]
