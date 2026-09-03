"""Replay physical amplitude queries and derive view-relative V5.1 inputs."""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np

from .amplitude_query_v5 import V5AmplitudeQuery
from .contract import ClosedInterval


def amplitude_query_from_json(encoded: str, expected_sha256: str) -> V5AmplitudeQuery:
    """Strictly reconstruct one persisted observation-independent query."""

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate amplitude-query field {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(encoded, object_pairs_hook=reject_duplicates)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("amplitude query is not strict JSON") from exc
    expected_fields = {
        "schema",
        "version",
        "embedding_version",
        "gui_amplitude_constraint_version",
        "canonical_gauge",
        "numeric_policy_version",
        "numeric_policy_sha256",
        "background",
        "k",
        "component_intensities",
        "resolution_presence_policy",
        "int_res",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("amplitude-query fields are incomplete or unsupported")
    try:
        result = V5AmplitudeQuery.create(
            background=ClosedInterval(**payload["background"]),
            k=ClosedInterval(**payload["k"]),
            component_intensities=tuple(
                ClosedInterval(**value) for value in payload["component_intensities"]
            ),
            resolution_presence_policy=payload["resolution_presence_policy"],
            int_res=(None if payload["int_res"] is None else ClosedInterval(**payload["int_res"])),
            numeric_policy_version=payload["numeric_policy_version"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("persisted amplitude query is invalid") from exc
    if result.canonical_json != encoded or result.sha256 != expected_sha256:
        raise ValueError("amplitude query JSON/SHA-256 does not reproduce")
    return result


def joined_amplitude_embeddings(
    *,
    query_json: Sequence[str],
    query_sha256: Sequence[str],
    observation_intensity_reference: Sequence[float],
    observation_recipe_index: Sequence[int],
    candidate_recipe_index: Sequence[int],
    observation_indices: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    """Create 21D conditions only for requested observation/candidate pairs."""

    per_observation = observation_amplitude_embeddings(
        query_json=query_json,
        query_sha256=query_sha256,
        observation_intensity_reference=observation_intensity_reference,
        observation_recipe_index=observation_recipe_index,
    )
    observation_recipe = np.asarray(observation_recipe_index, dtype=np.int64)
    candidate_recipe = np.asarray(candidate_recipe_index, dtype=np.int64)
    observed = np.asarray(observation_indices, dtype=np.int64)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if observed.shape != candidates.shape or observed.ndim != 1:
        raise ValueError("joined observation/candidate indices must be aligned vectors")
    if np.any(observation_recipe[observed] != candidate_recipe[candidates]):
        raise ValueError("amplitude embedding join crossed clean recipe parents")
    result = np.ascontiguousarray(per_observation[observed])
    result.setflags(write=False)
    return result


def observation_amplitude_embeddings(
    *,
    query_json: Sequence[str],
    query_sha256: Sequence[str],
    observation_intensity_reference: Sequence[float],
    observation_recipe_index: Sequence[int],
) -> np.ndarray:
    """Derive one compact 21D model condition per observation view."""

    encoded_queries, query_digests = tuple(query_json), tuple(query_sha256)
    if not encoded_queries or len(encoded_queries) != len(query_digests):
        raise ValueError("amplitude query JSON and hashes must have equal non-zero length")
    recipes = tuple(
        amplitude_query_from_json(str(encoded), str(digest))
        for encoded, digest in zip(encoded_queries, query_digests)
    )
    references = np.asarray(observation_intensity_reference, dtype=np.float64)
    observation_recipe = np.asarray(observation_recipe_index, dtype=np.int64)
    if references.ndim != 1 or observation_recipe.shape != references.shape:
        raise ValueError("observation amplitude references and recipe indices must align")
    if (
        not np.all(np.isfinite(references))
        or np.any(references <= 0.0)
        or np.any(observation_recipe < 0)
        or np.any(observation_recipe >= len(recipes))
    ):
        raise ValueError("observation amplitude references or recipe indices are invalid")
    result = np.asarray(
        [
            recipes[recipe_index].model_embedding(reference)
            for recipe_index, reference in zip(observation_recipe, references)
        ],
        dtype=np.float32,
    )
    result.setflags(write=False)
    return result


__all__ = [
    "amplitude_query_from_json",
    "joined_amplitude_embeddings",
    "observation_amplitude_embeddings",
]
