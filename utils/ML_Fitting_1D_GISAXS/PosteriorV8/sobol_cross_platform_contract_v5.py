"""Frozen identities and hash layers for the V5.2 cross-platform Sobol gate."""

from __future__ import annotations

from hashlib import sha256
import json
from numbers import Integral
import re
from typing import Mapping, Sequence

import numpy as np

from .contract import NUM_TOPOLOGIES
from .sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_NAMES,
    V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
    V5_SOBOL_RECIPE_COORDINATE_VERSION,
    V5_SOBOL_RECIPE_DIM,
)


V5_SOBOL_CROSS_PLATFORM_MANIFEST_SCHEMA = (
    "gisaxs.posterior_v8.sobol_cross_platform_scientific_manifest/v2"
)
V5_SOBOL_CROSS_PLATFORM_MANIFEST_VERSION = (
    "posterior_v8_v5_2_1024_recipe_10x34_query_and_model_embedding_attestation_v2"
)
V5_SOBOL_SEMANTIC_DESIGN_SCHEMA = (
    "gisaxs.posterior_v8.runtime_free_sobol_semantic_design/v1"
)
V5_SOBOL_SEMANTIC_DESIGN_VERSION = (
    "posterior_v8_scrambled_sobol_168d_bits52_seed20260903_semantics_v1"
)
V5_SOBOL_CROSS_PLATFORM_PREFIX_START = 0
V5_SOBOL_CROSS_PLATFORM_PREFIX_COUNT = 1024
V5_SOBOL_CROSS_PLATFORM_PREFIX_STOP = 1024
V5_SOBOL_CROSS_PLATFORM_SCRAMBLE_SEED = 20260903
V5_SOBOL_CROSS_PLATFORM_BITS = 52
V5_SOBOL_CROSS_PLATFORM_PROBE_INDICES = (
    0,
    1,
    17,
    118,
    233,
    511,
    589,
    700,
    753,
    1023,
)
V5_SOBOL_CROSS_PLATFORM_TOPOLOGY_IDS = tuple(range(NUM_TOPOLOGIES))
V5_SOBOL_CROSS_PLATFORM_CLEAN_GROUP_POLICY = (
    "sha256(runtime_free_semantic_design_sha256,NUL,decimal_sobol_index);"
    "neutral_train_attestation_parent_not_a_dataset_split_assignment_v1"
)

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
TOP_LEVEL_FIELDS = {
    "schema",
    "version",
    "source",
    "design",
    "scope",
    "validation_contract",
    "generating_entries",
    "topology_probe_entries",
    "layer_sha256",
}
SOURCE_FIELDS = {
    "source_archive_sha256",
    "source_manifest_sha256",
    "source_tree_sha256",
}
DESIGN_FIELDS = {"semantic_payload", "semantic_sha256"}
SCOPE_FIELDS = {
    "prefix_start",
    "prefix_stop",
    "prefix_count",
    "probe_indices",
    "selected_topology_ids",
    "probe_topology_query_count",
    "assigned_split",
    "ood_label",
    "clean_group_policy",
}
GENERATING_ENTRY_FIELDS = {
    "sobol_index",
    "clean_group_id",
    "unit_coordinates_sha256",
    "generating_topology_id",
    "branch_pattern_id",
    "geometry_query_sha256",
    "amplitude_query_sha256",
    "geometry_bounds_embedding_sha256",
    "amplitude_embedding_ref1_sha256",
    "amplitude_embedding_ref1e4_sha256",
    "amplitude_range_regimes_sha256",
    "target_sha256",
    "amplitude_composition_sha256",
    "normalized_clean_recipe_sha256",
}
TOPOLOGY_ENTRY_FIELDS = {
    "topology_id",
    "topology_query_sha256",
    "geometry_query_sha256",
    "amplitude_query_sha256",
    "geometry_bounds_embedding_sha256",
    "amplitude_embedding_ref1_sha256",
    "amplitude_embedding_ref1e4_sha256",
}
PROBE_ENTRY_FIELDS = {
    "sobol_index",
    "unit_coordinates_sha256",
    "generating_topology_id",
    "normalized_query_design_sha256",
    "topology_queries",
    "topology_queries_sha256",
}
LAYER_FIELDS = {
    "coordinate_prefix_sha256",
    "generating_geometry_queries_sha256",
    "generating_amplitude_queries_sha256",
    "generating_geometry_bounds_embeddings_sha256",
    "generating_amplitude_embeddings_ref1_sha256",
    "generating_amplitude_embeddings_ref1e4_sha256",
    "generating_amplitude_range_regimes_sha256",
    "generating_targets_sha256",
    "generating_amplitude_compositions_sha256",
    "normalized_clean_recipes_sha256",
    "generating_entries_sha256",
    "topology_probe_queries_sha256",
    "topology_probe_model_embeddings_sha256",
    "topology_probe_entries_sha256",
    "scientific_content_sha256",
}
VALIDATION_CONTRACT = {
    "coordinate_binary_encoding": "little_endian_float64_C_order",
    "coordinate_engine_replay": (
        "random_base2_10_byte_equals_uint52_gray_code_random_access_for_all_1024"
    ),
    "generating_recipe_replay": "exact_dataclass_and_canonical_json_sha256",
    "geometry_query_replay": "strict_canonical_json_and_sha256",
    "amplitude_query_replay": "strict_canonical_json_and_sha256",
    "geometry_bounds_embedding_replay": "exact_little_endian_float64_78d",
    "amplitude_model_embedding_replay": (
        "exact_little_endian_float64_21d_at_intensity_references_1_and_10000"
    ),
    "geometry_target_containment": "closed_interval_without_added_tolerance",
    "branch_codec_target_replay": (
        "exact_decode_to_stored_gui_truth_and_strict_physical_reencode_in_closed_"
        "unit_cube"
    ),
    "active_local_target_containment": (
        "epsilon_closed_interval_without_added_tolerance"
    ),
    "gui_amplitude_containment": "closed_interval_without_added_tolerance",
    "effective_amplitude_constraint": "shared_k_witness_with_atol_zero",
    "topology_query_replay": "strict_full_artifact_from_json",
    "runtime_metadata_in_scientific_identity": False,
    "runtime_contaminated_identity_fields_replaced": [
        "clean_recipe.source.sobol_design_sha256",
        "universal_query_design.source.sobol_design_sha256",
        "universal_query_design.source.design_point_sha256",
    ],
    "scientific_float_or_derived_query_hash_normalization": "forbidden",
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def pretty_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def sha256_json(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def exact_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def exact_keys(value: object, fields: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields are incomplete or unsupported")
    return value


def coordinate_sha256(values: Sequence[float]) -> str:
    coordinates = np.asarray(values, dtype="<f8")
    if coordinates.shape != (V5_SOBOL_RECIPE_DIM,):  # pragma: no cover
        raise RuntimeError("cross-platform coordinate row has the wrong shape")
    return sha256(coordinates.tobytes(order="C")).hexdigest()


def float64_vector_sha256(values: Sequence[float], *, expected_size: int) -> str:
    vector = np.asarray(values, dtype="<f8")
    if vector.shape != (expected_size,) or not np.all(np.isfinite(vector)):
        raise RuntimeError("cross-platform model embedding has invalid shape or values")
    return sha256(vector.tobytes(order="C")).hexdigest()


def v5_sobol_runtime_free_semantic_design() -> dict[str, object]:
    payload = {
        "schema": V5_SOBOL_SEMANTIC_DESIGN_SCHEMA,
        "version": V5_SOBOL_SEMANTIC_DESIGN_VERSION,
        "engine_semantics": "scrambled_Sobol_digital_net",
        "dimension": V5_SOBOL_RECIPE_DIM,
        "scramble": True,
        "bits": V5_SOBOL_CROSS_PLATFORM_BITS,
        "scramble_seed": V5_SOBOL_CROSS_PLATFORM_SCRAMBLE_SEED,
        "coordinate_schema": V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
        "coordinate_version": V5_SOBOL_RECIPE_COORDINATE_VERSION,
        "coordinate_contract_sha256": V5_SOBOL_RECIPE_COORDINATE_SHA256,
        "coordinate_names": list(V5_SOBOL_RECIPE_COORDINATE_NAMES),
        "runtime_fields_excluded": [
            "hostname",
            "operating_system",
            "python_version",
            "numpy_version",
            "scipy_version",
        ],
    }
    return {"semantic_payload": payload, "semantic_sha256": sha256_json(payload)}


def neutral_clean_group_id(semantic_design_sha256: str, sobol_index: int) -> str:
    return sha256(
        b"\0".join(
            (semantic_design_sha256.encode("ascii"), str(sobol_index).encode("ascii"))
        )
    ).hexdigest()


def scope_payload() -> dict[str, object]:
    return {
        "prefix_start": V5_SOBOL_CROSS_PLATFORM_PREFIX_START,
        "prefix_stop": V5_SOBOL_CROSS_PLATFORM_PREFIX_STOP,
        "prefix_count": V5_SOBOL_CROSS_PLATFORM_PREFIX_COUNT,
        "probe_indices": list(V5_SOBOL_CROSS_PLATFORM_PROBE_INDICES),
        "selected_topology_ids": list(V5_SOBOL_CROSS_PLATFORM_TOPOLOGY_IDS),
        "probe_topology_query_count": (
            len(V5_SOBOL_CROSS_PLATFORM_PROBE_INDICES)
            * len(V5_SOBOL_CROSS_PLATFORM_TOPOLOGY_IDS)
        ),
        "assigned_split": "train",
        "ood_label": None,
        "clean_group_policy": V5_SOBOL_CROSS_PLATFORM_CLEAN_GROUP_POLICY,
    }


def _indexed_hash(entries: Sequence[Mapping[str, object]], field: str) -> str:
    return sha256_json(
        [
            {"sobol_index": entry["sobol_index"], field: entry[field]}
            for entry in entries
        ]
    )


def layer_hashes(
    *,
    design: Mapping[str, object],
    scope: Mapping[str, object],
    generating_entries: Sequence[Mapping[str, object]],
    probe_entries: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    topology_query_rows = [
        {"sobol_index": entry["sobol_index"], **query}
        for entry in probe_entries
        for query in entry["topology_queries"]
    ]
    layers = {
        "coordinate_prefix_sha256": _indexed_hash(
            generating_entries, "unit_coordinates_sha256"
        ),
        "generating_geometry_queries_sha256": _indexed_hash(
            generating_entries, "geometry_query_sha256"
        ),
        "generating_amplitude_queries_sha256": _indexed_hash(
            generating_entries, "amplitude_query_sha256"
        ),
        "generating_geometry_bounds_embeddings_sha256": _indexed_hash(
            generating_entries, "geometry_bounds_embedding_sha256"
        ),
        "generating_amplitude_embeddings_ref1_sha256": _indexed_hash(
            generating_entries, "amplitude_embedding_ref1_sha256"
        ),
        "generating_amplitude_embeddings_ref1e4_sha256": _indexed_hash(
            generating_entries, "amplitude_embedding_ref1e4_sha256"
        ),
        "generating_amplitude_range_regimes_sha256": _indexed_hash(
            generating_entries, "amplitude_range_regimes_sha256"
        ),
        "generating_targets_sha256": _indexed_hash(generating_entries, "target_sha256"),
        "generating_amplitude_compositions_sha256": _indexed_hash(
            generating_entries, "amplitude_composition_sha256"
        ),
        "normalized_clean_recipes_sha256": _indexed_hash(
            generating_entries, "normalized_clean_recipe_sha256"
        ),
        "generating_entries_sha256": sha256_json(generating_entries),
        "topology_probe_queries_sha256": sha256_json(topology_query_rows),
        "topology_probe_model_embeddings_sha256": sha256_json(
            [
                {
                    "sobol_index": row["sobol_index"],
                    "topology_id": row["topology_id"],
                    "geometry_bounds_embedding_sha256": row[
                        "geometry_bounds_embedding_sha256"
                    ],
                    "amplitude_embedding_ref1_sha256": row[
                        "amplitude_embedding_ref1_sha256"
                    ],
                    "amplitude_embedding_ref1e4_sha256": row[
                        "amplitude_embedding_ref1e4_sha256"
                    ],
                }
                for row in topology_query_rows
            ]
        ),
        "topology_probe_entries_sha256": sha256_json(probe_entries),
    }
    layers["scientific_content_sha256"] = sha256_json(
        {
            "design": design,
            "scope": scope,
            "validation_contract": VALIDATION_CONTRACT,
            "generating_entries": generating_entries,
            "topology_probe_entries": probe_entries,
            "layer_sha256": layers,
        }
    )
    return layers


__all__ = [
    "V5_SOBOL_CROSS_PLATFORM_BITS",
    "V5_SOBOL_CROSS_PLATFORM_CLEAN_GROUP_POLICY",
    "V5_SOBOL_CROSS_PLATFORM_MANIFEST_SCHEMA",
    "V5_SOBOL_CROSS_PLATFORM_MANIFEST_VERSION",
    "V5_SOBOL_CROSS_PLATFORM_PREFIX_COUNT",
    "V5_SOBOL_CROSS_PLATFORM_PREFIX_START",
    "V5_SOBOL_CROSS_PLATFORM_PREFIX_STOP",
    "V5_SOBOL_CROSS_PLATFORM_PROBE_INDICES",
    "V5_SOBOL_CROSS_PLATFORM_SCRAMBLE_SEED",
    "V5_SOBOL_CROSS_PLATFORM_TOPOLOGY_IDS",
    "V5_SOBOL_SEMANTIC_DESIGN_SCHEMA",
    "V5_SOBOL_SEMANTIC_DESIGN_VERSION",
    "coordinate_sha256",
    "float64_vector_sha256",
    "layer_hashes",
    "neutral_clean_group_id",
    "scope_payload",
    "v5_sobol_runtime_free_semantic_design",
]
