"""Build the finite, byte-exact V5.2 cross-platform Sobol attestation.

The fixed scope covers generating physics and clean recipes at indices
0..1023 plus all 34 topology queries at ten preregistered probes.  Scientific
floats and derived hashes are never normalized.  Only runtime metadata and
the design/point identities contaminated by that metadata are replaced with
their declared runtime-free semantic identities.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json

import numpy as np
from scipy.stats import qmc

from .bounds_query_v5 import V5_LOCAL_TARGET_OPEN_EPSILON, bounds_query_from_json
from .clean_recipe_forward_v5 import validate_v5_clean_recipe_like
from .grouped_amplitude_join_v5 import amplitude_query_from_json
from .sobol_cross_platform_contract_v5 import (
    LAYER_FIELDS,
    VALIDATION_CONTRACT,
    V5_SOBOL_CROSS_PLATFORM_BITS,
    V5_SOBOL_CROSS_PLATFORM_CLEAN_GROUP_POLICY,
    V5_SOBOL_CROSS_PLATFORM_MANIFEST_SCHEMA,
    V5_SOBOL_CROSS_PLATFORM_MANIFEST_VERSION,
    V5_SOBOL_CROSS_PLATFORM_PREFIX_COUNT,
    V5_SOBOL_CROSS_PLATFORM_PREFIX_START,
    V5_SOBOL_CROSS_PLATFORM_PREFIX_STOP,
    V5_SOBOL_CROSS_PLATFORM_PROBE_INDICES,
    V5_SOBOL_CROSS_PLATFORM_SCRAMBLE_SEED,
    V5_SOBOL_CROSS_PLATFORM_TOPOLOGY_IDS,
    V5_SOBOL_SEMANTIC_DESIGN_SCHEMA,
    V5_SOBOL_SEMANTIC_DESIGN_VERSION,
    canonical_json,
    coordinate_sha256,
    digest,
    float64_vector_sha256,
    layer_hashes,
    neutral_clean_group_id,
    scope_payload,
    sha256_json,
    v5_sobol_runtime_free_semantic_design,
)
from .sobol_cross_platform_manifest_io_v5 import (
    V5SobolCrossPlatformComparison,
    V5SobolCrossPlatformManifest,
    compare_v5_sobol_cross_platform_manifests,
)
from .sobol_design_v5 import (
    V5DesignPoint,
    _random_access_sobol_points,  # version-bound formal engine replay
)
from .sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
    V5_SOBOL_RECIPE_DIM,
    v5_sobol_recipe_design,
)
from .sobol_recipe_v5 import V5SobolCleanRecipe, materialize_v5_sobol_clean_recipe
from .sobol_universal_query_design_v5 import (
    V5SobolUniversalTopologyQueryDesign,
    materialize_v5_sobol_universal_topology_query_design,
)


# Kept module-local so focused tests can recompute a deliberately tampered
# aggregate without widening the public contract API.
_LAYER_FIELDS = LAYER_FIELDS
_layer_hashes = layer_hashes


def _model_embedding_hashes(geometry, amplitude) -> dict[str, str]:
    return {
        "geometry_bounds_embedding_sha256": float64_vector_sha256(
            geometry.bounds_embedding,
            expected_size=78,
        ),
        "amplitude_embedding_ref1_sha256": float64_vector_sha256(
            amplitude.model_embedding(1.0),
            expected_size=21,
        ),
        "amplitude_embedding_ref1e4_sha256": float64_vector_sha256(
            amplitude.model_embedding(1.0e4),
            expected_size=21,
        ),
    }


def _target_payload(recipe: V5SobolCleanRecipe) -> dict[str, object]:
    target = recipe.target
    return {
        "version": target.version,
        "query_sha256": target.query.sha256,
        "pattern_id": target.pattern_id,
        "target_seed": target.target_seed,
        "physical_numeric_policy_version": target.physical_numeric_policy_version,
        "local_target_unit": list(target.local_target_unit),
        "truth_components": [asdict(value) for value in target.truth_components],
        "truth_resolution": (
            None if target.truth_resolution is None else asdict(target.truth_resolution)
        ),
    }


def _normalized_clean_recipe_payload(
    recipe: V5SobolCleanRecipe,
    *,
    semantic_design_sha256: str,
) -> dict[str, object]:
    payload = json.loads(recipe.canonical_json)
    payload["source"]["sobol_design_sha256"] = semantic_design_sha256
    payload["source"]["sobol_design_identity"] = "runtime_free_semantic_design_sha256"
    return payload


def _semantic_design_point_sha256(
    *,
    semantic_design_sha256: str,
    sobol_index: int,
    clean_group_id: str,
    unit_coordinates_sha256: str,
) -> str:
    return sha256_json(
        {
            "version": "posterior_v8_runtime_free_attestation_point_identity_v1",
            "semantic_design_sha256": semantic_design_sha256,
            "sobol_index": sobol_index,
            "assigned_split": "train",
            "ood_label": None,
            "clean_group_id": clean_group_id,
            "coordinate_contract_sha256": V5_SOBOL_RECIPE_COORDINATE_SHA256,
            "unit_coordinates_binary_encoding": "little_endian_float64_C_order",
            "unit_coordinates_sha256": unit_coordinates_sha256,
        }
    )


def _normalized_query_design_payload(
    artifact: V5SobolUniversalTopologyQueryDesign,
    *,
    semantic_design_sha256: str,
    unit_coordinates_sha256: str,
) -> dict[str, object]:
    payload = json.loads(artifact.canonical_json)
    source = payload["source"]
    source["sobol_design_sha256"] = semantic_design_sha256
    source["sobol_design_identity"] = "runtime_free_semantic_design_sha256"
    source["design_point_sha256"] = _semantic_design_point_sha256(
        semantic_design_sha256=semantic_design_sha256,
        sobol_index=artifact.sobol_index,
        clean_group_id=artifact.clean_group_id,
        unit_coordinates_sha256=unit_coordinates_sha256,
    )
    return payload


def _assert_closed_interval(low: float, value: float, high: float, name: str) -> None:
    if not low <= value <= high:
        raise RuntimeError(f"{name} escaped its closed query interval")


def _strict_query_replay(recipe: V5SobolCleanRecipe) -> None:
    if validate_v5_clean_recipe_like(recipe) is not recipe:
        raise RuntimeError("direct Sobol clean-recipe protocol replay failed")
    if bounds_query_from_json(recipe.query.canonical_json, recipe.query.sha256) != recipe.query:
        raise RuntimeError("geometry query strict replay failed")
    amplitude = amplitude_query_from_json(
        recipe.amplitude_query.canonical_json,
        recipe.amplitude_query.sha256,
    )
    if amplitude != recipe.amplitude_query:
        raise RuntimeError("amplitude query strict replay failed")


def _strict_geometry_containment(recipe: V5SobolCleanRecipe) -> None:
    codec = recipe.query.codec_for(recipe.target.pattern_id)
    latent_components, resolution = codec.decode(recipe.target.local_target_unit)
    reencoded = codec.encode(latent_components, resolution)
    if (
        codec.latent_components_to_gui(latent_components)
        != recipe.target.truth_components
        or resolution != recipe.target.truth_resolution
    ):
        raise RuntimeError("branch codec did not exactly replay the stored physical target")
    for value, present in zip(reencoded.unit_cube, codec.active_mask):
        if present:
            _assert_closed_interval(
                0.0,
                value,
                1.0,
                "re-encoded active local target",
            )
    for value, present in zip(recipe.target.local_target_unit, codec.active_mask):
        if present:
            _assert_closed_interval(
                V5_LOCAL_TARGET_OPEN_EPSILON,
                value,
                1.0 - V5_LOCAL_TARGET_OPEN_EPSILON,
                "active local target",
            )
    for component_index, (bounds, truth) in enumerate(
        zip(recipe.query.component_bounds, recipe.target.truth_components)
    ):
        for axis in ("R", "sigma_R", "h", "sigma_h", "D", "sigma_D"):
            interval, value = getattr(bounds, axis), getattr(truth, axis)
            if value is not None:
                if interval is None:
                    raise RuntimeError(
                        f"component {component_index} axis {axis} lacks bounds"
                    )
                _assert_closed_interval(
                    interval.low,
                    value,
                    interval.high,
                    f"component {component_index} axis {axis}",
                )
    resolution = recipe.target.truth_resolution
    if resolution is None:
        if recipe.amplitude.resolution_present:
            raise RuntimeError("Resolution amplitude exists without Resolution target")
        return
    bounds = recipe.query.resolution_bounds
    if bounds is None:
        raise RuntimeError("Resolution target lacks query bounds")
    _assert_closed_interval(
        bounds.sigma_res.low,
        resolution.sigma_res,
        bounds.sigma_res.high,
        "resolution sigma_res",
    )
    _assert_closed_interval(
        bounds.nu_res.low,
        resolution.nu_res,
        bounds.nu_res.high,
        "resolution nu_res",
    )


def _strict_amplitude_containment(recipe: V5SobolCleanRecipe) -> None:
    amplitude, query = recipe.amplitude, recipe.amplitude_query
    _assert_closed_interval(
        query.background.low, amplitude.background, query.background.high, "BG"
    )
    _assert_closed_interval(query.k.low, amplitude.k, query.k.high, "k")
    for slot, (interval, value) in enumerate(
        zip(query.component_intensities, amplitude.component_intensities), 1
    ):
        _assert_closed_interval(interval.low, value, interval.high, f"Int_{slot}")
    if amplitude.resolution_present:
        if query.int_res is None:
            raise RuntimeError("Resolution amplitude lacks int_Res bounds")
        _assert_closed_interval(
            query.int_res.low,
            amplitude.int_res,
            query.int_res.high,
            "int_Res",
        )
    constraint = query.constraint_for_branch(
        resolution_present=amplitude.resolution_present
    )
    if not constraint.contains(amplitude.coefficient_vector, k=amplitude.k, atol=0.0):
        raise RuntimeError("effective amplitudes violate the exact shared-k constraint")


def _strict_recipe_replay_and_containment(
    recipe: V5SobolCleanRecipe,
    *,
    point: V5DesignPoint,
    design,
) -> None:
    if materialize_v5_sobol_clean_recipe(point, design) != recipe:
        raise RuntimeError("direct Sobol clean recipe did not replay exactly")
    _strict_query_replay(recipe)
    _strict_geometry_containment(recipe)
    _strict_amplitude_containment(recipe)


def _generating_entry(
    recipe: V5SobolCleanRecipe,
    *,
    semantic_design_sha256: str,
) -> dict[str, object]:
    return {
        "sobol_index": recipe.sobol_index,
        "clean_group_id": recipe.clean_group_id,
        "unit_coordinates_sha256": coordinate_sha256(recipe.unit_coordinates),
        "generating_topology_id": recipe.query.topology_id,
        "branch_pattern_id": recipe.target.pattern_id,
        "geometry_query_sha256": recipe.query.sha256,
        "amplitude_query_sha256": recipe.amplitude_query.sha256,
        **_model_embedding_hashes(recipe.query, recipe.amplitude_query),
        "amplitude_range_regimes_sha256": recipe.physics.amplitude_range_regimes.sha256,
        "target_sha256": sha256_json(_target_payload(recipe)),
        "amplitude_composition_sha256": sha256_json(recipe.amplitude.audit_payload()),
        "normalized_clean_recipe_sha256": sha256_json(
            _normalized_clean_recipe_payload(
                recipe,
                semantic_design_sha256=semantic_design_sha256,
            )
        ),
    }


def _probe_entry(
    artifact: V5SobolUniversalTopologyQueryDesign,
    *,
    semantic_design_sha256: str,
) -> dict[str, object]:
    coordinate_digest = coordinate_sha256(artifact.unit_coordinates)
    topology_queries = [
        {
            "topology_id": query.topology_id,
            "topology_query_sha256": query.sha256,
            "geometry_query_sha256": query.geometry.sha256,
            "amplitude_query_sha256": query.amplitude.sha256,
            **_model_embedding_hashes(query.geometry, query.amplitude),
        }
        for query in artifact.topology_queries
    ]
    return {
        "sobol_index": artifact.sobol_index,
        "unit_coordinates_sha256": coordinate_digest,
        "generating_topology_id": artifact.generating_topology_id,
        "normalized_query_design_sha256": sha256_json(
            _normalized_query_design_payload(
                artifact,
                semantic_design_sha256=semantic_design_sha256,
                unit_coordinates_sha256=coordinate_digest,
            )
        ),
        "topology_queries": topology_queries,
        "topology_queries_sha256": sha256_json(topology_queries),
    }


def _replayed_prefix(design) -> np.ndarray:
    coordinates = qmc.Sobol(
        d=V5_SOBOL_RECIPE_DIM,
        scramble=True,
        bits=V5_SOBOL_CROSS_PLATFORM_BITS,
        seed=V5_SOBOL_CROSS_PLATFORM_SCRAMBLE_SEED,
    ).random_base2(10)
    expected_shape = (V5_SOBOL_CROSS_PLATFORM_PREFIX_COUNT, V5_SOBOL_RECIPE_DIM)
    if coordinates.shape != expected_shape:
        raise RuntimeError("frozen Sobol engine did not return the 1024x168 prefix")
    random_access = _random_access_sobol_points(
        design,
        tuple(range(V5_SOBOL_CROSS_PLATFORM_PREFIX_START, V5_SOBOL_CROSS_PLATFORM_PREFIX_STOP)),
    )
    if (
        random_access.dtype != np.float64
        or random_access.shape != coordinates.shape
        or random_access.tobytes(order="C") != coordinates.tobytes(order="C")
    ):
        raise RuntimeError(
            "Sobol random_base2 prefix disagrees byte-for-byte with frozen Gray-code "
            "random access"
        )
    return coordinates


def build_v5_sobol_cross_platform_manifest(
    *,
    source_archive_sha256: str,
    source_manifest_sha256: str,
    source_tree_sha256: str,
) -> V5SobolCrossPlatformManifest:
    """Materialize and attest the complete finite cross-platform gate scope."""

    source = {
        "source_archive_sha256": digest(source_archive_sha256, "source_archive_sha256"),
        "source_manifest_sha256": digest(
            source_manifest_sha256, "source_manifest_sha256"
        ),
        "source_tree_sha256": digest(source_tree_sha256, "source_tree_sha256"),
    }
    semantic_design = v5_sobol_runtime_free_semantic_design()
    semantic_digest = str(semantic_design["semantic_sha256"])
    design = v5_sobol_recipe_design(
        scramble_seed=V5_SOBOL_CROSS_PLATFORM_SCRAMBLE_SEED
    )
    if design.bits != V5_SOBOL_CROSS_PLATFORM_BITS:
        raise RuntimeError("live direct-Sobol design does not use frozen bits=52")
    coordinates = _replayed_prefix(design)

    points: list[V5DesignPoint] = []
    generating_entries = []
    for index, row in enumerate(coordinates):
        point = V5DesignPoint(
            sobol_index=index,
            assigned_split="train",
            ood_label=None,
            clean_group_id=neutral_clean_group_id(semantic_digest, index),
            unit_coordinates=tuple(float(value) for value in np.asarray(row, dtype="<f8")),
        )
        recipe = materialize_v5_sobol_clean_recipe(point, design)
        _strict_recipe_replay_and_containment(recipe, point=point, design=design)
        points.append(point)
        generating_entries.append(
            _generating_entry(recipe, semantic_design_sha256=semantic_digest)
        )

    probe_entries = []
    for index in V5_SOBOL_CROSS_PLATFORM_PROBE_INDICES:
        artifact = materialize_v5_sobol_universal_topology_query_design(
            points[index],
            design,
            selected_topology_ids=V5_SOBOL_CROSS_PLATFORM_TOPOLOGY_IDS,
        )
        if V5SobolUniversalTopologyQueryDesign.from_json(artifact.to_json()) != artifact:
            raise RuntimeError("universal topology-query design did not replay exactly")
        for query in artifact.topology_queries:
            for resolution_present in query.amplitude.allowed_resolution_states:
                query.amplitude.constraint_for_branch(resolution_present=resolution_present)
        probe_entries.append(
            _probe_entry(artifact, semantic_design_sha256=semantic_digest)
        )

    scope = scope_payload()
    layers = layer_hashes(
        design=semantic_design,
        scope=scope,
        generating_entries=generating_entries,
        probe_entries=probe_entries,
    )
    core = {
        "schema": V5_SOBOL_CROSS_PLATFORM_MANIFEST_SCHEMA,
        "version": V5_SOBOL_CROSS_PLATFORM_MANIFEST_VERSION,
        "source": source,
        "design": semantic_design,
        "scope": scope,
        "validation_contract": VALIDATION_CONTRACT,
        "generating_entries": generating_entries,
        "topology_probe_entries": probe_entries,
        "layer_sha256": layers,
    }
    canonical = canonical_json(core)
    return V5SobolCrossPlatformManifest(
        canonical_json=canonical,
        sha256=sha256(canonical.encode("utf-8")).hexdigest(),
    )


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
    "V5SobolCrossPlatformComparison",
    "V5SobolCrossPlatformManifest",
    "build_v5_sobol_cross_platform_manifest",
    "compare_v5_sobol_cross_platform_manifests",
    "v5_sobol_runtime_free_semantic_design",
]
