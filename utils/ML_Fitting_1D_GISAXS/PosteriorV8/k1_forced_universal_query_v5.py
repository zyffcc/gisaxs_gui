"""Recipe-bound all-K1 query replay for balanced full-search supervision."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np

from .formal_production_search_contract_v5 import (
    topology_ids_for_v5_formal_production_stage,
)
from .grouped_artifact_v5 import canonical_json
from .k1_forced_sobol_recipe_v5 import (
    V5K1ForcedRecipeIdentity,
    decode_v5_k1_forced_recipe_identity,
)
from .k1_phase_c_contract_v5 import K1_PHASE_C_BRANCH_BY_ID
from .sobol_universal_query_design_v5 import (
    materialize_v5_topology_queries_from_named_coordinates,
)
from .universal_query_contract_v5 import V5TopologyQuery


V5_K1_FORCED_UNIVERSAL_QUERY_SET_SCHEMA = (
    "gisaxs.posterior_v8.k1_forced_universal_query_set/v1"
)
V5_K1_FORCED_UNIVERSAL_QUERY_SET_VERSION = (
    "posterior_v8_v5_2_recipe_bound_all_k1_named_coordinate_queries_v1"
)


def _coordinate_sha256(values: object) -> str:
    coordinates = np.asarray(values, dtype="<f8")
    return sha256(coordinates.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class _DerivedQuerySet:
    identity: V5K1ForcedRecipeIdentity
    selected_topology_ids: tuple[int, ...]
    generating_topology_id: int
    topology_queries: tuple[V5TopologyQuery, ...]
    audit_payload: dict[str, object]
    canonical_json: str
    sha256: str


def _derive_query_set(
    recipe_canonical_json: str,
    *,
    expected_recipe_sha256: str | None,
) -> _DerivedQuerySet:
    identity = decode_v5_k1_forced_recipe_identity(
        recipe_canonical_json,
        expected_sha256=expected_recipe_sha256,
    )
    payload = json.loads(recipe_canonical_json)
    source = payload["source"]
    physics = payload["physics"]
    selected = topology_ids_for_v5_formal_production_stage("K1")
    generating_topology_id, queries = (
        materialize_v5_topology_queries_from_named_coordinates(
            sobol_index=identity.sobol_index,
            unit_coordinates=source["forced_unit_coordinates"],
            selected_topology_ids=selected,
        )
    )
    branch = K1_PHASE_C_BRANCH_BY_ID[identity.branch_id]
    generating_matches = tuple(
        query for query in queries if query.topology_id == generating_topology_id
    )
    if len(generating_matches) != 1:
        raise ValueError("forced recipe has no unique generating K1 query")
    generating = generating_matches[0]
    if (
        generating.topology != (branch.shape,)
        or generating.geometry.canonical_json
        != canonical_json(physics["geometry_query"])
        or generating.amplitude.canonical_json
        != canonical_json(physics["amplitude_query"])
        or generating.feasible_wire_pattern_ids != (branch.pattern_id,)
        or physics["branch_pattern_id"] != branch.pattern_id
    ):
        raise ValueError("all-K1 query replay changed the persisted generating query")
    query_rows = [
        {
            "topology_id": query.topology_id,
            "topology_query_sha256": query.sha256,
            "geometry_query": json.loads(query.geometry.canonical_json),
            "amplitude_query": json.loads(query.amplitude.canonical_json),
            "feasible_wire_pattern_ids": list(query.feasible_wire_pattern_ids),
            "is_generating_topology": query.topology_id == generating_topology_id,
        }
        for query in queries
    ]
    audit = {
        "schema": V5_K1_FORCED_UNIVERSAL_QUERY_SET_SCHEMA,
        "version": V5_K1_FORCED_UNIVERSAL_QUERY_SET_VERSION,
        "scientific_role": "balanced_parent_all_k1_full_search_query_input",
        "forced_recipe_sha256": identity.recipe_sha256,
        "balanced_dataset_plan_sha256": identity.balanced_dataset_plan_sha256,
        "balanced_sobol_block_sha256": identity.balanced_sobol_block_sha256,
        "role": identity.role,
        "split_id": identity.split_id,
        "generating_branch_id": identity.branch_id,
        "generating_branch_pattern_id": branch.pattern_id,
        "sobol_index": identity.sobol_index,
        "clean_group_id": identity.clean_group_id,
        "parent_sobol_design_sha256": identity.sobol_design_sha256,
        "branch_forcing_sha256": source["branch_forcing_sha256"],
        "forced_unit_coordinates_sha256": _coordinate_sha256(
            source["forced_unit_coordinates"]
        ),
        "selected_topology_ids": list(selected),
        "generating_topology_id": generating_topology_id,
        "topology_queries": query_rows,
        "all_and_only_k1_topologies": True,
        "generating_query_matches_persisted_recipe": True,
        "curve_or_model_score_conditions_query": False,
        "training_authorization_granted": False,
    }
    encoded = canonical_json(audit)
    return _DerivedQuerySet(
        identity=identity,
        selected_topology_ids=selected,
        generating_topology_id=generating_topology_id,
        topology_queries=queries,
        audit_payload=audit,
        canonical_json=encoded,
        sha256=sha256(encoded.encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True)
class V5K1ForcedUniversalQuerySet:
    """All K1 topology queries replayed from one strict forced recipe."""

    recipe_canonical_json: str
    identity: V5K1ForcedRecipeIdentity
    selected_topology_ids: tuple[int, ...]
    generating_topology_id: int
    topology_queries: tuple[V5TopologyQuery, ...]
    canonical_json: str
    sha256: str

    def __post_init__(self) -> None:
        replay = _derive_query_set(
            self.recipe_canonical_json,
            expected_recipe_sha256=self.identity.recipe_sha256,
        )
        actual = (
            self.identity,
            self.selected_topology_ids,
            self.generating_topology_id,
            self.topology_queries,
            self.canonical_json,
            self.sha256,
        )
        expected = (
            replay.identity,
            replay.selected_topology_ids,
            replay.generating_topology_id,
            replay.topology_queries,
            replay.canonical_json,
            replay.sha256,
        )
        if actual != expected:
            raise ValueError("forced all-K1 query-set identity does not reproduce")

    def audit_payload(self) -> dict[str, object]:
        return json.loads(self.canonical_json)


def materialize_v5_k1_forced_universal_query_set(
    recipe_canonical_json: str,
    *,
    expected_recipe_sha256: str | None = None,
) -> V5K1ForcedUniversalQuerySet:
    """Strictly replay one forced recipe into all and only K1 queries."""

    derived = _derive_query_set(
        recipe_canonical_json,
        expected_recipe_sha256=expected_recipe_sha256,
    )
    return V5K1ForcedUniversalQuerySet(
        recipe_canonical_json=recipe_canonical_json,
        identity=derived.identity,
        selected_topology_ids=derived.selected_topology_ids,
        generating_topology_id=derived.generating_topology_id,
        topology_queries=derived.topology_queries,
        canonical_json=derived.canonical_json,
        sha256=derived.sha256,
    )


__all__ = [
    "V5_K1_FORCED_UNIVERSAL_QUERY_SET_SCHEMA",
    "V5_K1_FORCED_UNIVERSAL_QUERY_SET_VERSION",
    "V5K1ForcedUniversalQuerySet",
    "materialize_v5_k1_forced_universal_query_set",
]
