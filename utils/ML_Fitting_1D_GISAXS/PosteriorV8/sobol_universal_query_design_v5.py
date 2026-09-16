"""Replayable cross-topology user-range queries from one direct Sobol point.

The clean recipe uses ``discrete.topology`` to select one generating topology.
Frozen cross-topology search needs a query for every explicitly selected
topology, including alternatives.  This module reuses the same named 168D
point for those queries without deriving a random seed or consulting the
generating parameters, curve, search result, or model score.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral
import re
from typing import Sequence

import numpy as np

from .contract import NUM_TOPOLOGIES, topology_from_id
from .amplitude_query_sampling_v5 import V5AmplitudeRangeRegimes
from .sobol_amplitude_recipe_v5 import direct_v5_amplitude_query
from .sobol_design_v5 import V5DesignPoint, V5SobolDesign
from .sobol_geometry_recipe_v5 import (
    direct_v5_geometry_query,
    direct_v5_geometry_query_for_topology,
)
from .sobol_numeric_canonicalization_v5 import (
    V5_DETERMINISTIC_NUMERIC_POLICY_VERSION,
    v5_numeric_policy_payload,
    v5_numeric_policy_sha256,
)
from .sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_INDEX,
    V5_SOBOL_RECIPE_COORDINATE_NAMES,
    V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
    V5_SOBOL_RECIPE_COORDINATE_VERSION,
    V5SobolCoordinateReader,
    validate_v5_sobol_recipe_coordinates,
)
from .split_design_v5 import MAIN_SPLITS
from .universal_query_contract_v5 import V5TopologyQuery


V5_SOBOL_UNIVERSAL_QUERY_DESIGN_SCHEMA = (
    "gisaxs.posterior_v8.direct_sobol_universal_topology_query_design/v6"
)
V5_SOBOL_UNIVERSAL_QUERY_DESIGN_VERSION = (
    "posterior_v8_same_named_point_numeric_contract_cross_topology_queries_v6"
)
V5_SOBOL_UNIVERSAL_POINT_HASH_VERSION = (
    "posterior_v8_sobol_design_bound_168d_exact_point_identity_v3"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_SPLITS = tuple(value for value in MAIN_SPLITS if value != "ood")


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _strict_json_object(encoded: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate universal-query design field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(encoded, object_pairs_hook=reject_duplicates)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid universal-query design JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("universal-query design JSON must contain one object")
    return value


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _split(value: object) -> str:
    if not isinstance(value, str) or value not in _ID_SPLITS:
        raise ValueError(f"assigned_split must be one of {_ID_SPLITS}")
    return value


def _selected_topology_ids(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("selected_topology_ids must be a sequence of integers")
    try:
        supplied = tuple(values)
    except TypeError as exc:
        raise TypeError("selected_topology_ids must be a sequence of integers") from exc
    if not supplied:
        raise ValueError("selected_topology_ids cannot be empty")
    selected_values = []
    for value in supplied:
        topology_from_id(value)
        selected_values.append(int(value))
    selected = tuple(selected_values)
    if len(set(selected)) != len(selected):
        raise ValueError("selected_topology_ids must be unique")
    return tuple(sorted(selected))


def _coordinate_bytes_sha256(values: tuple[float, ...]) -> str:
    coordinates = np.asarray(values, dtype="<f8")
    return sha256(coordinates.tobytes(order="C")).hexdigest()


def v5_sobol_universal_design_point_sha256(
    *,
    sobol_index: int,
    assigned_split: str,
    clean_group_id: str,
    sobol_design_sha256: str,
    unit_coordinates: Sequence[float],
) -> str:
    """Hash the exact point and its frozen design/split parent identity."""

    index = _nonnegative_integer(sobol_index, "sobol_index")
    split = _split(assigned_split)
    group = _digest(clean_group_id, "clean_group_id")
    design = _digest(sobol_design_sha256, "sobol_design_sha256")
    coordinates = validate_v5_sobol_recipe_coordinates(unit_coordinates)
    return sha256(
        _canonical_json(
            {
                "version": V5_SOBOL_UNIVERSAL_POINT_HASH_VERSION,
                "sobol_index": index,
                "assigned_split": split,
                "clean_group_id": group,
                "sobol_design_sha256": design,
                "coordinate_contract_sha256": V5_SOBOL_RECIPE_COORDINATE_SHA256,
                "unit_coordinates_binary_encoding": "little_endian_float64_C_order",
                "unit_coordinates_sha256": _coordinate_bytes_sha256(coordinates),
            }
        ).encode("utf-8")
    ).hexdigest()


def _generating_topology_id(coordinates: tuple[float, ...]) -> int:
    value = coordinates[V5_SOBOL_RECIPE_COORDINATE_INDEX["discrete.topology"]]
    return min(int(value * NUM_TOPOLOGIES), NUM_TOPOLOGIES - 1)


@dataclass(frozen=True)
class _TopologyReplay:
    query: V5TopologyQuery
    amplitude_range_regimes: V5AmplitudeRangeRegimes
    used_coordinate_names: tuple[str, ...]
    inactive_coordinate_names: tuple[str, ...]


def _replay_topology_queries(
    coordinates: tuple[float, ...],
    *,
    sobol_index: int,
    selected_topology_ids: tuple[int, ...],
) -> tuple[int, tuple[_TopologyReplay, ...]]:
    generating_reader = V5SobolCoordinateReader.create(coordinates)
    generating_geometry = direct_v5_geometry_query(
        generating_reader,
        sobol_index=sobol_index,
    )
    generating_amplitude, generating_regimes = direct_v5_amplitude_query(
        generating_reader,
        generating_geometry,
    )
    generating = V5TopologyQuery(generating_geometry, generating_amplitude)
    generating_topology_id = _generating_topology_id(coordinates)
    if generating.topology_id != generating_topology_id:  # pragma: no cover
        raise RuntimeError("generating topology coordinate mapping disagrees with its query")
    if generating_topology_id not in selected_topology_ids:
        raise ValueError("selected_topology_ids must include the generating topology")

    replays = []
    for topology_id in selected_topology_ids:
        if topology_id == generating_topology_id:
            reader = generating_reader
            query = generating
            regimes = generating_regimes
        else:
            reader = V5SobolCoordinateReader.create(coordinates)
            geometry = direct_v5_geometry_query_for_topology(
                reader,
                topology_id=topology_id,
                sobol_index=sobol_index,
            )
            amplitude, regimes = direct_v5_amplitude_query(reader, geometry)
            query = V5TopologyQuery(geometry, amplitude)
        replays.append(
            _TopologyReplay(
                query=query,
                amplitude_range_regimes=regimes,
                used_coordinate_names=tuple(
                    name
                    for name in V5_SOBOL_RECIPE_COORDINATE_NAMES
                    if name in reader.used
                ),
                inactive_coordinate_names=reader.inactive,
            )
        )
    return generating_topology_id, tuple(replays)


def _correlation_policy() -> dict[str, object]:
    return {
        "numeric_policy_version": V5_DETERMINISTIC_NUMERIC_POLICY_VERSION,
        "numeric_policy_contract": v5_numeric_policy_payload(
            V5_DETERMINISTIC_NUMERIC_POLICY_VERSION
        ),
        "numeric_policy_sha256": v5_numeric_policy_sha256(
            V5_DETERMINISTIC_NUMERIC_POLICY_VERSION
        ),
        "same_exact_168d_named_point_reused_for_every_selected_topology": True,
        "topology_queries_are_statistically_independent": False,
        "selected_topologies_source": "explicit_external_topology_id_subset",
        "discrete_topology_coordinate_role": "generating_topology_only",
        "alternative_topology_rule": (
            "replace_only_the_topology_catalog_entry_then_apply_the_same_direct_named_"
            "geometry_and_amplitude_range_transforms"
        ),
        "geometry_slot_rule": (
            "slot_i_uses_geometry.slot_i coordinates; active axes and physical domains "
            "follow the explicitly selected slot shape"
        ),
        "D_rule": (
            "slot_i_D_policy_coordinate_is_shared; a shape-specific authoritative_exclusion_"
            "size_lower_bound_is_applied_when_D_is_enabled"
        ),
        "resolution_rule": (
            "one_shared_resolution_presence_coordinate_and_shared_sigma_res_nu_res_range_"
            "coordinates_apply_to_every_selected_topology"
        ),
        "amplitude_axis_rule": (
            "each_BG_k_Int_i_and_int_Res_axis_has_its_own_regime_width_and_position_"
            "coordinates; shared named axis coordinates apply to every selected topology"
        ),
        "amplitude_independent_intensity_rule": (
            "slot_i uses independent Int_i regime, width, and position coordinates; the "
            "active prefix follows the selected topology count and no sum-to-one "
            "normalization is applied"
        ),
        "unused_coordinate_rule": "retained_in_point_hash_and_explicitly_listed_per_topology",
        "curve_or_search_result_conditions_range_generation": False,
        "generating_parameters_condition_alternative_ranges": False,
        "coordinate_derived_PRNG_seed": False,
        "rejection_or_retry": False,
    }


def _topology_payload(
    replay: _TopologyReplay,
    *,
    generating_topology_id: int,
) -> dict[str, object]:
    query = replay.query
    return {
        **query.audit_payload(),
        "topology_query_sha256": query.sha256,
        "is_generating_topology": query.topology_id == generating_topology_id,
        "generating_pair_uses_original_direct_query_transform": (
            query.topology_id == generating_topology_id
        ),
        "amplitude_range_regime": replay.amplitude_range_regimes.summary,
        "amplitude_range_regimes": replay.amplitude_range_regimes.audit_payload(),
        "amplitude_range_regimes_sha256": replay.amplitude_range_regimes.sha256,
        "geometry_query": json.loads(query.geometry.canonical_json),
        "amplitude_query": json.loads(query.amplitude.canonical_json),
        "used_coordinate_names": list(replay.used_coordinate_names),
        "inactive_coordinate_names": list(replay.inactive_coordinate_names),
    }


def _artifact_payload(
    *,
    sobol_index: int,
    assigned_split: str,
    clean_group_id: str,
    sobol_design_sha256: str,
    design_point_sha256: str,
    unit_coordinates: tuple[float, ...],
    generating_topology_id: int,
    selected_topology_ids: tuple[int, ...],
    replays: tuple[_TopologyReplay, ...],
) -> dict[str, object]:
    return {
        "schema": V5_SOBOL_UNIVERSAL_QUERY_DESIGN_SCHEMA,
        "version": V5_SOBOL_UNIVERSAL_QUERY_DESIGN_VERSION,
        "source": {
            "sobol_index": sobol_index,
            "assigned_split": assigned_split,
            "ood_label": None,
            "clean_group_id": clean_group_id,
            "sobol_design_sha256": sobol_design_sha256,
            "design_point_sha256": design_point_sha256,
            "coordinate_schema": V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
            "coordinate_version": V5_SOBOL_RECIPE_COORDINATE_VERSION,
            "coordinate_contract_sha256": V5_SOBOL_RECIPE_COORDINATE_SHA256,
            "unit_coordinates_binary_encoding": "little_endian_float64_C_order",
            "unit_coordinates_sha256": _coordinate_bytes_sha256(unit_coordinates),
            "unit_coordinates": list(unit_coordinates),
            "coordinate_consumption": "direct_named_transforms_no_seed_bridge",
        },
        "generating_topology_id": generating_topology_id,
        "selected_topology_ids": list(selected_topology_ids),
        "selected_topology_count": len(selected_topology_ids),
        "range_policy": _correlation_policy(),
        "topology_queries": [
            _topology_payload(value, generating_topology_id=generating_topology_id)
            for value in replays
        ],
    }


@dataclass(frozen=True)
class _DerivedDesign:
    sobol_index: int
    assigned_split: str
    clean_group_id: str
    sobol_design_sha256: str
    design_point_sha256: str
    unit_coordinates: tuple[float, ...]
    generating_topology_id: int
    selected_topology_ids: tuple[int, ...]
    replays: tuple[_TopologyReplay, ...]
    canonical_json: str
    sha256: str


def _derive_design(
    *,
    sobol_index: int,
    assigned_split: str,
    clean_group_id: str,
    sobol_design_sha256: str,
    unit_coordinates: Sequence[float],
    selected_topology_ids: Sequence[int],
) -> _DerivedDesign:
    index = _nonnegative_integer(sobol_index, "sobol_index")
    split = _split(assigned_split)
    group = _digest(clean_group_id, "clean_group_id")
    design_digest = _digest(sobol_design_sha256, "sobol_design_sha256")
    coordinates = validate_v5_sobol_recipe_coordinates(unit_coordinates)
    selected = _selected_topology_ids(selected_topology_ids)
    generating_topology_id, replays = _replay_topology_queries(
        coordinates,
        sobol_index=index,
        selected_topology_ids=selected,
    )
    point_digest = v5_sobol_universal_design_point_sha256(
        sobol_index=index,
        assigned_split=split,
        clean_group_id=group,
        sobol_design_sha256=design_digest,
        unit_coordinates=coordinates,
    )
    payload = _artifact_payload(
        sobol_index=index,
        assigned_split=split,
        clean_group_id=group,
        sobol_design_sha256=design_digest,
        design_point_sha256=point_digest,
        unit_coordinates=coordinates,
        generating_topology_id=generating_topology_id,
        selected_topology_ids=selected,
        replays=replays,
    )
    canonical = _canonical_json(payload)
    return _DerivedDesign(
        sobol_index=index,
        assigned_split=split,
        clean_group_id=group,
        sobol_design_sha256=design_digest,
        design_point_sha256=point_digest,
        unit_coordinates=coordinates,
        generating_topology_id=generating_topology_id,
        selected_topology_ids=selected,
        replays=replays,
        canonical_json=canonical,
        sha256=sha256(canonical.encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True)
class V5SobolUniversalTopologyQueryDesign:
    """Immutable explicit topology-query set bound to one frozen Sobol point."""

    sobol_index: int
    assigned_split: str
    clean_group_id: str
    sobol_design_sha256: str
    design_point_sha256: str
    unit_coordinates: tuple[float, ...]
    generating_topology_id: int
    selected_topology_ids: tuple[int, ...]
    topology_queries: tuple[V5TopologyQuery, ...]
    canonical_json: str
    sha256: str
    schema_version: str = V5_SOBOL_UNIVERSAL_QUERY_DESIGN_SCHEMA
    version: str = V5_SOBOL_UNIVERSAL_QUERY_DESIGN_VERSION

    @classmethod
    def create(
        cls,
        *,
        point: V5DesignPoint,
        design: V5SobolDesign,
        selected_topology_ids: Sequence[int],
    ) -> "V5SobolUniversalTopologyQueryDesign":
        if not isinstance(point, V5DesignPoint):
            raise TypeError("point must be a V5DesignPoint")
        if not isinstance(design, V5SobolDesign):
            raise TypeError("design must be a V5SobolDesign")
        if design.coordinate_names != V5_SOBOL_RECIPE_COORDINATE_NAMES:
            raise ValueError("Sobol design does not use the frozen direct-recipe coordinates")
        if design.coordinate_contract_sha256 != V5_SOBOL_RECIPE_COORDINATE_SHA256:
            raise ValueError("Sobol design is not bound to the frozen coordinate contract hash")
        if point.ood_label is not None or point.assigned_split == "ood":
            raise ValueError(
                "universal direct-Sobol topology queries are fail-closed for unregistered OOD "
                "transforms"
            )
        return cls._from_source(
            sobol_index=point.sobol_index,
            assigned_split=point.assigned_split,
            clean_group_id=point.clean_group_id,
            sobol_design_sha256=design.sha256,
            unit_coordinates=point.unit_coordinates,
            selected_topology_ids=selected_topology_ids,
        )

    @classmethod
    def _from_source(
        cls,
        *,
        sobol_index: int,
        assigned_split: str,
        clean_group_id: str,
        sobol_design_sha256: str,
        unit_coordinates: Sequence[float],
        selected_topology_ids: Sequence[int],
    ) -> "V5SobolUniversalTopologyQueryDesign":
        derived = _derive_design(
            sobol_index=sobol_index,
            assigned_split=assigned_split,
            clean_group_id=clean_group_id,
            sobol_design_sha256=sobol_design_sha256,
            unit_coordinates=unit_coordinates,
            selected_topology_ids=selected_topology_ids,
        )
        return cls(
            sobol_index=derived.sobol_index,
            assigned_split=derived.assigned_split,
            clean_group_id=derived.clean_group_id,
            sobol_design_sha256=derived.sobol_design_sha256,
            design_point_sha256=derived.design_point_sha256,
            unit_coordinates=derived.unit_coordinates,
            generating_topology_id=derived.generating_topology_id,
            selected_topology_ids=derived.selected_topology_ids,
            topology_queries=tuple(value.query for value in derived.replays),
            canonical_json=derived.canonical_json,
            sha256=derived.sha256,
        )

    def __post_init__(self) -> None:
        if self.schema_version != V5_SOBOL_UNIVERSAL_QUERY_DESIGN_SCHEMA:
            raise ValueError("unsupported universal-query design schema")
        if self.version != V5_SOBOL_UNIVERSAL_QUERY_DESIGN_VERSION:
            raise ValueError("unsupported universal-query design version")
        topology_from_id(self.generating_topology_id)
        replay = _derive_design(
            sobol_index=self.sobol_index,
            assigned_split=self.assigned_split,
            clean_group_id=self.clean_group_id,
            sobol_design_sha256=self.sobol_design_sha256,
            unit_coordinates=self.unit_coordinates,
            selected_topology_ids=self.selected_topology_ids,
        )
        expected = (
            replay.sobol_index,
            replay.assigned_split,
            replay.clean_group_id,
            replay.sobol_design_sha256,
            replay.design_point_sha256,
            replay.unit_coordinates,
            replay.generating_topology_id,
            replay.selected_topology_ids,
            tuple(value.query for value in replay.replays),
            replay.canonical_json,
            replay.sha256,
        )
        actual = (
            self.sobol_index,
            self.assigned_split,
            self.clean_group_id,
            self.sobol_design_sha256,
            self.design_point_sha256,
            self.unit_coordinates,
            self.generating_topology_id,
            self.selected_topology_ids,
            self.topology_queries,
            self.canonical_json,
            self.sha256,
        )
        if actual != expected:
            raise ValueError("universal-query design does not replay from its named Sobol point")

    def to_json(self) -> str:
        payload = json.loads(self.canonical_json)
        payload["artifact_sha256"] = self.sha256
        return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, encoded: str) -> "V5SobolUniversalTopologyQueryDesign":
        payload = _strict_json_object(encoded)
        expected_fields = {
            "schema",
            "version",
            "source",
            "generating_topology_id",
            "selected_topology_ids",
            "selected_topology_count",
            "range_policy",
            "topology_queries",
            "artifact_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError("universal-query design fields are incomplete or unsupported")
        source = payload.get("source")
        expected_source_fields = {
            "sobol_index",
            "assigned_split",
            "ood_label",
            "clean_group_id",
            "sobol_design_sha256",
            "design_point_sha256",
            "coordinate_schema",
            "coordinate_version",
            "coordinate_contract_sha256",
            "unit_coordinates_binary_encoding",
            "unit_coordinates_sha256",
            "unit_coordinates",
            "coordinate_consumption",
        }
        if not isinstance(source, dict) or set(source) != expected_source_fields:
            raise ValueError("universal-query design source fields are incomplete or unsupported")
        try:
            replay = cls._from_source(
                sobol_index=source["sobol_index"],
                assigned_split=source["assigned_split"],
                clean_group_id=source["clean_group_id"],
                sobol_design_sha256=source["sobol_design_sha256"],
                unit_coordinates=source["unit_coordinates"],
                selected_topology_ids=payload["selected_topology_ids"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid universal-query design payload") from exc
        supplied_hash = payload.pop("artifact_sha256")
        if payload != json.loads(replay.canonical_json) or supplied_hash != replay.sha256:
            raise ValueError("universal-query design payload/hash does not reproduce")
        return replay


def materialize_v5_sobol_universal_topology_query_design(
    point: V5DesignPoint,
    design: V5SobolDesign,
    *,
    selected_topology_ids: Sequence[int],
) -> V5SobolUniversalTopologyQueryDesign:
    """Materialize explicit replayable topology queries for one exact point."""

    return V5SobolUniversalTopologyQueryDesign.create(
        point=point,
        design=design,
        selected_topology_ids=selected_topology_ids,
    )


def materialize_v5_topology_queries_from_named_coordinates(
    *,
    sobol_index: int,
    unit_coordinates: Sequence[float],
    selected_topology_ids: Sequence[int],
) -> tuple[int, tuple[V5TopologyQuery, ...]]:
    """Replay queries from an upstream-validated exact named-coordinate vector.

    This lower-level adapter does not claim that the supplied coordinates are a
    raw Sobol point.  The caller owns and must bind any deterministic projection
    (for example, the balanced K1 categorical forcing contract).
    """

    index = _nonnegative_integer(sobol_index, "sobol_index")
    coordinates = validate_v5_sobol_recipe_coordinates(unit_coordinates)
    selected = _selected_topology_ids(selected_topology_ids)
    generating_topology_id, replays = _replay_topology_queries(
        coordinates,
        sobol_index=index,
        selected_topology_ids=selected,
    )
    return generating_topology_id, tuple(value.query for value in replays)


__all__ = [
    "V5_SOBOL_UNIVERSAL_POINT_HASH_VERSION",
    "V5_SOBOL_UNIVERSAL_QUERY_DESIGN_SCHEMA",
    "V5_SOBOL_UNIVERSAL_QUERY_DESIGN_VERSION",
    "V5SobolUniversalTopologyQueryDesign",
    "materialize_v5_sobol_universal_topology_query_design",
    "materialize_v5_topology_queries_from_named_coordinates",
    "v5_sobol_universal_design_point_sha256",
]
