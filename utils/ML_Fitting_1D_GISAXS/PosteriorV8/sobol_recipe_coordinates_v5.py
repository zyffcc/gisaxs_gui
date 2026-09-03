"""Frozen named coordinates for direct V5.2 Sobol recipe materialization.

Every coordinate has one stable semantic role.  The recipe materializer reads
these values directly; it must never compress a Sobol point into a PRNG seed.
Coordinates for axes absent from a selected topology remain part of the design
and are recorded as inactive by the materialized recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Sequence

import numpy as np

from .amplitude_query_sampling_v5 import (
    V5_AMPLITUDE_RANGE_REGIMES,
    V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA,
    V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION,
)
from .amplitude_sampling_v5 import V5_AMPLITUDE_REGIMES
from .bounds_query_v5 import AXIS_RANGE_PLACEMENTS, AXIS_RANGE_REGIMES
from .branch_codec import COMPONENT_UNIT_AXES, RESOLUTION_UNIT_AXES
from .contextual_branch_catalog import PRESENCE_POLICIES
from .contract import MAX_COMPONENTS, TOPOLOGIES
from .sobol_design_v5 import V5SobolDesign


V5_SOBOL_RECIPE_COORDINATE_SCHEMA = "gisaxs.posterior_v8.direct_sobol_recipe_coordinates/v4"
V5_SOBOL_RECIPE_COORDINATE_VERSION = (
    "posterior_v8_v5_2_named_per_amplitude_axis_query_branch_mapping_v5"
)

_INTERVAL_FIELDS = ("regime", "placement", "width", "position")
_GEOMETRY_INTERVAL_AXES = (
    "R",
    "sigma_R_fraction",
    "h",
    "sigma_h_fraction",
    "D",
    "sigma_D_fraction",
)
_AMPLITUDE_QUERY_AXES = (
    "BG",
    "k",
    *(f"Int_{slot + 1}" for slot in range(MAX_COMPONENTS)),
    "int_Res",
)
_AMPLITUDE_QUERY_FIELDS = ("regime", "width", "position")


def _interval_names(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}.{field}" for field in _INTERVAL_FIELDS)


def _coordinate_names() -> tuple[str, ...]:
    names: list[str] = ["discrete.topology"]
    names.extend(f"geometry.slot_{slot + 1}.D_policy" for slot in range(MAX_COMPONENTS))
    names.append("geometry.resolution_policy")
    for slot in range(MAX_COMPONENTS):
        for axis in _GEOMETRY_INTERVAL_AXES:
            names.extend(_interval_names(f"geometry.slot_{slot + 1}.{axis}"))
    for axis in ("sigma_res", "nu_res"):
        names.extend(_interval_names(f"geometry.resolution.{axis}"))
    for axis in _AMPLITUDE_QUERY_AXES:
        names.extend(
            f"amplitude.query.{axis}.{field}" for field in _AMPLITUDE_QUERY_FIELDS
        )
    names.append("discrete.branch_within_feasible_catalog")
    for slot in range(MAX_COMPONENTS):
        names.extend(f"target.slot_{slot + 1}.{axis}" for axis in COMPONENT_UNIT_AXES)
    names.extend(f"target.resolution.{axis}" for axis in RESOLUTION_UNIT_AXES)
    names.extend(
        (
            "amplitude.composition.regime_within_feasible_set",
            "amplitude.composition.selected_particle_within_feasible_set",
            "amplitude.composition.intensity_order",
        )
    )
    names.extend(
        f"amplitude.composition.Int_fraction_{slot + 1}" for slot in range(MAX_COMPONENTS)
    )
    names.extend(
        (
            "amplitude.composition.k",
            "amplitude.composition.BG",
            "amplitude.composition.int_Res",
        )
    )
    return tuple(names)


V5_SOBOL_RECIPE_COORDINATE_NAMES = _coordinate_names()
V5_SOBOL_RECIPE_DIM = len(V5_SOBOL_RECIPE_COORDINATE_NAMES)
V5_SOBOL_RECIPE_COORDINATE_INDEX = MappingProxyType(
    {name: index for index, name in enumerate(V5_SOBOL_RECIPE_COORDINATE_NAMES)}
)

if len(V5_SOBOL_RECIPE_COORDINATE_INDEX) != V5_SOBOL_RECIPE_DIM:  # pragma: no cover
    raise RuntimeError("direct Sobol coordinate names must be unique")
if V5_SOBOL_RECIPE_DIM != 168:  # pragma: no cover
    raise RuntimeError(f"direct V5.2 Sobol recipe must remain 168D, got {V5_SOBOL_RECIPE_DIM}")


def _topology_labels() -> tuple[str, ...]:
    return tuple("+".join(topology) for topology in TOPOLOGIES)


def v5_sobol_recipe_coordinate_contract() -> dict[str, object]:
    """Return the complete paper-auditable coordinate dictionary."""

    interval_semantics = {
        "regime": list(AXIS_RANGE_REGIMES),
        "placement": list(AXIS_RANGE_PLACEMENTS),
        "width": "continuous interval-width coordinate",
        "position": "continuous fixed value or interval-position coordinate",
    }
    return {
        "schema": V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
        "version": V5_SOBOL_RECIPE_COORDINATE_VERSION,
        "dimension": V5_SOBOL_RECIPE_DIM,
        "coordinate_names": list(V5_SOBOL_RECIPE_COORDINATE_NAMES),
        "coordinate_domain": "half_open_unit_cube_[0,1)",
        "consumption": "direct_coordinate_transforms_only_no_coordinate_derived_PRNG_seed",
        "amplitude_range_assignment": {
            "schema": V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA,
            "version": V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION,
        },
        "fixed_discrete_mappings": {
            "discrete.topology": list(_topology_labels()),
            "geometry.slot_[1:4].D_policy": list(PRESENCE_POLICIES),
            "geometry.resolution_policy": list(PRESENCE_POLICIES),
            "geometry.*.{regime,placement}": interval_semantics,
            "amplitude.query.{BG,k,Int_[1:4],int_Res}.regime": list(
                V5_AMPLITUDE_RANGE_REGIMES
            ),
        },
        "contextual_discrete_mappings": {
            "discrete.branch_within_feasible_catalog": (
                "floor(u*N) into the paired complete geometry_plus_amplitude query's "
                "ascending feasible_wire_pattern_ids"
            ),
            "amplitude.composition.regime_within_feasible_set": (
                "floor(u*N) into the ordered feasible subset of V5_AMPLITUDE_REGIMES"
            ),
            "amplitude.composition.selected_particle_within_feasible_set": (
                "floor(u*N) into ascending feasible particle slots when the selected regime "
                "requires one"
            ),
            "amplitude.composition.intensity_order": (
                "floor(u*K) circular rotation of active particle slots; ordering only and "
                "never a normalization constraint"
            ),
        },
        "independent_gui_amplitude_coordinates": {
            "amplitude.query.{axis}.regime": (
                "independent categorical range regime for each active BG, k, Int_i, and "
                "int_Res axis"
            ),
            "amplitude.query.{axis}.width": (
                "independent log-domain interval width for that axis when its regime varies"
            ),
            "amplitude.query.{axis}.position": (
                "independent fixed value or interior interval position for that axis"
            ),
            "amplitude.composition.Int_fraction_[1:4]": (
                "independent within-range coordinate for each active GUI Int_i; the historical "
                "fraction suffix is a coordinate fraction, not a component simplex fraction"
            ),
            "amplitude.composition.k": "independent shared GUI multiplier k",
            "normalization": "none; GUI Int_i values need not sum to one",
            "range_regime_correlation": "none_between_active_amplitude_axes",
        },
        "target_coordinates": {
            "component_axes": list(COMPONENT_UNIT_AXES),
            "resolution_axes": list(RESOLUTION_UNIT_AXES),
            "transform": "epsilon+(1-2*epsilon)*u on active branch axes",
        },
        "amplitude_regime_order": list(V5_AMPLITUDE_REGIMES),
        "inactive_coordinate_policy": (
            "absent-axis and regime-irrelevant coordinates are retained in the recipe hash, "
            "listed as inactive, and ignored by the physical map"
        ),
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


V5_SOBOL_RECIPE_COORDINATE_SHA256 = sha256(
    _canonical_json(v5_sobol_recipe_coordinate_contract()).encode("utf-8")
).hexdigest()


def validate_v5_sobol_recipe_coordinates(values: Sequence[float]) -> tuple[float, ...]:
    """Validate and freeze one exact half-open 168D Sobol point."""

    if isinstance(values, (str, bytes)):
        raise TypeError("unit_coordinates must be a numeric sequence")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("unit_coordinates must contain numeric values") from exc
    if array.shape != (V5_SOBOL_RECIPE_DIM,):
        raise ValueError(f"unit_coordinates must have shape ({V5_SOBOL_RECIPE_DIM},)")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array >= 1.0):
        raise ValueError("unit_coordinates must be finite and lie in [0, 1)")
    return tuple(float(value) for value in array)


@dataclass
class V5SobolCoordinateReader:
    """Track which named coordinates affect one contextual physical recipe."""

    values: tuple[float, ...]
    used: set[str]

    @classmethod
    def create(cls, values: Sequence[float]) -> "V5SobolCoordinateReader":
        return cls(validate_v5_sobol_recipe_coordinates(values), set())

    def take(self, name: str) -> float:
        try:
            index = V5_SOBOL_RECIPE_COORDINATE_INDEX[name]
        except KeyError as exc:  # pragma: no cover - internal coordinate dictionary invariant
            raise RuntimeError(f"unknown direct Sobol coordinate {name!r}") from exc
        self.used.add(name)
        return self.values[index]

    @property
    def inactive(self) -> tuple[str, ...]:
        return tuple(name for name in V5_SOBOL_RECIPE_COORDINATE_NAMES if name not in self.used)


def v5_sobol_recipe_design(*, scramble_seed: int) -> V5SobolDesign:
    """Build a frozen SciPy Sobol design with the exact recipe dictionary."""

    return V5SobolDesign(
        coordinate_names=V5_SOBOL_RECIPE_COORDINATE_NAMES,
        scramble_seed=scramble_seed,
        coordinate_contract_sha256=V5_SOBOL_RECIPE_COORDINATE_SHA256,
    )


__all__ = [
    "V5_SOBOL_RECIPE_COORDINATE_INDEX",
    "V5_SOBOL_RECIPE_COORDINATE_NAMES",
    "V5_SOBOL_RECIPE_COORDINATE_SCHEMA",
    "V5_SOBOL_RECIPE_COORDINATE_SHA256",
    "V5_SOBOL_RECIPE_COORDINATE_VERSION",
    "V5_SOBOL_RECIPE_DIM",
    "V5SobolCoordinateReader",
    "v5_sobol_recipe_coordinate_contract",
    "v5_sobol_recipe_design",
    "validate_v5_sobol_recipe_coordinates",
]
