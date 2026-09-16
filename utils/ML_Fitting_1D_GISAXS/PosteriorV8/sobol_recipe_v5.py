"""Paper-grade direct Sobol-coordinate clean recipes for V5.2.

This is intentionally independent from the pilot seed-replay recipe.  A
scrambled Sobol point is retained verbatim and each named coordinate is mapped
directly into the query, branch, local target, and query-contained amplitude
composition.  No coordinate-derived random seed exists on this path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
import re
import numpy as np
from .canonical_component_slots import CANONICAL_COMPONENT_SLOTS_VERSION
from .clean_recipe_forward_v5 import (
    V5_CLEAN_EXACT_FORWARD_PATH,
    authoritative_v5_gui_parameters,
    evaluate_v5_clean_recipe_forward,
)
from .simulation import GridProvenance
from .sobol_design_v5 import V5DesignPoint, V5SobolDesign
from .sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_NAMES,
    V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
    V5_SOBOL_RECIPE_COORDINATE_VERSION,
    validate_v5_sobol_recipe_coordinates,
)
from .sobol_recipe_physics_v5 import (
    V5_DIRECT_PHYSICS_VERSION,
    V5DirectAmplitudeComposition,
    V5DirectPhysics,
    direct_v5_physics_from_sobol,
)
from .sobol_numeric_canonicalization_v5 import (
    V5_DETERMINISTIC_NUMERIC_POLICY_VERSION,
    v5_numeric_policy_payload,
    v5_numeric_policy_sha256,
)
from .split_design_v5 import OOD_LABELS


V5_SOBOL_CLEAN_RECIPE_SCHEMA = "gisaxs.posterior_v8.direct_sobol_clean_recipe/v7"
V5_SOBOL_CLEAN_RECIPE_VERSION = (
    "posterior_v8_v5_2_numeric_contract_bound_direct_sobol_clean_recipe_v7"
)
V5_SOBOL_EXACT_FORWARD_PATH = V5_CLEAN_EXACT_FORWARD_PATH
V5_SOBOL_OOD_MATERIALIZATION_STATUS = (
    "fail_closed_until_concrete_support_and_acquisition_transforms_are_preregistered_v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _digest(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_label(value: str | None, name: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{name} must be None or a non-empty string")
    return value


def _validate_id_split(split: str, ood_label: str | None) -> None:
    if ood_label is not None and ood_label not in OOD_LABELS:
        raise ValueError(f"ood_label must be None or one of {OOD_LABELS}")
    if split == "ood" or ood_label is not None:
        if split != "ood" or ood_label is None:
            raise ValueError("OOD split and ood_label must be supplied together")
        raise ValueError(
            "direct Sobol OOD materialization is fail-closed until concrete support and "
            "acquisition transforms are preregistered"
        )


def v5_sobol_ood_materialization_contract() -> dict[str, object]:
    """Expose the machine-checkable reason ordinary IID points cannot masquerade as OOD."""

    return {
        "status": V5_SOBOL_OOD_MATERIALIZATION_STATUS,
        "labels": list(OOD_LABELS),
        "topology_holdout": "requires frozen held-out and ID topology ID sets",
        "range_width_holdout": "requires frozen disjoint ID/OOD width-transform supports",
        "weak_component_holdout": "requires frozen disjoint particle-weight threshold supports",
        "acquisition_policy_holdout": {
            "owner": "observation_v5",
            "clean_physics_transform": None,
            "requires_held_out_acquisition_policy_certificate": True,
        },
        "ordinary_iid_point_may_be_relabelled_ood": False,
    }


def v5_sobol_physics_payload(physics: V5DirectPhysics) -> dict[str, object]:
    resolution_present = physics.amplitude.resolution_present
    return {
        "direct_physics_version": physics.version,
        "direct_geometry_target_version": physics.geometry_target_version,
        "numeric_policy_version": V5_DETERMINISTIC_NUMERIC_POLICY_VERSION,
        "numeric_policy_contract": v5_numeric_policy_payload(
            V5_DETERMINISTIC_NUMERIC_POLICY_VERSION
        ),
        "numeric_policy_sha256": v5_numeric_policy_sha256(
            V5_DETERMINISTIC_NUMERIC_POLICY_VERSION
        ),
        "canonical_component_slots_version": CANONICAL_COMPONENT_SLOTS_VERSION,
        "local_target_version": physics.target.version,
        "geometry_query": json.loads(physics.query.canonical_json),
        "geometry_query_sha256": physics.query.sha256,
        "amplitude_query": json.loads(physics.amplitude_query.canonical_json),
        "amplitude_query_sha256": physics.amplitude_query.sha256,
        "amplitude_range_regime": physics.amplitude_range_regime,
        "amplitude_range_regimes": physics.amplitude_range_regimes.audit_payload(),
        "amplitude_range_regimes_sha256": physics.amplitude_range_regimes.sha256,
        "branch_pattern_id": physics.target.pattern_id,
        "target_coordinate_transform": (
            "epsilon+(1-2*epsilon)*named_Sobol_coordinate_on_active_branch_axes"
        ),
        "target_seed_field_semantics": (
            "compatibility-only Sobol index; no pseudorandom generator is invoked"
        ),
        "local_target_unit": list(physics.target.local_target_unit),
        "target_numeric_policy_version": physics.target.physical_numeric_policy_version,
        "truth_components": [asdict(value) for value in physics.target.truth_components],
        "truth_resolution": (
            None
            if physics.target.truth_resolution is None
            else asdict(physics.target.truth_resolution)
        ),
        "feasible_amplitude_regimes": list(physics.feasible_amplitude_regimes),
        "amplitude_composition": physics.amplitude.audit_payload(),
        "branch_amplitude_constraint": physics.amplitude_query.constraint_for_branch(
            resolution_present=resolution_present
        ).to_audit_dict(),
    }


def _recipe_payload(
    *,
    sobol_index: int,
    assigned_split: str,
    ood_label: str | None,
    clean_group_id: str,
    sobol_design_sha256: str,
    unit_coordinates: tuple[float, ...],
    physics: V5DirectPhysics,
    grid: GridProvenance,
) -> dict[str, object]:
    return {
        "schema": V5_SOBOL_CLEAN_RECIPE_SCHEMA,
        "version": V5_SOBOL_CLEAN_RECIPE_VERSION,
        "exact_forward_path": V5_SOBOL_EXACT_FORWARD_PATH,
        "source": {
            "sobol_index": sobol_index,
            "assigned_split": assigned_split,
            "ood_label": ood_label,
            "clean_group_id": clean_group_id,
            "sobol_design_sha256": sobol_design_sha256,
            "coordinate_schema": V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
            "coordinate_version": V5_SOBOL_RECIPE_COORDINATE_VERSION,
            "coordinate_contract_sha256": V5_SOBOL_RECIPE_COORDINATE_SHA256,
            "coordinate_consumption": "direct_no_coordinate_derived_seed_or_PRNG",
            "recipe_seed_semantics": "Sobol index used only for downstream observation views",
            "unit_coordinates": list(unit_coordinates),
            "inactive_coordinate_names": list(physics.inactive_coordinate_names),
        },
        "physics": v5_sobol_physics_payload(physics),
        "grid": asdict(grid),
    }


@dataclass(frozen=True)
class V5SobolCleanRecipe:
    """One immutable clean parent generated directly from a named Sobol point."""

    sobol_index: int
    assigned_split: str
    ood_label: str | None
    clean_group_id: str
    sobol_design_sha256: str
    unit_coordinates: tuple[float, ...]
    physics: V5DirectPhysics
    grid: GridProvenance
    canonical_json: str
    sha256: str
    schema_version: str = V5_SOBOL_CLEAN_RECIPE_SCHEMA
    generator_version: str = V5_SOBOL_CLEAN_RECIPE_VERSION

    @classmethod
    def create(
        cls,
        *,
        point: V5DesignPoint,
        design: V5SobolDesign,
        grid: GridProvenance | None = None,
    ) -> "V5SobolCleanRecipe":
        if not isinstance(point, V5DesignPoint):
            raise TypeError("point must be a V5DesignPoint")
        if not isinstance(design, V5SobolDesign):
            raise TypeError("design must be a V5SobolDesign")
        if design.coordinate_names != V5_SOBOL_RECIPE_COORDINATE_NAMES:
            raise ValueError("Sobol design does not use the frozen direct-recipe coordinates")
        if design.coordinate_contract_sha256 != V5_SOBOL_RECIPE_COORDINATE_SHA256:
            raise ValueError("Sobol design is not bound to the frozen coordinate contract hash")
        index = _nonnegative_integer(point.sobol_index, "sobol_index")
        split = _nonempty(point.assigned_split, "assigned_split")
        ood = _optional_label(point.ood_label, "ood_label")
        _validate_id_split(split, ood)
        group = _digest(point.clean_group_id, "clean_group_id")
        coordinates = validate_v5_sobol_recipe_coordinates(point.unit_coordinates)
        selected_grid = GridProvenance(n_points=256) if grid is None else grid
        if not isinstance(selected_grid, GridProvenance):
            raise TypeError("grid must be a GridProvenance")
        physics = direct_v5_physics_from_sobol(coordinates, sobol_index=index)
        payload = _recipe_payload(
            sobol_index=index,
            assigned_split=split,
            ood_label=ood,
            clean_group_id=group,
            sobol_design_sha256=design.sha256,
            unit_coordinates=coordinates,
            physics=physics,
            grid=selected_grid,
        )
        canonical = _canonical_json(payload)
        return cls(
            sobol_index=index,
            assigned_split=split,
            ood_label=ood,
            clean_group_id=group,
            sobol_design_sha256=design.sha256,
            unit_coordinates=coordinates,
            physics=physics,
            grid=selected_grid,
            canonical_json=canonical,
            sha256=sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def __post_init__(self) -> None:
        index = _nonnegative_integer(self.sobol_index, "sobol_index")
        split = _nonempty(self.assigned_split, "assigned_split")
        ood = _optional_label(self.ood_label, "ood_label")
        _validate_id_split(split, ood)
        group = _digest(self.clean_group_id, "clean_group_id")
        design_digest = _digest(self.sobol_design_sha256, "sobol_design_sha256")
        coordinates = validate_v5_sobol_recipe_coordinates(self.unit_coordinates)
        if not isinstance(self.physics, V5DirectPhysics):
            raise TypeError("physics must be V5DirectPhysics")
        if self.physics.version != V5_DIRECT_PHYSICS_VERSION:
            raise ValueError("unsupported direct Sobol physics version")
        replay = direct_v5_physics_from_sobol(coordinates, sobol_index=index)
        if self.physics != replay:
            raise ValueError("direct Sobol physics does not replay from its named coordinates")
        if not isinstance(self.grid, GridProvenance):
            raise TypeError("grid must be GridProvenance")
        canonical = _canonical_json(
            _recipe_payload(
                sobol_index=index,
                assigned_split=split,
                ood_label=ood,
                clean_group_id=group,
                sobol_design_sha256=design_digest,
                unit_coordinates=coordinates,
                physics=replay,
                grid=self.grid,
            )
        )
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        if self.canonical_json != canonical or self.sha256 != digest:
            raise ValueError("direct Sobol recipe does not reproduce its audit hash")
        if self.schema_version != V5_SOBOL_CLEAN_RECIPE_SCHEMA:
            raise ValueError("unsupported direct Sobol clean-recipe schema")
        if self.generator_version != V5_SOBOL_CLEAN_RECIPE_VERSION:
            raise ValueError("unsupported direct Sobol clean-recipe generator")

    @property
    def query(self):
        return self.physics.query

    @property
    def recipe_seed(self) -> int:
        """Return an observation-only seed; clean physics never consumes it as a PRNG seed."""

        return self.sobol_index

    @property
    def amplitude_query(self):
        return self.physics.amplitude_query

    @property
    def target(self):
        return self.physics.target

    @property
    def amplitude(self) -> V5DirectAmplitudeComposition:
        return self.physics.amplitude


def materialize_v5_sobol_clean_recipe(
    point: V5DesignPoint,
    design: V5SobolDesign,
    *,
    grid: GridProvenance | None = None,
) -> V5SobolCleanRecipe:
    """Materialize one design point without any PRNG bridge or rejection loop."""

    return V5SobolCleanRecipe.create(point=point, design=design, grid=grid)


authoritative_v5_sobol_gui_parameters = authoritative_v5_gui_parameters
evaluate_v5_sobol_clean_recipe = evaluate_v5_clean_recipe_forward


__all__ = [
    "V5_SOBOL_CLEAN_RECIPE_SCHEMA",
    "V5_SOBOL_CLEAN_RECIPE_VERSION",
    "V5_SOBOL_EXACT_FORWARD_PATH",
    "V5_SOBOL_OOD_MATERIALIZATION_STATUS",
    "V5SobolCleanRecipe",
    "authoritative_v5_sobol_gui_parameters",
    "evaluate_v5_sobol_clean_recipe",
    "materialize_v5_sobol_clean_recipe",
    "v5_sobol_ood_materialization_contract",
    "v5_sobol_physics_payload",
]
