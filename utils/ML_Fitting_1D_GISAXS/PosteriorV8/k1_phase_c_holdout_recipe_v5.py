"""Persisted clean-parent identities for the formal K1 Phase-C holdout."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from .k1_branch_forcing_v5 import force_v5_k1_branch_coordinates
from .k1_phase_c_contract_v5 import (
    K1_PHASE_C_BRANCH_BY_ID,
    K1_PHASE_C_OBSERVATION_EFFECTS,
    K1_PHASE_C_SPLIT_ID,
    v5_k1_phase_c_contract_payload,
)
from .k1_phase_c_plan_v5 import (
    V5K1PhaseCPlan,
    V5K1PhaseCSobolBlock,
    planned_v5_k1_phase_c_stress_cell,
    validate_v5_k1_phase_c_plan,
)
from .k1_phase_c_stress_v5 import (
    V5K1PhaseCObservationDesign,
    V5K1PhaseCStressCoordinates,
    build_v5_k1_phase_c_observation_design,
    force_v5_k1_phase_c_range_stress,
    validate_v5_k1_phase_c_stress_semantics,
)
from .simulation import GridProvenance
from .sobol_design_v5 import (
    V5_SOBOL_DESIGN_SCHEMA,
    V5_SOBOL_DESIGN_VERSION,
    materialize_v5_unit_coordinates_for_indices,
)
from .sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
    V5_SOBOL_RECIPE_COORDINATE_VERSION,
    v5_sobol_recipe_design,
)
from .sobol_recipe_physics_v5 import V5DirectPhysics, direct_v5_physics_from_sobol
from .sobol_recipe_v5 import V5_SOBOL_EXACT_FORWARD_PATH, v5_sobol_physics_payload


V5_K1_PHASE_C_HOLDOUT_RECIPE_SCHEMA = (
    "gisaxs.posterior_v8.k1_phase_c_holdout_clean_recipe/v1"
)
V5_K1_PHASE_C_HOLDOUT_RECIPE_VERSION = (
    "posterior_v8_v5_2_phase_c_branch_range_observation_bound_parent_v1"
)
_TOP_LEVEL_FIELDS = {
    "schema",
    "version",
    "exact_forward_path",
    "source",
    "physics",
    "grid",
    "observation_design",
}
_SOURCE_FIELDS = {
    "phase_c_contract_sha256",
    "phase_c_plan_sha256",
    "phase_c_sobol_block_sha256",
    "role",
    "assigned_split",
    "generating_branch_id",
    "branch_ordinal",
    "sobol_index",
    "clean_group_id",
    "sobol_design_sha256",
    "scramble_seed",
    "coordinate_schema",
    "coordinate_version",
    "coordinate_contract_sha256",
    "original_unit_coordinates",
    "branch_forcing",
    "branch_forcing_sha256",
    "range_stress",
    "range_stress_sha256",
    "stressed_unit_coordinates",
    "range_stress_stratum",
    "observation_stress_stratum",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _strict_object(encoded: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate Phase-C recipe field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(encoded, object_pairs_hook=reject_duplicates)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Phase-C holdout recipe is not strict JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != encoded:
        raise ValueError("Phase-C holdout recipe must be one canonical JSON object")
    return value


def _clean_group_id(plan_sha: str, block_sha: str, sobol_index: int) -> str:
    return sha256(
        b"\0".join(
            (
                V5_K1_PHASE_C_HOLDOUT_RECIPE_VERSION.encode("ascii"),
                plan_sha.encode("ascii"),
                block_sha.encode("ascii"),
                str(sobol_index).encode("ascii"),
            )
        )
    ).hexdigest()


def _phase_c_design_sha256(scramble_seed: int) -> str:
    """Replay the deliberately SciPy-version-independent Phase-C design identity."""

    return sha256(
        _canonical_json(
            {
                "sobol_schema": V5_SOBOL_DESIGN_SCHEMA,
                "sobol_version": V5_SOBOL_DESIGN_VERSION,
                "coordinate_contract_sha256": V5_SOBOL_RECIPE_COORDINATE_SHA256,
                "scramble_seed": scramble_seed,
                "bits": 52,
            }
        ).encode()
    ).hexdigest()


def _physics_stress_axes(
    physics: V5DirectPhysics,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    designs = physics.query.axis_designs
    geometry_regimes = tuple(value.regime for value in designs)
    geometry_placements = tuple(value.placement for value in designs)
    amplitude_regimes = tuple(
        value
        for value in physics.amplitude_range_regimes.axis_regimes
        if value is not None
    )
    return geometry_regimes, geometry_placements, amplitude_regimes


@dataclass(frozen=True)
class V5K1PhaseCHoldoutCleanRecipe:
    plan: V5K1PhaseCPlan
    block: V5K1PhaseCSobolBlock
    sobol_index: int
    clean_group_id: str
    branch_forcing_sha256: str
    stress: V5K1PhaseCStressCoordinates
    observation_design: V5K1PhaseCObservationDesign
    physics: V5DirectPhysics
    grid: GridProvenance
    canonical_json: str
    sha256: str
    schema_version: str = V5_K1_PHASE_C_HOLDOUT_RECIPE_SCHEMA
    generator_version: str = V5_K1_PHASE_C_HOLDOUT_RECIPE_VERSION

    @property
    def recipe_seed(self) -> int:
        return int.from_bytes(bytes.fromhex(self.clean_group_id)[:8], "big")

    @property
    def assigned_split(self) -> str:
        return K1_PHASE_C_SPLIT_ID


@dataclass(frozen=True)
class V5K1PhaseCHoldoutRecipeIdentity:
    recipe_sha256: str
    phase_c_contract_sha256: str
    phase_c_plan_sha256: str
    phase_c_sobol_block_sha256: str
    split_id: str
    branch_id: str
    branch_ordinal: int
    sobol_index: int
    clean_group_id: str
    sobol_design_sha256: str
    range_stress_stratum: str
    observation_stress_stratum: str
    observation_design_sha256: str


def _materialize(
    *,
    plan: V5K1PhaseCPlan,
    block: V5K1PhaseCSobolBlock,
    sobol_index: int,
    original_unit_coordinates: Sequence[float],
) -> V5K1PhaseCHoldoutCleanRecipe:
    checked = validate_v5_k1_phase_c_plan(plan)
    if not checked.formal or block not in checked.sobol_blocks:
        raise ValueError("formal Phase-C block must belong to its plan")
    index = _nonnegative_integer(sobol_index, "sobol_index")
    if not block.sobol_index_start <= index < block.sobol_index_start + block.parent_count:
        raise ValueError("sobol_index is outside its Phase-C block")
    design = v5_sobol_recipe_design(scramble_seed=block.scramble_seed)
    if _phase_c_design_sha256(block.scramble_seed) != block.sobol_design_sha256:
        raise ValueError("Phase-C block Sobol design does not reproduce")
    original = tuple(float(value) for value in original_unit_coordinates)
    expected = materialize_v5_unit_coordinates_for_indices(design, (index,))[0]
    if original != expected:
        raise ValueError("Phase-C original coordinates do not match design/index")
    branch_forcing = force_v5_k1_branch_coordinates(original, branch_id=block.branch_id)
    range_stratum, observation_stratum = planned_v5_k1_phase_c_stress_cell(
        checked,
        branch_id=block.branch_id,
        sobol_index=index,
    )
    stress = force_v5_k1_phase_c_range_stress(
        branch_forcing,
        range_stress_stratum=range_stratum,
    )
    physics = direct_v5_physics_from_sobol(stress.stressed_coordinates, sobol_index=index)
    branch = K1_PHASE_C_BRANCH_BY_ID[block.branch_id]
    if (
        physics.query.topology != (branch.shape,)
        or physics.target.pattern_id != branch.pattern_id
        or physics.query.feasible_wire_pattern_ids != (branch.pattern_id,)
    ):
        raise RuntimeError("Phase-C holdout recipe escaped its branch block")
    geometry_regimes, geometry_placements, amplitude_regimes = _physics_stress_axes(physics)
    validate_v5_k1_phase_c_stress_semantics(
        range_stress_stratum=range_stratum,
        geometry_axis_regimes=geometry_regimes,
        geometry_axis_placements=geometry_placements,
        amplitude_axis_regimes=amplitude_regimes,
    )
    group = _clean_group_id(checked.sha256, block.block_sha256, index)
    observation = build_v5_k1_phase_c_observation_design(
        clean_group_id=group,
        observation_stress_stratum=observation_stratum,
    )
    grid = observation.grid
    contract_sha = v5_k1_phase_c_contract_payload()["contract_sha256"]
    source = {
        "phase_c_contract_sha256": contract_sha,
        "phase_c_plan_sha256": checked.sha256,
        "phase_c_sobol_block_sha256": block.block_sha256,
        "role": "phase_c_holdout",
        "assigned_split": K1_PHASE_C_SPLIT_ID,
        "generating_branch_id": block.branch_id,
        "branch_ordinal": block.branch_ordinal,
        "sobol_index": index,
        "clean_group_id": group,
        "sobol_design_sha256": block.sobol_design_sha256,
        "scramble_seed": block.scramble_seed,
        "coordinate_schema": V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
        "coordinate_version": V5_SOBOL_RECIPE_COORDINATE_VERSION,
        "coordinate_contract_sha256": V5_SOBOL_RECIPE_COORDINATE_SHA256,
        "original_unit_coordinates": list(original),
        "branch_forcing": branch_forcing.audit_payload(),
        "branch_forcing_sha256": branch_forcing.sha256,
        "range_stress": stress.audit_payload(),
        "range_stress_sha256": stress.sha256,
        "stressed_unit_coordinates": list(stress.stressed_coordinates),
        "range_stress_stratum": range_stratum,
        "observation_stress_stratum": observation_stratum,
    }
    payload = {
        "schema": V5_K1_PHASE_C_HOLDOUT_RECIPE_SCHEMA,
        "version": V5_K1_PHASE_C_HOLDOUT_RECIPE_VERSION,
        "exact_forward_path": V5_SOBOL_EXACT_FORWARD_PATH,
        "source": source,
        "physics": v5_sobol_physics_payload(physics),
        "grid": asdict(grid),
        "observation_design": observation.audit_payload(),
    }
    encoded = _canonical_json(payload)
    return V5K1PhaseCHoldoutCleanRecipe(
        plan=checked,
        block=block,
        sobol_index=index,
        clean_group_id=group,
        branch_forcing_sha256=branch_forcing.sha256,
        stress=stress,
        observation_design=observation,
        physics=physics,
        grid=grid,
        canonical_json=encoded,
        sha256=sha256(encoded.encode()).hexdigest(),
    )


def materialize_v5_k1_phase_c_holdout_recipe(
    *,
    plan: V5K1PhaseCPlan,
    block: V5K1PhaseCSobolBlock,
    sobol_index: int,
    original_unit_coordinates: Sequence[float],
) -> V5K1PhaseCHoldoutCleanRecipe:
    return _materialize(
        plan=plan,
        block=block,
        sobol_index=sobol_index,
        original_unit_coordinates=original_unit_coordinates,
    )


def decode_v5_k1_phase_c_holdout_recipe_identity(
    encoded: str,
    *,
    expected_sha256: str | None = None,
) -> V5K1PhaseCHoldoutRecipeIdentity:
    """Replay every transform in one persisted Phase-C recipe."""

    payload = _strict_object(encoded)
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise ValueError("Phase-C holdout recipe fields are unsupported")
    if (
        payload["schema"] != V5_K1_PHASE_C_HOLDOUT_RECIPE_SCHEMA
        or payload["version"] != V5_K1_PHASE_C_HOLDOUT_RECIPE_VERSION
        or payload["exact_forward_path"] != V5_SOBOL_EXACT_FORWARD_PATH
    ):
        raise ValueError("Phase-C holdout recipe schema/version drifted")
    recipe_sha = sha256(encoded.encode()).hexdigest()
    if expected_sha256 is not None and recipe_sha != _digest(
        expected_sha256, "expected_sha256"
    ):
        raise ValueError("Phase-C holdout recipe SHA-256 does not reproduce")
    source = payload["source"]
    if not isinstance(source, Mapping) or set(source) != _SOURCE_FIELDS:
        raise ValueError("Phase-C holdout source fields are unsupported")
    if (
        source["role"] != "phase_c_holdout"
        or source["assigned_split"] != K1_PHASE_C_SPLIT_ID
        or source["coordinate_schema"] != V5_SOBOL_RECIPE_COORDINATE_SCHEMA
        or source["coordinate_version"] != V5_SOBOL_RECIPE_COORDINATE_VERSION
        or source["coordinate_contract_sha256"] != V5_SOBOL_RECIPE_COORDINATE_SHA256
    ):
        raise ValueError("Phase-C holdout source contract drifted")
    contract_sha = _digest(source["phase_c_contract_sha256"], "phase_c_contract_sha256")
    if contract_sha != v5_k1_phase_c_contract_payload()["contract_sha256"]:
        raise ValueError("Phase-C contract SHA-256 drifted")
    plan_sha = _digest(source["phase_c_plan_sha256"], "phase_c_plan_sha256")
    block_sha = _digest(source["phase_c_sobol_block_sha256"], "block_sha256")
    design_sha = _digest(source["sobol_design_sha256"], "sobol_design_sha256")
    branch_id = source["generating_branch_id"]
    if not isinstance(branch_id, str) or branch_id not in K1_PHASE_C_BRANCH_BY_ID:
        raise ValueError("Phase-C branch is outside the canonical K1 catalog")
    branch_ordinal = _nonnegative_integer(source["branch_ordinal"], "branch_ordinal")
    if tuple(K1_PHASE_C_BRANCH_BY_ID)[branch_ordinal] != branch_id:
        raise ValueError("Phase-C branch ordinal does not reproduce")
    index = _nonnegative_integer(source["sobol_index"], "sobol_index")
    scramble_seed = _nonnegative_integer(source["scramble_seed"], "scramble_seed")
    design = v5_sobol_recipe_design(scramble_seed=scramble_seed)
    if _phase_c_design_sha256(scramble_seed) != design_sha:
        raise ValueError("Phase-C Sobol design does not reproduce")
    original = materialize_v5_unit_coordinates_for_indices(design, (index,))[0]
    if list(original) != source["original_unit_coordinates"]:
        raise ValueError("Phase-C original Sobol point does not reproduce")
    branch_forcing = force_v5_k1_branch_coordinates(original, branch_id=branch_id)
    if (
        branch_forcing.audit_payload() != source["branch_forcing"]
        or branch_forcing.sha256 != _digest(
            source["branch_forcing_sha256"], "branch_forcing_sha256"
        )
    ):
        raise ValueError("Phase-C branch forcing does not reproduce")
    stress = force_v5_k1_phase_c_range_stress(
        branch_forcing,
        range_stress_stratum=source["range_stress_stratum"],
    )
    if (
        stress.audit_payload() != source["range_stress"]
        or stress.sha256 != _digest(source["range_stress_sha256"], "range_stress_sha256")
        or list(stress.stressed_coordinates) != source["stressed_unit_coordinates"]
    ):
        raise ValueError("Phase-C range stress does not reproduce")
    physics = direct_v5_physics_from_sobol(stress.stressed_coordinates, sobol_index=index)
    if v5_sobol_physics_payload(physics) != payload["physics"]:
        raise ValueError("Phase-C holdout physics does not reproduce")
    group = _digest(source["clean_group_id"], "clean_group_id")
    if group != _clean_group_id(plan_sha, block_sha, index):
        raise ValueError("Phase-C clean-group identity does not reproduce")
    observation = build_v5_k1_phase_c_observation_design(
        clean_group_id=group,
        observation_stress_stratum=source["observation_stress_stratum"],
    )
    if observation.audit_payload() != payload["observation_design"]:
        raise ValueError("Phase-C observation design does not reproduce")
    if asdict(observation.grid) != payload["grid"]:
        raise ValueError("Phase-C clean grid does not reproduce")
    if list(K1_PHASE_C_OBSERVATION_EFFECTS[observation.stress_stratum]) != payload[
        "observation_design"
    ]["effects"]:
        raise ValueError("Phase-C observation effects drifted")
    geometry_regimes, geometry_placements, amplitude_regimes = _physics_stress_axes(physics)
    validate_v5_k1_phase_c_stress_semantics(
        range_stress_stratum=stress.range_stress_stratum,
        geometry_axis_regimes=geometry_regimes,
        geometry_axis_placements=geometry_placements,
        amplitude_axis_regimes=amplitude_regimes,
    )
    return V5K1PhaseCHoldoutRecipeIdentity(
        recipe_sha256=recipe_sha,
        phase_c_contract_sha256=contract_sha,
        phase_c_plan_sha256=plan_sha,
        phase_c_sobol_block_sha256=block_sha,
        split_id=source["assigned_split"],
        branch_id=branch_id,
        branch_ordinal=branch_ordinal,
        sobol_index=index,
        clean_group_id=group,
        sobol_design_sha256=design_sha,
        range_stress_stratum=stress.range_stress_stratum,
        observation_stress_stratum=observation.stress_stratum,
        observation_design_sha256=observation.sha256,
    )


__all__ = [
    "V5_K1_PHASE_C_HOLDOUT_RECIPE_SCHEMA",
    "V5_K1_PHASE_C_HOLDOUT_RECIPE_VERSION",
    "V5K1PhaseCHoldoutCleanRecipe",
    "V5K1PhaseCHoldoutRecipeIdentity",
    "decode_v5_k1_phase_c_holdout_recipe_identity",
    "materialize_v5_k1_phase_c_holdout_recipe",
]
