"""Branch-balanced K1 clean recipes with truthful forced-Sobol provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from .k1_balanced_dataset_plan_v5 import (
    V5K1BalancedDatasetPlan,
    V5K1BalancedSobolBlock,
    validate_v5_k1_balanced_dataset_plan,
)
from .k1_branch_forcing_v5 import (
    V5K1ForcedSobolCoordinates,
    force_v5_k1_branch_coordinates,
)
from .k1_phase_c_contract_v5 import K1_PHASE_C_BRANCH_BY_ID
from .simulation import GridProvenance
from .sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
    V5_SOBOL_RECIPE_COORDINATE_VERSION,
    v5_sobol_recipe_design,
)
from .sobol_recipe_physics_v5 import (
    V5DirectAmplitudeComposition,
    V5DirectPhysics,
    direct_v5_physics_from_sobol,
)
from .sobol_recipe_v5 import V5_SOBOL_EXACT_FORWARD_PATH, v5_sobol_physics_payload
from .sobol_design_v5 import materialize_v5_unit_coordinates_for_indices


V5_K1_FORCED_SOBOL_RECIPE_SCHEMA = (
    "gisaxs.posterior_v8.k1_branch_forced_sobol_clean_recipe/v2"
)
V5_K1_FORCED_SOBOL_RECIPE_VERSION = (
    "posterior_v8_v5_2_original_and_forced_coordinate_bound_k1_recipe_v2"
)
_TOP_LEVEL_FIELDS = {"schema", "version", "exact_forward_path", "source", "physics", "grid"}
_SOURCE_FIELDS = {
    "balanced_dataset_plan_sha256",
    "balanced_sobol_block_sha256",
    "role",
    "assigned_split",
    "generating_branch_id",
    "branch_ordinal",
    "sobol_index",
    "clean_group_id",
    "sobol_design_sha256",
    "coordinate_schema",
    "coordinate_version",
    "coordinate_contract_sha256",
    "original_unit_coordinates",
    "forced_unit_coordinates",
    "branch_forcing",
    "branch_forcing_sha256",
}
_GRID_FIELDS = {"kind", "q_min", "q_max", "n_points"}
_SHA256_ALPHABET = frozenset("0123456789abcdef")


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
        or any(character not in _SHA256_ALPHABET for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _strict_canonical_object(encoded: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate forced-recipe field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(encoded, object_pairs_hook=reject_duplicates)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("forced K1 recipe is not strict JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != encoded:
        raise ValueError("forced K1 recipe must be one canonical JSON object")
    return value


def _clean_group_id(
    plan: V5K1BalancedDatasetPlan,
    block: V5K1BalancedSobolBlock,
    sobol_index: int,
) -> str:
    return _clean_group_id_from_hashes(plan.sha256, block.block_sha256, sobol_index)


def _clean_group_id_from_hashes(
    plan_sha256: str,
    block_sha256: str,
    sobol_index: int,
) -> str:
    return sha256(
        b"\0".join(
            (
                V5_K1_FORCED_SOBOL_RECIPE_VERSION.encode("ascii"),
                plan_sha256.encode("ascii"),
                block_sha256.encode("ascii"),
                str(sobol_index).encode("ascii"),
            )
        )
    ).hexdigest()


def _observation_seed(clean_group_id: str) -> int:
    return int.from_bytes(bytes.fromhex(clean_group_id)[:8], "big")


def _recipe_payload(
    *,
    plan: V5K1BalancedDatasetPlan,
    block: V5K1BalancedSobolBlock,
    sobol_index: int,
    clean_group_id: str,
    forcing: V5K1ForcedSobolCoordinates,
    physics: V5DirectPhysics,
    grid: GridProvenance,
) -> dict[str, object]:
    return {
        "schema": V5_K1_FORCED_SOBOL_RECIPE_SCHEMA,
        "version": V5_K1_FORCED_SOBOL_RECIPE_VERSION,
        "exact_forward_path": V5_SOBOL_EXACT_FORWARD_PATH,
        "source": {
            "balanced_dataset_plan_sha256": plan.sha256,
            "balanced_sobol_block_sha256": block.block_sha256,
            "role": block.role,
            "assigned_split": block.split_id,
            "generating_branch_id": block.branch_id,
            "branch_ordinal": block.branch_ordinal,
            "sobol_index": sobol_index,
            "clean_group_id": clean_group_id,
            "sobol_design_sha256": block.sobol_design_sha256,
            "coordinate_schema": V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
            "coordinate_version": V5_SOBOL_RECIPE_COORDINATE_VERSION,
            "coordinate_contract_sha256": V5_SOBOL_RECIPE_COORDINATE_SHA256,
            "original_unit_coordinates": list(forcing.original_coordinates),
            "forced_unit_coordinates": list(forcing.forced_coordinates),
            "branch_forcing": forcing.audit_payload(),
            "branch_forcing_sha256": forcing.sha256,
        },
        "physics": v5_sobol_physics_payload(physics),
        "grid": asdict(grid),
    }


@dataclass(frozen=True)
class _V5K1ForcedRecipeMaterialization:
    plan: V5K1BalancedDatasetPlan
    block: V5K1BalancedSobolBlock
    sobol_index: int
    clean_group_id: str
    forcing: V5K1ForcedSobolCoordinates
    physics: V5DirectPhysics
    grid: GridProvenance
    canonical_json: str
    sha256: str


def _materialize_recipe_fields(
    *,
    plan: V5K1BalancedDatasetPlan,
    block: V5K1BalancedSobolBlock,
    sobol_index: int,
    original_unit_coordinates: Sequence[float],
    grid: GridProvenance | None,
) -> _V5K1ForcedRecipeMaterialization:
    checked_plan = validate_v5_k1_balanced_dataset_plan(plan)
    if block not in checked_plan.blocks:
        raise ValueError("block is not owned by the balanced dataset plan")
    index = _nonnegative_integer(sobol_index, "sobol_index")
    if not block.sobol_index_start <= index < (
        block.sobol_index_start + block.parent_count
    ):
        raise ValueError("sobol_index is outside the selected branch block")
    design = v5_sobol_recipe_design(scramble_seed=block.scramble_seed)
    if design.sha256 != block.sobol_design_sha256:
        raise ValueError("branch block Sobol design does not reproduce")
    original = tuple(float(value) for value in original_unit_coordinates)
    expected_original = materialize_v5_unit_coordinates_for_indices(design, (index,))[0]
    if original != expected_original:
        raise ValueError("original coordinates do not match the branch Sobol design/index")
    forcing = force_v5_k1_branch_coordinates(
        original,
        branch_id=block.branch_id,
    )
    physics = direct_v5_physics_from_sobol(
        forcing.forced_coordinates,
        sobol_index=index,
    )
    branch = K1_PHASE_C_BRANCH_BY_ID[block.branch_id]
    if (
        physics.query.topology != (branch.shape,)
        or physics.target.pattern_id != branch.pattern_id
        or physics.query.feasible_wire_pattern_ids != (branch.pattern_id,)
    ):
        raise RuntimeError("forced K1 recipe escaped its labelled branch block")
    selected_grid = GridProvenance(n_points=256) if grid is None else grid
    if not isinstance(selected_grid, GridProvenance):
        raise TypeError("grid must be a GridProvenance")
    group = _clean_group_id(checked_plan, block, index)
    payload = _recipe_payload(
        plan=checked_plan,
        block=block,
        sobol_index=index,
        clean_group_id=group,
        forcing=forcing,
        physics=physics,
        grid=selected_grid,
    )
    encoded = _canonical_json(payload)
    return _V5K1ForcedRecipeMaterialization(
        plan=checked_plan,
        block=block,
        sobol_index=index,
        clean_group_id=group,
        forcing=forcing,
        physics=physics,
        grid=selected_grid,
        canonical_json=encoded,
        sha256=sha256(encoded.encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True)
class V5K1ForcedSobolCleanRecipe:
    """One clean K1 parent with both raw and branch-forced Sobol evidence."""

    plan: V5K1BalancedDatasetPlan
    block: V5K1BalancedSobolBlock
    sobol_index: int
    clean_group_id: str
    forcing: V5K1ForcedSobolCoordinates
    physics: V5DirectPhysics
    grid: GridProvenance
    canonical_json: str
    sha256: str
    schema_version: str = V5_K1_FORCED_SOBOL_RECIPE_SCHEMA
    generator_version: str = V5_K1_FORCED_SOBOL_RECIPE_VERSION

    @classmethod
    def create(
        cls,
        *,
        plan: V5K1BalancedDatasetPlan,
        block: V5K1BalancedSobolBlock,
        sobol_index: int,
        original_unit_coordinates: Sequence[float],
        grid: GridProvenance | None = None,
    ) -> "V5K1ForcedSobolCleanRecipe":
        materialized = _materialize_recipe_fields(
            plan=plan,
            block=block,
            sobol_index=sobol_index,
            original_unit_coordinates=original_unit_coordinates,
            grid=grid,
        )
        return cls(
            plan=materialized.plan,
            block=materialized.block,
            sobol_index=materialized.sobol_index,
            clean_group_id=materialized.clean_group_id,
            forcing=materialized.forcing,
            physics=materialized.physics,
            grid=materialized.grid,
            canonical_json=materialized.canonical_json,
            sha256=materialized.sha256,
        )

    def __post_init__(self) -> None:
        replay = _materialize_recipe_fields(
            plan=self.plan,
            block=self.block,
            sobol_index=self.sobol_index,
            original_unit_coordinates=self.forcing.original_coordinates,
            grid=self.grid,
        )
        expected = (
            replay.clean_group_id,
            replay.forcing,
            replay.physics,
            replay.canonical_json,
            replay.sha256,
            V5_K1_FORCED_SOBOL_RECIPE_SCHEMA,
            V5_K1_FORCED_SOBOL_RECIPE_VERSION,
        )
        actual = (
            self.clean_group_id,
            self.forcing,
            self.physics,
            self.canonical_json,
            self.sha256,
            self.schema_version,
            self.generator_version,
        )
        if actual != expected:
            raise ValueError("forced K1 clean recipe identity does not reproduce")

    @property
    def recipe_seed(self) -> int:
        """Return a split-and-branch-specific seed used only for observation views."""

        return _observation_seed(self.clean_group_id)

    @property
    def assigned_split(self) -> str:
        return self.block.split_id

    @property
    def sobol_design_sha256(self) -> str:
        return self.block.sobol_design_sha256

    @property
    def query(self):
        return self.physics.query

    @property
    def amplitude_query(self):
        return self.physics.amplitude_query

    @property
    def target(self):
        return self.physics.target

    @property
    def amplitude(self) -> V5DirectAmplitudeComposition:
        return self.physics.amplitude


@dataclass(frozen=True)
class V5K1ForcedRecipeIdentity:
    recipe_sha256: str
    balanced_dataset_plan_sha256: str
    balanced_sobol_block_sha256: str
    role: str
    split_id: str
    branch_id: str
    branch_ordinal: int
    sobol_index: int
    clean_group_id: str
    sobol_design_sha256: str


@dataclass(frozen=True)
class _V5K1PersistedRecipeFields:
    identity: V5K1ForcedRecipeIdentity
    physics: V5DirectPhysics
    grid: GridProvenance


def _decode_v5_k1_persisted_recipe_fields(
    encoded: str,
    *,
    expected_sha256: str | None = None,
) -> _V5K1PersistedRecipeFields:
    """Strictly replay the complete persisted forced-recipe payload."""

    payload = _strict_canonical_object(encoded)
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise ValueError("forced K1 recipe fields are incomplete or unsupported")
    if (
        payload["schema"] != V5_K1_FORCED_SOBOL_RECIPE_SCHEMA
        or payload["version"] != V5_K1_FORCED_SOBOL_RECIPE_VERSION
        or payload["exact_forward_path"] != V5_SOBOL_EXACT_FORWARD_PATH
    ):
        raise ValueError("forced K1 recipe schema/version or forward path drifted")
    recipe_sha = sha256(encoded.encode("utf-8")).hexdigest()
    if expected_sha256 is not None and recipe_sha != _digest(
        expected_sha256, "expected_sha256"
    ):
        raise ValueError("forced K1 recipe JSON/SHA-256 does not reproduce")

    source = payload["source"]
    if not isinstance(source, Mapping) or set(source) != _SOURCE_FIELDS:
        raise ValueError("forced K1 recipe source fields are incomplete or unsupported")
    if (
        source["coordinate_schema"] != V5_SOBOL_RECIPE_COORDINATE_SCHEMA
        or source["coordinate_version"] != V5_SOBOL_RECIPE_COORDINATE_VERSION
        or source["coordinate_contract_sha256"] != V5_SOBOL_RECIPE_COORDINATE_SHA256
    ):
        raise ValueError("forced K1 recipe coordinate contract drifted")
    plan_sha = _digest(
        source["balanced_dataset_plan_sha256"],
        "balanced_dataset_plan_sha256",
    )
    block_sha = _digest(
        source["balanced_sobol_block_sha256"],
        "balanced_sobol_block_sha256",
    )
    design_sha = _digest(source["sobol_design_sha256"], "sobol_design_sha256")
    role = source["role"]
    split_id = source["assigned_split"]
    expected_split = {"train": "train", "tuning_validation": "tuning_validation"}.get(
        role
    )
    if expected_split is None or split_id != expected_split:
        raise ValueError("forced K1 recipe role/split binding is invalid")
    branch_id = source["generating_branch_id"]
    if not isinstance(branch_id, str) or branch_id not in K1_PHASE_C_BRANCH_BY_ID:
        raise ValueError("forced K1 recipe branch is outside the canonical K1 catalog")
    branch_ordinal = _nonnegative_integer(source["branch_ordinal"], "branch_ordinal")
    branches = tuple(K1_PHASE_C_BRANCH_BY_ID)
    if branch_ordinal >= len(branches) or branches[branch_ordinal] != branch_id:
        raise ValueError("forced K1 recipe branch ordinal does not reproduce")
    sobol_index = _nonnegative_integer(source["sobol_index"], "sobol_index")
    forcing = force_v5_k1_branch_coordinates(
        source["original_unit_coordinates"],
        branch_id=branch_id,
    )
    if (
        list(forcing.forced_coordinates) != source["forced_unit_coordinates"]
        or forcing.audit_payload() != source["branch_forcing"]
        or forcing.sha256 != _digest(
            source["branch_forcing_sha256"], "branch_forcing_sha256"
        )
    ):
        raise ValueError("forced K1 recipe branch transform does not reproduce")
    physics = direct_v5_physics_from_sobol(
        forcing.forced_coordinates,
        sobol_index=sobol_index,
    )
    if v5_sobol_physics_payload(physics) != payload["physics"]:
        raise ValueError("forced K1 recipe physics does not reproduce")
    grid_payload = payload["grid"]
    if not isinstance(grid_payload, Mapping) or set(grid_payload) != _GRID_FIELDS:
        raise ValueError("forced K1 recipe grid fields are unsupported")
    try:
        grid = GridProvenance(**grid_payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("forced K1 recipe grid is invalid") from exc
    if asdict(grid) != dict(grid_payload):
        raise ValueError("forced K1 recipe grid does not reproduce")
    clean_group_id = _digest(source["clean_group_id"], "clean_group_id")
    if clean_group_id != _clean_group_id_from_hashes(
        plan_sha,
        block_sha,
        sobol_index,
    ):
        raise ValueError("forced K1 recipe clean-group identity does not reproduce")
    return _V5K1PersistedRecipeFields(
        identity=V5K1ForcedRecipeIdentity(
            recipe_sha256=recipe_sha,
            balanced_dataset_plan_sha256=plan_sha,
            balanced_sobol_block_sha256=block_sha,
            role=role,
            split_id=split_id,
            branch_id=branch_id,
            branch_ordinal=branch_ordinal,
            sobol_index=sobol_index,
            clean_group_id=clean_group_id,
            sobol_design_sha256=design_sha,
        ),
        physics=physics,
        grid=grid,
    )


@dataclass(frozen=True)
class V5K1PersistedForcedCleanRecipe:
    """Artifact-authoritative forced recipe reconstructed without its authoring plan."""

    identity: V5K1ForcedRecipeIdentity
    physics: V5DirectPhysics
    grid: GridProvenance
    canonical_json: str
    sha256: str
    schema_version: str = V5_K1_FORCED_SOBOL_RECIPE_SCHEMA
    generator_version: str = V5_K1_FORCED_SOBOL_RECIPE_VERSION

    def __post_init__(self) -> None:
        replay = _decode_v5_k1_persisted_recipe_fields(
            self.canonical_json,
            expected_sha256=self.sha256,
        )
        if (
            self.identity,
            self.physics,
            self.grid,
            self.schema_version,
            self.generator_version,
        ) != (
            replay.identity,
            replay.physics,
            replay.grid,
            V5_K1_FORCED_SOBOL_RECIPE_SCHEMA,
            V5_K1_FORCED_SOBOL_RECIPE_VERSION,
        ):
            raise ValueError("persisted forced K1 clean recipe does not reproduce")

    @property
    def recipe_seed(self) -> int:
        return _observation_seed(self.identity.clean_group_id)

    @property
    def assigned_split(self) -> str:
        return self.identity.split_id

    @property
    def clean_group_id(self) -> str:
        return self.identity.clean_group_id

    @property
    def sobol_index(self) -> int:
        return self.identity.sobol_index

    @property
    def sobol_design_sha256(self) -> str:
        return self.identity.sobol_design_sha256

    @property
    def query(self):
        return self.physics.query

    @property
    def amplitude_query(self):
        return self.physics.amplitude_query

    @property
    def target(self):
        return self.physics.target

    @property
    def amplitude(self) -> V5DirectAmplitudeComposition:
        return self.physics.amplitude


def decode_v5_k1_forced_recipe_identity(
    encoded: str,
    *,
    expected_sha256: str | None = None,
) -> V5K1ForcedRecipeIdentity:
    """Strictly replay a persisted forced recipe and return its bound identity."""

    return _decode_v5_k1_persisted_recipe_fields(
        encoded,
        expected_sha256=expected_sha256,
    ).identity


def persisted_v5_k1_forced_clean_recipe_from_json(
    encoded: str,
    expected_sha256: str,
) -> V5K1PersistedForcedCleanRecipe:
    """Recover a complete artifact-bound forced recipe for worker-side replay."""

    replay = _decode_v5_k1_persisted_recipe_fields(
        encoded,
        expected_sha256=expected_sha256,
    )
    return V5K1PersistedForcedCleanRecipe(
        identity=replay.identity,
        physics=replay.physics,
        grid=replay.grid,
        canonical_json=encoded,
        sha256=replay.identity.recipe_sha256,
    )


__all__ = [
    "V5_K1_FORCED_SOBOL_RECIPE_SCHEMA",
    "V5_K1_FORCED_SOBOL_RECIPE_VERSION",
    "V5K1ForcedRecipeIdentity",
    "V5K1ForcedSobolCleanRecipe",
    "V5K1PersistedForcedCleanRecipe",
    "decode_v5_k1_forced_recipe_identity",
    "persisted_v5_k1_forced_clean_recipe_from_json",
]
