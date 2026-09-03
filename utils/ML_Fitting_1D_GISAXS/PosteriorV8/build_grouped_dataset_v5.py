"""Deterministic builder for compact V5.1 solution-stage dataset shards."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from .candidate_batch_v5 import build_v5_candidate_context_batch
from .candidate_supervision_v5 import (
    V5CandidateSupervision,
    stack_candidate_supervision_v5,
)
from .clean_recipe_forward_v5 import (
    V5_CLEAN_RECIPE_PROTOCOL_VERSION,
    V5CleanRecipeLike,
    validate_v5_clean_recipe_like,
)
from .grouped_artifact_v5 import array_sha256, canonical_json
from .grouped_dataset_v5 import (
    V5GroupedDataset,
    build_grouped_manifest,
    candidate_context_array,
    candidate_input,
    candidate_label,
    clean_array,
    observation_array,
    observation_input,
)
from .grouped_known_truth_oracle_v5 import (
    V5_ORACLE_EXACT_ARTIFACT_SCHEMA,
    V5_ORACLE_PROTOCOL_ID,
    V5_ORACLE_PROTOCOL_PAYLOAD,
    V5_ORACLE_PROTOCOL_SHA256,
    V5_ORACLE_SEARCH_ARTIFACT_SCHEMA,
    run_v5_known_truth_exact_oracle,
)
from .model_v5_contract import MODEL_V5_INPUT_KEYS
from .observation_v5 import build_v5_observation_data_views
from .synthetic_recipe_v5 import sample_v5_clean_recipe
from .universal_query_contract_v5 import V5TopologyQuery

_OBSERVATION_MODEL_INPUTS = {
    "x",
    "point_mask",
    "global_features",
    "uncertainty_provenance",
}
_DERIVED_JOIN_MODEL_INPUTS = {"amplitude_bounds_embedding"}


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _digest(value: str, name: str) -> str:
    result = _nonempty(value, name)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def _uint64(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= (1 << 64) - 1:
        raise ValueError(f"{name} must fit in uint64")
    return result


class V5CleanDesignPoint(Protocol):
    sobol_index: int
    assigned_split: str
    clean_group_id: str


@dataclass(frozen=True, kw_only=True)
class V5GroupedRecipeSpec:
    recipe: V5CleanRecipeLike
    split_id: str
    view_indices: tuple[int, ...] = (0,)
    clean_group_id: str | None = None
    sobol_index: int | None = None
    split_plan_sha256: str | None = None
    sobol_design_sha256: str | None = None

    def __post_init__(self) -> None:
        recipe = validate_v5_clean_recipe_like(self.recipe)
        _recipe_identity(recipe)
        split = _nonempty(self.split_id, "split_id")
        views = tuple(_uint64(value, "view_index") for value in self.view_indices)
        if not views or len(set(views)) != len(views):
            raise ValueError("view_indices must be non-empty and unique")
        group = self.recipe.sha256 if self.clean_group_id is None else self.clean_group_id
        group = _digest(group, "clean_group_id")
        parent_values = (self.sobol_index, self.split_plan_sha256, self.sobol_design_sha256)
        if all(value is None for value in parent_values):
            sobol_index, split_hash, design_hash = None, None, None
        elif any(value is None for value in parent_values):
            raise ValueError(
                "Sobol parent index, split hash, and design hash must be supplied together"
            )
        else:
            sobol_index = _uint64(self.sobol_index, "sobol_index")
            split_hash = _digest(self.split_plan_sha256, "split_plan_sha256")
            design_hash = _digest(self.sobol_design_sha256, "sobol_design_sha256")
        recipe_sobol_index = getattr(recipe, "sobol_index", None)
        if recipe_sobol_index is not None:
            if sobol_index is None:
                raise ValueError("direct Sobol recipes require complete Sobol parent provenance")
            if _uint64(recipe_sobol_index, "recipe.sobol_index") != sobol_index:
                raise ValueError("recipe and grouped spec use different Sobol indices")
            if getattr(recipe, "assigned_split", split) != split:
                raise ValueError("recipe and grouped spec use different assigned splits")
            if getattr(recipe, "clean_group_id", group) != group:
                raise ValueError("recipe and grouped spec use different clean group IDs")
            if getattr(recipe, "sobol_design_sha256", design_hash) != design_hash:
                raise ValueError("recipe and grouped spec use different Sobol designs")
        object.__setattr__(self, "split_id", split)
        object.__setattr__(self, "view_indices", views)
        object.__setattr__(self, "clean_group_id", group)
        object.__setattr__(self, "sobol_index", sobol_index)
        object.__setattr__(self, "split_plan_sha256", split_hash)
        object.__setattr__(self, "sobol_design_sha256", design_hash)

    @classmethod
    def from_design_point(
        cls,
        recipe: V5CleanRecipeLike,
        point: V5CleanDesignPoint,
        *,
        split_plan_sha256: str,
        sobol_design_sha256: str,
        view_indices: Sequence[int] = (0,),
    ) -> "V5GroupedRecipeSpec":
        """Attach stable Sobol parent identity without owning split generation."""

        return cls(
            recipe=recipe,
            split_id=point.assigned_split,
            view_indices=tuple(view_indices),
            clean_group_id=point.clean_group_id,
            sobol_index=point.sobol_index,
            split_plan_sha256=split_plan_sha256,
            sobol_design_sha256=sobol_design_sha256,
        )


def _recipe_identity(recipe: V5CleanRecipeLike) -> tuple[str, str]:
    """Return the persisted schema/generator identity of one clean recipe."""

    schema = _nonempty(getattr(recipe, "schema_version", None), "recipe.schema_version")
    generator = _nonempty(
        getattr(recipe, "generator_version", None),
        "recipe.generator_version",
    )
    return schema, generator


def _source_hashes() -> dict[str, str]:
    posterior_root = Path(__file__).resolve().parent
    repository_root = posterior_root.parents[2]
    # Cover direct samplers and observation primitives as well as wrappers;
    # the authoritative empirical GUI forward lives outside this bundle.
    sources = sorted(posterior_root.glob("*.py"))
    sources.append(repository_root / "src/gimap/features/fitting/domain/scattering_model.py")
    sources.append(repository_root / "src/gimap/features/fitting/domain/physical_constraints.py")
    return {
        path.relative_to(repository_root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sources
    }


def _observation_id(recipe_id: str, view) -> str:
    payload = {
        "recipe_id": recipe_id,
        "view_index": view.view_index,
        "acquisition_policy_id": view.acquisition_policy_id,
        "x_sha256": array_sha256("observation_x", view.preprocessed.x),
        "point_mask_sha256": array_sha256("observation_point_mask", view.preprocessed.point_mask),
        "global_features_sha256": array_sha256(
            "observation_global_features", view.preprocessed.global_features
        ),
    }
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _append(mapping: dict[str, list[object]], name: str, value: object) -> None:
    mapping.setdefault(name, []).append(value)


def _table_arrays(
    values: dict[str, list[object]], dtypes: dict[str, object]
) -> dict[str, np.ndarray]:
    return {name: np.asarray(items, dtype=dtypes.get(name)) for name, items in values.items()}


def build_v5_grouped_solution_dataset(
    specs: Sequence[V5GroupedRecipeSpec],
    *,
    dataset_id: str,
    generating_only: bool = False,
    shard_selection: Mapping[str, object] | None = None,
) -> V5GroupedDataset:
    """Build a replayable solution-stage shard without inventing negatives."""

    items = tuple(specs)
    if not items or not all(isinstance(value, V5GroupedRecipeSpec) for value in items):
        raise ValueError("specs must contain V5GroupedRecipeSpec values")
    if type(generating_only) is not bool:
        raise TypeError("generating_only must be a bool")
    if len({value.recipe.sha256 for value in items}) != len(items):
        raise ValueError("clean recipes must be unique within one grouped shard")
    if len({value.clean_group_id for value in items}) != len(items):
        raise ValueError("clean parent group IDs must be unique")
    recipe_identities = {_recipe_identity(value.recipe) for value in items}
    if len(recipe_identities) != 1:
        raise ValueError("one grouped shard cannot mix clean recipe schemas or generators")
    recipe_schema, recipe_generator = recipe_identities.pop()
    clean_values: dict[str, list[object]] = {}
    observation_values: dict[str, list[object]] = {}
    context_values: dict[str, list[object]] = {}
    candidates: list[V5CandidateSupervision] = []
    candidate_recipe_indices: list[int] = []
    supervision_audits: list[str] = []
    exact_jsons: list[str] = []
    search_jsons: list[str] = []
    generating_matches: list[bool] = []

    for recipe_index, spec in enumerate(items):
        recipe = spec.recipe
        recipe_id = f"clean-{spec.clean_group_id}"
        _append(clean_values, clean_array("recipe_id"), recipe_id)
        _append(clean_values, clean_array("clean_group_id"), spec.clean_group_id)
        _append(clean_values, clean_array("recipe_sha256"), recipe.sha256)
        _append(clean_values, clean_array("recipe_canonical_json"), recipe.canonical_json)
        _append(clean_values, clean_array("recipe_schema_version"), recipe_schema)
        _append(
            clean_values,
            clean_array("recipe_generator_version"),
            recipe_generator,
        )
        _append(clean_values, clean_array("recipe_seed"), recipe.recipe_seed)
        _append(clean_values, clean_array("geometry_query_sha256"), recipe.query.sha256)
        _append(
            clean_values,
            clean_array("amplitude_query_sha256"),
            recipe.amplitude_query.sha256,
        )
        _append(
            clean_values,
            clean_array("amplitude_query_canonical_json"),
            recipe.amplitude_query.canonical_json,
        )
        _append(clean_values, clean_array("target_pattern_id"), recipe.target.pattern_id)
        _append(clean_values, clean_array("target_local"), recipe.target.local_target_unit)
        _append(clean_values, clean_array("split_id"), spec.split_id)
        _append(
            clean_values,
            clean_array("sobol_index"),
            -1 if spec.sobol_index is None else spec.sobol_index,
        )
        _append(
            clean_values,
            clean_array("split_plan_sha256"),
            "" if spec.split_plan_sha256 is None else spec.split_plan_sha256,
        )
        _append(
            clean_values,
            clean_array("sobol_design_sha256"),
            "" if spec.sobol_design_sha256 is None else spec.sobol_design_sha256,
        )

        search, exact, exact_json, search_json = run_v5_known_truth_exact_oracle(recipe, recipe_id)
        views = build_v5_observation_data_views(
            recipe,
            spec.view_indices,
            split_id=spec.split_id,
        )
        paired_query = V5TopologyQuery(recipe.query, recipe.amplitude_query)
        patterns = (
            (recipe.target.pattern_id,)
            if generating_only
            else paired_query.feasible_wire_pattern_ids
        )
        first_batch = None
        for view in views:
            batch = build_v5_candidate_context_batch(
                view.preprocessed,
                view.uncertainty,
                recipe.query,
                recipe.amplitude_query,
                pattern_ids=patterns,
            )
            first_batch = batch if first_batch is None else first_batch
            _append(
                observation_values,
                observation_array("observation_id"),
                _observation_id(recipe_id, view),
            )
            _append(observation_values, observation_array("recipe_index"), recipe_index)
            _append(observation_values, observation_array("view_index"), view.view_index)
            _append(observation_values, observation_array("split_id"), spec.split_id)
            _append(
                observation_values,
                observation_array("acquisition_policy_id"),
                view.acquisition_policy_id,
            )
            _append(
                observation_values,
                observation_array("intensity_reference"),
                view.preprocessed.stats["intensity_reference"],
            )
            _append(
                observation_values,
                observation_array("audit_json"),
                canonical_json(view.audit_payload()),
            )
            for name in _OBSERVATION_MODEL_INPUTS:
                _append(observation_values, observation_input(name), batch.model_inputs[name][0])
        assert first_batch is not None
        for candidate_index, (condition, constraint) in enumerate(
            zip(first_batch.branch_conditions, first_batch.amplitude_constraints)
        ):
            candidate_id = f"{recipe_id}:wire-{condition.pattern_id}"
            constraint_json = canonical_json(constraint.to_audit_dict())
            _append(context_values, candidate_context_array("candidate_id"), candidate_id)
            _append(context_values, candidate_context_array("recipe_index"), recipe_index)
            _append(
                context_values,
                candidate_context_array("geometry_query_sha256"),
                recipe.query.sha256,
            )
            _append(
                context_values,
                candidate_context_array("amplitude_query_sha256"),
                recipe.amplitude_query.sha256,
            )
            _append(
                context_values,
                candidate_context_array("amplitude_constraint_json"),
                constraint_json,
            )
            _append(
                context_values,
                candidate_context_array("amplitude_constraint_sha256"),
                sha256(constraint_json.encode("utf-8")).hexdigest(),
            )
            for name in MODEL_V5_INPUT_KEYS:
                if name not in _OBSERVATION_MODEL_INPUTS | _DERIVED_JOIN_MODEL_INPUTS:
                    _append(
                        context_values,
                        candidate_input(name),
                        first_batch.model_inputs[name][candidate_index],
                    )
            generating = condition.pattern_id == recipe.target.pattern_id
            supervision = V5CandidateSupervision(
                clean_recipe_id=recipe_id,
                candidate_id=candidate_id,
                outcome="compatible_found" if generating else "unverified",
                active_dimension_mask=condition.active_dimension_mask,
                varying_dimension_mask=condition.varying_dimension_mask,
                search_provenance=search if generating else None,
                exact_compatible=exact if generating else None,
                target_local=recipe.target.local_target_unit if generating else None,
                generating_candidate_match=generating,
            )
            candidates.append(supervision)
            candidate_recipe_indices.append(recipe_index)
            supervision_audits.append(canonical_json(supervision.audit_payload()))
            exact_jsons.append(exact_json if generating else "")
            search_jsons.append(search_json if generating else "")
            generating_matches.append(generating)

    arrays = {
        **_table_arrays(
            clean_values,
            {
                clean_array("recipe_seed"): np.uint64,
                clean_array("target_pattern_id"): np.int32,
                clean_array("target_local"): np.float32,
                clean_array("sobol_index"): np.int64,
            },
        ),
        **_table_arrays(
            observation_values,
            {
                observation_array("recipe_index"): np.int32,
                observation_array("view_index"): np.uint64,
                observation_array("intensity_reference"): np.float64,
                **{
                    observation_input(name): (np.bool_ if name == "point_mask" else np.float32)
                    for name in _OBSERVATION_MODEL_INPUTS
                },
            },
        ),
        **_table_arrays(
            context_values,
            {
                candidate_context_array("recipe_index"): np.int32,
                candidate_input("branch_topology_id"): np.int32,
                candidate_input("branch_pattern_id"): np.int32,
                **{
                    candidate_input(name): np.float32
                    for name in MODEL_V5_INPUT_KEYS
                    if name not in _OBSERVATION_MODEL_INPUTS | _DERIVED_JOIN_MODEL_INPUTS
                    and name not in {"branch_topology_id", "branch_pattern_id"}
                },
            },
        ),
    }
    labels = stack_candidate_supervision_v5(
        candidates,
        clean_recipe_indices=candidate_recipe_indices,
    )
    arrays.update({candidate_label(name): value for name, value in labels.items()})
    arrays.update(
        {
            candidate_label("supervision_audit_json"): np.asarray(
                supervision_audits, dtype=np.str_
            ),
            candidate_label("oracle_exact_artifact_json"): np.asarray(exact_jsons, dtype=np.str_),
            candidate_label("oracle_search_artifact_json"): np.asarray(search_jsons, dtype=np.str_),
            candidate_label("generating_candidate_match"): np.asarray(
                generating_matches, dtype=np.bool_
            ),
        }
    )
    observation_recipe = arrays[observation_array("recipe_index")]
    candidate_recipe = arrays[candidate_context_array("recipe_index")]
    joined_count = sum(
        np.count_nonzero(observation_recipe == index) * np.count_nonzero(candidate_recipe == index)
        for index in range(len(items))
    )
    counts = {
        "clean_recipes": len(items),
        "observation_views": int(observation_recipe.size),
        "candidates": len(candidates),
        "joined_examples": int(joined_count),
    }
    manifest = build_grouped_manifest(
        dataset_id=dataset_id,
        generating_only=generating_only,
        arrays=arrays,
        counts=counts,
        source_sha256=_source_hashes(),
        oracle_protocol={
            "payload": V5_ORACLE_PROTOCOL_PAYLOAD,
            "sha256": V5_ORACLE_PROTOCOL_SHA256,
        },
        clean_recipe_identity={
            "protocol_version": V5_CLEAN_RECIPE_PROTOCOL_VERSION,
            "schema_version": recipe_schema,
            "generator_version": recipe_generator,
        },
        shard_selection=shard_selection,
    )
    return V5GroupedDataset(manifest, arrays)


def build_tiny_v5_grouped_dataset(
    *,
    recipe_count: int = 1,
    topology: Sequence[str] = ("sphere",),
    base_seed: int = 20260903,
    view_indices: Sequence[int] = (0,),
    split_id: str = "train",
    pattern_id: int | None = 0,
    generating_only: bool = True,
) -> V5GroupedDataset:
    """Build the deterministic K=1 memorization fixture or a tiny pilot shard."""

    if isinstance(recipe_count, (bool, np.bool_)) or not isinstance(recipe_count, Integral):
        raise TypeError("recipe_count must be an integer")
    if int(recipe_count) < 1:
        raise ValueError("recipe_count must be positive")
    seed = _uint64(base_seed, "base_seed")
    recipes = tuple(
        sample_v5_clean_recipe(
            topology,
            recipe_seed=seed + index,
            amplitude_range_regime="full",
            pattern_id=pattern_id,
        )
        for index in range(int(recipe_count))
    )
    specs = tuple(
        V5GroupedRecipeSpec(
            recipe=recipe,
            split_id=split_id,
            view_indices=tuple(view_indices),
        )
        for recipe in recipes
    )
    identity = {
        "kind": "tiny_v5_grouped_solution_dataset",
        "recipe_count": int(recipe_count),
        "topology": list(topology),
        "base_seed": seed,
        "view_indices": list(view_indices),
        "split_id": split_id,
        "pattern_id": pattern_id,
        "generating_only": generating_only,
    }
    dataset_id = f"tiny-v5-{sha256(canonical_json(identity).encode('utf-8')).hexdigest()}"
    return build_v5_grouped_solution_dataset(
        specs,
        dataset_id=dataset_id,
        generating_only=generating_only,
    )


__all__ = [
    "V5_ORACLE_EXACT_ARTIFACT_SCHEMA",
    "V5_ORACLE_PROTOCOL_ID",
    "V5_ORACLE_PROTOCOL_SHA256",
    "V5_ORACLE_SEARCH_ARTIFACT_SCHEMA",
    "V5GroupedRecipeSpec",
    "build_tiny_v5_grouped_dataset",
    "build_v5_grouped_solution_dataset",
]
