"""Lossless one-view parent projection for balanced all-K1 full search."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Sequence

import numpy as np

from .build_grouped_dataset_v5 import (
    V5GroupedRecipeSpec,
    build_v5_grouped_solution_dataset,
)
from .formal_label_observation_policy_v5 import (
    V5FormalLabelObservationSelection,
    select_v5_formal_label_observation,
)
from .grouped_artifact_v5 import canonical_json
from .grouped_dataset_v5 import (
    V5GroupedDataset,
    clean_array,
    observation_array,
)
from .k1_forced_sobol_recipe_v5 import (
    V5K1PersistedForcedCleanRecipe,
    persisted_v5_k1_forced_clean_recipe_from_json,
)
from .k1_forced_universal_query_v5 import (
    V5K1ForcedUniversalQuerySet,
    materialize_v5_k1_forced_universal_query_set,
)
from .observation_v5 import sample_v5_uncertainty_provenance, v5_acquisition_policy_id
from .simulation import sample_observation_view


V5_K1_BALANCED_FULL_SEARCH_PARENT_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_full_search_parent_projection/v1"
)
V5_K1_BALANCED_FULL_SEARCH_PARENT_VERSION = (
    "posterior_v8_v5_2_artifact_recipe_bound_curve_blind_one_view_parent_v1"
)


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _candidate_views(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("candidate_view_indices must be an integer sequence")
    result = tuple(values)
    if (
        not result
        or len(result) != len(set(result))
        or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, int) for value in result)
        or any(value < 0 for value in result)
    ):
        raise ValueError("candidate_view_indices must be unique non-negative integers")
    return result


def _assert_equal_array(left: np.ndarray, right: np.ndarray, name: str) -> None:
    dtype_matches = left.dtype == right.dtype or (
        left.dtype.kind in {"S", "U"} and left.dtype.kind == right.dtype.kind
    )
    if (
        not dtype_matches
        or left.shape != right.shape
        or not np.array_equal(left, right)
    ):
        raise RuntimeError(f"projected grouped parent changed {name}")


def _verify_lossless_projection(
    source: V5GroupedDataset,
    projected: V5GroupedDataset,
) -> None:
    for name, values in projected.arrays.items():
        if name.startswith(("clean__", "candidate_context__", "candidate_label__")):
            _assert_equal_array(values, source.arrays[name], name)

    source_ids = tuple(str(value) for value in source.arrays[observation_array("observation_id")])
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source parent observation IDs are duplicated")
    source_index = {value: index for index, value in enumerate(source_ids)}
    projected_ids = tuple(
        str(value) for value in projected.arrays[observation_array("observation_id")]
    )
    try:
        selected = np.asarray([source_index[value] for value in projected_ids], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError("projected observation is absent from its source parent") from exc
    for name, values in projected.arrays.items():
        if name.startswith("observation__"):
            _assert_equal_array(values, source.arrays[name][selected], name)


@dataclass(frozen=True)
class V5K1BalancedFullSearchParentProjection:
    """A search parent plus the artifact-authoritative derivation inputs."""

    parent: V5GroupedDataset
    recipes: tuple[V5K1PersistedForcedCleanRecipe, ...]
    query_sets: tuple[V5K1ForcedUniversalQuerySet, ...]
    observation_selections: tuple[V5FormalLabelObservationSelection, ...]
    audit: Mapping[str, object]
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.parent, V5GroupedDataset):
            raise TypeError("parent must be a V5GroupedDataset")
        count = self.parent.recipe_count
        if not (
            count
            == len(self.recipes)
            == len(self.query_sets)
            == len(self.observation_selections)
            == self.parent.observation_count
        ):
            raise ValueError("projected parent membership is incomplete")
        if self.parent.manifest["build_policy"]["generating_candidate_only"] is not False:
            raise ValueError("projected search parent must expose full source-query branches")
        encoded = canonical_json(dict(self.audit))
        if self.sha256 != sha256(encoded.encode("utf-8")).hexdigest():
            raise ValueError("projected parent audit SHA-256 does not reproduce")


def project_v5_k1_balanced_full_search_parent(
    source_parent: V5GroupedDataset,
    *,
    source_parent_artifact_sha256: str,
    expected_balanced_dataset_plan_sha256: str,
    expected_balanced_sobol_block_sha256: str,
    expected_role: str,
    expected_split_id: str,
    candidate_view_indices: Sequence[int] = (0, 1),
) -> V5K1BalancedFullSearchParentProjection:
    """Project a checked generating shard without changing recipes or observations."""

    if not isinstance(source_parent, V5GroupedDataset):
        raise TypeError("source_parent must be a V5GroupedDataset")
    source_artifact = _digest(
        source_parent_artifact_sha256,
        "source_parent_artifact_sha256",
    )
    plan_sha = _digest(
        expected_balanced_dataset_plan_sha256,
        "expected_balanced_dataset_plan_sha256",
    )
    block_sha = _digest(
        expected_balanced_sobol_block_sha256,
        "expected_balanced_sobol_block_sha256",
    )
    if expected_role not in ("train", "tuning_validation"):
        raise ValueError("expected_role is unsupported")
    if expected_split_id != expected_role:
        raise ValueError("expected role/split binding is invalid")
    views = _candidate_views(candidate_view_indices)
    if source_parent.manifest["build_policy"]["generating_candidate_only"] is not True:
        raise ValueError("source parent must be the immutable generating-only shard")

    recipes = []
    query_sets = []
    selections = []
    specs = []
    arrays = source_parent.arrays
    for index in range(source_parent.recipe_count):
        recipe_sha = str(arrays[clean_array("recipe_sha256")][index])
        recipe = persisted_v5_k1_forced_clean_recipe_from_json(
            str(arrays[clean_array("recipe_canonical_json")][index]),
            recipe_sha,
        )
        identity = recipe.identity
        if (
            identity.balanced_dataset_plan_sha256 != plan_sha
            or identity.balanced_sobol_block_sha256 != block_sha
            or identity.role != expected_role
            or identity.split_id != expected_split_id
            or identity.clean_group_id
            != str(arrays[clean_array("clean_group_id")][index])
            or identity.sobol_index != int(arrays[clean_array("sobol_index")][index])
            or identity.sobol_design_sha256
            != str(arrays[clean_array("sobol_design_sha256")][index])
        ):
            raise ValueError("source parent recipe escaped its balanced shard binding")
        # Policy identity includes the numerical noise algorithm version. A
        # historical observation must never silently acquire today's identity.
        observation_rows = np.flatnonzero(arrays[observation_array("recipe_index")] == index)
        for row in observation_rows:
            view_index = int(arrays[observation_array("view_index")][row])
            expected_policy = v5_acquisition_policy_id(
                sample_observation_view(recipe.recipe_seed, view_index),
                sample_v5_uncertainty_provenance(recipe.recipe_seed, view_index),
            )
            if str(arrays[observation_array("acquisition_policy_id")][row]) != expected_policy:
                raise ValueError(
                    "source observation acquisition policy is incompatible with current "
                    "reconstruction; preserve the historical artifact and use a version-matched dataset"
                )
        query_set = materialize_v5_k1_forced_universal_query_set(
            recipe.canonical_json,
            expected_recipe_sha256=recipe.sha256,
        )
        selection = select_v5_formal_label_observation(recipe.recipe_seed, views)
        recipes.append(recipe)
        query_sets.append(query_set)
        selections.append(selection)
        specs.append(
            V5GroupedRecipeSpec(
                recipe=recipe,
                split_id=expected_split_id,
                view_indices=(selection.selected_view_index,),
                clean_group_id=identity.clean_group_id,
                sobol_index=identity.sobol_index,
                split_plan_sha256=plan_sha,
                sobol_design_sha256=identity.sobol_design_sha256,
            )
        )

    sobol_indices = tuple(value.sobol_index for value in recipes)
    if tuple(sorted(set(sobol_indices))) != sobol_indices:
        raise ValueError("source parent Sobol membership is duplicated or reordered")
    identity = {
        "schema": V5_K1_BALANCED_FULL_SEARCH_PARENT_SCHEMA,
        "version": V5_K1_BALANCED_FULL_SEARCH_PARENT_VERSION,
        "source_parent_artifact_sha256": source_artifact,
        "balanced_dataset_plan_sha256": plan_sha,
        "balanced_sobol_block_sha256": block_sha,
        "role": expected_role,
        "split_id": expected_split_id,
        "candidate_view_indices": list(views),
        "recipe_sha256s": [value.sha256 for value in recipes],
        "query_set_sha256s": [value.sha256 for value in query_sets],
        "observation_selection_sha256s": [
            value.audit_sha256 for value in selections
        ],
        "one_curve_blind_sigma_present_view_per_clean_parent": True,
        "source_recipe_or_observation_changed": False,
        "training_authorization_granted": False,
    }
    projection_sha = sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    parent = build_v5_grouped_solution_dataset(
        specs,
        dataset_id=f"balanced-k1-full-search-parent-v1-{projection_sha}",
        generating_only=False,
        shard_selection=None,
    )
    _verify_lossless_projection(source_parent, parent)
    return V5K1BalancedFullSearchParentProjection(
        parent=parent,
        recipes=tuple(recipes),
        query_sets=tuple(query_sets),
        observation_selections=tuple(selections),
        audit=identity,
        sha256=projection_sha,
    )


__all__ = [
    "V5K1BalancedFullSearchParentProjection",
    "V5_K1_BALANCED_FULL_SEARCH_PARENT_SCHEMA",
    "V5_K1_BALANCED_FULL_SEARCH_PARENT_VERSION",
    "project_v5_k1_balanced_full_search_parent",
]
