"""Compact relational dataset contract for V5.1 candidate proposals.

Clean physical recipes, stochastic observation views, and branch candidates
are stored as three tables.  A padded 1000-point curve therefore appears once
per observation rather than once per candidate.  NumPy and ``tf.data`` joins
materialize the exact model/label dictionaries only when training asks for
them.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import os
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .candidate_supervision_v5 import (
    CANDIDATE_SUPERVISION_TENSOR_KEYS,
    SEARCH_OUTCOME_CODE,
    candidate_supervision_v5_contract_payload,
)
from .grouped_artifact_v5 import (
    V5_CHECKED_ARRAY_ARTIFACT_SCHEMA,
    V5_CHECKED_ARRAY_ARTIFACT_VERSION,
    V5ArtifactReceipt,
    array_manifest,
    canonical_json,
    read_checked_array_artifact,
    validate_array_manifest,
    write_checked_array_artifact,
)
from .grouped_amplitude_join_v5 import amplitude_query_from_json, joined_amplitude_embeddings
from .grouped_manifest_validation_v5 import validate_grouped_manifest_semantics
from .grouped_provenance_validation_v5 import validate_grouped_provenance
from .grouped_shard_metadata_v5 import build_grouped_shard_metadata
from .grouped_tensor_contract_v5 import validate_grouped_tensor_contract
from .model_v5_contract import MODEL_V5_INPUT_KEYS, model_v5_contract_payload


V5_GROUPED_DATASET_SCHEMA = "gisaxs.posterior_v8.grouped_candidate_dataset/v4"
V5_GROUPED_DATASET_VERSION = (
    "posterior_v8_v5_2_center_aligned_complete_slot_contract_varying_axes_relational_stage_v4"
)
V5_GROUPED_SOLUTION_STAGE = "oracle_known_truth_solution_stage_no_automatic_negatives_v1"

CLEAN_FIELDS = (
    "recipe_id",
    "clean_group_id",
    "recipe_sha256",
    "recipe_canonical_json",
    "recipe_schema_version",
    "recipe_generator_version",
    "recipe_seed",
    "geometry_query_sha256",
    "amplitude_query_sha256",
    "amplitude_query_canonical_json",
    "target_pattern_id",
    "target_local",
    "split_id",
    "sobol_index",
    "split_plan_sha256",
    "sobol_design_sha256",
)
OBSERVATION_FIELDS = (
    "observation_id",
    "recipe_index",
    "view_index",
    "split_id",
    "acquisition_policy_id",
    "intensity_reference",
    "audit_json",
)
CANDIDATE_CONTEXT_FIELDS = (
    "candidate_id",
    "recipe_index",
    "geometry_query_sha256",
    "amplitude_query_sha256",
    "amplitude_constraint_json",
    "amplitude_constraint_sha256",
)
CANDIDATE_LABEL_EXTRA_FIELDS = (
    "supervision_audit_json",
    "oracle_exact_artifact_json",
    "oracle_search_artifact_json",
    "generating_candidate_match",
)


def _array_name(table: str, category: str, name: str) -> str:
    return f"{table}__{category}__{name}"


def clean_array(name: str) -> str:
    return f"clean__{name}"


def observation_array(name: str) -> str:
    return f"observation__{name}"


def observation_input(name: str) -> str:
    return _array_name("observation", "input", name)


def candidate_context_array(name: str) -> str:
    return f"candidate_context__{name}"


def candidate_input(name: str) -> str:
    return _array_name("candidate_context", "input", name)


def candidate_label(name: str) -> str:
    return f"candidate_label__{name}"


def grouped_contract_bundle() -> dict[str, object]:
    """Return current model, label, and local-only objective identities."""

    from .training_objective_v5 import (
        DEFAULT_LOCAL_COVERAGE_WEIGHT,
        V5CandidateObjectiveConfig,
    )

    return {
        "model": model_v5_contract_payload(),
        "candidate_supervision": candidate_supervision_v5_contract_payload(),
        "solution_stage_objective": V5CandidateObjectiveConfig(
            search_yield_weight=0.0,
            pairwise_ranking_weight=0.0,
            local_mdn_weight=1.0,
            local_coverage_weight=DEFAULT_LOCAL_COVERAGE_WEIGHT,
        ).audit_payload(),
    }


def _manifest_core(
    *,
    dataset_id: str,
    generating_only: bool,
    arrays: Mapping[str, np.ndarray],
    counts: Mapping[str, int],
    source_sha256: Mapping[str, str],
    oracle_protocol: Mapping[str, object],
    clean_recipe_identity: Mapping[str, str],
    shard_selection: Mapping[str, object] | None,
) -> dict[str, object]:
    bundle = grouped_contract_bundle()
    return {
        "container_schema": V5_CHECKED_ARRAY_ARTIFACT_SCHEMA,
        "container_version": V5_CHECKED_ARRAY_ARTIFACT_VERSION,
        "dataset_schema": V5_GROUPED_DATASET_SCHEMA,
        "dataset_version": V5_GROUPED_DATASET_VERSION,
        "dataset_id": dataset_id,
        "stage": V5_GROUPED_SOLUTION_STAGE,
        "build_policy": {
            "generating_candidate_only": generating_only,
            "non_generating_feasible_branch_outcome": "unverified",
            "unsearched_branch_is_negative": False,
            "curve_storage": "once_per_observation_view",
            "candidate_context_storage": "once_per_clean_recipe_branch",
        },
        "counts": dict(counts),
        "table_fields": {
            "clean": list(CLEAN_FIELDS),
            "observation": list(OBSERVATION_FIELDS),
            "candidate_context": list(CANDIDATE_CONTEXT_FIELDS),
            "candidate_label": [
                *CANDIDATE_SUPERVISION_TENSOR_KEYS,
                *CANDIDATE_LABEL_EXTRA_FIELDS,
            ],
        },
        "model_input_keys": list(MODEL_V5_INPUT_KEYS),
        "candidate_label_keys": list(CANDIDATE_SUPERVISION_TENSOR_KEYS),
        "contract_bundle": bundle,
        "contract_bundle_sha256": sha256(canonical_json(bundle).encode("utf-8")).hexdigest(),
        "clean_recipe_identity": dict(clean_recipe_identity),
        "source_sha256": dict(sorted(source_sha256.items())),
        "oracle_protocol": deepcopy(dict(oracle_protocol)),
        "shard_metadata": build_grouped_shard_metadata(
            arrays,
            shard_selection,
            generating_only=generating_only,
            clean_recipe_identity=clean_recipe_identity,
        ),
        "split_semantics": {
            "statistical_unit": "independent_clean_physical_recipe",
            "inheritance": "all_observation_views_and_candidates_follow_clean_parent",
            "sobol_parent_fields_are_empty_together_or_complete_together": True,
        },
        "arrays": array_manifest(arrays),
    }


def build_grouped_manifest(
    *,
    dataset_id: str,
    generating_only: bool,
    arrays: Mapping[str, np.ndarray],
    counts: Mapping[str, int],
    source_sha256: Mapping[str, str],
    oracle_protocol: Mapping[str, object],
    clean_recipe_identity: Mapping[str, str],
    shard_selection: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be non-empty")
    if type(generating_only) is not bool:
        raise TypeError("generating_only must be a bool")
    core = _manifest_core(
        dataset_id=dataset_id.strip(),
        generating_only=generating_only,
        arrays=arrays,
        counts=counts,
        source_sha256=source_sha256,
        oracle_protocol=oracle_protocol,
        clean_recipe_identity=clean_recipe_identity,
        shard_selection=shard_selection,
    )
    return {
        **core,
        "manifest_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }


def _first_dimension(arrays: Mapping[str, np.ndarray], names: tuple[str, ...], size: int) -> None:
    for name in names:
        value = arrays[name]
        if value.ndim < 1 or value.shape[0] != size:
            raise ValueError(f"{name} must have table first dimension {size}")


@dataclass(frozen=True)
class V5GroupedDataset:
    manifest: Mapping[str, object]
    arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        manifest = deepcopy(dict(self.manifest))
        arrays = validate_array_manifest(self.arrays, manifest.get("arrays", {}))
        self._validate_manifest(manifest)
        self._validate_tables(manifest, arrays)
        object.__setattr__(self, "manifest", MappingProxyType(manifest))
        object.__setattr__(self, "arrays", arrays)

    @property
    def recipe_count(self) -> int:
        return int(self.manifest["counts"]["clean_recipes"])

    @property
    def observation_count(self) -> int:
        return int(self.manifest["counts"]["observation_views"])

    @property
    def candidate_count(self) -> int:
        return int(self.manifest["counts"]["candidates"])

    @property
    def joined_count(self) -> int:
        return int(self.manifest["counts"]["joined_examples"])

    def _validate_manifest(self, manifest: dict[str, object]) -> None:
        validate_grouped_manifest_semantics(
            manifest,
            dataset_schema=V5_GROUPED_DATASET_SCHEMA,
            dataset_version=V5_GROUPED_DATASET_VERSION,
            stage=V5_GROUPED_SOLUTION_STAGE,
            clean_fields=CLEAN_FIELDS,
            observation_fields=OBSERVATION_FIELDS,
            candidate_context_fields=CANDIDATE_CONTEXT_FIELDS,
            candidate_label_extra_fields=CANDIDATE_LABEL_EXTRA_FIELDS,
            contract_bundle=grouped_contract_bundle(),
        )

    def _validate_tables(
        self,
        manifest: Mapping[str, object],
        arrays: Mapping[str, np.ndarray],
    ) -> None:
        counts = manifest["counts"]
        if not isinstance(counts, Mapping) or set(counts) != {
            "clean_recipes",
            "observation_views",
            "candidates",
            "joined_examples",
        }:
            raise ValueError("grouped counts are incomplete or unsupported")
        recipe_count, observation_count, candidate_count = (
            int(counts[name]) for name in ("clean_recipes", "observation_views", "candidates")
        )
        if min(recipe_count, observation_count, candidate_count) < 1:
            raise ValueError("grouped tables cannot be empty")
        clean_names = tuple(clean_array(name) for name in CLEAN_FIELDS)
        observation_names = tuple(observation_array(name) for name in OBSERVATION_FIELDS)
        context_names = tuple(candidate_context_array(name) for name in CANDIDATE_CONTEXT_FIELDS)
        label_names = tuple(
            candidate_label(name)
            for name in (*CANDIDATE_SUPERVISION_TENSOR_KEYS, *CANDIDATE_LABEL_EXTRA_FIELDS)
        )
        model_names = []
        for name in MODEL_V5_INPUT_KEYS:
            if name == "amplitude_bounds_embedding":
                continue
            options = (observation_input(name), candidate_input(name))
            present = tuple(value for value in options if value in arrays)
            if len(present) != 1:
                raise ValueError(f"model input {name} must belong to exactly one grouped table")
            model_names.extend(present)
        expected_names = set(
            (*clean_names, *observation_names, *context_names, *label_names, *model_names)
        )
        if set(arrays) != expected_names:
            raise ValueError("grouped array inventory is incomplete or unsupported")
        _first_dimension(arrays, clean_names, recipe_count)
        _first_dimension(
            arrays,
            (
                *observation_names,
                *(name for name in model_names if name.startswith("observation__")),
            ),
            observation_count,
        )
        _first_dimension(
            arrays,
            (
                *context_names,
                *label_names,
                *(name for name in model_names if name.startswith("candidate_")),
            ),
            candidate_count,
        )
        recipe_ids = arrays[clean_array("recipe_id")]
        if len(set(recipe_ids.tolist())) != recipe_count:
            raise ValueError("clean recipe IDs must be unique")
        for index, encoded in enumerate(arrays[clean_array("recipe_canonical_json")]):
            digest = sha256(str(encoded).encode("utf-8")).hexdigest()
            if digest != arrays[clean_array("recipe_sha256")][index]:
                raise ValueError("clean recipe JSON/hash does not reproduce")
        identity = manifest["clean_recipe_identity"]
        if not np.all(
            arrays[clean_array("recipe_schema_version")] == identity["schema_version"]
        ) or not np.all(
            arrays[clean_array("recipe_generator_version")] == identity["generator_version"]
        ):
            raise ValueError("clean recipe rows do not match the shard recipe identity")
        for encoded, digest in zip(
            arrays[clean_array("amplitude_query_canonical_json")],
            arrays[clean_array("amplitude_query_sha256")],
        ):
            amplitude_query_from_json(str(encoded), str(digest))
        observation_recipe = arrays[observation_array("recipe_index")]
        candidate_recipe = arrays[candidate_context_array("recipe_index")]
        if (
            observation_recipe.dtype != np.int32
            or candidate_recipe.dtype != np.int32
            or np.any(observation_recipe < 0)
            or np.any(observation_recipe >= recipe_count)
            or np.any(candidate_recipe < 0)
            or np.any(candidate_recipe >= recipe_count)
        ):
            raise ValueError("grouped foreign-key indices are invalid")
        clean_split = arrays[clean_array("split_id")]
        if not np.array_equal(
            arrays[observation_array("split_id")], clean_split[observation_recipe]
        ):
            raise ValueError("observation views do not inherit the clean-parent split")
        label_recipe = arrays[candidate_label("clean_recipe_index")]
        if not np.array_equal(label_recipe, candidate_recipe):
            raise ValueError("candidate labels do not inherit the clean-parent index")
        for mask_name in ("active_dimension_mask", "varying_dimension_mask"):
            if not np.array_equal(
                arrays[candidate_label(mask_name)], arrays[candidate_input(mask_name)]
            ):
                raise ValueError(f"candidate label/context {mask_name} disagree")
        outcomes = arrays[candidate_label("search_outcome_code")]
        if np.any(
            ~np.isin(
                outcomes,
                (SEARCH_OUTCOME_CODE["unverified"], SEARCH_OUTCOME_CODE["compatible_found"]),
            )
        ):
            raise ValueError("solution-stage dataset cannot contain automatic negatives")
        positive = outcomes == SEARCH_OUTCOME_CODE["compatible_found"]
        unverified = ~positive
        generating = arrays[candidate_label("generating_candidate_match")]
        if not np.all(generating[positive]) or np.any(generating[unverified]):
            raise ValueError("only the generating known-truth candidate may be positive")
        if np.any(arrays[candidate_label("has_local_target")][unverified]):
            raise ValueError("unverified candidates cannot carry local targets")
        pattern = arrays[candidate_input("branch_pattern_id")].reshape(-1)
        target_pattern = arrays[clean_array("target_pattern_id")][candidate_recipe]
        if not np.array_equal(pattern[positive], target_pattern[positive]):
            raise ValueError("positive branch does not match the generating target")
        positive_per_recipe = np.bincount(candidate_recipe[positive], minlength=recipe_count)
        if not np.all(positive_per_recipe == 1):
            raise ValueError("each clean recipe must have exactly one generating positive")
        if manifest["build_policy"]["generating_candidate_only"] and np.any(unverified):
            raise ValueError("generating-only artifact contains unverified candidates")
        expected_joined = sum(
            np.count_nonzero(observation_recipe == index)
            * np.count_nonzero(candidate_recipe == index)
            for index in range(recipe_count)
        )
        if int(counts["joined_examples"]) != expected_joined:
            raise ValueError("joined example count disagrees with grouped foreign keys")
        shard_metadata = manifest["shard_metadata"]
        if not isinstance(
            shard_metadata, Mapping
        ) or shard_metadata != build_grouped_shard_metadata(
            arrays,
            shard_metadata.get("formal_sobol_selection"),
            generating_only=manifest["build_policy"]["generating_candidate_only"],
            clean_recipe_identity=manifest["clean_recipe_identity"],
        ):
            raise ValueError("grouped split/Sobol shard metadata does not reproduce")
        validate_grouped_tensor_contract(
            arrays,
            recipe_count=recipe_count,
            observation_count=observation_count,
            candidate_count=candidate_count,
        )
        validate_grouped_provenance(arrays, manifest, positive=positive)

    def join_indices(self, *, include_unverified: bool = True) -> tuple[np.ndarray, np.ndarray]:
        observation_recipe = self.arrays[observation_array("recipe_index")]
        candidate_recipe = self.arrays[candidate_context_array("recipe_index")]
        outcomes = self.arrays[candidate_label("search_outcome_code")]
        pairs = [
            (observation_index, candidate_index)
            for observation_index, recipe_index in enumerate(observation_recipe)
            for candidate_index in np.flatnonzero(candidate_recipe == recipe_index)
            if include_unverified
            or outcomes[candidate_index] == SEARCH_OUTCOME_CODE["compatible_found"]
        ]
        if not pairs:
            raise ValueError("grouped join selected no examples")
        observed, candidates = np.asarray(pairs, dtype=np.int32).T
        return observed, candidates

    def joined_numpy(
        self, *, include_unverified: bool = True
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        observed, candidates = self.join_indices(include_unverified=include_unverified)
        inputs = {}
        for name in MODEL_V5_INPUT_KEYS:
            if name == "amplitude_bounds_embedding":
                inputs[name] = joined_amplitude_embeddings(
                    query_json=self.arrays[clean_array("amplitude_query_canonical_json")],
                    query_sha256=self.arrays[clean_array("amplitude_query_sha256")],
                    observation_intensity_reference=self.arrays[
                        observation_array("intensity_reference")
                    ],
                    observation_recipe_index=self.arrays[observation_array("recipe_index")],
                    candidate_recipe_index=self.arrays[candidate_context_array("recipe_index")],
                    observation_indices=observed,
                    candidate_indices=candidates,
                )
                continue
            observation_name, context_name = observation_input(name), candidate_input(name)
            if observation_name in self.arrays:
                inputs[name] = self.arrays[observation_name][observed]
            else:
                inputs[name] = self.arrays[context_name][candidates]
        labels = {
            name: self.arrays[candidate_label(name)][candidates]
            for name in CANDIDATE_SUPERVISION_TENSOR_KEYS
        }
        return inputs, labels

    def as_tensorflow_dataset(
        self,
        *,
        batch_size: int,
        include_unverified: bool = True,
        shuffle: bool = False,
        seed: int = 0,
    ):
        from .grouped_tf_loader_v5 import as_tensorflow_dataset

        return as_tensorflow_dataset(
            self,
            batch_size=batch_size,
            include_unverified=include_unverified,
            shuffle=shuffle,
            seed=seed,
        )


def write_v5_grouped_dataset(
    dataset: V5GroupedDataset,
    path: str | os.PathLike[str],
) -> V5ArtifactReceipt:
    if not isinstance(dataset, V5GroupedDataset):
        raise TypeError("dataset must be a V5GroupedDataset")
    return write_checked_array_artifact(path, manifest=dataset.manifest, arrays=dataset.arrays)


def read_v5_grouped_dataset(
    path: str | os.PathLike[str],
) -> tuple[V5GroupedDataset, V5ArtifactReceipt]:
    manifest, arrays, receipt = read_checked_array_artifact(path)
    return V5GroupedDataset(manifest, arrays), receipt


__all__ = [
    "CANDIDATE_CONTEXT_FIELDS",
    "CANDIDATE_LABEL_EXTRA_FIELDS",
    "CLEAN_FIELDS",
    "OBSERVATION_FIELDS",
    "V5_GROUPED_DATASET_SCHEMA",
    "V5_GROUPED_DATASET_VERSION",
    "V5_GROUPED_SOLUTION_STAGE",
    "V5GroupedDataset",
    "build_grouped_manifest",
    "candidate_context_array",
    "candidate_input",
    "candidate_label",
    "clean_array",
    "grouped_contract_bundle",
    "observation_array",
    "observation_input",
    "read_v5_grouped_dataset",
    "write_v5_grouped_dataset",
]
