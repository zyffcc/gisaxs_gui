"""Strict training join for a V5.1 solution shard and search sidecar.

The parent contributes clean-parent and observation rows.  The sidecar
contributes observation-specific cross-topology branch contexts and frozen
search labels.  No array in the solution-stage artifact is rewritten.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import numpy as np

from .candidate_supervision_v5 import CANDIDATE_SUPERVISION_TENSOR_KEYS, SEARCH_OUTCOME_CODE
from .grouped_amplitude_join_v5 import amplitude_query_from_json
from .grouped_dataset_v5 import (
    V5GroupedDataset,
    candidate_context_array,
    candidate_input,
    clean_array,
    observation_array,
    observation_input,
    read_v5_grouped_dataset,
)
from .model_v5_contract import MODEL_V5_INPUT_KEYS
from .search_supervision_contract_v5 import V5CompatibleRepresentativeReference
from .search_supervision_sidecar_v5 import (
    BRANCH_MODEL_INPUT_KEYS,
    OBSERVATION_MODEL_INPUT_KEYS,
    V5_SEARCH_SIDECAR_ALLOWED_SPLITS,
    V5SearchSupervisionSidecar,
    branch_array,
    branch_input,
    branch_label,
    observation_inputs_sha256,
    query_array,
    read_v5_search_supervision_sidecar,
)


@dataclass(frozen=True)
class V5SearchSupervisionOverlay:
    """Fully checked parent/sidecar relationship and training materializer."""

    parent_path: Path
    sidecar_path: Path
    parent: V5GroupedDataset
    sidecar: V5SearchSupervisionSidecar
    parent_artifact_sha256: str
    sidecar_artifact_sha256: str

    def __post_init__(self) -> None:
        _validate_overlay(self.parent, self.sidecar, self.parent_artifact_sha256)

    @property
    def recipe_count(self) -> int:
        return self.parent.recipe_count

    @property
    def branch_count(self) -> int:
        return self.sidecar.branch_count

    @property
    def protocol_sha256(self) -> str:
        return str(self.sidecar.manifest["protocol_sha256"])

    def outcome_counts(self, recipes: Sequence[int] | None = None) -> dict[str, int]:
        labels = self.sidecar.arrays[branch_label("search_outcome_code")]
        if recipes is not None:
            selected = np.isin(
                self.sidecar.arrays[branch_label("clean_recipe_index")],
                np.asarray(tuple(recipes), dtype=np.int32),
            )
            labels = labels[selected]
        return {
            name: int(np.count_nonzero(labels == code))
            for name, code in SEARCH_OUTCOME_CODE.items()
        }

    def recipes(self) -> np.ndarray:
        return np.arange(self.parent.recipe_count, dtype=np.int32)

    def joined_numpy(
        self,
        *,
        require_completed_catalog: bool = True,
        require_positive_and_negative: bool = True,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        return self.numpy_batch(
            self.recipes(),
            phase="full",
            require_completed_catalog=require_completed_catalog,
            require_positive_and_negative=require_positive_and_negative,
        )

    def numpy_batch(
        self,
        recipes: Sequence[int],
        *,
        phase: str,
        require_completed_catalog: bool = True,
        require_positive_and_negative: bool = True,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        requested = np.asarray(tuple(recipes), dtype=np.int32)
        if requested.ndim != 1 or requested.size == 0 or len(set(requested.tolist())) != len(
            requested
        ):
            raise ValueError("recipes must be a non-empty unique sequence")
        if np.any(requested < 0) or np.any(requested >= self.parent.recipe_count):
            raise ValueError("recipe selection is outside the parent artifact")
        arrays = self.sidecar.arrays
        selected = np.flatnonzero(
            np.isin(arrays[branch_label("clean_recipe_index")], requested)
        )
        outcomes = arrays[branch_label("search_outcome_code")][selected]
        if phase == "warmup":
            selected = selected[outcomes == SEARCH_OUTCOME_CODE["compatible_found"]]
            outcomes = arrays[branch_label("search_outcome_code")][selected]
            require_positive_and_negative = False
        elif phase != "full":
            raise ValueError("phase must be warmup or full")
        if selected.size == 0:
            raise ValueError("selected recipe batch has no eligible sidecar rows")
        if require_completed_catalog and np.any(
            outcomes == SEARCH_OUTCOME_CODE["unverified"]
        ):
            raise ValueError("full search-yield training requires a completed branch catalog")
        if require_positive_and_negative:
            positive = np.any(outcomes == SEARCH_OUTCOME_CODE["compatible_found"])
            negative = np.any(
                outcomes
                == SEARCH_OUTCOME_CODE[
                    "no_compatible_found_within_frozen_search_budget"
                ]
            )
            if not positive or not negative:
                raise ValueError(
                    "full search-yield training requires completed positives and negatives"
                )
        selected, target_references, branch_multiplicity = _expand_local_targets(
            arrays, selected
        )
        query_indices = arrays[branch_array("query_index")][selected]
        observation_indices = self.sidecar.arrays[
            query_array("parent_observation_index")
        ][query_indices]
        inputs: dict[str, np.ndarray] = {}
        for name in MODEL_V5_INPUT_KEYS:
            if name in OBSERVATION_MODEL_INPUT_KEYS:
                inputs[name] = self.parent.arrays[observation_input(name)][
                    observation_indices
                ]
            else:
                inputs[name] = arrays[branch_input(name)][selected]
        label_values = {
            name: [value for value in arrays[branch_label(name)][selected]]
            for name in CANDIDATE_SUPERVISION_TENSOR_KEYS
        }
        for index, (reference, multiplicity) in enumerate(
            zip(target_references, branch_multiplicity)
        ):
            label_values["candidate_weight"][index] = (
                float(label_values["candidate_weight"][index]) / multiplicity
            )
            if reference is None:
                continue
            label_values["exact_artifact_id"][index] = reference.artifact_id
            label_values["exact_artifact_sha256"][index] = reference.artifact_sha256
            label_values["exact_metric_value"][index] = reference.metric_value
            label_values["exact_bounds_passed"][index] = reference.bounds_passed
            label_values["exact_physics_passed"][index] = reference.physics_passed
            label_values["target_local"][index] = np.asarray(
                reference.target_local, dtype=np.float32
            )
            label_values["has_local_target"][index] = True
        labels = {}
        for name, values in label_values.items():
            source_dtype = arrays[branch_label(name)].dtype
            dtype = np.str_ if source_dtype.kind == "U" else source_dtype
            labels[name] = np.asarray(values, dtype=dtype)
        return inputs, labels

    def tensor_batch(self, recipes: Sequence[int], *, phase: str):
        import tensorflow as tf

        inputs, labels = self.numpy_batch(recipes, phase=phase)
        return (
            {name: tf.convert_to_tensor(value) for name, value in inputs.items()},
            {name: tf.convert_to_tensor(value) for name, value in labels.items()},
        )


def read_v5_search_supervision_overlay(
    parent_dataset_path: str | os.PathLike[str],
    sidecar_path: str | os.PathLike[str],
) -> V5SearchSupervisionOverlay:
    parent, parent_receipt = read_v5_grouped_dataset(parent_dataset_path)
    sidecar, sidecar_receipt = read_v5_search_supervision_sidecar(sidecar_path)
    return V5SearchSupervisionOverlay(
        parent_path=Path(parent_dataset_path).resolve(),
        sidecar_path=Path(sidecar_path).resolve(),
        parent=parent,
        sidecar=sidecar,
        parent_artifact_sha256=parent_receipt.artifact_sha256,
        sidecar_artifact_sha256=sidecar_receipt.artifact_sha256,
    )


def _expand_local_targets(arrays, selected):
    """Repeat a positive branch per stored local target without reweighting its BCE."""

    expanded: list[int] = []
    references: list[V5CompatibleRepresentativeReference | None] = []
    multiplicities: list[int] = []
    positive_code = SEARCH_OUTCOME_CODE["compatible_found"]
    for row in selected:
        targets: tuple[V5CompatibleRepresentativeReference, ...] = ()
        if int(arrays[branch_label("search_outcome_code")][row]) == positive_code:
            payload = json.loads(
                str(arrays[branch_array("compatible_representatives_json")][row])
            )
            targets = tuple(
                reference
                for reference in (
                    V5CompatibleRepresentativeReference(**value) for value in payload
                )
                if reference.target_local is not None
            )
        if not targets:
            expanded.append(int(row))
            references.append(None)
            multiplicities.append(1)
            continue
        for reference in targets:
            expanded.append(int(row))
            references.append(reference)
            multiplicities.append(len(targets))
    return (
        np.asarray(expanded, dtype=np.int32),
        tuple(references),
        tuple(multiplicities),
    )


def _validate_overlay(
    parent: V5GroupedDataset,
    sidecar: V5SearchSupervisionSidecar,
    parent_artifact_sha256: str,
) -> None:
    binding = sidecar.manifest["parent_grouped_artifact"]
    if (
        binding.get("artifact_sha256") != parent_artifact_sha256
        or binding.get("manifest_sha256") != parent.manifest["manifest_sha256"]
        or binding.get("dataset_id") != parent.manifest["dataset_id"]
        or binding.get("dataset_schema") != parent.manifest["dataset_schema"]
        or binding.get("dataset_version") != parent.manifest["dataset_version"]
        or binding.get("stage") != parent.manifest["stage"]
    ):
        raise ValueError("search sidecar is bound to a different grouped parent artifact")
    if sidecar.query_count != parent.observation_count:
        raise ValueError("search sidecar does not cover every parent observation")
    split_id = str(sidecar.manifest["split_id"])
    if split_id not in V5_SEARCH_SIDECAR_ALLOWED_SPLITS or set(
        parent.arrays[clean_array("split_id")].tolist()
    ) != {split_id}:
        raise ValueError("search-supervision overlay crossed its development split")
    arrays = sidecar.arrays
    for query_index in range(sidecar.query_count):
        observation_index = int(
            arrays[query_array("parent_observation_index")][query_index]
        )
        recipe_index = int(parent.arrays[observation_array("recipe_index")][observation_index])
        if (
            int(arrays[query_array("parent_recipe_index")][query_index]) != recipe_index
            or arrays[query_array("clean_group_id")][query_index]
            != parent.arrays[clean_array("clean_group_id")][recipe_index]
            or arrays[query_array("recipe_id")][query_index]
            != parent.arrays[clean_array("recipe_id")][recipe_index]
            or arrays[query_array("observation_id")][query_index]
            != parent.arrays[observation_array("observation_id")][observation_index]
            or arrays[query_array("split_id")][query_index] != split_id
            or arrays[query_array("exact_curve_id")][query_index]
            != parent.arrays[observation_array("observation_id")][observation_index]
        ):
            raise ValueError("clean-group/observation join disagrees with grouped parent")
        observation_hash = observation_inputs_sha256(
            {
                name: parent.arrays[observation_input(name)][observation_index]
                for name in OBSERVATION_MODEL_INPUT_KEYS
            }
        )
        if arrays[query_array("observation_inputs_sha256")][query_index] != observation_hash:
            raise ValueError("universal query is bound to different observation tensors")
        sidecar_reference = np.ascontiguousarray(
            arrays[query_array("intensity_reference")][query_index : query_index + 1]
        )
        parent_reference = np.ascontiguousarray(
            parent.arrays[observation_array("intensity_reference")][
                observation_index : observation_index + 1
            ]
        )
        if (
            sidecar_reference.dtype != parent_reference.dtype
            or sidecar_reference.shape != parent_reference.shape
            or sidecar_reference.tobytes(order="C")
            != parent_reference.tobytes(order="C")
        ):
            raise ValueError(
                "persisted intensity reference disagrees byte-for-byte with grouped parent"
            )
        parent_audit = str(
            parent.arrays[observation_array("audit_json")][observation_index]
        )
        if (
            sha256(parent_audit.encode("utf-8")).hexdigest()
            != arrays[query_array("parent_observation_audit_sha256")][query_index]
            or not str(
                arrays[query_array("acceptance_sigma_source_id")][query_index]
            ).startswith(
                f"{parent.arrays[observation_array('acquisition_policy_id')][observation_index]}|"
            )
        ):
            raise ValueError(
                "exact-search curve evidence disagrees with parent observation provenance"
            )
        _validate_source_query_anchor(parent, sidecar, query_index, recipe_index, observation_index)
    label_recipes = arrays[branch_label("clean_recipe_index")]
    expected_recipes = arrays[query_array("parent_recipe_index")][
        arrays[branch_array("query_index")]
    ]
    if not np.array_equal(label_recipes, expected_recipes):
        raise ValueError("search supervision crossed clean-parent recipes")


def _validate_source_query_anchor(parent, sidecar, query_index, recipe_index, observation_index):
    arrays = sidecar.arrays
    audit = json.loads(str(arrays[query_array("universal_query_audit_json")][query_index]))
    geometry_sha = str(parent.arrays[clean_array("geometry_query_sha256")][recipe_index])
    amplitude_sha = str(parent.arrays[clean_array("amplitude_query_sha256")][recipe_index])
    source = [
        value
        for value in audit["topology_queries"]
        if value["geometry_query_sha256"] == geometry_sha
        and value["amplitude_query_sha256"] == amplitude_sha
    ]
    if len(source) != 1:
        raise ValueError("universal query lost or duplicated its source grouped query")
    source = source[0]
    query_rows = np.flatnonzero(arrays[branch_array("query_index")] == query_index)
    source_rows = query_rows[
        (arrays[branch_array("geometry_query_sha256")][query_rows] == geometry_sha)
        & (arrays[branch_array("amplitude_query_sha256")][query_rows] == amplitude_sha)
    ]
    parent_rows = np.flatnonzero(
        parent.arrays[candidate_context_array("recipe_index")] == recipe_index
    )
    parent_patterns = parent.arrays[candidate_input("branch_pattern_id")][parent_rows].reshape(-1)
    source_patterns = arrays[branch_array("pattern_id")][source_rows]
    if (
        source_rows.size != parent_rows.size
        or not np.array_equal(source_patterns, parent_patterns)
        or np.any(arrays[branch_array("topology_id")][source_rows] != source["topology_id"])
    ):
        raise ValueError("source topology branch catalog is missing, duplicated, or reordered")
    amplitude_query = amplitude_query_from_json(
        str(parent.arrays[clean_array("amplitude_query_canonical_json")][recipe_index]),
        amplitude_sha,
    )
    expected_amplitude = np.asarray(
        amplitude_query.model_embedding(
            float(parent.arrays[observation_array("intensity_reference")][observation_index])
        ),
        dtype=np.float32,
    )
    for sidecar_row, parent_row in zip(source_rows, parent_rows):
        if (
            arrays[branch_array("amplitude_constraint_sha256")][sidecar_row]
            != parent.arrays[candidate_context_array("amplitude_constraint_sha256")][
                parent_row
            ]
            or arrays[branch_array("amplitude_constraint_json")][sidecar_row]
            != parent.arrays[candidate_context_array("amplitude_constraint_json")][parent_row]
        ):
            raise ValueError("source branch amplitude context disagrees with grouped parent")
        for name in BRANCH_MODEL_INPUT_KEYS:
            expected = (
                expected_amplitude
                if name == "amplitude_bounds_embedding"
                else parent.arrays[candidate_input(name)][parent_row]
            )
            if not np.array_equal(arrays[branch_input(name)][sidecar_row], expected):
                raise ValueError("source query branch tensors disagree with grouped parent")


__all__ = [
    "V5SearchSupervisionOverlay",
    "read_v5_search_supervision_overlay",
]
