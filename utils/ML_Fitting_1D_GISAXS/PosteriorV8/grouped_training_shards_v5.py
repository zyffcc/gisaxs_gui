"""Cross-shard identity audit and lazy recipe references for V5.1 training."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import tensorflow as tf

from .candidate_supervision_v5 import CANDIDATE_SUPERVISION_TENSOR_KEYS, SEARCH_OUTCOME_CODE
from .grouped_dataset_v5 import (
    V5GroupedDataset,
    clean_array,
    observation_input,
    read_v5_grouped_dataset,
)
from .grouped_training_data_v5 import (
    RecipeMacroAdapter,
    V5GroupedTrainingConfig,
)
from .grouped_training_inventory_v5 import (
    V5GroupedShard,
    V5GroupedTrainingAudit,
    V5_MAX_RESIDENT_ARRAY_BYTES,
    V5_SHARD_LOADING_POLICY,
    audit_sidecar_training_expansion,
    preflight_resident_array_bytes,
)
from .formal_production_search_plan_v5 import (
    V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE,
    V5_FORMAL_PRODUCTION_TUNING_CONSUMER_ROLE,
)
from .search_evidence_receipt_v5 import (
    evidence_receipt_path_for_sidecar,
    read_v5_search_evidence_receipt,
)
from .search_supervision_overlay_v5 import V5SearchSupervisionOverlay
from .search_supervision_sidecar_v5 import read_v5_search_supervision_sidecar


def _paths(values: str | os.PathLike[str] | Sequence[str | os.PathLike[str]]) -> tuple[Path, ...]:
    if isinstance(values, (str, os.PathLike)):
        raw = (values,)
    else:
        raw = tuple(values)
    result = tuple(Path(value).resolve() for value in raw)
    if not result or len(set(result)) != len(result):
        raise ValueError("dataset paths must be non-empty and unique within each role")
    return result


def _sidecar_paths(
    values: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None,
) -> tuple[Path, ...]:
    if values is None:
        return ()
    return _paths(values)


def _recipe_generator_identity(dataset: V5GroupedDataset) -> tuple[str, str, str]:
    manifest_identity = dataset.manifest.get("clean_recipe_identity")
    if not isinstance(manifest_identity, Mapping):
        raise ValueError("grouped artifact has no clean recipe identity")
    try:
        schema = str(manifest_identity["schema_version"])
        generator = str(manifest_identity["generator_version"])
    except (KeyError, TypeError) as exc:
        raise ValueError("grouped artifact has no complete recipe identity") from exc
    if not schema or not generator:
        raise ValueError("grouped artifact has an empty recipe identity")

    exact_paths = set()
    for encoded in dataset.arrays[clean_array("recipe_canonical_json")]:
        try:
            payload = json.loads(str(encoded))
            exact_path = str(payload["exact_forward_path"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("recipe JSON has no exact-forward identity") from exc
        if not exact_path:
            raise ValueError("recipe JSON has an empty exact-forward identity")
        exact_paths.add(exact_path)
    if len(exact_paths) != 1:
        raise ValueError("one grouped artifact mixes exact-forward identities")
    return schema, generator, exact_paths.pop()


class V5GroupedShardCollection:
    """Keep shard arrays separate while exposing stable global recipe references."""

    def __init__(
        self,
        train_paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        validation_paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        config: V5GroupedTrainingConfig,
        *,
        train_sidecar_paths: (
            str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None
        ) = None,
        validation_sidecar_paths: (
            str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None
        ) = None,
    ):
        train = _paths(train_paths)
        validation = _paths(validation_paths)
        train_sidecars = _sidecar_paths(train_sidecar_paths)
        validation_sidecars = _sidecar_paths(validation_sidecar_paths)
        supplied = bool(train_sidecars or validation_sidecars)
        if supplied and not (train_sidecars and validation_sidecars):
            raise ValueError(
                "train and validation search-sidecar subsets must both be supplied"
            )
        if config.full_epochs and not supplied:
            raise ValueError(
                "full search-yield stage requires checked train and validation search "
                "sidecars; parent unverified branches cannot be negatives"
            )
        if not config.allow_same_split_for_smoke and set(train) & set(validation):
            raise ValueError("train and validation artifact paths overlap")
        sidecar_values = set((*train_sidecars, *validation_sidecars))
        if (
            not config.allow_same_split_for_smoke
            and len(sidecar_values) != len(train_sidecars) + len(validation_sidecars)
        ):
            raise ValueError("train and validation search-sidecar paths overlap")
        self.resident_array_bytes = preflight_resident_array_bytes(
            (*train, *validation, *train_sidecars, *validation_sidecars)
        )
        self.shards: list[V5GroupedShard] = []
        offset = 0
        for role, paths, expected_split in (
            ("train", train, config.train_split),
            ("validation", validation, config.validation_split),
        ):
            for path in paths:
                dataset, receipt = read_v5_grouped_dataset(path)
                observed_splits = set(dataset.arrays[clean_array("split_id")].tolist())
                if observed_splits != {expected_split}:
                    raise ValueError(
                        f"{role} artifact clean split_id must be exactly {expected_split!r}"
                    )
                self.shards.append(
                    V5GroupedShard(
                        path,
                        dataset,
                        receipt.artifact_sha256,
                        receipt.manifest_sha256,
                        role,
                        offset,
                        None,
                    )
                )
                offset += dataset.recipe_count
        self._attach_sidecars(
            "train", train_sidecars, require_evidence=bool(config.full_epochs)
        )
        self._attach_sidecars(
            "validation",
            validation_sidecars,
            require_evidence=bool(config.full_epochs),
        )
        self._adapters = tuple(RecipeMacroAdapter(value.dataset) for value in self.shards)
        self._validate_cross_shard(config)
        self.train_recipes = self._global_recipes("train")
        self.validation_recipes = self._global_recipes("validation")
        self.full_train_recipes = self._global_recipes("train", require_sidecar=True)
        self.full_validation_recipes = self._global_recipes(
            "validation", require_sidecar=True
        )

    def _attach_sidecars(
        self, role: str, paths: Sequence[Path], *, require_evidence: bool
    ) -> None:
        for sidecar_path in paths:
            sidecar, receipt = read_v5_search_supervision_sidecar(sidecar_path)
            binding = sidecar.manifest["parent_grouped_artifact"]
            matching = [
                index
                for index, shard in enumerate(self.shards)
                if shard.role == role
                and shard.artifact_sha256 == binding["artifact_sha256"]
                and shard.manifest_sha256 == binding["manifest_sha256"]
            ]
            if len(matching) != 1:
                raise ValueError(
                    f"{role} search sidecar does not uniquely bind one supplied parent shard"
                )
            index = matching[0]
            shard = self.shards[index]
            if shard.full_overlay is not None:
                raise ValueError("more than one search sidecar binds the same parent shard")
            overlay = V5SearchSupervisionOverlay(
                parent_path=shard.path,
                sidecar_path=sidecar_path,
                parent=shard.dataset,
                sidecar=sidecar,
                parent_artifact_sha256=shard.artifact_sha256,
                sidecar_artifact_sha256=receipt.artifact_sha256,
            )
            evidence_path = evidence_receipt_path_for_sidecar(sidecar_path)
            evidence_receipt = None
            if require_evidence and not evidence_path.exists():
                raise ValueError(
                    "full search-yield stage requires a task-bound evidence receipt "
                    f"for sidecar {sidecar_path}"
                )
            if evidence_path.exists() or require_evidence:
                evidence_receipt = read_v5_search_evidence_receipt(
                    evidence_path,
                    parent_dataset_path=shard.path,
                    sidecar_path=sidecar_path,
                    require_training_eligible=require_evidence,
                    expected_consumer_role=(
                        V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE
                        if role == "train"
                        else V5_FORMAL_PRODUCTION_TUNING_CONSUMER_ROLE
                    )
                    if require_evidence
                    else None,
                )
            self.shards[index] = replace(
                shard,
                full_overlay=overlay,
                full_evidence_receipt=evidence_receipt,
            )

    def _validate_cross_shard(self, config: V5GroupedTrainingConfig) -> None:
        first = self.shards[0]
        contract = first.dataset.manifest["contract_bundle_sha256"]
        source = first.dataset.manifest["source_sha256"]
        oracle = first.dataset.manifest["oracle_protocol"]
        max_points = first.dataset.arrays[observation_input("x")].shape[1]
        generator = _recipe_generator_identity(first.dataset)
        split_plans: set[str] = set()
        designs: set[str] = set()
        group_ids: list[str] = []
        sobol_indices: list[int] = []
        dataset_ids: list[str] = []
        full_protocols: set[str] = set()
        sidecar_contracts: set[str] = set()
        launch_source_bundles: set[str] = set()
        executor_source_bundles: set[str] = set()
        for shard in self.shards:
            dataset = shard.dataset
            if dataset.manifest["contract_bundle_sha256"] != contract:
                raise ValueError("grouped shards use different V5.1 contracts")
            if dataset.manifest["source_sha256"] != source:
                raise ValueError("grouped shards were built from different source bundles")
            if dataset.manifest["oracle_protocol"] != oracle:
                raise ValueError("grouped shards use different oracle protocols")
            if dataset.arrays[observation_input("x")].shape[1] != max_points:
                raise ValueError("grouped shards use different padded curve lengths")
            if _recipe_generator_identity(dataset) != generator:
                raise ValueError("grouped shards use different recipe generators")
            plans = np.unique(dataset.arrays[clean_array("split_plan_sha256")])
            shard_designs = np.unique(dataset.arrays[clean_array("sobol_design_sha256")])
            if plans.size != 1 or shard_designs.size != 1:
                raise ValueError("each shard must have one Sobol plan and design identity")
            split_plans.add(str(plans[0]))
            designs.add(str(shard_designs[0]))
            group_ids.extend(str(value) for value in dataset.arrays[clean_array("clean_group_id")])
            sobol_indices.extend(int(value) for value in dataset.arrays[clean_array("sobol_index")])
            dataset_ids.append(str(dataset.manifest["dataset_id"]))
            if shard.full_overlay is not None:
                full_protocols.add(shard.full_overlay.protocol_sha256)
                sidecar_contracts.add(
                    str(shard.full_overlay.sidecar.manifest["contract_bundle_sha256"])
                )
                if shard.full_evidence_receipt is not None:
                    launch_source_bundles.add(
                        str(
                            shard.full_evidence_receipt.manifest[
                                "launch_source_bundle_sha256"
                            ]
                        )
                    )
                    executor_source_bundles.add(
                        str(
                            shard.full_evidence_receipt.manifest[
                                "executor_source_bundle_sha256"
                            ]
                        )
                    )
        engineering = config.allow_same_split_for_smoke
        if not engineering and (split_plans == {""} or designs == {""}):
            raise ValueError("formal train/validation shards require Sobol plan and design hashes")
        if len(split_plans) != 1 or len(designs) != 1:
            raise ValueError(
                "train/validation shards use different Sobol plan or design identities"
            )
        if not engineering and any(value < 0 for value in sobol_indices):
            raise ValueError("formal grouped shards require non-negative Sobol indices")
        if not engineering and len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("formal grouped dataset IDs must be unique")
        if not engineering and len(group_ids) != len(set(group_ids)):
            raise ValueError("clean_group_id overlaps across train/validation shards")
        if not engineering and len(sobol_indices) != len(set(sobol_indices)):
            raise ValueError("Sobol index overlaps across train/validation shards")
        if full_protocols and len(full_protocols) != 1:
            raise ValueError(
                "train/validation search sidecars use different frozen search protocols"
            )
        if sidecar_contracts and len(sidecar_contracts) != 1:
            raise ValueError("search sidecars use different model/training contracts")
        if len(launch_source_bundles) > 1 or len(executor_source_bundles) > 1:
            raise ValueError(
                "train/validation search evidence was produced by different source bundles"
            )
        self.contract_bundle_sha256 = str(contract)
        self.split_plan_sha256 = next(iter(split_plans))
        self.sobol_design_sha256 = next(iter(designs))
        self.recipe_generator_identity = generator
        self.max_points = int(max_points)
        self.full_search_protocol_sha256 = (
            next(iter(full_protocols)) if full_protocols else None
        )

    def _global_recipes(self, role: str, *, require_sidecar: bool = False) -> np.ndarray:
        values = [
            np.arange(shard.recipe_offset, shard.recipe_offset + shard.dataset.recipe_count)
            for shard in self.shards
            if shard.role == role
            and (
                not require_sidecar
                or (
                    shard.full_overlay is not None
                    and shard.full_evidence_receipt is not None
                    and shard.full_evidence_receipt.full_training_eligible
                )
            )
        ]
        if not values:
            return np.empty((0,), dtype=np.int32)
        return np.concatenate(values).astype(np.int32, copy=False)

    def _locate(self, recipe: int) -> tuple[int, int]:
        for index, shard in enumerate(self.shards):
            local = int(recipe) - shard.recipe_offset
            if 0 <= local < shard.dataset.recipe_count:
                return index, local
        raise ValueError("global recipe reference is outside the checked shards")

    def numpy_batch(self, recipes: Sequence[int], *, phase: str):
        parts = []
        for recipe in recipes:
            shard_index, local_recipe = self._locate(int(recipe))
            shard = self.shards[shard_index]
            if phase == "warmup":
                inputs, labels = self._adapters[shard_index].numpy_batch(
                    (local_recipe,), phase="warmup"
                )
            elif phase == "full":
                if (
                    shard.full_overlay is None
                    or shard.full_evidence_receipt is None
                    or not shard.full_evidence_receipt.full_training_eligible
                ):
                    raise ValueError(
                        "full batches require a checked search sidecar and an eligible "
                        "task-bound evidence receipt for every parent shard"
                    )
                inputs, labels = shard.full_overlay.numpy_batch(
                    (local_recipe,),
                    phase="full",
                    require_completed_catalog=True,
                    require_positive_and_negative=False,
                )
            else:
                raise ValueError("phase must be warmup or full")
            labels = dict(labels)
            labels["clean_recipe_index"] = np.full(
                labels["clean_recipe_index"].shape,
                int(recipe),
                dtype=np.int32,
            )
            parts.append((inputs, labels))
        inputs = {
            name: np.concatenate([value[0][name] for value in parts], axis=0)
            for name in parts[0][0]
        }
        labels = {
            name: np.concatenate([value[1][name] for value in parts], axis=0)
            for name in CANDIDATE_SUPERVISION_TENSOR_KEYS
        }
        return inputs, labels

    def tensor_batch(self, recipes: Sequence[int], *, phase: str):
        inputs, labels = self.numpy_batch(recipes, phase=phase)
        return (
            {name: tf.convert_to_tensor(value) for name, value in inputs.items()},
            {name: tf.convert_to_tensor(value) for name, value in labels.items()},
        )

    def outcome_counts(self, role: str, *, source: str = "parent") -> dict[str, int]:
        result = {name: 0 for name in SEARCH_OUTCOME_CODE}
        for shard, adapter in zip(self.shards, self._adapters):
            if shard.role != role:
                continue
            if source == "parent":
                recipes = np.arange(shard.dataset.recipe_count, dtype=np.int32)
                counts = adapter.outcome_counts(recipes)
            elif source == "sidecar":
                if shard.full_overlay is None:
                    continue
                counts = shard.full_overlay.outcome_counts()
            else:
                raise ValueError("source must be parent or sidecar")
            for name, value in counts.items():
                result[name] += value
        return result

    def protocol_sha256(self, role: str) -> str:
        values = {
            adapter.protocol_sha256(np.arange(shard.dataset.recipe_count, dtype=np.int32))
            for shard, adapter in zip(self.shards, self._adapters)
            if shard.role == role
        }
        if len(values) != 1:
            raise ValueError(f"{role} shards use different frozen search protocols")
        return values.pop()

    def artifact_audit(self, role: str) -> tuple[Mapping[str, object], ...]:
        return tuple(
            {
                name: value
                for name, value in shard.audit_payload().items()
                if name != "search_sidecar"
            }
            for shard in self.shards
            if shard.role == role
        )

    def sidecar_artifact_audit(self, role: str) -> tuple[Mapping[str, object], ...]:
        values = (
            shard.sidecar_audit_payload()
            for shard in self.shards
            if shard.role == role
        )
        return tuple(value for value in values if value is not None)

    @property
    def has_full_sidecars(self) -> bool:
        return all(
            any(
                shard.role == role and shard.full_overlay is not None
                for shard in self.shards
            )
            for role in ("train", "validation")
        )

    @property
    def has_full_evidence_receipts(self) -> bool:
        labeled = tuple(
            shard for shard in self.shards if shard.full_overlay is not None
        )
        return self.has_full_sidecars and bool(labeled) and all(
            shard.full_evidence_receipt is not None
            and shard.full_evidence_receipt.full_training_eligible
            for shard in labeled
        )

    def input_artifact_audit(self) -> tuple[Mapping[str, object], ...]:
        result: list[Mapping[str, object]] = []
        for shard in self.shards:
            result.append(
                {
                    "path": str(shard.path),
                    "role": shard.role,
                    "kind": "grouped_parent",
                    "artifact_sha256": shard.artifact_sha256,
                    "manifest_sha256": shard.manifest_sha256,
                }
            )
            if shard.full_overlay is not None:
                result.append(
                    {
                        "path": str(shard.full_overlay.sidecar_path),
                        "role": shard.role,
                        "kind": "frozen_search_sidecar",
                        "artifact_sha256": shard.full_overlay.sidecar_artifact_sha256,
                        "manifest_sha256": shard.full_overlay.sidecar.manifest[
                            "manifest_sha256"
                        ],
                        "protocol_sha256": shard.full_overlay.protocol_sha256,
                        "counts": dict(shard.full_overlay.sidecar.manifest["counts"]),
                    }
                )
                if shard.full_evidence_receipt is not None:
                    result.append(
                        {
                            "path": str(shard.full_evidence_receipt.path),
                            "role": shard.role,
                            "kind": "task_bound_search_evidence_receipt",
                            "artifact_sha256": shard.full_evidence_receipt.file_sha256,
                            "receipt_sha256": shard.full_evidence_receipt.manifest[
                                "receipt_sha256"
                            ],
                        }
                    )
        return tuple(result)


def inspect_v5_grouped_training(
    train_dataset_paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    validation_dataset_paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    config: V5GroupedTrainingConfig,
    *,
    train_sidecar_paths: (
        str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None
    ) = None,
    validation_sidecar_paths: (
        str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None
    ) = None,
    replicas: int = 1,
) -> tuple[V5GroupedShardCollection, V5GroupedTrainingAudit]:
    if not isinstance(config, V5GroupedTrainingConfig):
        raise TypeError("config must be V5GroupedTrainingConfig")
    if isinstance(replicas, bool) or int(replicas) != replicas or int(replicas) < 1:
        raise ValueError("replicas must be a positive integer")
    collection = V5GroupedShardCollection(
        train_dataset_paths,
        validation_dataset_paths,
        config,
        train_sidecar_paths=train_sidecar_paths,
        validation_sidecar_paths=validation_sidecar_paths,
    )
    train_counts = collection.outcome_counts("train", source="parent")
    validation_counts = collection.outcome_counts("validation", source="parent")
    train_full_counts = (
        collection.outcome_counts("train", source="sidecar")
        if collection.has_full_sidecars
        else None
    )
    validation_full_counts = (
        collection.outcome_counts("validation", source="sidecar")
        if collection.has_full_sidecars
        else None
    )
    expansion_safety = audit_sidecar_training_expansion(
        collection.shards,
        train_recipes_per_replica=config.recipes_per_replica,
        validation_recipes_per_batch=config.validation_recipes_per_batch,
        allow_unsafe_for_engineering=(
            config.allow_unsafe_sidecar_expansion_for_engineering
        ),
    )
    full_permitted = (
        train_full_counts is not None
        and validation_full_counts is not None
        and collection.has_full_evidence_receipts
        and all(
            values["unverified"] == 0
            and values["compatible_found"] > 0
            and values["no_compatible_found_within_frozen_search_budget"] > 0
            for values in (train_full_counts, validation_full_counts)
        )
    )
    if config.full_epochs and not full_permitted:
        raise ValueError(
            "full search-yield stage requires sidecar-only completed-search positive and "
            "negative labels plus task-bound executor evidence receipts in both train "
            "and validation splits; any unverified branch or pilot-only branch rejects "
            "the stage"
        )
    global_recipes = config.recipes_per_replica * int(replicas)
    available_steps = len(collection.train_recipes) // global_recipes
    selected_steps = available_steps if config.steps_per_epoch is None else config.steps_per_epoch
    if config.warmup_epochs and (
        selected_steps < 1 or selected_steps > available_steps
    ):
        raise ValueError("requested steps exceed complete equal-recipe batches")
    full_available_steps = len(collection.full_train_recipes) // global_recipes
    full_selected_steps = (
        full_available_steps
        if config.full_steps_per_epoch is None
        else config.full_steps_per_epoch
    )
    if config.full_epochs and (
        full_selected_steps < 1 or full_selected_steps > full_available_steps
    ):
        raise ValueError(
            "requested full-stage steps exceed complete equal-recipe sidecar batches"
        )
    audit = V5GroupedTrainingAudit(
        train_artifacts=collection.artifact_audit("train"),
        validation_artifacts=collection.artifact_audit("validation"),
        train_sidecar_artifacts=collection.sidecar_artifact_audit("train"),
        validation_sidecar_artifacts=collection.sidecar_artifact_audit("validation"),
        train_recipe_count=len(collection.train_recipes),
        validation_recipe_count=len(collection.validation_recipes),
        train_full_recipe_count=len(collection.full_train_recipes),
        validation_full_recipe_count=len(collection.full_validation_recipes),
        train_outcome_counts=train_counts,
        validation_outcome_counts=validation_counts,
        train_full_outcome_counts=train_full_counts,
        validation_full_outcome_counts=validation_full_counts,
        train_protocol_sha256=collection.protocol_sha256("train"),
        validation_protocol_sha256=collection.protocol_sha256("validation"),
        full_search_protocol_sha256=collection.full_search_protocol_sha256,
        phase_data_sources={
            "warmup": {
                "train": "grouped_parent_known_positive",
                "validation": "grouped_parent_known_positive",
            },
            "full": {
                "train": (
                    "frozen_search_sidecar_plus_task_bound_executor_receipt_labeled_parent_subset"
                    if collection.has_full_evidence_receipts
                    else "unavailable"
                ),
                "validation": (
                    "frozen_search_sidecar_plus_task_bound_executor_receipt_labeled_parent_subset"
                    if collection.has_full_evidence_receipts
                    else "unavailable"
                ),
            },
        },
        split_plan_sha256=collection.split_plan_sha256,
        sobol_design_sha256=collection.sobol_design_sha256,
        recipe_generator_identity=collection.recipe_generator_identity,
        max_points=collection.max_points,
        replicas=int(replicas),
        global_recipes_per_step=global_recipes,
        available_steps_per_epoch=available_steps,
        selected_steps_per_epoch=selected_steps if config.warmup_epochs else 0,
        full_available_steps_per_epoch=full_available_steps,
        full_selected_steps_per_epoch=(
            full_selected_steps if collection.has_full_evidence_receipts else 0
        ),
        resident_array_bytes=collection.resident_array_bytes,
        resident_array_limit_bytes=V5_MAX_RESIDENT_ARRAY_BYTES,
        shard_loading_policy=V5_SHARD_LOADING_POLICY,
        paper_scale_streaming_gate=(
            "required_before_declared_arrays_exceed_resident_array_limit"
        ),
        sidecar_expansion_safety=expansion_safety,
        full_stage_permitted=full_permitted,
    )
    return collection, audit


__all__ = [
    "V5GroupedShard",
    "V5GroupedShardCollection",
    "V5GroupedTrainingAudit",
    "V5_MAX_RESIDENT_ARRAY_BYTES",
    "V5_SHARD_LOADING_POLICY",
    "inspect_v5_grouped_training",
]
