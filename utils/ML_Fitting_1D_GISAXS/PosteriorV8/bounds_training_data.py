"""Strict V4 bounds-first dataset adapter and distributed training steps."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
import tensorflow as tf

from .bounds_first_contract import (
    BOUNDS_EMBEDDING_DIM,
    BOUNDS_EMBEDDING_VERSION,
    LOCAL_TARGET_SEMANTICS,
)
from .bounds_first_shards import (
    PILOT_LIMITATIONS,
    SHARD_GENERATOR_VERSION,
    SHARD_SCHEMA_VERSION,
    SOLUTION_ONLY_PHASE,
    SPLIT_CODE,
    SPLIT_NAMES,
)
from .bounds_model_contract import MODEL_INPUT_KEYS
from .branch_catalog import decode_branch_pattern
from .canonical_branch_catalog import (
    CANONICAL_BRANCH_CATALOG_VERSION,
    canonical_branch_pattern_is_valid,
)
from .canonical_component_slots import CANONICAL_COMPONENT_SLOTS_VERSION
from .contract import MAX_COMPONENTS, NUM_TOPOLOGIES
from .build_bounds_first_shards import (
    BoundsFirstNumpyShard,
    canonical_json_bytes,
    file_sha256,
    load_shard,
)
from .local_target import local_labels_from_bounds_first
from .model import BRANCH_DIM, GLOBAL_FEATURE_DIM, POINT_FEATURE_DIM


BOUNDS_DATASET_ADAPTER_VERSION = "posterior_v8_bounds_first_solution_adapter_v3"
BOUNDS_DATASET_AUDIT_SCHEMA = "gisaxs.posterior_v8.bounds_training_dataset_audit/v3"
TRAINING_SPLITS = ("train", "tuning_validation")
EXCLUDED_TRAINING_SPLITS = ("calibration", "test")


@dataclass(frozen=True)
class BoundsTrainingDatasetAudit:
    payload: Mapping[str, object]
    shard_paths: tuple[Path, ...]
    train_count: int
    tuning_validation_count: int
    calibration_count: int
    test_count: int
    max_points: int

    @property
    def validation_count(self) -> int:
        return self.tuning_validation_count

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


def _validate_training_identity(metadata: Mapping[str, object]) -> dict[str, object]:
    identity = {
        "dataset_identity_sha256": metadata.get("dataset_identity_sha256"),
        "dataset_schema_version": metadata.get("dataset_schema_version"),
        "dataset_generator_version": metadata.get("dataset_generator_version"),
        "phase": metadata.get("phase"),
        "versions": metadata.get("versions"),
        "config": metadata.get("config"),
        "preprocessing_contract": metadata.get("preprocessing_contract"),
        "split_assignment": metadata.get("split_assignment"),
        "pilot_limitations": metadata.get("pilot_limitations"),
        "source_sha256_aggregate": metadata.get("source_sha256_aggregate"),
    }
    if identity["dataset_schema_version"] != SHARD_SCHEMA_VERSION:
        raise ValueError("trainer requires the V4 bounds-first shard schema")
    if identity["dataset_generator_version"] != SHARD_GENERATOR_VERSION:
        raise ValueError("trainer requires the V4 bounds-first shard generator")
    if identity["phase"] != SOLUTION_ONLY_PHASE:
        raise ValueError("trainer accepts only the bounds-first solution-only pilot")
    if tuple(identity["pilot_limitations"] or ()) != PILOT_LIMITATIONS:
        raise ValueError("bounds-first pilot limitations are missing or inconsistent")
    versions = identity["versions"]
    if not isinstance(versions, Mapping):
        raise ValueError("bounds-first scientific versions are missing")
    if versions.get("bounds_embedding") != BOUNDS_EMBEDDING_VERSION:
        raise ValueError("bounds-first embedding version is unsupported")
    if versions.get("local_target_semantics") != LOCAL_TARGET_SEMANTICS:
        raise ValueError("bounds-first local target semantics are unsupported")
    if versions.get("canonical_branch_catalog") != CANONICAL_BRANCH_CATALOG_VERSION:
        raise ValueError("bounds-first canonical branch catalog is unsupported")
    if versions.get("canonical_component_slots") != CANONICAL_COMPONENT_SLOTS_VERSION:
        raise ValueError("bounds-first component-slot canonicalization is unsupported")
    split = identity["split_assignment"]
    if (
        not isinstance(split, Mapping)
        or split.get("classification") != "recipe_grouped_interpolation_pilot"
        or split.get("no_parameter_guard_band") is not True
        or split.get("strong_holdout_or_ood_claim_allowed") is not False
        or split.get("all_observation_views_inherit_recipe_split") is not True
    ):
        raise ValueError("V4 split provenance cannot support this pilot trainer")
    return identity


def inspect_bounds_first_shards(
    shard_paths: Sequence[str | os.PathLike[str]],
    *,
    shard_loader: Callable[[str | os.PathLike[str]], BoundsFirstNumpyShard] | None = None,
) -> BoundsTrainingDatasetAudit:
    """Audit immutable V4 shards and bind training to two allowed splits."""

    if not shard_paths:
        raise ValueError("at least one V4 bounds-first NPZ shard is required")
    loader = load_shard if shard_loader is None else shard_loader
    paths = tuple(sorted(Path(value).resolve() for value in shard_paths))
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate V4 shard paths are not allowed")
    identity = None
    max_points = None
    rows_seen: set[tuple[int, int]] = set()
    recipe_splits: dict[bytes, int] = {}
    split_counts = {name: 0 for name in SPLIT_NAMES}
    records: list[dict[str, object]] = []
    for path in paths:
        if path.suffix != ".npz" or path.is_symlink() or not path.is_file():
            raise ValueError(f"V4 shard must be a regular .npz file: {path}")
        metadata_path = path.with_suffix(".json")
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise ValueError("V4 shard metadata must be a regular JSON file")
        shard = loader(path)
        current_identity = _validate_training_identity(shard.metadata)
        if file_sha256(path) != shard.metadata.get("npz_sha256"):
            raise ValueError("V4 shard checksum changed during training audit")
        if identity is None:
            identity = current_identity
        elif canonical_json_bytes(identity) != canonical_json_bytes(current_identity):
            raise ValueError("V4 shards do not share one schema/config/source identity")
        points = int(shard.arrays["x"].shape[1])
        max_points = points if max_points is None else max_points
        if points != max_points:
            raise ValueError("V4 shards disagree on padded curve length")
        for recipe, view, group, split_code in zip(
            shard.arrays["global_recipe_index"],
            shard.arrays["view_index"],
            shard.arrays["recipe_group_id"],
            shard.arrays["assigned_split"],
        ):
            key = (int(recipe), int(view))
            if key in rows_seen:
                raise ValueError("duplicate recipe/view row across V4 shards")
            rows_seen.add(key)
            group_key, code = bytes(group), int(split_code)
            prior = recipe_splits.setdefault(group_key, code)
            if prior != code:
                raise ValueError("observation views of one recipe cross split boundaries")
        shard_splits = {}
        for name, code in SPLIT_CODE.items():
            count = int(np.count_nonzero(shard.arrays["assigned_split"] == code))
            split_counts[name] += count
            shard_splits[name] = count
        if sum(shard_splits.values()) != shard.row_count:
            raise ValueError("V4 shard contains an unknown split code")
        records.append(
            {
                "path": str(path),
                "npz_sha256": shard.metadata["npz_sha256"],
                "metadata_sha256": file_sha256(metadata_path),
                "recipe_count": shard.recipe_count,
                "row_count": shard.row_count,
                "split_counts": shard_splits,
                "start_recipe_index": shard.metadata["shard"]["start_recipe_index"],
                "stop_recipe_index_exclusive": shard.metadata["shard"][
                    "stop_recipe_index_exclusive"
                ],
            }
        )
    assert identity is not None and max_points is not None
    if split_counts["train"] == 0 or split_counts["tuning_validation"] == 0:
        raise ValueError("both train and tuning_validation must contain V4 rows")
    fingerprint_payload = {"identity": identity, "shards": records}
    payload = {
        "schema_version": BOUNDS_DATASET_AUDIT_SCHEMA,
        "identity": identity,
        "shards": records,
        "shard_count": len(records),
        "row_count": sum(split_counts.values()),
        "split_counts": split_counts,
        "consumed_splits": list(TRAINING_SPLITS),
        "excluded_splits": list(EXCLUDED_TRAINING_SPLITS),
        "calibration_test_rows_consumed": 0,
        "max_points": max_points,
        "fingerprint_sha256": sha256(
            canonical_json_bytes(fingerprint_payload)
        ).hexdigest(),
    }
    return BoundsTrainingDatasetAudit(
        MappingProxyType(payload),
        paths,
        split_counts["train"],
        split_counts["tuning_validation"],
        split_counts["calibration"],
        split_counts["test"],
        max_points,
    )


def bounds_training_data(
    shard: BoundsFirstNumpyShard, *, split: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return V3 model inputs/native labels for one authorized split."""

    name = str(split).strip().lower()
    if name not in TRAINING_SPLITS:
        raise ValueError("bounds trainer may only consume train or tuning_validation")
    selected = shard.arrays["assigned_split"] == SPLIT_CODE[name]
    if np.any(~shard.arrays["truth_available"][selected]):
        raise ValueError("bounds trainer cannot consume rows without solution truth")
    count = int(np.count_nonzero(selected))
    topology_id = np.asarray(shard.arrays["topology_id"][selected], dtype=np.int32)
    pattern_id = np.asarray(
        shard.arrays["branch_pattern_id"][selected], dtype=np.int32
    )
    if any(
        not canonical_branch_pattern_is_valid(int(topology), int(pattern))
        for topology, pattern in zip(topology_id, pattern_id)
    ):
        raise ValueError("V4 training row uses a non-canonical physical branch")
    topology = np.zeros((count, NUM_TOPOLOGIES), dtype=np.float32)
    topology[np.arange(count), topology_id] = 1.0
    d_present = np.zeros((count, MAX_COMPONENTS), dtype=np.float32)
    resolution_present = np.zeros((count, 1), dtype=np.float32)
    for row, value in enumerate(pattern_id):
        flags, resolution = decode_branch_pattern(int(value))
        d_present[row] = flags
        resolution_present[row, 0] = resolution
    raw_labels = {
        "topology_id": topology_id,
        "branch_pattern_id": pattern_id,
        "target_local_unit": shard.arrays["target_local_unit"][selected],
        "active_dimension_mask": shard.arrays["active_dimension_mask"][selected],
        "local_varying_mask": shard.arrays["local_varying_mask"][selected],
    }
    labels = local_labels_from_bounds_first(
        raw_labels,
        target_coordinate_semantics=LOCAL_TARGET_SEMANTICS,
    )
    bounds = np.asarray(shard.arrays["bounds_embedding"][selected], dtype=np.float32)
    if bounds.shape != (count, BOUNDS_EMBEDDING_DIM):
        raise ValueError("bounds_embedding has the wrong V3 shape")
    presence = bounds[:, 2::3]
    if not np.array_equal(presence, labels["active_dimension_mask"]):
        raise ValueError("physical bounds presence disagrees with local active mask")
    inputs = {
        "x": np.asarray(shard.arrays["x"][selected], dtype=np.float32),
        "point_mask": np.asarray(shard.arrays["point_mask"][selected], dtype=np.bool_),
        "global_features": np.asarray(
            shard.arrays["global_features"][selected], dtype=np.float32
        ),
        "branch_topology": topology,
        "branch_d_present": d_present,
        "branch_resolution_present": resolution_present,
        "bounds_embedding": bounds,
        "active_dimension_mask": labels["active_dimension_mask"],
        "varying_dimension_mask": labels["varying_dimension_mask"],
    }
    if tuple(inputs) != MODEL_INPUT_KEYS:
        raise RuntimeError("V3 adapter input order drifted from its model contract")
    for field, value in (*inputs.items(), *labels.items()):
        array = np.asarray(value)
        if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
            raise ValueError(f"V3 training tensor {field!r} contains NaN/Inf")
    return inputs, labels


def _output_signature(max_points: int):
    inputs = {
        "x": tf.TensorSpec((None, max_points, POINT_FEATURE_DIM), tf.float32),
        "point_mask": tf.TensorSpec((None, max_points), tf.bool),
        "global_features": tf.TensorSpec((None, GLOBAL_FEATURE_DIM), tf.float32),
        "branch_topology": tf.TensorSpec((None, NUM_TOPOLOGIES), tf.float32),
        "branch_d_present": tf.TensorSpec((None, MAX_COMPONENTS), tf.float32),
        "branch_resolution_present": tf.TensorSpec((None, 1), tf.float32),
        "bounds_embedding": tf.TensorSpec((None, BOUNDS_EMBEDDING_DIM), tf.float32),
        "active_dimension_mask": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
        "varying_dimension_mask": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
    }
    labels = {
        "topology_id": tf.TensorSpec((None,), tf.int32),
        "branch_pattern_id": tf.TensorSpec((None,), tf.int32),
        "target_local": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
        "active_dimension_mask": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
        "varying_dimension_mask": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
    }
    return inputs, labels


def epoch_dataset(
    audit: BoundsTrainingDatasetAudit,
    config,
    split: str,
    *,
    epoch: int,
    shard_loader: Callable[[str | os.PathLike[str]], BoundsFirstNumpyShard] | None = None,
):
    if split not in TRAINING_SPLITS:
        raise ValueError("bounds trainer may only consume train or tuning_validation")
    loader = load_shard if shard_loader is None else shard_loader
    records = {item["path"]: item for item in audit.payload["shards"]}

    def batches():
        for path in audit.shard_paths:
            record = records[str(path)]
            if (
                file_sha256(path) != record["npz_sha256"]
                or file_sha256(path.with_suffix(".json"))
                != record["metadata_sha256"]
            ):
                raise ValueError("V4 shard provenance changed after training audit")
            shard = loader(path)
            inputs, labels = bounds_training_data(shard, split=split)
            count = labels["topology_id"].shape[0]
            for start in range(0, count, 1024):
                stop = min(start + 1024, count)
                yield (
                    {key: np.asarray(value[start:stop]) for key, value in inputs.items()},
                    {key: np.asarray(value[start:stop]) for key, value in labels.items()},
                )

    dataset = tf.data.Dataset.from_generator(
        batches, output_signature=_output_signature(config.max_points)
    ).unbatch()
    if split == "train":
        dataset = dataset.shuffle(
            min(config.shuffle_buffer, audit.train_count),
            seed=(config.seed + epoch) % (2**31 - 1),
            reshuffle_each_iteration=False,
        )
    dataset = dataset.batch(
        config.global_batch_size, drop_remainder=split == "train"
    )
    options = tf.data.Options()
    options.deterministic = True
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
    return dataset.with_options(options).prefetch(1)


__all__ = [
    "BOUNDS_DATASET_ADAPTER_VERSION",
    "BOUNDS_DATASET_AUDIT_SCHEMA",
    "BoundsTrainingDatasetAudit",
    "EXCLUDED_TRAINING_SPLITS",
    "TRAINING_SPLITS",
    "bounds_training_data",
    "epoch_dataset",
    "inspect_bounds_first_shards",
]
