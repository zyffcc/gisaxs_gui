"""Dataset and distributed-step plumbing for local-target v2 training."""

from __future__ import annotations

from math import isfinite
from typing import Mapping
import time

import numpy as np
import tensorflow as tf

from .dataset import RANGE_GENERATOR_VERSION
from .local_target import (
    TRUTH_CENTERED_RANGE_CONSTRUCTION_SEMANTICS,
    local_labels_from_global_box,
)
from .model import BRANCH_DIM, GLOBAL_FEATURE_DIM, POINT_FEATURE_DIM
from .proposal_training_audit import DatasetAudit, _file_sha256
from .training_objective_v2 import compute_local_training_objective


LOCAL_DATASET_ADAPTER_VERSION = "posterior_v8_phase2_global_box_local_target_adapter_v1"


def range_construction_semantics(audit: DatasetAudit) -> str:
    versions = audit.payload["identity"].get("versions")
    if not isinstance(versions, Mapping) or versions.get("range_generator") != RANGE_GENERATOR_VERSION:
        raise ValueError("dataset does not declare the supported Phase-2 range generator")
    return TRUTH_CENTERED_RANGE_CONSTRUCTION_SEMANTICS


def require_scientific_data_contract(
    audit: DatasetAudit, *, allow_truth_centered_range_pilot: bool
) -> str:
    semantics = range_construction_semantics(audit)
    if semantics == TRUTH_CENTERED_RANGE_CONSTRUCTION_SEMANTICS and not (
        allow_truth_centered_range_pilot
    ):
        raise ValueError(
            "Phase-2 ranges are truth-centered and may leak target position; local-target v2 "
            "requires bounds-first data unless allow_truth_centered_range_pilot=True is "
            "explicitly selected for an engineering comparison"
        )
    return semantics


def _output_signature(max_points: int):
    inputs = {
        "x": tf.TensorSpec((None, max_points, POINT_FEATURE_DIM), tf.float32),
        "point_mask": tf.TensorSpec((None, max_points), tf.bool),
        "global_features": tf.TensorSpec((None, GLOBAL_FEATURE_DIM), tf.float32),
        "branch_topology": tf.TensorSpec((None, 34), tf.float32),
        "branch_d_present": tf.TensorSpec((None, 4), tf.float32),
        "branch_resolution_present": tf.TensorSpec((None, 1), tf.float32),
        "branch_low": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
        "branch_high": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
        "active_dimension_mask": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
    }
    labels = {
        "topology_id": tf.TensorSpec((None,), tf.int32),
        "branch_pattern_id": tf.TensorSpec((None,), tf.int32),
        "target_local": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
        "active_dimension_mask": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
        "varying_dimension_mask": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
        "branch_low": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
        "branch_high": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
    }
    return inputs, labels


def epoch_dataset(
    audit: DatasetAudit,
    config,
    split: str,
    *,
    epoch: int,
    shard_loader,
    range_semantics: str,
):
    if split not in {"train", "validation"}:
        raise ValueError("proposal trainer may only consume train or validation rows")
    records = {item["path"]: item for item in audit.payload["shards"]}

    def batches():
        for path in audit.shard_paths:
            shard = shard_loader(path)
            record = records[str(path)]
            if (
                shard.metadata.get("npz_sha256") != record["npz_sha256"]
                or _file_sha256(path.with_suffix(".json")) != record["metadata_sha256"]
            ):
                raise ValueError("shard provenance changed after run audit")
            inputs, global_labels = shard.training_data(split=split)
            labels = local_labels_from_global_box(
                inputs,
                global_labels,
                range_construction_semantics=range_semantics,
            )
            for name, value in (*inputs.items(), *labels.items()):
                array = np.asarray(value)
                if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
                    raise ValueError(f"training tensor {name!r} contains NaN/Inf")
            for start in range(0, labels["topology_id"].shape[0], 1024):
                stop = min(start + 1024, labels["topology_id"].shape[0])
                yield (
                    {name: np.asarray(value[start:stop]) for name, value in inputs.items()},
                    {name: np.asarray(value[start:stop]) for name, value in labels.items()},
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
        config.global_batch_size,
        drop_remainder=split == "train",
    )
    options = tf.data.Options()
    options.deterministic = True
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    return dataset.with_options(options).prefetch(1)


def _assert_input_label_coordinates(inputs, labels):
    assertions = (
        tf.debugging.assert_equal(inputs["branch_low"], labels["branch_low"]),
        tf.debugging.assert_equal(inputs["branch_high"], labels["branch_high"]),
        tf.debugging.assert_equal(
            inputs["active_dimension_mask"], labels["active_dimension_mask"]
        ),
    )
    with tf.control_dependencies(assertions):
        return (
            {name: tf.identity(value) for name, value in inputs.items()},
            {name: tf.identity(value) for name, value in labels.items()},
        )


def training_steps(strategy, model, optimizer, objective_config, gradient_clip_norm):
    replicas = tf.cast(strategy.num_replicas_in_sync, tf.float32)
    mixed = isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer)

    def replica_train(inputs, labels):
        inputs, labels = _assert_input_label_coordinates(inputs, labels)
        with tf.GradientTape() as tape:
            metrics = compute_local_training_objective(
                model(inputs, training=True), labels, objective_config
            )
            gradient_loss = metrics["loss"] / replicas
            tape_loss = optimizer.get_scaled_loss(gradient_loss) if mixed else gradient_loss
        gradients = tape.gradient(tape_loss, model.trainable_variables)
        if any(value is None for value in gradients):
            raise RuntimeError("local-target objective left disconnected model variables")
        if mixed:
            gradients = optimizer.get_unscaled_gradients(gradients)
        finite_checks = (
            []
            if mixed
            else [
                tf.debugging.assert_all_finite(
                    value.values if isinstance(value, tf.IndexedSlices) else value,
                    "local proposal gradient contains NaN/Inf",
                )
                for value in gradients
            ]
        )
        iteration_before = tf.identity(optimizer.iterations)
        with tf.control_dependencies(finite_checks):
            gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip_norm)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        with tf.control_dependencies(
            [
                tf.debugging.assert_all_finite(value, "local proposal weight contains NaN/Inf")
                for value in model.trainable_variables
            ]
        ):
            result = {name: tf.identity(value) for name, value in metrics.items()}
            result["optimizer_update_applied"] = tf.cast(
                optimizer.iterations > iteration_before, tf.float32
            )
            return result

    def replica_validation(inputs, labels):
        inputs, labels = _assert_input_label_coordinates(inputs, labels)
        return compute_local_training_objective(
            model(inputs, training=False), labels, objective_config
        )

    @tf.function
    def train_step(inputs, labels):
        values = strategy.run(replica_train, args=(inputs, labels))
        return {
            name: strategy.reduce(tf.distribute.ReduceOp.MEAN, value, axis=None)
            for name, value in values.items()
        }

    @tf.function
    def distributed_validation_step(inputs, labels):
        values = strategy.run(replica_validation, args=(inputs, labels))
        return {
            name: strategy.reduce(tf.distribute.ReduceOp.MEAN, value, axis=None)
            for name, value in values.items()
        }

    @tf.function
    def coordinator_validation_step(inputs, labels):
        return replica_validation(inputs, labels)

    return train_step, distributed_validation_step, coordinator_validation_step


def _distribute_full_batch(strategy, inputs, labels, global_batch_size: int):
    replica_count = int(strategy.num_replicas_in_sync)
    if replica_count < 1 or global_batch_size % replica_count:
        raise ValueError("full validation batch must divide evenly across replicas")
    local_batch_size = global_batch_size // replica_count

    def values_for(context, values):
        start = context.replica_id_in_sync_group * local_batch_size
        stop = start + local_batch_size
        return tf.nest.map_structure(lambda value: value[start:stop], values)

    return (
        strategy.experimental_distribute_values_from_function(
            lambda context: values_for(context, inputs)
        ),
        strategy.experimental_distribute_values_from_function(
            lambda context: values_for(context, labels)
        ),
    )


def run_hybrid_validation(
    dataset,
    *,
    strategy,
    global_batch_size: int,
    expected_examples: int,
    distributed_step,
    coordinator_step,
):
    """Distribute full batches and evaluate the unique tail on the coordinator."""

    totals = None
    seen = distributed_examples = coordinator_tail_examples = 0
    steps = 0
    started = time.monotonic()
    for inputs, labels in dataset:
        count = int(tf.shape(labels["topology_id"])[0].numpy())
        if count == global_batch_size:
            dist_inputs, dist_labels = _distribute_full_batch(
                strategy, inputs, labels, global_batch_size
            )
            tensors = distributed_step(dist_inputs, dist_labels)
            distributed_examples += count
        else:
            if coordinator_tail_examples or count >= global_batch_size:
                raise RuntimeError("validation produced more than one or an invalid tail batch")
            tensors = coordinator_step(inputs, labels)
            coordinator_tail_examples = count
        metrics = {name: float(value.numpy()) for name, value in tensors.items()}
        if not all(isfinite(value) for value in metrics.values()):
            raise FloatingPointError("validation step returned a NaN/Inf metric")
        if totals is None:
            totals = {name: 0.0 for name in metrics}
        for name, value in metrics.items():
            totals[name] += value * count
        seen += count
        steps += 1
    if totals is None or seen != expected_examples:
        raise RuntimeError("validation dataset did not match its audited sample count")
    return (
        {name: value / seen for name, value in totals.items()},
        time.monotonic() - started,
        {
            "validation_steps": steps,
            "validation_examples": seen,
            "validation_distributed_examples": distributed_examples,
            "validation_coordinator_tail_examples": coordinator_tail_examples,
        },
    )


__all__ = [
    "LOCAL_DATASET_ADAPTER_VERSION",
    "epoch_dataset",
    "range_construction_semantics",
    "require_scientific_data_contract",
    "run_hybrid_validation",
    "training_steps",
]
