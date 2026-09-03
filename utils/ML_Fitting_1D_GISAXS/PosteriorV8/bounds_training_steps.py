"""TensorFlow steps for V3 bounds-first proposal training."""

from __future__ import annotations

import tensorflow as tf

from .local_training_data import run_hybrid_validation
from .training_objective_v3 import compute_bounds_training_objective


def _assert_input_label_coordinates(inputs, labels):
    assertions = (
        tf.debugging.assert_equal(
            inputs["active_dimension_mask"], labels["active_dimension_mask"]
        ),
        tf.debugging.assert_equal(
            inputs["varying_dimension_mask"], labels["varying_dimension_mask"]
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
            metrics = compute_bounds_training_objective(
                model(inputs, training=True), labels, objective_config
            )
            gradient_loss = metrics["loss"] / replicas
            tape_loss = optimizer.get_scaled_loss(gradient_loss) if mixed else gradient_loss
        gradients = tape.gradient(tape_loss, model.trainable_variables)
        if any(value is None for value in gradients):
            raise RuntimeError("bounds-first objective left disconnected model variables")
        if mixed:
            gradients = optimizer.get_unscaled_gradients(gradients)
        finite_checks = (
            []
            if mixed
            else [
                tf.debugging.assert_all_finite(
                    value.values if isinstance(value, tf.IndexedSlices) else value,
                    "bounds proposal gradient contains NaN/Inf",
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
                tf.debugging.assert_all_finite(
                    value, "bounds proposal weight contains NaN/Inf"
                )
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
        return compute_bounds_training_objective(
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


__all__ = ["run_hybrid_validation", "training_steps"]
