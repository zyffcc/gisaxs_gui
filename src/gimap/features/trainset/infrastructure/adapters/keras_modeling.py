"""Keras model construction adapter；TensorFlow API is injected lazily。"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ...domain import (
    SUPPORTED_LAYER_TYPES,
    normalized_layers,
    static_contract,
)


def resolve_keras_api(tf: Any) -> Any:
    """Return a working Keras API, including TensorFlow 2.15 legacy fallback.

    Some scientific Conda environments contain TensorFlow 2.15 alongside an
    incompatible standalone Keras package. TensorFlow's bundled legacy API is
    still usable there and keeps local smoke tests self-contained.
    """
    try:
        keras_api = tf.keras
        _ = keras_api.Input
        return keras_api
    except Exception:
        from tensorflow.python.distribute import input_lib  # type: ignore
        from tensorflow.python import keras as keras_api  # type: ignore

        if not hasattr(input_lib, "DistributedDatasetInterface"):
            input_lib.DistributedDatasetInterface = type("DistributedDatasetInterface", (), {})
        # TensorFlow's legacy HDF5 saver imports this attribute directly.
        # Some Conda builds omit it from the private bundled namespace.
        if not hasattr(keras_api, "__version__"):
            keras_api.__version__ = str(getattr(tf, "__version__", "legacy"))
        return keras_api


def build_optimizer(keras_api: Any, name: str, learning_rate: float) -> Any:
    normalized = str(name).lower()
    if hasattr(keras_api.optimizers, "Adam"):
        if normalized == "sgd":
            return keras_api.optimizers.SGD(learning_rate)
        if normalized == "adamw" and hasattr(keras_api.optimizers, "AdamW"):
            return keras_api.optimizers.AdamW(learning_rate)
        return keras_api.optimizers.Adam(learning_rate)
    # TensorFlow 2.15's private legacy namespace exposes optimizer modules but
    # not the convenience class attributes.
    from tensorflow.python.keras.optimizer_v2 import adam, gradient_descent  # type: ignore

    if normalized == "sgd":
        return gradient_descent.SGD(learning_rate)
    return adam.Adam(learning_rate)


def build_keras_model(
    tf: Any,
    input_shape: Tuple[int, int, int],
    output_size: int,
    model_config: Dict[str, Any],
    smoke: bool = False,
) -> Any:
    keras_api = resolve_keras_api(tf)
    inputs = keras_api.Input(shape=input_shape)
    x = inputs
    spatial = True
    for index, spec in enumerate(normalized_layers(model_config)):
        kind = str(spec.get("type", "")).lower()
        if kind not in SUPPORTED_LAYER_TYPES:
            raise ValueError(f"Layer {index + 1}: unsupported type {kind!r}.")
        if kind == "conv2d":
            if not spatial:
                raise ValueError(f"Layer {index + 1}: Conv2D requires a spatial tensor.")
            units = max(1, int(spec.get("units", 32)))
            if smoke:
                units = min(units, 16)
            x = keras_api.layers.Conv2D(
                units,
                max(1, int(spec.get("kernel", 3))),
                padding="same",
                activation=str(spec.get("activation", "relu")) or None,
            )(x)
        elif kind == "maxpool2d":
            if not spatial:
                raise ValueError(f"Layer {index + 1}: MaxPool2D requires a spatial tensor.")
            x = keras_api.layers.MaxPool2D(pool_size=max(1, int(spec.get("pool", 2))))(x)
        elif kind == "batch_normalization":
            x = keras_api.layers.BatchNormalization()(x)
        elif kind == "dropout":
            x = keras_api.layers.Dropout(min(0.95, max(0.0, float(spec.get("rate", 0.3)))))(x)
        elif kind == "global_average_pooling2d":
            if not spatial:
                raise ValueError(f"Layer {index + 1}: global pooling requires a spatial tensor.")
            x = keras_api.layers.GlobalAveragePooling2D()(x)
            spatial = False
        elif kind == "flatten":
            x = keras_api.layers.Flatten()(x)
            spatial = False
        elif kind == "dense":
            if spatial:
                raise ValueError(f"Layer {index + 1}: add Flatten or GlobalAveragePooling2D before Dense.")
            units = max(1, int(spec.get("units", 128)))
            if smoke:
                units = min(units, 32)
            x = keras_api.layers.Dense(units, activation=str(spec.get("activation", "relu")) or None)(x)
    if spatial:
        x = keras_api.layers.GlobalAveragePooling2D(name="automatic_global_average_pooling")(x)
    outputs = keras_api.layers.Dense(int(output_size), name="parameter_output")(x)
    return keras_api.Model(inputs, outputs)
