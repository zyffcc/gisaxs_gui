from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


SUPPORTED_LAYER_TYPES = (
    "conv2d",
    "maxpool2d",
    "batch_normalization",
    "dropout",
    "global_average_pooling2d",
    "flatten",
    "dense",
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


def normalized_layers(model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    layers = model_config.get("layers")
    if isinstance(layers, list) and layers:
        return [dict(layer) for layer in layers if isinstance(layer, dict)]
    # Compatibility with schema v1 projects.
    output: List[Dict[str, Any]] = []
    for channels in model_config.get("channels", [32, 64, 128]):
        output.append(
            {
                "type": "conv2d",
                "units": int(channels),
                "kernel": int(model_config.get("kernel_size", 3)),
                "activation": "relu",
            }
        )
        output.append({"type": "maxpool2d", "pool": 2})
    output.append({"type": "global_average_pooling2d"})
    if float(model_config.get("dropout", 0.0)) > 0:
        output.append({"type": "dropout", "rate": float(model_config["dropout"])})
    return output


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


def static_contract(
    input_shape: Tuple[int, int, int],
    output_size: int,
    layers: Iterable[Dict[str, Any]],
) -> str:
    rows = [f"Input  {input_shape}"]
    spatial = True
    for index, spec in enumerate(layers, start=1):
        kind = str(spec.get("type", ""))
        rows.append(f"{index:02d}  {kind}")
        if kind in {"global_average_pooling2d", "flatten"}:
            spatial = False
        if kind == "dense" and spatial:
            rows.append("    ! Dense needs Flatten or GlobalAveragePooling2D first")
    if spatial:
        rows.append("Auto  global_average_pooling2d")
    rows.append(f"Output ({output_size},) regression parameters")
    return "\n".join(rows)
