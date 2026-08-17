"""仅在 Job worker process 内调用的 TensorFlow handlers。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..jobs import decode_numpy_tree, encode_numpy_tree
from .errors import TensorFlowModelError, TensorFlowNotInstalledError
from .legacy_keras import install_legacy_keras_load_shims


def _import_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        if exc.name == "tensorflow":
            raise TensorFlowNotInstalledError(
                "TensorFlow is not installed in this Python environment."
            ) from exc
        raise TensorFlowModelError(f"A TensorFlow dependency is missing: {exc}") from exc
    except Exception as exc:
        raise TensorFlowModelError(f"TensorFlow import failed: {exc}") from exc
    return tf


def _known_custom_objects() -> dict[str, Any]:
    custom_objects: dict[str, Any] = {
        "TYPE_MASK_LOGIT": -1.0e9,
        "FORCE_EXIST_LOGIT": 20.0,
        "FORCE_EMPTY_LOGIT": -20.0,
    }
    try:
        import sys

        legacy_package_root = (
            Path(__file__).resolve().parents[4] / "utils" / "ML_Fitting_1D_GISAXS"
        )
        if str(legacy_package_root) not in sys.path:
            sys.path.insert(0, str(legacy_package_root))
        from Training.model import SlotQueryBase

        custom_objects["SlotQueryBase"] = SlotQueryBase
    except Exception:
        pass
    return custom_objects


def _configure_precision(tf, precision_policy: str | None) -> None:
    if precision_policy is None:
        return
    if precision_policy == "auto":
        gpus = tf.config.list_physical_devices("GPU")
        precision_policy = "mixed_float16" if gpus else "float32"
    if precision_policy not in {"float32", "mixed_float16"}:
        raise TensorFlowModelError(f"Unsupported TensorFlow precision policy: {precision_policy}")
    tf.keras.mixed_precision.set_global_policy(precision_policy)


def _load_keras(tf, path: Path, allow_unsafe_lambda: bool):
    custom_objects = _known_custom_objects()
    try:
        try:
            return tf.keras.models.load_model(
                str(path),
                custom_objects=custom_objects,
                compile=False,
                safe_mode=not allow_unsafe_lambda,
            )
        except TypeError:
            return tf.keras.models.load_model(
                str(path), custom_objects=custom_objects, compile=False
            )
    except ValueError as exc:
        if "Lambda layer" in str(exc) and not allow_unsafe_lambda:
            raise TensorFlowModelError(
                "Model contains Lambda layers and safe deserialization blocked loading. "
                "Only enable unsafe Lambda loading for a trusted model."
            ) from exc
        first_error = exc
    except Exception as exc:
        first_error = exc

    install_legacy_keras_load_shims()
    try:
        import keras

        try:
            return keras.models.load_model(
                str(path),
                custom_objects=custom_objects,
                compile=False,
                safe_mode=not allow_unsafe_lambda,
            )
        except TypeError:
            return keras.models.load_model(
                str(path), custom_objects=custom_objects, compile=False
            )
    except Exception as exc:
        raise TensorFlowModelError(
            f"Keras model loading failed for {path}: {first_error}; fallback failed: {exc}"
        ) from exc


def _load_model(path: Path, allow_unsafe_lambda: bool, precision_policy: str | None):
    tf = _import_tensorflow()
    _configure_precision(tf, precision_policy)
    if path.is_file() and path.suffix.lower() == ".keras":
        return tf, _load_keras(tf, path, allow_unsafe_lambda), "keras"
    if path.is_dir() and (path / "saved_model.pb").is_file():
        try:
            return tf, _load_keras(tf, path, allow_unsafe_lambda), "keras_saved_model"
        except TensorFlowModelError as keras_error:
            try:
                return tf, tf.saved_model.load(str(path)), "saved_model_signature"
            except Exception as exc:
                raise TensorFlowModelError(
                    f"SavedModel loading failed for {path}: {keras_error}; fallback failed: {exc}"
                ) from exc
    raise TensorFlowModelError(f"Unsupported TensorFlow model artifact: {path}")


def _shape_tuple(value: Any) -> tuple[Any, ...] | None:
    shape = getattr(value, "shape", value)
    try:
        dimensions = shape.as_list() if hasattr(shape, "as_list") else list(shape)
    except Exception:
        return None
    return tuple(None if item is None else int(item) for item in dimensions)


def _tensor_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(str(name) for name in value)
    values = value if isinstance(value, (list, tuple)) else [value]
    names = []
    for item in values:
        name = str(getattr(item, "name", "")).split(":", maxsplit=1)[0]
        if name:
            names.append(name.split("/")[-1])
    return tuple(names)


def _runtime_info(tf, model, artifact_path: Path, loader: str) -> dict[str, Any]:
    input_names = tuple(getattr(model, "input_names", ()) or ())
    output_names = tuple(getattr(model, "output_names", ()) or ())
    input_shape = getattr(model, "input_shape", None)
    output_shape = getattr(model, "output_shape", None)
    if not input_names:
        input_names = _tensor_names(getattr(model, "inputs", ()))
    if not output_names:
        output_names = _tensor_names(getattr(model, "outputs", ()))

    signatures = getattr(model, "signatures", None)
    if signatures:
        signature = signatures.get("serving_default") or next(iter(signatures.values()))
        _args, kwargs = signature.structured_input_signature
        if not input_names:
            input_names = tuple(str(name) for name in kwargs)
        if input_shape is None and kwargs:
            input_shape = _shape_tuple(next(iter(kwargs.values())))
        outputs = getattr(signature, "structured_outputs", {})
        if not output_names:
            output_names = _tensor_names(outputs)
        if output_shape is None and outputs:
            output_shape = _shape_tuple(next(iter(outputs.values())))

    if isinstance(input_shape, dict):
        input_shape = next(iter(input_shape.values()), None)
    if isinstance(output_shape, dict):
        output_shape = next(iter(output_shape.values()), None)
    if (
        isinstance(input_shape, list)
        and input_shape
        and (isinstance(input_shape[0], (list, tuple)) or hasattr(input_shape[0], "as_list"))
    ):
        input_shape = input_shape[0]
    if (
        isinstance(output_shape, list)
        and output_shape
        and (isinstance(output_shape[0], (list, tuple)) or hasattr(output_shape[0], "as_list"))
    ):
        output_shape = output_shape[0]
    return {
        "artifact_path": str(artifact_path),
        "runtime_name": f"tensorflow:{loader}",
        "runtime_version": str(getattr(tf, "__version__", "")),
        "input_names": list(input_names),
        "output_names": list(output_names),
        "input_shape": list(_shape_tuple(input_shape) or ()),
        "output_shape": list(_shape_tuple(output_shape) or ()),
    }


def _to_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_numpy(item) for item in value]
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _predict(model, inputs: Any) -> Any:
    if hasattr(model, "predict"):
        return model.predict(inputs, verbose=0)
    signatures = getattr(model, "signatures", None)
    if signatures:
        function = signatures.get("serving_default") or next(iter(signatures.values()))
        _args, kwargs = function.structured_input_signature
        if isinstance(inputs, dict):
            return function(**inputs)
        names = list(kwargs)
        if names:
            return function(**{names[0]: inputs})
        return function(inputs)
    if callable(model):
        return model(inputs, training=False)
    raise TensorFlowModelError("Loaded TensorFlow artifact has no callable prediction interface.")


def inspect_tensorflow_model(payload, report, is_cancelled):
    if is_cancelled():
        raise RuntimeError("TensorFlow inspection cancelled before start.")
    path = Path(payload["artifact_path"])
    report(0, 1, "Loading TensorFlow model in isolated worker")
    tf, model, loader = _load_model(
        path,
        bool(payload.get("allow_unsafe_lambda", False)),
        payload.get("precision_policy"),
    )
    result = _runtime_info(tf, model, path, loader)
    report(1, 1, "TensorFlow model compatibility check complete")
    return result


def predict_tensorflow_model(payload, report, is_cancelled):
    if is_cancelled():
        raise RuntimeError("TensorFlow prediction cancelled before start.")
    path = Path(payload["artifact_path"])
    report(0, 2, "Loading TensorFlow model in isolated worker")
    tf, model, loader = _load_model(
        path,
        bool(payload.get("allow_unsafe_lambda", False)),
        payload.get("precision_policy"),
    )
    if is_cancelled():
        raise RuntimeError("TensorFlow prediction cancelled after model loading.")
    report(1, 2, "Running TensorFlow prediction")
    outputs = _predict(model, decode_numpy_tree(payload["inputs"]))
    report(2, 2, "TensorFlow prediction complete")
    return {
        "runtime": _runtime_info(tf, model, path, loader),
        "outputs": encode_numpy_tree(_to_numpy(outputs)),
    }
