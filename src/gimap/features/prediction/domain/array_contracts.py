"""模型输入 shape 和模型输出的稳定 NumPy 规则。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


ResizeNhwc = Callable[[np.ndarray, int, int], np.ndarray]


def normalize_input_rank(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    while array.ndim > 4 and 1 in array.shape:
        array = np.squeeze(array, axis=array.shape.index(1))
    if array.ndim == 2:
        return array[None, ..., None].astype(np.float32, copy=False)
    if array.ndim == 3:
        if array.shape[0] == 1:
            return array[..., None].astype(np.float32, copy=False)
        if array.shape[-1] in (1, 2, 3, 4):
            return array[None, ...].astype(np.float32, copy=False)
        return array[..., None].astype(np.float32, copy=False)
    return array.astype(np.float32, copy=False)


def nearest_resize_nhwc(array: np.ndarray, height: int, width: int) -> np.ndarray:
    if array.ndim != 4 or array.shape[1:3] == (height, width):
        return array
    y_indices = np.linspace(0, array.shape[1] - 1, height).astype(np.int32)
    x_indices = np.linspace(0, array.shape[2] - 1, width).astype(np.int32)
    return array[:, y_indices][:, :, x_indices]


def coerce_array_to_shape(
    value: np.ndarray,
    shape: tuple[object, ...],
    resize_nhwc: ResizeNhwc = nearest_resize_nhwc,
) -> np.ndarray:
    array = normalize_input_rank(value)
    if len(shape) == 4:
        _, height, width, channels = shape
        target_height = int(height) if isinstance(height, (int, np.integer)) else array.shape[1]
        target_width = int(width) if isinstance(width, (int, np.integer)) else array.shape[2]
        target_channels = (
            int(channels)
            if isinstance(channels, (int, np.integer))
            else array.shape[-1]
        )
        array = resize_nhwc(array, target_height, target_width)
        if array.shape[-1] != target_channels:
            if target_channels == 1:
                array = array[..., :1]
            elif array.shape[-1] == 1:
                array = np.repeat(array, target_channels, axis=-1)
            else:
                array = array[..., :target_channels]
        return array.astype(np.float32, copy=False)

    if len(shape) == 3:
        _, height, width = shape
        target_height = int(height) if isinstance(height, (int, np.integer)) else array.shape[1]
        target_width = int(width) if isinstance(width, (int, np.integer)) else array.shape[2]
        if array.ndim == 4 and array.shape[-1] == 1:
            array = array[..., 0]
        elif array.ndim == 2:
            array = array[None, ...]
        if array.ndim == 3:
            resized = resize_nhwc(array[..., None], target_height, target_width)
            return resized[..., 0].astype(np.float32, copy=False)
    return array.astype(np.float32, copy=False)


def normalize_parameter_prediction(
    prediction: object,
    module_values: Mapping[str, Any],
) -> dict[str, object] | None:
    output_type = str(module_values.get("output_type") or "").lower()
    is_parameters = output_type in {"sf_4_parameters", "sf_parameters", "parameters"}
    if not is_parameters and isinstance(prediction, dict):
        is_parameters = "branch_thickness" in prediction and "branch_size" in prediction
    if not is_parameters:
        return None

    normalized = None
    if isinstance(prediction, dict):
        if "branch_thickness" in prediction and "branch_size" in prediction:
            thickness = np.asarray(prediction["branch_thickness"], dtype=np.float32)
            size = np.asarray(prediction["branch_size"], dtype=np.float32)
            normalized = np.concatenate([thickness, size], axis=-1)
        elif "parameters" in prediction:
            normalized = np.asarray(prediction["parameters"], dtype=np.float32)
        elif prediction:
            values = [np.asarray(value, dtype=np.float32) for value in prediction.values()]
            if values:
                normalized = np.concatenate(values, axis=-1)
    else:
        array = np.asarray(prediction, dtype=np.float32)
        if array.size:
            normalized = array
    if normalized is None:
        return None

    array = np.asarray(normalized, dtype=np.float32)
    if array.ndim > 1:
        array = array.reshape((-1, array.shape[-1]))[0]
    array = array.reshape(-1)
    raw_names = module_values.get("parameter_names")
    names = (
        [str(name) for name in raw_names]
        if isinstance(raw_names, list) and raw_names
        else ["t_Cu", "t_polymer", "D", "sigma"]
    )
    target_min = np.asarray(
        module_values.get("target_min") or [0.0, 10.0, 4.0, 0.2], dtype=np.float32
    )
    target_max = np.asarray(
        module_values.get("target_max") or [25.0, 50.0, 20.0, 4.0], dtype=np.float32
    )
    count = min(array.size, len(names), target_min.size, target_max.size)
    if count <= 0:
        return None
    normalized_values = array[:count]
    values = normalized_values * (target_max[:count] - target_min[:count]) + target_min[:count]
    return {
        "parameters": values.astype(np.float32),
        "parameters_normalized": normalized_values.astype(np.float32),
        "parameter_names": names[:count],
    }


def normalize_prediction_output(
    prediction: object,
    module_values: Mapping[str, Any],
) -> dict[str, object] | None:
    parameters = normalize_parameter_prediction(prediction, module_values)
    if parameters is not None:
        return parameters

    image_output = None
    scalar_output = None
    if isinstance(prediction, (list, tuple)) and prediction:
        candidate = np.asarray(prediction[0])
        image_output = candidate if candidate.squeeze().ndim >= 2 else None
        scalar_output = None if image_output is not None else candidate.squeeze()
    elif isinstance(prediction, dict):
        value = prediction.get("hr")
        if value is None:
            value = prediction.get("output")
        if value is None and prediction:
            value = next(iter(prediction.values()))
        if value is not None:
            candidate = np.asarray(value)
            image_output = candidate if candidate.squeeze().ndim >= 2 else None
            scalar_output = None if image_output is not None else candidate.squeeze()
    else:
        candidate = np.asarray(prediction)
        image_output = candidate if candidate.squeeze().ndim >= 2 else None
        scalar_output = None if image_output is not None else candidate.squeeze()

    if image_output is None and scalar_output is not None:
        return {"scalars": np.asarray(scalar_output).reshape(-1)}
    if image_output is None:
        return None
    image = image_output.squeeze()
    if image.ndim == 3:
        image = image[..., 0]
    if image.ndim != 2:
        return None
    return {
        "hr": image,
        "h": np.sum(image, axis=0),
        "r": np.sum(image, axis=1),
    }
