"""NumPy arrays 与 JSON-safe worker payload 之间的显式转换。"""

from __future__ import annotations

from typing import Any

import numpy as np


ARRAY_MARKER = "gimap.ndarray.v1"


def encode_array(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
    return {
        "format": ARRAY_MARKER,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "data": value.reshape(-1).tolist(),
    }


def decode_array(payload: dict[str, Any]) -> np.ndarray:
    if payload.get("format") != ARRAY_MARKER:
        raise ValueError("Unsupported array worker payload.")
    shape = tuple(int(value) for value in payload.get("shape", []))
    array = np.asarray(payload.get("data", []), dtype=str(payload.get("dtype", "float32")))
    expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if array.size != expected:
        raise ValueError(
            f"Array payload contains {array.size} values; expected {expected} for shape {shape}."
        )
    return array.reshape(shape)


def encode_numpy_tree(value: Any) -> Any:
    """Recursively convert NumPy values into JSON-safe worker data."""
    if isinstance(value, np.ndarray):
        return encode_array(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): encode_numpy_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_numpy_tree(item) for item in value]
    return value


def decode_numpy_tree(value: Any) -> Any:
    """Restore arrays from a recursively encoded worker result."""
    if isinstance(value, dict):
        if value.get("format") == ARRAY_MARKER:
            return decode_array(value)
        return {key: decode_numpy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_numpy_tree(item) for item in value]
    return value
