"""JSON-safe XRR JobRunner payload conversion."""

from __future__ import annotations

import base64
from dataclasses import asdict
from pathlib import Path
import zlib

import numpy as np

from ..application import (
    XrrExtractionRequest,
    XrrExtractionResult,
    XrrExtractionSettings,
    XrrPoint,
    XrrSeriesSpec,
)
from ..domain import SpecularGeometry


def request_to_payload(request: XrrExtractionRequest) -> dict:
    series = asdict(request.series)
    series["source_path"] = str(request.series.source_path)
    return {
        "series": series,
        "geometry": asdict(request.geometry),
        "extraction": asdict(request.extraction),
    }


def request_from_payload(payload: dict) -> XrrExtractionRequest:
    series_values = dict(payload["series"])
    series_values["source_path"] = Path(series_values["source_path"])
    return XrrExtractionRequest(
        series=XrrSeriesSpec(**series_values),
        geometry=SpecularGeometry(**payload["geometry"]),
        extraction=XrrExtractionSettings(**payload["extraction"]),
    )


def point_to_payload(point: XrrPoint) -> dict:
    return asdict(point)


def point_from_payload(payload: dict) -> XrrPoint:
    return XrrPoint(**payload)


def result_to_payload(result: XrrExtractionResult) -> dict:
    return {"points": [point_to_payload(point) for point in result.points]}


def result_from_payload(payload: dict) -> XrrExtractionResult:
    return XrrExtractionResult(
        tuple(point_from_payload(value) for value in payload.get("points", ()))
    )


def encode_preview(image: np.ndarray, max_side: int = 360) -> dict:
    values = np.asarray(image, dtype=np.float32)
    stride = max(1, int(np.ceil(max(values.shape) / max_side)))
    sampled = np.ascontiguousarray(values[::stride, ::stride], dtype=np.float32)
    compressed = zlib.compress(sampled.tobytes(), level=1)
    return {
        "shape": [int(value) for value in sampled.shape],
        "full_shape": [int(value) for value in values.shape],
        "data": base64.b64encode(compressed).decode("ascii"),
    }


def decode_preview(payload: dict) -> tuple[np.ndarray, tuple[int, int]]:
    raw = zlib.decompress(base64.b64decode(payload["data"]))
    shape = tuple(int(value) for value in payload["shape"])
    full_shape = tuple(int(value) for value in payload["full_shape"])
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy(), full_shape


__all__ = [
    "decode_preview",
    "encode_preview",
    "point_from_payload",
    "point_to_payload",
    "request_from_payload",
    "request_to_payload",
    "result_from_payload",
    "result_to_payload",
]
