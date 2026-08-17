"""Prediction domain public API。"""

from .models import ModelRuntimeInfo, PredictionRequest, PredictionResult
from .array_contracts import (
    coerce_array_to_shape,
    nearest_resize_nhwc,
    normalize_input_rank,
    normalize_parameter_prediction,
    normalize_prediction_output,
)
from .modules import ModelSpec, OutputSpec, PredictionModule, PreprocessSpec
from .sequences import build_complete_batches, extract_cbf_index, parse_index_range

__all__ = [
    "ModelRuntimeInfo",
    "ModelSpec",
    "OutputSpec",
    "PredictionModule",
    "PredictionRequest",
    "PredictionResult",
    "PreprocessSpec",
    "build_complete_batches",
    "coerce_array_to_shape",
    "extract_cbf_index",
    "nearest_resize_nhwc",
    "normalize_input_rank",
    "normalize_parameter_prediction",
    "normalize_prediction_output",
    "parse_index_range",
]
