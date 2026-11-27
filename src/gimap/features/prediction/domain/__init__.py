"""Prediction domain public API。"""

from .models import ModelRuntimeInfo, PredictionRequest, PredictionResult
from .array_contracts import (
    coerce_array_to_shape,
    nearest_resize_nhwc,
    normalize_input_rank,
    normalize_parameter_prediction,
    normalize_prediction_output,
)
from .modules import (
    DistributionAxisSpec,
    ModelSpec,
    OutputSpec,
    PredictionModule,
    PreprocessSpec,
)
from .sequences import (
    FRAME_SUFFIXES,
    build_complete_batches,
    extract_cbf_index,
    extract_frame_index,
    parse_index_range,
)

__all__ = [
    "FRAME_SUFFIXES",
    "ModelRuntimeInfo",
    "DistributionAxisSpec",
    "ModelSpec",
    "OutputSpec",
    "PredictionModule",
    "PredictionRequest",
    "PredictionResult",
    "PreprocessSpec",
    "build_complete_batches",
    "coerce_array_to_shape",
    "extract_cbf_index",
    "extract_frame_index",
    "nearest_resize_nhwc",
    "normalize_input_rank",
    "normalize_parameter_prediction",
    "normalize_prediction_output",
    "parse_index_range",
]
