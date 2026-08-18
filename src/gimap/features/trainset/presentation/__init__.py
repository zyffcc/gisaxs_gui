"""Trainset presentation public API."""

from .page import (
    ArrayCanvas,
    HistogramWidget,
    ParameterCoverageWidget,
    TrainsetBuildPage,
)
from .state import TrainsetState
from .view_binding import TrainsetViewBinding
from .view_model import TrainsetViewModel

TrainsetController = TrainsetViewBinding

__all__ = [
    "ArrayCanvas",
    "HistogramWidget",
    "ParameterCoverageWidget",
    "TrainsetBuildPage",
    "TrainsetController",
    "TrainsetState",
    "TrainsetViewBinding",
    "TrainsetViewModel",
]
