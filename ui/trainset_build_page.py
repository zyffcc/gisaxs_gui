"""Legacy import compatibility for the feature-owned Trainset page."""

from src.gimap.features.trainset.presentation.page import (
    ArrayCanvas,
    HistogramWidget,
    ParameterCoverageWidget,
    TrainsetBuildPage,
)

__all__ = [
    "ArrayCanvas",
    "HistogramWidget",
    "ParameterCoverageWidget",
    "TrainsetBuildPage",
]
