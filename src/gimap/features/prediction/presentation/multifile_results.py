"""Stable public API for multi-file Prediction presentation."""

from .result_types import (
    PredictStatus,
    PredictResult,
)

from .trend_windows import (
    DistributionHeatmapWindow,
    ParameterTrendWindow,
)

from .result_models import (
    PredictResultsTableModel,
    PredictResultsFilterModel,
)

from .export_dialog import (
    ExportDialog,
)

from .results_widget import (
    MultiFilePredictResultsWidget,
)

from .batch_manager import (
    MultiFilePredictManager,
)

__all__ = [
    "PredictStatus",
    "PredictResult",
    "DistributionHeatmapWindow",
    "ParameterTrendWindow",
    "PredictResultsTableModel",
    "PredictResultsFilterModel",
    "ExportDialog",
    "MultiFilePredictResultsWidget",
    "MultiFilePredictManager",
]
