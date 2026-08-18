"""Feature-owned binding between Classification Qt views and ViewModel."""

from __future__ import annotations


from typing import Optional


from PyQt5.QtCore import QThreadPool, pyqtSignal, QObject


from src.gimap.features.classification.application import (
    AlgorithmConfig,
    ClassificationPageState,
    ClassificationSample,
    DatasetSource,
    DatasetSummary,
    ExperimentResult,
    ModelEvaluationResult,
    PredictionResult,
    SavedModelPackage,
)


from src.gimap.features.classification.presentation.page import ClassificationPage

from src.gimap.features.classification.presentation.view_model import (
    ClassificationViewModel,
)

from .bindings.lifecycle_config import LifecycleConfigMixin
from .bindings.dataset_sources import DatasetSourcesMixin
from .bindings.algorithms_training import AlgorithmsTrainingMixin
from .bindings.dataset_table import DatasetTableMixin
from .bindings.result_rendering import ResultRenderingMixin
from .bindings.run_state import RunStateMixin
from .bindings.session_export import SessionExportMixin
from .bindings.selection_controls import SelectionControlsMixin
from .bindings.formatting import FormattingMixin
from .bindings.compatibility_slots import CompatibilitySlotsMixin

__all__ = ["ClassificationViewBinding"]


class ClassificationViewBinding(
    LifecycleConfigMixin,
    DatasetSourcesMixin,
    AlgorithmsTrainingMixin,
    DatasetTableMixin,
    ResultRenderingMixin,
    RunStateMixin,
    SessionExportMixin,
    SelectionControlsMixin,
    FormattingMixin,
    CompatibilitySlotsMixin,
    QObject,
):
    """Translate Qt events and ViewModel results into presentation state."""

    status_updated = pyqtSignal(str)

    progress_updated = pyqtSignal(int)

    parameters_changed = pyqtSignal(dict)

    classification_completed = pyqtSignal(dict)

    def __init__(
        self,
        ui,
        parent=None,
        *,
        classification_view_model: ClassificationViewModel | None = None,
        page: ClassificationPage | None = None,
    ):
        super().__init__(parent)
        self.ui = ui
        self.main_window = getattr(parent, "parent", None)
        if classification_view_model is None:
            raise ValueError("ClassificationViewBinding requires ClassificationViewModel")
        if page is None:
            raise ValueError("ClassificationViewBinding requires an injected ClassificationPage")
        self.classification_view_model = classification_view_model
        self.page = page

        self.thread_pool = QThreadPool.globalInstance()
        self.algorithm_configs: list[AlgorithmConfig] = (
            self.classification_view_model.default_algorithms()
        )

        self.sources: dict[str, DatasetSource] = {}
        self.samples: list[ClassificationSample] = []
        self.summary = DatasetSummary()
        self.experiment_result: Optional[ExperimentResult] = None
        self.feature_matrix = None
        self.active_result: Optional[ModelEvaluationResult] = None
        self.active_model_package: Optional[SavedModelPackage] = None
        self.prediction_results: list[PredictionResult] = []
        self.current_preview_sample_id: Optional[str] = None
        self.current_worker = None
        self._initialized = False
        self._table_updating = False
        self._results_outdated = False
        self.state = ClassificationPageState.EMPTY
