"""Feature-owned binding between the Prediction workspace and ViewModel."""

from __future__ import annotations


from typing import Dict, List, Optional, Tuple

import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal, QTimer

from PyQt5.QtGui import QPixmap

from PyQt5.QtWidgets import (
    QGraphicsScene,
    QLabel,
    QShortcut,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QDialog,
    QTextBrowser,
)


from src.gimap.features.prediction.presentation.image_worker import PredictionImageLoader

from src.gimap.features.prediction.presentation.view_model import PredictionViewModel

from src.gimap.features.prediction.presentation.multifile_results import (
    MultiFilePredictResultsWidget,
    MultiFilePredictManager,
)


from .bindings.setup_status import SetupStatusMixin
from .bindings.multifile_setup import MultifileSetupMixin
from .bindings.input_parameters import InputParametersMixin
from .bindings.image_loading import ImageLoadingMixin
from .bindings.rendering_setup import RenderingSetupMixin
from .bindings.render_controls import RenderControlsMixin
from .bindings.prediction_results import PredictionResultsMixin
from .bindings.display_controls import DisplayControlsMixin
from .bindings.module_catalog import ModuleCatalogMixin
from .bindings.prediction_execution import PredictionExecutionMixin
from .bindings.multifile_results import MultifileResultsMixin
from .bindings.widget_access import WidgetAccessMixin

__all__ = ["PredictionViewBinding"]


class PredictionViewBinding(
    SetupStatusMixin,
    MultifileSetupMixin,
    InputParametersMixin,
    ImageLoadingMixin,
    RenderingSetupMixin,
    RenderControlsMixin,
    PredictionResultsMixin,
    DisplayControlsMixin,
    ModuleCatalogMixin,
    PredictionExecutionMixin,
    MultifileResultsMixin,
    WidgetAccessMixin,
    QObject,
):
    """Translate Prediction Qt events and render ViewModel results."""

    status_updated = pyqtSignal(str)

    progress_updated = pyqtSignal(int)

    parameters_changed = pyqtSignal(dict)

    prediction_completed = pyqtSignal(dict)

    model_load_finished = pyqtSignal(object, str, str)

    _DEFAULT_COLORMAPS = [
        "viridis",
        "cividis",
        "plasma",
        "magma",
        "inferno",
        "turbo",
        "jet",
        "coolwarm",
        "gray",
    ]

    _mpl_cm = None

    def __init__(
        self,
        ui,
        parent=None,
        *,
        prediction_view_model: PredictionViewModel | None = None,
    ) -> None:
        super().__init__(parent)
        self.ui = ui
        self.main_window = parent.parent if hasattr(parent, "parent") else None
        if prediction_view_model is None:
            raise ValueError("PredictionViewBinding requires PredictionViewModel")
        self.prediction_view_model = prediction_view_model

        self.current_parameters: Dict[str, Optional[str]] = {}
        self.prediction_results: Dict[str, object] = {}

        # 初始化运行时状态
        self._initialized = False
        self._ui_updating = False
        self._synchronizing = False

        self._graphics_scene: Optional[QGraphicsScene] = None
        self._current_pixmap: Optional[QPixmap] = None
        self._current_image: Optional[np.ndarray] = None
        self._current_image_path: Optional[str] = None
        self._view_zoom_steps = 0

        # Predict-2D view state
        self._predict_scene: Optional[QGraphicsScene] = None
        self._predict_pixmap: Optional[QPixmap] = None
        self._predict_zoom_steps = 0

        self._index_to_file: Dict[int, str] = {}
        self._available_indices: List[int] = []
        self._sequence_indices: List[int] = []
        self._current_file_index: Optional[int] = None
        self._folder_entries: List[Tuple[str, int]] = []

        self._load_request_seq = 0
        self._latest_display_request = 0
        self._active_loaders: Dict[int, PredictionImageLoader] = {}
        self._pending_contexts: Dict[int, Dict[str, object]] = {}

        # Module system state
        self._modules_by_name: Dict[str, Dict[str, object]] = {}
        self._modules_by_id: Dict[str, Dict[str, object]] = {}
        self._current_module: Optional[Dict[str, object]] = None
        self._module_edit_watch_timer: Optional[QTimer] = None
        self._module_edit_watch_path: Optional[str] = None
        self._module_edit_watch_mtime: Optional[float] = None
        self._module_edit_watch_ticks: int = 0
        self._current_mask: Optional[np.ndarray] = None
        self._current_model: Optional[object] = None
        self._model_loading: bool = False
        self._model_cancel_requested: bool = False
        self._model_loader_thread = None
        self._model_status_label: Optional[QLabel] = None
        self._status_text_window: Optional[QDialog] = None
        self._status_text_window_browser: Optional[QTextBrowser] = None
        self._cancel_shortcut: Optional[QShortcut] = None
        self._predict_tabs: Optional[QTabWidget] = None
        self._predict_panel: Optional[QWidget] = None
        self._predict_panel_layout: Optional[QVBoxLayout] = None
        self._predict_import_button: Optional[QPushButton] = None
        self._predict_tab_specs: List[Dict[str, object]] = []
        self._predict_current_kind: Optional[str] = None
        self._predict_current_image: Optional[np.ndarray] = None
        self._predict_current_curve: Optional[np.ndarray] = None
        self._predict_curve_controls: Dict[str, object] = {}
        self._current_step_index: int = 0

        # 多文件预测相关
        self._multifile_results_widget: Optional[MultiFilePredictResultsWidget] = None
        self._multifile_manager: Optional[MultiFilePredictManager] = None
        self._multifile_prediction_active: bool = False
        self._prediction_active: bool = False
        self._multifile_batch_map: Dict[str, List[str]] = {}

        # 读取全局参数
        self._set_default_parameters()
        self._load_saved_parameters()
        self.model_load_finished.connect(self._on_model_load_finished)
