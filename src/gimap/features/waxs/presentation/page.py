"""Python-owned WAXS workspace and stable compatibility exports."""

from __future__ import annotations


from typing import Optional

import numpy as np


from PyQt5.QtCore import QThread, pyqtSignal

from PyQt5.QtWidgets import (
    QWidget,
)


from matplotlib.widgets import RectangleSelector


from .views import (
    WaxsPageView,
)

from .file_types import SCATTERING_FILTER, SUPPORTED_EXTENSIONS
from .image_viewer import ScatteringImageViewer
from .widget_factory import make_double_spin
from .workers import BatchWorker, ImageLoadResult, ImageLoadWorker

from .bindings.form_setup import FormSetupMixin
from .bindings.file_loading import FileLoadingMixin
from .bindings.selection_overlay import SelectionOverlayMixin
from .bindings.integration_export import IntegrationExportMixin
from .bindings.batch_processing import BatchProcessingMixin
from .bindings.view_state import ViewStateMixin

__all__ = [
    "BatchWorker",
    "ImageLoadResult",
    "ImageLoadWorker",
    "InSituProcessingWidget",
    "SCATTERING_FILTER",
    "SUPPORTED_EXTENSIONS",
    "ScatteringImageViewer",
    "make_double_spin",
]


class InSituProcessingWidget(
    FormSetupMixin,
    FileLoadingMixin,
    SelectionOverlayMixin,
    IntegrationExportMixin,
    BatchProcessingMixin,
    ViewStateMixin,
    QWidget,
    WaxsPageView,
):
    """Modern embedded replacement for the legacy in-situ data window."""

    statusChanged = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None, *, view_model=None):
        super().__init__(parent)
        if view_model is None:
            raise ValueError("InSituProcessingWidget requires WaxsViewModel")
        self.view_model = view_model
        self.current_file: Optional[str] = None
        self.current_image: Optional[np.ndarray] = None
        self.current_frame_count = 1
        self._loader_thread: Optional[QThread] = None
        self._loader_worker: Optional[ImageLoadWorker] = None
        self._batch_thread: Optional[QThread] = None
        self._batch_worker: Optional[BatchWorker] = None
        self._roi_selector: Optional[RectangleSelector] = None
        self._circle_pick_cid: Optional[int] = None
        self._center_pick_cid: Optional[int] = None
        self._circle_pick_points: list[tuple[float, float]] = []
        self._cut_extent: tuple[float, float, float, float] | None = None
        self._current_view_is_cut = False
        self._active_view = "2d"

        self.setupUi(self)
        self._bind_form()
        self._connect_signals()
        self._set_frame_controls_enabled(False)
        self._set_status("Ready")
