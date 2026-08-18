"""Feature-owned PyQt presentation for geometry calibration."""

from __future__ import annotations


from typing import Optional


from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal

from PyQt5.QtWidgets import (
    QDialog,
)

from ..application import (
    CalibrationResult,
    DetectorImage,
)

from src.gimap.app.bootstrap import create_standalone_legacy_context


from src.gimap.app.presentation.assets import app_icon

from .views import GeometryCalibrationDialogView

from .workers import CalibrationWorker, ImageLoaderWorker

from .bindings.form_setup import FormSetupMixin
from .bindings.image_loading import ImageLoadingMixin
from .bindings.calibration_run import CalibrationRunMixin
from .bindings.result_preview import ResultPreviewMixin
from .bindings.manual_refinement import ManualRefinementMixin
from .bindings.persistence import PersistenceMixin

__all__ = ["CalibrationWorker", "GeometryCalibrationDialog", "ImageLoaderWorker"]


class GeometryCalibrationDialog(
    FormSetupMixin,
    ImageLoadingMixin,
    CalibrationRunMixin,
    ResultPreviewMixin,
    ManualRefinementMixin,
    PersistenceMixin,
    QDialog,
    GeometryCalibrationDialogView,
):
    calibrationApplied = pyqtSignal(object)

    @property
    def image(self) -> Optional[DetectorImage]:
        return self.view_model.image

    @image.setter
    def image(self, value: Optional[DetectorImage]) -> None:
        self.view_model.image = value

    @property
    def result(self) -> Optional[CalibrationResult]:
        return self.view_model.result

    @result.setter
    def result(self, value: Optional[CalibrationResult]) -> None:
        self.view_model.result = value

    def __init__(self, main_window=None, app_context=None, view_model=None):
        super().__init__(main_window)
        self.setupUi(self)
        self.main_window = main_window
        self.app_context = (
            app_context
            or getattr(main_window, "app_context", None)
            or getattr(view_model, "app_context", None)
            or create_standalone_legacy_context()
        )
        if view_model is None:
            from ..bootstrap import create_calibration_view_model

            view_model = create_calibration_view_model(self.app_context)
        self.view_model = view_model
        self.image: Optional[DetectorImage] = None
        self.result: Optional[CalibrationResult] = None
        self._load_thread: Optional[QThread] = None
        self._load_worker: Optional[ImageLoaderWorker] = None
        self._cal_thread: Optional[QThread] = None
        self._cal_worker: Optional[CalibrationWorker] = None
        self._dragging_center = False
        self._close_when_idle = False
        self._reset_preview_view = True
        self._preview_cache: dict[tuple[int, bool], tuple] = {}
        self._overlay_timer = QTimer(self)
        self._overlay_timer.setSingleShot(True)
        self._overlay_timer.setInterval(80)
        self._overlay_timer.timeout.connect(self.redraw_preview)
        self.detector_models = self.view_model.detector_models
        self.setWindowIcon(app_icon())
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint
        )
        self._bind_form()
        self._apply_dialog_style()
        self._connect_signals()
        self._set_running(False)
