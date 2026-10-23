"""Sputter workspace: particle size R, spacing D and the metal thickness over an in-situ sputter series, and how
far to trust them.

The user chooses the scan folder and presses Predict; the model bundled with GIMaP runs in GIMaP's own
Python (PyTorch is installed there on the first Predict). Left: Data, Sample and geometry, Predict with its
job status, the model's environment, the result line and More options. Right: Size Trend (faint per-window points
and the trend lines; the metal thickness), Literature (the series against published growth over the nominal
thickness), Structure Sketch, Windows and Reliability.
The layout is in ``views/sputter_page_view.py``; the folder and the form in ``geometry.py``, Predict in
``predict_flow.py``, showing a result in ``results_panel.py``.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.task_runner import TaskRunner
from src.gimap.app.presentation.theme.engine import theme_manager

from ..application import DOWNLOAD_SIZE, SputterService
from .geometry import INSPECT_KEY, SputterGeometryMixin
from .literature_tab import LiteratureTab
from .predict_flow import PREDICT_KEY, SputterPredictMixin
from .results_panel import SputterResultsMixin
from .sketch_view import SketchView
from .texts import message_text, text
from .trend_plot import TrendPlot
from .views.sputter_page_view import SputterPageView

ENVIRONMENT_READY = "Model: bundled with GIMaP · Python: {python} · PyTorch {version}: ready"
ENVIRONMENT_MISSING = ("Model: bundled with GIMaP · Python: {python} · PyTorch is installed on the first Predict "
                       "({size}, needs the internet)")
ENVIRONMENT_UNKNOWN = "Model: bundled with GIMaP · Python: {python} · PyTorch is checked when you press Predict"


class SputterPage(SputterGeometryMixin, SputterPredictMixin, SputterResultsMixin, QWidget, SputterPageView):
    eventReceived = pyqtSignal(object)

    def __init__(self, service: SputterService, *, task_runner: Optional[TaskRunner] = None, preferences=None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setup_ui(self)
        self.service = service
        self.tasks = task_runner or TaskRunner(self)
        self.preferences = preferences
        self.result = None
        self.scan_info = None
        self.sources: dict = {}
        self._edited: set = set()
        self._requested_scan = ""
        self._summary = None
        self._busy: Optional[str] = None
        self._stop = False
        self._tracker = None
        self._step = ""
        self._installing_writing = False
        self._inspecting = False          # the header of the chosen folder is being read
        self._predict_queued = False      # Predict pressed while the header was read or the files rewritten
        self._result_owner = ""           # folder_key of the scan the result shown belongs to
        self._drawn: Optional[tuple] = None   # (rate, t0, end) the result's physical files were drawn with
        self._rebuild_values: Optional[tuple] = None
        self._thickness_typed = False     # the rate, t0 or end was typed while the model ran
        self._status_parts = ("idle", [("Ready", {})], 0.0, True)
        self.trend = TrendPlot(self.trend_layout)
        self.sketch = SketchView()
        self.literature = LiteratureTab()
        self.tabs.addTab(self.literature, "Literature")
        self.tabs.addTab(self.sketch, "Structure Sketch")
        self.tabs.addTab(self.table, "Windows")
        self.tabs.addTab(self.report, "Reliability")
        self.results_picker.path_edit.setPlaceholderText(str(service.results_root()))
        self.mask_picker.browseRequested.connect(self._choose_mask)
        self.results_picker.browseRequested.connect(self._choose_results_root)
        self._connect_geometry()
        self._connect_predict()
        self._connect_results()
        self._restore()
        self._metal_changed()
        self._show_status()
        self._show_environment()
        theme_manager().changed.connect(self._theme_changed)
        last = self._stored().get("last_result")
        if last:
            self.open_results(last, quiet=True)

    # -- small pickers -------------------------------------------------------------------------------------
    def _choose_mask(self) -> None:
        from PyQt5.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(self, tr("Choose a Detector Mask"), self.mask_picker.path(),
                                              "NumPy mask (*.npy);;All files (*)")
        if path:
            self.mask_picker.set_path(path)

    def _choose_results_root(self) -> None:
        from PyQt5.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(self, tr("Save Results In"), self.results_picker.path()
                                                  or str(self.service.results_root()))
        if folder:
            self.results_picker.set_path(folder)

    # -- texts -----------------------------------------------------------------------------------------------
    def _show_environment(self) -> None:
        missing = self.service.toolkit_problem()
        status = self.service.environment()
        if status.ready:
            words = text(ENVIRONMENT_READY, python=status.python, version=status.torch_version)
        elif status.problem == "unknown":
            words = text(ENVIRONMENT_UNKNOWN, python=status.python)
        else:
            words = text(ENVIRONMENT_MISSING, python=status.python, size=tr(DOWNLOAD_SIZE))
        self.environment_label.setText(message_text(missing) if missing else words)
        self.environment_label.setToolTip(status.python)

    def _notify(self, message: str, *, error: bool = False) -> None:
        try:
            show_toast(self, message, level="error" if error else "info")
        except Exception:  # noqa: BLE001 - a missing toast host must not hide the message
            self.job_status.message_label.setText(message)

    def _theme_changed(self, *_args) -> None:
        """The plots in the colours of the new theme (data colours stay)."""
        self._draw_trend()
        self.literature.redraw()
        if self.result is not None:
            self.sketch.redraw()

    def refresh_language(self) -> None:
        """Run-time texts in the new interface language (the walker has translated the fixed ones)."""
        self._show_status()
        self._show_sources()
        self._show_scan_summary()
        self._show_environment()
        self.refresh_results_language()
        self.sketch.refresh_language()
        if self.result is None:
            self.literature.refresh_language()

    def dispose(self) -> None:
        SputterPredictMixin.dispose(self)
        self.tasks.cancel(INSPECT_KEY)
        try:
            theme_manager().changed.disconnect(self._theme_changed)
        except (TypeError, RuntimeError):  # already disconnected
            pass


__all__ = ["PREDICT_KEY", "SputterPage"]
