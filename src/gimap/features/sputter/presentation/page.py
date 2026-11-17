"""Sputter workspace: the grains of an in-situ sputter series, time window by time window.

**The main screen** (shown first). The input bar takes the scan folder, the metal, the substrate and the deposition
rate with its unit (``rate_flow.py``); **Analyze** runs the client analysis package (``client_flow.py``); while it runs,
the main screen shows its steps and what each has found so far (``live_panel.py``). The result is shown one time
window at a time (``client_panel.py``): the measured image beside the model image of a solution, the solutions to
choose from (the network's candidates, the refined one and the joint fit of the whole scan), the values of the
solution shown — radius, height, distance, gap, size distribution width, coverage — and their evolution over the scan.

**Details** (the second view): the geometry, the options of Analyze, Static Frames (samples measured after their
deposition, analysed one at a time into a series the main screen shows: ``static_flow.py``; the parallel processes of
both: ``workers_flow.py``) and More options; at the
right the result shown, from the files of its folder: Windows, Report and Log (``result_tabs.py``).

The layout is in ``views/sputter_page_view.py`` and ``views/sputter_main_view.py``; the folder and the form in
``geometry.py``.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.task_runner import TaskRunner
from src.gimap.app.presentation.theme.engine import theme_manager

from ..application import SputterService
from .client_flow import CLIENT_KEY, SputterClientMixin
from .client_panel import ClientPanelMixin
from .geometry import INSPECT_KEY, SputterGeometryMixin
from .live_panel import LivePanelMixin
from .project_state import SputterProjectMixin
from .rate_flow import RateMixin
from .result_tabs import ResultTabsMixin
from .static_flow import STATIC_KEY, SputterStaticMixin
from .views.sputter_page_view import SputterPageView
from .workers_flow import WorkersMixin


class SputterPage(SputterProjectMixin, SputterClientMixin, ClientPanelMixin, ResultTabsMixin, LivePanelMixin,
                  SputterStaticMixin, RateMixin, WorkersMixin, SputterGeometryMixin, QWidget, SputterPageView):
    clientEvent = pyqtSignal(object)
    """A line or a step of Analyze, from its background task (delivered in the GUI thread)."""
    staticEvent = pyqtSignal(object)
    """The same for Analyze Sample (static frames)."""

    def __init__(self, service: SputterService, *, task_runner: Optional[TaskRunner] = None, preferences=None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setup_ui(self)
        self.service = service
        self.tasks = task_runner or TaskRunner(self)
        self.preferences = preferences
        self.scan_info = None
        self.sources: dict = {}
        self._edited: set = set()
        self._requested_scan = ""
        self._summary = None
        self._inspecting = False          # the header of the chosen folder is being read
        self.results_picker.path_edit.setPlaceholderText(str(service.results_root()))
        self.mask_picker.browseRequested.connect(self._choose_mask)
        self.results_picker.browseRequested.connect(self._choose_results_root)
        self._init_client()
        self._init_static()
        self._init_client_panel()
        self._init_result_tabs()
        self._init_live()
        self._init_workers()
        self._connect_geometry()
        self._connect_client()
        self._connect_static()
        self._empty_project_state = self.project_state()
        self._pending_project_state = None
        self._restore()
        self._restore_client_last()
        self._restore_static_last()
        theme_manager().changed.connect(self._theme_changed)
        last = self._stored().get("last_client_result")
        if last:
            self.open_client_result(last, quiet=True)

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
    def _notify(self, message: str, *, error: bool = False) -> None:
        try:
            show_toast(self, message, level="error" if error else "info")
        except Exception:  # noqa: BLE001 - a missing toast host must not hide the message
            self.analyze_status.message_label.setText(message)

    def _theme_changed(self, *_args) -> None:
        """The plots in the colours of the new theme (data colours stay)."""
        self.redraw_client_plots()
        self.redraw_live_plots()

    def refresh_language(self) -> None:
        """Run-time texts in the new interface language (the walker has translated the fixed ones)."""
        self._show_sources()
        self._show_scan_summary()
        self.refresh_client_language()
        self.refresh_static_language()
        self.refresh_result_tabs_language()
        self._fill_workers()  # Automatic (N) in the new language

    def running(self) -> bool:
        """Analyze or Analyze Sample is at work (the window asks before it closes)."""
        return self.analyzing() or self.static_analyzing()

    def dispose(self) -> None:
        self._dispose_client()
        self._dispose_static()
        self.tasks.cancel(INSPECT_KEY)
        try:
            theme_manager().changed.disconnect(self._theme_changed)
        except (TypeError, RuntimeError):  # already disconnected
            pass


__all__ = ["CLIENT_KEY", "STATIC_KEY", "SputterPage"]
