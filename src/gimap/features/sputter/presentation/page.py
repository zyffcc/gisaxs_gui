"""Sputter workspace: the grains of an in-situ sputter series, time window by time window.

**The main screen** (shown first). The input bar takes the scan folder, the metal, the substrate and the deposition
rate with its unit (``rate_flow.py``); **Analyze** runs the client analysis package (``client_flow.py``); while it runs,
the main screen shows its steps and what each has found so far (``live_panel.py``). The result is shown one time
window at a time (``client_panel.py``): the measured image beside the model image of a solution, the solutions to
choose from (the network's candidates, the refined one and the joint fit of the whole scan), the values of the
solution shown — radius, height, distance, gap, size distribution width, coverage — and their evolution over the scan. **Export…**
writes the result shown for a paper (``export_flow.py``).

**After Deposition** (the switch of the bar): Add Samples… fills the sample list beside the main screen
(``static_list.py``) and Analyze works through its ticked samples, each into the series of its folder name
(``static_queue.py``, one sample's run: ``static_flow.py``).

**Details** (the second view): the geometry, the options of Analyze, Static Frames (the folder the series go to, the
mode and the geometry of samples measured after their deposition; the parallel processes of both analyses:
``workers_flow.py``), More options and Storage (the import caches: ``storage_panel.py``); at the right the result
shown, from the files of its folder: Windows, Report, Log (``result_tabs.py``) and Steps (``live_steps_panel.py``).

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
from .export_flow import EXPORT_KEY, ExportMixin
from .geometry import INSPECT_KEY, SputterGeometryMixin
from .live_panel import LivePanelMixin
from .project_state import SputterProjectMixin
from .rate_flow import RateMixin
from .result_tabs import ResultTabsMixin
from .static_export import STATIC_EXPORT_KEY, StaticExportMixin
from .static_flow import STATIC_KEY, SputterStaticMixin
from .static_list import StaticListMixin
from .static_queue import StaticQueueMixin
from .storage_panel import STORAGE_KEY, StoragePanelMixin
from .views.sputter_page_view import SputterPageView
from .workers_flow import WorkersMixin


class SputterPage(SputterProjectMixin, SputterClientMixin, ClientPanelMixin, ExportMixin, ResultTabsMixin, LivePanelMixin,
                  SputterStaticMixin, StaticListMixin, StaticQueueMixin, StaticExportMixin, RateMixin, WorkersMixin,
                  StoragePanelMixin, SputterGeometryMixin, QWidget, SputterPageView):
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
        self._init_export()
        self._init_result_tabs()
        self._init_live()
        self._init_workers()
        self._init_storage()
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
        self.refresh_steps_language()
        self.refresh_storage()
        self._fill_workers()  # Automatic (N) in the new language

    def running(self) -> bool:
        """Analyze or Analyze Sample is at work (the window asks before it closes)."""
        return self.analyzing() or self.static_analyzing()

    def dispose(self) -> None:
        self._dispose_client()
        self._dispose_static()
        self.tasks.cancel(INSPECT_KEY)
        self.tasks.cancel(STORAGE_KEY)
        self.tasks.cancel(EXPORT_KEY)
        self.tasks.cancel(STATIC_EXPORT_KEY)
        try:
            theme_manager().changed.disconnect(self._theme_changed)
        except (TypeError, RuntimeError):  # already disconnected
            pass


__all__ = ["CLIENT_KEY", "STATIC_KEY", "SputterPage"]
