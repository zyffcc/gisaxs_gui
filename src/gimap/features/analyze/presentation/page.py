"""Analyze workspace: open or drop frames and get reduced curves without clicks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtCore import QSignalBlocker, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QFileDialog, QShortcut, QWidget

from src.gimap.app.presentation.task_runner import TaskRunner

from ..application import GISAXS, GIWAXS, FrameAnalysis
from .bindings.batch_export import BatchExportMixin
from .bindings.batch_watch import BatchWatchMixin
from .bindings.beam_center import BeamCenterMixin
from .bindings.display import DisplayMixin
from .bindings.options import OptionsMixin
from .bindings.marks import MarksMixin
from .bindings.masks import MaskToolsMixin
from .bindings.profile_actions import ProfileActionsMixin
from .bindings.project import ProjectMixin
from .bindings.region_pick import RegionPickMixin
from .bindings.regions import RegionsMixin
from .bindings.series import SeriesMixin
from .bindings.batch_run import BatchRunMixin
from .bindings.session_memory import SessionMemoryMixin
from .bindings.undo import UndoMixin
from .bindings.workspace import WorkspaceMixin
from .view_model import AnalyzeViewModel
from .views.analyze_page_view import FILE_FILTER, INCIDENCE_FROM_PROFILE, AnalyzePageView
WATCH_INTERVAL_MS = 2000
STATUS_LEVELS = ("info", "ok", "warning", "error")


class AnalyzePage(
    ProjectMixin,
    WorkspaceMixin,
    RegionsMixin,
    RegionPickMixin,
    MaskToolsMixin,
    SeriesMixin,
    ProfileActionsMixin,
    BatchWatchMixin,
    BatchExportMixin,
    BatchRunMixin,
    SessionMemoryMixin,
    UndoMixin,
    MarksMixin,
    BeamCenterMixin,
    OptionsMixin,
    DisplayMixin,
    QWidget,
    AnalyzePageView,
):
    """Open → geometry → reduction → curves, with every step visible and adjustable."""

    analysisShown = pyqtSignal(object)
    analysisFailed = pyqtSignal(str)

    def __init__(
        self,
        view_model: AnalyzeViewModel,
        *,
        task_runner: Optional[TaskRunner] = None,
        calibrate: Optional[Callable[[Optional[Path]], None]] = None,
        send_to_fitting: Optional[Callable[[Path, str], None]] = None,
        send_series_to_fitting: Optional[Callable[[Path], None]] = None,
        assistant: Optional[Callable[[], None]] = None,
        find_geometry: Optional[Callable[[], None]] = None,
        run_pipeline: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.view_model = view_model
        self.tasks = task_runner or TaskRunner(self)
        self._calibrate = calibrate
        self._send_to_fitting = send_to_fitting
        self._send_series_to_fitting = send_series_to_fitting
        self._assistant = assistant
        self._find_geometry = find_geometry
        self._run_pipeline = run_pipeline
        self._automation = None
        self._batch_then = None
        self._started_at = 0.0
        self._batch: list = []
        self._batch_total = 0
        self._batch_failures: list[str] = []
        self._batch_destination: Optional[Path] = None
        self._batch_series = None
        self._batch_series_state: dict = {}
        self._batch_dialog = None
        self._batch_pending = False
        self._lower_profile = "azimuthal"
        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(WATCH_INTERVAL_MS)
        self._watch_timer.timeout.connect(self.poll_watch)
        self.setup_ui(self)
        self.setAcceptDrops(True)
        self.calibrate_button.setEnabled(calibrate is not None)
        self.banner_calibrate_button.setEnabled(calibrate is not None)
        self.fit_button.setEnabled(False)
        self.fit_series_action.setEnabled(send_series_to_fitting is not None)
        self.assistant_button.setVisible(assistant is not None)
        self.assistant_button.setEnabled(False)
        self._last_folder = ""
        self._refresh_profiles()
        self._apply_preferences()
        self._update_frame_controls(None)
        self._sync_fit_side()
        self._connect()
        self._connect_center()
        self._connect_options()
        self._connect_workspace()
        self._connect_series()
        self._connect_batch_run()
        self._connect_regions()
        self._connect_undo()
        self._connect_marks()
        self._connect_region_pick()
        self._connect_masks()

    # -- wiring ------------------------------------------------------------------------

    def _connect(self) -> None:
        self.open_files_button.clicked.connect(self.open_files)
        self.open_folder_action.triggered.connect(self.open_folder)
        self.clear_button.clicked.connect(self.clear_files)
        self.file_list.currentRowChanged.connect(self._file_selected)
        self.frame_spin.valueChanged.connect(self._frame_changed)
        self.profile_combo.activated.connect(self._profile_chosen)
        self.mode_combo.activated.connect(self._mode_chosen)
        self.incidence_spin.valueChanged.connect(self._incidence_changed)
        self.auto_cuts_button.clicked.connect(self._auto_cuts)
        self.export_button.clicked.connect(self.export_current)
        self.fit_button.clicked.connect(self.fit_current)
        self.assistant_button.clicked.connect(self._start_assistant)
        self.fit_side_group.triggered.connect(self._fit_side_chosen)
        self.batch_button.triggered.connect(lambda: self.batch_export_dialog())
        self.stop_pipeline_button.clicked.connect(self._stop_clicked)
        for button in (self.batch_export_button, self.data_batch_button, self.series_batch_button):
            button.clicked.connect(lambda _checked=False: self.batch_export_dialog())
        self.save_settings_action.triggered.connect(lambda: self.save_settings())
        self.load_settings_action.triggered.connect(lambda: self.load_settings())
        self.fit_series_action.triggered.connect(self.send_series_dialog)
        self.save_image_action.triggered.connect(self.save_image)
        self.save_upper_plot_action.triggered.connect(lambda: self.save_plot(self.top_plot, "upper"))
        self.save_lower_plot_action.triggered.connect(lambda: self.save_plot(self.bottom_plot, "lower"))
        self.watch_button.clicked.connect(self._watch_clicked)
        self.auto_export_check.toggled.connect(lambda _checked: self._remember())
        self.calibrate_button.clicked.connect(self._open_calibration)
        self.banner_calibrate_button.clicked.connect(self._open_calibration)
        self.use_fitting_button.clicked.connect(self._use_fitting_geometry)
        self.enter_geometry_button.clicked.connect(self._enter_geometry)
        self.edit_geometry_button.clicked.connect(self._enter_geometry)
        # ``activated`` fires only for user choices, never while Qt tears the combo down.
        self.view_combo.activated.connect(self._view_chosen)
        self.detector_view.horizontalBandChanged.connect(self._horizontal_band_moved)
        self.detector_view.verticalBandChanged.connect(self._vertical_band_moved)
        self.detector_view.positionClicked.connect(self._position_clicked)
        self.top_plot.windowChanged.connect(self._chi_window_moved)
        self.lower_choice.activated.connect(self._lower_profile_chosen)
        self.canvas_save_figure_action.triggered.connect(self.save_image)
        self.canvas_save_data_action.triggered.connect(self.save_view_data)
        for plot, suffix in ((self.top_plot, "upper"), (self.bottom_plot, "lower")):
            plot.saveFigureRequested.connect(lambda plot=plot, suffix=suffix: self.save_plot(plot, suffix))
            plot.saveDataRequested.connect(lambda name=suffix: self.save_plot_data(name))
        self.detector_view.set_readout(self._readout)
        # Ctrl+O / Ctrl+Shift+O are File-menu shortcuts of the main window.
        for keys, slot in (
            ("Ctrl+E", self.export_current),
            ("F5", self.run_analysis),
            ("Esc", lambda: self.detector_view.set_pick_mode(False)),
        ):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

    # -- files ---------------------------------------------------------------------------

    def open_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open detector frames", self._last_folder, FILE_FILTER
        )
        if paths:
            self._remember(last_folder=str(Path(paths[0]).parent))
            self.add_paths(paths)

    def open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Open a folder of detector frames", self._last_folder
        )
        if folder:
            self._remember(last_folder=folder)
            self.add_paths([folder])

    def add_paths(self, paths) -> list[Path]:
        added = self.view_model.add_paths(paths)
        if not added:
            self._status("No new detector frames (CBF, NXS, TIFF, EDF) found.", "warning")
            return added
        self._append_list_items(added)
        first_new = len(self.view_model.state.files) - len(added)
        self.show_step("data")
        self.file_list.setCurrentRow(first_new)
        self.refresh_batch_entry()
        self.remember_recent(paths)
        return added

    def clear_files(self) -> None:
        self.tasks.cancel("analyze")
        self.view_model.clear_files()
        with QSignalBlocker(self.file_list):
            self.file_list.clear()
        self.detector_view.clear()
        self.top_plot.clear_curves()
        self.bottom_plot.clear_curves()
        self.banner.hide()
        self.summary_label.setText("Open or drop detector files (CBF, NXS, TIFF, EDF) to start")
        self.file_chip.setText("No data yet — open or drop detector frames")
        self.file_meta.setText("")
        self._refresh_workspace(None)
        self._update_frame_controls(None)
        self.refresh_batch_entry()
        self._status("Ready")

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.add_paths(paths)
            event.acceptProposedAction()

    def _file_selected(self, row: int) -> None:
        if self.view_model.select(row):
            path = self.view_model.current_path
            if path is not None:
                self._file_loading(path)
            self.run_analysis()

    def _frame_changed(self, value: int) -> None:
        self.view_model.set_frame(int(value) - 1)
        self.run_analysis()

    # -- analysis ------------------------------------------------------------------------

    def run_analysis(self) -> None:
        self._observe_setup()
        request = self.view_model.request()
        if request is None:
            return
        self._started_at = time.perf_counter()
        self._status(f"Analysing {request.path.name} …")
        self.tasks.submit(
            "analyze",
            lambda: self.view_model.analyze(request),
            on_done=self.show_analysis,
            on_error=self._analysis_failed,
        )

    def show_analysis(self, analysis: FrameAnalysis) -> None:
        previous = self.view_model.state.analysis
        self.view_model.accept(analysis)
        self.summary_label.setText(self.view_model.geometry_summary())
        self._update_frame_controls(analysis)
        self._update_incidence(analysis)
        same_frame_shape = previous is not None and previous.shape == analysis.shape
        self.banner.setVisible(analysis.reduction is None)
        if analysis.reduction is None:
            self.banner_label.setText(" ".join(analysis.messages))
            self._offer_previous_geometry()
        has_map = analysis.reduction is not None and analysis.reduction.reciprocal_space_map is not None
        giwaxs = analysis.reduction is not None and analysis.kind != GISAXS
        with QSignalBlocker(self.view_combo):
            self.view_combo.setEnabled(has_map)
            self.view_combo.button(2).setEnabled(giwaxs)  # the cake is a GIWAXS view
            if not has_map or (self.view_combo.currentIndex() == 2 and not giwaxs):
                self.view_combo.setCurrentIndex(0)
        self._show_image(keep_view=same_frame_shape)
        self._show_curves(analysis)
        self._update_center_control()
        self._sync_options(analysis.kind)
        self.fit_button.setEnabled(
            self._send_to_fitting is not None and analysis.reduction is not None
        )
        self.assistant_button.setEnabled(self._assistant is not None)
        self._refresh_workspace(analysis)
        self._refresh_series_curves(analysis)
        elapsed = time.perf_counter() - self._started_at
        if analysis.reduction is None:
            self._status(analysis.messages[0] if analysis.messages else "No geometry.", "warning")
        elif analysis.messages:
            self._status(" · ".join(analysis.messages), "warning")
        else:
            curves = sum(not curve.is_empty for curve in analysis.reduction.curves)
            name = analysis.path.name
            if analysis.summed_frames:
                name += f" (sum of {analysis.frame_total} frames)"
            self._status(f"{name}: {curves} curves in {elapsed:.1f} s", "ok")
        self.analysisShown.emit(analysis)
        self.refresh_batch_entry()
        if analysis.reduction is not None:
            self._offer_last_setup()
        self._batch_when_ready(analysis)

    def _analysis_failed(self, message: str, _details: str) -> None:
        self._status(f"Could not analyse the frame: {message}", "error")
        self.file_meta.setText("could not be read")
        self.set_step_state("data", "error", message)
        self.analysisFailed.emit(message)

    # -- automation (the assistant) --------------------------------------------------------

    def automation(self):
        """The page's automation surface (created once), for another component to operate it."""
        if self._automation is None:
            from .automation import AnalyzeAutomation

            self._automation = AnalyzeAutomation(self)
        return self._automation

    def _start_assistant(self) -> None:
        if self._assistant is not None:
            self._assistant()

    def _view_chosen(self, _index: int) -> None:
        self.detector_view.set_pick_mode(False)
        self._show_image(keep_view=False)
        self._update_center_control()
        self._refresh_overlay()

    def _update_frame_controls(self, analysis: Optional[FrameAnalysis]) -> None:
        count = analysis.frame_count if analysis is not None else 1
        visible = count > 1
        self.frame_label.setVisible(visible)
        self.frame_spin.setVisible(visible)
        if analysis is not None:
            with QSignalBlocker(self.frame_spin):
                self.frame_spin.setMaximum(count)
                self.frame_spin.setValue(analysis.frame_index + 1)

    def _update_incidence(self, analysis: FrameAnalysis) -> None:
        if self.view_model.state.incidence_deg is not None or analysis.geometry is None:
            return
        with QSignalBlocker(self.incidence_spin):
            self.incidence_spin.setValue(analysis.geometry.incidence_deg)

    def _horizontal_band_moved(self, low: float, high: float) -> None:
        self.view_model.set_horizontal_band(low, high)
        self.run_analysis()

    def _vertical_band_moved(self, low: float, high: float) -> None:
        self.view_model.set_vertical_band(low, high)
        self.run_analysis()

    def _position_clicked(self, x: float, _y: float) -> None:
        analysis = self.view_model.state.analysis
        if analysis is None or analysis.kind != GISAXS or self.view_combo.currentIndex() != 0:
            return
        half = self.view_model.state.gisaxs.vertical_half_width_px
        self.view_model.set_vertical_band(x - half, x + half)
        self.run_analysis()

    def _auto_cuts(self) -> None:
        self.view_model.reset_cuts()
        self.run_analysis()

    def _sync_fit_side(self) -> None:
        side = self.view_model.fit_side
        for key, action in self.fit_side_actions.items():
            action.setChecked(key == side)
        self.fit_button.setToolTip(
            "Open the curve in Fitting (q in Å⁻¹). GISAXS: the horizontal cut, "
            f"{self.fit_side_actions[side].text().lower()}; GIWAXS: I(q). "
            "The arrow chooses the half."
        )

    def _fit_side_chosen(self, action) -> None:
        self.view_model.set_fit_side(action.data())
        self._sync_fit_side()

    def _mode_chosen(self, _index: int) -> None:
        self.view_model.set_mode(self.mode_combo.currentData())
        self._remember()
        self.run_analysis()

    def _incidence_changed(self, value: float) -> None:
        self.view_model.set_incidence(None if value <= INCIDENCE_FROM_PROFILE + 1e-9 else value)
        self._remember()
        self.run_analysis()

    def dispose(self) -> None:
        """Stop background work and release the pyqtgraph views (call before deletion)."""
        self.save_last_setup()
        self._watch_timer.stop()
        self.view_model.stop_watch()
        self.tasks.cancel("analyze")
        self.tasks.cancel("batch")
        self.tasks.cancel("series")
        self.tasks.cancel("cake")
        self.shape_layer.cancel_drawing()
        self.shutdown_batch()
        self._dispose_undo()
        for view in (self.detector_view, self.top_plot, self.bottom_plot, self.series_map_view,
                     self.series_profile_plot, self.series_trace_plot):
            view.dispose()

    # -- status ------------------------------------------------------------------------

    def _status(self, text: str, level: str = "info") -> None:
        """Status line instead of pop-ups; the colour comes from the ``level`` QSS property."""
        self.status_label.setText(text)
        self.status_label.setProperty("level", level if level in STATUS_LEVELS else "info")
        style = self.status_label.style()
        style.unpolish(self.status_label)
        style.polish(self.status_label)

    def status_level(self) -> str:
        return str(self.status_label.property("level") or "info")

    def status_text(self) -> str:
        return self.status_label.text()


__all__ = ["AnalyzePage"]
