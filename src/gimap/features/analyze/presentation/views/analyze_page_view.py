"""Static layout of the Analyze workspace (behaviour lives in ``page.py`` and ``bindings/``).

A command bar — open, the current file, mode and αi, and what to do next (run the automatic
analysis, ask the AI, export, send to Fitting) — then three panels: the process steps with
their controls (``analyze_steps_view.py``), the image with its overlays, and the curves. The
status line with a progress bar closes the page.
"""

from __future__ import annotations

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import CurvePlot, DetectorView, EmptyState, SegmentedControl, ShapeLayer

from .analyze_steps_view import FIT_SIDE_ITEMS, STEPS, AnalyzeStepsView
from .batch_progress_view import BatchProgressPanel
from .command_bar import CommandBar, ElidedLabel
from .options_panel_view import OptionsPanelView
from .regions_view import RegionsView
from .series_view import SeriesView

AUTO_PROFILE_TEXT = "Automatic (match detector)"
FILE_FILTER = "Detector frames (*.cbf *.nxs *.tif *.tiff *.edf);;All files (*)"
MODE_ITEMS = (("Auto", "auto"), ("GISAXS", "gisaxs"), ("GIWAXS", "giwaxs"))
VIEW_ITEMS = ("Detector", "q map", "Cake")
VIEW_DETECTOR, VIEW_Q_MAP, VIEW_CAKE = range(3)
RIGHT_TABS = (("Curves", "curves"), ("Results", "results"), ("Series", "series"))
INCIDENCE_FROM_PROFILE = -0.001
SPLITTER_SIZES = [300, 610, 610]
"""Steps | image | curves and results before the page has its width."""
INCIDENCE_TIP = (
    "Grazing angle αi of this measurement. At the minimum (“from profile”) the "
    "instrument profile's value is used."
)
TOP_SIDE_TIPS = (
    "Both halves, as measured (display only)",
    "Only the positive half (display only)",
    "Only the negative half (display only)",
    "Both halves on |qy|, the negative one dashed in a second shade (display only)",
)
"""The halves control of the upper plot, item by item (also the entries of its “⋯” menu when the plot is narrow):
it starts at the halves chosen for Fitting, then is the plot's own."""
TOP_SIDES_TIP = "The halves shown in this plot (display only): the halves for Fitting are chosen in the Cuts step"
INCIDENCE_FROM_PROFILE_TEXT = "αi from profile"
"""What the αi field shows at its minimum: Qt draws no prefix there, so the name is in the text."""


def _separator(parent: QWidget) -> QFrame:
    line = QFrame(parent)
    line.setObjectName("toolbarSeparator")
    line.setFrameShape(QFrame.VLine)
    line.setFixedWidth(1)
    return line


FILE_STEP_SIZE = 24
"""The previous / next file buttons: at least this square (px), an easy target; the chevron is drawn at 16 px."""


def _step_button(bar: QWidget, name: str, text: str, tip: str) -> QToolButton:
    """A previous / next file button: an icon only (the text is its accessible fallback)."""
    button = QToolButton(bar)
    button.setObjectName(name)
    button.setText(text)
    button.setAccessibleName(tip)
    button.setToolButtonStyle(Qt.ToolButtonIconOnly)
    button.setIconSize(QSize(16, 16))
    button.setMinimumSize(FILE_STEP_SIZE, FILE_STEP_SIZE)
    button.setAutoRaise(True)
    button.setToolTip(tip)
    return button


class AnalyzePageView(AnalyzeStepsView, OptionsPanelView, RegionsView, SeriesView):
    """Command bar, then process | image | curves, then the status line."""

    def setup_ui(self, page: QWidget) -> None:
        page.setObjectName("analyzePage")
        root = QVBoxLayout(page)
        root.setContentsMargins(12, 10, 12, 6)
        root.setSpacing(8)
        root.addWidget(self._command_bar(page))
        root.addWidget(self._banner(page))

        self.splitter = QSplitter(Qt.Horizontal, page)
        self.splitter.setObjectName("analyzeSplitter")
        self.splitter.setHandleWidth(8)
        self.splitter.setChildrenCollapsible(False)
        self.process_panel = self.setup_process_panel(self.splitter)
        self.process_panel.setMinimumWidth(290)  # the step texts and buttons are never cut
        self.splitter.addWidget(self.process_panel)
        self.splitter.addWidget(self._detector_panel(self.splitter))
        self.splitter.addWidget(self._curves_panel(self.splitter))
        # The steps keep their width; the image and the curves / results share the rest (``bindings/workspace.py``
        # ``EvenSplit`` balances them until the person moves a handle).
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 1)
        self.splitter.setSizes(SPLITTER_SIZES)
        root.addWidget(self.splitter, 1)
        root.addLayout(self._status_row(page))
        # Kept for callers of the former Options toggle: checking it shows the Mask step.
        self.options_button = QToolButton(page)
        self.options_button.setObjectName("analyzeOptionsButton")
        self.options_button.setCheckable(True)
        self.options_button.hide()

    # -- command bar ---------------------------------------------------------------------

    def _command_bar(self, page: QWidget) -> QFrame:
        bar = self.command_bar = CommandBar(page)
        bar.setObjectName("analyzeCommandBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(6)

        self.open_files_button = QToolButton(bar)
        self.open_files_button.setObjectName("analyzeOpenButton")
        self.open_files_button.setText("Open…")
        self.open_files_button.setPopupMode(QToolButton.MenuButtonPopup)
        self.open_files_button.setToolTip(
            "Open detector frames (CBF, NXS, TIFF, EDF) — Ctrl+O. You can also drop files or folders here."
        )
        open_menu = QMenu(self.open_files_button)
        self.open_folder_action = QAction("Open Folder…", page)
        self.open_folder_action.setToolTip("Open every detector frame in a folder — Ctrl+Shift+O")
        open_menu.addAction(self.open_folder_action)
        self.open_files_button.setMenu(open_menu)

        self.undo_button = QToolButton(bar)
        self.undo_button.setObjectName("analyzeUndoButton")
        self.undo_button.setAutoRaise(True)
        self.undo_button.setToolTip("Undo the last change of the set-up (Ctrl+Z)")
        self.redo_button = QToolButton(bar)
        self.redo_button.setObjectName("analyzeRedoButton")
        self.redo_button.setAutoRaise(True)
        self.redo_button.setToolTip("Redo (Ctrl+Shift+Z)")

        # Chevrons drawn in the theme's text colour (``bindings/file_list.py``), as large as undo / redo.
        self.previous_file_button = _step_button(bar, "analyzePreviousFile", "‹", "Previous file (Page Up)")
        self.next_file_button = _step_button(bar, "analyzeNextFile", "›", "Next file (Page Down)")
        self.file_position_label = QLabel("", bar)
        self.file_position_label.setObjectName("analyzeFilePosition")
        self.file_position_label.setProperty("gimapRole", "muted")
        for widget in (self.previous_file_button, self.next_file_button, self.file_position_label):
            widget.setProperty("gimapWanted", False)  # shown (“3 / 40”) once more than one file is listed
            widget.hide()
        self.file_chip = ElidedLabel("No data yet — open or drop detector frames", bar)
        self.file_chip.setObjectName("analyzeFileChip")
        self.file_chip.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.file_meta = QLabel("", bar)
        self.file_meta.setObjectName("analyzeFileMeta")
        self.file_meta.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        self.mode_combo = SegmentedControl(bar)
        self.mode_combo.setObjectName("analyzeModeControl")
        self.mode_combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        for text, value in MODE_ITEMS:
            self.mode_combo.addItem(text, value)
        self.mode_combo.setItemToolTip(0, "Decide from the largest scattering angle on the detector")
        self.mode_combo.setItemToolTip(1, "Small angles: Yoneda cut, vertical cut, the symmetric halves")
        self.mode_combo.setItemToolTip(2, "Wide angles: rings, sectors, the q map and the cake")
        self.incidence_spin = QDoubleSpinBox(bar)
        self.incidence_spin.setObjectName("analyzeIncidenceSpin")
        self.incidence_spin.setDecimals(3)
        self.incidence_spin.setRange(INCIDENCE_FROM_PROFILE, 10.0)
        self.incidence_spin.setSingleStep(0.01)
        self.incidence_spin.setPrefix("αi ")  # the name stays in view when the narrow bar drops captions
        self.incidence_spin.setSuffix(" °")
        self.incidence_spin.setSpecialValueText(INCIDENCE_FROM_PROFILE_TEXT)
        self.incidence_spin.setKeyboardTracking(False)
        self.incidence_spin.setValue(INCIDENCE_FROM_PROFILE)
        # No fixed minimum width: its size hint (the special text, the widest value) is its minimum, so
        # “αi from profile” is never cut, in any language.
        self.incidence_spin.setContextMenuPolicy(Qt.CustomContextMenu)  # “Back to Profile αi”
        self.incidence_spin.setToolTip(INCIDENCE_TIP)
        self.incidence_reset_action = QAction("Back to Profile αi", page)
        self.incidence_reset_action.setToolTip("Use the grazing angle of the instrument profile again")

        self.run_pipeline_button = QPushButton("Run Automatic Analysis", bar)
        self.run_pipeline_button.setObjectName("analyzeRunButton")
        self.run_pipeline_button.setToolTip(
            "The standard procedure without AI: geometry (finding a calibration if needed), "
            "mask, cuts and results, each step with what it found"
        )
        self.run_pipeline_button.hide()
        self.stop_pipeline_button = QPushButton("Stop", bar)
        self.stop_pipeline_button.setObjectName("analyzeStopButton")
        self.stop_pipeline_button.setProperty("gimapDangerAction", True)
        self.stop_pipeline_button.setToolTip(
            "End the automatic analysis before its next step; what it found so far is kept"
        )
        self.stop_pipeline_button.hide()
        self.assistant_button = QPushButton("Ask AI…", bar)
        self.assistant_button.setObjectName("analyzeAssistantButton")
        self.assistant_button.setToolTip(
            "Let the AI process this frame with the same tools while you watch; its changes come "
            "back as cards you can preview, apply or undo"
        )
        self.export_button = self._export_button(bar, page)
        self.batch_export_button = QPushButton("Batch Export…", bar)
        self.batch_export_button.setObjectName("analyzeBatchExportCommand")
        self.batch_export_button.setProperty("gimapRole", "accent")
        self.batch_export_button.setToolTip(
            "Every listed frame with the current settings, to a folder you choose — Ctrl+Shift+E"
        )
        self.batch_export_button.hide()  # shown once more than one frame is listed
        self.fit_button = self._fit_button(bar, page)

        row.addWidget(self.open_files_button)
        row.addWidget(self.undo_button)
        row.addWidget(self.redo_button)
        row.addWidget(_separator(bar))
        row.addWidget(self.previous_file_button)
        row.addWidget(self.file_chip)
        row.addWidget(self.next_file_button)
        row.addWidget(self.file_position_label)
        row.addWidget(self.file_meta, 1)
        row.addWidget(self.mode_combo)
        row.addWidget(self.incidence_spin)
        row.addWidget(_separator(bar))
        row.addWidget(self.run_pipeline_button)
        row.addWidget(self.stop_pipeline_button)
        row.addWidget(self.assistant_button)
        row.addWidget(self.export_button)
        row.addWidget(self.batch_export_button)
        row.addWidget(self.fit_button)
        bar.set_compact_parts(
            hidden=(self.previous_file_button, self.next_file_button, self.file_position_label, self.file_meta),
            texts=((self.run_pipeline_button, "Run Automatic Analysis", "Run Analysis"),
                   (self.batch_export_button, "Batch Export…", "Batch…"),
                   (self.fit_button, "Send to Fitting", "Fitting")),
        )
        return bar

    def _export_button(self, bar: QWidget, page: QWidget) -> QToolButton:
        button = QToolButton(bar)
        button.setObjectName("analyzeExportButton")
        button.setText("Export")
        button.setPopupMode(QToolButton.MenuButtonPopup)
        button.setToolTip("Write CSV curves and a JSON record next to the data (gimap_analysis/) — Ctrl+E")
        menu = QMenu(button)
        self.batch_button = QAction("Batch Export…", page)
        self.batch_button.setToolTip("Every listed frame with the current settings: choose what to save and where")
        self.auto_export_check = QAction("Export New Frames While Watching", page)
        self.auto_export_check.setCheckable(True)
        menu.addAction(self.batch_button)
        menu.addAction(self.auto_export_check)
        menu.addSection("Settings")
        self.save_settings_action = QAction("Save Settings…", page)
        self.save_settings_action.setToolTip("Geometry, masks, corrections and cut regions as a file, for the next data set")
        self.load_settings_action = QAction("Load Settings…", page)
        self.load_settings_action.setToolTip("Use a set-up saved before")
        menu.addAction(self.save_settings_action)
        menu.addAction(self.load_settings_action)
        menu.addSection("Figures and data")
        self.save_image_action = QAction("Save Image…", page)
        self.save_image_action.setToolTip("The detector or q map as shown, with a colour bar")
        self.save_map_action = QAction("Save q-Map Data…", page)
        self.save_map_action.setToolTip("The intensity on a regular q grid, as a CSV table with its axes")
        self.save_upper_plot_action = QAction("Save Upper Plot…", page)
        self.save_lower_plot_action = QAction("Save Lower Plot…", page)
        for action in (self.save_image_action, self.save_map_action, self.save_upper_plot_action, self.save_lower_plot_action):
            menu.addAction(action)
        button.setMenu(menu)
        return button

    def _fit_button(self, bar: QWidget, page: QWidget) -> QToolButton:
        button = QToolButton(bar)
        button.setObjectName("analyzeFitButton")
        button.setText("Send to Fitting")
        button.setProperty("gimapRole", "primary")
        button.setPopupMode(QToolButton.MenuButtonPopup)
        menu = QMenu(button)
        menu.addSection("Horizontal cut (GISAXS)")
        self.fit_side_group = QActionGroup(page)
        self.fit_side_group.setExclusive(True)
        self.fit_side_actions: dict[str, QAction] = {}
        for key, title in FIT_SIDE_ITEMS:
            action = QAction(title, page)
            action.setCheckable(True)
            action.setData(key)
            self.fit_side_group.addAction(action)
            menu.addAction(action)
            self.fit_side_actions[key] = action
        menu.addSeparator()
        self.fit_series_action = QAction("Send Series to Fitting…", page)
        self.fit_series_action.setToolTip(
            "Export every listed frame (or sum of frames) and open the series in Fitting ▸ In-situ series"
        )
        menu.addAction(self.fit_series_action)
        button.setMenu(menu)
        return button

    def _banner(self, page: QWidget) -> QFrame:
        self.banner = QFrame(page)
        self.banner.setObjectName("analyzeGeometryBanner")
        banner_layout = QHBoxLayout(self.banner)
        banner_layout.setContentsMargins(10, 6, 8, 6)
        self.banner_label = QLabel(self.banner)
        self.banner_label.setWordWrap(True)
        self.use_fitting_button = QPushButton("Use Previous Geometry…", self.banner)
        self.use_fitting_button.hide()
        self.banner_find_button = QPushButton("Find Calibration Automatically", self.banner)
        self.banner_find_button.setObjectName("analyzeBannerFindButton")
        self.banner_find_button.setProperty("gimapRole", "primary")
        self.banner_find_button.hide()
        self.enter_geometry_button = QPushButton("Enter Geometry…", self.banner)
        self.banner_calibrate_button = QPushButton("Calibrate…", self.banner)
        banner_layout.addWidget(self.banner_label, 1)
        banner_layout.addWidget(self.use_fitting_button)
        banner_layout.addWidget(self.banner_find_button)
        banner_layout.addWidget(self.enter_geometry_button)
        banner_layout.addWidget(self.banner_calibrate_button)
        self.banner.hide()
        return self.banner

    # -- image and curves --------------------------------------------------------------

    def _detector_panel(self, parent: QWidget) -> QWidget:
        center = QFrame(parent)
        center.setObjectName("analyzeCanvasPanel")
        layout = QVBoxLayout(center)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self.view_combo = SegmentedControl(center)
        self.view_combo.setObjectName("analyzeViewControl")
        self.view_combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.view_combo.addItems(VIEW_ITEMS)
        self.view_combo.setToolTip("Show the detector, or the intensity regridded onto q")
        self.view_combo.setItemToolTip(0, "The detector frame as measured, in pixels")
        self.view_combo.setItemToolTip(1, "Intensity regridded onto q∥ (qr, qy) and qz")
        self.view_combo.setItemToolTip(2, "GIWAXS unwrapped: χ against q; a ring is a vertical line, a region a rectangle")
        self.detector_view = DetectorView(center)
        self.detector_view.toolbar_layout.insertWidget(0, self.view_combo)
        self.sources_button = QToolButton(center)
        self.sources_button.setObjectName("analyzeSourcesButton")
        self.sources_button.setText("Sources")
        self.sources_button.setCheckable(True)
        self.sources_button.setToolTip(
            "Show on the detector image which pixels each curve comes from, in the colours of the plots; "
            "click a curve to show only its pixels"
        )
        self.detector_view.toolbar_layout.addWidget(self.sources_button)
        self.shape_layer = ShapeLayer(self.detector_view)
        self.canvas_save_button = QToolButton(center)
        self.canvas_save_button.setObjectName("analyzeCanvasSave")
        self.canvas_save_button.setText("Save")
        self.canvas_save_button.setToolTip("Save the view as shown (a figure) or its data (a table)")
        self.canvas_save_button.setPopupMode(QToolButton.InstantPopup)
        canvas_menu = QMenu(self.canvas_save_button)
        self.canvas_save_figure_action = canvas_menu.addAction("View as Figure…")
        self.canvas_save_data_action = canvas_menu.addAction("View Data as CSV…")
        self.canvas_save_data_action.setToolTip("The q map (q∥–qz) or the cake (χ–q) as a table with its axes")
        self.canvas_save_button.setMenu(canvas_menu)
        self.detector_view.toolbar_layout.addWidget(self.canvas_save_button)
        layout.addWidget(self.detector_view, 1)
        return center

    def _curves_panel(self, parent: QWidget) -> QWidget:
        panel = QFrame(parent)
        panel.setObjectName("analyzeCurvesPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.right_tabs = SegmentedControl(panel)
        self.right_tabs.setObjectName("analyzeRightTabs")
        for text, key in RIGHT_TABS:
            self.right_tabs.addItem(text, key)
        self.right_tabs.setItemToolTip(0, "The curves of this frame (cuts, rings, sectors)")
        self.right_tabs.setItemToolTip(1, "What the automatic analysis (or the AI) found, with its evidence")
        self.right_tabs.setItemToolTip(2, "Every frame of a series: intensity against frame and q")
        header = QHBoxLayout()
        header.addWidget(self.right_tabs)
        header.addStretch(1)
        layout.addLayout(header)
        self.batch_panel = BatchProgressPanel(panel)
        layout.addWidget(self.batch_panel)
        self.right_stack = QStackedWidget(panel)
        self.right_pages: dict[str, QWidget] = {}

        curves = QWidget(self.right_stack)
        curves_layout = QVBoxLayout(curves)
        curves_layout.setContentsMargins(0, 0, 0, 0)
        curves_layout.setSpacing(6)
        self.top_plot = CurvePlot("", curves)
        for index, tip in enumerate(TOP_SIDE_TIPS):
            self.top_plot.side_control.setItemToolTip(index, tip)
        self.top_plot.side_control.setToolTip(TOP_SIDES_TIP)
        self.bottom_plot = CurvePlot("", curves)
        for plot in (self.top_plot, self.bottom_plot):
            plot.add_save_menu()
        self.lower_choice = QComboBox(curves)
        self.lower_choice.setObjectName("analyzeLowerProfile")
        self.lower_choice.setToolTip("Which GIWAXS profile the lower plot shows")
        self.lower_choice.hide()
        self.bottom_plot.header_layout.insertWidget(0, self.lower_choice)
        curves_layout.addWidget(self.top_plot, 1)
        curves_layout.addWidget(self.bottom_plot, 1)

        results = QScrollArea(self.right_stack)
        results.setObjectName("analyzeResultsPage")
        results.setWidgetResizable(True)
        results.setFrameShape(QFrame.NoFrame)
        results.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # everything wraps or scrolls on its own
        results_content = QWidget(results)
        self.results_panel_host = QVBoxLayout(results_content)
        self.results_panel_host.setContentsMargins(2, 2, 6, 2)
        self.results_panel_empty = EmptyState(
            "No results yet",
            "Run Automatic Analysis (no AI needed) or Ask AI; what they find — peaks, orientation, sizes, "
            "checks — appears here with its evidence.",
            results_content,
        )
        self.results_panel_host.addWidget(self.results_panel_empty)
        self.results_panel_host.addStretch(1)
        results.setWidget(results_content)

        series = QWidget(self.right_stack)
        self.series_host = QVBoxLayout(series)
        self.series_host.setContentsMargins(0, 0, 0, 0)
        self.series_host.setSpacing(6)
        self.setup_series_panel(series, self.series_host)

        for key, page in (("curves", curves), ("results", results), ("series", series)):
            self.right_stack.addWidget(page)
            self.right_pages[key] = page
        layout.addWidget(self.right_stack, 1)
        return panel

    def _status_row(self, page: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        self.status_label = QLabel("Ready", page)
        self.status_label.setObjectName("analyzeStatus")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.progress_bar = QProgressBar(page)
        self.progress_bar.setObjectName("analyzeProgress")
        self.progress_bar.setMaximumWidth(220)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        self.cancel_button = QToolButton(page)
        self.cancel_button.setObjectName("analyzeCancelButton")
        self.cancel_button.setText("Cancel")
        self.cancel_button.hide()
        row.addWidget(self.status_label, 1)
        row.addWidget(self.progress_bar)
        row.addWidget(self.cancel_button)
        return row


__all__ = [
    "AUTO_PROFILE_TEXT",
    "AnalyzePageView",
    "FILE_FILTER",
    "FIT_SIDE_ITEMS",
    "INCIDENCE_FROM_PROFILE",
    "INCIDENCE_TIP",
    "MODE_ITEMS",
    "RIGHT_TABS",
    "STEPS",
    "VIEW_ITEMS",
]
