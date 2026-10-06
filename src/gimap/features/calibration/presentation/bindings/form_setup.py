"""Form Setup behavior for Calibration."""

from __future__ import annotations

import logging


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure


from PyQt5.QtCore import QCoreApplication, QEvent, QSignalBlocker, QSize, QTimer, Qt

from PyQt5.QtWidgets import (
    QHeaderView,
)


from src.gimap.app.presentation.i18n import language_changed, tr
from src.gimap.app.presentation.theme import set_role
from src.gimap.app.presentation.theme.figures import exported_colors, theme_figure
from src.gimap.app.presentation.section_bindings import (
    bind_advanced_section,
    bind_parameter_section,
)


from ..preview_style import (
    CENTER_COLOR,
    DETECTED_RING_COLOR,
    MATCHED_RING_COLOR,
    UNMATCHED_RING_COLOR,
)

LOGGER = logging.getLogger(__name__)


class PreviewToolbar(NavigationToolbar):
    """Home, back/forward, pan, zoom and save; the cursor readout goes to a label.

    The subplot and figure-option dialogs would fight the fixed preview margins, and the
    toolbar's own two-line coordinate label would make the action row jump in height.
    """

    toolitems = [
        item for item in NavigationToolbar.toolitems if item[0] not in ("Subplots", "Customize")
    ]

    def __init__(self, canvas, parent, readout=None):
        super().__init__(canvas, parent, coordinates=False)
        self._readout = readout

    def set_message(self, s) -> None:
        readout = getattr(self, "_readout", None)
        if readout is not None:
            readout.setText(" ".join(str(s).split()))

    def save_figure(self, *args):
        """Saved files get matplotlib's own colours, not the screen theme's."""
        with exported_colors(self.canvas.figure):
            saved = super().save_figure(*args)
        self.canvas.draw_idle()
        return saved


class FormSetupMixin:
    """Own form setup presentation behavior."""

    def _apply_dialog_style(self) -> None:
        """Semantic roles only; colours come from the application theme."""
        set_role(self.calibrate_button, "primary")
        set_role(self.overlay_legend, "overlay")
        for label in (
            self.preview_info_label,
            self.preview_cursor_label,
            self.manual_hint,
            self.preview_empty_label,
        ):
            set_role(label, "muted")

    def _bind_form(self) -> None:
        """Attach behavior and dynamic content to the Designer-owned form."""
        bind_parameter_section(
            self.calibration_input_section,
            self.calibrationInputTitle,
            self.calibrationInputDescription,
            self.calibrationInputContent,
            self.calibrationInputContentLayout,
        )
        bind_parameter_section(
            self.calibration_run_section,
            self.calibrationRunTitle,
            self.calibrationRunDescription,
            self.calibrationRunContent,
            self.calibrationRunContentLayout,
        )
        bind_parameter_section(
            self.calibration_preview_panel,
            self.calibrationPreviewTitle,
            self.calibrationPreviewDescription,
            self.calibrationPreviewContent,
            self.calibrationPreviewContentLayout,
        )
        bind_parameter_section(
            self.calibration_results_section,
            self.calibrationResultsTitle,
            self.calibrationResultsDescription,
            self.calibrationResultsContent,
            self.calibrationResultsContentLayout,
        )
        bind_parameter_section(
            self.calibration_export_section,
            self.calibrationExportTitle,
            self.calibrationExportDescription,
            self.calibrationExportContent,
            self.calibrationExportContentLayout,
        )
        bind_advanced_section(
            self.calibration_advanced_section,
            self.calibrationAdvancedToggle,
            self.calibrationAdvancedDescription,
            self.calibrationAdvancedContent,
            self.calibrationAdvancedContentLayout,
        )
        bind_advanced_section(
            self.calibration_manual_section,
            self.calibrationManualToggle,
            self.calibrationManualDescription,
            self.calibrationManualContent,
            self.calibrationManualContentLayout,
        )

        # Logic reads itemData ("auto", None, "custom"), never the displayed text.
        self.standard_combo.addItem(tr("Auto Detect"), "auto")
        for standard in self.view_model.standard_options():
            self.standard_combo.addItem(standard.display_name, standard.key)
        self.detector_combo.addItem(tr("Auto detected"), None)
        for detector_name in self.detector_models:
            self.detector_combo.addItem(detector_name, detector_name)
        self.detector_combo.addItem(tr("Custom pixel size"), "custom")

        self.calibrate_button.setObjectName("primaryCalibrationButton")
        for button in (
            self.fit_image_button,
            self.clean_preview_button,
            self.expand_preview_button,
        ):
            button.setObjectName("previewActionButton")
        self.manual_refine_button.setObjectName("manualRefineButton")
        self.manual_group.setObjectName("manualRefinementGroup")
        self.manual_hint.setObjectName("manualHint")
        self.preview_info_label.setObjectName("previewInfo")
        self.overlay_legend.setObjectName("overlayLegend")

        self.job_status.set_actions_visible(
            pause=False,
            cancel=False,
            details=False,
        )
        self.progress = self.job_status.progress_bar
        self.stage_label = self.job_status.message_label

        self.figure = Figure(figsize=(7, 5), constrained_layout=False)
        self.figure.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.96)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("resize_event", self._fit_figure_margins)
        self.axes = self.figure.add_subplot(111)
        self.toolbar = PreviewToolbar(self.canvas, self, self.preview_cursor_label)
        self.toolbar.setIconSize(QSize(18, 18))
        self.calibrationToolbarHostLayout.addWidget(self.toolbar)
        # Every navigation tool stays visible: see _place_preview_toolbar.
        self.calibrationPreviewContent.installEventFilter(self)
        self.calibrationFigureHostLayout.addWidget(self.canvas, 0, 0)
        self.preview_empty_label.raise_()
        theme_figure(self.figure, self.canvas)

        self.overlay_legend.setText(
            f'<span style="color:{CENTER_COLOR}">━━</span> '
            '<span style="color:#f8fafc">Center</span> &nbsp;&nbsp; '
            f'<span style="color:{DETECTED_RING_COLOR}">┄┄┄</span> '
            '<span style="color:#f8fafc">Detected</span> &nbsp;&nbsp; '
            f'<span style="color:{MATCHED_RING_COLOR}">━━</span> '
            '<span style="color:#f8fafc">Matched</span> &nbsp;&nbsp; '
            f'<span style="color:{UNMATCHED_RING_COLOR}">╌╌╌</span> '
            '<span style="color:#f8fafc">Other theoretical</span>'
        )
        self.overlay_legend.setAttribute(Qt.WA_StyledBackground, True)
        self.overlay_legend.setVisible(False)

        self.result_labels = {
            "Beam center X": self.result_center_x,
            "Beam center Y": self.result_center_y,
            "Distance": self.result_distance,
            "Detector rotation": self.result_rotation,
            "Matched rings": self.result_rings,
            "RMS residual": self.result_rms,
            "Confidence": self.result_confidence,
            "Warning": self.result_warning,
        }
        self.candidate_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.candidate_table.horizontalHeader().setStretchLastSection(True)
        self.results_splitter.setStretchFactor(0, 1)
        self.results_splitter.setStretchFactor(1, 2)
        self._result_two_columns = True  # as the View places the rows
        self.results_splitter.installEventFilter(self)
        language_changed().connect(self._result_language_changed)
        # Results grows for a longer solution only until the user sets the split by hand.
        self._results_user_sized = False
        self.right_splitter.splitterMoved.connect(self._right_split_moved)
        self.right_splitter.setStretchFactor(0, 4)
        self.right_splitter.setStretchFactor(1, 2)
        # The inputs column grows a little with wide windows; the preview takes the rest.
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 3)
        self._initial_layout_applied = False
        self._results_sizes = None
        self._fitted_geometry = None
        self._manual_section_opened = False
        # The empty state ("Open a … calibration image") instead of a bare 0–1 axis.
        self.redraw_preview()

    def _connect_signals(self) -> None:
        self.open_button.clicked.connect(self.open_image_dialog)
        self.path_edit.returnPressed.connect(self._load_path_edit)
        self.calibrate_button.clicked.connect(self.start_calibration)
        self.cancel_button.clicked.connect(self.cancel_calibration)
        self.close_button.clicked.connect(self.close)
        self.apply_button.clicked.connect(self.apply_result)
        self.export_button.clicked.connect(self.export_result)
        self.import_button.clicked.connect(self.import_result)
        self.candidate_table.itemSelectionChanged.connect(self._candidate_selected)
        self.log_check.toggled.connect(self.redraw_preview)
        self.mask_check.toggled.connect(self.redraw_preview)
        self.rings_check.toggled.connect(self.redraw_preview)
        self.center_check.toggled.connect(self.redraw_preview)
        self.fit_image_button.clicked.connect(self.fit_preview_to_image)
        self.clean_preview_button.toggled.connect(self._clean_preview_toggled)
        self.expand_preview_button.clicked.connect(self._toggle_preview_expanded)
        # Manual mode: 'Manual refine' or the 'Use manual values' check box. The section toggle
        # only opens and closes the section, so looking at the fields changes nothing.
        self.manual_refine_button.toggled.connect(self.manual_group.setChecked)
        self.manual_group.toggled.connect(self._manual_group_toggled)
        self.standard_combo.currentIndexChanged.connect(self._populate_theory_rings)
        self.detector_combo.currentIndexChanged.connect(self._detector_model_changed)
        for widget in (self.manual_x, self.manual_y, self.manual_distance):
            widget.valueChanged.connect(lambda _value: self._overlay_timer.start())
        self.refine_ring_button.clicked.connect(self.fit_selected_ring)
        self.reset_manual_button.clicked.connect(self.reset_manual_to_fitted)
        self.canvas.mpl_connect("button_press_event", self._preview_press)
        self.canvas.mpl_connect("motion_notify_event", self._preview_move)
        self.canvas.mpl_connect("button_release_event", self._preview_release)

    def _set_running(self, running: bool) -> None:
        self.open_button.setEnabled(not running)
        for widget in (
            self.path_edit,
            self.energy_spin,
            self.standard_combo,
            self.estimated_distance_spin,
            self.range_combo,
            self.detector_combo,
            self.pixel_x_spin,
            self.pixel_y_spin,
            self.custom_min_spin,
            self.custom_max_spin,
            self.background_check,
        ):
            widget.setEnabled(not running)
        self.calibrate_button.setEnabled(not running and self.image is not None)
        self.cancel_button.setEnabled(running)
        self.apply_button.setEnabled(not running and self.result is not None)
        self.export_button.setEnabled(not running and self.result is not None)
        self.clean_preview_button.setEnabled(not running and self.result is not None)
        self.manual_refine_button.setEnabled(not running and self.result is not None)
        self.manual_group.setEnabled(not running and self.result is not None)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Shown again (the Tools menu reuses the window): a close deferred until an earlier
        # run ended no longer applies.
        self._close_when_idle = False
        if not self._initial_layout_applied:
            self._initial_layout_applied = True
            # Splitter sizes only stick once the window has its real size.
            QTimer.singleShot(0, self._apply_initial_splitter_sizes)

    def _apply_initial_splitter_sizes(self) -> None:
        try:
            width = self.main_splitter.width()
            height = self.right_splitter.height()
        except RuntimeError:  # closed before the first layout pass
            return
        left = min(max(int(width * 0.35), 372), 540)
        self.main_splitter.setSizes([left, max(1, width - left)])
        results = min(max(int(height * 0.36), 200), 340)
        self.right_splitter.setSizes([max(1, height - results), results])
        self._arrange_result_form(initial=True)
        self._reset_preview_view = True
        self.redraw_preview()
        self._fit_results_height()

    def _right_split_moved(self, *_args) -> None:
        self._results_user_sized = True

    def eventFilter(self, watched, event) -> bool:
        if watched is getattr(self, "calibrationPreviewContent", None) and event.type() in (
            QEvent.Resize,
            QEvent.LayoutRequest,  # a button text changed (language, Clean image/Show overlays)
        ):
            self._place_preview_toolbar()
        elif (
            watched is getattr(self, "results_splitter", None)
            and event.type() == QEvent.Resize
            and getattr(self, "_initial_layout_applied", False)
        ):
            # After QSplitter's own resize handling: one or two columns for the new width.
            QTimer.singleShot(0, self._arrange_result_form)
        return super().eventFilter(watched, event)

    def _place_preview_toolbar(self) -> None:
        """Keep every navigation tool visible.

        The toolbar shares the row of the preview buttons while the row is wide enough and
        takes a row of its own below them when it is not, instead of folding zoom and pan into
        the toolbar's overflow menu.
        """
        try:
            available = self.calibrationPreviewContent.width()
            host = self.calibrationToolbarHost
            buttons = [
                button
                for button in (
                    self.fit_image_button,
                    self.clean_preview_button,
                    self.expand_preview_button,
                    self.manual_refine_button,
                )
                if not button.isHidden()
            ]
            spacing = max(self.previewActionsLayout.spacing(), 0)
            needed = (
                sum(button.sizeHint().width() for button in buttons)
                + self.toolbar.sizeHint().width()
                + 12  # the spacer between the buttons and the toolbar
                + spacing * (len(buttons) + 1)
            )
            own_row = needed > available
            if own_row == (self.previewToolbarRowLayout.indexOf(host) >= 0):
                return
            if own_row:
                self.previewActionsLayout.removeWidget(host)
                self.previewToolbarRowLayout.addWidget(host)
            else:
                self.previewToolbarRowLayout.removeWidget(host)
                self.previewActionsLayout.addWidget(host)
        except RuntimeError:  # the dialog is being destroyed
            return

    def _fit_figure_margins(self, _event=None) -> None:
        """Fixed margins in points, so tick labels and axis titles fit at every canvas size."""
        width, height = self.figure.get_size_inches() * 72.0
        if width < 120 or height < 120:
            return
        self.figure.subplots_adjust(
            left=46.0 / width,
            right=1.0 - 8.0 / width,
            bottom=34.0 / height,
            top=1.0 - 8.0 / height,
        )

    def _manual_group_toggled(self, checked: bool) -> None:
        if checked:
            # Show the fields; close the section again afterwards only if it was opened here.
            self._manual_section_opened = not self.calibration_manual_section.is_expanded()
            self.calibration_manual_section.set_expanded(True)
            # Start from the fitted solution, never from values left over from an earlier session.
            self._load_fitted_into_manual_fields()
        else:
            if self._manual_section_opened:
                self.calibration_manual_section.set_expanded(False)
            self._manual_section_opened = False
            # Manual mode off means the fitted solution, also after values were committed.
            self._restore_fitted_geometry()
        self.manual_panel.setVisible(checked)
        self.manual_group.setMaximumHeight(16777215 if checked else 40)
        blocker = QSignalBlocker(self.manual_refine_button)
        self.manual_refine_button.setChecked(checked)
        self.manual_refine_button.setText(tr("Finish manual") if checked else tr("Manual refine"))
        del blocker
        if checked:
            QTimer.singleShot(0, self._reveal_manual_fields)
        self._overlay_timer.start()

    def _reveal_manual_fields(self) -> None:
        """The manual fields live in the inputs column: scroll them into view."""
        try:
            # Let the expanded section reach its final size before scrolling to it.
            for _ in range(4):
                QCoreApplication.sendPostedEvents(None, QEvent.LayoutRequest)
            self.calibrationControlsScroll.ensureWidgetVisible(self.manual_panel, 0, 8)
        except RuntimeError:  # the dialog was closed meanwhile
            pass

    def fit_preview_to_image(self) -> None:
        self._reset_preview_view = True
        self.redraw_preview()

    def _clean_preview_toggled(self, checked: bool) -> None:
        self.clean_preview_button.setText(tr("Show overlays") if checked else tr("Clean image"))
        self.redraw_preview()

    def _toggle_preview_expanded(self) -> None:
        """'Focus image' hides the whole Results section; 'Show results' brings it back."""
        focus = not self.calibration_results_section.isHidden()
        if focus:
            self._results_sizes = self.right_splitter.sizes()
        self.calibration_results_section.setVisible(not focus)
        if not focus and self._results_sizes:
            self.right_splitter.setSizes(self._results_sizes)
        self.expand_preview_button.setText(tr("Show results") if focus else tr("Focus image"))
        self._reset_preview_view = True
        QTimer.singleShot(0, self.redraw_preview)
