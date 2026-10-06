"""The process steps of the Analyze workspace and the controls each step shows.

Data → Geometry → Mask & corrections → Cuts → Results → Export. Each step page has a
title, one sentence of what it found (``step_intro[key]``, filled by the page), the few
controls used most, and the rest under a collapsed "More" section. Behaviour lives in
``page.py`` and ``bindings/``; this module only builds widgets.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMenu,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import EmptyState, FlowLayout, StepRail

STEPS = (
    ("data", "Data"),
    ("geometry", "Geometry"),
    ("mask", "Mask & corrections"),
    ("cuts", "Cuts"),
    ("results", "Results"),
    ("export", "Export"),
)
FIT_SIDE_ITEMS = (
    ("both_abs", "Both halves on |qy| (two colours)"),
    ("mean", "Mean of both halves"),
    ("negative", "qy < 0 half"),
    ("positive", "qy > 0 half"),
)


def muted(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setProperty("gimapRole", "muted")
    return label


def info_card(parent: QWidget) -> tuple[QFrame, QLabel]:
    card = QFrame(parent)
    card.setProperty("gimapInfoCard", True)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(10, 8, 10, 8)
    label = QLabel("", card)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    layout.addWidget(label)
    return card, label


def action_row(button: QPushButton, description: str, parent: QWidget, tip: str = "") -> QWidget:
    """A button with one short line saying what it produces (the Export step); the details in its tooltip."""
    if tip or not button.toolTip():
        button.setToolTip(tip or description)
    row = QWidget(parent)
    layout = QVBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 2)
    layout.setSpacing(2)
    layout.addWidget(button, 0, Qt.AlignLeft)
    layout.addWidget(muted(description, row))
    return row


class AnalyzeStepsView:
    """Builds ``self.step_rail`` and ``self.step_stack`` with one page per step."""

    def setup_process_panel(self, parent: QWidget) -> QFrame:
        panel = QFrame(parent)
        panel.setObjectName("analyzeProcessPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.step_rail = StepRail(STEPS, panel)
        layout.addWidget(self.step_rail)
        divider = QFrame(panel)
        divider.setFrameShape(QFrame.HLine)
        divider.setProperty("gimapDivider", True)
        layout.addWidget(divider)
        self.step_stack = QStackedWidget(panel)
        self.step_stack.setObjectName("analyzeStepStack")
        self.step_pages: dict[str, QWidget] = {}
        self.step_titles: dict[str, QLabel] = {}
        self.step_intro: dict[str, QLabel] = {}
        self.build_option_sections(panel)
        for key, title in STEPS:
            scroll = QScrollArea(self.step_stack)
            scroll.setObjectName(f"analyzeStep_{key}")
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            content = QWidget(scroll)
            page_layout = QVBoxLayout(content)
            page_layout.setContentsMargins(4, 2, 6, 4)
            page_layout.setSpacing(8)
            heading = QLabel(title, content)
            heading.setProperty("gimapInspectorTitle", True)
            intro = QLabel("", content)
            intro.setProperty("gimapInspectorIntro", True)
            intro.setWordWrap(True)
            intro.setTextInteractionFlags(Qt.TextSelectableByMouse)
            page_layout.addWidget(heading)
            page_layout.addWidget(intro)
            getattr(self, f"_{key}_step")(content, page_layout)
            scroll.setWidget(content)
            self.step_stack.addWidget(scroll)
            self.step_pages[key] = scroll
            self.step_titles[key] = heading
            self.step_intro[key] = intro
        layout.addWidget(self.step_stack, 1)
        self.step_rail.set_current("data")
        return panel

    # -- 1 data ------------------------------------------------------------------------

    def _data_step(self, page: QWidget, layout: QVBoxLayout) -> None:
        header = QHBoxLayout()
        header.setSpacing(4)
        self.files_title = QLabel("Files", page)
        self.files_title.setProperty("gimapRole", "strong")
        self.watch_button = QToolButton(page)
        self.watch_button.setObjectName("analyzeWatchButton")
        self.watch_button.setText("Watch")
        self.watch_button.setCheckable(True)
        self.watch_button.setAutoRaise(True)
        self.watch_button.setToolTip(
            "Watch a folder: new frames are added and shown as soon as they are completely written"
        )
        self.clear_button = QToolButton(page)
        self.clear_button.setText("Clear")
        self.clear_button.setAutoRaise(True)
        self.clear_button.setToolTip(
            "Remove every file from the list. One file: select it and press Delete, or right-click it")
        header.addWidget(self.files_title, 1)
        header.addWidget(self.watch_button)
        header.addWidget(self.clear_button)
        layout.addLayout(header)
        self.file_list = QListWidget(page)
        self.file_list.setObjectName("analyzeFileList")
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.setTextElideMode(Qt.ElideMiddle)
        self.file_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.file_list.setMinimumHeight(120)
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)  # Remove from List, Show in Folder, Copy Path
        layout.addWidget(self.file_list, 1)
        frame_row = QHBoxLayout()
        self.frame_label = muted("Frame", page)
        self.frame_spin = QSpinBox(page)
        self.frame_spin.setObjectName("analyzeFrameSpin")
        self.frame_spin.setMinimum(1)
        self.frame_spin.setKeyboardTracking(False)
        frame_row.addWidget(self.frame_label)
        frame_row.addWidget(self.frame_spin, 1)
        layout.addLayout(frame_row)
        self.data_batch_button = QPushButton("Batch Export…", page)
        self.data_batch_button.setObjectName("analyzeDataBatchButton")
        self.data_batch_button.setProperty("gimapRole", "accent")
        self.data_batch_button.setToolTip(
            "Set up one frame (geometry, mask, cuts), then export every listed frame the same way"
        )
        self.data_batch_button.hide()
        layout.addWidget(self.data_batch_button)
        card, self.data_info_label = info_card(page)
        self.data_info_label.setObjectName("analyzeDataInfo")
        layout.addWidget(card)
        self.frames_section.set_expanded(False)
        layout.addWidget(self.frames_section)

    # -- 2 geometry --------------------------------------------------------------------

    def _geometry_step(self, page: QWidget, layout: QVBoxLayout) -> None:
        self.profile_combo = QComboBox(page)
        self.profile_combo.setObjectName("analyzeProfileCombo")
        self.profile_combo.setMinimumContentsLength(14)
        self.profile_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.profile_combo.setToolTip("Instrument profile: the detector geometry used for q")
        layout.addWidget(muted("Instrument profile", page))
        layout.addWidget(self.profile_combo)
        card, self.summary_label = info_card(page)
        self.summary_label.setObjectName("analyzeGeometrySummary")
        self.summary_label.setText("Open or drop detector files (CBF, NXS, TIFF, EDF) to start")
        layout.addWidget(card)
        self.find_geometry_button = QPushButton("Find Calibration Automatically", page)
        self.find_geometry_button.setObjectName("analyzeFindGeometryButton")
        self.find_geometry_button.setProperty("gimapRole", "primary")
        self.find_geometry_button.setToolTip(
            "Look for calibration files and images of a standard near the data, fit the geometry "
            "and check it against the standard's lines"
        )
        self.find_geometry_button.hide()
        layout.addWidget(self.find_geometry_button)
        buttons = QHBoxLayout()
        self.edit_geometry_button = QPushButton("Geometry…", page)
        self.edit_geometry_button.setObjectName("analyzeEditGeometryButton")
        self.edit_geometry_button.setToolTip(
            "Edit the geometry of the profile in use (or save it under a new name), or delete it"
        )
        self.calibrate_button = QPushButton("Calibrate…", page)
        self.calibrate_button.setToolTip("Find the beam centre and distance from a standard (AgBh …)")
        buttons.addWidget(self.edit_geometry_button)
        buttons.addWidget(self.calibrate_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        layout.addWidget(muted("Beam centre", page))
        self.center_button = QToolButton(page)
        self.center_button.setObjectName("analyzeCenterButton")
        self.center_button.setPopupMode(QToolButton.InstantPopup)
        self.center_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.center_button.setText("Beam centre")
        self.center_button.setEnabled(False)
        self.center_menu = QMenu(self.center_button)
        self.pick_center_action = QAction("Pick on Image", page)
        self.pick_center_action.setCheckable(True)
        self.pick_center_action.setToolTip("Click the direct-beam position on the detector image")
        self.enter_center_action = QAction("Enter Coordinates…", page)
        self.header_center_action = QAction("Use File Header Centre", page)
        self.symmetry_center_action = QAction("Refine x by Symmetry", page)
        self.symmetry_center_action.setToolTip(
            "GISAXS: move the centre column to the left–right symmetry axis of the "
            "horizontal cut (qy = 0), useful when the direct beam is hidden"
        )
        self.reset_center_action = QAction("Back to Profile Centre", page)
        self.save_center_action = QAction("Save to Profile", page)
        for action in (
            self.pick_center_action,
            self.enter_center_action,
            self.header_center_action,
            self.symmetry_center_action,
        ):
            self.center_menu.addAction(action)
        self.center_menu.addSeparator()
        self.center_menu.addAction(self.reset_center_action)
        self.center_menu.addAction(self.save_center_action)
        self.center_button.setMenu(self.center_menu)
        layout.addWidget(self.center_button, 0, Qt.AlignLeft)
        layout.addWidget(muted(
            "Drag the cyan cross on the image to move the centre; the change holds for every file "
            "of this detector until you go back to the profile centre.", page,
        ))
        layout.addStretch(1)

    # -- 3 mask and corrections ----------------------------------------------------------

    def _mask_step(self, page: QWidget, layout: QVBoxLayout) -> None:
        card, self.mask_summary_label = info_card(page)
        self.mask_summary_label.setObjectName("analyzeMaskSummary")
        layout.addWidget(card)
        self.bad_pixels_check = QCheckBox("Leave out hot and dead pixels", page)
        self.bad_pixels_check.setObjectName("analyzeBadPixelsCheck")
        self.bad_pixels_check.setToolTip(
            "Single pixels far brighter than all their neighbours (stuck pixels, zingers, overflow codes) "
            "or reading zero among bright neighbours. Found in every frame; anything wider than one pixel "
            "(a peak, a streak, the beam-stop edge) is kept. Remembered between sessions."
        )
        layout.addWidget(self.bad_pixels_check)
        self.mirror_fill_check = QCheckBox("Fill gaps from the mirror side (GIWAXS)", page)
        self.mirror_fill_check.setObjectName("analyzeMirrorFillCheck")
        self.mirror_fill_check.setToolTip(
            "GIWAXS is symmetric in ±q∥: a pixel without data (module gap, mask, hot pixel) takes the value "
            "at its mirror position about the beam-centre column when that one was measured. Filled pixels "
            "are counted and recorded in the export."
        )
        layout.addWidget(self.mirror_fill_check)
        masks_title = QLabel("Masks you draw", page)
        masks_title.setProperty("gimapSectionTitle", True)
        layout.addWidget(masks_title)
        draw_row = FlowLayout()  # wraps in a narrow panel
        self.draw_rect_button = QPushButton("Rectangle", page)
        self.draw_rect_button.setObjectName("analyzeDrawRectangle")
        self.draw_rect_button.setCheckable(True)
        self.draw_rect_button.setToolTip("Click two opposite corners on the detector image (Esc cancels)")
        self.draw_polygon_button = QPushButton("Polygon", page)
        self.draw_polygon_button.setObjectName("analyzeDrawPolygon")
        self.draw_polygon_button.setCheckable(True)
        self.draw_polygon_button.setToolTip(
            "Click the corners on the detector image; double-click or Enter closes it, Backspace removes "
            "the last corner, Esc cancels"
        )
        self.mask_load_button = QPushButton("Load…", page)
        self.mask_load_button.setToolTip("Masks saved by GIMaP (.json) or a mask image of the same size (EDF, TIFF: non-zero = masked)")
        self.mask_save_button = QPushButton("Save…", page)
        self.mask_save_button.setToolTip("The drawn masks as a .json file, to reuse with other data of this set-up")
        for button in (self.draw_rect_button, self.draw_polygon_button, self.mask_load_button, self.mask_save_button):
            draw_row.addWidget(button)
        layout.addLayout(draw_row)
        self.mask_list = QListWidget(page)
        self.mask_list.setObjectName("analyzeMaskList")
        self.mask_list.setMaximumHeight(96)
        layout.addWidget(self.mask_list)
        mask_actions = QHBoxLayout()
        self.mask_remove_button = QPushButton("Remove Selected", page)
        self.mask_remove_button.setToolTip("Remove the mask selected in the list above (or press Delete in the list)")
        self.mask_clear_button = QPushButton("Clear Masks", page)
        self.mask_clear_button.setToolTip("Remove every drawn or loaded mask (the detector gaps stay masked)")
        mask_actions.addWidget(self.mask_remove_button)
        mask_actions.addWidget(self.mask_clear_button)
        mask_actions.addStretch(1)
        layout.addLayout(mask_actions)
        self.show_mask_button = QPushButton("Show Masked Pixels on the Image", page)
        self.show_mask_button.setObjectName("analyzeShowMaskButton")
        self.show_mask_button.setToolTip("Circle every pixel left out (gaps, hot and dead pixels, your masks) on the image")
        self.show_mask_button.setCheckable(True)
        layout.addWidget(self.show_mask_button, 0, Qt.AlignLeft)
        self.corrections_section.set_expanded(False)
        layout.addWidget(self.corrections_section)
        layout.addWidget(self.intensity_section)
        layout.addStretch(1)

    # -- 4 cuts ------------------------------------------------------------------------

    def _cuts_step(self, page: QWidget, layout: QVBoxLayout) -> None:
        self.auto_cuts_button = QPushButton("Auto Cuts", page)
        self.auto_cuts_button.setToolTip("Return to the automatic cut positions")
        layout.addWidget(self.auto_cuts_button, 0, Qt.AlignLeft)
        self.gisaxs_cuts = QWidget(page)
        gisaxs = QVBoxLayout(self.gisaxs_cuts)
        gisaxs.setContentsMargins(0, 0, 0, 0)
        gisaxs.setSpacing(6)
        card, self.cuts_info_label = info_card(self.gisaxs_cuts)
        self.cuts_info_label.setObjectName("analyzeCutsInfo")
        gisaxs.addWidget(card)
        self.symmetry_button = QPushButton("Make Left and Right Symmetric", self.gisaxs_cuts)
        self.symmetry_button.setObjectName("analyzeSymmetryButton")
        self.symmetry_button.setToolTip(
            "Move the centre column to the symmetry axis of the horizontal cut (qy = 0); the "
            "direct beam is usually hidden by the beam stop"
        )
        gisaxs.addWidget(self.symmetry_button, 0, Qt.AlignLeft)
        gisaxs.addWidget(muted("Halves of the horizontal cut", self.gisaxs_cuts))
        self.halves_combo = QComboBox(self.gisaxs_cuts)
        self.halves_combo.setObjectName("analyzeHalvesCombo")
        for key, title in FIT_SIDE_ITEMS:
            self.halves_combo.addItem(title, key)
        self.halves_combo.setToolTip("Which half of I(qy) the curve for Fitting uses (also in Send to Fitting ▾)")
        gisaxs.addWidget(self.halves_combo)
        gisaxs.addWidget(muted(
            "Drag the orange band on the image to move the horizontal cut, or double-click a column to "
            "move the vertical cut.", self.gisaxs_cuts,
        ))
        layout.addWidget(self.gisaxs_cuts)
        layout.addWidget(self.setup_regions_panel(page))
        layout.addWidget(self.giwaxs_section)
        layout.addStretch(1)

    # -- 5 results ---------------------------------------------------------------------

    def _results_step(self, page: QWidget, layout: QVBoxLayout) -> None:
        self.results_empty = EmptyState(
            "No results yet",
            "Run Automatic Analysis for peaks, orientation and sizes (GIWAXS), or the Yoneda cut, "
            "its halves, the spacing and a model fit (GISAXS).",
            page,
        )
        layout.addWidget(self.results_empty)
        self.results_host = QVBoxLayout()
        self.results_host.setSpacing(8)
        layout.addLayout(self.results_host, 0)  # panels at the top, the space left below them
        layout.addStretch(1)

    # -- 6 export ----------------------------------------------------------------------

    def _export_step(self, page: QWidget, layout: QVBoxLayout) -> None:
        self.export_curves_button = QPushButton("Export Curves…", page)
        self.export_curves_button.setObjectName("analyzeExportCurvesButton")
        self.export_image_button = QPushButton("Save Image…", page)
        self.export_map_button = QPushButton("Save q-Map Data…", page)
        self.export_map_button.setObjectName("analyzeExportMapButton")
        self.export_plots_button = QPushButton("Save Plots…", page)
        self.export_all_button = QPushButton("Batch Export…", page)
        self.export_all_button.setObjectName("analyzeBatchExportButton")
        self.export_fit_button = QPushButton("Send to Fitting", page)
        self.export_series_button = QPushButton("Send Series to Fitting…", page)
        self.export_all_button.setProperty("gimapRole", "primary")
        for button, text, tip in (
            (self.export_all_button, "Every listed frame, to a folder you choose.",
             "Every listed frame with the current settings, to a folder you choose: a table per curve with every frame a column, and per-frame files. Settings can be saved for the next data set."),
            (self.export_curves_button, "This frame's curves as CSV, next to the data.",
             "One CSV per curve (q, I, σ, pixels) and a JSON record of the settings, next to the data (gimap_analysis/)."),
            (self.export_image_button, "The image as shown, with its colour bar.",
             "The detector image or q map as shown, with a colour bar (PNG, TIFF, SVG, PDF)."),
            (self.export_map_button, "The q map as a table.",
             "The intensity on a regular q grid (qy–qz or q∥–qz) as a CSV table with its axes."),
            (self.export_plots_button, "The two plots as figures.", "The upper and lower plots as figures, one column wide."),
            (self.export_fit_button, "The cut, fitted in Fitting.", "The horizontal cut (GISAXS) or I(q) (GIWAXS) opened in Fitting."),
            (self.export_series_button, "Every frame, fitted in Fitting ▸ In-situ series.",
             "Every listed frame exported and opened in Fitting ▸ In-situ series."),
        ):
            layout.addWidget(action_row(button, text, page, tip))
        self.export_extra_host = QVBoxLayout()
        layout.addLayout(self.export_extra_host)
        layout.addStretch(1)


__all__ = ["AnalyzeStepsView", "FIT_SIDE_ITEMS", "STEPS"]
