"""Static layout of the Batch Export dialog: which frames, with which settings, what to save, where.

Top to bottom: the frames and the settings they are reduced with (Load / Save Settings…); then, in a
scrolling area, four groups — **Data** (numbers: tables, per-frame curves, the Fitting input, q maps),
**Pictures**, **Converted detector frames** (the raw data in another format) and **Fitting**
(optional: peaks of the ring regions or a particle model, with its starting values and a trial on
the frame on screen) — each option with the file it writes next to it; at the bottom the folder.
Behaviour: ``batch_dialog.py``.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

DATA_OUTPUTS = (
    ("tables", "All frames in one table per curve", "x in the first column, then one column per frame — ready to plot a series"),
    ("per_frame", "Each frame: its curves, with error bars and pixel counts", "x, I, σ and the pixels behind each point; a JSON record of the settings per frame"),
    ("fit_input", "Each frame: the curve for Fitting", "q, I, σ, pixels — what Fitting ▸ In-situ series reads"),
    ("q_map", "Each frame: the q map as numbers", "Intensity on a regular q∥–qz (or qy–qz) grid, with its axes"),
    ("cake", "Each frame: the unwrapped χ–q map as numbers", "GIWAXS: intensity on a χ × q grid, with its axes"),
)
PICTURE_OUTPUTS = (
    ("detector_image", "Each frame: picture of the detector image", "With a colour bar; colour limits as chosen below"),
    ("q_map_image", "Each frame: picture of the q map", "With a colour bar; colour limits as chosen below"),
)
FRAME_OUTPUT = (
    "frames", "Each frame: the detector data in another format",
    "Values as read (modules stitched, frames summed), for other programs: a format conversion",
)
OUTPUTS = DATA_OUTPUTS + PICTURE_OUTPUTS + (FRAME_OUTPUT,)
PEAK_SHAPES = (("Pseudo-Voigt", "pseudo_voigt"), ("Gaussian", "gaussian"), ("Lorentzian", "lorentzian"))
MODELS = (
    ("Automatic (compare sphere and cylinders)", "auto"), ("Sphere", "sphere"),
    ("Random cylinder", "random_cylinder"), ("Vertical cylinder", "vertical_cylinder"),
)
STARTS = (
    ("previous", "The previous frame's result", "For a series that changes slowly: the fit follows the peak or the particles"),
    ("first", "The first frame's result, for every frame", "The same start everywhere: frames do not depend on each other"),
    ("fresh", "Found anew in each frame", "Every frame on its own, the start taken from its own data"),
)


CURVE_LIST_HEIGHT = 130
"""The most height (px) the list of curves takes; fewer rows: just their height."""
SCREEN_SHARE = 0.9
"""The dialog opens as tall as its content, at most this share of the screen's free height."""


def _muted(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setProperty("gimapRole", "muted")
    return label


def _file_label(parent: QWidget) -> QLabel:
    """The file an option writes (muted, next to it; the text can be selected and copied)."""
    label = QLabel("", parent)
    label.setProperty("gimapRole", "muted")
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def _spin(parent, low, high, decimals, step) -> QDoubleSpinBox:
    spin = QDoubleSpinBox(parent)
    spin.setRange(low, high)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    return spin


class BatchExportView:
    """Builds the widgets on ``self`` (a ``QDialog``)."""

    def setup_batch_export(self) -> None:
        self.setObjectName("analyzeBatchExportDialog")
        self.setMinimumSize(760, 640)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        self.frames_label = QLabel("", self)
        self.frames_label.setObjectName("batchFramesLabel")
        self.frames_label.setWordWrap(True)
        font = self.frames_label.font()
        font.setBold(True)
        self.frames_label.setFont(font)
        layout.addWidget(self.frames_label)
        top = QGridLayout()
        top.setHorizontalSpacing(10)
        top.addWidget(QLabel("Frames", self), 0, 0, Qt.AlignRight)
        frames = QHBoxLayout()
        frames.addWidget(QLabel("Every", self))
        self.every_spin = QSpinBox(self)
        self.every_spin.setObjectName("batchEvery")
        self.every_spin.setRange(1, 10000)
        self.every_spin.setToolTip("Take every n-th listed frame (1: all) — a long run first at a glance")
        frames.addWidget(self.every_spin)
        self.every_label = QLabel("", self)
        frames.addWidget(self.every_label)
        frames.addStretch(1)
        top.addLayout(frames, 0, 1)
        top.addWidget(QLabel("Settings", self), 1, 0, Qt.AlignRight | Qt.AlignTop)
        settings = QHBoxLayout()
        self.settings_label = _muted("", self)
        self.settings_label.setObjectName("batchSettingsLabel")
        self.load_settings_button = QPushButton("Load Settings…", self)
        self.load_settings_button.setToolTip("Use a set-up saved before: geometry, masks, corrections, cut regions")
        self.save_settings_button = QPushButton("Save Settings…", self)
        self.save_settings_button.setToolTip("Keep this set-up (and these export choices) for the next data set")
        settings.addWidget(self.settings_label, 1)
        settings.addWidget(self.load_settings_button)
        settings.addWidget(self.save_settings_button)
        top.addLayout(settings, 1, 1)
        top.addWidget(QLabel("Speed", self), 2, 0, Qt.AlignRight)
        speed = QHBoxLayout()
        self.speed_combo = QComboBox(self)
        self.speed_combo.setObjectName("batchSpeed")
        self.speed_combo.setToolTip(
            "How many frames are reduced at the same time, in separate processes at low priority. Sized to "
            "this computer's cores and free memory; it can be lowered, paused or stopped while the batch runs."
        )
        speed.addWidget(self.speed_combo)
        speed.addStretch(1)
        top.addLayout(speed, 2, 1)
        top.setColumnStretch(1, 1)
        layout.addLayout(top)

        scroll = self.options_scroll = QScrollArea(self)  # the dialog opens as tall as it (``batch_dialog.py``)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget(scroll)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 6, 0)
        body_layout.setSpacing(10)
        self.output_checks: dict[str, QCheckBox] = {}
        self.output_files: dict[str, QLabel] = {}
        # data
        self.data_group, data = self._group("Data (numbers)", body)
        self.text_format_combo = QComboBox(self.data_group)
        self.text_format_combo.setObjectName("batchTextFormat")
        data.addLayout(self._format_row("Format", self.text_format_combo, self.data_group), 0, 0, 1, 2)
        for row, item in enumerate(DATA_OUTPUTS, start=1):
            self._output_row(data, row, item, self.data_group)
        curves = QVBoxLayout()
        curves.addWidget(_muted("Curves to save (tables and per-frame curves):", self.data_group))
        self.curve_list = QListWidget(self.data_group)
        self.curve_list.setObjectName("batchCurveList")
        self.curve_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.curve_list.setMaximumHeight(CURVE_LIST_HEIGHT)  # as tall as its rows, up to this (``batch_dialog.py``)
        curves.addWidget(self.curve_list)
        curve_buttons = QHBoxLayout()
        self.all_curves_button = QPushButton("All", self.data_group)
        self.no_curves_button = QPushButton("None", self.data_group)
        curve_buttons.addWidget(self.all_curves_button)
        curve_buttons.addWidget(self.no_curves_button)
        curve_buttons.addStretch(1)
        curves.addLayout(curve_buttons)
        data.addLayout(curves, len(DATA_OUTPUTS) + 1, 0, 1, 2)
        body_layout.addWidget(self.data_group)
        # pictures
        self.pictures_group, pictures = self._group("Pictures", body)
        self.image_format_combo = QComboBox(self.pictures_group)
        self.image_format_combo.setObjectName("batchImageFormat")
        pictures.addLayout(self._format_row("Format", self.image_format_combo, self.pictures_group), 0, 0, 1, 2)
        for row, item in enumerate(PICTURE_OUTPUTS, start=1):
            self._output_row(pictures, row, item, self.pictures_group)
        scale = QHBoxLayout()
        scale.addWidget(QLabel("Colour scale", self.pictures_group))
        self.image_scale_combo = QComboBox(self.pictures_group)
        self.image_scale_combo.setObjectName("batchImageScale")
        self.image_scale_combo.setToolTip(
            "Each frame its own limits (the brightest parts always visible), or the limits on screen for every "
            "frame (the pictures can be compared: the same colour is the same intensity)"
        )
        scale.addWidget(self.image_scale_combo, 1)
        pictures.addLayout(scale, len(PICTURE_OUTPUTS) + 1, 0, 1, 2)
        self.image_scale_label = _muted("", self.pictures_group)
        self.image_scale_label.setObjectName("batchImageScaleText")
        pictures.addWidget(self.image_scale_label, len(PICTURE_OUTPUTS) + 2, 0, 1, 2)
        body_layout.addWidget(self.pictures_group)
        # converted frames
        self.frames_group, converted = self._group("Converted detector frames", body)
        self.frame_format_combo = QComboBox(self.frames_group)
        self.frame_format_combo.setObjectName("batchFrameFormat")
        converted.addLayout(self._format_row("Format", self.frame_format_combo, self.frames_group), 0, 0, 1, 2)
        self._output_row(converted, 1, FRAME_OUTPUT, self.frames_group)
        body_layout.addWidget(self.frames_group)
        # fitting
        self.fit_group, fitting = self._group("Fitting (optional)", body)
        choice = QHBoxLayout()
        choice.setSpacing(18)
        self.fit_none_radio = QRadioButton("No fitting", self.fit_group)
        self.fit_peaks_radio = QRadioButton("Peaks of the ring regions", self.fit_group)
        self.fit_peaks_radio.setToolTip(
            "GIWAXS: every region with a q window (Cuts ▸ Ring or Spot) is one peak, fitted in its I(q)"
        )
        self.fit_model_radio = QRadioButton("Particle model of the horizontal cut", self.fit_group)
        self.fit_model_radio.setToolTip(
            "GISAXS: sphere or cylinders with size dispersity and a spacing D, fitted to the curve for Fitting"
        )
        self.fit_kind_group = QButtonGroup(self.fit_group)
        for button in (self.fit_none_radio, self.fit_peaks_radio, self.fit_model_radio):
            self.fit_kind_group.addButton(button)
            choice.addWidget(button)
        choice.addStretch(1)
        fitting.addLayout(choice, 0, 0, 1, 2)
        self.fit_details = QWidget(self.fit_group)
        details = QGridLayout(self.fit_details)
        details.setContentsMargins(20, 0, 0, 0)
        details.setHorizontalSpacing(10)
        self.fit_targets_label = _muted("", self.fit_details)
        self.fit_targets_label.setObjectName("batchFitTargets")
        details.addWidget(self.fit_targets_label, 0, 0, 1, 2)
        self.fit_shape_caption = QLabel("Peak shape", self.fit_details)
        self.fit_profile_combo = QComboBox(self.fit_details)
        self.fit_profile_combo.setObjectName("batchFitProfile")
        for text, key in PEAK_SHAPES:
            self.fit_profile_combo.addItem(text, key)
        self.fit_profile_combo.setToolTip("Fitted on a straight background, in the region's q window")
        self.fit_model_caption = QLabel("Model", self.fit_details)
        self.fit_model_combo = QComboBox(self.fit_details)
        self.fit_model_combo.setObjectName("batchFitModel")
        for text, key in MODELS:
            self.fit_model_combo.addItem(text, key)
        details.addWidget(self.fit_shape_caption, 1, 0, Qt.AlignRight)
        details.addWidget(self.fit_profile_combo, 1, 1, Qt.AlignLeft)
        details.addWidget(self.fit_model_caption, 2, 0, Qt.AlignRight)
        details.addWidget(self.fit_model_combo, 2, 1, Qt.AlignLeft)
        details.addWidget(QLabel("Start values", self.fit_details), 3, 0, Qt.AlignRight | Qt.AlignTop)
        starts = QVBoxLayout()
        self.fit_start_radios: dict[str, QRadioButton] = {}
        self.fit_start_group = QButtonGroup(self.fit_details)
        for key, text, tip in STARTS:
            radio = QRadioButton(text, self.fit_details)
            radio.setToolTip(tip)
            self.fit_start_group.addButton(radio)
            self.fit_start_radios[key] = radio
            starts.addWidget(radio)
        details.addLayout(starts, 3, 1)
        self.fit_curves_check = QCheckBox("Also save the fitted curve of each frame", self.fit_details)
        self.fit_curves_check.setObjectName("batchFitCurves")
        self.fit_curves_file = _file_label(self.fit_details)
        details.addWidget(self.fit_curves_check, 4, 0, 1, 2)
        details.addWidget(self.fit_curves_file, 5, 0, 1, 2)
        self.fit_try_button = QPushButton("Try on This Frame", self.fit_details)
        self.fit_try_button.setObjectName("batchFitTry")
        self.fit_try_button.setToolTip("Fit the frame on screen with these choices, to check them before the batch")
        details.addWidget(self.fit_try_button, 6, 0, Qt.AlignLeft | Qt.AlignTop)
        self.fit_try_label = QLabel("", self.fit_details)
        self.fit_try_label.setObjectName("batchFitTryResult")
        self.fit_try_label.setWordWrap(True)
        self.fit_try_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        details.addWidget(self.fit_try_label, 6, 1)
        self.fit_file_label = _file_label(self.fit_details)
        self.fit_file_label.setObjectName("batchFitFile")
        details.addWidget(self.fit_file_label, 7, 0, 1, 2)
        details.setColumnStretch(1, 1)
        fitting.addWidget(self.fit_details, 1, 0, 1, 2)
        body_layout.addWidget(self.fit_group)
        # in-situ corrections, folded away
        self.series_toggle = QToolButton(body)
        self.series_toggle.setObjectName("batchSeriesToggle")
        self.series_toggle.setText("In-situ corrections (reference peak)")
        self.series_toggle.setCheckable(True)
        self.series_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.series_toggle.setArrowType(Qt.RightArrow)
        self.series_toggle.setAutoRaise(True)
        body_layout.addWidget(self.series_toggle)
        self.series_box = QFrame(body)
        self.series_box.setObjectName("batchSeriesBox")
        box = QVBoxLayout(self.series_box)
        box.setContentsMargins(20, 0, 0, 0)
        box.addWidget(_muted(
            "A peak that does not change during the run (substrate, internal standard) corrects the drift "
            "of sample height and incident flux (GIWAXS).", self.series_box,
        ))
        grid = QGridLayout()
        self.reference_q_spin = _spin(self.series_box, 0.01, 50.0, 4, 0.01)
        self.reference_q_spin.setSuffix(" Å⁻¹")
        self.half_width_spin = _spin(self.series_box, 0.001, 5.0, 4, 0.005)
        self.half_width_spin.setSuffix(" Å⁻¹")
        grid.addWidget(QLabel("Reference peak q", self.series_box), 0, 0)
        grid.addWidget(self.reference_q_spin, 0, 1)
        grid.addWidget(QLabel("Search ±", self.series_box), 0, 2)
        grid.addWidget(self.half_width_spin, 0, 3)
        box.addLayout(grid)
        self.align_check = QCheckBox("Align the detector distance so the peak sits at this q", self.series_box)
        self.normalize_check = QCheckBox("Normalise the intensity to the peak height", self.series_box)
        box.addWidget(self.align_check)
        box.addWidget(self.normalize_check)
        normalize = QHBoxLayout()
        normalize.setContentsMargins(20, 0, 0, 0)
        self.target_spin = _spin(self.series_box, 1e-12, 1e12, 4, 0.1)
        normalize.addWidget(QLabel("Peak height after", self.series_box))
        normalize.addWidget(self.target_spin)
        normalize.addStretch(1)
        box.addLayout(normalize)
        self.first_frame_radio = QRadioButton("One factor for the whole series (from the first frame)", self.series_box)
        self.per_frame_radio = QRadioButton("Every frame to its own peak", self.series_box)
        box.addWidget(self.first_frame_radio)
        box.addWidget(self.per_frame_radio)
        self.series_box.hide()
        body_layout.addWidget(self.series_box)
        body_layout.addStretch(1)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        # destination
        bottom = QGridLayout()
        bottom.setHorizontalSpacing(10)
        bottom.addWidget(QLabel("Save to", self), 0, 0, Qt.AlignRight)
        row = QHBoxLayout()
        self.destination_edit = QLineEdit(self)
        self.destination_edit.setObjectName("batchDestination")
        self.browse_button = QPushButton("Browse…", self)
        row.addWidget(self.destination_edit, 1)
        row.addWidget(self.browse_button)
        bottom.addLayout(row, 0, 1)
        self.subfolder_check = QCheckBox("", self)
        self.subfolder_check.setObjectName("batchSubfolder")
        bottom.addWidget(self.subfolder_check, 1, 1)
        self.target_label = _muted("", self)
        self.target_label.setObjectName("batchTarget")
        self.target_label.setWordWrap(False)
        bottom.addWidget(self.target_label, 2, 1)
        bottom.setColumnStretch(1, 1)
        layout.addLayout(bottom)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel, self)
        self.export_button = self.buttons.addButton("Export", QDialogButtonBox.AcceptRole)
        self.export_button.setObjectName("batchExportButton")
        self.export_button.setProperty("gimapRole", "primary")
        layout.addWidget(self.buttons)

    def _group(self, title: str, parent: QWidget) -> tuple[QGroupBox, QGridLayout]:
        group = QGroupBox(title, parent)
        grid = QGridLayout(group)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 1)
        return group, grid

    def _format_row(self, caption: str, combo: QComboBox, parent: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(_muted(caption, parent))
        row.addWidget(combo)
        row.addStretch(1)
        return row

    def _output_row(self, grid: QGridLayout, row: int, item, parent: QWidget) -> None:
        key, text, tip = item
        check = QCheckBox(text, parent)
        check.setObjectName(f"batchOutput_{key}")
        check.setToolTip(tip)
        name = _file_label(parent)
        name.setObjectName(f"batchFile_{key}")
        grid.addWidget(check, row, 0)
        grid.addWidget(name, row, 1)
        self.output_checks[key] = check
        self.output_files[key] = name


__all__ = [
    "BatchExportView", "CURVE_LIST_HEIGHT", "DATA_OUTPUTS", "FRAME_OUTPUT", "MODELS", "OUTPUTS", "PEAK_SHAPES",
    "PICTURE_OUTPUTS", "SCREEN_SHARE", "STARTS",
]
