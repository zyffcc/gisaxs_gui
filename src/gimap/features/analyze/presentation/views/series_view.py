"""Static layout of the Series tab: every frame's curve as one image, and two linked cuts of it.

The map (frames down, x across) is a detector view, so the colour scale, log and levels
are adjusted the same way as on the image. A horizontal band picks a frame (its curve is
drawn below), a vertical band a q window (its intensity against frame is drawn below);
both can be dragged. Behaviour lives in ``bindings/series.py``.

The tab stays narrow (about 380 px), so a map does not squeeze the image beside it: the controls take two
rows (the buttons wrap further), and the two plots under the map stand one above the other when the panel
has no room for them side by side.
"""

from __future__ import annotations

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMenu,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import AdvancedSection, CurvePlot, DetectorView, EmptyState, FlowLayout

TRACE_ITEMS = (
    ("Mean intensity", "intensity"),
    ("Peak position", "position"),
    ("Peak FWHM", "fwhm"),
    ("Peak area", "area"),
    ("Peak height", "height"),
    ("Change along the series", "change"),
)
CHANGE_EMPTY = "Change along the series: once the stages are found"
"""The trace plot as “Change along the series” before the stages are found."""
CHANGE_FEW = "Change along the series: needs a map of three frames or more"
"""… and with a map of two frames, where no stages are looked for."""
CHANGE_TITLE = "Change along the series"
"""The trace plot's short title as “Change along the series”; the share of the change is in its y label."""
TRACE_LEGEND_OFFSET = (-10, -10)
TRACE_COMBO_CHARACTERS = 6
"""The least width of the trace combo, in characters: with it, the frame and trace plots stand side by side from
a right panel of about 480 px (a 1366 px window) instead of 550."""
SERIES_CONTROLS_WIDTH = 380
"""The Series controls need no more than this (px), so a map does not squeeze the image beside it."""
SERIES_PLOTS_HEIGHT = (180, 96)
"""The least height (px) of the frame and trace plots side by side, and of each when one is above the other."""
SERIES_EMPTY_TITLE = "One frame"
SERIES_EMPTY_TEXT = (
    "List several files, open a folder, or open a multi-frame NeXus series: Build Map then shows the chosen "
    "curve of every frame as intensity against frame and q."
)


class ShrinkingCombo(QComboBox):
    """A combo as wide as its longest item while there is room, down to ``least_characters`` when there is not
    (``minimumSizeHint``), its text then shortened with “…”; its list shows the items whole."""

    def __init__(self, parent: QWidget, *, least_characters: int):
        super().__init__(parent)
        self._least_characters = int(least_characters)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        full = super().minimumSizeHint()
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        text = QSize(self._least_characters * self.fontMetrics().horizontalAdvance("x"), full.height())
        least = self.style().sizeFromContents(QStyle.CT_ComboBox, option, text, self)
        return QSize(min(full.width(), least.width()), full.height())

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        field = self.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, self)
        option.currentText = self.fontMetrics().elidedText(option.currentText, Qt.ElideRight, max(0, field.width()))
        painter.drawComplexControl(QStyle.CC_ComboBox, option)
        painter.drawControl(QStyle.CE_ComboBoxLabel, option)


class SideBySide(QWidget):
    """Two widgets side by side while both get their least width, else one above the other, so a narrow panel
    keeps both whole (the frame and trace plots of the Series tab). Its least width is the wider one's."""

    def __init__(self, first: QWidget, second: QWidget, parent: QWidget, *, spacing: int = 6,
                 least_height: tuple[int, int] = (0, 0)):
        super().__init__(parent)
        self._pair = (first, second)
        self._least_height = tuple(int(value) for value in least_height)
        """``(side by side, each when one is above the other)``: the least heights (px)."""
        self._box = QBoxLayout(QBoxLayout.LeftToRight, self)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(spacing)
        self._box.setSizeConstraint(QLayout.SetNoConstraint)  # the least width is ``minimumSizeHint``'s
        for widget in self._pair:
            self._box.addWidget(widget, 1)

    def stacked(self) -> bool:
        return self._box.direction() == QBoxLayout.TopToBottom

    def side_by_side_width(self) -> int:
        return sum(widget.minimumSizeHint().width() for widget in self._pair) + self._box.spacing()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        hints = [widget.minimumSizeHint() for widget in self._pair]
        side_by_side, stacked = self._least_height
        if self.stacked():
            height = sum(max(hint.height(), stacked) for hint in hints) + self._box.spacing()
        else:
            height = max(side_by_side, *(hint.height() for hint in hints))
        return QSize(max(hint.width() for hint in hints), height)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        """As tall as side by side, also when one is above the other: the map above keeps its share of the
        height, and the two plots share theirs."""
        hints = [widget.sizeHint() for widget in self._pair]
        width = sum(hint.width() for hint in hints) + self._box.spacing()
        return QSize(width, max(self.minimumSizeHint().height(), *(hint.height() for hint in hints)))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        stacked = event.size().width() < self.side_by_side_width()
        if stacked != self.stacked():
            self._box.setDirection(QBoxLayout.TopToBottom if stacked else QBoxLayout.LeftToRight)
            self.updateGeometry()


class SeriesView:
    """Adds the Series tab widgets to ``host`` (a vertical layout on ``parent``)."""

    def setup_series_panel(self, parent: QWidget, host: QVBoxLayout) -> None:
        # Two rows, so the panel stays narrow (``SERIES_CONTROLS_WIDTH``): what the map stacks, then what to do
        # with it (the buttons wrap onto a further line when the panel is narrower still).
        self.series_controls = QWidget(parent)
        self.series_controls.setObjectName("analyzeSeriesControls")
        rows = QVBoxLayout(self.series_controls)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(6)
        controls = QHBoxLayout()
        controls.setSpacing(6)
        actions = FlowLayout(spacing=6)
        rows.addLayout(controls)
        rows.addLayout(actions)
        caption = QLabel("Curve", parent)
        caption.setProperty("gimapRole", "muted")
        self.series_curve_combo = QComboBox(parent)
        self.series_curve_combo.setObjectName("analyzeSeriesCurve")
        self.series_curve_combo.setToolTip("Which curve of every frame the map stacks")
        # As wide as the row allows, at least 12 characters; the list shows the names whole (``bindings/series.py``).
        self.series_curve_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.series_curve_combo.setMinimumContentsLength(12)
        self.series_build_button = QPushButton("Build Map", parent)
        self.series_build_button.setObjectName("analyzeSeriesBuild")
        self.series_build_button.setProperty("gimapRole", "primary")
        self.series_build_button.setToolTip(
            "Reduce every listed frame (or group of summed frames) with the current settings and stack the curve"
        )
        self.series_export_button = QToolButton(parent)
        self.series_export_button.setObjectName("analyzeSeriesExport")
        self.series_export_button.setText("Export")
        self.series_export_button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.series_export_button)
        self.series_export_csv_action = menu.addAction("Map as CSV Table…")
        self.series_export_figure_action = menu.addAction("Map as Figure…")
        self.series_export_stages_action = menu.addAction("Stages as Table…")
        self.series_export_stages_action.setToolTip(
            "CSV: every frame's stage, whether it is odd and why, and its place along the main changes; a JSON record next to it")
        menu.addSeparator()
        self.series_export_profile_action = menu.addAction("Selected Frame's Curve…")
        self.series_export_trace_action = menu.addAction("Intensity against Frame…")
        self.series_export_track_action = menu.addAction("Peak Table of Every Frame…")
        self.series_export_track_action.setToolTip("CSV: frame, file, peak position, FWHM, area and height in the q window")
        self.series_export_button.setMenu(menu)
        self.series_export_button.setToolTip("Save the map, a frame's curve, a trace or the peak table (once a map is built)")
        self.series_export_button.hide()  # shown once there is a map to export
        self.series_step_spin = QSpinBox(parent)
        self.series_step_spin.setObjectName("analyzeSeriesStep")
        self.series_step_spin.setRange(1, 1000)
        self.series_step_spin.setPrefix("every ")
        self.series_step_spin.setSuffix(" frame")
        self.series_step_spin.setToolTip(
            "Use every n-th frame (or group of summed frames): a quick first look at a long series"
        )
        controls.addWidget(caption)
        controls.addWidget(self.series_curve_combo, 1)
        controls.addWidget(self.series_step_spin)
        self.series_batch_button = QPushButton("Batch Export…", parent)
        self.series_batch_button.setObjectName("analyzeSeriesBatch")
        self.series_batch_button.setToolTip(
            "Every frame's curves to a folder: a table per curve with every frame as a column, and per-frame files"
        )
        actions.addWidget(self.series_build_button)
        actions.addWidget(self.series_export_button)
        self.series_compare_button = QPushButton("Send to Compare", parent)
        self.series_compare_button.setObjectName("analyzeSeriesCompare")
        self.series_compare_button.setToolTip(
            "Add this series (every frame's curve) to Compare, to set it beside other samples or series")
        self.series_compare_button.hide()  # shown once there is a map
        actions.addWidget(self.series_compare_button)
        actions.addWidget(self.series_batch_button)
        self.series_controls.hide()  # shown once several frames are listed
        host.addWidget(self.series_controls)
        self.series_info_label = QLabel("", parent)
        self.series_info_label.setObjectName("analyzeSeriesInfo")
        self.series_info_label.setProperty("gimapRole", "muted")
        self.series_info_label.setWordWrap(True)
        host.addWidget(self.series_info_label)
        self._setup_stages(parent, host)
        self.series_empty = EmptyState(SERIES_EMPTY_TITLE, SERIES_EMPTY_TEXT, parent)
        host.addWidget(self.series_empty)
        self.series_map_view = DetectorView(parent)
        self.series_map_view.setObjectName("analyzeSeriesMap")
        self.series_map_view.set_aspect_locked(False)
        self.series_map_view.hide()
        host.addWidget(self.series_map_view, 3)
        self.series_profile_plot = CurvePlot("", parent)
        self.series_profile_plot.setObjectName("analyzeSeriesProfile")
        self.series_open_button = QPushButton("Open", parent)
        self.series_open_button.setObjectName("analyzeSeriesOpen")
        self.series_open_button.setToolTip("Show this frame in Analyze (its image and all its curves)")
        self.series_profile_plot.header_layout.addWidget(self.series_open_button)
        self.series_trace_plot = CurvePlot("", parent, log_y=False)
        self.series_trace_plot.setObjectName("analyzeSeriesTrace")
        self.series_trace_plot.set_empty_text(CHANGE_EMPTY)  # only that kind: ``bindings/series.py`` clears it else
        # Bottom right: the traces (the change of every stage) rise to the top right, under a legend there.
        self.series_trace_plot.legend.setOffset(TRACE_LEGEND_OFFSET)
        # Whole when there is room, narrower when the plots would otherwise stand one above the other.
        self.series_trace_combo = ShrinkingCombo(parent, least_characters=TRACE_COMBO_CHARACTERS)
        self.series_trace_combo.setObjectName("analyzeSeriesTraceKind")
        for text, key in TRACE_ITEMS:
            self.series_trace_combo.addItem(text, key)
        popup = self.series_trace_combo.view()  # narrower than an item: its list still shows them whole
        popup.setMinimumWidth(popup.sizeHintForColumn(0) + 2 * popup.frameWidth() + 24)
        self.series_trace_combo.setToolTip(
            "In the q window of the vertical band, for every frame: the mean intensity, or the peak found there "
            "(centroid position, FWHM, area and height above a straight background)"
        )
        self.series_trace_plot.header_layout.insertWidget(0, self.series_trace_combo)
        # The frame's curve and the trace side by side, or one above the other in a narrow panel.
        self.series_plots = SideBySide(self.series_profile_plot, self.series_trace_plot, parent,
                                       least_height=SERIES_PLOTS_HEIGHT)
        self.series_plots.hide()
        host.addWidget(self.series_plots, 2)
        self.series_stretch_index = host.count()
        host.addStretch(1)

    def _setup_stages(self, parent: QWidget, host: QVBoxLayout) -> None:
        """Under the controls: how many stages, where they are; folded: what changes and the odd frames."""
        self.series_stages_row = QWidget(parent)
        self.series_stages_row.setObjectName("analyzeSeriesStages")
        row = QHBoxLayout(self.series_stages_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        caption = QLabel("Stages", parent)
        caption.setProperty("gimapRole", "muted")
        self.series_stages_combo = QComboBox(parent)
        self.series_stages_combo.setObjectName("analyzeSeriesStagesCount")
        self.series_stages_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)  # “Auto (4)” whole, in any language
        self.series_stages_combo.setToolTip(
            "How many stages the series is cut into. Auto: a stage is added while it explains at least 5 % of the "
            "change and more than noise would")
        self.series_stages_label = QLabel("", parent)
        self.series_stages_label.setObjectName("analyzeSeriesStagesText")
        self.series_stages_label.setWordWrap(True)
        self.series_stages_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(caption)
        row.addWidget(self.series_stages_combo)
        row.addWidget(self.series_stages_label, 1)
        self.series_stages_row.hide()  # shown once a map has stages
        host.addWidget(self.series_stages_row)
        section = AdvancedSection(
            "What changes, and the odd frames",
            "Where the curves change course; a stage describes the series, it is not a phase by itself — what "
            "grows or falls between stages says whether the structure changed.",
            parent,
        )
        section.setObjectName("analyzeSeriesStagesDetails")
        self.series_stages_details = section
        scroll = QScrollArea(section)  # a long list of odd frames must not squeeze the map
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setMaximumHeight(170)
        self.series_changes_label = QLabel("", scroll)
        self.series_changes_label.setObjectName("analyzeSeriesChanges")
        self.series_changes_label.setWordWrap(True)
        self.series_changes_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.series_changes_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(self.series_changes_label)
        section.add_widget(scroll)
        self.series_skip_odd_check = QCheckBox("Leave the odd frames out of Batch Export", section)
        self.series_skip_odd_check.setObjectName("analyzeSeriesSkipOdd")
        self.series_skip_odd_check.setToolTip(
            "A Batch Export of this list then skips the frames marked odd here (the map keeps them)")
        section.add_widget(self.series_skip_odd_check)
        section.hide()
        host.addWidget(section)


__all__ = [
    "CHANGE_EMPTY", "CHANGE_FEW", "CHANGE_TITLE", "SERIES_CONTROLS_WIDTH", "SERIES_EMPTY_TEXT", "SERIES_EMPTY_TITLE",
    "SERIES_PLOTS_HEIGHT", "SeriesView", "ShrinkingCombo", "SideBySide", "TRACE_COMBO_CHARACTERS", "TRACE_LEGEND_OFFSET",
]
