"""Static layout of the Series tab: every frame's curve as one image, and two linked cuts of it.

The map (frames down, x across) is a detector view, so the colour scale, log and levels
are adjusted the same way as on the image. A horizontal band picks a frame (its curve is
drawn below), a vertical band a q window (its intensity against frame is drawn below);
both can be dragged. Behaviour lives in ``bindings/series.py``.

The tab stays narrow (about 380 px), so a map does not squeeze the image beside it: the controls take two
rows (the buttons wrap further), and the two plots under the map always stand one above the other, each
using the full panel width. The map's picture and the plots keep a least height; in a window too short
for them (a laptop screen) the tab scrolls instead of squeezing them to strips (``SeriesScroll``).

Every line above the map counts in such a window. What the map is (``series_title_label``) and the stages
stand above it; the how-to sentence takes one line while a map is shown (``HintLabel``); what is read or set
now and then is folded below the plots, where opening it moves neither the map nor the plots: the options of
a series run (``series_options``) and what changes between the stages. “Wide” gives the tab the room of the
image beside it; both curves grow across that width while keeping their vertical order.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import AdvancedSection, CurvePlot, DetectorView, EmptyState, FlowLayout

from .series_widgets import HintLabel, SeriesScroll, ShrinkingCombo, SideBySide, Snug, StackedPlots
from .series_fit_view import SeriesFitView

TRACE_ITEMS = (
    ("Mean intensity", "intensity"),
    ("Peak position", "position"),
    ("Peak width (2.355 σ)", "fwhm"),
    ("Peak area", "area"),
    ("Peak height", "height"),
    ("Change along the series", "change"),
)
CHANGE_EMPTY = "Change along the series: once the stages are found"
"""The trace plot as “Change along the series” before the stages are found."""
CHANGE_FEW = "Change along the series: needs a map of five frames or more"
"""… and with a map of fewer frames than stages are looked for in."""
VIEW_ITEMS = (("±", "both"), ("+", "positive"), ("−", "negative"), ("|x|", "folded"), ("mean", "mean"))
"""The halves of a signed cut the map stacks (``domain/series_map.VIEWS``), as ``CurvePlot`` writes them."""
VIEW_TIPS = (
    "Both halves, on the signed axis (as measured)",
    "Only the half with x > 0",
    "Only the half with x < 0, on |x|",
    "Both halves on |x|: every measured point of the two",
    "The mean of both halves on |x|; where one half has no points (a detector gap), the other half alone",
)
AXIS_ITEMS = (("Frame number", "frame"), ("Time (s)", "time"))
"""What the frame axis of the map and the traces count in."""
STAGES_RANGE_ITEMS = (("The whole x axis", "all"), ("The x range shown in the map", "shown"))
EACH_CUT_TEXT = "Find the cut again in every frame"
EACH_CUT_TIP = (
    "Off (the usual way): every frame of a Series map, a Batch Export and a watched folder is cut where the frame "
    "on screen is — the Yoneda rows of a GISAXS frame, the ring of I(χ) of a GIWAXS frame — so a row differs from "
    "the next by the sample, not by where it was cut. On: the cut is found in each frame anew."
)
CHANGE_TITLE = "Change along the series"
"""The trace plot's short title as “Change along the series”; the share of the change is in its y label."""
TRACE_LEGEND_OFFSET = (-10, -10)
TRACE_COMBO_CHARACTERS = 6
"""The least width of the trace combo, in characters, so the header remains usable in a narrow panel."""
SERIES_CONTROLS_WIDTH = 380
"""The Series controls need no more than this (px), so a map does not squeeze the image beside it."""
SERIES_MAP_HEIGHT = 170
"""The least height (px) of the map's picture (the image with its axes, under the toolbar): about 110 px of
image. In a window too short for it the tab scrolls instead of squeezing the map to a strip."""
SERIES_PLOTS_HEIGHT = (180, 200)
"""The legacy responsive-pair heights; Series uses the second value for each vertically stacked plot.
The picture and curves keep readable heights and scroll when the window is shorter."""
SERIES_EMPTY_TITLE = "One frame"
SERIES_EMPTY_TEXT = (
    "List several files, open a folder, or open a multi-frame NeXus series: Build Map then shows the chosen "
    "curve of every frame as intensity against frame and q."
)
SERIES_WAITING_TITLE = "No map yet"
SERIES_WAITING_TEXT = (
    "The map shows the chosen curve of every frame as intensity against frame and q. A Batch Export fills it "
    "as it goes."
)
"""The card of the Series tab while several frames are listed and no map is built (``bindings/series.py``)."""
SERIES_EMPTY_ACTION = "Open Folder…"
SERIES_WAITING_ACTION = "Build Map"
"""What the card offers: the frames of a folder while one frame is listed, the map once several are."""
SECTION_MARGINS = (8, 3, 8, 3)
"""The folded sections of the tab are slim: every line above the fold counts."""


class SeriesView(SeriesFitView):
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
        self.series_export_profile_action = menu.addAction("Selected Frame's Curve as Figure…")
        self.series_export_profile_data_action = menu.addAction("Selected Frame's Curve as Table…")
        self.series_export_trace_action = menu.addAction("Trace as Figure…")
        self.series_export_trace_data_action = menu.addAction("Trace as Table…")
        self.series_export_trace_data_action.setToolTip("CSV: the value of the trace plot for every frame")
        self.series_export_track_action = menu.addAction("Peak Table of Every Frame…")
        self.series_export_track_action.setToolTip(
            "CSV: frame, file, and the peak in the x window — centroid position, moment width (2.355 σ), area and height")
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
        self.series_wide_button = QToolButton(parent)
        self.series_wide_button.setObjectName("analyzeSeriesWide")
        self.series_wide_button.setText("Wide")
        self.series_wide_button.setCheckable(True)
        self.series_wide_button.setToolTip(
            "More room for this tab: the image beside it shrinks to its least width. The map and the plots below "
            "it fill the tab width and stay vertically stacked. Click again for the image.")
        self.series_wide_button.hide()  # shown once there is a map
        self.series_controls.hide()  # shown once several frames are listed
        host.addWidget(self.series_controls)
        self.series_info_label = HintLabel(parent)
        self.series_info_label.setObjectName("analyzeSeriesInfo")
        self.series_info_label.setProperty("gimapRole", "muted")
        # The halves of a signed cut the map stacks (shown for a signed cut only), beside the sentence.
        self.series_view_combo = QComboBox(parent)
        self.series_view_combo.setObjectName("analyzeSeriesHalves")
        for (text, key), tip in zip(VIEW_ITEMS, VIEW_TIPS):
            self.series_view_combo.addItem(text, key)
            self.series_view_combo.setItemData(self.series_view_combo.count() - 1, tip, Qt.ToolTipRole)
        self.series_view_combo.setToolTip(
            "The halves of the signed cut this map stacks: ± both as measured, + or − one half, |x| both on |x|, or "
            "their mean. The stages, the traces and the exports follow it.")
        self.series_view_combo.hide()
        info = QHBoxLayout()  # the sentence, and — beside its one line while a map is shown — the halves and “Wide”
        info.setSpacing(6)
        info.addWidget(self.series_info_label, 1)
        info.addWidget(self.series_view_combo, 0, Qt.AlignTop)
        info.addWidget(self.series_wide_button, 0, Qt.AlignTop)
        host.addLayout(info)
        # What the map shown is: its curve and halves, its frames, the cut of the series (``bindings/series_display.py``).
        self.series_title_label = QLabel("", parent)
        self.series_title_label.setObjectName("analyzeSeriesTitle")
        self.series_title_label.setWordWrap(True)
        self.series_title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.series_title_label.hide()
        host.addWidget(self.series_title_label)
        self.series_stale_label = QLabel("", parent)  # the map was built with another curve or other settings
        self.series_stale_label.setObjectName("analyzeSeriesStale")
        self.series_stale_label.setProperty("gimapRole", "warning")
        self.series_stale_label.setWordWrap(True)
        self.series_stale_label.hide()
        host.addWidget(self.series_stale_label)
        self._setup_stages(parent, host)
        self.series_empty = EmptyState(SERIES_EMPTY_TITLE, SERIES_EMPTY_TEXT, parent, action_text=SERIES_EMPTY_ACTION)
        host.addWidget(self.series_empty)
        self.series_map_view = DetectorView(parent)
        self.series_map_view.setObjectName("analyzeSeriesMap")
        self.series_map_view.set_aspect_locked(False)
        # The picture keeps its height whatever the toolbar above it needs (two or three lines in a narrow
        # panel); a tab too short for it scrolls (``SeriesScroll``).
        self.series_map_view.graphics.setMinimumHeight(SERIES_MAP_HEIGHT)
        self.series_map_view.hide()
        # … and asks for no more than that as its least, however tall it last was (``Snug``).
        view = self.series_map_view
        host.addWidget(Snug(view, parent, lambda width: view.layout().minimumHeightForWidth(width)), 3)
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
            "In the x window of the vertical band, for every frame: the mean intensity, or the peak found there "
            "above a straight background — its centroid, its moment width (2.355 × the standard deviation inside "
            "the window: the FWHM of a Gaussian peak, larger for a wider window), its area and its height. "
            "A fitted FWHM with its error: Batch Export ▸ Fit ▸ Peaks."
        )
        self.series_trace_plot.header_layout.insertWidget(0, self.series_trace_combo)
        # Both curves use the full width, including after Wide or a window resize.
        self.series_plots = StackedPlots(self.series_profile_plot, self.series_trace_plot, parent,
                                       least_height=SERIES_PLOTS_HEIGHT)
        self.series_plots.hide()
        host.addWidget(Snug(self.series_plots, parent), 2)
        self.setup_series_fit_results(parent, host)
        self._setup_series_options(parent, host)
        host.addWidget(self.series_stages_details)
        self.series_stretch_index = host.count()
        host.addStretch(1)

    def _setup_series_options(self, parent: QWidget, host: QVBoxLayout) -> None:
        """Folded under the plots: which frames a map takes, where they are cut, what its axes count in."""
        section = AdvancedSection(
            "Options: frames, cut, axes",
            "Which frames Build Map takes and where they are cut; what the frame axis counts in; what the stages "
            "compare. The cut holds for a Batch Export and a watched folder too.",
            parent,
        )
        section.setObjectName("analyzeSeriesOptions")
        section.layout().setContentsMargins(*SECTION_MARGINS)
        self.series_options = section
        frames = QHBoxLayout()
        frames.setSpacing(6)
        self.series_first_spin = QSpinBox(section)
        self.series_first_spin.setObjectName("analyzeSeriesFirst")
        self.series_first_spin.setRange(1, 10_000_000)
        self.series_first_spin.setToolTip("The first frame of the list a map takes (the frames before it are left out)")
        self.series_last_spin = QSpinBox(section)
        self.series_last_spin.setObjectName("analyzeSeriesLast")
        self.series_last_spin.setRange(0, 10_000_000)
        self.series_last_spin.setSpecialValueText("the last")
        self.series_last_spin.setToolTip("The last frame a map takes; at its least value: the last frame listed")
        for widget in (QLabel("Frames", section), self.series_first_spin, QLabel("to", section), self.series_last_spin):
            frames.addWidget(widget)
        frames.addStretch(1)
        section.add_layout(frames)
        self.series_each_cut_check = QCheckBox(EACH_CUT_TEXT, section)
        self.series_each_cut_check.setObjectName("analyzeSeriesEachCut")
        self.series_each_cut_check.setToolTip(EACH_CUT_TIP)
        section.add_widget(self.series_each_cut_check)
        self.series_axis_combo = QComboBox(section)
        self.series_axis_combo.setObjectName("analyzeSeriesAxis")
        for text, key in AXIS_ITEMS:
            self.series_axis_combo.addItem(text, key)
        self.series_axis_combo.setToolTip(
            "What the frame axis of the map and of the traces counts in: the number of the frame (in its file, "
            "or in the list), or the time since the first row when the frames carry one")
        self.series_stages_range_combo = QComboBox(section)
        self.series_stages_range_combo.setObjectName("analyzeSeriesStagesRange")
        for text, key in STAGES_RANGE_ITEMS:
            self.series_stages_range_combo.addItem(text, key)
        self.series_stages_range_combo.setToolTip(
            "What the stages and odd frames are found in: every x of the map, or only the x range the map shows "
            "(zoom the map: the stages follow)")
        for caption, combo in (("Frame axis", self.series_axis_combo), ("Stages compare", self.series_stages_range_combo)):
            row = QHBoxLayout()  # a row each: the tab stays as narrow as its controls
            row.setSpacing(6)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
            row.addWidget(QLabel(caption, section))
            row.addWidget(combo, 1)
            section.add_layout(row)
        self.series_follow_check = QCheckBox("Show the picked frame's image", section)
        self.series_follow_check.setObjectName("analyzeSeriesFollow")
        self.series_follow_check.setToolTip(
            "Picking a row of the map also shows that frame in Analyze (its image and curves); a large frame takes "
            "a moment to read. Off: Open shows it.")
        section.add_widget(self.series_follow_check)
        self.series_watch_follow_check = QCheckBox("Show the newest frame", section)
        self.series_watch_follow_check.setObjectName("analyzeSeriesWatchFollow")
        self.series_watch_follow_check.setChecked(True)
        self.series_watch_follow_check.setToolTip(
            "In the Series tab, the curve below the map follows the newest frame done; click a row of the "
            "map to look at another one")
        self.series_watch_follow_check.hide()
        section.add_widget(self.series_watch_follow_check)
        section.hide()  # shown once several frames are listed
        host.addWidget(section)

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
        section.layout().setContentsMargins(*SECTION_MARGINS)
        self.series_stages_details = section  # added under the plots: opening it moves neither map nor plots
        self.series_changes_label = QLabel("", section)  # as long as it is: the tab scrolls, nothing inside it does
        self.series_changes_label.setObjectName("analyzeSeriesChanges")
        self.series_changes_label.setWordWrap(True)
        self.series_changes_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.series_changes_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        section.add_widget(self.series_changes_label)
        self.series_skip_odd_check = QCheckBox("Leave the odd frames out of Batch Export", section)
        self.series_skip_odd_check.setObjectName("analyzeSeriesSkipOdd")
        self.series_skip_odd_check.setToolTip(
            "A Batch Export of this list then skips the frames marked odd here (the map keeps them)")
        section.add_widget(self.series_skip_odd_check)
        section.hide()


__all__ = [
    "AXIS_ITEMS", "CHANGE_EMPTY", "CHANGE_FEW", "CHANGE_TITLE", "EACH_CUT_TEXT", "EACH_CUT_TIP", "HintLabel",
    "SERIES_CONTROLS_WIDTH", "SERIES_EMPTY_ACTION", "SERIES_EMPTY_TEXT", "SERIES_EMPTY_TITLE", "SERIES_MAP_HEIGHT",
    "SERIES_PLOTS_HEIGHT", "SERIES_WAITING_ACTION", "SERIES_WAITING_TEXT", "SERIES_WAITING_TITLE",
    "STAGES_RANGE_ITEMS", "SeriesScroll", "SeriesView", "ShrinkingCombo", "SideBySide", "Snug",
    "TRACE_COMBO_CHARACTERS", "TRACE_LEGEND_OFFSET", "VIEW_ITEMS", "VIEW_TIPS",
]
