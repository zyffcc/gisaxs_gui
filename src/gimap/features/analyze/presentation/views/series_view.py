"""Static layout of the Series tab: every frame's curve as one image, and two linked cuts of it.

The map (frames down, x across) is a detector view, so the colour scale, log and levels
are adjusted the same way as on the image. A horizontal band picks a frame (its curve is
drawn below), a vertical band a q window (its intensity against frame is drawn below);
both can be dragged. Behaviour lives in ``bindings/series.py``.
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
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

from src.gimap.app.presentation.components import CurvePlot, DetectorView, EmptyState

TRACE_ITEMS = (
    ("Mean intensity", "intensity"),
    ("Peak position", "position"),
    ("Peak FWHM", "fwhm"),
    ("Peak area", "area"),
    ("Peak height", "height"),
)
SERIES_EMPTY_TITLE = "One frame"
SERIES_EMPTY_TEXT = (
    "List several files, open a folder, or open a multi-frame NeXus series: Build Map then shows the chosen "
    "curve of every frame as intensity against frame and q."
)


class SeriesView:
    """Adds the Series tab widgets to ``host`` (a vertical layout on ``parent``)."""

    def setup_series_panel(self, parent: QWidget, host: QVBoxLayout) -> None:
        self.series_controls = QWidget(parent)
        self.series_controls.setObjectName("analyzeSeriesControls")
        controls = QHBoxLayout(self.series_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        caption = QLabel("Curve", parent)
        caption.setProperty("gimapRole", "muted")
        self.series_curve_combo = QComboBox(parent)
        self.series_curve_combo.setObjectName("analyzeSeriesCurve")
        self.series_curve_combo.setToolTip("Which curve of every frame the map stacks")
        self.series_curve_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.series_curve_combo.setMinimumContentsLength(16)
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
        controls.addWidget(self.series_build_button)
        controls.addWidget(self.series_export_button)
        controls.addWidget(self.series_batch_button)
        self.series_controls.hide()  # shown once several frames are listed
        host.addWidget(self.series_controls)
        self.series_info_label = QLabel("", parent)
        self.series_info_label.setObjectName("analyzeSeriesInfo")
        self.series_info_label.setProperty("gimapRole", "muted")
        self.series_info_label.setWordWrap(True)
        host.addWidget(self.series_info_label)
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
        self.series_trace_combo = QComboBox(parent)
        self.series_trace_combo.setObjectName("analyzeSeriesTraceKind")
        for text, key in TRACE_ITEMS:
            self.series_trace_combo.addItem(text, key)
        self.series_trace_combo.setToolTip(
            "In the q window of the vertical band, for every frame: the mean intensity, or the peak found there "
            "(centroid position, FWHM, area and height above a straight background)"
        )
        self.series_trace_plot.header_layout.insertWidget(0, self.series_trace_combo)
        self.series_plots = QWidget(parent)
        plots = QHBoxLayout(self.series_plots)
        plots.setContentsMargins(0, 0, 0, 0)
        plots.setSpacing(6)
        plots.addWidget(self.series_profile_plot, 1)
        plots.addWidget(self.series_trace_plot, 1)
        self.series_plots.setMinimumHeight(180)
        self.series_plots.hide()
        host.addWidget(self.series_plots, 2)
        self.series_stretch_index = host.count()
        host.addStretch(1)


__all__ = ["SERIES_EMPTY_TEXT", "SERIES_EMPTY_TITLE", "SeriesView"]
