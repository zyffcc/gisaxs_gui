"""The results of Batch Export fits, below the map and its two vertically stacked curves."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QGroupBox, QHBoxLayout, QLabel, QTableWidget, QTabWidget, QVBoxLayout, QWidget,
)

from src.gimap.app.presentation.components import CurvePlot
from src.gimap.app.presentation.components.table_copy import enable_table_copy
from src.gimap.app.presentation.i18n import DATA_HEADERS


class SeriesFitView:
    def setup_series_fit_results(self, parent: QWidget, host: QVBoxLayout) -> None:
        self.series_fit_results = QGroupBox("Results: batch fitting", parent)
        self.series_fit_results.setObjectName("analyzeSeriesFitResults")
        box = QVBoxLayout(self.series_fit_results)
        box.setContentsMargins(8, 8, 8, 8)
        self.series_fit_summary = QLabel("", self.series_fit_results)
        self.series_fit_summary.setWordWrap(True)
        self.series_fit_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box.addWidget(self.series_fit_summary)
        self.series_fit_table = QTableWidget(self.series_fit_results)
        self.series_fit_table.setObjectName("analyzeSeriesFitTable")
        self.series_fit_table.setProperty(DATA_HEADERS, True)
        self.series_fit_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.series_fit_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.series_fit_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.series_fit_table.verticalHeader().hide()
        self.series_fit_table.setMinimumHeight(140)
        self.series_fit_table.setMaximumHeight(220)
        enable_table_copy(self.series_fit_table)
        box.addWidget(self.series_fit_table)
        self.series_fit_note = QLabel("", self.series_fit_results)
        self.series_fit_note.setWordWrap(True)
        self.series_fit_note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box.addWidget(self.series_fit_note)
        self.series_fit_tabs = QTabWidget(self.series_fit_results)
        self.series_fit_plot = CurvePlot("", self.series_fit_tabs, log_x=False, sides=False)
        self.series_fit_plot.setObjectName("analyzeSeriesFitCurve")
        self.series_fit_plot.setMinimumHeight(230)
        self.series_fit_tabs.addTab(self.series_fit_plot, "Fit curve")
        trend = QWidget(self.series_fit_tabs)
        layout = QVBoxLayout(trend)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Parameter", trend))
        self.series_fit_parameter = QComboBox(trend)
        self.series_fit_parameter.setObjectName("analyzeSeriesFitParameter")
        self.series_fit_parameter.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.series_fit_parameter.setMinimumContentsLength(12)
        row.addWidget(self.series_fit_parameter, 1)
        layout.addLayout(row)
        self.series_fit_trend_plot = CurvePlot("", trend, log_y=False, sides=False)
        self.series_fit_trend_plot.setObjectName("analyzeSeriesFitTrend")
        self.series_fit_trend_plot.setMinimumHeight(210)
        layout.addWidget(self.series_fit_trend_plot)
        self.series_fit_tabs.addTab(trend, "Parameter trend")
        box.addWidget(self.series_fit_tabs)
        self.series_fit_results.hide()
        host.addWidget(self.series_fit_results)


__all__ = ["SeriesFitView"]
