"""Layout of the indexed-cell time series."""

from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from src.gimap.app.presentation.components import JobStatus
from src.gimap.app.presentation.components.curve_plot import CurvePlot
from src.gimap.app.presentation.i18n import DATA_VALUES


class CellSeriesView:
    def setup_ui(self):
        self.setWindowTitle("Indexed cell series")
        self.resize(670, 620)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Phase", self))
        self.phase_combo = QComboBox(self)
        self.phase_combo.setProperty(DATA_VALUES, True)
        controls.addWidget(self.phase_combo)
        controls.addWidget(QLabel("Cell parameter", self))
        self.parameter_combo = QComboBox(self)
        self.parameter_combo.setProperty(DATA_VALUES, True)
        controls.addWidget(self.parameter_combo)
        controls.addStretch(1)
        self.export_button = QPushButton("Export Table…", self)
        self.export_button.setEnabled(False)
        controls.addWidget(self.export_button)
        layout.addLayout(controls)
        self.source_label = QLabel(self)
        self.source_label.setWordWrap(True)
        self.source_label.setProperty("gimapRole", "muted")
        layout.addWidget(self.source_label)
        self.job = JobStatus(self)
        self.job.set_actions_visible(pause=False, details=False)
        layout.addWidget(self.job)
        self.length_plot = CurvePlot("", self, log_y=False, sides=False)
        self.volume_plot = CurvePlot("", self, log_y=False, sides=False)
        for plot in (self.length_plot, self.volume_plot):
            plot.log_check.hide()
            plot.marks_button.hide()
            layout.addWidget(plot, 1)
        self.details_label = QLabel(self)
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)
