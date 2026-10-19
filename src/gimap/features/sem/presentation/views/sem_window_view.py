"""Static layout for SEM particle analysis; behavior lives in the window."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget,
    QSizePolicy, QVBoxLayout, QWidget,
)

from src.gimap.app.presentation.components import (
    AdvancedSection, JobStatus, ParameterSection, PlotPanel, ResultTable,
)
from src.gimap.app.presentation.components.table_copy import enable_table_copy
from src.gimap.app.presentation.i18n import DATA_HEADERS, DATA_VALUES

from .sem_instance_view import SemInstanceView


def _decimal(parent, *, minimum=0, maximum=1e9, value=0, decimals=4):
    control = QDoubleSpinBox(parent)
    control.setRange(minimum, maximum)
    control.setDecimals(decimals)
    control.setValue(value)
    control.setKeyboardTracking(False)
    return control


def _integer(parent, *, minimum=0, maximum=10000000, value=0):
    control = QSpinBox(parent)
    control.setRange(minimum, maximum)
    control.setValue(value)
    control.setKeyboardTracking(False)
    return control


class SemWindowView(SemInstanceView):
    def setupUi(self, window):  # noqa: N802 - view convention
        window.resize(1260, 860)
        window.setMinimumSize(880, 640)
        window.setWindowTitle("SEM Particle Analysis")
        root = QVBoxLayout(window)
        root.setContentsMargins(12, 12, 12, 12)
        self.description_label = QLabel(
            "Extract particle sizes from an SEM image, or import measured sizes from CSV. "
            "Prediction comparison is optional.", window,
        )
        self.description_label.setWordWrap(True)
        self.description_label.setProperty("gimapMeta", True)
        root.addWidget(self.description_label)
        split = QSplitter(Qt.Horizontal, window)
        root.addWidget(split, 1)
        scroll = QScrollArea(split)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(340)
        settings = QWidget(scroll)
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 0, 6, 0)
        scroll.setWidget(settings)
        split.addWidget(scroll)
        self._source_controls(settings_layout, settings)
        self._instance_controls(settings_layout, settings)
        self._calibration_controls(settings_layout, settings)
        self._segmentation_controls(settings_layout, settings)
        self._csv_controls(settings_layout, settings)
        self._comparison_controls(settings_layout, settings)
        settings_layout.addStretch(1)
        for combo in settings.findChildren(QComboBox):
            combo.setMinimumContentsLength(10)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for form in settings.findChildren(QFormLayout):
            form.setRowWrapPolicy(QFormLayout.WrapLongRows)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        output = QWidget(split)
        output_layout = QVBoxLayout(output)
        output_layout.setContentsMargins(4, 0, 0, 0)
        toolbar = QHBoxLayout()
        self.summary_label = QLabel("No SEM measurements yet", output)
        self.summary_label.setWordWrap(True)
        toolbar.addWidget(self.summary_label, 1)
        self.export_button = QPushButton("Export SEM results…", output)
        self.export_button.setEnabled(False)
        toolbar.addWidget(self.export_button)
        output_layout.addLayout(toolbar)
        self.result_tabs = QTabWidget(output)
        self.image_panel = PlotPanel(
            "Segmentation review", parent=output,
            empty_title="No SEM image yet",
            empty_message="Open an SEM image. Review the retained particle outlines after extraction.",
        )
        self.distribution_panel = PlotPanel(
            "Particle size distribution", parent=output,
            empty_title="No particle sizes yet",
            empty_message="Extract particles or import a CSV to see the size distribution.",
        )
        self.particles_table = ResultTable(
            ("Particle", "Accepted", "Radius (nm)", "Diameter (nm)", "Area (px²)",
             "Circularity", "Center x (px)", "Center y (px)", "Reason"),
            output, empty_message="No measured particles yet",
        )
        self.particles_table.setProperty(DATA_HEADERS, True)
        enable_table_copy(self.particles_table)
        self.result_tabs.addTab(self.image_panel, "SEM image / outlines")
        self.result_tabs.addTab(self.distribution_panel, "Distribution / comparison")
        self.result_tabs.addTab(self.particles_table, "Particle measurements")
        output_layout.addWidget(self.result_tabs, 1)
        self.notes_label = QLabel("", output)
        self.notes_label.setWordWrap(True)
        self.notes_label.setProperty("gimapMeta", True)
        output_layout.addWidget(self.notes_label)
        split.addWidget(output)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 860])
        self.job_status = JobStatus(window)
        self.job_status.set_actions_visible(pause=False, details=False)
        root.addWidget(self.job_status)

    def _source_controls(self, layout, parent):
        section = ParameterSection("SEM source", parent=parent)
        self.sem_source_edit = QLineEdit(section)
        self.sem_source_edit.setReadOnly(True)
        self.sem_source_edit.setPlaceholderText("Open an image or a particle-size CSV")
        section.add_widget(self.sem_source_edit)
        row = QHBoxLayout()
        self.open_image_button = QPushButton("Open image…", section)
        self.open_csv_button = QPushButton("Open CSV…", section)
        row.addWidget(self.open_image_button)
        row.addWidget(self.open_csv_button)
        section.add_layout(row)
        self.run_button = QPushButton("Extract particles", section)
        self.run_button.setProperty("gimapPrimaryAction", True)
        self.run_button.setEnabled(False)
        section.add_widget(self.run_button)
        layout.addWidget(section)

    def _calibration_controls(self, layout, parent):
        self.calibration_section = ParameterSection(
            "Image calibration", "Enter a known scale. Image annotations are not used automatically.", parent,
        )
        form = QFormLayout()
        self.calibration_mode_combo = QComboBox(parent)
        self.calibration_mode_combo.addItems(["nm per pixel", "Scale-bar length / pixels"])
        self.nm_per_pixel_spin = _decimal(parent, decimals=6)
        self.nm_per_pixel_spin.setSpecialValueText("Enter calibration")
        self.scale_bar_nm_spin = _decimal(parent)
        self.scale_bar_pixels_spin = _decimal(parent)
        form.addRow("Calibration method", self.calibration_mode_combo)
        form.addRow("nm per pixel", self.nm_per_pixel_spin)
        form.addRow("Scale-bar length (nm)", self.scale_bar_nm_spin)
        form.addRow("Scale-bar length (px)", self.scale_bar_pixels_spin)
        self.calibration_section.add_layout(form)
        layout.addWidget(self.calibration_section)

    def _segmentation_controls(self, layout, parent):
        self.segmentation_section = AdvancedSection(
            "Segmentation and image ROI", "Exclude scale bars and annotations using the image ROI.", parent,
        )
        form = QFormLayout()
        self.threshold_mode_combo = QComboBox(parent)
        self.threshold_mode_combo.addItems(["Otsu threshold", "Manual threshold"])
        self.threshold_spin = _decimal(parent, minimum=-1e12, maximum=1e12, decimals=6)
        self.threshold_spin.setToolTip("Manual threshold in the stored image intensity units")
        self.polarity_combo = QComboBox(parent)
        self.polarity_combo.addItems(["Bright particles", "Dark particles"])
        self.min_area_spin = _integer(parent, minimum=1, value=9)
        self.max_area_spin = _integer(parent)
        self.max_area_spin.setSpecialValueText("No upper limit")
        self.circularity_spin = _decimal(parent, maximum=1, decimals=3)
        self.clear_border_check = QCheckBox("Exclude particles touching the ROI boundary", parent)
        self.clear_border_check.setChecked(True)
        self.split_touching_check = QCheckBox("Split touching particles (watershed)", parent)
        self.peak_distance_spin = _integer(parent, minimum=1, maximum=10000, value=3)
        self.roi_check = QCheckBox("Use image ROI (pixel edges)", parent)
        self.roi_x0_spin = _integer(parent)
        self.roi_y0_spin = _integer(parent)
        self.roi_x1_spin = _integer(parent)
        self.roi_y1_spin = _integer(parent)
        for text, control in (
            ("Threshold", self.threshold_mode_combo), ("Stored intensity", self.threshold_spin),
            ("Particle contrast", self.polarity_combo), ("Minimum area (px²)", self.min_area_spin),
            ("Maximum area (px²)", self.max_area_spin), ("Minimum circularity", self.circularity_spin),
            ("Watershed peak distance (px)", self.peak_distance_spin),
        ):
            form.addRow(text, control)
        form.addRow(self.clear_border_check)
        form.addRow(self.split_touching_check)
        form.addRow(self.roi_check)
        for text, control in (("ROI left x", self.roi_x0_spin), ("ROI top y", self.roi_y0_spin),
                              ("ROI right x", self.roi_x1_spin), ("ROI bottom y", self.roi_y1_spin)):
            form.addRow(text, control)
        self.segmentation_section.add_layout(form)
        layout.addWidget(self.segmentation_section)

    def _csv_controls(self, layout, parent):
        self.csv_section = AdvancedSection("CSV measurement settings", parent=parent, expanded=True)
        form = QFormLayout()
        self.csv_column_combo = QComboBox(parent)
        self.csv_column_combo.setEditable(True)
        self.csv_column_combo.setProperty(DATA_VALUES, True)
        self.csv_quantity_combo = QComboBox(parent)
        self.csv_quantity_combo.addItems(["Diameter", "Radius"])
        self.csv_unit_combo = QComboBox(parent)
        for text, data in (("nm", "nm"), ("µm", "um"), ("pixels", "px")):
            self.csv_unit_combo.addItem(text, data)
        self.csv_skip_spin = _integer(parent, maximum=100000)
        self.csv_delimiter_edit = QLineEdit(",", parent)
        self.csv_delimiter_edit.setMaximumWidth(100)
        self.refresh_columns_button = QPushButton("Read CSV columns", parent)
        for text, control in (("Value column", self.csv_column_combo), ("Stored size quantity", self.csv_quantity_combo),
                              ("Stored unit", self.csv_unit_combo), ("Rows before header", self.csv_skip_spin),
                              ("Delimiter", self.csv_delimiter_edit)):
            form.addRow(text, control)
        self.csv_section.add_layout(form)
        self.csv_section.add_widget(self.refresh_columns_button)
        layout.addWidget(self.csv_section)

    def _comparison_controls(self, layout, parent):
        section = ParameterSection("Optional prediction comparison", parent=parent)
        self.prediction_label = QLabel("No prediction attached", section)
        self.prediction_label.setWordWrap(True)
        self.clear_prediction_button = QPushButton("Remove prediction", section)
        self.clear_prediction_button.setEnabled(False)
        section.add_widget(self.prediction_label)
        section.add_widget(self.clear_prediction_button)
        form = QFormLayout()
        self.bins_spin = _integer(parent, minimum=2, maximum=500, value=30)
        self.display_mode_combo = QComboBox(parent)
        self.display_mode_combo.addItems(["Probability (same bins)", "Paper display (independent peak scaling)"])
        self.radius_scale_spin = _decimal(parent, minimum=0.001, maximum=1000, value=1, decimals=5)
        self.radius_scale_spin.setToolTip("Explicit scale applied to SEM radii only; original measurements are retained")
        self.radius_min_spin = _decimal(parent)
        self.radius_max_spin = _decimal(parent)
        self.radius_max_spin.setSpecialValueText("From data and prediction")
        form.addRow("Histogram bins", self.bins_spin)
        form.addRow("Display", self.display_mode_combo)
        form.addRow("SEM radius scale", self.radius_scale_spin)
        form.addRow("Minimum radius (nm)", self.radius_min_spin)
        form.addRow("Maximum radius (nm)", self.radius_max_spin)
        section.add_layout(form)
        self.update_plot_button = QPushButton("Update distribution", section)
        self.update_plot_button.setEnabled(False)
        section.add_widget(self.update_plot_button)
        layout.addWidget(section)
