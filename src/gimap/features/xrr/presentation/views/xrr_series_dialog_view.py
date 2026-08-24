"""Hand-maintained Python View for the XRR series extractor."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from src.gimap.app.presentation import (
    FilePicker,
    JobStatus,
    ParameterSection,
    PlotPanel,
    ResultTable,
    SafeWheelComboBox,
    SafeWheelDoubleSpinBox,
    SafeWheelSpinBox,
)


class XrrSeriesDialogView:
    def setupUi(self, dialog) -> None:
        dialog.setObjectName("xrrSeriesDialog")
        dialog.resize(1280, 820)
        dialog.setMinimumSize(QtCore.QSize(960, 640))
        dialog.setModal(False)
        self.root_layout = QtWidgets.QVBoxLayout(dialog)
        self.root_layout.setContentsMargins(12, 12, 12, 12)
        self.root_layout.setSpacing(8)

        self.heading = QtWidgets.QLabel("XRR series extractor", dialog)
        self.heading.setProperty("gimapPageTitle", True)
        self.subtitle = QtWidgets.QLabel(
            "Stream GIWAXS/GISAXS detector frames and extract the moving specular reflection.",
            dialog,
        )
        self.subtitle.setProperty("gimapSectionDescription", True)
        self.subtitle.setWordWrap(True)
        self.root_layout.addWidget(self.heading)
        self.root_layout.addWidget(self.subtitle)

        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, dialog)
        self.main_splitter.setObjectName("xrrMainSplitter")
        self.root_layout.addWidget(self.main_splitter, 1)
        self._build_controls()
        self._build_preview_tabs()
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([440, 800])
        self._set_tab_order(dialog)

    def _build_controls(self) -> None:
        self.controls_rail = QtWidgets.QWidget(self.main_splitter)
        self.controls_rail.setObjectName("xrrControlsRail")
        self.controls_rail.setMinimumWidth(350)
        self.controls_rail.setMaximumWidth(480)
        self.controls_rail_layout = QtWidgets.QVBoxLayout(self.controls_rail)
        self.controls_rail_layout.setContentsMargins(0, 0, 6, 0)
        self.controls_rail_layout.setSpacing(8)
        self.controls_scroll = QtWidgets.QScrollArea(self.controls_rail)
        self.controls_scroll.setObjectName("xrrControlsScroll")
        self.controls_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.controls = QtWidgets.QWidget()
        self.controls_layout = QtWidgets.QVBoxLayout(self.controls)
        self.controls_layout.setContentsMargins(0, 0, 6, 0)
        self.controls_layout.setSpacing(8)
        self._build_input_section()
        self._build_angle_section()
        self._build_geometry_section()
        self.controls_layout.addStretch(1)
        self.controls_scroll.setWidget(self.controls)
        self.controls_rail_layout.addWidget(self.controls_scroll, 1)
        self._build_extraction_section()

    def _build_input_section(self) -> None:
        self.input_section = ParameterSection(
            "Series data",
            "Choose one NXS module file or a CBF file/folder. NXS modules are stitched per frame.",
            self.controls,
        )
        self.source_picker = FilePicker(
            self.input_section.content,
            placeholder="NXS module, CBF file, or CBF folder…",
        )
        self.source_picker.setObjectName("xrrSourcePicker")
        self.source_picker.clear_button.hide()
        self.input_section.add_widget(self.source_picker)
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.source_kind_combo = SafeWheelComboBox(self.input_section.content)
        self.source_kind_combo.setObjectName("xrrSourceKindCombo")
        self.source_kind_combo.addItems(["Auto detect", "NXS scan", "CBF series"])
        self.pattern_edit = QtWidgets.QLineEdit("*.cbf", self.input_section.content)
        self.pattern_edit.setObjectName("xrrPatternEdit")
        form.addRow("Source type", self.source_kind_combo)
        form.addRow("CBF pattern", self.pattern_edit)
        self.input_section.add_layout(form)
        self.inspect_button = QtWidgets.QPushButton("Load first frame", self.input_section.content)
        self.inspect_button.setObjectName("xrrInspectButton")
        self.input_section.add_widget(self.inspect_button)
        self.series_summary = QtWidgets.QLabel("No series inspected", self.input_section.content)
        self.series_summary.setProperty("gimapMeta", True)
        self.series_summary.setWordWrap(True)
        self.input_section.add_widget(self.series_summary)
        self.controls_layout.addWidget(self.input_section)

    def _build_angle_section(self) -> None:
        self.angle_section = ParameterSection(
            "Sample-angle series",
            "Theta is the sample-table angle; the beam and detector remain fixed.",
            self.controls,
        )
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.angle_mode_combo = SafeWheelComboBox(self.angle_section.content)
        self.angle_mode_combo.setObjectName("xrrAngleModeCombo")
        self.angle_mode_combo.addItems(["Linear start + step", "NXS motor dataset"])
        self.theta_start_spin = self._double(0.0, -10.0, 10.0, 5, 0.01)
        self.theta_start_spin.setObjectName("xrrThetaStartSpin")
        self.theta_step_spin = self._double(0.01, -10.0, 10.0, 6, 0.001)
        self.theta_step_spin.setObjectName("xrrThetaStepSpin")
        self.angle_dataset_edit = QtWidgets.QLineEdit(self.angle_section.content)
        self.angle_dataset_edit.setObjectName("xrrAngleDatasetEdit")
        self.angle_dataset_edit.setPlaceholderText("/entry/sample/transformations/omega")
        form.addRow("Angle source", self.angle_mode_combo)
        form.addRow("Theta start (°)", self.theta_start_spin)
        form.addRow("Theta step (°)", self.theta_step_spin)
        form.addRow("NXS dataset", self.angle_dataset_edit)
        self.angle_section.add_layout(form)
        self.controls_layout.addWidget(self.angle_section)

    def _build_geometry_section(self) -> None:
        self.geometry_section = ParameterSection(
            "Detector geometry",
            "The center is the direct-beam position at theta = 0. Pick it on the preview if needed.",
            self.controls,
        )
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.distance_spin = self._double(5000.0, 0.001, 100000.0, 3, 10.0)
        self.distance_spin.setObjectName("xrrDistanceSpin")
        self.energy_spin = self._double(12.0, 0.01, 200.0, 5, 0.1)
        self.energy_spin.setObjectName("xrrEnergySpin")
        self.pixel_x_spin = self._double(172.0, 0.001, 10000.0, 4, 1.0)
        self.pixel_x_spin.setObjectName("xrrPixelXSpin")
        self.pixel_y_spin = self._double(172.0, 0.001, 10000.0, 4, 1.0)
        self.pixel_y_spin.setObjectName("xrrPixelYSpin")
        self.center_x_spin = self._double(0.0, -100000.0, 100000.0, 2, 1.0)
        self.center_x_spin.setObjectName("xrrCenterXSpin")
        self.center_y_spin = self._double(0.0, -100000.0, 100000.0, 2, 1.0)
        self.center_y_spin.setObjectName("xrrCenterYSpin")
        self.direction_combo = SafeWheelComboBox(self.geometry_section.content)
        self.direction_combo.setObjectName("xrrDirectionCombo")
        self.direction_combo.addItems(["Specular moves up (−y)", "Specular moves down (+y)"])
        form.addRow("Distance (mm)", self.distance_spin)
        form.addRow("Energy (keV)", self.energy_spin)
        form.addRow("Pixel X (µm)", self.pixel_x_spin)
        form.addRow("Pixel Y (µm)", self.pixel_y_spin)
        form.addRow("Beam center X (px)", self.center_x_spin)
        form.addRow("Beam center Y (px)", self.center_y_spin)
        form.addRow("Reflection direction", self.direction_combo)
        self.geometry_section.add_layout(form)
        self.pick_center_button = QtWidgets.QPushButton(
            "Pick direct-beam center", self.geometry_section.content
        )
        self.pick_center_button.setObjectName("xrrPickCenterButton")
        self.pick_center_button.setCheckable(True)
        self.geometry_section.add_widget(self.pick_center_button)
        self.controls_layout.addWidget(self.geometry_section)

    def _build_extraction_section(self) -> None:
        self.run_section = ParameterSection(
            "Extract XRR",
            "Radius 0 reads one pixel; larger radii integrate a circular neighborhood.",
            self.controls_rail,
        )
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.radius_spin = SafeWheelSpinBox(self.run_section.content)
        self.radius_spin.setRange(0, 500)
        self.radius_spin.setValue(2)
        self.radius_spin.setObjectName("xrrRadiusSpin")
        self.aggregation_combo = SafeWheelComboBox(self.run_section.content)
        self.aggregation_combo.addItems(["Sum", "Mean"])
        self.aggregation_combo.setObjectName("xrrAggregationCombo")
        form.addRow("ROI radius (px)", self.radius_spin)
        form.addRow("Intensity", self.aggregation_combo)
        self.run_section.add_layout(form)
        actions = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run extraction", self.run_section.content)
        self.run_button.setObjectName("xrrRunButton")
        self.run_button.setProperty("gimapPrimaryAction", True)
        self.export_button = QtWidgets.QPushButton("Export CSV…", self.run_section.content)
        self.export_button.setObjectName("xrrExportButton")
        self.export_button.setEnabled(False)
        actions.addWidget(self.run_button, 1)
        actions.addWidget(self.export_button)
        self.run_section.add_layout(actions)
        self.job_status = JobStatus(self.run_section.content)
        self.job_status.setObjectName("xrrJobStatus")
        self.job_status.set_actions_visible(pause=False, cancel=True, details=False)
        self.run_section.add_widget(self.job_status)
        self.controls_rail_layout.addWidget(self.run_section, 0)

    def _build_preview_tabs(self) -> None:
        self.preview_tabs = QtWidgets.QTabWidget(self.main_splitter)
        self.preview_tabs.setObjectName("xrrPreviewTabs")
        self.live_tab = QtWidgets.QWidget()
        live_layout = QtWidgets.QVBoxLayout(self.live_tab)
        live_layout.setContentsMargins(0, 0, 0, 0)
        self.live_panel = PlotPanel(
            "Live detector frame",
            "The calculated circular ROI follows the specular beam for the current theta.",
            self.live_tab,
            empty_title="No detector frame",
            empty_message="Load the first frame or start extraction.",
        )
        self.live_frame_label = QtWidgets.QLabel("", self.live_panel.content)
        self.live_frame_label.setProperty("gimapMeta", True)
        self.live_panel.add_toolbar_widget(self.live_frame_label)
        self.live_panel.add_toolbar_stretch()
        live_layout.addWidget(self.live_panel)
        self.preview_tabs.addTab(self.live_tab, "Live frame")

        self.curve_tab = QtWidgets.QWidget()
        curve_layout = QtWidgets.QVBoxLayout(self.curve_tab)
        curve_layout.setContentsMargins(0, 0, 0, 0)
        self.curve_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self.curve_tab)
        self.curve_panel = PlotPanel(
            "XRR curve",
            "Extracted intensity versus specular qz.",
            self.curve_splitter,
            empty_title="No XRR points",
            empty_message="Run extraction to build the curve point by point.",
        )
        self.log_y_check = QtWidgets.QCheckBox("Log intensity", self.curve_panel.content)
        self.log_y_check.setChecked(True)
        self.curve_panel.add_toolbar_widget(self.log_y_check)
        self.results_table = ResultTable(
            ("#", "Theta (°)", "qz (Å⁻¹)", "Intensity", "ROI x", "ROI y", "Pixels"),
            self.curve_splitter,
            empty_message="No extracted points",
        )
        self.results_table.setObjectName("xrrResultsTable")
        self.curve_splitter.addWidget(self.curve_panel)
        self.curve_splitter.addWidget(self.results_table)
        self.curve_splitter.setSizes([480, 220])
        curve_layout.addWidget(self.curve_splitter)
        self.preview_tabs.addTab(self.curve_tab, "XRR points")

    def _double(self, value, minimum, maximum, decimals, step):
        spin = SafeWheelDoubleSpinBox(self.controls)
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        return spin

    def _set_tab_order(self, dialog) -> None:
        dialog.setTabOrder(self.source_picker.path_edit, self.source_picker.browse_button)
        dialog.setTabOrder(self.source_picker.browse_button, self.source_kind_combo)
        dialog.setTabOrder(self.source_kind_combo, self.pattern_edit)
        dialog.setTabOrder(self.pattern_edit, self.inspect_button)
        dialog.setTabOrder(self.inspect_button, self.angle_mode_combo)
        dialog.setTabOrder(self.angle_mode_combo, self.theta_start_spin)
        dialog.setTabOrder(self.theta_start_spin, self.theta_step_spin)
        dialog.setTabOrder(self.theta_step_spin, self.angle_dataset_edit)
        dialog.setTabOrder(self.angle_dataset_edit, self.distance_spin)
        dialog.setTabOrder(self.distance_spin, self.energy_spin)
        dialog.setTabOrder(self.energy_spin, self.pixel_x_spin)
        dialog.setTabOrder(self.pixel_x_spin, self.pixel_y_spin)
        dialog.setTabOrder(self.pixel_y_spin, self.center_x_spin)
        dialog.setTabOrder(self.center_x_spin, self.center_y_spin)
        dialog.setTabOrder(self.center_y_spin, self.direction_combo)
        dialog.setTabOrder(self.direction_combo, self.pick_center_button)
        dialog.setTabOrder(self.pick_center_button, self.radius_spin)
        dialog.setTabOrder(self.radius_spin, self.aggregation_combo)
        dialog.setTabOrder(self.aggregation_combo, self.run_button)


__all__ = ["XrrSeriesDialogView"]
