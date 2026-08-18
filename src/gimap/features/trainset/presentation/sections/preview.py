"""Preview section for the Trainset page."""

from __future__ import annotations

from typing import Dict


from PyQt5.QtCore import Qt


from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation import (
    JobStatus,
    apply_design_system,
)

from src.gimap.app.presentation.section_bindings import (
    bind_advanced_section,
    bind_parameter_section,
)


from ..views import (
    TrainsetPreviewPageView,
)

from ..visualization_widgets import ArrayCanvas, HistogramWidget, ParameterCoverageWidget


class PreviewMixin:
    """Own the preview section."""

    def _preview_page(self) -> QWidget:
        page = QWidget()
        ui = TrainsetPreviewPageView()
        ui.setupUi(page)
        self._preview_page_ui = ui
        self.trainset_preview_run_section = ui.trainsetPreviewRunSection
        self.trainset_preview_advanced_section = ui.trainsetPreviewAdvancedSection
        self.trainset_preview_panel = ui.trainsetPreviewPanel
        bind_parameter_section(
            self.trainset_preview_run_section,
            ui.previewRunTitle,
            ui.previewRunDescription,
            ui.previewRunContent,
            ui.previewRunContentLayout,
        )
        bind_advanced_section(
            self.trainset_preview_advanced_section,
            ui.previewAdvancedToggle,
            ui.previewAdvancedDescription,
            ui.previewAdvancedContent,
            ui.previewAdvancedContentLayout,
        )
        bind_parameter_section(
            self.trainset_preview_panel,
            ui.previewPanelTitle,
            ui.previewPanelDescription,
            ui.previewPanelContent,
            ui.previewPanelContentLayout,
        )
        apply_design_system(page)
        intro = QLabel(
            "Simulation-first preview: BornAgain creates the scattering pattern. Choose any sampled physics, background or noise range "
            "to compare its minimum, midpoint and maximum without using the experimental image as training data."
        )
        intro.setWordWrap(True)
        controls_widget = QWidget()
        controls = QVBoxLayout(controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        selection_controls = QGridLayout()
        selection_controls.addWidget(QLabel("Compare range"), 0, 0)
        self.impact_parameter_combo = QComboBox()
        self.impact_parameter_combo.setMinimumWidth(180)
        self.impact_parameter_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        selection_controls.addWidget(self.impact_parameter_combo, 0, 1)
        selection_controls.addWidget(QLabel("Coverage samples"), 1, 0)
        self.preview_count = QSpinBox()
        self.preview_count.setRange(3, 1000)
        self.preview_count.setValue(16)
        self.preview_count.setToolTip(
            "Number of label samples drawn for the parameter-coverage diagnostic; it does not rerun BornAgain."
        )
        selection_controls.addWidget(self.preview_count, 1, 1)
        selection_controls.setColumnStretch(1, 1)
        controls.addLayout(selection_controls)

        action_controls = QGridLayout()
        self.generate_preview_button = QPushButton("Update simulated comparison")
        self.generate_preview_button.setObjectName("primaryAction")
        self.generate_preview_button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.generate_preview_button.setMinimumWidth(190)
        self.generate_preview_button.setMinimumHeight(38)
        action_controls.addWidget(self.generate_preview_button, 0, 0, 1, 2)
        self.force_simulation_button = QPushButton("Recompute BornAgain")
        self.force_simulation_button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.force_simulation_button.setMinimumWidth(160)
        self.force_simulation_button.setMinimumHeight(38)
        self.force_simulation_button.setToolTip(
            "Clear the in-memory physics cache and explicitly rerun BornAgain."
        )
        action_controls.addWidget(self.force_simulation_button, 1, 0)
        self.new_realization_button = QPushButton("New noise / mask realization")
        self.new_realization_button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.new_realization_button.setMinimumWidth(180)
        self.new_realization_button.setMinimumHeight(38)
        self.new_realization_button.setToolTip(
            "Keep cached BornAgain images, but draw fresh stochastic background, noise, mask and edge-crop values."
        )
        action_controls.addWidget(self.new_realization_button, 1, 1)
        self.preview_parameters_button = QPushButton("View parameters used…")
        self.preview_parameters_button.setEnabled(False)
        self.preview_parameters_button.setToolTip(
            "Open a roomy read-only window showing the exact physics, preprocessing realization, beam, detector and ROI values for all three images."
        )
        self.preview_parameters_button.clicked.connect(self.show_comparison_parameters)
        action_controls.addWidget(self.preview_parameters_button, 2, 0, 1, 2)
        action_controls.setColumnStretch(0, 1)
        action_controls.setColumnStretch(1, 1)
        controls.addLayout(action_controls)
        self.preview_cache_status = QLabel("BornAgain cache: empty")
        self.preview_cache_status.setWordWrap(True)
        controls.addWidget(self.preview_cache_status)
        self.preview_job_status = JobStatus()
        self.preview_job_status.set_actions_visible(
            pause=False,
            cancel=False,
            details=False,
        )
        self.preview_job_status.setVisible(False)
        self.preview_progress = self.preview_job_status.progress_bar
        self.preview_progress.setRange(0, 100)
        self.preview_progress.setValue(0)
        self.preview_progress.setTextVisible(True)
        self.preview_activity = self.preview_job_status.message_label
        self.preview_activity.setWordWrap(True)
        controls.addWidget(self.preview_job_status)
        # Kept as an internal compatibility field for local-run code. Local
        # Preview itself is always simulated.
        self.preview_mode = QComboBox()
        self.preview_mode.addItem("Simulated impact")

        self.preprocessing_tabs = QTabWidget()
        self.preprocessing_tabs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        background_page = QWidget()
        background_layout = QVBoxLayout(background_page)
        background_enable = self._check("pre.background.enabled", False, "Add physical background")
        background_enable.setToolTip(
            "Synthetic GISAXS background based on the former Yuxin_train model. Every component can be ranged and inspected below."
        )
        background_layout.addWidget(background_enable)
        background_groups = (
            (
                "General",
                {
                    "target_fraction",
                    "constant_fraction",
                    "plane_qy_slope",
                    "plane_qz_slope",
                    "low_qz_cut_fraction",
                    "blur_sigma_px",
                },
            ),
            (
                "Specular ridge",
                {
                    "specular_amplitude",
                    "specular_width_fraction",
                    "specular_widening",
                    "specular_decay_fraction",
                },
            ),
            (
                "Yoneda band",
                {
                    "yoneda_amplitude",
                    "yoneda_center_fraction",
                    "yoneda_width_fraction",
                    "yoneda_center_hole",
                },
            ),
            (
                "Diffuse wedge",
                {
                    "wedge_amplitude",
                    "wedge_anisotropy",
                    "wedge_porod_exponent",
                    "wedge_rg_fraction",
                },
            ),
        )
        background_component_tabs = QTabWidget()
        self.background_component_tabs = background_component_tabs
        self.background_parameter_tables = []
        for group_name, keys in background_groups:
            definitions = [
                item for item in self.catalog.background_parameters() if item["key"] in keys
            ]
            background_table = QTableWidget(len(definitions), 3)
            background_table.setHorizontalHeaderLabels(("Parameter", "Minimum", "Maximum"))
            background_table.verticalHeader().setVisible(False)
            background_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            background_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
            background_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
            background_table.setColumnWidth(1, 126)
            background_table.setColumnWidth(2, 126)
            for row, definition in enumerate(definitions):
                label = QTableWidgetItem(str(definition["label"]))
                label.setToolTip(str(definition["help"]))
                label.setFlags(label.flags() & ~Qt.ItemIsEditable)
                background_table.setItem(row, 0, label)
                key = str(definition["key"])
                minimum = self._double(
                    f"pre.background.{key}.min",
                    float(definition["minimum"]),
                    -1e6,
                    1e6,
                    int(definition["decimals"]),
                )
                maximum = self._double(
                    f"pre.background.{key}.max",
                    float(definition["maximum"]),
                    -1e6,
                    1e6,
                    int(definition["decimals"]),
                )
                minimum.setToolTip(str(definition["help"]))
                maximum.setToolTip(str(definition["help"]))
                background_table.setCellWidget(row, 1, minimum)
                background_table.setCellWidget(row, 2, maximum)
            visible_rows = max(4, len(definitions))
            background_table.setMinimumHeight(34 + visible_rows * 31)
            self.background_parameter_tables.append(background_table)
            background_component_tabs.addTab(background_table, group_name)
        background_layout.addWidget(background_component_tabs)
        self.background_parameter_table = self.background_parameter_tables[0]
        self.preprocessing_tabs.addTab(background_page, "Physical background")

        noise_page = QWidget()
        noise_layout = QGridLayout(noise_page)
        gaussian = self._check("pre.gaussian.enabled", True, "Gaussian readout noise")
        gaussian.setToolTip(
            "Adds zero-mean Gaussian noise. Lower SNR means stronger noise; SNR is computed from mean signal power."
        )
        noise_layout.addWidget(gaussian, 0, 0, 1, 3)
        snr_label = QLabel("SNR range (dB)")
        snr_label.setToolTip("Signal-to-noise ratio in dB. Lower values add more Gaussian noise.")
        noise_layout.addWidget(snr_label, 1, 0)
        noise_layout.addWidget(self._double("pre.gaussian.min", 80.0, -100.0, 300.0, 2), 1, 1)
        noise_layout.addWidget(self._double("pre.gaussian.max", 110.0, -100.0, 300.0, 2), 1, 2)
        poisson = self._check("pre.poisson.enabled", False, "Poisson photon-count noise")
        poisson.setToolTip(
            "Converts intensity to expected photon counts, draws Poisson counts, then converts back."
        )
        noise_layout.addWidget(poisson, 2, 0, 1, 3)
        poisson_label = QLabel("Photon-count scale")
        poisson_help = (
            "Intensity multiplier before Poisson sampling. Low scale means few counts and stronger relative shot noise; "
            "high scale means more counts and weaker relative noise."
        )
        poisson_label.setToolTip(poisson_help)
        poisson_min = self._double("pre.poisson.min", 1.0, 1e-6, 1e9, 3)
        poisson_max = self._double("pre.poisson.max", 20.0, 1e-6, 1e9, 3)
        poisson_min.setToolTip(poisson_help)
        poisson_max.setToolTip(poisson_help)
        noise_layout.addWidget(poisson_label, 3, 0)
        noise_layout.addWidget(poisson_min, 3, 1)
        noise_layout.addWidget(poisson_max, 3, 2)
        independent_help = QLabel(
            "Gaussian and Poisson are independent: enable neither, either one, or both. Applied order: Gaussian → Poisson."
        )
        independent_help.setWordWrap(True)
        noise_layout.addWidget(independent_help, 4, 0, 1, 3)
        noise_layout.setColumnStretch(1, 1)
        noise_layout.setColumnStretch(2, 1)
        self.preprocessing_tabs.addTab(noise_page, "Noise")

        transform_page = QWidget()
        chain = QGridLayout(transform_page)
        chain.addWidget(self._check("pre.mask.enabled", True), 0, 0)
        mask_stage_label = QLabel("1. Apply configured detector mask")
        mask_stage_label.setWordWrap(True)
        mask_stage_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        chain.addWidget(mask_stage_label, 0, 1, 1, 3)
        chain.addWidget(self._check("pre.log.enabled", True), 1, 0)
        log_stage_label = QLabel("2. Log transform compresses scattering dynamic range")
        log_stage_label.setWordWrap(True)
        log_stage_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        chain.addWidget(log_stage_label, 1, 1, 1, 3)
        chain.addWidget(self._check("pre.normalize.enabled", True), 2, 0)
        normalize_stage_label = QLabel("3. Normalize valid pixels")
        normalize_stage_label.setWordWrap(True)
        normalize_stage_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        chain.addWidget(normalize_stage_label, 2, 1)
        chain.addWidget(
            self._combo("pre.normalize.mode", ("range", "upper", "lower"), "range"), 2, 2
        )
        normalize_bounds = QHBoxLayout()
        normalize_bounds.addWidget(self._double("pre.normalize.lower", 0.0))
        normalize_bounds.addWidget(self._double("pre.normalize.upper", 1.0))
        chain.addLayout(normalize_bounds, 2, 3)
        chain.addWidget(self._check("pre.edge.enabled", False), 3, 0)
        crop_stage_label = QLabel("4. Random edge crop then resize back")
        crop_stage_label.setWordWrap(True)
        crop_stage_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        chain.addWidget(crop_stage_label, 3, 1)
        chain.addWidget(QLabel("Maximum px"), 3, 2)
        chain.addWidget(self._spin("pre.edge.maximum", 4, 0, 128), 3, 3)
        transform_help = QLabel(
            "Each enabled stage is shown separately in the Pipeline stages view."
        )
        transform_help.setWordWrap(True)
        chain.addWidget(transform_help, 4, 0, 1, 4)
        chain.setColumnStretch(1, 1)
        self.preprocessing_tabs.addTab(transform_page, "Mask & transforms")
        self.preprocessing_tabs.setMinimumHeight(330)
        self.preview_views = QTabWidget()
        self.preview_views.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        comparison_page = QWidget()
        comparison_layout = QVBoxLayout(comparison_page)
        self.impact_canvases: Dict[str, list[ArrayCanvas]] = {
            "minimum": [],
            "midpoint": [],
            "maximum": [],
        }
        self.impact_value_labels: Dict[str, list[QLabel]] = {
            "minimum": [],
            "midpoint": [],
            "maximum": [],
        }
        self.impact_responsive_stack = QStackedWidget()
        self.impact_responsive_stack.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        wide = QWidget()
        wide_layout = QHBoxLayout(wide)
        for key, heading in (
            ("minimum", "Minimum"),
            ("midpoint", "Midpoint"),
            ("maximum", "Maximum"),
        ):
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            value_label = QLabel(heading)
            value_label.setAlignment(Qt.AlignCenter)
            canvas = ArrayCanvas(f"{heading} simulated result")
            canvas.setMinimumSize(180, 210)
            panel_layout.addWidget(value_label)
            panel_layout.addWidget(canvas, 1)
            wide_layout.addWidget(panel, 1)
            self.impact_canvases[key].append(canvas)
            self.impact_value_labels[key].append(value_label)
        self.impact_responsive_stack.addWidget(wide)
        compact = QTabWidget()
        for key, heading in (
            ("minimum", "Minimum"),
            ("midpoint", "Midpoint"),
            ("maximum", "Maximum"),
        ):
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            value_label = QLabel(heading)
            value_label.setAlignment(Qt.AlignCenter)
            canvas = ArrayCanvas(f"{heading} simulated result")
            canvas.setMinimumSize(180, 210)
            panel_layout.addWidget(value_label)
            panel_layout.addWidget(canvas, 1)
            compact.addTab(panel, heading)
            self.impact_canvases[key].append(canvas)
            self.impact_value_labels[key].append(value_label)
        self.impact_responsive_stack.addWidget(compact)
        comparison_layout.addWidget(self._make_display_bar("preview"))
        comparison_layout.addWidget(self.impact_responsive_stack, 1)
        self.preview_views.addTab(comparison_page, "Range impact")

        pipeline_page = QWidget()
        pipeline_layout = QVBoxLayout(pipeline_page)
        self.preview_tabs = QTabWidget()
        placeholder = QLabel(
            "Run Update simulated comparison. Only BornAgain Raw and the preprocessing stages you enabled will appear here, in execution order."
        )
        placeholder.setWordWrap(True)
        placeholder.setAlignment(Qt.AlignCenter)
        self.preview_tabs.addTab(placeholder, "No result yet")
        pipeline_layout.addWidget(self.preview_tabs)
        self.preview_views.addTab(pipeline_page, "Pipeline stages")

        diagnostics = QWidget()
        diagnostics_layout = QVBoxLayout(diagnostics)
        self.preview_stats = QLabel(
            "Update the simulated comparison to inspect cache use, tensor shape and dynamic range."
        )
        self.preview_stats.setWordWrap(True)
        self.preview_stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.diagnostic_tabs = QTabWidget()
        self.histogram = HistogramWidget()
        self.parameter_coverage = ParameterCoverageWidget()
        self.diagnostic_tabs.addTab(self.histogram, "Intensity")
        self.diagnostic_tabs.addTab(self.parameter_coverage, "Ground-truth distribution")
        self.preview_gate_table = QTableWidget(4, 2)
        self.preview_gate_table.setHorizontalHeaderLabels(("Local readiness check", "State"))
        for row, gate in enumerate(
            (
                "Configuration valid",
                "Local samples generated",
                "Tensor shapes compatible",
                "Storage estimate accepted",
            )
        ):
            self.preview_gate_table.setItem(row, 0, QTableWidgetItem(gate))
            self.preview_gate_table.setItem(row, 1, QTableWidgetItem("Pending"))
        self.preview_gate_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.preview_gate_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.storage_accept_check = QCheckBox("Storage estimate reviewed")
        self.storage_accept_check.setToolTip(
            "Confirm that you reviewed the estimated local storage for the configured full dataset."
        )
        diagnostics_layout.addWidget(self.preview_stats)
        diagnostics_layout.addWidget(self.diagnostic_tabs, 1)
        diagnostics_layout.addWidget(self.storage_accept_check)
        diagnostics_layout.addWidget(self.preview_gate_table)
        self.preview_views.addTab(diagnostics, "Diagnostics")
        ui.previewRunContentLayout.addWidget(intro)
        ui.previewRunContentLayout.addWidget(controls_widget)
        ui.previewAdvancedContentLayout.addWidget(self.preprocessing_tabs)
        ui.previewPanelContentLayout.addWidget(self.preview_views, 1)
        self.preview_capability = self.preview_cache_status
        return page
