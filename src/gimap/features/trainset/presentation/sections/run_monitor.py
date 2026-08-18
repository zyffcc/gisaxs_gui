"""Run Monitor section for the Trainset page."""

from __future__ import annotations


from PyQt5.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTabWidget,
    QTextEdit,
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
    TrainsetModelPageView,
    TrainsetMonitorPageView,
    TrainsetRunPageView,
)


class RunMonitorMixin:
    """Own the run monitor section."""

    def _model_page(self) -> QWidget:
        page = QWidget()
        ui = TrainsetModelPageView()
        ui.setupUi(page)
        self._model_page_ui = ui
        self.trainset_model_configure_section = ui.trainsetModelConfigureSection
        self.trainset_model_advanced_section = ui.trainsetModelAdvancedSection
        self.trainset_model_preview_section = ui.trainsetModelPreviewSection
        self.trainset_model_run_section = ui.trainsetModelRunSection
        bind_parameter_section(
            self.trainset_model_configure_section,
            ui.modelConfigureTitle,
            ui.modelConfigureDescription,
            ui.modelConfigureContent,
            ui.modelConfigureContentLayout,
        )
        bind_advanced_section(
            self.trainset_model_advanced_section,
            ui.modelAdvancedToggle,
            ui.modelAdvancedDescription,
            ui.modelAdvancedContent,
            ui.modelAdvancedContentLayout,
        )
        bind_parameter_section(
            self.trainset_model_preview_section,
            ui.modelPreviewTitle,
            ui.modelPreviewDescription,
            ui.modelPreviewContent,
            ui.modelPreviewContentLayout,
        )
        bind_parameter_section(
            self.trainset_model_run_section,
            ui.modelRunTitle,
            ui.modelRunDescription,
            ui.modelRunContent,
            ui.modelRunContentLayout,
        )
        self.model_layer_table = ui.model_layer_table
        self.model_layer_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.add_model_layer_button = ui.add_model_layer_button
        self.remove_model_layer_button = ui.remove_model_layer_button
        self.move_model_layer_up_button = ui.move_model_layer_up_button
        self.move_model_layer_down_button = ui.move_model_layer_down_button
        training = QGroupBox("Training controls")
        training_form = QFormLayout(training)
        training_form.addRow(
            "Output mode", self._combo("model.output_mode", ("regression",), "regression")
        )
        training_form.addRow("Batch size", self._spin("training.batch_size", 64, 1, 100000))
        training_form.addRow("Epochs", self._spin("training.epochs", 100, 1, 100000))
        training_form.addRow(
            "Optimizer", self._combo("training.optimizer", ("adam", "adamw", "sgd"), "adam")
        )
        training_form.addRow(
            "Learning rate", self._double("training.learning_rate", 0.0001, 1e-9, 10.0, 8)
        )
        training_form.addRow(
            "Scheduler",
            self._combo("training.scheduler", ("cosine", "plateau", "constant"), "cosine"),
        )
        ui.modelConfigureContentLayout.addWidget(training)
        self.model_summary = ui.model_summary
        self.model_validate_button = ui.model_validate_button
        apply_design_system(page)
        self.add_model_layer_button.clicked.connect(
            lambda: self.add_model_layer(
                {"type": "conv2d", "units": 32, "kernel": 3, "activation": "relu"}
            )
        )
        self.remove_model_layer_button.clicked.connect(
            lambda: self._remove_selected_rows(self.model_layer_table)
        )
        self.move_model_layer_up_button.clicked.connect(lambda: self._move_model_layer(-1))
        self.move_model_layer_down_button.clicked.connect(lambda: self._move_model_layer(1))
        return page

    def _hpc_page(self) -> QWidget:
        page = QWidget()
        ui = TrainsetRunPageView()
        ui.setupUi(page)
        self._run_page_ui = ui
        self.trainset_run_section = ui.trainsetRunSection
        self.trainset_export_section = ui.trainsetExportSection
        bind_parameter_section(
            self.trainset_run_section,
            ui.runTitle,
            ui.runDescription,
            ui.runContent,
            ui.runContentLayout,
        )
        bind_parameter_section(
            self.trainset_export_section,
            ui.exportTitle,
            ui.exportDescription,
            ui.exportContent,
            ui.exportContentLayout,
        )
        apply_design_system(page)
        tabs = QTabWidget()
        local = QWidget()
        local_form = QFormLayout(local)
        local_intro = QLabel(
            "Local physical workflow: choose an output folder/Python, prepare the reproducible package, "
            "generate a small BornAgain test first, then generate the full dataset and train."
        )
        local_intro.setWordWrap(True)
        local_form.addRow(local_intro)
        local_form.addRow("Output folder", self._line("project.workspace", ""))
        local_form.addRow("Dataset folder", self._line("runtime.dataset_output_dir", ""))
        local_form.addRow("Training results folder", self._line("runtime.results_output_dir", ""))
        local_form.addRow("Python executable", self._line("training.local_python", ""))
        self.local_python_button = QPushButton("Choose Python executable…")
        local_form.addRow(self.local_python_button)
        self.local_folder_button = QPushButton("Choose output folder…")
        local_form.addRow(self.local_folder_button)
        self.local_dataset_folder_button = QPushButton("Choose dataset folder…")
        self.local_results_folder_button = QPushButton("Choose training results folder…")
        local_form.addRow(self.local_dataset_folder_button)
        local_form.addRow(self.local_results_folder_button)
        cache_group = QGroupBox("BornAgain form-factor grid cache")
        cache_form = QFormLayout(cache_group)
        cache_form.addRow(
            self._check(
                "simulation.grid_cache.enabled",
                True,
                "Use the precomputed particle-parameter matrix during dataset generation",
            )
        )
        cache_form.addRow(
            "Cache folder", self._line("simulation.grid_cache.directory", "_bornagain_cache")
        )
        self.local_cache_folder_button = QPushButton("Choose cache folder…")
        cache_form.addRow(self.local_cache_folder_button)
        cache_form.addRow(
            "Maximum cache files", self._spin("simulation.grid_cache.max_files", 5, 1, 50)
        )
        self.cache_grid_summary = QLabel(
            "Grid points are set per particle parameter in Dataset Design. "
            "For radius=30 and height=30, BornAgain precomputes one 30 × 30 form-factor matrix."
        )
        self.cache_grid_summary.setWordWrap(True)
        cache_form.addRow(self.cache_grid_summary)
        local_form.addRow(cache_group)
        local_form.addRow("Test samples", self._spin("training.smoke_samples", 64, 8, 10000))
        local_form.addRow("Test epochs", self._spin("training.smoke_epochs", 2, 1, 20))
        self.local_prepare_button = QPushButton("1 · Prepare local job package")
        self.local_generate_test_button = QPushButton("2 · Generate small physical BornAgain test")
        self.local_generate_test_button.setToolTip(
            "Generate Test samples with the real BornAgain pipeline."
        )
        self.local_generate_button = QPushButton("3 · Generate full physical dataset")
        self.local_train_button = QPushButton("4 · Train on generated dataset")
        self.local_smoke_button = QPushButton("Optional · Reference-based I/O smoke test")
        self.local_smoke_button.setToolTip(
            "Fast non-physical I/O/model check using a loaded reference image. It is not a replacement for the small BornAgain physical test."
        )
        for button in (
            self.local_prepare_button,
            self.local_generate_test_button,
            self.local_generate_button,
            self.local_train_button,
            self.local_smoke_button,
        ):
            local_form.addRow(button)
        self.trainset_job_status = JobStatus()
        self.trainset_job_status.set_actions_visible(
            pause=False,
            cancel=False,
            details=False,
        )
        self.trainset_job_status.set_state("idle", "Idle", progress=0.0)
        self.local_activity = self.trainset_job_status.message_label
        self.local_activity.setWordWrap(True)
        self.local_progress = self.trainset_job_status.progress_bar
        self.local_progress.setRange(0, 100)
        self.local_progress.setValue(0)
        local_controls = QHBoxLayout()
        self.local_pause_button = QPushButton("Pause")
        self.local_pause_button.setEnabled(False)
        self.local_stop_button = QPushButton("Stop safely")
        self.local_stop_button.setEnabled(False)
        local_controls.addWidget(self.local_pause_button)
        local_controls.addWidget(self.local_stop_button)
        local_controls.addStretch(1)
        local_form.addRow(self.trainset_job_status)
        local_form.addRow(local_controls)
        output_help = QLabel(
            "Generation writes HDF5 shards under <output>/<project name>/dataset. Progress and errors appear in Monitor & Results."
        )
        output_help.setWordWrap(True)
        local_form.addRow(output_help)
        tabs.addTab(local, "Local")

        maxwell = QWidget()
        maxwell_form = QFormLayout(maxwell)
        maxwell_form.addRow("Host", self._line("hpc.host", "maxwell.desy.de"))
        maxwell_form.addRow("User", self._line("hpc.user", ""))
        maxwell_form.addRow("Remote project path", self._line("hpc.remote_path", ""))
        maxwell_form.addRow("Partition", self._line("hpc.partition", "allgpu"))
        maxwell_form.addRow("GPUs", self._spin("hpc.gpus", 1, 0, 64))
        maxwell_form.addRow("CPUs", self._spin("hpc.cpus", 8, 1, 1024))
        maxwell_form.addRow("Memory", self._line("hpc.memory", "64G"))
        maxwell_form.addRow("Run time", self._line("hpc.time", "24:00:00"))
        maxwell_form.addRow("Remote Python command", self._line("hpc.python_command", "python"))
        maxwell_form.addRow(
            "Job array", self._check("hpc.job_array", True, "Generate HDF5 shards in parallel")
        )
        self.connection_button = QPushButton("Test SSH and remote path")
        self.hpc_prepare_button = QPushButton("Prepare reproducible HPC job")
        self.hpc_submit_button = QPushButton("Upload and submit dependent jobs")
        self.hpc_submit_button.setObjectName("primaryAction")
        reserved = QLabel(
            "Reserved interface: Maxwell SSH upload/submission is disabled for this local demo. Job packaging and Slurm scripts remain exportable."
        )
        reserved.setWordWrap(True)
        maxwell_form.addRow(reserved)
        maxwell_form.addRow(self.connection_button)
        maxwell_form.addRow(self.hpc_prepare_button)
        maxwell_form.addRow(self.hpc_submit_button)
        self.connection_button.setEnabled(False)
        self.hpc_prepare_button.setEnabled(False)
        self.hpc_submit_button.setEnabled(False)
        tabs.addTab(maxwell, "Maxwell")
        ui.runContentLayout.addWidget(tabs)
        self.package_tree = QTextEdit()
        self.package_tree.setReadOnly(True)
        self.package_tree.setPlainText(
            "Prepare a job package to see its path and reproducibility manifest."
        )
        ui.exportContentLayout.addWidget(self.package_tree, 1)
        return page

    def _monitor_page(self) -> QWidget:
        page = QWidget()
        ui = TrainsetMonitorPageView()
        ui.setupUi(page)
        self._monitor_page_ui = ui
        self.job_state = ui.job_state
        self.job_state.setObjectName("jobState")
        self.job_id_label = ui.job_id_label
        self.refresh_job_button = ui.refresh_job_button
        self.sync_results_button = ui.sync_results_button
        self.monitor_splitter = ui.monitor_splitter
        self.job_log = ui.job_log
        self.trainset_log_section = ui.trainsetLogSection
        self.trainset_results_section = ui.trainsetResultsSection
        bind_advanced_section(
            self.trainset_log_section,
            ui.logToggle,
            ui.logDescription,
            ui.logContent,
            ui.logContentLayout,
        )
        bind_parameter_section(
            self.trainset_results_section,
            ui.resultsTitle,
            ui.resultsDescription,
            ui.resultsContent,
            ui.resultsContentLayout,
        )
        self.metrics_table = ui.metrics_table
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.register_model_button = ui.register_model_button
        apply_design_system(page)
        return page
