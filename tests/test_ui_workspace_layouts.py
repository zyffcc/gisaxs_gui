import ast
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from main import MainWindow
from src.gimap.app import AppContext
from src.gimap.integrations.jobs import LocalProcessJobRunner
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)
from ui.classification_page import ClassificationPage
from ui.format_converter_dialog import ConversionProgressDialog, FormatConverterDialog
from ui.geometry_calibration_dialog import GeometryCalibrationDialog
from ui.trainset_build_page import TrainsetBuildPage


_TEST_APP = None
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _app():
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


def _context():
    return AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        jobs=LocalProcessJobRunner(),
    )


def test_main_window_composition_is_owned_by_app_composition_boundary() -> None:
    from main import MainWindowComponents
    from src.gimap.app.presentation.navigation import NavigationSidebar
    from src.gimap.app.presentation.shell import ContentStack, MainShell
    from ui.components.main_window_components import (
        MainWindowComponents as LegacyMainWindowComponents,
    )
    from ui.components.main_window_components import (
        NavigationSidebar as LegacyNavigationSidebar,
    )
    from ui.components.main_window_components import ContentStack as LegacyContentStack
    from ui.components.main_window_components import MainShell as LegacyMainShell

    assert LegacyMainWindowComponents is MainWindowComponents
    assert MainWindowComponents.__module__ == "src.gimap.app.main_window"
    assert LegacyNavigationSidebar is NavigationSidebar
    assert LegacyContentStack is ContentStack
    assert LegacyMainShell is MainShell

    legacy_source = (PROJECT_ROOT / "ui" / "components" / "main_window_components.py").read_text(
        encoding="utf-8"
    )
    legacy_tree = ast.parse(legacy_source)
    assert not any(isinstance(node, ast.ClassDef) for node in legacy_tree.body)
    assert len(legacy_source.splitlines()) <= 70

    main_source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
    assert "src.gimap.app.main_window" in main_source
    assert "from ui.components import MainWindowComponents" not in main_source

    composition_source = (PROJECT_ROOT / "src" / "gimap" / "app" / "main_window.py").read_text(
        encoding="utf-8"
    )
    composition_tree = ast.parse(composition_source)
    assert not any(
        isinstance(node, ast.ClassDef)
        and node.name in {"ContentStack", "MainShell", "NavigationSidebar"}
        for node in composition_tree.body
    )


def test_format_converter_uses_six_stage_information_sections():
    _app()
    dialog = FormatConverterDialog(app_context=_context())

    assert dialog.format_input_section.title_label.text() == "Input"
    assert dialog.format_configure_section.title_label.text() == "Configure"
    assert dialog.format_preview_panel.title_label.text() == "Preview"
    assert dialog.format_output_advanced.is_expanded() is False
    assert dialog.format_run_section.title_label.text() == "Run, Results & Export"
    assert dialog.input_tree.parent() is dialog.format_input_section.content
    assert dialog.selection_table.parent().parent() is not None
    dialog.close()


def test_format_converter_progress_uses_shared_job_status():
    _app()
    dialog = ConversionProgressDialog("/tmp")
    dialog.job_status.set_state("running", "Working", progress=0.5)

    assert dialog.bar is dialog.job_status.progress_bar
    assert dialog.pause_button is dialog.job_status.pause_button
    assert dialog.cancel_button is dialog.job_status.cancel_button
    assert dialog.bar.value() == 500
    dialog.running = False
    dialog.close()


def test_calibration_layout_separates_input_preview_results_and_export():
    _app()
    dialog = GeometryCalibrationDialog(app_context=_context())

    assert dialog.calibration_input_section.title_label.text() == "Input"
    assert dialog.calibration_advanced_section.is_expanded() is False
    assert dialog.calibration_run_section.title_label.text() == "Run"
    assert dialog.calibration_preview_panel.title_label.text() == "Preview"
    assert dialog.calibration_results_section.title_label.text() == "Results"
    assert dialog.calibration_export_section.title_label.text() == "Export"
    assert dialog.progress is dialog.job_status.progress_bar
    assert dialog.stage_label is dialog.job_status.message_label
    dialog.close()


def test_calibration_manual_refinement_uses_advanced_section_without_value_reset():
    _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.manual_x.setValue(123.5)

    dialog.calibration_manual_section.set_expanded(True)
    dialog.calibration_manual_section.set_expanded(False)

    assert dialog.manual_x.value() == 123.5
    assert dialog.manual_group.isChecked() is False
    dialog.close()


def test_trainset_layout_maps_existing_steps_to_shared_workspace_sections():
    _app()
    page = TrainsetBuildPage()

    assert page.trainset_input_section.title_label.text() == "Input"
    assert page.trainset_configure_section.title_label.text() == "Configure"
    assert page.trainset_advanced_section.is_expanded() is False
    assert page.trainset_design_preview_panel.title_label.text() == "Preview"
    assert page.trainset_preview_run_section.title_label.text() == "Run"
    assert page.trainset_preview_advanced_section.is_expanded() is False
    assert page.trainset_model_run_section.title_label.text() == "Run"
    assert page.trainset_export_section.title_label.text() == "Export"
    assert page.trainset_results_section.title_label.text() == "Results"
    page.close()


def test_trainset_job_status_preserves_legacy_controller_aliases_and_percent_range():
    _app()
    page = TrainsetBuildPage()

    page.set_local_job_status("running", "Generating shard 1", 25)

    assert page.local_progress is page.trainset_job_status.progress_bar
    assert page.local_activity is page.trainset_job_status.message_label
    assert page.local_progress.maximum() == 100
    assert page.local_progress.value() == 25
    assert page.trainset_job_status.state_label.text() == "RUNNING"
    assert page.local_activity.text() == "Generating shard 1"
    page.close()


def test_fitting_layout_uses_shared_six_stage_sections_without_replacing_actions():
    _app()
    window = MainWindow(_context())
    workspace = window.components.fitting_workspace

    assert workspace.fitting_input_section.title_label.text() == "Input"
    assert workspace.fitting_configure_section.title_label.text() == "Configure"
    assert workspace.fitting_preview_panel.title_label.text() == "Preview"
    assert workspace.fitting_run_section.title_label.text() == "Fit"
    assert workspace.fitting_results_panel.title_label.text() == "Results"
    assert workspace.fitting_export_section.title_label.text() == "Export"
    assert workspace.fitting_advanced_section.is_expanded() is False
    assert window.FittingExportButton.parent() is workspace.fitting_export_section.content
    assert window.fitExportPlotButton.parent() is workspace.fitting_export_section.content
    window.close()


def test_prediction_layout_uses_modern_three_step_workbench_and_original_actions():
    _app()
    window = MainWindow(_context())
    workspace = window.components.predict_workspace

    assert workspace.prediction_input_section.title_label.text() == "1. Import data"
    assert workspace.prediction_configure_section.title_label.text() == "2. Import model"
    assert workspace.prediction_preview_panel.title_label.text() == "Preview"
    assert workspace.prediction_run_section.title_label.text() == "3. Predict"
    assert workspace.prediction_results_section.title_label.text() == "Results"
    assert workspace.prediction_advanced_section.is_expanded() is False
    assert window.gisaxsPredictWorkspaceSplitter.count() == 2
    assert window.predictionPlotPanel.title_label.text() == "Prediction canvas"
    assert window.predictionBatchResultsSection.title_label.text() == "Batch results"
    assert workspace.prediction_export_section.isHidden()
    assert window.predictStatusTextBrowser.parent() is workspace.prediction_results_section.content
    toolbar = workspace.workbench_layout.plot_panel.toolbar_widget
    assert window.gisaxsImageExportButton.parent() is toolbar
    assert window.predict2dExportButton.parent() is toolbar
    window.close()


def test_classification_layout_uses_shared_stages_advanced_sections_and_job_status():
    _app()
    page = ClassificationPage()

    assert page.classification_input_section.title_label.text() == "Input"
    assert page.classification_configure_section.title_label.text() == "Configure"
    assert page.classification_preview_panel.title_label.text() == "Preview"
    assert page.classification_run_section.title_label.text() == "Run"
    assert page.classification_results_section.title_label.text() == "Results"
    assert page.classification_export_section.title_label.text() == "Export"
    assert page.classification_preprocessing_advanced.is_expanded() is False
    assert page.classification_algorithm_advanced.is_expanded() is False
    assert page.classification_log_section.is_expanded() is False

    page.set_job_state("running", progress=40)
    assert page.taskProgressBar is page.classification_job_status.progress_bar
    assert page.runStatusLabel is page.classification_job_status.message_label
    assert page.taskProgressBar.maximum() == 100
    assert page.taskProgressBar.value() == 40
    assert page.classification_job_status.state_label.text() == "RUNNING"
    assert page.exportResultsButton.parent() is page.classification_export_section.content
    page.close()


def test_waxs_layout_uses_shared_stages_basic_advanced_and_job_status():
    _app()
    window = MainWindow(_context())
    page = window.components.waxs_page

    assert window.waxsPageIndex == 4
    assert window.mainWindowWidget.count() == 5
    assert window.mainWindowWidget.widget(4) is page
    assert not hasattr(window, "waxsPageHost")

    assert page.waxs_input_section.title_label.text() == "Load data"
    assert page.waxs_configure_section.title_label.text() == "Cut and integrate"
    assert page.waxs_preview_panel.title_label.text() == "Preview"
    assert page.waxs_run_section.title_label.text() == "Batch processing"
    assert page.waxs_results_section.title_label.text() == "Results"
    assert page.waxs_export_section.title_label.text() == "Export"
    assert page.waxs_advanced_section.is_expanded() is True
    assert page.waxs_workflow_tabs.count() == 3
    assert [page.tabs.tabText(index) for index in range(page.tabs.count())] == [
        "ROI / Cut",
        "1D Integration",
    ]
    assert [page.advanced_tabs.tabText(index) for index in range(page.advanced_tabs.count())] == [
        "Display",
        "Mask",
        "Geometry",
    ]

    page.set_job_state("running", "Processing frame", progress=35)
    assert page.progress is page.waxs_job_status.progress_bar
    assert page.status_label is page.waxs_job_status.message_label
    assert page.progress.maximum() == 100
    assert page.progress.value() == 35
    assert page.export_button.parent() is page.waxs_export_section.content
    assert page.export_1d_button.parent() is page.waxs_export_section.content
    window.close()
