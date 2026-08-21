"""Prediction presentation ownership and offscreen compatibility tests。"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QTabWidget, QWidget

from main import MainWindow
from src.gimap.features.prediction.presentation.bindings.module_catalog import (
    ModuleCatalogMixin,
)
from src.gimap.features.prediction.presentation.bindings.setup_status import (
    SetupStatusMixin,
)
from src.gimap.features.prediction.presentation.workflow_state import (
    PredictionWorkflowSnapshot,
)
from src.gimap.app import AppContext
from src.gimap.features.prediction.presentation import (
    GisaxsPredictWorkspace,
    PredictCard,
    PredictionViewModel,
    PredictModelLibraryCard,
    build_prediction_controls,
    translate_prediction_controls,
)
from src.gimap.features.prediction.presentation.views import (
    PredictionPageView,
    PredictionWorkspaceView,
)
from src.gimap.integrations.jobs import LocalProcessJobRunner
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)
from ui.components.main_window_components import (
    GisaxsPredictWorkspace as LegacyGisaxsPredictWorkspace,
)
from ui.components.main_window_components import PredictCard as LegacyPredictCard
from ui.components.main_window_components import (
    PredictModelLibraryCard as LegacyPredictModelLibraryCard,
)

ROOT = Path(__file__).resolve().parents[1]
LEGACY_COMPONENTS = ROOT / "ui" / "components" / "main_window_components.py"
GENERATED_MAIN_WINDOW = ROOT / "src" / "gimap" / "app" / "window_view.py"
PRESENTATION_ROOT = ROOT / "src" / "gimap" / "features" / "prediction" / "presentation"
_TEST_APP = None


def _app() -> QApplication:
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


def _context() -> AppContext:
    return AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        jobs=LocalProcessJobRunner(),
    )


def test_module_catalog_ignores_final_event_from_deleted_qt_widget():
    class DeletedQtBase:
        def eventFilter(self, _obj, _event):  # noqa: N802 - Qt signature
            raise RuntimeError("wrapped C/C++ object has been deleted")

    class Binding(ModuleCatalogMixin, DeletedQtBase):
        ui = object()

    assert Binding().eventFilter(object(), None) is False


def test_legacy_component_path_reexports_feature_owned_prediction_classes():
    assert LegacyGisaxsPredictWorkspace is GisaxsPredictWorkspace
    assert LegacyPredictCard is PredictCard
    assert LegacyPredictModelLibraryCard is PredictModelLibraryCard
    assert PredictionViewModel.__module__.startswith("src.gimap.features.prediction.presentation")

    legacy_tree = ast.parse(LEGACY_COMPONENTS.read_text(encoding="utf-8"))
    legacy_class_names = {node.name for node in legacy_tree.body if isinstance(node, ast.ClassDef)}
    assert "GisaxsPredictWorkspace" not in legacy_class_names
    assert "PredictCard" not in legacy_class_names
    assert "PredictModelLibraryCard" not in legacy_class_names


def test_prediction_controls_are_owned_by_feature_factory():
    assert build_prediction_controls.__module__ == (
        "src.gimap.features.prediction.presentation.control_view_factory"
    )

    generated_source = GENERATED_MAIN_WINDOW.read_text(encoding="utf-8")
    generated_tree = ast.parse(generated_source)
    setup = next(
        node
        for node in ast.walk(generated_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "setupUi"
    )
    calls = {
        node.func.id
        for node in ast.walk(setup)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assigned_attributes = {
        target.attr
        for node in ast.walk(setup)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }

    assert "build_prediction_controls" in calls
    assert "gisaxsPredictPage" not in assigned_attributes


def test_prediction_static_controls_and_workspace_are_python_view_owned():
    views = PRESENTATION_ROOT / "views"
    factory_source = (PRESENTATION_ROOT / "control_view_factory.py").read_text(
        encoding="utf-8"
    )
    workspace_source = (PRESENTATION_ROOT / "workspace.py").read_text(
        encoding="utf-8"
    )

    assert PredictionPageView.__module__.endswith("views.prediction_page_view")
    assert PredictionWorkspaceView.__module__.endswith("views.prediction_workspace_view")
    assert "PredictionPageView" in factory_source
    assert "QtWidgets.QPushButton" not in factory_source
    assert len(factory_source.splitlines()) <= 40
    assert {path.name for path in views.glob("*_view.py")} == {
        "distribution_heatmap_dialog_view.py",
        "export_dialog_view.py",
        "multifile_results_widget_view.py",
        "parameter_trend_dialog_view.py",
        "prediction_page_view.py",
        "prediction_workspace_view.py",
    }
    for legacy_section_builder in (
        'ParameterSection("Input"',
        'ParameterSection("Configure"',
        'ParameterSection("Run"',
        'ParameterSection("Results"',
        'ParameterSection("Export"',
        'AdvancedSection("Advanced model sources"',
        'PlotPanel("Preview"',
    ):
        assert legacy_section_builder not in workspace_source


def test_prediction_translation_is_owned_by_feature_presentation():
    assert translate_prediction_controls.__module__ == (
        "src.gimap.features.prediction.presentation.control_view_factory"
    )

    generated_tree = ast.parse(GENERATED_MAIN_WINDOW.read_text(encoding="utf-8"))
    retranslate = next(
        node
        for node in ast.walk(generated_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "retranslateUi"
    )
    plain_calls = {
        node.func.id
        for node in ast.walk(retranslate)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    translated_feature_owners: set[str] = set()
    for node in ast.walk(retranslate):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if not (
            node.func.attr in {"setItemText", "setTabText", "setText"}
            and isinstance(owner, ast.Attribute)
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "self"
        ):
            continue
        if owner.attr.startswith(("gisaxsPredict", "gisaxsImage", "predict2d")):
            translated_feature_owners.add(owner.attr)

    assert "translate_prediction_controls" in plain_calls
    assert not translated_feature_owners


def test_feature_owned_prediction_workspace_preserves_controls_offscreen():
    _app()
    window = MainWindow(_context())
    workspace = window.components.predict_workspace

    assert type(workspace) is GisaxsPredictWorkspace
    assert workspace.prediction_input_section.objectName() == "predictionInputSection"
    assert workspace.prediction_configure_section.objectName() == "predictionConfigureSection"
    assert workspace.prediction_preview_panel.objectName() == "predictionPreviewPanel"
    assert workspace.prediction_run_section.objectName() == "predictionRunSection"
    assert workspace.prediction_results_section.objectName() == "predictionResultsSection"
    assert workspace.prediction_export_section.objectName() == "predictionExportSection"

    assert window.gisaxsPredictChooseGisaxsFileButton.objectName() == (
        "gisaxsPredictChooseGisaxsFileButton"
    )
    assert window.gisaxsPredictPredictButton.objectName() == "gisaxsPredictPredictButton"
    assert window.gisaxsPredictStopButton.objectName() == "gisaxsPredictStopButton"
    assert window.gisaxsPredictShowMultiFileResultsButton.objectName() == (
        "gisaxsPredictShowMultiFileResultsButton"
    )
    assert window.gisaxsPredictImageShowTabWidget.count() == 2
    assert window.gisaxsPredictChooseFolderButton.text() == "Choose Folder"
    assert window.gisaxsPredictStackValue.text() == "1"
    assert window.gisaxsPredictImageShowTabWidget.tabText(0) == "Input preview"
    assert window.gisaxsPredictImageShowTabWidget.tabText(1) == "Prediction result"
    assert window.gisaxsPredictImageShowTabWidget.currentWidget() is window.gisaxsImageTab
    assert window.predictStatusTextBrowser.parent() is (
        workspace.prediction_results_section.content
    )
    toolbar = workspace.workbench_layout.plot_panel.toolbar_widget
    assert window.gisaxsImageExportButton.parent() is toolbar
    assert window.predict2dExportButton.parent() is toolbar
    assert window.gisaxsImageExportButton.isHidden()
    assert window.gisaxsPredictStopButton.isHidden()
    window.close()


def test_prediction_workbench_exposes_three_step_single_and_batch_flow_offscreen():
    app = _app()
    window = MainWindow(_context())
    window.mainWindowWidget.setCurrentIndex(1)
    window.show()
    app.processEvents()
    workspace = window.components.predict_workspace

    assert workspace.workflow_header.STEP_TITLES == (
        "Import data",
        "Import model",
        "Predict",
    )
    assert window.gisaxsPredictWorkspaceSplitter.count() == 2
    assert window.predictionInputModePanel.pages.currentIndex() == 0
    assert window.gisaxsPredictChooseGisaxsFileButton.isVisibleTo(window)
    assert not window.gisaxsPredictChooseFolderButton.isVisibleTo(window)
    assert window.gisaxsPredictShowMultiFileResultsButton.isHidden()
    assert window.predictionTechnicalModelDisclosure.content.isHidden()
    assert window.predictionActivityDisclosure.content.isHidden()
    assert window.gisaxsPredictModelStatusTextLabel.property("modelState") == "idle"

    window.gisaxsPredictMultiFilesRadioButton.setChecked(True)
    app.processEvents()

    assert window.predictionInputModePanel.pages.currentIndex() == 1
    assert not window.gisaxsPredictChooseGisaxsFileButton.isVisibleTo(window)
    assert window.gisaxsPredictChooseFolderButton.isVisibleTo(window)
    assert not window.gisaxsPredictShowMultiFileResultsButton.isHidden()
    assert window.predictionActivityDisclosure.toggle.text() == "Activity log"
    assert window.predictionBatchResultsSection.isVisibleTo(window)
    workspace.input_mode_panel.set_batch_summary(files=10, jobs=3, skipped=1)
    assert window.predictionBatchPlanSummary.text() == (
        "10 files selected · 3 prediction jobs · 1 trailing file skipped"
    )
    window.close()


def test_prediction_workflow_snapshot_is_derived_from_successful_states():
    missing = PredictionWorkflowSnapshot()
    assert missing.active_step == 1
    assert missing.step_states() == ("active", "upcoming", "upcoming")

    input_ready = PredictionWorkflowSnapshot(input_ready=True)
    assert input_ready.active_step == 2
    assert input_ready.step_states() == ("complete", "active", "upcoming")

    stale_result = PredictionWorkflowSnapshot(
        input_ready=True,
        prediction_succeeded=True,
    )
    assert stale_result.step_states() == ("complete", "active", "upcoming")

    ready = PredictionWorkflowSnapshot(
        input_ready=True,
        model_ready=True,
        framework_ready=True,
        prediction_running=True,
    )
    assert ready.step_states() == ("complete", "complete", "running")

    failed = PredictionWorkflowSnapshot(
        input_ready=True,
        model_ready=True,
        framework_ready=True,
        error_step=3,
    )
    assert failed.step_states() == ("complete", "complete", "error")


def test_prediction_tab_navigation_uses_widget_identity_not_visible_text():
    _app()
    tabs = QTabWidget()
    input_page = QWidget()
    result_page = QWidget()
    tabs.addTab(input_page, "Input preview")
    tabs.addTab(result_page, "Prediction result")
    binding = SetupStatusMixin()
    binding.ui = SimpleNamespace(
        gisaxsPredictImageShowTabWidget=tabs,
        gisaxsImageTab=input_page,
        predict2dImageTab=result_page,
    )

    tabs.setCurrentWidget(result_page)
    binding._set_predict_main_tab("GISAXS")
    assert tabs.currentWidget() is input_page
    binding._set_predict_main_tab("Predict-2D")
    assert tabs.currentWidget() is result_page


def test_prediction_layout_modules_do_not_import_workflow_or_scientific_runtimes():
    forbidden_roots = {
        "bornagain",
        "controllers",
        "keras",
        "tensorflow",
    }
    violations: list[str] = []
    for path in (
        PRESENTATION_ROOT / "cards.py",
        PRESENTATION_ROOT / "control_style.py",
        PRESENTATION_ROOT / "control_view_factory.py",
        PRESENTATION_ROOT / "preview_layout.py",
        PRESENTATION_ROOT / "workbench_layout.py",
        PRESENTATION_ROOT / "workflow_components.py",
        PRESENTATION_ROOT / "workflow_state.py",
        PRESENTATION_ROOT / "workspace.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name.split(".", maxsplit=1)[0].casefold() in forbidden_roots:
                    violations.append(f"{path.name}:{node.lineno}: {name}")

    assert not violations, "Prediction layout contains workflow/runtime imports:\n" + "\n".join(
        violations
    )
