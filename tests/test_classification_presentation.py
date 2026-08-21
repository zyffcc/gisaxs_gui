"""Classification presentation ownership and offscreen compatibility tests."""

from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget

from controllers.classification_controller import ClassificationController
from src.gimap.app import AppContext
from src.gimap.app.main_window import MainWindowComponents
from src.gimap.features.classification.bootstrap import create_classification_view_model
from src.gimap.features.classification.presentation.page import (
    STYLE_PATH,
    ClassificationPage,
)
from src.gimap.features.classification.presentation.views import ClassificationPageView
from src.gimap.features.classification.presentation.view_binding import (
    ClassificationViewBinding,
)
from ui.classification_page import ClassificationPage as LegacyClassificationPage
from src.gimap.integrations.jobs import LocalProcessJobRunner
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)
from ui.main_window import Ui_MainWindow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_MAIN_WINDOW = PROJECT_ROOT / "ui" / "main_window.py"
_TEST_APP = None


def _app() -> QApplication:
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


def test_legacy_classification_entry_reexports_feature_owned_page() -> None:
    assert ClassificationController is ClassificationViewBinding
    assert LegacyClassificationPage is ClassificationPage

    legacy_source = (PROJECT_ROOT / "ui" / "classification_page.py").read_text(encoding="utf-8")
    assert "class ClassificationPage" not in legacy_source
    assert len(legacy_source.splitlines()) <= 8


def test_feature_page_owns_its_only_stylesheet_implementation() -> None:
    expected_root = PROJECT_ROOT / "src" / "gimap" / "features" / "classification" / "presentation"
    assert STYLE_PATH == expected_root / "styles" / "classification_page.qss"
    assert STYLE_PATH.is_file()
    assert not (PROJECT_ROOT / "ui" / "styles" / "classification_page.qss").exists()


def test_feature_page_preserves_widgets_signals_steps_and_job_status_offscreen() -> None:
    _app()
    page = ClassificationPage()
    emitted_steps: list[str] = []
    dropped_paths: list[list[str]] = []
    page.stepChanged.connect(emitted_steps.append)
    page.filesDropped.connect(dropped_paths.append)

    assert page.objectName() == "ClassificationPageRoot"
    assert page.acceptDrops()
    assert page.styleSheet()
    assert page.findChild(QStackedWidget, "classificationWorkflowStack") is page.workflowStack
    for object_name in (
        "classificationInputSection",
        "classificationPreviewPanel",
        "classificationConfigureSection",
        "classificationAlgorithmSection",
        "classificationResultsSection",
        "classificationExportSection",
        "classificationLogSection",
        "classificationPageTextBrowser",
    ):
        assert page.findChild(QObject, object_name) is not None

    page.set_step("Results")
    assert page.workflowStack.currentIndex() == 3
    assert emitted_steps == ["Results"]
    page.filesDropped.emit(["one.npy", "two.npy"])
    assert dropped_paths == [["one.npy", "two.npy"]]

    page.runStatusLabel.setText("Training classifiers")
    page.set_job_state("running", progress=42)
    assert page.classification_job_status.state_label.text() == "RUNNING"
    assert page.runStatusLabel.text() == "Training classifiers"
    assert page.taskProgressBar.maximum() == 100
    assert page.taskProgressBar.value() == 42
    page.close()


def test_classification_modern_workflow_uses_action_steps_and_progressive_disclosure() -> None:
    _app()
    page = ClassificationPage()

    assert page.titleLabel.text() == "Classifier workbench"
    assert [
        page.datasetStepButton.text(),
        page.preprocessingStepButton.text(),
        page.algorithmsStepButton.text(),
        page.resultsStepButton.text(),
    ] == [
        "1  Import dataset",
        "2  Preprocess",
        "3  Compare models",
        "4  Results",
    ]
    assert page.scanImportButton.property("classificationPrimaryAction") is True
    assert page.algorithmConfigSplitter.count() == 1
    assert page.classification_algorithm_advanced.parentWidget() is not (
        page.algorithmConfigSplitter
    )
    assert page.preview_empty_state.parentWidget() is page.previewGraphicsView.viewport()

    page.set_step("Preprocessing")
    page.preprocessing_continue_button.click()
    assert page.workflowStack.currentIndex() == 2
    page.close()


def test_page_shell_and_panels_are_owned_by_feature_python_views() -> None:
    views = (
        PROJECT_ROOT
        / "src/gimap/features/classification/presentation/views"
    )
    page_source = (
        PROJECT_ROOT / "src/gimap/features/classification/presentation/page.py"
    ).read_text(encoding="utf-8")

    assert issubclass(ClassificationPage, ClassificationPageView)
    assert "def _build_ui(" not in page_source
    assert "def _build_header(" not in page_source
    assert "def _build_stepper(" not in page_source
    assert "def _build_log_panel(" not in page_source
    assert {path.name for path in views.glob("*_view.py")} == {
        "classification_dataset_panel_view.py",
        "classification_experiment_panel_view.py",
        "classification_inspection_panel_view.py",
        "classification_page_view.py",
        "classification_preprocessing_panel_view.py",
        "classification_results_panel_view.py",
    }
    for old_builder in (
        "_build_dataset_panel",
        "_build_inspection_panel",
        "_build_preprocessing_panel",
        "_build_experiment_panel",
        "_build_results_panel",
    ):
        assert f"def {old_builder}(" not in page_source


def test_feature_page_contains_only_presentation_dependencies() -> None:
    source = (
        PROJECT_ROOT / "src" / "gimap" / "features" / "classification" / "presentation" / "page.py"
    ).read_text(encoding="utf-8")
    imported_modules: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    forbidden = (
        "controllers",
        "global_params",
        "tensorflow",
        "keras",
        "sklearn",
        "src.gimap.features.classification.application",
        "src.gimap.features.classification.domain",
        "src.gimap.features.classification.infrastructure",
    )
    assert not any(module.startswith(forbidden) for module in imported_modules)
    assert "QFileDialog" not in source
    assert "QMessageBox" not in source


def test_application_shell_keeps_only_classification_host() -> None:
    app = _app()
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)

    assert ui.mainWindowWidget.indexOf(ui.classificationPage) == 3
    assert ui.classificationPage.layout() is ui.verticalLayout_23
    assert ui.verticalLayout_23.count() == 0

    source = LEGACY_MAIN_WINDOW.read_text(encoding="utf-8")
    for removed_name in (
        "ClassificationImportGroupBox",
        "classificationPageMainScrollArea",
        "DimensionalityReductionGroupBox",
    ):
        assert removed_name not in source

    context = AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        jobs=LocalProcessJobRunner(),
    )
    assert ui.verticalLayout_23.count() == 0
    with pytest.raises(ValueError, match="injected ClassificationPage"):
        ClassificationViewBinding(
            ui,
            classification_view_model=create_classification_view_model(context),
        )
    for removed_alias in (
        "addClassButton",
        "datasetTable",
        "runComparisonButton",
        "algorithmList",
        "validationMethodCombo",
        "resultsTable",
        "confusionMatrixView",
        "misclassifiedTable",
        "activeModelCombo",
        "predictNewDataButton",
        "ClassificationPanelWidget",
        "ClassificationImportListWidget",
        "ClassificationImportPlusButton",
        "ClassificationImportMinusButton",
        "ClassificationImportImportButton",
        "ClassificationImportClassifyButton",
        "ClassificationImportFolderPathLabel",
        "ClassificationImportFolderPathValue",
        "ClassificationImportRuleLabel",
        "ClassificationImportRuleValue",
        "DimensionalityReductionMethodCombox",
        "DimensionalityReductionTargetDimValue",
        "DimensionalityReductionNNeighborValue",
        "DimensionalityReductionStartButton",
        "DimensionalityReductionShowResultButton",
        "ClassificationMethodCombox",
        "ClassificationKNnnNneighborsLabel",
        "ClassificationKNnnNneighborsValue",
        "ClassificationClassifyButton",
        "ClassificationSaveModelButton",
        "ClassificationLoadModelButton",
        "ClassificationImportTableWidget",
        "ClassificationGraphicsView",
        "classificationPageTextBrowser",
    ):
        assert not hasattr(ui, removed_alias)

    window.close()
    context.jobs.shutdown()
    app.processEvents()


def test_app_composition_installs_classification_page_before_controller_binding() -> None:
    app = _app()

    class ComposedWindow(QMainWindow, Ui_MainWindow):
        pass

    window = ComposedWindow()
    window.setupUi(window)
    context = AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        jobs=LocalProcessJobRunner(),
    )
    window.app_context = context

    components = MainWindowComponents(window)
    page = components.classification_page
    binding = ClassificationViewBinding(
        window,
        classification_view_model=create_classification_view_model(context),
        page=page,
    )
    binding.initialize()

    assert window.mainWindowWidget.indexOf(window.classificationPage) == 3
    assert window.verticalLayout_23.count() == 1
    assert window.classificationWorkspace is page
    assert binding.page is page
    assert not hasattr(window, "ClassificationImportTableWidget")
    assert not hasattr(window, "ClassificationGraphicsView")
    assert not hasattr(window, "classificationPageTextBrowser")
    assert not hasattr(binding, "_install_compatibility_aliases")
    assert not hasattr(binding, "_install_page")
    assert page.qualityStatusLabel.text() == "Waiting for data"
    assert "Add at least two labeled classes" in page.qualityListWidget.item(0).text()

    binding.log("Direct page logging")
    assert "Direct page logging" in page.logTextBrowser.toPlainText()

    window.close()
    context.jobs.shutdown()
    app.processEvents()
