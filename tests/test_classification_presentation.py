"""Classification presentation ownership and offscreen compatibility tests."""

from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QObject, Qt
from PyQt5.QtWidgets import QApplication, QGridLayout, QMainWindow, QStackedWidget

from controllers.classification_controller import ClassificationController
from src.gimap.app import AppContext
from src.gimap.app.main_window import MainWindowComponents
from src.gimap.features.classification.bootstrap import create_classification_view_model
from src.gimap.features.classification.presentation.page import (
    STYLE_PATH,
    ClassificationPage,
)
from src.gimap.features.classification.presentation.components import EmbeddingScatterView
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
        "classificationExploreSection",
        "classificationApplySection",
        "classificationExportSection",
        "classificationLogSection",
        "classificationPageTextBrowser",
    ):
        assert page.findChild(QObject, object_name) is not None

    page.set_step("Apply")
    assert page.workflowStack.currentIndex() == 4
    page.set_step("Train")
    assert page.workflowStack.currentIndex() == 3
    assert emitted_steps == ["Apply", "Train"]
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

    assert page.titleLabel.text() == "Classifier"
    assert page.classificationWorkflowHeader.property("classificationWorkflowHeader") is True
    assert [
        page.datasetStepButton.text(),
        page.preprocessingStepButton.text(),
        page.algorithmsStepButton.text(),
        page.resultsStepButton.text(),
        page.applyStepButton.text(),
    ] == [
        "Data",
        "Prepare",
        "Explore",
        "Train",
        "Apply",
    ]
    assert page.runEmbeddingButton.property("classificationPrimaryAction") is True
    assert page.algorithmConfigSplitter.count() == 1
    assert page.classification_algorithm_advanced.parentWidget() is not (
        page.algorithmConfigSplitter
    )
    assert page.preview_empty_state.parentWidget() is page.previewGraphicsView.viewport()

    page.set_step("Prepare")
    page.preprocessing_continue_button.click()
    assert page.workflowStack.currentIndex() == 2
    assert page.embeddingScatterView.objectName() == "embeddingScatterView"
    assert page.resultTabs.count() == 4
    assert page.predictionTable.parent() is page._apply_panel_ui.predictionTable.parent()
    page.close()


def test_classification_workflow_progress_is_independent_from_navigation() -> None:
    _app()
    page = ClassificationPage()

    states = {step.key: step.property("workflowState") for step in page.classificationWorkflowHeader.steps}
    assert states == {
        "Data": "available",
        "Prepare": "blocked",
        "Explore": "blocked",
        "Train": "blocked",
        "Apply": "blocked",
    }

    page.set_workflow_step_state("Explore", "complete", "24 accepted")
    page.set_step("Apply")
    explore = next(
        step for step in page.classificationWorkflowHeader.steps if step.key == "Explore"
    )
    apply = next(
        step for step in page.classificationWorkflowHeader.steps if step.key == "Apply"
    )
    assert explore.property("workflowState") == "complete"
    assert explore.property("workflowSelected") is False
    assert apply.property("workflowSelected") is True
    assert explore.message_label.text() == "24 accepted"
    page.close()


@pytest.mark.parametrize("size", [(1280, 800), (1440, 900), (1920, 1080)])
def test_classification_workflow_has_no_page_level_horizontal_overflow(size) -> None:
    app = _app()
    page = ClassificationPage()
    page.resize(*size)
    page.show()
    app.processEvents()

    for step, prefix in (
        ("Data", "dataset"),
        ("Prepare", "preprocessing"),
        ("Explore", "algorithms"),
        ("Train", "results"),
        ("Apply", "apply"),
    ):
        page.set_step(step)
        app.processEvents()
        area = getattr(page, f"{prefix}StepScrollArea")
        assert area.horizontalScrollBar().maximum() == 0

    assert all(
        button.width() > 0
        for button in (
            page.newSessionButton,
            page.loadSessionButton,
            page.saveSessionButton,
            page.helpButton,
        )
    )
    page.close()


def test_classification_small_screen_uses_dense_header_and_side_by_side_workspaces() -> None:
    app = _app()
    page = ClassificationPage()
    page.resize(1280, 800)
    page.show()
    app.processEvents()

    assert page._responsive_mode == "medium"
    assert page.datasetInspectionSplitter.orientation() == Qt.Horizontal
    assert page.explorationSplitter.orientation() == Qt.Horizontal
    assert not page.classificationWorkflowHeader.subtitle_label.isVisible()
    assert not page.classificationWorkflowHeader.mode_button.isVisible()
    assert not page._dataset_panel_ui.sectionTitle.isVisible()
    assert not page._inspection_panel_ui.sectionTitle.isVisible()
    assert isinstance(page._dataset_panel_ui.datasetActionsLayout, QGridLayout)
    assert page._dataset_panel_ui.datasetActionsLayout.rowCount() == 2
    page.close()


def test_embedding_scatter_supports_linked_multi_selection_offscreen() -> None:
    _app()
    scatter = EmbeddingScatterView()
    changes = []
    scatter.selectedSampleIdsChanged.connect(changes.append)
    scatter.set_points(
        [[0.0, 0.0], [1.0, 0.5], [0.5, 1.0]],
        ["one", "two", "three"],
        ["#2563eb", "#16a34a", "#dc2626"],
        ["one.npy", "two.npy", "three.npy"],
    )

    scatter.select_sample_ids(["one", "three"])

    assert set(scatter.selected_sample_ids()) == {"one", "three"}
    assert set(changes[-1]) == {"one", "three"}
    scatter.clear_selection()
    assert scatter.selected_sample_ids() == []
    scatter.close()


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
        "classification_apply_panel_view.py",
        "classification_dataset_panel_view.py",
        "classification_experiment_panel_view.py",
        "classification_exploration_panel_view.py",
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
    assert "Labels are optional until model training" in page.qualityListWidget.item(0).text()
    assert page.stateBadgeLabel.text() == "Waiting for data"
    assert {
        step.key: step.property("workflowState")
        for step in page.classificationWorkflowHeader.steps
    } == {
        "Data": "available",
        "Prepare": "blocked",
        "Explore": "blocked",
        "Train": "blocked",
        "Apply": "blocked",
    }

    binding.log("Direct page logging")
    assert "Direct page logging" in page.logTextBrowser.toPlainText()

    window.close()
    context.jobs.shutdown()
    app.processEvents()
