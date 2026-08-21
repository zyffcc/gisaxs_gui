"""Trainset presentation ownership and offscreen compatibility tests."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QApplication, QMainWindow

from controllers.trainset_controller import TrainsetController
from src.gimap.app import AppContext
from src.gimap.app.main_window import MainWindowComponents
from src.gimap.features.trainset.domain import PHYSICAL_BACKGROUND_PARAMETERS
from src.gimap.features.trainset.presentation.page import (
    ArrayCanvas,
    HistogramWidget,
    ParameterCoverageWidget,
    TrainsetBuildPage,
)
from src.gimap.features.trainset.presentation.views import (
    TrainsetModelPageView,
    TrainsetMonitorPageView,
    TrainsetPageView,
    TrainsetPreviewPageView,
    TrainsetRunPageView,
)
from src.gimap.features.trainset.presentation.view_binding import TrainsetViewBinding
from trainset.config import PHYSICAL_BACKGROUND_PARAMETERS as LegacyBackgroundParameters
from src.gimap.integrations.jobs import LocalProcessJobRunner
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)
from ui.trainset_build_page import ArrayCanvas as LegacyArrayCanvas
from ui.trainset_build_page import HistogramWidget as LegacyHistogramWidget
from ui.trainset_build_page import (
    ParameterCoverageWidget as LegacyParameterCoverageWidget,
)
from ui.trainset_build_page import TrainsetBuildPage as LegacyTrainsetBuildPage
from ui.main_window import Ui_MainWindow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_MAIN_WINDOW = PROJECT_ROOT / "ui" / "main_window.py"
_TEST_APP = None


class SimulationStub:
    def simulate(self, request):
        raise AssertionError("The presentation host test must not run a simulation")


class TrainsetViewModelStub:
    def default_config(self):
        return {"schema_version": 2}

    def validate_model_contract(self, request):
        raise AssertionError("The presentation host test must not validate a model")

    def generate_preview(self, request, *, on_progress=None):
        raise AssertionError("The presentation host test must not generate a preview")

    def simulate_what_if(self, request):
        raise AssertionError("The presentation host test must not simulate")


def _app() -> QApplication:
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


def test_legacy_trainset_entry_reexports_feature_owned_view_classes() -> None:
    assert TrainsetController is TrainsetViewBinding
    assert LegacyTrainsetBuildPage is TrainsetBuildPage
    assert LegacyArrayCanvas is ArrayCanvas
    assert LegacyHistogramWidget is HistogramWidget
    assert LegacyParameterCoverageWidget is ParameterCoverageWidget

    legacy_source = (PROJECT_ROOT / "ui" / "trainset_build_page.py").read_text(encoding="utf-8")
    assert "class TrainsetBuildPage" not in legacy_source
    assert len(legacy_source.splitlines()) <= 18


def test_physical_background_definitions_have_one_domain_owner() -> None:
    assert LegacyBackgroundParameters is PHYSICAL_BACKGROUND_PARAMETERS
    assert len(PHYSICAL_BACKGROUND_PARAMETERS) == 18
    assert PHYSICAL_BACKGROUND_PARAMETERS[0]["key"] == "target_fraction"
    assert PHYSICAL_BACKGROUND_PARAMETERS[-1]["key"] == "blur_sigma_px"


def test_feature_page_preserves_steps_sections_signals_and_status_offscreen() -> None:
    _app()
    page = TrainsetBuildPage()
    selected_steps: list[int] = []
    mask_regions: list[tuple[str, dict]] = []
    page.step_changed.connect(selected_steps.append)
    page.mask_region_created.connect(lambda kind, value: mask_regions.append((kind, value)))

    assert page.objectName() == "freshTrainsetBuildPage"
    assert page.styleSheet()
    assert page.STEPS == (
        "Dataset Design",
        "Local Preview",
        "Model Design",
        "Local Run",
        "Monitor & Results",
    )
    for object_name in (
        "pageTitle",
        "pageSubtitle",
        "validationBadge",
        "trainsetStepList",
        "designStageTabs",
        "designPreviewCard",
        "jobState",
    ):
        assert page.findChild(QObject, object_name) is not None
    assert page.trainset_input_section.title_label.text() == "Input"
    assert page.trainset_configure_section.title_label.text() == "Configure"
    assert page.trainset_design_preview_panel.title_label.text() == "Preview"
    assert page.trainset_run_section.title_label.text() == "Run"
    assert page.trainset_results_section.title_label.text() == "Results"
    assert page.trainset_export_section.title_label.text() == "Export"
    assert isinstance(page._preview_page_ui, TrainsetPreviewPageView)
    assert isinstance(page._model_page_ui, TrainsetModelPageView)
    assert isinstance(page._run_page_ui, TrainsetRunPageView)
    assert isinstance(page._monitor_page_ui, TrainsetMonitorPageView)
    for object_name in (
        "trainsetPreviewPage",
        "trainsetModelPage",
        "trainsetRunPage",
        "trainsetMonitorPage",
    ):
        assert page.findChild(QObject, object_name) is not None

    page.step_list.setCurrentRow(1)
    assert page.stack.currentIndex() == 1
    assert selected_steps == [1]
    page.full_detector_canvas.region_created.emit("rectangle", {"width": 12})
    assert mask_regions == [("rectangle", {"width": 12})]
    page.full_detector_canvas.set_data(np.ones((4, 5), dtype=np.float32))
    assert page.full_detector_canvas.image.shape == (4, 5)

    page.set_local_job_status("running", "Generating shard 1", 25)
    assert page.local_progress is page.trainset_job_status.progress_bar
    assert page.local_activity is page.trainset_job_status.message_label
    assert page.local_progress.maximum() == 100
    assert page.local_progress.value() == 25
    assert page.trainset_job_status.state_label.text() == "RUNNING"
    for button in (
        page.validate_button,
        page.load_button,
        page.save_button,
        page.preview_button,
        page.prepare_button,
        page.local_generate_button,
        page.local_train_button,
    ):
        assert button.shortcut().toString() == ""
    page.close()


def test_trainset_modern_workflow_groups_design_and_contextual_actions() -> None:
    _app()
    page = TrainsetBuildPage()

    assert page.pageTitle.text() == "Trainset builder"
    assert [
        page.dataset_configuration_tabs.tabText(index)
        for index in range(page.dataset_configuration_tabs.count())
    ] == ["Geometry + ROI", "Particle population", "Sampling + files"]
    assert page.trainset_action_hint.text().startswith("Validate the detector")
    assert page.validate_button.property("trainsetPrimaryAction") is True
    assert page.prepare_button.isHidden()

    page.step_list.setCurrentRow(3)

    assert not page.prepare_button.isHidden()
    assert page.prepare_button.property("trainsetPrimaryAction") is True
    assert page.validate_button.isHidden()
    assert page.trainset_action_hint.text() == (
        "Run locally or export a portable job package."
    )
    page.close()


def test_feature_page_has_no_controller_runtime_or_io_dependencies() -> None:
    source = (PROJECT_ROOT / "src/gimap/features/trainset/presentation/page.py").read_text(
        encoding="utf-8"
    )
    imported_modules: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    forbidden = (
        "controllers",
        "core.global_params",
        "tensorflow",
        "keras",
        "bornagain",
        "trainset.config",
        "trainset.generator",
        "trainset.backends",
        "src.gimap.features.trainset.infrastructure",
    )
    assert not any(module.startswith(forbidden) for module in imported_modules)
    for name in ("QFileDialog", "QMessageBox"):
        assert name not in source


def test_trainset_workflow_shell_is_owned_by_python_views() -> None:
    source = (PROJECT_ROOT / "src/gimap/features/trainset/presentation/page.py").read_text(
        encoding="utf-8"
    )
    views = (
        PROJECT_ROOT
        / "src/gimap/features/trainset/presentation/views"
    )

    assert issubclass(TrainsetBuildPage, TrainsetPageView)
    assert "def _build(self)" not in source
    assert {path.name for path in views.glob("*_view.py")} == {
        "dataset_page_view.py",
        "model_page_view.py",
        "monitor_page_view.py",
        "page_view.py",
        "preview_page_view.py",
        "run_page_view.py",
    }


def test_application_shell_keeps_only_trainset_host() -> None:
    app = _app()
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)

    assert ui.mainWindowWidget.indexOf(ui.trainsetBuildPage) == 0
    assert ui.trainsetBuildPage.layout() is ui.verticalLayout_6
    assert ui.verticalLayout_6.count() == 0

    source = LEGACY_MAIN_WINDOW.read_text(encoding="utf-8")
    for removed_name in (
        "beamParametersBox",
        "sampleParametersParticleStackedWidget",
        "trainsetGenerateStackedWidget",
        "preProcessingBox",
    ):
        assert removed_name not in source

    with pytest.raises(ValueError, match="injected TrainsetBuildPage"):
        TrainsetViewBinding(ui, simulation_port=SimulationStub())

    window.close()
    app.processEvents()


def test_app_composition_installs_trainset_page_before_controller_binding() -> None:
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
    page = components.trainset_page
    binding = TrainsetViewBinding(
        window,
        simulation_port=SimulationStub(),
        trainset_view_model=TrainsetViewModelStub(),
        page=page,
        project_root=PROJECT_ROOT,
    )

    assert window.mainWindowWidget.indexOf(window.trainsetBuildPage) == 0
    assert window.verticalLayout_6.count() == 1
    assert window.trainsetWorkspace is page
    assert binding.page is page
    assert not hasattr(binding, "_replace_legacy_page")
    assert not hasattr(binding, "_legacy_page_widgets")

    window.close()
    context.jobs.shutdown()
    app.processEvents()
