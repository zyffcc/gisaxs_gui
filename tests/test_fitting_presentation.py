"""Fitting presentation ownership and offscreen compatibility tests。"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from matplotlib.collections import QuadMesh

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtTest import QTest

from main import MainWindow
from src.gimap.app import AppContext
from src.gimap.features.fitting.presentation import (
    CutLineCard,
    DetectorSetupPanel,
    FittingControlsCard,
    FittingDataExportDialog,
    FittingPlotControlsCard,
    FittingViewModel,
    FittingViewBinding,
    GisaxsFittingWorkspace,
    GisaxsInputCard,
    ModelParameterCard,
    PlotPreviewCard,
    build_fitting_controls,
    translate_fitting_controls,
)
from src.gimap.features.fitting.presentation.views import (
    FittingPageView,
    FittingWorkspaceView,
    IndependentFitWindowView,
    IndependentImageWindowView,
    InSituSeriesPageView,
)
from src.gimap.features.fitting.application import (
    FittingImageCalculations,
    ScatteringFileData,
    SingleAnalysisRecipeSnapshot,
)
from src.gimap.features.fitting.domain import InSituFittingPolicy, InSituTrackingPolicy
from src.gimap.features.fitting.presentation.view_binding import (
    IndependentFitWindow,
    IndependentMatplotlibWindow,
)
from src.gimap.features.fitting.presentation.bindings.detector_configuration import (
    DetectorConfigurationMixin,
)
from src.gimap.features.fitting.presentation.bindings.particle_connections import (
    ParticleConnectionsMixin,
)
from src.gimap.features.fitting.presentation.workflow_state import (
    complete_workflow_step,
    initial_workflow_state,
)
from src.gimap.features.fitting.presentation.state import (
    CurveViewState,
    DetectorDisplayState,
)
from controllers.fitting_controller import FittingController
from src.gimap.integrations.jobs import LocalProcessJobRunner
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)
from ui.components.main_window_components import (
    CutLineCard as LegacyCutLineCard,
)
from ui.components.main_window_components import (
    FittingControlsCard as LegacyFittingControlsCard,
)
from ui.components.main_window_components import (
    GisaxsFittingWorkspace as LegacyGisaxsFittingWorkspace,
)
from ui.components.main_window_components import (
    GisaxsInputCard as LegacyGisaxsInputCard,
)


ROOT = Path(__file__).resolve().parents[1]
LEGACY_COMPONENTS = ROOT / "ui" / "components" / "main_window_components.py"
GENERATED_MAIN_WINDOW = ROOT / "src" / "gimap" / "app" / "window_view.py"
PRESENTATION_ROOT = ROOT / "src" / "gimap" / "features" / "fitting" / "presentation"
_TEST_APP = None

MIGRATED_CLASSES = {
    "CardFrame",
    "CutLineCard",
    "DetectorPreviewCard",
    "FittingControlsCard",
    "FittingPlotControlsCard",
    "FittingRegionControl",
    "GisaxsFittingWorkspace",
    "GisaxsInputCard",
    "ModelParameterCard",
    "NoWheelDoubleSpinBox",
    "ParticleOptionsLayout",
    "PlotCanvasArea",
    "PlotOptionsControl",
    "PlotPreviewCard",
    "PlotSamplingControl",
    "SectionCard",
    "StatusCard",
}


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


def test_legacy_component_path_reexports_feature_owned_fitting_classes():
    assert FittingController is FittingViewBinding
    assert LegacyGisaxsFittingWorkspace is GisaxsFittingWorkspace
    assert LegacyGisaxsInputCard is GisaxsInputCard
    assert LegacyCutLineCard is CutLineCard
    assert LegacyFittingControlsCard is FittingControlsCard
    assert FittingViewModel.__module__.startswith("src.gimap.features.fitting.presentation")

    legacy_tree = ast.parse(LEGACY_COMPONENTS.read_text(encoding="utf-8"))
    legacy_class_names = {node.name for node in legacy_tree.body if isinstance(node, ast.ClassDef)}
    assert MIGRATED_CLASSES.isdisjoint(legacy_class_names)


def test_fitting_controls_are_owned_by_feature_factory():
    assert build_fitting_controls.__module__ == (
        "src.gimap.features.fitting.presentation.control_view_factory"
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

    assert "build_fitting_controls" in calls
    assert "gisaxsFittingPage" not in assigned_attributes
    assert "QRangeSlider" not in generated_source


def test_fitting_static_controls_and_workspace_are_python_view_owned():
    views = PRESENTATION_ROOT / "views"
    factory_source = (PRESENTATION_ROOT / "control_view_factory.py").read_text(
        encoding="utf-8"
    )
    workspace_source = (PRESENTATION_ROOT / "workspace.py").read_text(
        encoding="utf-8"
    )

    assert FittingPageView.__module__.endswith("views.fitting_page_view")
    assert FittingWorkspaceView.__module__.endswith("views.fitting_workspace_view")
    assert "FittingPageView" in factory_source
    assert "QtWidgets.QPushButton" not in factory_source
    assert len(factory_source.splitlines()) <= 40
    assert {path.name for path in views.glob("*_view.py")} == {
        "detector_parameters_dialog_view.py",
        "fitting_page_view.py",
        "fitting_workspace_view.py",
        "independent_fit_window_view.py",
        "independent_image_window_view.py",
        "insitu_series_page_view.py",
    }
    for legacy_section_builder in (
        'ParameterSection("Input"',
        'ParameterSection("Configure"',
        'ParameterSection("Run"',
        'ParameterSection("Export"',
        'AdvancedSection("Advanced model configuration"',
        'AdvancedSection("Advanced plot controls"',
        'AdvancedSection("Log"',
        'PlotPanel("Preview"',
        'PlotPanel("Results"',
    ):
        assert legacy_section_builder not in workspace_source


def test_independent_plot_windows_use_python_view_shells() -> None:
    class ViewModelStub:
        @staticmethod
        def get_setting(_section, _key, default=None):
            return default

        @staticmethod
        def set_setting(_section, _key, _value):
            return None

        @staticmethod
        def save_settings():
            return None

    app = _app()
    fit_window = IndependentFitWindow()
    image_window = IndependentMatplotlibWindow(fitting_view_model=ViewModelStub())

    assert isinstance(fit_window, IndependentFitWindowView)
    assert isinstance(image_window, IndependentImageWindowView)
    assert fit_window.q_unit_combo.currentData() == "nm"
    assert fit_window.y_range_combo.currentData() == "all"
    assert image_window.centralWidget() is image_window.centralwidget

    fit_window.close()
    image_window.close()
    app.processEvents()


def test_independent_windows_project_the_same_typed_display_state() -> None:
    class ViewModelStub:
        def __init__(self):
            self.values = {}

        def get_setting(self, section, key, default=None):
            return self.values.get((section, key), default)

        def set_setting(self, section, key, value):
            self.values[(section, key)] = value

        @staticmethod
        def save_settings():
            return None

    app = _app()
    curve_window = IndependentFitWindow()
    image_window = IndependentMatplotlibWindow(fitting_view_model=ViewModelStub())
    curve_state = CurveViewState(
        q_mode="fold",
        layer_mode="data",
        log_x=True,
        log_y=True,
        normalize=True,
        q_unit="angstrom",
        y_range="experimental",
    )
    curve_window.set_curve_view_state(curve_state)
    assert curve_window.current_curve_view_state() == curve_state
    emitted_curve_states = []
    curve_window.view_state_changed.connect(emitted_curve_states.append)
    curve_window.q_view_combo.setCurrentIndex(
        curve_window.q_view_combo.findData("positive")
    )
    assert emitted_curve_states[-1].q_mode == "positive"

    detector_state = DetectorDisplayState(
        log_intensity=False,
        auto_scale=False,
        vmin=2.5,
        vmax=42.0,
        colormap="magma",
        show_cut_region=False,
        show_center=True,
        show_q_axis=True,
    )
    image_window.set_detector_display_state(detector_state)
    assert image_window.current_detector_display_state() == detector_state
    emitted_detector_states = []
    image_window.display_state_changed.connect(emitted_detector_states.append)
    image_window.log_action.setChecked(True)
    assert emitted_detector_states[-1].log_intensity

    curve_window.close()
    image_window.close()
    app.processEvents()


def test_legacy_range_slider_path_reexports_feature_owner():
    from src.gimap.features.fitting.presentation.range_slider import (
        QRangeSlider as FeatureRangeSlider,
    )
    from utils.widgets.qtrangeslider import QRangeSlider as LegacyRangeSlider

    assert LegacyRangeSlider is FeatureRangeSlider


def test_fitting_control_factory_has_no_workflow_or_runtime_imports():
    path = PRESENTATION_ROOT / "control_view_factory.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_roots = {
        "bornagain",
        "controllers",
        "keras",
        "tensorflow",
        "infrastructure",
    }
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports = [node.module]
        else:
            continue
        imported_roots.update(name.split(".", maxsplit=1)[0].casefold() for name in imports)

    assert imported_roots.isdisjoint(forbidden_roots)


def test_fitting_translation_is_owned_by_feature_presentation():
    assert translate_fitting_controls.__module__ == (
        "src.gimap.features.fitting.presentation.control_view_factory"
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
            node.func.attr in {"setItemText", "setText", "setTitle"}
            and isinstance(owner, ast.Attribute)
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "self"
        ):
            continue
        if owner.attr.startswith(("Fitting", "fit", "gisaxsInput")):
            translated_feature_owners.add(owner.attr)

    retranslate_source = ast.get_source_segment(
        GENERATED_MAIN_WINDOW.read_text(encoding="utf-8"), retranslate
    )
    assert "translate_fitting_controls" in plain_calls
    assert not translated_feature_owners
    assert "self.pushButton.setText" not in (retranslate_source or "")


def test_feature_owned_fitting_workspace_preserves_controls_and_defaults_offscreen():
    _app()
    window = MainWindow(_context())
    workspace = window.components.fitting_workspace

    assert type(workspace) is GisaxsFittingWorkspace
    assert type(workspace.fixed_controls_stack.findChild(GisaxsInputCard)) is GisaxsInputCard
    assert type(workspace.fixed_controls_stack.findChild(CutLineCard)) is CutLineCard
    assert (
        type(workspace.fixed_controls_stack.findChild(FittingControlsCard)) is FittingControlsCard
    )
    assert type(workspace.fixed_controls_stack.findChild(ModelParameterCard)) is ModelParameterCard
    assert type(workspace.fitting_plot_card) is PlotPreviewCard
    assert type(workspace.fitting_controls_card) is FittingPlotControlsCard

    assert workspace.fitting_input_section.objectName() == "fittingInputSection"
    assert workspace.fitting_configure_section.objectName() == "fittingConfigureSection"
    assert workspace.fitting_advanced_section.objectName() == "fittingAdvancedSection"
    assert workspace.fitting_run_section.objectName() == "fittingRunSection"
    assert workspace.fitting_preview_panel.objectName() == "fittingPreviewPanel"
    assert workspace.fitting_results_panel.objectName() == "fittingResultsPanel"
    assert workspace.fitting_export_section.objectName() == "fittingExportSection"

    assert window.fitBGValue.objectName() == "fitBGValue"
    assert window.fitBGStep.objectName() == "fitBGStep"
    assert window.fitBGStep.value() == 0.1
    assert window.fitKStep.value() == 0.1
    assert window.fitIntResStep.value() == 0.01
    assert window.fitSigmaResStep.value() == 0.0001
    assert window.fitSigmaResValue.singleStep() == 0.0001
    assert window.gisaxsAutoYonedaCutThicknessSpinBox.value() == 5
    assert window.aiFittingSamplesSpinBox.value() == 2000
    assert window.aiFittingRefineTopNSpinBox.value() == 5
    assert window.aiFittingRefineMaxEvalSpinBox.value() == 80
    assert window.aiFittingConstraintComboBox.itemText(0) == "Free Prediction"
    assert window.gisaxsInputCutLineLabel.text() == "Cut line:"
    assert [
        window.gisaxsInputModelCombox.itemText(index)
        for index in range(window.gisaxsInputModelCombox.count())
    ] == ["Single", "Stack"]
    assert window.FittingManualFittingButton.text() == "Plot Current Model"
    assert window.FittingManualFittingButton.parent().objectName() == (
        "fittingPersistentCommandBar"
    )

    parameter_binding = ParticleConnectionsMixin()
    parameter_binding.ui = window
    parameter_binding._iter_particle_widget_ids = lambda: []
    parameter_binding._add_fitting_success = lambda _message: None
    parameter_binding._setup_parameter_ranges([])
    assert window.fitSigmaResValue.singleStep() == window.fitSigmaResStep.value()
    assert window.fitMethodValue.itemText(3) == "Model: 1 Sphere + 1Cylinder"
    assert window.FittingExportButton.parent() is workspace.fitting_export_section.content
    assert window.fitExportPlotButton.parent() is workspace.fitting_export_section.content
    window.close()


def test_fitting_context_switch_preserves_single_navigation_and_versions_insitu_recipe():
    app = _app()
    window = MainWindow(_context())
    window.show()
    QTest.qWait(420)
    app.processEvents()
    workspace = window.components.fitting_workspace
    page = workspace.insitu_series_page

    assert isinstance(page.ui, InSituSeriesPageView)
    assert workspace.context_stack.currentWidget() is workspace.page_splitter
    assert workspace.single_context_button.isChecked()
    workspace.show_workflow_step("fit")
    workspace.preview_tabs.setCurrentIndex(1)
    single_step_index = workspace.workflow_content_stack.currentIndex()

    QTest.mouseClick(workspace.insitu_context_button, Qt.LeftButton)
    app.processEvents()
    assert workspace.context_stack.currentWidget() is page
    assert workspace.insitu_context_button.isChecked()
    assert [page.ui.previewTabs.tabText(index) for index in range(3)] == [
        "Preview",
        "Frames",
        "Log",
    ]
    assert tuple(page.ui.workflowButtons) == (
        "source",
        "preprocess",
        "geometry",
        "cut",
        "fit",
        "results",
    )
    assert page.ui.workflowControls.applyRecipeButton.isEnabled() is False
    QTest.mouseClick(page.ui.workflowButtons["geometry"], Qt.LeftButton)
    assert page.ui.workflowControls.stack.currentIndex() == 2

    recipe = window.components.fitting_view_model.insitu.create_recipe_from_single(
        SingleAnalysisRecipeSnapshot(
            experiment_setup={"distance_mm": 2000.0},
            preprocessing={"flip_ud": False},
            cut={"width_px": 5},
            model={"shapes": ["sphere"]},
            tracking=InSituTrackingPolicy(),
            fitting=InSituFittingPolicy(),
        )
    )
    page.render_recipe(recipe)
    single_center = window.gisaxsInputCenterParallelValue.value()
    page.ui.workflowControls.flipUdCheckBox.setChecked(True)
    page.ui.workflowControls.centerXSpinBox.setValue(411.0)
    page.ui.workflowControls.cutCenterParallelSpinBox.setValue(321.0)
    page.ui.workflowControls.refinementCombo.setCurrentText("Every N frames")
    page.ui.workflowControls.refineEverySpinBox.setValue(4)
    QTest.mouseClick(page.ui.workflowControls.applyRecipeButton, Qt.LeftButton)
    assert window.components.fitting_view_model.insitu.recipe.version == 2
    assert window.components.fitting_view_model.insitu.recipe.parent_version == 1
    assert window.components.fitting_view_model.insitu.recipe.fitting.refine_every_n == 4
    assert window.components.fitting_view_model.insitu.recipe.preprocessing["flip_ud"] is True
    assert (
        window.components.fitting_view_model.insitu.recipe.experiment_setup[
            "beam_center_x_px"
        ]
        == 411.0
    )
    assert window.components.fitting_view_model.insitu.recipe.cut["center_parallel_px"] == 321.0
    assert window.gisaxsInputCenterParallelValue.value() == single_center

    QTest.mouseClick(workspace.single_context_button, Qt.LeftButton)
    app.processEvents()
    assert workspace.context_stack.currentWidget() is workspace.page_splitter
    assert workspace.workflow_content_stack.currentIndex() == single_step_index
    assert workspace.preview_tabs.currentIndex() == 1
    window.runtime.fitting.initialize()
    assert (
        window.runtime.fitting._insitu_workflow_widgets["run_mode"]
        is page.ui.workflowControls.runModeCombo
    )
    page.ui.workflowControls.runModeCombo.setCurrentText("Live Watch")
    app.processEvents()
    assert not page.ui.workflowControls.liveSettingsWidget.isHidden()
    assert page.ui.workflowControls.sequenceSettingsWidget.isHidden()
    assert not page.ui.startWatchButton.isHidden()
    assert page.ui.startProcessButton.isHidden()
    page.ui.workflowControls.runModeCombo.setCurrentText("Process Existing Sequence")
    app.processEvents()
    assert page.ui.workflowControls.liveSettingsWidget.isHidden()
    assert not page.ui.workflowControls.sequenceSettingsWidget.isHidden()
    assert page.ui.startWatchButton.isHidden()
    assert not page.ui.startProcessButton.isHidden()
    assert window.gisaxsInputModelCombox.findText("In-situ") == -1
    assert window.findChild(QWidget, "gisaxsInputInsituWorkflowButton") is None
    assert not (
        PRESENTATION_ROOT / "bindings" / "insitu_dialog.py"
    ).exists()
    QTest.mouseClick(workspace.insitu_context_button, Qt.LeftButton)
    assert workspace.context_stack.currentWidget() is page
    window.close()


def test_single_analysis_explicitly_captures_serializable_insitu_recipe():
    app = _app()
    window = MainWindow(_context())
    window.show()
    QTest.qWait(420)
    app.processEvents()
    binding = window.runtime.fitting
    binding.initialize()
    binding.fitting_view_model.accept_loaded_image(
        ScatteringFileData(
            image=np.ones((4, 5)),
            source_path=Path("representative.cbf"),
            source_files=(Path("representative.cbf"),),
        )
    )
    binding.current_parameters["imported_gisaxs_file"] = "representative.cbf"
    binding._flip_ud = True
    binding._mirror_fill_detector_gaps = True

    QTest.mouseClick(
        window.components.fitting_workspace.insitu_series_page.ui.captureRecipeButton,
        Qt.LeftButton,
    )
    recipe = binding.fitting_view_model.insitu.recipe

    assert recipe.version == 1
    assert recipe.experiment_setup["distance_mm"] == 2000.0
    assert recipe.preprocessing["flip_ud"] is True
    assert recipe.preprocessing["mirror_fill_gaps"] is True
    assert recipe.cut["auto_horizontal_thickness_px"] == 5
    assert recipe.note.endswith("representative.cbf")
    assert binding._current_ui_matches_insitu_recipe() is True

    binding._flip_ud = False
    assert binding._current_ui_matches_insitu_recipe() is False
    QTest.mouseClick(
        window.components.fitting_workspace.insitu_series_page.ui.captureRecipeButton,
        Qt.LeftButton,
    )
    assert binding.fitting_view_model.insitu.recipe.version == 2
    assert binding.fitting_view_model.insitu.recipe.parent_version == 1
    window.close()


def test_fitting_layout_modules_do_not_import_workflow_or_scientific_runtimes():
    forbidden_roots = {"bornagain", "controllers", "keras", "tensorflow"}
    violations: list[str] = []
    layout_modules = (
        "ai_controls.py",
        "cut_card.py",
        "global_parameter_controls.py",
        "input_card.py",
        "layout_primitives.py",
        "model_card.py",
        "preview_cards.py",
        "run_card.py",
        "workspace.py",
        "workflow_header.py",
    )
    for name in layout_modules:
        path = PRESENTATION_ROOT / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports = [node.module]
            else:
                continue
            for imported in imports:
                if imported.split(".", maxsplit=1)[0].casefold() in forbidden_roots:
                    violations.append(f"{name}:{node.lineno}: {imported}")

    assert not violations, "Fitting layout contains workflow/runtime imports:\n" + "\n".join(
        violations
    )


def test_fitting_workbench_exposes_guided_progressive_disclosure_and_modes():
    app = _app()
    window = MainWindow(_context())
    window.show()
    QTest.qWait(80)
    app.processEvents()
    workspace = window.components.fitting_workspace

    assert tuple(workspace.workflow_header.STEP_TITLES) == (
        "Import data",
        "Experiment setup",
        "Yoneda & cut",
        "Fit",
    )
    assert workspace.workflow_header.steps[0].property("workflowState") == "available"
    assert window.fittingRemoteCacheDisclosure.content.isHidden()
    assert window.fittingCutStepDisclosure.content.isHidden()
    assert window.fittingAiTuningDisclosure.content.isHidden()

    assert window.fittingModeTabs.count() == 4
    assert [window.fittingModeTabs.tabText(index) for index in range(4)] == [
        "Components",
        "Global",
        "Data & refine",
        "Auto fit",
    ]
    assert workspace.preview_tabs.tabText(0) == "Detector"
    assert workspace.preview_tabs.count() == 2
    assert workspace.preview_tabs.tabText(1) == "Curve"
    assert workspace.fitting_plot_advanced_section.is_expanded() is False
    assert workspace.fitting_log_section.is_expanded() is False
    assert workspace.fitting_export_section.parent() is workspace.fitting_results_panel
    assert (
        workspace.fitting_plot_advanced_section.parent()
        is workspace.fitting_results_panel
    )
    workspace.preview_tabs.setCurrentIndex(0)
    assert workspace.fitting_plot_card.toolbar.isHidden()
    assert workspace.inline_feedback.parent() is workspace.fitting_preview_panel
    stable_tab_y = workspace.preview_tabs.tabBar().mapTo(
        workspace.right_panel, QPoint(0, 0)
    ).y()
    workspace.preview_tabs.setCurrentIndex(1)
    assert not workspace.fitting_plot_card.toolbar.isHidden()
    assert workspace.fitting_plot_card.toolbar.parent() is workspace.fitting_results_panel
    assert workspace.inline_feedback.parent() is workspace.fitting_results_panel
    assert workspace.preview_tabs.tabBar().mapTo(
        workspace.right_panel, QPoint(0, 0)
    ).y() == stable_tab_y
    workspace.inline_feedback.setText("Curve options need attention")
    workspace.inline_feedback.show()
    app.processEvents()
    assert workspace.preview_tabs.tabBar().mapTo(
        workspace.right_panel, QPoint(0, 0)
    ).y() == stable_tab_y
    workspace.inline_feedback.hide()

    workflow = initial_workflow_state()
    for key in ("import", "setup", "center", "cut"):
        workflow = complete_workflow_step(workflow, key)
    workspace.workflow_header.render(workflow)
    assert [step.property("workflowState") for step in workspace.workflow_header.steps] == [
        "complete",
        "complete",
        "complete",
        "available",
    ]
    window.fittingModeTabs.setCurrentIndex(3)
    workspace.preview_tabs.setCurrentIndex(1)
    app.processEvents()
    assert workspace.fitting_plot_card.toolbar.parent() is workspace.fitting_results_panel
    assert workspace.inline_feedback.parent() is workspace.fitting_results_panel
    assert workspace.preview_tabs.tabBar().mapTo(
        workspace.right_panel, QPoint(0, 0)
    ).y() == stable_tab_y
    assert window.aiFittingModelComboBox.parent() is not None
    assert workspace.fitting_plot_card.parent() is workspace.fitting_results_panel.content
    assert workspace.curve_plot_card is workspace.fitting_plot_card
    assert [
        window.fitCurveViewModeComboBox.itemData(index)
        for index in range(window.fitCurveViewModeComboBox.count())
    ] == ["data", "compare", "model"]
    window.close()


def test_fitting_empty_states_overlay_existing_graphics_views_without_replacement():
    _app()
    window = MainWindow(_context())
    workspace = window.components.fitting_workspace

    detector_state = workspace.detector_preview_card.empty_state
    result_state = workspace.fitting_plot_card.findChild(
        type(detector_state), "fitGraphicsViewEmptyState"
    )
    assert detector_state.parent() is window.gisaxsInputGraphicsView.viewport()
    assert result_state is not None
    assert result_state.parent() is window.fitGraphicsView.viewport()
    assert window.gisaxsInputGraphicsView.parent().objectName() == "fittingDetectorPreviewBody"
    window.close()


def test_fitting_primary_image_and_plot_controls_are_visible_at_point_of_use():
    app = _app()
    window = MainWindow(_context())
    workspace = window.components.fitting_workspace

    assert window.gisaxsInputAutoShowCheckBox.isChecked()
    assert window.gisaxsInputShowButton.parent() is window.gisaxsInputFileNavigationWidget
    assert window.gisaxsInputAutoShowCheckBox.parent() is (
        window.gisaxsInputFileNavigationWidget
    )
    assert not window.gisaxsInputShowButton.isHidden()
    assert window.gisaxsInputAutoScaleCheckBox.parent() is window.fittingDetectorDisplayInspector
    assert window.gisaxsInputVminValue.parent() is window.fittingDetectorDisplayInspector
    assert window.gisaxsInputColormapCombo.parent() is window.fittingDetectorDisplayInspector
    assert window.fittingDetectorPreprocessing.parent() is (
        window.fittingDetectorDisplayInspector
    )
    assert not window.fittingDetectorPreprocessing.isHidden()
    window.gisaxsInputThresholdMaskCheckBox.setChecked(True)
    window.gisaxsInputMirrorGapFillCheckBox.setChecked(True)
    app.processEvents()
    assert window.gisaxsInputThresholdMinSpinBox.isEnabled()
    assert window.gisaxsInputThresholdMaxSpinBox.isEnabled()
    assert window.gisaxsInputMirrorGapMarginSpinBox.isEnabled()
    assert window.fittingPickCenterButton.isCheckable()
    assert window.fittingSelectRegionButton.isCheckable()
    assert window.fitLogXCheckBox.parent().objectName() == "fittingResultToolBar"
    assert window.fitLogYCheckBox.parent().objectName() == "fittingResultToolBar"
    assert window.fitNormCheckBox.parent().objectName() == "fittingResultToolBar"
    assert workspace.workflow_header.parent() is workspace._workspace_ui.fittingWorkflowHost
    assert not workspace._workspace_ui.gisaxsFittingPageScrollArea.isAncestorOf(
        workspace.workflow_header
    )
    window.close()


def test_fitting_current_task_and_fit_mode_use_natural_height_without_blank_canvas():
    app = _app()
    window = MainWindow(_context())
    window.show()
    workspace = window.components.fitting_workspace
    workspace.show_workflow_step("fit")

    for index in range(window.fittingModeTabs.count()):
        window.fittingModeTabs.setCurrentIndex(index)
        QTest.qWait(90)
        app.processEvents()
        natural_height = workspace.fitting_fit_step_page.minimumSizeHint().height()
        assert workspace.workflow_content_stack.height() <= natural_height + 4
        assert window.FittingManualFittingButton.isVisible()
        assert window.FittingManualFittingButton.parent().objectName() == (
            "fittingPersistentCommandBar"
        )
        assert window.FittingManualFittingButton.mapTo(
            workspace.fitting_fit_step_page,
            window.FittingManualFittingButton.rect().topLeft(),
        ).y() < window.fittingModeTabs.mapTo(
            workspace.fitting_fit_step_page,
            window.fittingModeTabs.rect().topLeft(),
        ).y()

    bottom_controls = {
        1: window.findChild(QWidget, "fittingParameterStepHint"),
        2: window.findChild(QWidget, "FittingActionsGroup"),
        3: window.aiFittingStatusLabel,
    }
    for index, bottom_control in bottom_controls.items():
        assert bottom_control is not None
        window.fittingModeTabs.setCurrentIndex(index)
        QTest.qWait(90)
        app.processEvents()
        page = window.fittingModeTabs.currentWidget()
        control_bottom = bottom_control.mapTo(
            page, bottom_control.rect().bottomLeft()
        ).y()
        assert control_bottom <= page.contentsRect().bottom() + 1

    workspace.show_workflow_step("import")
    QTest.qWait(90)
    app.processEvents()
    assert workspace.workflow_content_stack.height() < 500
    assert window.gisaxsInputImportButton.isVisible()
    assert window.gisaxsInputAutoShowCheckBox.isVisible()
    window.close()


def test_fitting_yoneda_cut_and_parameter_step_preferences_are_persistent():
    app = _app()
    context = _context()
    window = MainWindow(context)

    assert window.gisaxsInputCenterAutoFindingButton.text() == "Find Yoneda & Set Cut"
    assert window.fittingCenterStepDisclosure is window.fittingCutStepDisclosure
    window.gisaxsAutoYonedaCutThicknessSpinBox.setValue(9)
    window.gisaxsAutoYonedaCutThicknessSpinBox.editingFinished.emit()
    window.fitSigmaResStep.setValue(0.00025)
    window.fitSigmaResStep.editingFinished.emit()

    preferences = context.preferences.snapshot()
    assert preferences["fitting.yoneda_cut.horizontal_thickness_pixels"] == 9
    assert preferences["fitting.parameter_step.resolution_sigma"] == 0.00025
    window.close()
    app.processEvents()

    restored = MainWindow(context)
    assert restored.gisaxsAutoYonedaCutThicknessSpinBox.value() == 9
    assert restored.fitSigmaResStep.value() == 0.00025
    assert restored.fitSigmaResValue.singleStep() == 0.00025
    restored.close()


def test_auto_yoneda_uses_the_configured_horizontal_cut_thickness():
    class ThicknessControl:
        @staticmethod
        def value():
            return 11

    binding = DetectorConfigurationMixin()
    binding.ui = SimpleNamespace(
        gisaxsAutoYonedaCutThicknessSpinBox=ThicknessControl()
    )
    assert binding._auto_horizontal_cut_thickness_pixels() == 11.0

    binding.ui = SimpleNamespace()
    assert binding._auto_horizontal_cut_thickness_pixels() == 5.0


def test_fitting_preview_tabs_ignore_hidden_curve_expansion():
    app = _app()
    window = MainWindow(_context())
    window.show()
    workspace = window.components.fitting_workspace

    workspace.preview_tabs.setCurrentIndex(0)
    app.processEvents()
    detector_height_before = workspace.preview_tabs.minimumSizeHint().height()

    workspace.preview_tabs.setCurrentIndex(1)
    workspace.fitting_plot_advanced_section.set_expanded(True)
    QTest.qWait(30)
    app.processEvents()
    expanded_fit_height = workspace.preview_tabs.minimumSizeHint().height()

    workspace.preview_tabs.setCurrentIndex(0)
    QTest.qWait(30)
    app.processEvents()
    detector_height_after = workspace.preview_tabs.minimumSizeHint().height()

    assert expanded_fit_height > detector_height_before
    assert abs(detector_height_after - detector_height_before) <= 2
    assert workspace.preview_tabs.currentWidget() is workspace.fitting_preview_panel
    window.close()


def test_fitting_workflow_header_compact_mode_keeps_navigation_available():
    _app()
    window = MainWindow(_context())
    header = window.components.fitting_workspace.workflow_header

    header.set_guided(False)

    assert header.mode_button.text() == "Compact"
    assert header.subtitle.isHidden()
    assert not header.steps[0].isHidden()
    assert header.steps[0].isEnabled()
    assert all(step.isEnabled() for step in header.steps)
    window.close()


def test_fitting_workflow_navigation_selects_one_task_without_completing_it():
    _app()
    window = MainWindow(_context())
    workspace = window.components.fitting_workspace
    initial_statuses = tuple(
        step.property("workflowState") for step in workspace.workflow_header.steps
    )
    workspace.preview_tabs.setCurrentIndex(1)

    workspace.show_workflow_step("center")

    assert workspace.workflow_content_stack.currentWidget() is workspace.fitting_configure_section
    assert workspace.cut_line_card.step_stack.currentWidget().objectName() == (
        "fittingYonedaCutPage"
    )
    assert workspace.workflow_header.steps[2].property("workflowSelected") is True
    assert tuple(
        step.property("workflowState") for step in workspace.workflow_header.steps
    ) == initial_statuses
    assert workspace.preview_tabs.currentIndex() == 1

    workspace.show_workflow_step("cut")
    assert workspace.cut_line_card.step_stack.currentWidget().objectName() == (
        "fittingYonedaCutPage"
    )
    assert workspace.workflow_header.steps[2].property("workflowSelected") is True
    assert workspace.preview_tabs.currentIndex() == 1

    workspace.show_workflow_step("center_cut")
    assert workspace.cut_line_card.step_stack.currentWidget().objectName() == (
        "fittingYonedaCutPage"
    )
    assert workspace.workflow_header.steps[2].property("workflowSelected") is True
    assert workspace.preview_tabs.currentIndex() == 1

    workspace.preview_tabs.setCurrentIndex(0)
    workspace.show_workflow_step("fit")
    assert workspace.preview_tabs.currentIndex() == 0
    window.close()


def test_fitting_model_parameters_are_primary_fit_content_and_wheel_safe():
    _app()
    window = MainWindow(_context())
    workspace = window.components.fitting_workspace
    workspace.show_workflow_step("fit")

    assert window.fittingModeTabs.indexOf(workspace.model_parameters_card) == 0
    assert not workspace.model_parameters_card.isHidden()
    assert workspace.fitting_advanced_section.isHidden()
    assert window.fitParticleShapeCombox_1.currentText() == "Sphere"
    assert "Sphere" in window.fitParticleStackWidget_1.currentWidget().objectName()
    assert window.fitParticleSphereRValue_1.property("gimapSafeWheelInput") is True
    assert window.fitParticleShapeCombox_1.property("gimapSafeWheelInput") is True
    window.close()


def test_fitting_signed_q_control_resolves_log_scale_without_exposing_internal_axes():
    app = _app()
    window = MainWindow(_context())
    QTest.qWait(120)
    QTest.qWait(260)
    app.processEvents()

    binding = window.runtime.fitting
    assert window.fitQViewModeComboBox.currentData() == "signed"
    assert binding._get_q_branch() == "both"
    assert binding._get_q_combination_mode() == "separate"
    assert binding._get_x_axis_scale() == "linear"
    window.fitLogXCheckBox.setChecked(True)
    app.processEvents()
    assert binding._get_x_axis_scale() == "symlog"
    assert "symmetric-log" in window.fitQViewHintLabel.text()

    window.fitQViewModeComboBox.setCurrentIndex(
        window.fitQViewModeComboBox.findData("fold")
    )
    app.processEvents()
    assert binding._get_q_branch() == "both"
    assert binding._get_q_combination_mode() == "fold"
    assert binding._get_x_axis_scale() == "log"
    assert not hasattr(window, "fitQBranchComboBox")
    assert not hasattr(window, "fitQCombinationComboBox")
    assert not hasattr(window, "fitXAxisScaleComboBox")

    shared_curve_state = CurveViewState(
        q_mode="average",
        layer_mode="data",
        log_x=True,
        log_y=True,
        normalize=True,
        q_unit="angstrom",
        y_range="experimental",
    )
    binding._apply_curve_view_state(shared_curve_state, refresh=False)
    assert binding.fitting_view_model.state.curve_view == shared_curve_state
    assert window.fitQViewModeComboBox.currentData() == "average"
    assert window.fitLogYCheckBox.isChecked()
    assert window.fitNormCheckBox.isChecked()

    shared_detector_state = DetectorDisplayState(
        log_intensity=False,
        auto_scale=False,
        vmin=1.25,
        vmax=9.75,
        colormap="magma",
        show_cut_region=False,
        show_center=True,
        show_q_axis=True,
    )
    binding._apply_detector_display_state(shared_detector_state)
    assert binding.fitting_view_model.state.detector_display == shared_detector_state
    assert not window.gisaxsInputIntLogCheckBox.isChecked()
    assert not window.gisaxsInputAutoScaleCheckBox.isChecked()
    assert window.gisaxsInputVminValue.value() == 1.25
    assert window.gisaxsInputVmaxValue.value() == 9.75
    assert window.gisaxsInputColormapCombo.currentText() == "magma"
    window.close()


def test_fitting_detector_setup_is_inline_and_uses_the_injected_view_model():
    app = _app()
    context = _context()
    window = MainWindow(context)
    workspace = window.components.fitting_workspace
    workspace.show_workflow_step("setup")
    app.processEvents()

    panel = window.fittingDetectorSetupPanel
    assert type(panel) is DetectorSetupPanel
    assert panel.parent().objectName() == "fittingDetectorSetupPage"
    assert window.gisaxsInputDetectorParaButton.parent() is panel
    assert window.gisaxsInputDetectorParaButton.text() == "Apply detector setup"
    assert not hasattr(window, "cutLineDetectorHintLabel")

    panel.distance_spinbox.setValue(2345.6)
    QTest.qWait(250)
    app.processEvents()
    assert context.settings.get("fitting", "detector.distance", None) == 2345.6
    assert panel.status_label.text() == "Detector setup applied"

    panel.distance_spinbox.setValue(2456.7)
    panel.distance_spinbox.editingFinished.emit()
    app.processEvents()
    assert context.settings.get("fitting", "detector.distance", None) == 2456.7
    window.close()


def test_fitting_export_dialog_makes_curve_representation_visible():
    _app()
    dialog = FittingDataExportDialog(("Cut Data", "Fitting Data"))

    assert dialog.source_combo.currentText() == "Cut Data"
    assert dialog.selection().preparation == "fitting"
    dialog.preparation_combo.setCurrentIndex(
        dialog.preparation_combo.findData("raw")
    )
    assert dialog.selection().preparation == "raw"
    assert "original signed q" in dialog.summary_label.text()
    dialog.close()


def test_selecting_center_or_region_preserves_preview_and_refreshes_existing_cut_once():
    app = _app()
    window = MainWindow(_context())
    QTest.qWait(120)
    QTest.qWait(260)
    app.processEvents()
    workspace = window.components.fitting_workspace
    binding = window.runtime.fitting
    workspace.preview_tabs.setCurrentIndex(0)

    binding._on_region_selected(
        {
            "is_q_space": False,
            "pixel_center_x": 128.0,
            "pixel_center_y": 96.0,
            "pixel_width": 20.0,
            "pixel_height": 16.0,
        }
    )
    app.processEvents()

    assert workspace.preview_tabs.currentIndex() == 0
    assert binding.fitting_view_model.state.cut_status != "ready"

    binding.current_cut_data = {
        "x_coords": [0.1, 0.2],
        "y_intensity": [10.0, 8.0],
    }
    refreshes = []
    binding._perform_cut = lambda *args, **kwargs: refreshes.append(kwargs)
    binding._on_region_selected(
        {
            "is_q_space": False,
            "pixel_center_x": 130.0,
            "pixel_center_y": 98.0,
            "pixel_width": 22.0,
            "pixel_height": 18.0,
        }
    )
    app.processEvents()

    assert refreshes == [{"reveal_result": False}]
    assert workspace.preview_tabs.currentIndex() == 0
    window.close()


def test_detector_q_preview_and_axis_switch_keep_the_same_detector_cells():
    app = _app()
    context = _context()
    window = MainWindow(context)
    QTest.qWait(300)
    app.processEvents()
    binding = window.runtime.fitting
    image = np.arange(48, dtype=np.float32).reshape(6, 8) + 1.0
    binding.current_detector_image = FittingImageCalculations().prepare(
        image,
        revision=1,
    )
    binding.fitting_view_model.set_setting("fitting", "detector.show_q_axis", True)
    binding.fitting_view_model.set_setting(
        "fitting", "detector.horizontal_q_axis", "qy"
    )
    binding._last_q_mode = True
    binding._last_horizontal_q_axis = "qy"
    binding._update_cutline_step_sizes()
    binding._compute_q_meshgrids_and_store()
    binding._apply_pixel_region_to_active_coordinates(
        (1, 3, 2, 5),
        q_mode=True,
        horizontal_axis="qy",
    )

    binding._refresh_image_display()
    assert isinstance(binding._preview_image_artist, QuadMesh)
    assert binding._preview_ax.get_xlabel() == r"$q_y$ (nm$^{-1}$)"

    binding.fitting_view_model.set_setting(
        "fitting", "detector.horizontal_q_axis", "qr"
    )
    binding._on_detector_parameters_changed({"horizontal_q_axis": "qr"})
    remapped = binding._current_selection_pixel_region(
        q_mode=True,
        horizontal_axis="qr",
    )

    assert remapped == (1, 3, 2, 5)
    assert binding.current_parameter_selection["horizontal_q_axis"] == "qr"
    assert binding._preview_ax.get_xlabel() == r"$q_r$ (nm$^{-1}$)"

    binding._has_displayed_image = True
    binding._initialize_roi_from_current_q = lambda **_kwargs: None
    binding._apply_roi_to_data_and_refresh = lambda: None
    binding._update_GUI_image = lambda _mode: None
    binding._update_outside_window = lambda _mode: None
    binding._auto_find_center()
    assert binding.current_parameter_selection["is_q_space"] is True
    assert binding.current_parameter_selection["horizontal_q_axis"] == "qr"
    assert binding.current_parameter_selection["pixel_height"] > 0
    window.close()
