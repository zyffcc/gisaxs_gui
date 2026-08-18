"""Fitting presentation ownership and offscreen compatibility tests。"""

from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from main import MainWindow
from src.gimap.app import AppContext
from src.gimap.features.fitting.presentation import (
    CutLineCard,
    FittingControlsCard,
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
)
from src.gimap.features.fitting.presentation.view_binding import (
    IndependentFitWindow,
    IndependentMatplotlibWindow,
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
    assert window.aiFittingSamplesSpinBox.value() == 2000
    assert window.aiFittingRefineTopNSpinBox.value() == 5
    assert window.aiFittingRefineMaxEvalSpinBox.value() == 80
    assert window.aiFittingConstraintComboBox.itemText(0) == "Free Prediction"
    assert window.gisaxsInputCutLineLabel.text() == "Cut line:"
    assert window.gisaxsInputModelCombox.itemText(2) == "In-situ"
    assert window.FittingManualFittingButton.text() == "Manual Fitting"
    assert window.fitMethodValue.itemText(3) == "Model: 1 Sphere + 1Cylinder"
    assert window.FittingExportButton.parent() is workspace.fitting_export_section.content
    assert window.fitExportPlotButton.parent() is workspace.fitting_export_section.content
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
