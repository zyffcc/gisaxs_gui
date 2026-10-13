"""Fitting presentation: feature-owned views and the curve-first workspace (offscreen)."""

from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QWidget,
)
from PyQt5.QtTest import QTest

from main import MainWindow
from src.gimap.app import AppContext
from src.gimap.features.fitting.presentation import (
    CurveSourceCard,
    FittingControlsCard,
    FittingDataExportDialog,
    FittingPlotControlsCard,
    FittingViewModel,
    FittingWorkspace,
    ModelParameterCard,
    PlotPreviewCard,
    build_fitting_controls,
    translate_fitting_controls,
)
from src.gimap.features.fitting.presentation.views import (
    FittingPageView,
    IndependentFitWindowView,
)
from src.gimap.features.fitting.presentation.view_binding import IndependentFitWindow
from src.gimap.features.fitting.presentation.bindings.particle_connections import (
    ParticleConnectionsMixin,
)
from src.gimap.features.fitting.presentation.state import CurveViewState
from src.gimap.integrations.jobs import LocalProcessJobRunner
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)


ROOT = Path(__file__).resolve().parents[1]
APP_COMPOSITION = ROOT / "src" / "gimap" / "app" / "main_window.py"
GENERATED_MAIN_WINDOW = ROOT / "src" / "gimap" / "app" / "window_view.py"
PRESENTATION_ROOT = ROOT / "src" / "gimap" / "features" / "fitting" / "presentation"
_TEST_APP = None

MIGRATED_CLASSES = {
    "CardFrame",
    "CurveSourceCard",
    "FittingControlsCard",
    "FittingPlotControlsCard",
    "FittingRegionControl",
    "FittingWorkspace",
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


def _fitting_window():
    """The main window on the Fitting page with its binding initialised."""
    app = _app()
    window = MainWindow(_context())
    window.show()
    for _ in range(30):
        QTest.qWait(50)
        app.processEvents()
        if hasattr(window, "runtime") and window.runtime.fitting._initialized:
            break
    window.menus.show_workspace("fitting")
    app.processEvents()
    return app, window


def test_app_composition_does_not_redefine_feature_owned_fitting_classes():
    assert FittingViewModel.__module__.startswith("src.gimap.features.fitting.presentation")

    composition_tree = ast.parse(APP_COMPOSITION.read_text(encoding="utf-8"))
    class_names = {node.name for node in composition_tree.body if isinstance(node, ast.ClassDef)}
    assert MIGRATED_CLASSES.isdisjoint(class_names)


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


def test_fitting_export_dialog_makes_curve_representation_visible():
    _app()
    dialog = FittingDataExportDialog(("Curve Data", "Fitting Data"))

    assert dialog.source_combo.currentText() == "Curve Data"
    assert dialog.selection().preparation == "fitting"
    dialog.preparation_combo.setCurrentIndex(dialog.preparation_combo.findData("raw"))
    assert dialog.selection().preparation == "raw"
    assert "original signed q" in dialog.summary_label.text()
    dialog.close()


def test_global_search_and_local_refine_open_mode_specific_bounds(
    monkeypatch,
):
    app, window = _fitting_window()

    binding = window.runtime.fitting
    assert binding.main_window is window

    binding.current_1d_data = {
        "q": np.linspace(0.01, 0.2, 20),
        "I": np.linspace(10.0, 2.0, 20),
        "q_source_unit": "angstrom",
    }
    binding._roi_min = None
    binding._roi_max = None
    binding._q_full_min = None
    binding._q_full_max = None

    ai_q, ai_intensity, _ai_sigma = binding._current_ai_curve_arrays()
    np.testing.assert_allclose(ai_q, binding.current_1d_data["q"] * 10.0)
    np.testing.assert_allclose(ai_intensity, binding.current_1d_data["I"])

    initial_setup = binding._build_manual_refine_setup()
    for meta in binding.param_trigger_manager._meta_registry.values():
        assert meta["last_value"] == pytest.approx(meta["widget"].value())
        if meta["meta"].get("persist") in {"model_particle", "model_global"}:
            assert meta["widget"].decimals() >= 12
    radius_desc = next(
        desc for desc in initial_setup["params"] if desc["name"].rstrip("0123456789") == "R"
    )
    radius_value = float(radius_desc["value"])
    expected_lower, expected_upper = binding._default_manual_refine_bounds(
        radius_desc["name"], radius_value
    )
    binding._sync_fitting_action_availability()
    assert window.FittingGlobalSearchButton.isEnabled()
    assert window.FittingGlobalSearchButton.text() == "Global Search"
    assert window.FittingAutoRefineButton.isEnabled()
    assert window.FittingAutoRefineButton.text() == "Local Refine"
    QTest.mouseClick(window.FittingAutoRefineButton, Qt.LeftButton)
    app.processEvents()

    dialog = binding._manual_auto_refine_dialog
    assert dialog is not None and dialog.isVisible()
    summary = dialog.findChild(QLabel, "manualAutoRefineInputSummary")
    table = dialog.findChild(QTableWidget, "manualAutoRefineParameterTable")
    assert "imported 1D data (20 fitting points)" in summary.text()
    assert table.rowCount() == len(initial_setup["params"])
    radius_row = next(
        row for row in range(table.rowCount()) if table.item(row, 1).text() == radius_desc["label"]
    )
    assert table.cellWidget(radius_row, 0).isChecked()
    assert table.cellWidget(radius_row, 3).value() == pytest.approx(expected_lower, abs=1e-8)
    assert table.cellWidget(radius_row, 4).value() == pytest.approx(expected_upper, abs=1e-8)
    assert not table.item(radius_row, 2).flags() & Qt.ItemIsEditable
    table.cellWidget(radius_row, 3).setValue(radius_value * 0.9)
    app.processEvents()
    dialog.close()
    app.processEvents()

    expected_global_lower, expected_global_upper = binding._default_manual_global_bounds(
        radius_desc["name"],
        radius_value,
        initial_setup["y"],
        initial_setup["q_model"],
    )
    QTest.mouseClick(window.FittingGlobalSearchButton, Qt.LeftButton)
    app.processEvents()
    dialog = binding._manual_auto_refine_dialog
    table = dialog.findChild(QTableWidget, "manualAutoRefineParameterTable")
    radius_row = next(
        row for row in range(table.rowCount()) if table.item(row, 1).text() == radius_desc["label"]
    )
    assert dialog.objectName() == "manualGlobalSearchDialog"
    assert (
        "Differential evolution explores the broad editable ranges"
        in dialog.findChild(QLabel, "manualAutoRefineInputSummary").text()
    )
    assert table.cellWidget(radius_row, 3).value() == pytest.approx(expected_global_lower, abs=1e-8)
    assert table.cellWidget(radius_row, 4).value() == pytest.approx(expected_global_upper, abs=1e-8)
    sigma_row = next(
        row for row in range(table.rowCount()) if "sigma_R" in table.item(row, 1).text()
    )
    k_row = next(row for row in range(table.rowCount()) if table.item(row, 1).text() == "Global k")
    assert table.cellWidget(sigma_row, 0).isChecked()
    assert not table.cellWidget(k_row, 0).isChecked()
    assert dialog.findChild(QSpinBox, "manualGlobalSamplesSpinBox").value() == 16384
    assert dialog.findChild(QSpinBox, "manualGlobalStartsSpinBox").value() == 3
    assert dialog.findChild(QDoubleSpinBox, "manualTargetLogRmseSpinBox").value() == 0.0
    dialog.close()
    app.processEvents()

    new_radius_value = radius_value * 2.0
    radius_widget = getattr(window, radius_desc["widget_name"])
    previous_block = radius_widget.blockSignals(True)
    radius_widget.setValue(new_radius_value)
    radius_widget.blockSignals(previous_block)
    QTest.mouseClick(window.FittingAutoRefineButton, Qt.LeftButton)
    app.processEvents()
    dialog = binding._manual_auto_refine_dialog
    table = dialog.findChild(QTableWidget, "manualAutoRefineParameterTable")
    radius_row = next(
        row for row in range(table.rowCount()) if table.item(row, 1).text() == radius_desc["label"]
    )
    expected_lower, expected_upper = binding._default_manual_refine_bounds(
        radius_desc["name"], new_radius_value
    )
    assert table.cellWidget(radius_row, 3).value() == pytest.approx(expected_lower, abs=1e-8)
    assert table.cellWidget(radius_row, 4).value() == pytest.approx(expected_upper, abs=1e-8)
    dialog.close()
    app.processEvents()

    loaded_candidates = []
    refine_modes = []

    def record_candidate(row, *, refresh_plot=True):
        loaded_candidates.append((row, refresh_plot))
        return True

    monkeypatch.setattr(binding, "_load_ai_candidate_params", record_candidate)
    monkeypatch.setattr(
        binding,
        "_show_manual_auto_refine_dialog",
        lambda mode="local": refine_modes.append(mode),
    )
    candidate = {
        "rank": 1,
        "combination": "sphere+sphere+sphere",
        "score_weighted_probability": 0.72,
        "posterior_frequency": 0.41,
        "best_log_rmse": 0.12,
        "best_chi2_weighted": 1.5,
        "best_source": "posterior_sample",
        "components": [
            {"type": "sphere", "weight": 1.0 / 3.0, "params": {"R": radius}}
            for radius in (4.0, 8.0, 14.0)
        ],
    }
    binding._show_ai_candidate_table(rows=[candidate], prefer_refine=True)
    candidate_dialog = binding._ai_results_dialog
    candidate_table = candidate_dialog.findChild(QTableWidget, "aiFittingCandidatesTable")
    refine_button = candidate_dialog.findChild(QPushButton, "aiRefineCandidateButton")
    assert candidate_table.currentRow() == 0
    assert refine_button.isDefault()
    QTest.mouseClick(refine_button, Qt.LeftButton)
    QTest.qWait(10)
    app.processEvents()
    assert loaded_candidates[-1][0]["combination"] == candidate["combination"]
    assert loaded_candidates[-1][1] is True
    assert refine_modes == ["local"]

    binding._show_ai_candidate_table(rows=[candidate])
    candidate_dialog = binding._ai_results_dialog
    global_button = candidate_dialog.findChild(QPushButton, "aiGlobalSearchCandidateButton")
    QTest.mouseClick(global_button, Qt.LeftButton)
    QTest.qWait(10)
    app.processEvents()
    assert refine_modes == ["local", "global"]

    binding.current_1d_data = None
    binding._sync_fitting_action_availability()
    assert not window.FittingAutoRefineButton.isEnabled()
    assert not window.FittingGlobalSearchButton.isEnabled()

    window.close()


def test_fitting_static_controls_and_workspace_are_python_view_owned():
    views = PRESENTATION_ROOT / "views"
    factory_source = (PRESENTATION_ROOT / "control_view_factory.py").read_text(encoding="utf-8")

    assert FittingPageView.__module__.endswith("views.fitting_page_view")
    assert "FittingPageView" in factory_source
    assert "QtWidgets.QPushButton" not in factory_source
    assert len(factory_source.splitlines()) <= 40
    assert {path.name for path in views.glob("*_view.py")} == {
        "fit_page_view.py",
        "fit_series_view.py",
        "fit_steps_view.py",
        "fitting_page_view.py",
        "independent_fit_window_view.py",
        "insitu_series_page_view.py",
    }


def test_independent_fit_window_projects_the_typed_curve_state() -> None:
    app = _app()
    window = IndependentFitWindow()
    assert isinstance(window, IndependentFitWindowView)
    assert window.q_unit_combo.currentData() == "nm"
    assert window.y_range_combo.currentData() == "all"
    state = CurveViewState(
        q_mode="fold",
        layer_mode="data",
        log_x=True,
        log_y=True,
        normalize=True,
        q_unit="angstrom",
        y_range="experimental",
    )
    window.set_curve_view_state(state)
    assert window.current_curve_view_state() == state
    emitted = []
    window.view_state_changed.connect(emitted.append)
    window.q_view_combo.setCurrentIndex(window.q_view_combo.findData("positive"))
    assert emitted[-1].q_mode == "positive"
    window.close()
    app.processEvents()


def test_fitting_workspace_is_curve_first_with_one_plot_and_no_detector_steps():
    app, window = _fitting_window()
    workspace = window.components.fitting_workspace

    assert type(workspace) is FittingWorkspace
    assert type(workspace.curve_card) is CurveSourceCard
    assert type(workspace.fitting_controls_card) is FittingControlsCard
    assert type(workspace.model_parameters_card) is ModelParameterCard
    assert type(workspace.fitting_plot_card) is PlotPreviewCard
    assert type(workspace.fitting_controls_plot_card) is FittingPlotControlsCard
    for detector_widget in (
        "gisaxsInputImportButton",
        "gisaxsInputGraphicsView",
        "fitCurrentDataCheckBox",
        "fittingDetectorSetupPanel",
        "fittingWorkflowHeader",
        "fittingPreviewTabs",
    ):
        assert not hasattr(window, detector_widget), detector_widget

    assert [window.fittingModeTabs.tabText(index) for index in range(4)] == [
        "Components",
        "Global",
        "Refine",
        "1D Predict",
    ]
    assert window.fitImport1dFileButton.parent() is workspace.curve_card
    assert window.fitImport1dFileButton.text() == "Open Curve…"
    assert window.fitImport1dFileValue.isHidden()
    assert workspace.curve_card.name_label.text() == "No curve"
    assert window.fitLogYCheckBox.isChecked()  # scattering data are read on a log scale
    assert not window.fitLogXCheckBox.isChecked()
    assert window.fitLogXCheckBox.parent().objectName() == "fittingResultToolBar"
    assert workspace.plot_controls_section.is_expanded() is False
    assert workspace.log_section.is_expanded() is False
    assert window.FittingExportButton.parent() is workspace.results_panel
    assert window.fitExportPlotButton.parent() is workspace.results_panel
    assert [
        window.fitCurveViewModeComboBox.itemData(index)
        for index in range(window.fitCurveViewModeComboBox.count())
    ] == ["data", "compare", "model"]

    assert window.fitBGStep.value() == 0.1
    assert window.fitKStep.value() == 0.1
    assert window.fitIntResStep.value() == 0.01
    assert window.fitSigmaResStep.value() == 0.0001
    assert window.aiFittingConstraintComboBox.itemText(0) == "Free Prediction"
    assert window.FittingManualFittingButton.text() == "Plot Current Model"
    parameter_binding = ParticleConnectionsMixin()
    parameter_binding.ui = window
    parameter_binding._iter_particle_widget_ids = lambda: []
    parameter_binding._add_fitting_success = lambda _message: None
    parameter_binding._setup_parameter_ranges([])
    assert window.fitSigmaResValue.singleStep() == window.fitSigmaResStep.value()
    window.close()


def test_a_curve_from_a_file_is_described_and_plotted(tmp_path):
    """Single analysis is the new page; the former page follows its curve for In-situ series."""
    app, window = _fitting_window()
    workspace = window.components.fitting_workspace
    curve = tmp_path / "sample_fit_input.dat"
    q = np.linspace(-0.1, 0.1, 50)
    np.savetxt(curve, np.column_stack([q, 100 * np.exp(-(q / 0.03) ** 2) + 1, np.ones(50)]))

    assert workspace.open_curve(curve, "mean")
    app.processEvents()
    page = workspace.fit_page
    assert workspace.context_stack.currentWidget() is page
    assert page.curve_chip.text() == "sample_fit_input.dat"
    assert "25 points, |q| 0.02041–1 nm⁻¹" in page.curve_info.text() and not page.side_combo.isHidden()
    assert page.fit_button.isEnabled() and page.step_rail.state("curve") == "ok"
    # The range is in nm⁻¹: 0.1 Å⁻¹ of the file is 1 nm⁻¹.
    assert page.range_max_spin.value() == pytest.approx(1.0, abs=1e-6)
    page.range_max_spin.setValue(0.5)
    assert page.session.q_range == pytest.approx((page.range_min_spin.value(), 0.5))
    binding = window.runtime.fitting
    assert binding.current_1d_data is not None  # In-situ series takes its set-up from this curve
    window.close()

def test_export_plot_writes_the_curve_as_shown(tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog

    app, window = _fitting_window()
    workspace = window.components.fitting_workspace
    curve = tmp_path / "sample_fit_input.dat"
    q = np.linspace(-0.1, 0.1, 50)
    np.savetxt(curve, np.column_stack([q, 100 * np.exp(-(q / 0.03) ** 2) + 1, np.ones(50)]))
    window.runtime.fitting.import_1d_file(curve, q_view="fold")
    workspace.show_fit_curve()
    app.processEvents()
    target = tmp_path / "plot.svg"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *args, **kwargs: (str(target), ""))
    )

    series, labels = window.runtime.fitting._plotted_figure_series()
    window.fitExportPlotButton.click()

    # Both halves of the folded cut, in their two colours; the range guides are not data.
    assert [item.label for item in series] == ["Data · +q", "Data · −q mirrored"]
    assert labels["log_y"] is window.fitLogYCheckBox.isChecked()
    assert "[" not in labels["x_label"] and "q" in labels["x_label"]
    text = target.read_text(encoding="utf-8")
    assert text.lstrip().startswith("<?xml") and "Data · +q" in text
    window.close()


def test_fitting_model_parameters_are_primary_fit_content_and_wheel_safe():
    _app()
    window = MainWindow(_context())
    workspace = window.components.fitting_workspace

    assert window.fittingModeTabs.indexOf(workspace.model_parameters_card) == 0
    assert window.fittingModeTabs.currentIndex() == 3
    window.fittingModeTabs.setCurrentIndex(0)
    assert not workspace.model_parameters_card.isHidden()
    assert window.fitParticleShapeCombox_1.currentText() == "Sphere"
    assert "Sphere" in window.fitParticleStackWidget_1.currentWidget().objectName()
    assert window.fitParticleSphereRValue_1.property("gimapSafeWheelInput") is True
    assert window.fitParticleShapeCombox_1.property("gimapSafeWheelInput") is True
    window.close()


def test_fitting_parameter_step_preferences_are_persistent():
    app = _app()
    context = _context()
    window = MainWindow(context)
    window.fitSigmaResStep.setValue(0.00025)
    window.fitSigmaResStep.editingFinished.emit()
    assert context.preferences.snapshot()["fitting.parameter_step.resolution_sigma"] == 0.00025
    window.close()
    app.processEvents()

    restored = MainWindow(context)
    assert restored.fitSigmaResStep.value() == 0.00025
    assert restored.fitSigmaResValue.singleStep() == 0.00025
    restored.close()


def test_fitting_signed_q_control_resolves_log_scale_without_exposing_internal_axes():
    app, window = _fitting_window()
    binding = window.runtime.fitting
    assert window.fitQViewModeComboBox.currentData() == "signed"
    assert binding._get_q_branch() == "both"
    assert binding._get_q_combination_mode() == "separate"
    assert binding._get_x_axis_scale() == "linear"
    window.fitLogXCheckBox.setChecked(True)
    app.processEvents()
    assert binding._get_x_axis_scale() == "symlog"
    assert "symmetric-log" in window.fitQViewHintLabel.text()

    window.fitQViewModeComboBox.setCurrentIndex(window.fitQViewModeComboBox.findData("fold"))
    app.processEvents()
    assert binding._get_q_branch() == "both"
    assert binding._get_q_combination_mode() == "fold"
    assert binding._get_x_axis_scale() == "log"
    assert not hasattr(window, "fitQBranchComboBox")

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
    window.close()


def test_fitting_layout_modules_do_not_import_workflow_or_scientific_runtimes():
    forbidden_roots = {"bornagain", "controllers", "keras", "tensorflow"}
    violations: list[str] = []
    for name in (
        "ai_controls.py",
        "curve_card.py",
        "global_parameter_controls.py",
        "layout_primitives.py",
        "model_card.py",
        "preview_cards.py",
        "run_card.py",
        "workspace.py",
    ):
        tree = ast.parse((PRESENTATION_ROOT / name).read_text(encoding="utf-8"))
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
    assert not violations, "\n".join(violations)

