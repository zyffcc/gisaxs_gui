"""Fitting presentation: feature-owned views, the two pages of the workspace and 1D Predict (offscreen)."""

from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest

from main import MainWindow
from src.gimap.app import AppContext
from src.gimap.features.fitting.application.single_fit import point_key
from src.gimap.features.fitting.presentation import (
    FittingDataExportDialog,
    FittingViewModel,
    FittingWorkspace,
    build_fitting_controls,
)
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

MIGRATED_CLASSES = {"FittingContextContainer", "FittingWorkspace", "ScientificDoubleSpinBox"}


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


def test_fitting_export_dialog_makes_curve_representation_visible():
    _app()
    dialog = FittingDataExportDialog(("Curve Data", "Fitting Data"))

    assert dialog.source_combo.currentText() == "Curve Data"
    assert dialog.selection().preparation == "fitting"
    dialog.preparation_combo.setCurrentIndex(dialog.preparation_combo.findData("raw"))
    assert dialog.selection().preparation == "raw"
    assert "original signed q" in dialog.summary_label.text()
    dialog.close()


def test_fitting_views_are_python_view_owned():
    views = PRESENTATION_ROOT / "views"
    factory_source = (PRESENTATION_ROOT / "control_view_factory.py").read_text(encoding="utf-8")

    assert "QtWidgets.QPushButton" not in factory_source
    assert len(factory_source.splitlines()) <= 40
    assert {path.name for path in views.glob("*_view.py")} == {
        "curve_import_view.py",
        "fit_results_view.py",
        "fit_step_view.py",
        "fit_page_view.py",
        "fit_series_view.py",
        "fit_steps_view.py",
        "fit_uncertainty_view.py",
    }


def _write_curve(tmp_path):
    curve = tmp_path / "sample_fit_input.dat"
    q = np.linspace(-0.1, 0.1, 50)
    np.savetxt(curve, np.column_stack([q, 100 * np.exp(-(q / 0.03) ** 2) + 1, np.ones(50)]))
    return curve


def test_fitting_holds_single_analysis_and_the_series_only():
    app, window = _fitting_window()
    workspace = window.components.fitting_workspace

    assert type(workspace) is FittingWorkspace
    stack = workspace.context_stack
    assert [stack.widget(index) for index in range(stack.count())] == [workspace.fit_page, workspace.series_page]
    for former in ("fitGraphicsView", "FittingAutoFittingButton", "aiFittingModelComboBox",
                   "fittingInsituSeriesPage", "fitImport1dFileButton"):
        assert not hasattr(window, former), former
    window.close()


def test_a_curve_from_a_file_is_described_and_plotted(tmp_path):
    app, window = _fitting_window()
    workspace = window.components.fitting_workspace
    curve = _write_curve(tmp_path)

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
    assert binding.get_imported_file() == str(curve)
    assert binding.get_session_data()["last_1d_file"] == str(curve)
    window.close()


def test_1d_predict_fits_the_points_single_analysis_shows(tmp_path):
    """The curve of 1D Predict is that of Single analysis: signed q in nm⁻¹, the halves, range and
    left-out points of the page."""
    app, window = _fitting_window()
    workspace = window.components.fitting_workspace
    binding = window.runtime.fitting
    assert binding._current_ai_curve_arrays() is None  # no curve yet
    assert workspace.open_curve(_write_curve(tmp_path), "mean")
    page = workspace.fit_page

    q, intensity, sigma = binding._current_ai_curve_arrays()
    assert q.size == 50 and np.any(q < 0) and np.any(q > 0)
    assert np.max(np.abs(q)) == pytest.approx(1.0)  # 0.1 Å⁻¹ of the file
    assert intensity.shape == sigma.shape == q.shape and not binding._workflow_sigma_estimated

    page.set_side("positive")
    page.set_range((0.2, 0.8))
    left_out = point_key(q[(q > 0.4) & (q < 0.8)][0])
    page.session.excluded.add(left_out)
    q, _intensity, _sigma = binding._current_ai_curve_arrays()
    assert q.size and np.all(q > 0) and np.all((q >= 0.2) & (q <= 0.8))
    assert not np.any(np.isclose(q, left_out))

    page.batch_button.click()  # the 1D Predict button of the page opens the dialog on this curve
    dialog = binding._workflow_v5_dialog
    assert dialog is not None and dialog.isVisible() and dialog.current_curve == binding._current_ai_curve_arrays
    dialog.close()
    window.close()


def test_fitting_modules_do_not_import_workflow_or_scientific_runtimes():
    forbidden_roots = {"bornagain", "controllers", "keras", "tensorflow"}
    violations: list[str] = []
    for name in ("control_view_factory.py", "layout_primitives.py", "view_binding.py", "workspace.py",
                 "workspace_context.py"):
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
