"""Fitting ▸ In-situ series: a folder of curves from Analyze, fitted one by one."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest

from main import MainWindow
from src.gimap.app import AppContext
from src.gimap.features.fitting.application import SingleAnalysisRecipeSnapshot
from src.gimap.features.fitting.domain import InSituFittingPolicy, InSituTrackingPolicy
from src.gimap.features.fitting.presentation.bindings.insitu_sequence import curve_number
from src.gimap.features.fitting.presentation.views import InSituSeriesPageView
from src.gimap.integrations.jobs import LocalProcessJobRunner
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)

_APP = None


def _app() -> QApplication:
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _window():
    app = _app()
    context = AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        jobs=LocalProcessJobRunner(),
    )
    window = MainWindow(context)
    window.show()
    QTest.qWait(420)
    app.processEvents()
    binding = window.runtime.fitting
    binding.initialize()
    return app, window, binding


def _write_curve(path: Path, scale: float = 1.0, points: int = 40) -> Path:
    """An Analyze-style fit input: q (1/A), I, sigma, pixels."""
    q = np.linspace(-0.05, 0.05, points)
    intensity = scale * (100.0 * np.exp(-((q / 0.02) ** 2)) + 1.0)
    rows = np.column_stack([q, intensity, np.sqrt(intensity), np.full(points, 5)])
    lines = ["# GIMaP Analyze fit input (q in 1/A)"]
    lines += [" ".join(f"{value:.8g}" for value in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _plain_recipe(binding):
    return binding.fitting_view_model.insitu.create_recipe_from_single(
        SingleAnalysisRecipeSnapshot(
            experiment_setup={},
            preprocessing={},
            cut={"source": "curve"},
            model={"workflow_v5": {"method": "model", "numerical": True}},
            tracking=InSituTrackingPolicy(),
            fitting=InSituFittingPolicy(),
        )
    )


def test_curve_numbers_ignore_analyze_suffixes() -> None:
    assert curve_number("run_00042_fit_input.dat") == 42
    assert curve_number("run_frame0007_sum10_fit_input.dat") == 7
    assert curve_number("plain.dat") is None


def test_series_page_is_curve_based_and_versions_the_recipe() -> None:
    app, window, binding = _window()
    workspace = window.components.fitting_workspace
    page = workspace.insitu_series_page
    assert isinstance(page.ui, InSituSeriesPageView)
    workspace.show_context("insitu")
    app.processEvents()
    controls = page.ui.workflowControls
    assert [page.ui.previewTabs.tabText(index) for index in range(3)] == ["Preview", "Frames", "Log"]
    assert controls.sequencePatternEdit.text() == "*_fit_input.dat"
    assert not controls.recursiveCheckBox.isChecked()
    assert tuple(page.ui.workflowButtons) == ("source", "fit", "results")
    assert not controls.applyRecipeButton.isEnabled()
    assert not page.ui.startProcessButton.isEnabled()

    page.render_recipe(_plain_recipe(binding))
    assert page.ui.startProcessButton.isEnabled()
    controls.workflowModeCombo.setCurrentIndex(2)  # plot curves only
    controls.failurePolicyCombo.setCurrentText("Stop")
    QTest.mouseClick(controls.applyRecipeButton, Qt.LeftButton)
    revised = binding.fitting_view_model.insitu.recipe
    assert revised.version == 2 and revised.parent_version == 1
    assert revised.model["extract_only"] is True
    assert revised.fitting.failure == "stop"
    assert revised.cut["source"] == "curve"
    assert binding._insitu_workflow_settings()["auto_fit"] is False

    controls.runModeCombo.setCurrentText("Live Watch")
    app.processEvents()
    assert not controls.liveSettingsWidget.isHidden()
    assert controls.sequenceSettingsWidget.isHidden()
    assert not page.ui.startWatchButton.isHidden() and page.ui.startProcessButton.isHidden()
    controls.runModeCombo.setCurrentText("Process Existing Sequence")
    app.processEvents()
    assert page.ui.startWatchButton.isHidden() and not page.ui.startProcessButton.isHidden()
    window.close()


def test_series_page_fits_supported_viewports() -> None:
    app, window, _binding = _window()
    workspace = window.components.fitting_workspace
    page = workspace.insitu_series_page
    window.menus.show_workspace("fitting")
    workspace.show_context("insitu")
    for width, height in ((1280, 800), (1440, 900), (1920, 1080)):
        window.resize(width, height)
        QTest.qWait(40)
        app.processEvents()
        assert page.ui.startProcessButton.isVisible()
        browse = page.ui.workflowControls.sequenceBrowseButton
        browse_right = browse.mapTo(page, browse.rect().bottomRight()).x()
        viewport = page.ui.settingsScrollArea.viewport()
        assert browse_right <= viewport.mapTo(page, viewport.rect().bottomRight()).x()
        assert page.ui.settingsScrollArea.horizontalScrollBar().maximum() == 0
        assert page.ui.jobStatus.mapTo(page, QPoint(0, page.ui.jobStatus.height())).y() <= page.height()
    window.close()


def test_single_curve_is_captured_as_the_series_recipe(tmp_path: Path) -> None:
    _app_, window, binding = _window()
    curve = _write_curve(tmp_path / "run_00001_fit_input.dat")
    binding.import_1d_file(curve)
    page = window.components.fitting_workspace.insitu_series_page
    QTest.mouseClick(page.ui.captureRecipeButton, Qt.LeftButton)
    recipe = binding.fitting_view_model.insitu.recipe
    assert recipe.version == 1 and recipe.cut["source"] == "curve"
    assert recipe.note.endswith("run_00001_fit_input.dat")
    assert Path(page.ui.workflowControls.sequenceFolderEdit.text()) == tmp_path
    assert binding._current_ui_matches_insitu_recipe()

    binding._save_workflow_options({**binding._workflow_options(), "relative_noise": 0.2})
    assert not binding._current_ui_matches_insitu_recipe()
    QTest.mouseClick(page.ui.captureRecipeButton, Qt.LeftButton)
    assert binding.fitting_view_model.insitu.recipe.version == 2
    window.close()


def test_existing_curves_are_processed_in_order_and_a_bad_file_does_not_stop_the_run(
    tmp_path: Path,
) -> None:
    _app_, window, binding = _window()
    for index in (1, 2, 10):
        _write_curve(tmp_path / f"run_{index:05d}_fit_input.dat", scale=index)
    (tmp_path / "run_00011_fit_input.dat").write_text("not a curve\n", encoding="ascii")
    (tmp_path / "run_00001_horizontal.csv").write_text("x,y\n", encoding="ascii")  # not a fit input
    binding.import_1d_file(tmp_path / "run_00001_fit_input.dat")
    page = window.components.fitting_workspace.insitu_series_page
    QTest.mouseClick(page.ui.captureRecipeButton, Qt.LeftButton)
    controls = page.ui.workflowControls
    controls.workflowModeCombo.setCurrentIndex(2)  # plot curves only: no fitting job
    QTest.mouseClick(controls.applyRecipeButton, Qt.LeftButton)
    controls.sequenceStartSpinBox.setValue(2)  # skip run_00001

    binding._start_insitu_sequence_processing()
    deadline = time.monotonic() + 30
    while binding._insitu_workflow_state != "Idle" and time.monotonic() < deadline:
        QTest.qWait(50)

    rows = binding._load_insitu_session_records()
    assert [row["file_name"] for row in rows] == [
        "run_00002_fit_input.dat",
        "run_00010_fit_input.dat",
        "run_00011_fit_input.dat",
    ]
    assert [row["load_status"] for row in rows] == ["ok", "ok", "failed"]
    assert binding._insitu_workflow_processed_count == 3
    assert binding._insitu_workflow_failed_count == 1
    assert binding._insitu_heatmap_count == 2
    # The last good curve is the current data, with its pixel counts.
    assert Path(binding.current_1d_data["file_path"]).name == "run_00010_fit_input.dat"
    assert binding.current_1d_data["pixels"] is not None
    window.close()
