"""Labs / Trainset: draw-mode feedback, Reset with Undo, validation badge state, small fixes."""

from __future__ import annotations

import os
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QWidget

from main import MainWindow
from src.gimap.app import AppContext
from src.gimap.app.presentation.components import visible_toasts
from src.gimap.app.presentation.i18n import translate
from src.gimap.app.presentation.theme import theme_color, theme_manager
from src.gimap.features.trainset.presentation.bindings import validation_files
from src.gimap.features.trainset.presentation.page import ArrayCanvas, TrainsetBuildPage
from src.gimap.integrations.jobs import LocalProcessJobRunner
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)

_TEST_APP = None


def _app() -> QApplication:
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


def _wait(condition, seconds: float = 10.0) -> None:
    app = _app()
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def _settle(seconds: float) -> None:
    _wait(lambda: False, seconds)


def _trainset(tmp_path):
    _app()
    context = AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        jobs=LocalProcessJobRunner(),
    )
    window = MainWindow(context)
    window.resize(1280, 800)
    _wait(lambda: getattr(window, "_initialization_completed", False))
    binding = window.runtime.trainset
    _wait(lambda: binding._initialized)
    reference = tmp_path / "reference.npy"
    y, x = np.mgrid[0:96, 0:120]
    np.save(reference, (100.0 * np.exp(-((x - 60) ** 2 + (y - 70) ** 2) / 90.0) + 1.0).astype(np.float32))
    binding._load_reference(str(reference))
    return window, binding, binding.page


def test_escape_leaves_a_draw_mode_and_the_canvas_shows_what_to_do():
    _app()
    canvas = ArrayCanvas()
    canvas.resize(420, 320)
    canvas.set_data(np.linspace(2.0, 500.0, 120 * 90).reshape(90, 120))
    canvas.show()

    canvas.set_draw_mode("beam_center")
    picture = canvas.grab().toImage()
    assert picture.pixelColor(canvas.width() // 2, 6).name() == theme_color("info_soft").name()
    assert canvas._image_rect().top() >= canvas._instruction_height()  # the strip covers no pixel

    canvas._press = canvas.rect().center()
    QTest.keyClick(canvas, Qt.Key_Escape)

    assert canvas.mode == ""
    assert canvas._press is None
    assert canvas.focusPolicy() == Qt.StrongFocus
    canvas.close()


def test_colour_bar_shows_the_displayed_limits_in_theme_colours():
    _app()
    manager = theme_manager()
    mode = manager.mode
    try:
        manager.mode = "dark"
        canvas = ArrayCanvas()
        canvas.resize(420, 320)
        canvas.set_data(np.linspace(2.0, 500.0, 120 * 90).reshape(90, 120))
        canvas.set_display_options("viridis", False, False, 10.0, 400.0)
        canvas.grab()
        assert canvas.colorbar_labels == ("10", "400")
        canvas.set_display_options("viridis", True, False, 5.0, 250.0)  # log: the linear limits
        canvas.grab()
        assert canvas.colorbar_labels == ("5", "250")
        canvas.close()
    finally:
        manager.mode = mode


def test_colour_bar_limits_of_detector_counts_are_not_cut():
    _app()
    canvas = ArrayCanvas()
    canvas.resize(420, 320)
    canvas.set_data(np.linspace(-20000.0, 700000.0, 120 * 90).reshape(90, 120))
    metrics = canvas.fontMetrics()
    image_right = None
    for low, high, labels in (
        (-12345.0, 612345.0, ("-1.23e+04", "6.12e+05")),
        (0.000123, 4100.0, ("0.000123", "4.1e+03")),
        (10.0, 400.0, ("10", "400")),
    ):
        canvas.set_display_options("viridis", False, False, low, high)
        picture = canvas.grab().toImage()
        assert canvas.colorbar_labels == labels
        for label in labels:
            assert metrics.horizontalAdvance(label) <= canvas._colorbar_text_width(), label
        rect = canvas._image_rect()
        bar_x = rect.right() + 10  # the label column ends 6 px before the canvas edge
        assert bar_x + 13 + 4 + canvas._colorbar_text_width() <= canvas.width() - 6
        assert picture.width() == canvas.width()
        # The image keeps its place whatever the limits (the column fits the widest limit).
        assert image_right in (None, rect.right())
        image_right = rect.right()
    canvas.close()


def test_reset_offers_undo_that_restores_reference_roi_and_particles(tmp_path):
    window, binding, page = _trainset(tmp_path)
    page.fields["roi.x"].setValue(11)
    page.fields["roi.width"].setValue(40)
    page.particle_parameter_table.item(0, 3).setText("7.5")
    before = binding._collect_config()
    assert before["project"]["reference_file"].endswith("reference.npy")

    page.reset_defaults_button.click()

    reset = binding._collect_config()
    assert reset["project"]["reference_file"] != before["project"]["reference_file"]
    assert binding.reference_image is None
    toasts = [toast for toast in visible_toasts(page) if toast.text() == "Trainset reset to defaults"]
    assert toasts and toasts[-1].action_button.text() == "Undo"

    toasts[-1].action_button.click()

    restored = binding._collect_config()
    assert restored["project"]["reference_file"] == before["project"]["reference_file"]
    assert restored["roi"] == before["roi"]
    assert restored["sample"]["particles"] == before["sample"]["particles"]
    assert binding.reference_image is not None
    window.close()


def _gates(page):
    table = page.preview_gate_table
    return [table.item(row, 1).text() for row in range(table.rowCount())]


def test_undo_restores_the_progress_the_badge_speaks_of(tmp_path):
    window, binding, page = _trainset(tmp_path)
    for row in range(3):
        page.preview_gate_table.item(row, 1).setText("Ready")
    page.set_validation_state("Preview ready", "ok")
    page.set_step_state(1, "Preview ready")
    page.set_step_state(2, "Contract ready")
    steps, gates, stages = page.step_states(), _gates(page), page.design_stages_ready()
    assert steps[1] == "Preview ready" and gates[:3] == ["Ready"] * 3

    page.reset_defaults_button.click()

    assert page.step_states() == ["Not started"] * len(page.STEPS)
    assert _gates(page)[:3] == ["Pending"] * 3  # the old design's checks no longer hold
    assert page.validation_state() == "pending"

    toast = [t for t in visible_toasts(page) if t.text() == "Trainset reset to defaults"][-1]
    toast.action_button.click()

    assert page.step_states() == steps
    assert _gates(page) == gates
    assert page.design_stages_ready() == stages
    assert page.validation_badge.text() == "Preview ready"
    assert page.validation_state() == "ok"
    window.close()


def test_edit_after_validation_marks_the_badge_changed_but_loading_a_project_does_not(
    tmp_path, monkeypatch
):
    window, binding, page = _trainset(tmp_path)
    assert page.validation_state() == "pending"
    monkeypatch.setattr(binding.trainset_view_model, "validate_config", lambda *a, **k: (True, [], []))
    monkeypatch.setattr(validation_files.QMessageBox, "information", lambda *a, **k: None)

    assert binding._validate_and_report()
    assert page.validation_state() == "ok"
    assert page.validation_badge.text() == "Configuration valid"

    page.particle_parameter_table.item(0, 3).setText("6.5")

    assert page.validation_state() == "warn"
    assert page.validation_badge.text() == "Changed since validation"
    assert page.preview_gate_table.item(0, 1).text() == "Pending"

    project = tmp_path / "project.yaml"
    binding.trainset_view_model.save_project(binding._collect_config(), project)
    assert binding._validate_and_report()
    monkeypatch.setattr(
        validation_files.QFileDialog, "getOpenFileName", lambda *a, **k: (str(project), "")
    )
    binding._load_project_dialog()
    _settle(0.3)

    assert page.validation_state() != "warn"
    assert page.validation_badge.text() == "Not validated"
    window.close()


def test_step_list_keeps_the_state_line_and_the_log_toggle_is_a_run_log():
    _app()
    page = TrainsetBuildPage()
    page.resize(1096, 700)
    page.show()
    page.set_step_state(0, "Reference loaded")
    page.step_list.setCurrentRow(0)
    _app().processEvents()

    # The steps are the shared StepRail (wave 2b, labs-8): the current step keeps its state line.
    rail = page.step_list
    assert rail.currentRow() == 0
    assert (rail.state("dataset"), rail.detail("dataset")) == ("ok", "Reference loaded")
    step = rail.findChild(QWidget, "gimapStep_dataset")
    assert step.property("current") is True
    assert step.detail.isVisible() and step.detail.height() >= step.detail.fontMetrics().lineSpacing()
    assert page._monitor_page_ui.logToggle.text() == "Run Log"
    assert translate("Run Log", "zh") == "运行日志"
    assert "font-weight: 600" not in page.styleSheet().split("QTabBar::tab:selected")[1].split("}")[0]
    page.close()
