"""Labs / 2D Prediction: worker thread, thread-safe log, readiness, preview layout, themed figures."""

from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPoint, QRect, QThread, QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QGridLayout, QLayout, QWidget

from main import MainWindow
from src.gimap.app import AppContext
from src.gimap.app.presentation.components import visible_toasts
from src.gimap.app.presentation.theme import theme_color, theme_manager
from src.gimap.features.prediction.presentation.bindings.prediction_execution import (
    BUSY_TEXT,
    INPUT_CHANGED_TEXT,
    STOPPING_TEXT,
)
from src.gimap.features.prediction.presentation.bindings.prediction_results import (
    PredictionResultsMixin,
)
from src.gimap.features.prediction.presentation.preview_layout import ResponsivePreviewBody
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


def _window() -> MainWindow:
    _app()
    context = AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        jobs=LocalProcessJobRunner(),
    )
    window = MainWindow(context)
    window.resize(1280, 800)
    # The feature runtimes start from timers once the shell exists.
    _wait(lambda: getattr(window, "_initialization_completed", False))
    _wait(lambda: getattr(window.runtime.prediction, "_initialized", False))
    return window


def _wait(condition, seconds: float = 10.0) -> None:
    app = _app()
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


class _SlowViewModel:
    """The real ViewModel, except that preprocessing and the model call are fakes (1 s model)."""

    def __init__(self, real, delay: float = 1.0) -> None:
        self._real = real
        self.delay = delay
        self.threads: list[threading.Thread] = []
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self._real, name)

    def prepare_input(self, image, module):
        self.threads.append(threading.current_thread())
        return SimpleNamespace(values=np.ones((1, 8, 8, 1), dtype=np.float32), steps=())

    def predict_prepared(self, values, module, model_path, steps=()):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
        finally:
            with self._lock:
                self.active -= 1
        self.threads.append(threading.current_thread())
        return SimpleNamespace(outputs={"scalars": np.array([0.25], dtype=np.float32)})


def _ready_binding(window, tmp_path):
    binding = window.runtime.prediction
    source = tmp_path / "frame_00001.cbf"
    source.write_bytes(b"")
    binding.current_parameters.update(
        {"mode": "single_file", "input_file": str(source), "module_model_path": str(tmp_path / "model")}
    )
    binding._current_image = np.ones((8, 8), dtype=np.float32)
    binding._current_image_path = str(source)
    binding._current_module = {"_prediction_module": object()}
    binding._current_model = object()
    binding._framework_ready = lambda: True
    displayed = []
    binding._display_prediction = lambda outputs: displayed.append(outputs)
    binding.prediction_view_model = _SlowViewModel(binding.prediction_view_model)
    return binding, displayed


def test_single_prediction_runs_in_a_worker_and_keeps_the_window_responsive(tmp_path):
    window = _window()
    binding, displayed = _ready_binding(window, tmp_path)
    button = window.gisaxsPredictPredictButton
    label = window.predictionCanvasStatusLabel
    seen_while_running = []

    started = time.monotonic()
    QTimer.singleShot(100, lambda: seen_while_running.append(
        (binding._prediction_active, button.isEnabled(), label.text())
    ))
    binding._execute_prediction()

    assert time.monotonic() - started < 0.5  # the model call does not block the GUI thread
    assert binding._prediction_active
    assert not button.isEnabled()
    assert label.text() == BUSY_TEXT
    assert not window.gisaxsPredictStopButton.isHidden()

    _wait(lambda: not binding._prediction_active)

    assert seen_while_running == [(True, False, BUSY_TEXT)]  # a QTimer fired during the prediction
    assert len(displayed) == 1 and float(displayed[0]["scalars"][0]) == pytest.approx(0.25)
    assert all(thread is not threading.main_thread() for thread in binding.prediction_view_model.threads)
    assert label.text() != BUSY_TEXT
    assert window.gisaxsPredictStopButton.isHidden()
    window.close()


def test_stop_discards_the_late_result_and_keeps_one_model_run_at_a_time(tmp_path):
    window = _window()
    binding, displayed = _ready_binding(window, tmp_path)
    button = window.gisaxsPredictPredictButton
    binding._execute_prediction()
    worker = next(iter(binding._prediction_workers.values()))
    # Stop while the model process runs (Stop during preprocessing starts no model process).
    _wait(lambda: binding.prediction_view_model.active == 1, 5)

    binding._stop_gisaxs_predict()

    # The model process cannot be interrupted: the run stays active until it reports back.
    assert binding._prediction_active
    assert not button.isEnabled() and button.text() == "Stopping…"
    assert window.gisaxsPredictStopButton.isHidden()
    assert window.predictionCanvasStatusLabel.text() == STOPPING_TEXT
    for _ in range(2):  # Predict, Stop, Predict, Stop ... start nothing new meanwhile
        binding._execute_prediction()
        binding._stop_gisaxs_predict()
    assert list(binding._prediction_workers.values()) == [worker]

    worker.wait(5000)
    _wait(lambda: not binding._prediction_active)

    assert displayed == []
    assert "Ignored the result of a stopped prediction." in window.predictStatusTextBrowser.toPlainText()
    assert button.isEnabled() and button.text() == "Predict"
    assert window.predictionCanvasStatusLabel.text() != STOPPING_TEXT
    view_model = binding.prediction_view_model
    assert view_model.calls == 1 and view_model.max_active == 1

    binding._execute_prediction()  # the next run starts only now
    _wait(lambda: not binding._prediction_active)
    assert view_model.calls == 2 and view_model.max_active == 1
    assert len(displayed) == 1
    window.close()


@pytest.mark.parametrize(
    "change",
    [
        lambda b, tmp: setattr(b, "_current_image", np.zeros((8, 8), dtype=np.float32)),
        lambda b, tmp: setattr(b, "_current_module", {"_prediction_module": object()}),
        lambda b, tmp: b.current_parameters.update(module_model_path=str(tmp / "other_model")),
        lambda b, tmp: setattr(b, "_current_model", None),
    ],
    ids=["new frame", "module switched", "other model", "model unloaded"],
)
def test_a_result_is_discarded_when_the_input_changed_during_the_run(tmp_path, change):
    window = _window()
    binding, displayed = _ready_binding(window, tmp_path)
    binding.prediction_view_model.delay = 0.4
    binding._execute_prediction()

    change(binding, tmp_path)
    _wait(lambda: not binding._prediction_active)

    assert displayed == []  # one result view never mixes two frames, modules or models
    assert f"[WARN] {INPUT_CHANGED_TEXT}" in window.predictStatusTextBrowser.toPlainText()
    toasts = [t for t in visible_toasts(window.gisaxsPredictPage) if t.property("level") == "warning"]
    assert toasts and toasts[-1].text() == INPUT_CHANGED_TEXT
    window.close()


def test_activity_log_from_a_python_thread_is_queued_to_the_gui_thread():
    window = _window()
    binding = window.runtime.prediction
    browser = window.predictStatusTextBrowser
    app = _app()
    on_gui_thread = []
    original_append = browser.append

    def checked_append(text):
        on_gui_thread.append(QThread.currentThread() is app.thread())
        original_append(text)

    browser.append = checked_append
    worker = threading.Thread(
        target=lambda: binding._append_status_message("line from a worker", level="WARN")
    )
    worker.start()
    worker.join()

    assert "line from a worker" not in browser.toPlainText()  # queued, nothing touched yet
    app.processEvents()
    assert "[WARN] line from a worker" in browser.toPlainText()
    assert on_gui_thread == [True]
    window.close()


def test_readiness_chips_are_muted_while_pending_and_error_only_after_a_failed_load(tmp_path):
    window = _window()
    binding = window.runtime.prediction
    binding._refresh_predict_readiness()

    assert not hasattr(window, "gisaxsPredictModeLabel")
    assert window.gisaxsPredictInputReadyLabel.text() == "Input: Missing"
    assert window.gisaxsPredictInputReadyLabel.property("gimapRole") == "muted"
    assert window.gisaxsPredictModelReadyLabel.text() == "Model: Not loaded"
    assert window.gisaxsPredictModelReadyLabel.property("gimapRole") == "muted"
    assert not window.gisaxsPredictPredictButton.isEnabled()
    assert not window.gisaxsPredictReadinessHint.isHidden()
    assert window.gisaxsPredictRunLogTitle.isHidden()

    model_path = str(tmp_path / "broken_model")
    binding.current_parameters["module_model_path"] = model_path
    binding._model_loading = True
    binding._on_model_load_finished(None, "SavedModel not found", model_path)

    assert window.gisaxsPredictModelReadyLabel.property("gimapRole") == "error"
    assert window.gisaxsPredictModelReadyLabel.text() == "Model: Load failed"
    toasts = [toast for toast in visible_toasts(window.gisaxsPredictPage) if toast.property("level") == "error"]
    assert toasts and "SavedModel not found" in toasts[-1].text()
    assert toasts[-1].action_button.text() == "Show Log"
    assert window.predictionActivityDisclosure.content.isHidden()
    toasts[-1].action_button.click()
    assert not window.predictionActivityDisclosure.content.isHidden()
    window.close()


def test_show_log_scrolls_the_whole_activity_log_into_view():
    window = _window()  # 1280 x 800
    window.show()
    binding = window.runtime.prediction
    window.runtime.navigate("predict")
    for index in range(40):
        binding._append_status_message(f"log line {index}")
    _app().processEvents()

    binding._show_activity_log()
    _wait(lambda: False, 0.2)  # the deferred pass after the layout has grown

    viewport = window.gisaxsPredictCanvasScrollArea.viewport()
    browser = window.predictStatusTextBrowser
    shown = QRect(browser.mapTo(viewport, QPoint(0, 0)), browser.size())
    assert browser.height() < viewport.height()
    assert viewport.rect().contains(shown), (viewport.rect(), shown)
    window.close()


def test_preview_inspector_moves_under_the_view_when_the_page_is_narrow():
    _app()
    page = QWidget()
    # Like a tab page, whose width the tab widget decides (a top-level window would not shrink).
    QGridLayout(page).setSizeConstraint(QLayout.SetNoConstraint)
    view, panel = QWidget(page), QWidget(page)
    view.setMinimumSize(360, 280)
    panel.setMinimumWidth(270)
    body = ResponsivePreviewBody(page, view, panel)
    page.layout().addLayout(body.box, 1, 0)
    changes = []
    body.layout_changed.connect(lambda: changes.append(body.stacked))
    page.show()

    for width, stacked in ((520, True), (900, False), (600, True)):
        page.resize(width, 700)
        _app().processEvents()
        assert body.stacked is stacked, width
        view_box, panel_box = view.geometry(), panel.geometry()
        assert not view_box.intersects(panel_box), (width, view_box, panel_box)
        assert panel_box.right() <= page.width()
    assert changes[-3:] == [True, False, True]
    assert body.stacked_extra_height() > 0
    page.close()


class _FigureHost(PredictionResultsMixin):
    current_parameters = {"colormap": "viridis"}
    _DEFAULT_COLORMAPS = ("viridis",)

    @staticmethod
    def _auto_scale_values(image):
        return float(np.nanmin(image)), float(np.nanmax(image))

    def _append_status_message(self, message, level="INFO"):
        raise AssertionError(message)


def test_a_shown_distribution_is_rendered_again_when_the_theme_changes():
    rendered = []
    host = _FigureHost()
    host.prediction_results = {"hr": np.ones((4, 4))}
    host._predict_tab_specs = [{"kind": "steps"}, {"kind": "hr"}]
    host._predict_tabs = SimpleNamespace(currentIndex=lambda: 1)
    host._render_predict_tab_by_index = rendered.append

    host._on_theme_changed("dark")
    assert rendered == [1]
    host._predict_tabs = SimpleNamespace(currentIndex=lambda: 0)  # the step gallery is not redrawn
    host._on_theme_changed("light")
    assert rendered == [1]


def test_result_figure_follows_the_dark_theme_and_keeps_the_colour_bar_inside():
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    _app()
    manager = theme_manager()
    mode = manager.mode
    try:
        manager.mode = "dark"  # the colours the figure reads; no restyle of every open widget
        y, x = np.mgrid[0:64, 0:64]
        image = np.exp(-((x - 30) ** 2 + (y - 22) ** 2) / 60.0)
        host = _FigureHost()
        pixmap = host._render_hr_figure(image, target_pixels=(480, 420))
        corner = pixmap.toImage().pixelColor(3, 3)
        assert corner.name() == theme_color("surface").name()

        for target in ((400, 320), (900, 700)):
            figure = host._hr_figure(image, target_pixels=target)
            renderer = FigureCanvasAgg(figure).get_renderer()
            figure.draw(renderer)
            bounds = figure.bbox
            colour_bar = figure.axes[-1]
            texts = [label for label in colour_bar.get_yticklabels() if label.get_text()]
            texts.append(colour_bar.yaxis.label)
            assert colour_bar.yaxis.label.get_text() == "Probability"
            for text in texts:
                extent = text.get_window_extent(renderer)
                assert extent.x0 >= bounds.x0 and extent.x1 <= bounds.x1, (target, text.get_text())
                assert extent.y0 >= bounds.y0 and extent.y1 <= bounds.y1, (target, text.get_text())
    finally:
        manager.mode = mode


def test_image_export_saves_the_light_figure_while_the_view_follows_the_dark_theme(
    tmp_path, monkeypatch
):
    window = _window()
    binding = window.runtime.prediction
    manager = theme_manager()
    mode = manager.mode
    try:
        manager.mode = "dark"
        y, x = np.mgrid[0:32, 0:32]
        binding._display_prediction({"hr": np.exp(-((x - 15) ** 2 + (y - 11) ** 2) / 30.0)})
        shown = binding._predict_pixmap
        assert shown.toImage().pixelColor(3, 3).name() == theme_color("surface").name()

        # Wave 2b (labs-15): one save dialog, a PNG (not JPG) named by the user, data and record beside it.
        monkeypatch.setattr(binding, "_ask_save_path", lambda *_args: str(tmp_path / "frame_prediction.png"))
        binding._on_predict_export_clicked()

        saved = list(tmp_path.glob("*.png"))
        assert [path.name for path in saved] == ["frame_prediction.png"]
        exported = QImage(str(saved[0]))
        assert exported.size() == shown.size()
        corner = exported.pixelColor(3, 3)
        assert min(corner.red(), corner.green(), corner.blue()) >= 245  # white, not the dark surface
        assert binding._predict_pixmap is shown  # the view keeps the themed figure
    finally:
        manager.mode = mode
        window.close()
