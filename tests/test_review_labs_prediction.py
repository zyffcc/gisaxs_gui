"""Review fixes, Labs / 2D Prediction: module refresh during a run, Stop before the model
process, and quitting while a single-file prediction runs."""

from __future__ import annotations

import dataclasses
import threading
import time
from types import SimpleNamespace

import numpy as np

from PyQt5.QtCore import QEvent, QPoint, Qt
from PyQt5.QtGui import QFocusEvent, QMouseEvent

from src.gimap.features.prediction.presentation.bindings.prediction_execution import (
    INPUT_CHANGED_TEXT,
)
from tests.test_labs_prediction import _ready_binding, _wait, _window


def _select_real_module(binding, window):
    """Select a module discovered from modules/ (a typed PredictionModule read from its YAML)."""
    names = sorted(binding._modules_by_name)
    assert names, "the repository ships prediction modules"
    name = names[0]
    window.gisaxsPredictModuleSelectCombox.setCurrentText(name)
    binding._on_module_selected(name)
    return name


def test_clicking_or_focusing_the_module_list_during_a_run_keeps_the_result(tmp_path):
    window = _window()
    binding = window.runtime.prediction
    name = _select_real_module(binding, window)
    spec = binding._current_module
    binding, displayed = _ready_binding(window, tmp_path)
    binding._current_module = spec  # the real module, not a placeholder object
    binding.current_parameters["module_name"] = name
    binding.prediction_view_model.delay = 0.6
    combo = window.gisaxsPredictModuleSelectCombox

    binding._execute_prediction()
    assert binding._prediction_active
    started_with = next(iter(binding._prediction_workers.values())).module
    # What Qt delivers when the user clicks the list or tabs onto it: the modules are re-read.
    press = QMouseEvent(QEvent.MouseButtonPress, QPoint(5, 5), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    binding.eventFilter(combo, press)
    binding.eventFilter(combo, QFocusEvent(QEvent.FocusIn, Qt.TabFocusReason))
    assert binding._current_module["_prediction_module"] is not started_with  # new, equal object
    _wait(lambda: not binding._prediction_active)

    assert len(displayed) == 1
    assert INPUT_CHANGED_TEXT not in window.predictStatusTextBrowser.toPlainText()
    window.close()


def test_a_module_yaml_edited_during_the_run_still_discards_the_result(tmp_path):
    window = _window()
    binding = window.runtime.prediction
    _select_real_module(binding, window)
    spec = binding._current_module
    binding, displayed = _ready_binding(window, tmp_path)
    binding._current_module = spec
    binding.prediction_view_model.delay = 0.4

    binding._execute_prediction()
    module = spec["_prediction_module"]
    edited = dataclasses.replace(module, version=f"{module.version}-edited")
    binding._current_module = {**spec, "_prediction_module": edited}  # a saved module.yaml edit
    _wait(lambda: not binding._prediction_active)

    assert displayed == []
    assert f"[WARN] {INPUT_CHANGED_TEXT}" in window.predictStatusTextBrowser.toPlainText()
    window.close()


class _BlockingPreprocessing:
    """Preprocessing waits for the test; the model call is counted."""

    def __init__(self, real) -> None:
        self._real = real
        self.release = threading.Event()
        self.entered = threading.Event()
        self.model_calls = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def prepare_input(self, image, module):
        self.entered.set()
        self.release.wait(10)
        return SimpleNamespace(values=np.ones((1, 8, 8, 1), dtype=np.float32), steps=())

    def predict_prepared(self, values, module, model_path, steps=()):
        self.model_calls += 1
        return SimpleNamespace(outputs={"scalars": np.array([0.5], dtype=np.float32)})


def test_stop_during_preprocessing_starts_no_model_process(tmp_path):
    window = _window()
    binding, displayed = _ready_binding(window, tmp_path)
    fake = _BlockingPreprocessing(binding.prediction_view_model)
    binding.prediction_view_model = fake

    binding._execute_prediction()
    assert fake.entered.wait(5)
    assert binding.prediction_running()  # the quit question names the 2D prediction
    binding._stop_gisaxs_predict()
    assert not binding.prediction_running()  # stopped by the user: nothing to ask about
    fake.release.set()
    _wait(lambda: not binding._prediction_active)

    assert fake.model_calls == 0
    assert displayed == []
    window.close()


class _JobRunner:
    """The job runner part that matters at quit: shutdown() cancels the registered jobs."""

    def __init__(self) -> None:
        self.cancel = threading.Event()
        self.shutdowns = 0

    def shutdown(self) -> None:
        self.shutdowns += 1
        self.cancel.set()


class _LateModelJob:
    """A model job registered only after the window shut the job runner down."""

    def __init__(self, real, runner: _JobRunner) -> None:
        self._real = real
        self.context = SimpleNamespace(jobs=runner)
        self.runner = runner
        self.model_started = threading.Event()
        self.cancelled = False

    def __getattr__(self, name):
        return getattr(self._real, name)

    def prepare_input(self, image, module):
        return SimpleNamespace(values=np.ones((1, 8, 8, 1), dtype=np.float32), steps=())

    def predict_prepared(self, values, module, model_path, steps=()):
        self.model_started.set()
        self.cancelled = self.runner.cancel.wait(20)  # a long model run unless it is cancelled
        return None


def test_quitting_during_a_prediction_waits_for_the_worker_and_cancels_a_late_model_job(tmp_path):
    window = _window()
    binding, displayed = _ready_binding(window, tmp_path)
    runner = _JobRunner()
    fake = _LateModelJob(binding.prediction_view_model, runner)
    binding.prediction_view_model = fake

    binding._execute_prediction()
    assert fake.model_started.wait(5)
    worker = next(iter(binding._prediction_workers.values()))
    started = time.monotonic()

    binding._wait_for_prediction_workers()  # aboutToQuit

    assert time.monotonic() - started < 5
    assert fake.cancelled and runner.shutdowns >= 1
    assert worker.discard and not worker.isRunning()  # destroying the window cannot abort now
    assert displayed == []
    window.close()


def test_stop_predictions_before_the_job_runner_shuts_down_skips_the_model_call(tmp_path):
    window = _window()
    binding, displayed = _ready_binding(window, tmp_path)
    fake = _BlockingPreprocessing(binding.prediction_view_model)
    binding.prediction_view_model = fake

    binding._execute_prediction()
    assert fake.entered.wait(5)
    binding.stop_predictions()  # MainWindowComponents.shutdown(), before jobs.shutdown()
    fake.release.set()
    binding._wait_for_prediction_workers()

    assert fake.model_calls == 0
    assert all(not worker.isRunning() for worker in binding._prediction_workers.values())
    _wait(lambda: not binding._prediction_active)
    assert displayed == []
    window.close()
