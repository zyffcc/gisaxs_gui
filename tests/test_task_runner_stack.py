"""Background threads have room for numerical work.

macOS gives a thread that is not the main one a 512 KB stack unless it is asked for more. numpy.linalg.inv
and solve (and with them Matplotlib's transforms whenever a picture is saved) need more than that: the
process then ends with a bus error, which no ``try`` can catch. Every Qt thread that computes therefore asks
for ``WORKER_STACK_BYTES``: 16 MB on macOS, the system's own size (0) elsewhere. Each test checks the size
before it lets the thread compute, so a regression is a failed assertion and not a dead test run.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtCore import QCoreApplication, QEventLoop, QThread, QTimer
from PyQt5.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
ENOUGH = 16 * 1024 * 1024 if sys.platform == "darwin" else 0
"""What a thread must have asked for on this system before it may compute (0: the system's size is enough)."""


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _invert() -> float:
    return float(np.linalg.inv(np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]))[0, 0])


def test_the_stack_asked_for_is_the_one_python_gives_its_own_threads() -> None:
    """16 MB on macOS (what CPython asks for its own threads there); elsewhere nothing is asked for: Linux
    gives 8 MB, Windows 1 MB, and Qt 5 on Windows would commit, not just reserve, the size asked for."""
    from src.gimap.app.presentation.task_runner import WORKER_STACK_BYTES

    assert WORKER_STACK_BYTES == (16 * 1024 * 1024 if sys.platform == "darwin" else 0) == ENOUGH


def test_a_pool_task_can_call_lapack_and_save_a_figure(tmp_path: Path) -> None:
    from src.gimap.app.presentation.task_runner import TaskRunner

    _app()
    tasks = TaskRunner()
    try:
        assert tasks._pool.stackSize() == ENOUGH  # first: without it the next lines end the interpreter on macOS

        def picture() -> int:
            from src.gimap.shared.figures import publication_figure, save_figure

            figure = publication_figure()
            axes = figure.add_subplot()
            axes.imshow(np.arange(12.0).reshape(3, 4))
            axes.set_xlabel("x (pixel)")
            return save_figure(figure, tmp_path / "frame.png").stat().st_size

        results: dict[str, object] = {}
        errors: list[str] = []
        tasks.submit("inverse", _invert, on_done=lambda value: results.update(inverse=value),
                     on_error=lambda message, _details: errors.append(message))
        tasks.submit("picture", picture, on_done=lambda value: results.update(picture=value),
                     on_error=lambda message, _details: errors.append(message))
        assert tasks.wait(60) and not errors, errors
        assert results["inverse"] == pytest.approx(0.5) and results["picture"] > 0
    finally:
        tasks.shutdown()


def test_the_prediction_threads_ask_for_the_stack_and_can_compute() -> None:
    from src.gimap.features.prediction.presentation.image_worker import PredictionImageLoader
    from src.gimap.features.prediction.presentation.prediction_worker import PredictionWorker

    _app()

    class Model:  # what the workers call: here both invert a matrix on their thread
        state = None

        def load_stack(self, _path, _count):
            raise AssertionError("not started in this test")

        def prepare_input(self, _image, _module):
            from types import SimpleNamespace

            return SimpleNamespace(values=_invert(), steps=())

        def predict_prepared(self, values, _module, _path, _steps):
            return values

    loader = PredictionImageLoader(Model())
    assert loader.stackSize() == ENOUGH
    worker = PredictionWorker(Model(), image=None, typed_module=None, model_path="model.h5", run_id=7)
    assert worker.stackSize() == ENOUGH
    finished: list[tuple] = []
    worker.prediction_finished.connect(lambda *values: finished.append(values))
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    QTimer.singleShot(30000, loop.quit)
    worker.start()
    loop.exec_()
    assert worker.wait(5000)
    QCoreApplication.processEvents()
    assert finished and finished[0][0] == 7 and finished[0][3] == "" and finished[0][2] == pytest.approx(0.5)


def _threads_without_a_stack(path: Path) -> list[int]:
    """Lines of ``name = QThread(...)`` in ``path`` whose function never calls ``name.setStackSize(...)``."""
    missing = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for function in (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
        made, sized = {}, set()
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                    and getattr(node.value.func, "id", "") == "QThread":
                made[ast.unparse(node.targets[0])] = node.lineno
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "setStackSize":
                sized.add(ast.unparse(node.func.value))
        missing += [line for name, line in made.items() if name not in sized]
    return sorted(missing)


@pytest.mark.parametrize("module", [
    "src/gimap/features/fitting/presentation/bindings/ai_job_execution.py",
    "src/gimap/features/calibration/presentation/bindings/calibration_run.py",
])
def test_the_threads_of_the_numerical_jobs_ask_for_the_stack(module: str) -> None:
    assert _threads_without_a_stack(ROOT / module) == []


def test_a_plain_qthread_is_what_this_guard_is_about() -> None:
    """The check above is not empty: a thread nobody sized reports Qt's default, 0 (“the system's”)."""
    assert QThread().stackSize() == 0
