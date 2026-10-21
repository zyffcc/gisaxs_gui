"""The worker threads of the tool windows have room for numerical work.

macOS gives a thread that is not the main one a 512 KB stack; numpy.linalg.inv / solve and a Matplotlib
figure save overflow it and end the process (see ``tests/test_task_runner_stack.py``). Every ``QThread`` the
tool windows start, and the pool the Trainset page computes on, therefore asks for ``WORKER_STACK_BYTES``.
The size is checked in the source and on the thread, never by letting an unsized thread compute.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QThreadPool
from PyQt5.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
ENOUGH = 16 * 1024 * 1024 if sys.platform == "darwin" else 0


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
    "src/gimap/features/calibration/presentation/bindings/image_loading.py",
    "src/gimap/features/xrr/presentation/dialog.py",
    "src/gimap/features/format_converter/presentation/bindings/preview.py",
    "src/gimap/features/format_converter/presentation/bindings/conversion.py",
])
def test_the_threads_of_the_tool_windows_ask_for_the_stack(module: str) -> None:
    path = ROOT / module
    assert "QThread(" in path.read_text(encoding="utf-8")  # the check below is about something
    assert _threads_without_a_stack(path) == []


def test_every_qthread_of_the_tool_features_asks_for_the_stack() -> None:
    """No thread is added to Calibration, XRR or the Format Converter without the size."""
    unsized = {}
    for feature in ("calibration", "xrr", "format_converter"):
        for path in (ROOT / "src" / "gimap" / "features" / feature).rglob("*.py"):
            lines = _threads_without_a_stack(path)
            if lines:
                unsized[str(path.relative_to(ROOT))] = lines
    assert unsized == {}


def test_the_pool_trainset_computes_on_asks_for_the_stack() -> None:
    from src.gimap.app.presentation.task_runner import WORKER_STACK_BYTES

    QApplication.instance() or QApplication([])
    source = (ROOT / "src/gimap/features/trainset/presentation/view_binding.py").read_text(encoding="utf-8")
    assert "setStackSize(WORKER_STACK_BYTES)" in source
    pool = QThreadPool.globalInstance()
    before = pool.stackSize()
    try:
        # What the binding does when it is built (building the whole Trainset page is not needed for this).
        tree = ast.parse(source)
        sized = [
            ast.unparse(node.func.value) for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "setStackSize"
        ]
        assert sized == ["self.thread_pool"]
        pool.setStackSize(WORKER_STACK_BYTES)
        assert pool.stackSize() == ENOUGH
    finally:
        pool.setStackSize(before)
