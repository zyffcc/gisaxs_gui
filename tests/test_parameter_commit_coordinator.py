"""Shared parameter-commit interaction contract tests."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QDoubleSpinBox

from src.gimap.app.presentation import (
    ParameterCommitCoordinator,
    ParameterUpdatePolicy,
)


_TEST_APP = None


def _app() -> QApplication:
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


def test_rapid_numeric_changes_are_committed_once_after_debounce():
    app = _app()
    widget = QDoubleSpinBox()
    commits = []
    coordinator = ParameterCommitCoordinator(widget)
    coordinator.register_group(
        "geometry",
        commit=lambda: commits.append(widget.value()),
        policy=ParameterUpdatePolicy(debounce_ms=40),
    )
    coordinator.bind_numeric("geometry", widget)

    widget.setValue(1.0)
    widget.setValue(2.0)
    widget.setValue(3.0)
    app.processEvents()
    assert commits == []

    QTest.qWait(55)
    app.processEvents()
    assert commits == [3.0]


def test_editing_finished_flushes_a_pending_numeric_change_immediately():
    app = _app()
    widget = QDoubleSpinBox()
    commits = []
    coordinator = ParameterCommitCoordinator(widget)
    coordinator.register_group(
        "geometry",
        commit=lambda: commits.append(widget.value()),
        policy=ParameterUpdatePolicy(debounce_ms=500),
    )
    coordinator.bind_numeric("geometry", widget)

    widget.setValue(4.0)
    widget.editingFinished.emit()
    app.processEvents()
    assert commits == [4.0]

    QTest.qWait(30)
    app.processEvents()
    assert commits == [4.0]
