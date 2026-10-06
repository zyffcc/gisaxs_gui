"""A background result for a widget deleted meanwhile (a closed dialog) is dropped, not an error."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QApplication, QLabel


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _delete(widget) -> None:
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


class _Page(QLabel):
    def shown(self, text) -> None:
        self.setText(str(text))


def test_a_result_for_a_deleted_widget_is_dropped() -> None:
    from src.gimap.app.presentation.task_runner import TaskRunner

    _app()
    tasks = TaskRunner()
    page, label = _Page(), QLabel()
    tasks.submit("bound", lambda: "ok", on_done=page.shown)
    tasks.submit("lambda", lambda: "ok", on_done=lambda text: label.setText(text))
    _delete(page)
    _delete(label)
    assert sip.isdeleted(page) and sip.isdeleted(label)
    assert tasks.wait(10)  # neither callback raises


def test_other_runtime_errors_still_reach_the_error_guard() -> None:
    from src.gimap.app.presentation.task_runner import _call

    def broken(_value):
        raise RuntimeError("something else went wrong")

    with pytest.raises(RuntimeError, match="something else"):
        _call(broken, 1)
