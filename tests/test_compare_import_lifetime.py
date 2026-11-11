"""Parent deletion cancels the current import without calling back into a dying Compare QWidget."""

import gc
import threading

from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QWidget

from src.gimap.app.presentation.task_runner import TaskRunner
from src.gimap.features.compare.bootstrap import create_compare_service
from src.gimap.features.compare.presentation.page import ComparePage
from tests.test_compare import _run, _write_curves


def test_parent_deletion_cancels_the_renewed_import_and_rejects_its_late_result(tmp_path):
    """The shared runner outlives the page; destruction must cancel its newest token, not its initial one."""
    tasks = TaskRunner()
    owner = QWidget()
    page = ComparePage(create_compare_service(), task_runner=tasks, parent=owner)
    files = _write_curves(tmp_path / "synthetic", _run(4, seed=1))
    entered, release = threading.Event(), threading.Event()
    reader = page.service.reader
    calls = []

    class SlowReader:
        def read(self, path):
            calls.append(path)
            entered.set()
            assert release.wait(10)
            return reader.read(path)

    page.service.reader = SlowReader()
    initial_token = page._read_cancel
    try:
        page.add_files(files)
        assert entered.wait(5)
        first_token = page._read_cancel
        page.cancel_import()
        release.set()
        assert tasks.wait(10)
        assert first_token.is_set() and not page.series

        entered.clear()
        release.clear()
        page.add_files(files)
        assert entered.wait(5)
        current_token = page._read_cancel
        assert current_token is not first_token and current_token is not initial_token
        assert not current_token.is_set()

        # No dispose or close: exercise the destroyed connection when the owner deletes its children.
        owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert sip.isdeleted(page) and current_token.is_set()
        gc.collect()  # Python collection can occur before the blocked read returns.
        release.set()
        assert tasks.wait(10)
        assert len(calls) == 2  # one active file per import; neither proceeds to its next file
        assert not page.series  # a late result must not publish data or touch a deleted widget
    finally:
        release.set()
        tasks.wait(10)
        if not sip.isdeleted(page):
            page.dispose()
            page.close()
        if not sip.isdeleted(owner):
            owner.deleteLater()
        tasks.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_compare_collection_before_deferred_delete_is_safe(qt_application):
    """Repeat the layout test's cleanup, including collection before queued view deletion is delivered."""
    for _ in range(12):
        page = ComparePage(create_compare_service())
        page.dispose()
        page.close()
        del page
        gc.collect()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qt_application.processEvents()
