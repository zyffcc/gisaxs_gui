"""Compare imports report progress on the GUI thread, remain cancellable, and preserve scientific arrays."""

from __future__ import annotations

from pathlib import Path
from threading import Event, get_ident

import numpy as np
import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language, translate
from src.gimap.features.compare.application import PROGRESS_EVERY
from src.gimap.features.compare.bootstrap import create_compare_service
from src.gimap.features.compare.presentation.series_input import READING
from tests.test_compare import _close, _idle, _run, _shown_page, _wait, _write_curves

ZH_ENTRIES = {
    "Reading curve files: {done} of {total} …": "正在读取曲线文件：{done} / {total} …",
}


@pytest.fixture
def chinese(monkeypatch):
    """This round's Chinese in the table (until it is merged into ``zh_*.py``)."""
    for english, chinese in ZH_ENTRIES.items():
        if english not in i18n.ZH:
            monkeypatch.setitem(i18n.ZH, english, chinese)
            monkeypatch.setitem(i18n._TO_ENGLISH, chinese, english)
    yield
    apply_language(DEFAULT_LANGUAGE)


class _GatedReader:
    """Observe only worker data; the test releases selected reads from the GUI thread."""

    def __init__(self, reader, pauses=()):
        self.reader, self.paths, self.threads = reader, [], []
        self.gates = {index: (Event(), Event()) for index in pauses}

    def read(self, path):
        self.paths.append(Path(path))
        self.threads.append(get_ident())
        gate = self.gates.get(len(self.paths))
        if gate:
            reached, released = gate
            reached.set()
            if not released.wait(10):
                raise RuntimeError("test did not release the reader")
        return self.reader.read(path)

    def release_all(self):
        for _reached, released in self.gates.values():
            released.set()


def _observe_progress(page):
    seen = []
    page._read_signals.progress.connect(lambda _token, done, total: seen.append(
        (done, total, page.status_label.text(), get_ident(), QApplication.overrideCursor())))
    return seen


# -- the service and the reader ----------------------------------------------------------------------

def test_reading_curve_files_reports_before_the_first_file_and_every_25(tmp_path: Path) -> None:
    service = create_compare_service()
    files = _write_curves(tmp_path / "run", _run(60, seed=1))
    calls = []
    series = service.series_from_files(files, progress=lambda done, total: calls.append((done, total)))
    assert PROGRESS_EVERY == 25 and calls == [(0, 60), (25, 60), (50, 60)]
    calls.clear()
    folder = service.series_from_folder(tmp_path / "run", progress=lambda done, total: calls.append((done, total)))
    assert calls == [(0, 60), (25, 60), (50, 60)]
    plain = service.series_from_folder(tmp_path / "run")  # without it: the same series
    assert plain.labels == folder.labels == series.labels and plain.rows == 60
    assert np.array_equal(plain.image, folder.image) and np.array_equal(plain.x, series.x)
    calls.clear()
    with pytest.raises(ValueError, match="at least two"):  # no series to read: nothing reported
        service.series_from_files(files[:1], progress=lambda done, total: calls.append((done, total)))
    assert calls == []


def test_the_reader_takes_the_same_cells_as_before(tmp_path: Path) -> None:
    """Commas, semicolons, tabs and spaces (and runs of them) part the cells; nothing else does."""
    path = tmp_path / "mixed.dat"
    path.write_text("\n".join([
        "# columns: q (1/A)  I  sigma", "q,I,sigma",
        "0.1,10,1", "0.2;20", "0.3\t30\t3", "0.4   40   4", " 0.5 , 50 ", "0.6,,60", ";0.7;\t 70", "8e-1 8E+1 note",
        "0.9 90", "1.0 91",  # a no-break or thin space is not a separator: such a line was never a point
    ]) + "\n", encoding="utf-8")
    x, intensity, label = create_compare_service().reader.read(path)
    assert label == "q (Å⁻¹)"
    assert x.tolist() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    assert intensity.tolist() == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]


# -- the page ----------------------------------------------------------------------------------------

def test_the_page_counts_files_on_the_gui_thread_while_reading_in_a_worker(tmp_path: Path, monkeypatch) -> None:
    page = _shown_page()
    watched = _GatedReader(page.service.reader, pauses=(1,))
    try:
        folder = tmp_path / "run_A"
        _write_curves(folder, _run(60, seed=1))
        expected = page.service.series_from_folder(folder)
        monkeypatch.setattr(page.service, "reader", watched)
        seen = _observe_progress(page)
        drawn = []
        QTimer.singleShot(0, lambda: drawn.append(get_ident()))
        assert page.add_folder(folder) is None
        assert not page.series  # accepted without waiting for the disk
        _wait(page, lambda: watched.gates[1][0].is_set() and bool(seen))
        assert drawn == [get_ident()]
        assert page.import_job.isVisible() and QApplication.overrideCursor() is None
        # Input is not excluded while the reader is blocked on the disk.
        page.end_spin.setValue(3)
        assert page.end_spin.value() == 3
        watched.release_all()
        assert page.tasks.wait(10)
        assert [item.name for item in page.series] == ["run_A"]
        added = page.series[0]
        assert added.rows == 60 and added.labels == expected.labels
        np.testing.assert_array_equal(added.image, expected.image)
        np.testing.assert_array_equal(added.x, expected.x)
        assert [(done, total) for done, total, *_rest in seen] == [(0, 60), (25, 60), (50, 60)]
        assert [text for _done, _total, text, *_rest in seen] == [
            "Reading curve files: 0 of 60 …", "Reading curve files: 25 of 60 …", "Reading curve files: 50 of 60 …"]
        assert {thread for _done, _total, _text, thread, _cursor in seen} == {get_ident()}
        assert all(cursor is None for *_rest, cursor in seen)
        assert all(thread != get_ident() for thread in watched.threads)
        assert page.import_job.isHidden()
    finally:
        watched.release_all()
        _close(page)


def test_chosen_curve_files_are_counted_too_and_in_chinese(tmp_path: Path, monkeypatch, chinese) -> None:
    page = _shown_page()
    try:
        files = _write_curves(tmp_path / "run_B", _run(30, seed=2))
        apply_language("zh", [page])
        page.refresh_language()
        assert READING in ZH_ENTRIES
        line = translate(READING, "zh")
        assert line and line != READING and "{done}" in line and "{total}" in line
        seen = _observe_progress(page)
        assert page.add_files([str(path) for path in files]) is None
        assert page.tasks.wait(10)
        assert len(page.series) == 1 and page.series[0].rows == 30
        assert [text for _done, _total, text, *_rest in seen] == [
            line.format(done=0, total=30), line.format(done=25, total=30)]
        assert QApplication.overrideCursor() is None and page.import_job.isHidden()
    finally:
        apply_language(DEFAULT_LANGUAGE)
        _close(page)


def test_reading_errors_are_reported_without_leaving_a_busy_cursor(tmp_path: Path, monkeypatch) -> None:
    page = _shown_page()
    try:
        (tmp_path / "empty").mkdir()
        assert page.add_folder(tmp_path / "empty") is None
        assert page.tasks.wait(10)
        assert QApplication.overrideCursor() is None
        assert page.status_label.text() == "Fewer than two curve files (.dat, .txt, .csv, .xy, .chi) in empty."
        files = [str(path) for path in _write_curves(tmp_path / "run_C", _run(30, seed=3))]

        class Unreadable:
            def read(self, path):
                raise OSError("denied")

        monkeypatch.setattr(page.service, "reader", Unreadable())
        assert page.add_files(files) is None
        assert page.tasks.wait(10)
        assert page.series == [] and QApplication.overrideCursor() is None
        assert page.status_label.text().startswith("Fewer than two curves could be read: run_1_fit_input.dat (denied)")

        class Broken:
            def read(self, path):
                raise RuntimeError("not an error of the files")

        monkeypatch.setattr(page.service, "reader", Broken())
        assert page.add_files(files) is None
        assert page.tasks.wait(10)
        assert page.status_label.text() == "not an error of the files"
        assert QApplication.overrideCursor() is None and page.import_job.isHidden()
    finally:
        _close(page)


@pytest.mark.parametrize("action", ["cancel", "close", "dispose"])
def test_stopping_an_import_discards_the_unfinished_series_and_queued_folders(tmp_path: Path, monkeypatch, action) -> None:
    page = _shown_page()
    watched = _GatedReader(page.service.reader, pauses=(30,))
    try:
        _write_curves(tmp_path / "run_A", _run(60, seed=1))
        _write_curves(tmp_path / "run_B", _run(30, seed=2))
        monkeypatch.setattr(page.service, "reader", watched)
        assert page.add_dropped([tmp_path / "run_A", tmp_path / "run_B"]) == []
        _wait(page, lambda: watched.gates[30][0].is_set())
        if action == "cancel":
            page.import_job.cancel_button.click()
        elif action == "close":
            page.close()
        else:
            page.dispose()
        assert page._reading == 0 and not page._read_queue and page.import_job.isHidden()
        watched.release_all()
        assert page.tasks.wait(10)
        assert len(watched.paths) == 30 and {path.parent.name for path in watched.paths} == {"run_A"}
        assert page.series == [] and QApplication.overrideCursor() is None
        if action == "dispose":
            assert page.add_folder(tmp_path / "run_B") is None
            assert not page.tasks.is_busy()
        else:
            if action == "cancel":
                assert page.status_label.text() == "Import cancelled. Unfinished series were not added."
            else:
                assert page.status_label.text() == ""  # closing must not leave a phantom import on reopening
                assert page._status_text is None
            page.show()
            assert page.add_folder(tmp_path / "run_B") is None
            assert page.tasks.wait(10)
            assert [item.name for item in page.series] == ["run_B"]
    finally:
        watched.release_all()
        _close(page)


def test_folders_dropped_together_are_compared_once_with_one_notice(tmp_path: Path, monkeypatch) -> None:
    from src.gimap.features.compare.presentation import page as page_module

    toasts, compared, steps = [], [], []
    monkeypatch.setattr(page_module, "show_toast", lambda parent, text, **kw: toasts.append(text))
    page = _shown_page()
    watched = _GatedReader(page.service.reader, pauses=(1, 61, 121))
    try:
        folders = [tmp_path / name for name in ("run_A", "run_B", "run_C")]
        for seed, (folder, extra) in enumerate(zip(folders, (0.0, 0.0, 1.0))):
            _write_curves(folder, _run(60, seed=seed + 1, extra=extra))
        page._timer.setInterval(20)
        compare = page.service.compare
        monkeypatch.setattr(page.service, "reader", watched)
        monkeypatch.setattr(page.service, "compare",
                            lambda series, settings: (compared.append(len(series)), compare(series, settings))[1])
        assert page.add_dropped(folders) == []
        for reached, released in watched.gates.values():
            _wait(page, reached.is_set)
            QTest.qWait(40)  # debounce expires while an import remains blocked
            assert not compared and page._reading == 1
            steps.append(page.step_rail.current())
            released.set()
        assert page.tasks.wait(10)
        assert [item.name for item in page.series] == ["run_A", "run_B", "run_C"]
        _wait(page, lambda: _idle(page) and page.comparison is not None and len(page.comparison.results) == 3)
        assert compared == [3] and len(toasts) == 1 and toasts[0].startswith("3 series compared")
        assert set(steps) == {"series"} and page.step_rail.current() == "results"
        # A failed import does not lose a pending comparison of the remaining series.
        page.series.pop()
        page._schedule()
        assert page.add_folder(tmp_path / "nothing_here") is None
        assert page.tasks.wait(10)
        _wait(page, lambda: _idle(page) and len(page.comparison.results) == 2)
        assert compared == [3, 2]
    finally:
        watched.release_all()
        _close(page)


def test_a_deleted_page_refuses_new_imports_and_queued_drops(tmp_path: Path, monkeypatch) -> None:
    from PyQt5 import sip

    page = _shown_page()
    watched = _GatedReader(page.service.reader, pauses=(1,))
    _write_curves(tmp_path / "run_A", _run(60, seed=1))
    _write_curves(tmp_path / "run_B", _run(30, seed=2))
    monkeypatch.setattr(page.service, "reader", watched)
    try:
        assert page.add_dropped([tmp_path / "run_A", tmp_path / "run_B"]) == []
        _wait(page, lambda: watched.gates[1][0].is_set())
        page.dispose()
        watched.release_all()
        assert page.tasks.wait(10)
        page.close()
        sip.delete(page)
        assert sip.isdeleted(page) and len(watched.paths) == 1
        assert QApplication.overrideCursor() is None
        assert page.add_folder(tmp_path / "run_B") is None
        assert page.add_dropped([tmp_path / "run_B"]) == []
    finally:
        watched.release_all()
        if not sip.isdeleted(page):
            _close(page)
