"""Release 0.1.0, Compare: reading the curve files of a series says how far it is (the status line counts the
files, the cursor waits, the window is drawn again), ends when the window is closed (or gone), and reads the same
cells as before in a third of the time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtCore import QEventLoop, Qt, QTimer
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


class _Watched:
    """The curve reader, noting what the page shows (status line, cursor) while each file is read;
    ``at``: something to do before the n-th file (counted from 1)."""

    def __init__(self, page, at=None):
        self.reader, self.page, self.seen, self.at = page.service.reader, page, [], at or {}

    def read(self, path):
        cursor = QApplication.overrideCursor()
        self.seen.append((self.page.status_label.text(), None if cursor is None else cursor.shape()))
        self.at.get(len(self.seen), lambda: None)()
        return self.reader.read(path)


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

def test_the_page_counts_the_files_and_waits_while_a_folder_is_read(tmp_path: Path, monkeypatch) -> None:
    page = _shown_page()
    try:
        _write_curves(tmp_path / "run_A", _run(60, seed=1))
        watched = _Watched(page)
        monkeypatch.setattr(page.service, "reader", watched)
        flags, process = [], QApplication.processEvents
        monkeypatch.setattr(QApplication, "processEvents", staticmethod(lambda *args: (flags.append(args), process(*args))[1]))
        drawn = []
        QTimer.singleShot(0, lambda: drawn.append(page.status_label.text()))  # waits for the event loop to run
        added = page.add_folder(tmp_path / "run_A")
        events = list(flags)
        assert added is page.series[0] and added.rows == 60
        texts = [text for text, _cursor in watched.seen]
        assert texts[0] == texts[24] == "Reading curve files: 0 of 60 …"
        assert texts[25] == texts[49] == "Reading curve files: 25 of 60 …"
        assert texts[50] == texts[59] == "Reading curve files: 50 of 60 …"
        assert {cursor for _text, cursor in watched.seen} == {Qt.WaitCursor}
        # At every count the events are handled (the window is drawn again), without clicks and keys.
        assert events == [(QEventLoop.ExcludeUserInputEvents,)] * 3
        assert drawn == ["Reading curve files: 0 of 60 …"]
        assert QApplication.overrideCursor() is None
        assert page.status_label.text() == "Added run_A: 60 frames."
    finally:
        monkeypatch.undo()
        _close(page)


def test_chosen_curve_files_are_counted_too_and_in_chinese(tmp_path: Path, monkeypatch, chinese) -> None:
    page = _shown_page()
    try:
        files = _write_curves(tmp_path / "run_B", _run(30, seed=2))
        watched = _Watched(page)
        monkeypatch.setattr(page.service, "reader", watched)
        apply_language("zh", [page])
        page.refresh_language()
        assert READING in ZH_ENTRIES  # the page's text is the key of the table
        line = translate(READING, "zh")  # the table's Chinese, whatever its wording: with both numbers
        assert line and line != READING and "{done}" in line and "{total}" in line
        added = page.add_files([str(path) for path in files])
        assert added is not None and added.rows == 30
        texts = [text for text, _cursor in watched.seen]
        assert texts[0] == line.format(done=0, total=30) and texts[29] == line.format(done=25, total=30)
        assert QApplication.overrideCursor() is None
    finally:
        apply_language(DEFAULT_LANGUAGE)
        _close(page)


def test_the_cursor_comes_back_after_an_error(tmp_path: Path, monkeypatch) -> None:
    page = _shown_page()
    try:
        (tmp_path / "empty").mkdir()
        assert page.add_folder(tmp_path / "empty") is None and QApplication.overrideCursor() is None
        assert page.status_label.text() == "Fewer than two curve files (.dat, .txt, .csv, .xy, .chi) in empty."
        files = [str(path) for path in _write_curves(tmp_path / "run_C", _run(30, seed=3))]

        class Unreadable:
            def read(self, path):
                raise OSError("denied")

        monkeypatch.setattr(page.service, "reader", Unreadable())
        assert page.add_files(files) is None and page.series == [] and QApplication.overrideCursor() is None
        assert page.status_label.text().startswith("Fewer than two curves could be read: run_1_fit_input.dat (denied)")

        class Broken:
            def read(self, path):
                raise RuntimeError("not an error of the files")

        monkeypatch.setattr(page.service, "reader", Broken())
        with pytest.raises(RuntimeError):  # not the page's to say; the cursor is back all the same
            page.add_files(files)
        assert QApplication.overrideCursor() is None
    finally:
        _close(page)


def test_a_window_closed_while_files_are_read_ends_the_reading(tmp_path: Path, monkeypatch) -> None:
    page = _shown_page()
    try:
        _write_curves(tmp_path / "run_A", _run(60, seed=1))
        _write_curves(tmp_path / "run_B", _run(30, seed=2))
        watched = _Watched(page, at={30: page.hide})  # the window goes while the 30th file is read
        monkeypatch.setattr(page.service, "reader", watched)
        assert page.add_dropped([tmp_path / "run_A", tmp_path / "run_B"]) == []
        assert len(watched.seen) == 50  # ended at the next count; the second folder was not read at all
        assert page.series == [] and QApplication.overrideCursor() is None
        assert page.status_label.text() == ""  # no “Reading … 50 of 60” left for when the window is shown again
        page.show()
        QApplication.processEvents()
        apply_language("zh", [page])
        page.refresh_language()  # nothing of the ended reading is made again either
        assert page.status_label.text() == ""
        apply_language(DEFAULT_LANGUAGE, [page])
        page.refresh_language()
        assert [item.name for item in page.add_dropped([tmp_path / "run_B"])] == ["run_B"]  # shown again: read again
        assert page.status_label.text() == "Added run_B: 30 frames."
    finally:
        apply_language(DEFAULT_LANGUAGE)
        _close(page)


def test_folders_dropped_together_are_compared_once_with_one_notice(tmp_path: Path, monkeypatch) -> None:
    """The window handles events at every count, the comparison's timer among them: it waits for the reading, so
    one drop of three folders gives one comparison of three series and one toast — and the Series step stays."""
    import time

    from src.gimap.features.compare.presentation import page as page_module

    toasts, compared, steps = [], [], []
    monkeypatch.setattr(page_module, "show_toast", lambda parent, text, **kw: toasts.append(text))
    page = _shown_page()
    try:
        folders = [tmp_path / name for name in ("run_A", "run_B", "run_C")]
        for seed, (folder, extra) in enumerate(zip(folders, (0.0, 0.0, 1.0))):
            _write_curves(folder, _run(60, seed=seed + 1, extra=extra))
        page._timer.setInterval(20)  # the quiet time ends long before the next folder is read
        reader, compare = page.service.reader, page.service.compare

        class Slow:  # a first read from a disk: several counts apart in time
            def read(self, path):
                time.sleep(0.003)
                steps.append(page.step_rail.current())
                return reader.read(path)

        monkeypatch.setattr(page.service, "reader", Slow())
        monkeypatch.setattr(page.service, "compare",
                            lambda series, settings: (compared.append(len(series)), compare(series, settings))[1])
        assert [item.name for item in page.add_dropped(folders)] == ["run_A", "run_B", "run_C"]
        assert compared == [] and page._reading == 0 and page._timer.isActive()
        _wait(page, lambda: _idle(page) and page.comparison is not None and len(page.comparison.results) == 3)
        assert compared == [3] and len(toasts) == 1 and toasts[0].startswith("3 series compared")
        assert set(steps) == {"series"} and page.step_rail.current() == "results"
        # A reading that fails leaves the comparison of what is already there waiting, not lost.
        page.series.pop()
        page._schedule()
        assert page.add_folder(tmp_path / "nothing_here") is None and page._reading == 0
        _wait(page, lambda: _idle(page) and len(page.comparison.results) == 2)
        assert compared == [3, 2]
    finally:
        monkeypatch.undo()
        _close(page)


def test_a_page_that_goes_with_its_window_while_a_drop_is_read_ends_it_quietly(tmp_path: Path, monkeypatch) -> None:
    """Two folders and a file of another kind dropped; the window is closed and deleted while the first folder is
    read: the reading ends at the next count, and neither the second folder nor the warning touches the page."""
    from PyQt5 import sip

    page = _shown_page()
    _write_curves(tmp_path / "run_A", _run(60, seed=1))
    _write_curves(tmp_path / "run_B", _run(30, seed=2))
    (tmp_path / "notes.pdf").write_bytes(b"not a curve")

    def gone() -> None:
        _close(page)
        sip.delete(page)

    watched = _Watched(page, at={30: lambda: QTimer.singleShot(0, gone)})  # handled with the events of a count
    monkeypatch.setattr(page.service, "reader", watched)
    assert page.add_dropped([tmp_path / "run_A", tmp_path / "run_B", tmp_path / "notes.pdf"]) == []
    assert sip.isdeleted(page) and len(watched.seen) == 50
    assert QApplication.overrideCursor() is None
    assert page.add_folder(tmp_path / "run_B") is None  # asked once more by a caller that still holds it: nothing
