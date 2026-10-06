"""Fitting: files dropped on the pages and the remembered panel widths (fitting-12)."""

from __future__ import annotations

from PyQt5.QtCore import QMimeData, QPoint, Qt, QUrl
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import QApplication

from tests.test_fit_page import _curve_file, _page
from tests.test_fit_series import _model, _pages, _series
from tests.test_wave2b_fitting_series_list import _close


def _mime(*paths) -> QMimeData:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    return mime


def _enter(widget, mime) -> bool:
    event = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    event.ignore()
    QApplication.sendEvent(widget, event)
    return event.isAccepted()


def _drop(widget, mime) -> bool:
    """A drag that enters ``widget`` and is dropped there (Qt sends the drop to the widget that took the enter)."""
    if not _enter(widget, mime):
        return False
    event = QDropEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    event.ignore()
    QApplication.sendEvent(widget, event)
    return event.isAccepted()


def test_a_curve_file_dropped_on_single_analysis_opens(tmp_path) -> None:
    page = _page()
    assert page.acceptDrops()
    curve = _curve_file(tmp_path)
    project = tmp_path / "work.gimap"
    project.write_text("{}", encoding="utf-8")
    table = tmp_path / "table.csv"
    table.write_text("0.1,10\n0.2,20\n", encoding="utf-8")
    assert not _enter(page, _mime(tmp_path))  # a folder: In-situ series takes those
    assert not _enter(page, _mime(project))  # a project: the main window opens it
    assert not _enter(page, _mime(table))  # not a file the curve reader takes (.dat, .txt)
    assert _drop(page, _mime(curve))
    assert page.session.curve is not None and page.session.curve.path == str(curve)
    other = _curve_file(tmp_path, name="second_fit_input.dat")
    third = _curve_file(tmp_path, name="third_fit_input.dat")
    assert _drop(page, _mime(other, third))  # several: the first opens, the status says one at a time
    assert page.session.curve.path == str(other) and "One curve at a time" in page.status_label.text()
    _close(page)


def test_a_file_that_is_no_curve_says_why_in_the_interface_language(tmp_path, monkeypatch) -> None:
    from src.gimap.app.presentation import i18n
    from tests.test_wave2b_fitting_predict import ZH_ENTRIES

    page = _page()
    bad = tmp_path / "notes.txt"
    bad.write_text("no numbers here\n", encoding="utf-8")
    assert _drop(page, _mime(bad))
    assert page.status_label.text() == ("Could not open notes.txt: No data rows found (at least two numeric "
                                        "columns are needed).")
    reason = "No data rows found (at least two numeric columns are needed)."
    monkeypatch.setitem(i18n.ZH, reason, ZH_ENTRIES[reason])
    i18n.apply_language("zh")
    try:
        page.refresh_language()
        assert ZH_ENTRIES[reason] in page.status_label.text()
    finally:
        i18n.apply_language("en")
    _close(page)


def test_a_folder_dropped_on_in_situ_series_is_listed(tmp_path) -> None:
    single, series = _pages()
    folder = _series(tmp_path)
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    assert series.acceptDrops()
    assert not _enter(series, _mime(folder / "run_00002_fit_input.dat"))  # a curve: Single analysis opens those
    assert _drop(series, _mime(folder))
    assert series.folder == str(folder) and len(series.paths) == 5
    _close(series, single)


def test_the_steps_panel_width_is_remembered_per_page(tmp_path) -> None:
    from src.gimap.features.fitting.presentation.single.page import PREFERENCES_KEY
    from src.gimap.features.fitting.presentation.single.series_page import PREFERENCES_KEY as SERIES_KEY

    page = _page()
    page.splitter.setSizes([520, 860])
    page.splitter.splitterMoved.emit(520, 1)  # as a drag of the handle does
    assert page._splitter_timer.isActive()
    page._splitter_timer.timeout.emit()  # a moment later
    stored = page.preferences.get(PREFERENCES_KEY, {})["splitter"]
    assert stored[0] == page.splitter.sizes()[0] and stored[0] > 400
    again = _page()
    again.preferences = page.preferences
    again._restore()
    again.resize(1400, 900)
    QApplication.processEvents()
    assert abs(again.splitter.sizes()[0] - stored[0]) <= 2  # the steps panel keeps its width
    single, series = _pages()
    assert SERIES_KEY != PREFERENCES_KEY and "splitter" not in (series.preferences.get(SERIES_KEY, {}) or {})
    _close(page, again, series, single)
