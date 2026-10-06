"""Wave 2b, the Compare page: errors in the interface language, dropped folders and curve files, Remove All with
Undo, the marks on the change and paths plots, the paths plot's empty text, copying the tables, and saving a plot
on the light palette."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QImage
from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language, translate
from tests.test_compare import Q, _close, _idle, _map, _run, _shown_page, _wait, _write_curves

ZH_ENTRIES = {
    "A second main change is needed to draw the paths.": "需要第二个主要变化才能画出路径。",
    "Removed {n} series.": "已移除 {n} 个序列。",
    "{n} series put back.": "已恢复 {n} 个序列。",
    "Not added: {names}. Compare takes a folder or curve files ({suffixes}).":
        "未添加：{names}。比较页只接受文件夹或曲线文件（{suffixes}）。",
    "A series needs at least two curve files.": "一个序列至少需要两个曲线文件。",
    "Fewer than two curves could be read: {files}": "能读取的曲线不足两条：{files}",
    "The files have different x axes: {axes}": "这些文件的横轴不同：{axes}",
    "Fewer than two curve files ({suffixes}) in {folder}.": "{folder} 中的曲线文件（{suffixes}）不足两个。",
    "Add a series first.": "请先添加一个序列。",
    "{name} has no data.": "{name} 没有数据。",
    "The series have different x axes: {axes}": "这些序列的横轴不同：{axes}",
    "fewer than three points of two numbers": "两列数值的点不足三个",
    "Fewer than three q points have data in nearly every frame: widen the q range.":
        "几乎每帧都有数据的 q 点不足三个：请放宽 q 范围。",
    "How far each series has changed, frame by frame, along the chosen main change; × marks an odd frame "
    "(left out of the comparison) in the colour of its series":
        "每个序列沿所选主要变化逐帧变化了多少；× 标出异常帧（不参与比较），颜色与其序列相同",
    "Each series through the first two main changes: ○ its first kept frame, ■ its last; odd frames are left out":
        "每个序列在前两个主要变化上的路径：○ 为第一个保留的帧，■ 为最后一个；异常帧不计入",
    "Add the Series map of a sample from Analyze (Series ▸ Send to Compare), a folder of curve files or chosen curve "
    "files (or drop them here). Two or more series are compared with each other; one alone is described.":
        "从分析页添加一个样品的序列热图（序列 ▸ 发送到比较），或一个曲线文件夹、若干选中的曲线文件（也可以直接拖放到这里）。"
        "两个或更多序列相互比较；只有一个时给出它的描述。",
    "In Analyze, open a sample and build its map in the Series tab, then Send to Compare; repeat for every sample. "
    "Curve files (q and I columns, e.g. a Batch Export) can be added as a folder too, or dropped on this page.":
        "在分析页打开一个样品，在序列标签页生成热图，再点“发送到比较”；每个样品重复一次。曲线文件（q、I 两列，例如批量导出的结果）"
        "也可以按文件夹添加，或直接拖放到本页。",
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


def _zh(text: str) -> str:
    return translate(text, "zh") or text


def _two(page, first: int = 1, second: int = 2, **kw) -> None:
    page.add_map(_map(first), "a")
    page.add_map(_map(second, **kw), "b")
    _wait(page, lambda: _idle(page) and page.comparison is not None and len(page.comparison.results) == 2)


# -- errors ----------------------------------------------------------------------------------------

def test_every_compare_error_is_said_in_the_interface_language(chinese) -> None:
    from src.gimap.features.compare.application import ERRORS
    from src.gimap.features.compare.presentation.errors import error_text

    from tests.test_assistant_gui import _app

    _app()
    apply_language("zh")
    assert error_text("A series needs at least two curve files.") == "一个序列至少需要两个曲线文件。"
    assert error_text("Fewer than two curve files (.dat, .txt) in run (2).") == "run (2) 中的曲线文件（.dat, .txt）不足两个。"
    assert error_text("The series have different x axes: q (Å⁻¹), χ (°)") == "这些序列的横轴不同：q (Å⁻¹), χ (°)"
    assert error_text("peo1 (2) has no data.") == "peo1 (2) 没有数据。"
    assert (error_text("Fewer than two curves could be read: a (1).dat (fewer than three points of two numbers); b.dat "
                       "([Errno 13] Permission denied)")
            == "能读取的曲线不足两条：a (1).dat (两列数值的点不足三个); b.dat ([Errno 13] Permission denied)")
    assert error_text("The series share no common q range.") == _zh("The series share no common q range.")
    assert error_text("something else") == "something else"
    assert all(_zh(template) != template for template in ERRORS)  # every template has its Chinese
    apply_language(DEFAULT_LANGUAGE)
    assert error_text("peo1 (2) has no data.") == "peo1 (2) has no data."


def test_this_rounds_texts_have_their_chinese(chinese) -> None:
    from src.gimap.features.compare.presentation.views.compare_page_view import (
        CHANGE_TIP, EMPTY_TEXT, PATHS_EMPTY, PATHS_TIP)

    page = _shown_page()
    try:
        for text in (CHANGE_TIP, PATHS_TIP, PATHS_EMPTY, EMPTY_TEXT, page.series_hint.text()):
            assert _zh(text) != text, text
        apply_language("zh", [page])
        page.refresh_language()
        assert page.change_plot.toolTip() == _zh(CHANGE_TIP) and page.paths_plot.toolTip() == _zh(PATHS_TIP)
        assert page.empty_state.message_label.text() == _zh(EMPTY_TEXT)
    finally:
        apply_language(DEFAULT_LANGUAGE)
        _close(page)


def test_file_and_comparison_errors_follow_a_language_switch(tmp_path: Path, chinese) -> None:
    page = _shown_page()
    try:
        (tmp_path / "empty").mkdir()
        assert page.add_folder(tmp_path / "empty") is None
        english = "Fewer than two curve files (.dat, .txt, .csv, .xy, .chi) in empty."
        assert page.status_label.text() == english
        apply_language("zh", [page])
        page.refresh_language()
        assert page.status_label.text() == "empty 中的曲线文件（.dat, .txt, .csv, .xy, .chi）不足两个。"
        apply_language(DEFAULT_LANGUAGE, [page])
        page.refresh_language()
        assert page.status_label.text() == english
        # A comparison that fails: the status line, the summary and the empty state say why, in zh too.
        page.add_map(_map(1), "q")
        page.add_map(_map(2, x=np.linspace(-80, 80, Q.size), x_label="χ (°)"), "chi")
        _wait(page, lambda: _idle(page) and page.failure_reason())
        assert page.status_label.text() == "Could not compare: The series have different x axes: q (Å⁻¹), χ (°)"
        apply_language("zh", [page])
        page.refresh_language()
        reason = "这些序列的横轴不同：q (Å⁻¹), χ (°)"
        assert page.status_label.text() == _zh("Could not compare: {reason}").format(reason=reason)
        assert reason in page.summary_label.text() and reason in page.empty_state.message_label.text()
    finally:
        apply_language(DEFAULT_LANGUAGE)
        _close(page)


# -- dropping, Remove All and Undo -------------------------------------------------------------------

def _mime(paths) -> QMimeData:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    return mime


def test_folders_and_curve_files_can_be_dropped_and_other_files_are_refused(tmp_path: Path, monkeypatch) -> None:
    from src.gimap.features.compare.presentation import series_input

    toasts = []
    monkeypatch.setattr(series_input, "show_toast", lambda parent, text, **kw: toasts.append((text, kw)))
    page = _shown_page()
    try:
        assert page.acceptDrops()
        _write_curves(tmp_path / "run_A", _run(12, seed=1))
        files = _write_curves(tmp_path / "run_B", _run(12, seed=2, extra=1.0))
        other = tmp_path / "notes.pdf"
        other.write_bytes(b"%PDF")
        folder, nothing, mixed = _mime([tmp_path / "run_A"]), QMimeData(), _mime([*files, other])  # kept: events hold no ref
        enter = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, folder, Qt.LeftButton, Qt.NoModifier)
        page.dragEnterEvent(enter)
        assert enter.isAccepted()
        empty = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, nothing, Qt.LeftButton, Qt.NoModifier)
        page.dragEnterEvent(empty)
        assert not empty.isAccepted()
        # Over a child — a plot (which takes drops of its own), a table, the empty state: the page takes it.
        for child in (page.change_plot.plot_widget.viewport(), page.series_table.viewport(), page.empty_state):
            enter = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, folder, Qt.LeftButton, Qt.NoModifier)
            QApplication.sendEvent(child, enter)
            assert enter.isAccepted(), child
        # One folder: one series.
        drop = QDropEvent(QPointF(5, 5), Qt.CopyAction, folder, Qt.LeftButton, Qt.NoModifier)
        page.dropEvent(drop)
        assert drop.isAccepted() and page.series == []  # read once the drop has returned (the source is free)
        QApplication.processEvents()
        assert [item.name for item in page.series] == ["run_A"]
        # Curve files together: one series; the other file is left out with a warning.
        drop = QDropEvent(QPointF(5, 5), Qt.CopyAction, mixed, Qt.LeftButton, Qt.NoModifier)
        page.dropEvent(drop)
        QApplication.processEvents()
        assert [item.name for item in page.series] == ["run_A", "run_B"] and page.series[1].rows == 12
        assert page.status_label.text() == ("Not added: notes.pdf. Compare takes a folder or curve files "
                                            "(.dat, .txt, .csv, .xy, .chi).")
        # A single curve file: the error says a series needs two.
        assert page.add_dropped([files[0]]) == [] and page.status_label.text() == "A series needs at least two curve files."
        # The same folder again: refused as a duplicate.
        assert page.add_dropped([tmp_path / "run_A"])[0] is page.series[0] and len(page.series) == 2
        _wait(page, lambda: _idle(page) and page.comparison is not None and len(page.comparison.results) == 2)
    finally:
        _close(page)


def test_remove_all_offers_undo_and_the_same_map_twice_stays_one_series(monkeypatch) -> None:
    from src.gimap.features.compare.presentation import series_input

    toasts = []
    monkeypatch.setattr(series_input, "show_toast", lambda parent, text, **kw: toasts.append((text, kw)))
    page = _shown_page()
    try:
        page.add_map(_map(1), "a")
        page.add_map(_map(1), "a again")  # the same map sent twice (another name): refused
        assert [item.name for item in page.series] == ["a"]
        page.add_map(_map(2), "b")
        _wait(page, lambda: _idle(page) and page.comparison is not None and len(page.comparison.results) == 2)
        page.q_low_spin.setValue(0.5)
        _wait(page, lambda: _idle(page) and page.comparison.compared.min() >= 0.5)
        page.clear_all()
        assert page.series == [] and page.status_label.text() == ""  # the toast is the notice
        (text, options), = toasts
        assert text == "Removed 2 series." and options["action"][0] == "Undo" and options["timeout_ms"] >= 8000
        options["action"][1]()  # Undo
        assert [item.name for item in page.series] == ["a", "b"] and page._q_range == (0.5, page._q_range[1])
        _wait(page, lambda: _idle(page) and page.comparison is not None and len(page.comparison.results) == 2)
        assert page.comparison.compared.min() >= 0.5  # the range of before
        assert page.status_label.text() == "2 series compared."
        page.add_map(_map(1), "a once more")
        assert len(page.series) == 2  # still a duplicate after the Undo
        # Undo after a series was added meanwhile: the others come back after it, none doubled.
        page.clear_all()
        page.add_map(_map(2), "b")
        assert [item.name for item in page.undo_remove_all(*_removed(toasts))] == ["a"]
        assert [item.name for item in page.series] == ["b", "a"]
        page.clear_all()
        assert len(toasts) == 3
        page.clear_all()  # nothing to remove: no toast
        assert len(toasts) == 3
    finally:
        _close(page)


def _removed(toasts):
    """The series and range a Remove All toast's Undo puts back (its closure)."""
    callback = toasts[-1][1]["action"][1]
    cells = {name: cell.cell_contents for name, cell in zip(callback.__code__.co_freevars, callback.__closure__)}
    return cells["removed"], cells["q_range"]


# -- plots ------------------------------------------------------------------------------------------

def test_the_plots_mark_odd_frames_and_where_each_path_starts_and_ends(tmp_path: Path) -> None:
    from src.gimap.features.compare.presentation.render import END_MARK, ODD_MARK, START_MARK

    page = _shown_page(1500, 950)
    try:
        page.add_map(SimpleNamespace(x=Q, image=_run(40, seed=1, odd=17), labels=tuple(f"f{i}" for i in range(40)),
                                     x_label="q (Å⁻¹)"), "odd")
        page.add_map(_map(2, extra=1.0, rows=40), "plain")
        _wait(page, lambda: _idle(page) and page.comparison is not None and len(page.comparison.results) == 2)
        results = page.comparison.results
        assert 17 in {frame.row for frame in results[0].odd}
        change = page.change_plot
        names = [name for name, _x, _y in change.figure_state()["curves"]]
        assert names[:2] == ["odd", "plain"] and "odd (odd)" in names
        assert [label.text for _sample, label in change.legend.items] == ["odd", "plain"]  # the × have no entry
        mark = names.index("odd (odd)")
        assert change._source_markers[mark] == ODD_MARK and change.curve_colors()[mark] == change.curve_colors()[0]
        assert 18.0 in change.figure_state()["curves"][mark][1]  # frame 18 (row 17)
        assert "×" in change.toolTip()
        assert results[0].scores.shape[1] >= 2  # two main changes: paths
        paths = page.paths_plot
        names = [name for name, _x, _y in paths.figure_state()["curves"]]
        assert names == ["odd", "plain", "odd (first)", "odd (last)", "plain (first)", "plain (last)"]
        assert [label.text for _sample, label in paths.legend.items] == ["odd", "plain"]
        assert paths._source_markers[2:] == [START_MARK, END_MARK, START_MARK, END_MARK]
        kept = [row for row in range(40) if row not in {frame.row for frame in results[0].odd}]
        first, last = paths.figure_state()["curves"][2:4]
        assert first[1][0] == results[0].scores[kept[0], 0] and first[2][0] == results[0].scores[kept[0], 1]
        assert last[1][0] == results[0].scores[kept[-1], 0] and last[2][0] == results[0].scores[kept[-1], 1]
        assert "○" in paths.toolTip() and "■" in paths.toolTip()
        assert not paths.empty_overlay.label.isVisible()
        # One main change only: the paths plot keeps its title and says what is missing.
        real = page.comparison
        page.comparison = SimpleNamespace(results=[SimpleNamespace(scores=np.zeros((5, 1)))])
        page._draw_paths()
        QApplication.processEvents()
        assert page.paths_plot.title_label.text() == "Paths through the two main changes"
        assert page.paths_plot.empty_overlay.label.isVisible()
        assert page.paths_plot.empty_overlay.label.text() == "A second main change is needed to draw the paths."
        page.comparison = real
        page._draw_paths()
        # Saved on the light palette in the dark theme.
        from src.gimap.app.presentation.theme import LIGHT, apply_theme, theme_manager

        mode = theme_manager().mode
        try:
            apply_theme("dark")
            written = page._save_plot(page.change_plot, "change", path=str(tmp_path / "change.png"))
            assert QImage(str(written)).pixelColor(3, 3).name() == LIGHT["plot_bg"]
            assert page.change_plot.plot_widget.backgroundBrush().color().name() != LIGHT["plot_bg"]
        finally:
            apply_theme(mode)
    finally:
        _close(page)


# -- tables ----------------------------------------------------------------------------------------

def test_the_compare_tables_copy_their_rows_and_the_series_table_stays_editable() -> None:
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QAbstractItemView

    page = _shown_page()
    try:
        _two(page)
        page.show_step("results")
        QApplication.processEvents()
        for table in (page.series_table, page.results_table, page.distance_table):
            assert [action.text() for action in table.actions()] == ["Copy Rows", "Copy Table"]
        assert page.series_table.editTriggers() != QAbstractItemView.NoEditTriggers
        page.results_table.clearSelection()
        page.results_table.setFocus()
        QTest.keyClick(page.results_table, Qt.Key_C, Qt.ControlModifier)
        lines = QApplication.clipboard().text().splitlines()
        assert lines[0].split("\t")[0] == "Series" and len(lines) == 3 and lines[1].startswith("a\t")
        page.series_table.selectRow(1)
        page.series_table.actions()[0].trigger()
        assert QApplication.clipboard().text().splitlines() == ["Series\tFrames\tFrom", "b\t60\tAnalyze"]
        page.show_step("series")
        QApplication.processEvents()
        page.series_table.setFocus()
        QTest.keyClick(page.series_table, Qt.Key_Delete)  # its own Delete key still removes the series
        assert [item.name for item in page.series] == ["a"]
    finally:
        _close(page)
