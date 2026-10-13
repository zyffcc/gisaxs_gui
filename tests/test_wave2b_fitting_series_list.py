"""In-situ series: the frame list keeps the text colour with a stage square, the states in the warning and
danger colours, numbers padded with figure spaces, the trend's legend “stage n” at the bottom right
(fitting-15), and the empty plots say what will appear there."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation.theme import theme_manager
from tests.test_fit_page import _page
from tests.test_fit_series import _model, _pages, _series, _wait


def _close(*pages) -> None:
    """Dispose of the pages and delete them now, their plot views first (as ``conftest`` does): a page left
    for the garbage collector can be deleted while the next test's window is painted."""
    import gc

    import pyqtgraph
    from PyQt5.QtCore import QCoreApplication, QEvent

    for page in pages:
        page.dispose()
        page.hide()
    gc.collect()
    for page in pages:
        for view in page.findChildren(pyqtgraph.GraphicsView):
            view.deleteLater()
        page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()


def test_frame_numbers_are_padded_with_figure_spaces(tmp_path) -> None:
    from src.gimap.features.fitting.presentation.single.series_page import FIGURE_SPACE

    single, series = _pages()
    folder = _series(tmp_path)
    for index in range(5, 12):  # 12 frames: two digits
        (folder / f"run_{index + 1:05d}_fit_input.dat").write_bytes((folder / "run_00001_fit_input.dat").read_bytes())
    single.open_curve(folder / "run_00001_fit_input.dat")
    series.open_series(folder)
    texts = [series.frame_list.item(row).text() for row in range(series.frame_list.count())]
    assert texts[0].startswith(f"· {FIGURE_SPACE}1  run_00001") and texts[11].startswith("· 12  run_00012")
    assert all(" " * 2 + "run_" in text for text in texts) and " 1  run" not in texts[0].replace(FIGURE_SPACE, "#")
    _close(series, single)


def test_the_list_keeps_its_colour_with_a_stage_square_and_the_states_coloured(tmp_path) -> None:
    single, series = _pages()
    folder = _series(tmp_path)
    (folder / "run_00003_fit_input.dat").write_text("not a curve\n", encoding="utf-8")
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    series.open_series(folder)
    series.start()
    _wait(series)
    series.stage_of_frame = lambda index: 1 if index < 3 else 2  # two stages (the search needs more frames)
    series.is_odd_frame = lambda index: index == 4
    series._mark_stages()
    items = [series.frame_list.item(row) for row in range(5)]
    assert items[0].text().startswith("✓") and items[0].data(Qt.ForegroundRole) is None  # the list's own colour
    assert not items[0].icon().isNull() and items[0].toolTip() == "Stage 1"
    assert items[2].text().startswith("✗") and "No data rows found" in items[2].text()  # why, in English
    assert items[2].foreground().color().name() == theme_manager().color("danger").name()
    assert items[4].toolTip() == "Odd frame (left out of the stages)" and items[4].text().endswith("odd")
    image = items[0].icon().pixmap(10, 10).toImage()
    assert image.pixelColor(5, 5).name() == "#2563eb"  # stage 1's colour, filled
    odd = items[4].icon().pixmap(10, 10).toImage()
    assert odd.pixelColor(5, 5).alpha() == 0  # an odd frame: hollow
    series._draw_trend()
    names = [curve[0] for curve in series.trend_plot.figure_state()["curves"]]
    assert names == ["stage 1", "stage 2"]  # the value is in the list and on the axis
    legend = series.trend_plot.legend
    assert legend.opts["offset"] == (-10, -10)  # the bottom right, off the late frames' points
    QApplication.processEvents()
    bottom = legend.pos().y()
    series.stage_of_frame = lambda index: None
    series._draw_trend()
    assert [curve[0] for curve in series.trend_plot.figure_state()["curves"]] == ["R (nm)"]
    assert legend.opts["offset"] == (-10, 10)
    QApplication.processEvents()
    assert legend.pos().y() < bottom  # back at the top right
    _close(series, single)


def test_a_frame_that_did_not_converge_is_in_the_warning_colour(tmp_path) -> None:
    from types import SimpleNamespace

    single, series = _pages()
    folder = _series(tmp_path)
    single.open_curve(folder / "run_00001_fit_input.dat")
    series.open_series(folder)
    series.fits[1] = SimpleNamespace(ok=True, result=SimpleNamespace(converged=False, chi2_reduced=7.0))
    series._mark_stages()
    item = series.frame_list.item(1)
    assert item.text().startswith("!") and "χ²ᵣ 7" in item.text()
    assert item.foreground().color().name() == theme_manager().color("warning").name()
    _close(series, single)


def test_why_no_stages_were_found_is_in_the_interface_language(monkeypatch) -> None:
    from src.gimap.app.presentation import i18n
    from src.gimap.features.fitting.presentation.single.series_stages import _failure_text

    reason = "No data rows found (at least two numeric columns are needed)."
    assert _failure_text(f"run_00001.dat: {reason}") == f"run_00001.dat: {reason}"
    monkeypatch.setitem(i18n.ZH, reason, "未找到数据行（至少需要两列数字）。")
    i18n.apply_language("zh")
    try:  # the file name as it is, the reason translated
        assert _failure_text(f"run_00001.dat: {reason}") == "run_00001.dat: 未找到数据行（至少需要两列数字）。"
    finally:
        i18n.apply_language("en")


def test_empty_plots_say_what_appears_there(tmp_path) -> None:
    from src.gimap.features.fitting.presentation.single.series_stages import TREND_EMPTY

    page = _page()
    assert page.plot.empty_overlay.label.text().startswith("Open a curve (q, I, σ)")
    assert not page.plot.empty_overlay.label.isHidden()
    assert page.residual_plot.empty_overlay.label.text() == "Open a curve: the residuals of the model appear here."
    _close(page)
    single, series = _pages()
    folder = _series(tmp_path)
    single.open_curve(folder / "run_00001_fit_input.dat")
    series.open_series(folder)
    assert series.trend_plot.figure_state()["title"] == ""  # the sentence once: over the plot, not in the title
    assert series.trend_plot.empty_overlay.label.text() == TREND_EMPTY
    assert not series.trend_plot.empty_overlay.label.isHidden()
    assert series.frame_plot.empty_overlay.label.isHidden()  # the first frame is drawn
    _close(series, single)
