"""Stages and Compare after the cross-area review: too few frames for a stage, stage and series names readable
on a dark theme, the Compare result table in view, run-time texts in the interface language again after a
switch, and the Chinese of the Compare texts."""

from __future__ import annotations

import ast
import re
import time
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language, translate
from src.gimap.app.presentation.theme import apply_theme, theme_manager
from src.gimap.shared.series_stages import SHORTEST, find_stages
from src.gimap.shared.series_stages.comparable import OddFrame
from tests.test_compare import Q, _close, _idle, _map, _run, _shown_page, _wait

ROOT = Path(__file__).resolve().parents[1]


def _luminance(color: str) -> float:
    shade = QColor(color)
    channels = [value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
                for value in (shade.redF(), shade.greenF(), shade.blueF())]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast(first: str, second: str) -> float:
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


class _Theme:
    """Switch the theme for a test and back to the one before."""

    def __enter__(self):
        self.mode, self.font_pt = theme_manager().mode, theme_manager().font_pt
        return self

    def __exit__(self, *_exc):
        if (theme_manager().mode, theme_manager().font_pt) != (self.mode, self.font_pt):
            apply_theme(self.mode, self.font_pt)


# -- too few frames for a stage ----------------------------------------------------------------------

def test_too_few_frames_for_a_stage_is_a_value_error_every_caller_handles() -> None:
    for rows in (3, 4):
        with pytest.raises(ValueError, match=f"At least {SHORTEST} frames are needed to find stages"):
            find_stages(Q, _run(rows))
    assert find_stages(Q, _run(SHORTEST)).count == 1


def test_odd_frames_that_leave_too_few_say_so(monkeypatch) -> None:
    from src.gimap.shared.series_stages import stages as module

    monkeypatch.setattr(module, "odd_frames", lambda _data: (OddFrame(1, 9.0, 1.0, False), OddFrame(4, 9.0, 1.0, False)))
    with pytest.raises(ValueError, match=f"At least {SHORTEST} frames besides the odd ones are needed"):
        find_stages(Q, _run(6))


def test_the_in_situ_stages_of_too_few_curves_are_a_value_error() -> None:
    from src.gimap.features.fitting.application.series_fit import SeriesSettings, stages_of_curves
    from src.gimap.features.fitting.application.single_fit import Curve

    q_nm = Q * 10.0
    curves = [Curve.from_arrays(q_nm, row, None, name=f"c{index}") for index, row in enumerate(_run(4))]
    with pytest.raises(ValueError, match="frames are needed"):
        stages_of_curves(curves, SeriesSettings())


def test_the_in_situ_page_says_why_four_curves_have_no_stages(tmp_path) -> None:
    from tests.test_fit_series import _pages, _series

    folder = _series(tmp_path, radii=(5.0, 5.4, 5.8, 6.2))
    single, series = _pages()
    try:
        series.open_series(folder)
        end = time.monotonic() + 30
        while not series.stages_label.text().startswith("No stages") and time.monotonic() < end:
            series.stage_tasks.wait(0.1)
            QApplication.processEvents()
        reason = f"At least {SHORTEST} frames are needed to find stages."
        assert series.stages_label.text() == f"No stages: {reason}"
        assert series._series_stages is None and series.skip_odd_check.isHidden()
        apply_language("zh", [series])
        series.refresh_language()  # the reason in Chinese once the table has it (zh_entries of this round)
        assert series.stages_label.text() == translate("No stages: {reason}", "zh").format(
            reason=translate(reason, "zh") or reason)
        apply_language(DEFAULT_LANGUAGE, [series])
        series.refresh_language()
        assert series.stages_label.text() == f"No stages: {reason}"
    finally:
        apply_language(DEFAULT_LANGUAGE)
        series.dispose()
        single.dispose()


# -- names readable on a dark theme ----------------------------------------------------------------

def test_stage_and_series_name_colours_are_readable_on_both_themes() -> None:
    from src.gimap.app.presentation.components import STAGE_COLORS, stage_color, stage_text_color, text_color
    from src.gimap.app.presentation.components.curve_plot import CURVE_COLORS
    from src.gimap.app.presentation.theme import DARK, LIGHT
    from tests.test_assistant_gui import _app

    _app()
    with _Theme():
        apply_theme("light")  # darker shades of the same hue where the colour is too light for text on white
        for color in (*STAGE_COLORS, *CURVE_COLORS):
            for surface in (LIGHT["surface"], LIGHT["surface_alt"]):
                assert _contrast(text_color(color), surface) >= 4.5, (color, surface)
        assert stage_text_color(1) == stage_color(1) == "#2563eb"  # a colour that is readable stays as it is
        apply_theme("dark")
        for color in (*STAGE_COLORS, *CURVE_COLORS):
            for surface in (DARK["surface"], DARK["surface_alt"]):
                assert _contrast(text_color(color), surface) >= 4.5, (color, surface)
        assert stage_text_color(1) == QColor(STAGE_COLORS[0]).lighter(150).name()
        assert STAGE_COLORS[0] == "#2563eb" and stage_color(1) == "#2563eb"  # the strips keep their colours


# -- Compare: the result table in view ------------------------------------------------------------

def _compared(page, count: int = 3) -> None:
    for index in range(count):
        page.add_map(_map(index + 1, extra=1.0 if index == 2 else 0.0), f"run {index + 1}")
        _wait(page, lambda: _idle(page) and page.comparison is not None and len(page.comparison.results) == index + 1)
    page.show_step("results")
    QApplication.processEvents()
    QApplication.processEvents()


def _in_view(table) -> bool:
    last = max(column for column in range(table.columnCount()) if not table.isColumnHidden(column))
    return (table.columnViewportPosition(last) + table.columnWidth(last) <= table.viewport().width()
            and table.horizontalScrollBar().maximum() == 0)


def test_the_compare_result_table_keeps_every_column_in_view() -> None:
    page = _shown_page(1400, 900)
    try:
        _compared(page)
        results = page.results_table
        assert not results.isColumnHidden(1)  # three series: the Group column
        assert _in_view(results), [results.columnWidth(column) for column in range(results.columnCount())]
        headers = [results.horizontalHeaderItem(column).text() for column in range(results.columnCount())]
        assert headers[0] == "Series" and headers[1] == "Group"
        assert headers[-1] in ("90 % done", "90 %\ndone") and headers[-2] in ("Half done", "Half\ndone")
        # A wider panel: one line again, still in view; back to the default width: in view again.
        page.splitter.setSizes([760, 640])
        QApplication.processEvents()
        assert results.horizontalHeaderItem(6).text() == "90 % done" and _in_view(results)
        page.splitter.setSizes([440, 960])
        QApplication.processEvents()
        assert _in_view(results)
        # The narrowest panel: two lines; what still does not fit scrolls, and the bar does not hide a row.
        page.splitter.setSizes([300, 1100])
        QApplication.processEvents()
        assert results.horizontalHeaderItem(6).text() == "90 %\ndone"
        assert results.columnWidth(0) >= results.horizontalHeader().sectionSizeHint(0)
        bar = results.horizontalScrollBar()
        rows = sum(results.rowHeight(row) for row in range(results.rowCount()))
        assert results.height() >= results.horizontalHeader().sizeHint().height() + rows + (
            bar.sizeHint().height() if bar.maximum() > 0 else 0)
    finally:
        _close(page)


def test_long_names_give_way_to_the_other_columns() -> None:
    page = _shown_page(1400, 900)
    long_name = "lyx_cu_peo{}_20pl_2p5fr_1{}_00002"
    try:
        for index in range(3):
            page.add_map(_map(index + 1, extra=1.0 if index == 2 else 0.0), long_name.format(index, index) + "_x" * 8)
        _wait(page, lambda: _idle(page) and page.comparison is not None and len(page.comparison.results) == 3)
        page.show_step("results")
        QApplication.processEvents()
        results = page.results_table
        assert _in_view(results) and results.columnWidth(0) >= results.horizontalHeader().sectionSizeHint(0)
        assert results.item(0, 0).toolTip().startswith("lyx_cu_peo0")  # the whole name, elided in the cell
    finally:
        _close(page)


def test_compare_names_follow_the_theme() -> None:
    from src.gimap.features.compare.presentation.render import series_color

    with _Theme():
        page = _shown_page(1400, 900)
        try:
            apply_theme("light")
            _compared(page, 2)
            name = page.distance_table.item(1, 0)
            assert page.results_table.item(0, 0).foreground().color().name() == series_color(0)
            apply_theme("dark")
            QApplication.processEvents()
            assert page.results_table.item(0, 0).foreground().color().name() == QColor(series_color(0)).lighter(150).name()
            assert name.foreground().color().name() == QColor(series_color(1)).lighter(150).name()
            apply_theme("light")
            assert page.results_table.item(0, 0).foreground().color().name() == series_color(0)
        finally:
            _close(page)
    with pytest.raises(TypeError):  # a disposed page no longer follows the theme
        theme_manager().changed.disconnect(page._restyle)


# -- run-time texts in the interface language again --------------------------------------------------

def test_the_compare_page_composes_its_texts_again_after_a_language_switch(tmp_path) -> None:
    page = _shown_page(1400, 900)
    try:
        _compared(page)
        assert page.status_label.text() == "3 series compared: 2 groups."
        apply_language("zh", [page])
        page.refresh_language()
        assert page.chip.text() == "3 个序列 · 180 帧"
        assert page.status_label.text() == "已比较 3 个序列：分成 2 组。"
        assert page.step_rail.detail("results") == "2 组" and "第 1 组：" in page.summary_label.text()
        assert page.component_combo.itemText(0).startswith("成分 1")
        assert page.end_plot.title_label.text().startswith("终态")
        assert page.results_table.horizontalHeaderItem(1).text() == "组"
        assert page.results_table.horizontalHeaderItem(6).text().replace("\n", " ") == "完成 90 %"
        assert _in_view(page.results_table)
        written = page._save_plot(page.change_plot, "change", path=str(tmp_path / "change.png"))
        assert written is not None and page.status_label.text() == "已保存 change.png。"
        apply_language(DEFAULT_LANGUAGE, [page])
        page.refresh_language()
        assert page.chip.text() == "3 series · 180 frames" and page.status_label.text() == "Saved change.png."
        assert page.results_table.horizontalHeaderItem(1).text() == "Group"
        assert page.component_combo.itemText(0).startswith("Component 1")
    finally:
        apply_language(DEFAULT_LANGUAGE)
        _close(page)


def _odd_series(folder: Path, frames: int = 12, odd: int = 5) -> None:
    from tests.test_fit_series import Q_NM, _model
    from src.gimap.features.fitting.application.single_fit import evaluate

    folder.mkdir()
    for index in range(frames):
        exact = evaluate(_model(5.0 + 0.1 * min(index, 6)), Q_NM)
        if index == odd:
            exact = exact * np.exp(-Q_NM)  # a curve unlike its neighbours
        measured = exact + np.random.default_rng(index).normal(0.0, 0.01 * exact)
        np.savetxt(folder / f"run_{index + 1:05d}_fit_input.dat", np.column_stack([Q_NM / 10.0, measured, 0.01 * exact]))


def test_in_situ_stage_names_follow_the_theme_and_the_language(tmp_path) -> None:
    from PyQt5.QtCore import Qt

    from tests.test_fit_series import _model, _pages

    folder = tmp_path / "gimap_analysis"
    _odd_series(folder)
    single, series = _pages()
    with _Theme():
        try:
            apply_theme("light")
            single.open_curve(folder / "run_00001_fit_input.dat")
            single.set_model(_model(5.0))
            series.open_series(folder)
            end = time.monotonic() + 60
            while series._series_stages is None and time.monotonic() < end:
                series.stage_tasks.wait(0.1)
                QApplication.processEvents()
            assert series._series_stages is not None and series.is_odd_frame(5)
            item = series.frame_list.item(5)
            stage = series.stage_of_frame(5)
            # The name keeps the normal text colour; a small square in the stage colour says the stage.
            assert stage >= 1 and item.text().endswith("· odd") and item.data(Qt.ForegroundRole) is None
            assert not item.icon().isNull()
            apply_theme("dark")
            assert item.data(Qt.ForegroundRole) is None and not item.icon().isNull()
            assert item.text().count("odd") == 1  # coloured again, not marked twice
            series._set_item(5)
            assert item.text().endswith("· odd")  # a frame's line made again keeps its mark
            apply_language("zh", [series])
            series.refresh_language()
            assert series.frame_list.item(5).text().endswith("· 异常")
            assert "阶段" in series.stages_label.text() and series.skip_odd_check.text() == "跳过异常帧（1）"
            assert series.step_rail.detail("curves") == "12 条曲线中的 12 条"
            assert series.status_label.text() == translate("{count} curves listed.", "zh").format(count=12)
            apply_language(DEFAULT_LANGUAGE, [series])
            series.refresh_language()
            assert series.frame_list.item(5).text().endswith("· odd") and "odd frames: 6" in series.stages_label.text()
            assert series.status_label.text() == "12 curves listed."
        finally:
            apply_language(DEFAULT_LANGUAGE)
            series.dispose()
            single.dispose()


def test_single_analysis_composes_its_texts_again_after_a_language_switch(tmp_path) -> None:
    from tests.test_fit_series import _pages, _series

    folder = _series(tmp_path)
    single, series = _pages()
    try:
        assert single.open_curve(folder / "run_00001_fit_input.dat")
        width = single.model_editor.row(("globals", "res_width"))
        assert width.name.text() == "Peak w (nm⁻¹)"
        apply_language("zh", [single])
        single.refresh_language()
        assert single.model_editor.row(("globals", "res_width")).name.text() == "峰 w (nm⁻¹)"
        assert "拟合范围内" in single.curve_info.text() and "拟合" in single.step_rail.detail("curve")
        opened = translate("Opened {name}.", "zh").format(name="run_00001_fit_input.dat")
        assert single.status_label.text() == opened != "Opened run_00001_fit_input.dat."
        apply_language(DEFAULT_LANGUAGE, [single])
        single.refresh_language()
        assert single.model_editor.row(("globals", "res_width")).name.text() == "Peak w (nm⁻¹)"
        assert "in the fitting range" in single.curve_info.text()
        assert single.status_label.text() == "Opened run_00001_fit_input.dat."
    finally:
        apply_language(DEFAULT_LANGUAGE)
        series.dispose()
        single.dispose()


# -- the Chinese of the Compare texts ----------------------------------------------------------------

_SHOWN = {"setText", "setToolTip", "addAction", "addItem", "setPrefix", "setSuffix", "addRow", "QLabel", "QPushButton",
          "QCheckBox", "AdvancedSection", "EmptyState", "CurvePlot", "_muted", "tr", "trf", "set_title"}


def _words(text: str) -> bool:
    """Text a person reads (not a template of values only, like “{axis} {span} · {what}”)."""
    return bool(re.search(r"[A-Za-z]{2}", re.sub(r"\{[^}]*\}", "", text)))


def test_the_compare_texts_have_their_chinese() -> None:
    from src.gimap.app.presentation.i18n import ZH
    from src.gimap.features.compare.application import METHOD
    from src.gimap.features.compare.presentation.render import FAILED_HINT, FAILED_TITLE, WAITING_TEXT
    from src.gimap.features.compare.presentation.views.compare_page_view import (
        COMPARE_STEPS, EMPTY_TEXT, EMPTY_TITLE, RESULT_COLUMNS, X_AXES)

    named = (METHOD, FAILED_HINT, FAILED_TITLE, WAITING_TEXT, EMPTY_TITLE, EMPTY_TEXT, "Could not compare",
             "The series changed: save when they are compared again.", "{name} is already a series: this one is {unique}.",
             *(title for _key, title in COMPARE_STEPS), *(text for text, _key in X_AXES), *RESULT_COLUMNS)
    assert [text for text in named if text not in ZH] == []
    missing = []
    for path in sorted((ROOT / "src/gimap/features/compare/presentation").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            name = getattr(node, "func", None) and (getattr(node.func, "id", None) or getattr(node.func, "attr", None))
            if name not in _SHOWN or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str) and _words(first.value):
                if translate(first.value, "zh") is None:
                    missing.append((path.name, node.lineno, first.value))
    assert missing == []
