"""Release 0.1.0, Fitting: a file that is not a curve is refused with a sentence that says what Fitting reads
(R7: Analyze's .csv tables are not read in this version), and a save dialog never starts in the working folder
of the program (R13)."""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtWidgets import QFileDialog

from src.gimap.features.fitting.application import CURVE_SUFFIXES, UNSUPPORTED_CURVE, LoadCurve, LoadCurveRequest
from src.gimap.features.fitting.infrastructure.adapters import LocalCurveRepository
from tests.test_fit_page import _curve_file, _page

REFUSAL_ZH = "拟合读取 .dat 和 .txt 曲线（q、I、σ）。来自分析的切线，请用“发送到拟合”，或打开它的 _fit_input.dat 文件。"
"""The entry this release adds to the Chinese table (put in here until it is in it)."""


def _table(tmp_path, name: str = "frame_horizontal.csv") -> Path:
    """A table as Analyze exports a cut: a ``#`` header, then comma-separated rows."""
    path = tmp_path / name
    rows = "\n".join(f"{q:.4f},{100.0 / q:.3f},{1.0:.3f},12" for q in np.linspace(0.01, 0.2, 30))
    path.write_text("# GIMaP Analyze cut\n# q in 1/A\nq,I,sigma,pixels\n" + rows + "\n", encoding="utf-8")
    return path


def test_a_csv_table_is_refused_with_what_fitting_reads_and_where_the_curve_is(tmp_path) -> None:
    assert UNSUPPORTED_CURVE == ("Fitting reads .dat and .txt curves (q, I, σ). For a cut from Analyze, use Send to "
                                 "Fitting or open its _fit_input.dat file.")
    assert all(suffix in UNSUPPORTED_CURVE for suffix in CURVE_SUFFIXES) and ".csv" not in CURVE_SUFFIXES
    use_case = LoadCurve(LocalCurveRepository())
    for name in ("frame_horizontal.csv", "frame_horizontal.CSV", "curve.xy", "curve"):
        outcome = use_case.execute(LoadCurveRequest(_table(tmp_path, name)))
        assert outcome.error is not None and outcome.error.code == "unsupported_format"
        assert outcome.error.message == UNSUPPORTED_CURVE  # one fixed sentence: the page can translate it
    for name in ("curve.dat", "curve.TXT"):  # what it does read is read as before
        path = tmp_path / name
        np.savetxt(path, np.column_stack([np.linspace(0.01, 0.2, 30), np.linspace(50, 5, 30)]))
        assert use_case.execute(LoadCurveRequest(path)).succeeded


def test_the_page_says_the_refusal_in_the_interface_language_and_keeps_its_curve(tmp_path, monkeypatch) -> None:
    from src.gimap.app.presentation import i18n

    if UNSUPPORTED_CURVE not in i18n.ZH:
        monkeypatch.setitem(i18n.ZH, UNSUPPORTED_CURVE, REFUSAL_ZH)
        monkeypatch.setitem(i18n._TO_ENGLISH, REFUSAL_ZH, UNSUPPORTED_CURVE)
    page = _page()
    try:
        assert page.open_curve(_curve_file(tmp_path))
        curve = page.session.curve
        table = _table(tmp_path)
        assert page.open_curve(table) is False
        assert page.status_label.text() == f"Could not open {table.name}: {UNSUPPORTED_CURVE}"
        assert "Unsupported curve format" not in page.status_label.text()
        assert page.session.curve is curve  # the curve that was open stays
        i18n.apply_language("zh", [page])
        page.refresh_language()
        assert page.status_label.text() == i18n.ZH["Could not open {name}: {reason}"].format(
            name=table.name, reason=i18n.ZH[UNSUPPORTED_CURVE])
    finally:
        i18n.apply_language("en")
        page.dispose()


@pytest.fixture
def asked(tmp_path, monkeypatch):
    """Save dialogs answer “cancel” and note where they would have started; the program runs in ``program``,
    the person's home is ``home``."""
    home, program = tmp_path / "home", tmp_path / "program"
    home.mkdir()
    program.mkdir()
    monkeypatch.chdir(program)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    starts: list[Path] = []

    def save_name(_parent, _title, start="", *_rest, **_options):
        starts.append(Path(start))
        return "", ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(save_name))
    return types.SimpleNamespace(home=home, program=program, starts=starts)


def test_save_plot_and_save_model_with_no_curve_start_at_home_not_in_the_programs_folder(tmp_path, asked) -> None:
    page = _page()
    assert page.session.curve is None
    assert page.export_plot_dialog() is None and page.save_model_dialog() is None
    assert asked.starts == [asked.home / "fit_fit.png", asked.home / "fit_model.json"]
    assert all(start.is_absolute() and asked.program not in start.parents for start in asked.starts)
    assert sorted(asked.program.iterdir()) == []  # and nothing was written there
    del asked.starts[:]
    page._remember(save_folder=str(tmp_path / "no_longer_there"))  # a remembered folder that is gone
    page.save_model_dialog()
    assert asked.starts == [asked.home / "fit_model.json"]
    del asked.starts[:]
    curves = tmp_path / "curves"
    curves.mkdir()
    page.open_curve(_curve_file(curves))
    page.export_plot_dialog()
    assert asked.starts == [curves / "sample_fit_input_fit.png"]  # the folder of the curve opened last
    del asked.starts[:]
    saved = tmp_path / "figures"
    saved.mkdir()
    page._remember(save_folder=str(saved))
    page.export_data_dialog()
    page.save_model_dialog()
    assert asked.starts == [saved / "sample_fit_input_fit.csv", saved / "sample_fit_input_model.json"]  # the last save
    page.dispose()


def test_the_series_page_saves_a_plot_from_home_when_no_folder_was_chosen(asked) -> None:
    from tests.test_fit_series import _pages
    from tests.test_wave2b_fitting_series_list import _close

    single, series = _pages()
    assert not series.folder
    assert series._save_plot(series.trend_plot, "trend") is None
    assert asked.starts == [asked.home / "series_trend.png"]
    _close(series, single)
