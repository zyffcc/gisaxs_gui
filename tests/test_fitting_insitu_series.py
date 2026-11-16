"""Fitting ▸ In-situ series: a frame of the series opened in Single analysis."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_fit_series import _loaded, _open_series


def test_a_frame_opened_in_single_analysis_keeps_the_settings_of_the_series(tmp_path: Path, monkeypatch) -> None:
    """Open Frame in Single Analysis: the frame's model, on the points the series fitted, so that the
    next Start reads the same halves, range and left-out points."""
    from types import SimpleNamespace

    import pytest

    from src.gimap.features.fitting.application.single_fit import point_key
    from src.gimap.features.fitting.presentation.workspace import FittingWorkspace
    from PyQt5.QtWidgets import QMessageBox

    from tests.test_fit_series import RADII, _answer, _model, _pages, _series
    from tests.test_fit_series import _wait as _wait_series

    _answer(monkeypatch, QMessageBox.Discard)  # the second Start: the first run was not saved
    single, series = _pages()
    folder = _series(tmp_path, RADII[:3])
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    single.set_range((0.5, 1.5))
    left_out = point_key(single.session.data().q[10])
    single.session.excluded.add(left_out)
    single._exclusions_changed()
    _open_series(series, folder)
    series.start()
    _wait_series(series)
    assert len(series.fits) == 3 and series._settings.q_range == (0.5, 1.5)
    single.set_range(None)  # meanwhile the person looks at something else in Single analysis
    single.include_all()
    contexts = []
    host = SimpleNamespace(show_context=contexts.append, series_page=series, fit_page=single)
    series.openFrameRequested.connect(lambda path, model: FittingWorkspace._frame_in_single(host, path, model))
    series.frame_list.setCurrentRow(2)
    _loaded(series)
    series.to_single_button.click()
    session = single.session
    assert contexts == ["single"] and session.curve.name == "run_00003_fit_input.dat"
    assert session.q_range == pytest.approx((0.5, 1.5)) and session.excluded == {left_out}
    assert session.model == series.fits[2].result.model and session.can_undo()
    assert "range and left-out points of the series are kept" in single.status_label.text()
    series.start()  # the next run reads the same settings
    assert series._settings.q_range == pytest.approx((0.5, 1.5)) and series._settings.excluded == frozenset({left_out})
    series.stop()
    _wait_series(series)
    series.dispose()
    single.dispose()
