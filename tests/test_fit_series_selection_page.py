"""Selection probabilities, mode quantiles and discarded Predict retries on the series page."""

from dataclasses import replace

import numpy as np
import pytest

from src.gimap.features.fitting.application.series_selection import PredictAttempt
from src.gimap.features.fitting.application.single_predict import Predicted
from tests.test_fit_series import _model, _open_series, _pages, _series, _wait


def test_series_path_probabilities_quantiles_and_least_squares_toggle(tmp_path, monkeypatch):
    single, page = _pages()
    try:
        folder = _series(tmp_path, (5, 5.05, 5.10, 5.15))
        single.open_curve(folder / "run_00001_fit_input.dat")
        single.set_model(_model(5))
        assert _open_series(page, folder)
        frame_numbers = iter(range(4))

        def predict(data, model):
            i = next(frame_numbers)
            candidates = []
            for p, radius in ((0.55, 8.0 if i % 2 else 13.0), (0.45, 5 + i * 0.05)):
                interval = (radius, radius + 0.05, radius - 0.10, radius + 0.25, radius - 0.20, radius + 0.40, 0.15)
                candidates.append(Predicted(_model(radius), 1.0, "sphere", p, {"R_1": interval}))
            return candidates

        monkeypatch.setattr(page, "predict_solutions", lambda: predict)
        page.method_predict.setChecked(True)
        page.smooth_check.setChecked(True)
        page.skip_odd_check.setChecked(False)
        assert not page.follow_check.isChecked() and not page.backward_check.isEnabled()
        assert page.smooth_scale.value() == pytest.approx(0.1)
        page.start()
        _wait(page)
        assert not page.running and len(page.fits) == 4
        table = page.results_table
        headers = {table.horizontalHeaderItem(i).text(): i for i in range(table.columnCount())}
        for i in range(4):
            assert table.item(i, headers["Chosen by"]).text() == "HMM"
            assert table.item(i, headers["Rank"]).text() == "2"
            assert float(table.item(i, headers["p chosen"]).text()) == pytest.approx(0.45)
            assert float(table.item(i, headers["p max"]).text()) == pytest.approx(0.55)
            assert float(table.item(i, headers["p smoothed"]).text()) > 0.99
        assert "HMM (s = 0.1" in page.selection_summary.text() and "68%" in page.selection_summary.text()
        plot = page.trend_plot
        points = np.arange(4) * 0.05 + 5
        assert len(plot._source) == 1 and plot._source[0][0] == "Posterior median (68%)"
        np.testing.assert_allclose(plot._source[0][2], points + 0.05)
        assert plot.legend.isVisible() and not plot._source_markers[0].hollow
        _, bounds = plot._items[-1].getOriginalDataset()
        np.testing.assert_allclose(bounds[0::2], points - 0.10)
        np.testing.assert_allclose(bounds[1::2], points + 0.25)
        page.least_squares_check.setChecked(True)
        assert [name for name, _, _ in plot._source] == ["Posterior median (68%)", "Predict least squares"]
        np.testing.assert_allclose(plot._source[1][2], points)
        assert plot._source_markers[1].hollow and plot._source_markers[1].size < plot._source_markers[0].size
        page.least_squares_check.setChecked(False)
        assert len(plot._source) == 1
        assert page.project_state()["smooth"] and page.project_state()["smooth_scale"] == 0.1

        # A discarded Predict retry gets probability columns, while its interval cannot appear on the trend.
        final = page.fits[0]
        attempt = PredictAttempt("forward", "a new stage begins", replace(final.selection, rule="closest",
                                                                          reference_frame=1), False)
        page.fits = {0: replace(final, selection=None, predict_attempts=(attempt,))}
        page._settings = replace(page._settings, method="local", start="chain", smooth=False)
        page._rebuild_results()
        page.refresh()
        page._draw_trend()
        assert table.item(0, headers["Chosen by"]).text() == "closest to frame 1 (discarded retry)"
        assert page.least_squares_check.isHidden() and not plot._source_markers[0].hollow
        assert "above 2 and 3×" in page.selection_summary.text() and "0.1 p max" in page.selection_summary.text()
    finally:
        page.dispose()
        single.dispose()
