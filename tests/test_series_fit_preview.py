"""Batch fits can be inspected without fitted-curve exports; failures never display another frame's fit."""

from pathlib import Path
from types import SimpleNamespace
from threading import Event

import numpy as np
import pytest

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.application import BatchChoices, batch_record, model_preview
from src.gimap.features.analyze.infrastructure.preview_cache import TemporaryFitPreviewCache
from tests.test_analyze_workspace import _app, _done
from tests.test_series_cut import _close, _gisaxs_page
from tests.test_series_map import _wait


def _solution(radius=4.0):
    return {
        "combination": "sphere", "best_chi2_weighted": 1.2, "best_log_rmse": 0.003,
        "components": [{"type": "sphere", "weight": 1.0, "params": {"R": radius, "D": 35.0}}],
        "native_q": [0.1, 0.123456789, 0.25], "observed": [10.2, 8.3, 5.4], "native_fit": [10.0, 8.1, 5.3],
        "preparation": {"halves": "mean", "points": 3, "averaged": 3, "merged": 1},
    }


def test_preview_copies_only_best_solution_and_converts_q_without_changing_fit_values():
    outcome = SimpleNamespace(label="stack.h5 frame 4 of 8", number=4,
                              metadata={"source_file": "/synthetic/stack.h5", "frame_index": 3})
    best = _solution()
    preview = model_preview(outcome, [best, _solution(12.0)], {"frame": outcome.label})
    assert preview.ref == (Path("/synthetic/stack.h5"), 3) and preview.number == 4 and preview.ok
    assert len(preview.curves) == 1
    curve = preview.curves[0]
    np.testing.assert_array_equal(curve.x, np.asarray(best["native_q"]) * 0.1)
    np.testing.assert_array_equal(curve.observed, best["observed"])
    np.testing.assert_array_equal(curve.fitted, best["native_fit"])
    best["observed"][0] = 99
    assert curve.observed[0] == 10.2


def test_gui_browses_batch_fits_without_files_and_clears_failed_frame(tmp_path):
    page = _gisaxs_page(tmp_path / "data", bands=(204, 204, 204))
    calls = []

    def fit(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("No intensity above zero on the q > 0 half")
        return [_solution(4.0 if len(calls) == 1 else 5.0)]

    try:
        page.view_model.model_fitter = fit
        output = tmp_path / "out"
        choices = BatchChoices(tables=False, fit="model", fit_model="sphere", fit_curves=False, speed="gentle")
        page.run_batch(output, choices)
        _wait(page, lambda: not page.batch_running())
        _done(page)
        run = page._batch_run
        assert len(run.fit_previews) == 3 and not (output / "fits").exists()
        assert "fit_previews" not in batch_record(run, [])
        assert not page.series_fit_results.isHidden() and page.series_fit_table.rowCount() == 3
        assert "2 fitted" in page.series_fit_summary.text() and "1 failed" in page.series_fit_summary.text()
        page.series_fit_table.setCurrentCell(0, 0)
        data, fitted = page.series_fit_plot._source
        np.testing.assert_array_equal(data[1], np.array(_solution()["native_q"]) * 0.1)
        np.testing.assert_array_equal(data[2], _solution()["observed"])
        np.testing.assert_array_equal(fitted[2], _solution()["native_fit"])
        assert "Å⁻¹" in page.series_fit_plot.plot.getAxis("bottom").labelText
        page.series_fit_parameter.setCurrentIndex(page.series_fit_parameter.findData("c1 sphere R"))
        trend = page.series_fit_trend_plot._source[0]
        np.testing.assert_array_equal(trend[1], [1, 2, 3])
        np.testing.assert_allclose(trend[2], [4, np.nan, 5], equal_nan=True)
        assert "nm" in page.series_fit_trend_plot.plot.getAxis("left").labelText
        page.series_fit_table.setCurrentCell(1, 0)
        assert page.series_fit_plot._source == [] and not page.series_fit_plot.has_curves()
        assert "No intensity above zero" in page.series_fit_note.text()
        apply_language("zh", [page])
        page.refresh_language()
        assert page.series_fit_results.title() == "结果：批量拟合"
        assert "失败" in page.series_fit_table.item(1, 1).text() and "没有大于零" in page.series_fit_note.text()
        page.clear_files()
        _app().processEvents()
        assert page.series_fit_results.isHidden() and page._series_fit_run is None
        assert page.series_fit_plot._source == []
    finally:
        apply_language("en", [page])
        _close(page)


def test_selected_fit_stays_selected_while_more_frames_arrive_and_clear_rejects_late_results(tmp_path):
    page = _gisaxs_page(tmp_path / "data", bands=(204, 204))
    try:
        run = page.view_model.start_batch(tmp_path / "out", BatchChoices(fit="model"), "series")
        run.requested = 2
        page._batch_run = run
        page._series_fit_start(run)
        for index in range(2):
            outcome = SimpleNamespace(label=f"frame {index + 1}", number=index + 1,
                                      metadata={"source_file": f"/synthetic/frame_{index}.tif", "frame_index": 0})
            values = {"frame": outcome.label, "chi2": 1.2, "c1 sphere R": 4.0 + index}
            run.fit_table.add(values, index + 1)
            run.fit_previews.append(model_preview(outcome, [_solution(4 + index)], values))
            page._series_fit_render()
            if index == 0:
                page.series_fit_table.setCurrentCell(0, 1)  # an intentional selection stops following
        assert page.series_fit_table.currentRow() == 0
        assert page.series_fit_plot.title_label.text() == "frame 1"
        run.stopped = True
        page._series_fit_changed(finished=True)
        assert page.series_fit_summary.text().startswith("Stopped:")
        page._reset_series_fit()
        page._series_fit_changed(finished=True)
        assert page.series_fit_results.isHidden() and page.series_fit_table.rowCount() == 0
    finally:
        _close(page)


def test_older_fit_curves_spill_to_a_bounded_private_cache_and_load_on_demand(tmp_path):
    page = _gisaxs_page(tmp_path / "data", bands=(204, 204, 204))
    try:
        page.view_model.preview_cache_factory = lambda: TemporaryFitPreviewCache(max_frames=1, max_bytes=1024)
        page.view_model.model_fitter = lambda *_args, **_kwargs: [_solution()]
        output = tmp_path / "out"
        page.run_batch(output, BatchChoices(tables=False, fit="model", fit_model="sphere", fit_curves=False))
        _wait(page, lambda: not page.batch_running())
        _done(page)
        run, cache = page._batch_run, page._batch_run.preview_cache
        assert cache.resident_frames == 1 and cache.resident_bytes <= 1024
        assert run.fit_previews[0].curves == () and run.fit_previews[0].cache_key == 0
        folder = cache.directory
        assert folder.exists() and not folder.is_relative_to(output)
        assert "preview_cache" not in batch_record(run, []) and not (output / "fits").exists()
        page.series_fit_table.setCurrentCell(0, 0)
        assert page.tasks.is_busy() and page.series_fit_plot._source == []
        _done(page)
        data, fitted = page.series_fit_plot._source
        np.testing.assert_array_equal(data[1], np.array(_solution()["native_q"]) * 0.1)
        np.testing.assert_array_equal(fitted[2], _solution()["native_fit"])
        assert cache.resident_frames == 1  # looking at an old curve does not grow the resident cache
        page.clear_files()
        assert not folder.exists() and cache.closed and not run.fit_previews
    finally:
        _close(page)


def test_preview_cache_respects_bytes_and_returns_the_exact_prepared_points():
    outcome = SimpleNamespace(label="f1", number=1, metadata={})
    preview = model_preview(outcome, [_solution()], {})
    cache = TemporaryFitPreviewCache(max_frames=100, max_bytes=80)
    try:
        assert cache.remember(0, preview.curves) == []
        assert cache.remember(1, preview.curves) == [0]
        assert cache.resident_frames == 1 and cache.resident_bytes == 72
        recovered = cache.read(0)[0]
        for name in ("x", "observed", "fitted"):
            np.testing.assert_array_equal(getattr(recovered, name), getattr(preview.curves[0], name))
    finally:
        folder = cache.directory
        cache.close()
        assert not folder.exists()


def test_read_failure_keeps_its_original_frame_number_in_results_and_trends(tmp_path):
    page = _gisaxs_page(tmp_path / "data", bands=(204, 204, 204))
    try:
        page.view_model.state.files[1].write_bytes(b"not a detector frame")
        page.view_model.model_fitter = lambda *_args, **_kwargs: [_solution()]
        page.run_batch(tmp_path / "out", BatchChoices(tables=False, fit="model", fit_model="sphere"))
        _wait(page, lambda: not page.batch_running())
        _done(page)
        assert [entry.number for entry in page._series_fit_entries] == [1, 2, 3]
        assert [entry.ok for entry in page._series_fit_entries] == [True, False, True]
        page.series_fit_parameter.setCurrentIndex(page.series_fit_parameter.findData("c1 sphere R"))
        _name, x, y = page.series_fit_trend_plot._source[0]
        np.testing.assert_array_equal(x, [1, 2, 3])
        np.testing.assert_allclose(y, [4, np.nan, 4], equal_nan=True)
        page.series_fit_table.setCurrentCell(1, 0)
        assert page.series_fit_plot._source == [] and page.series_fit_note.text()
    finally:
        _close(page)


def test_final_summary_is_saved_in_a_worker_and_keeps_batch_busy_until_done(tmp_path, monkeypatch):
    from PyQt5.QtCore import QTimer

    page = _gisaxs_page(tmp_path / "data", bands=(204, 204))
    entered, release = Event(), Event()
    finish = page.view_model.finish_batch

    def held(run):
        entered.set()
        release.wait(10)
        return finish(run)

    ticks = []
    timer = QTimer(page)
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(1))
    try:
        monkeypatch.setattr(page.view_model, "finish_batch", held)
        timer.start()
        page.run_batch(tmp_path / "out", BatchChoices(tables=True))
        _wait(page, lambda: entered.is_set())
        before = len(ticks)
        _wait(page, lambda: len(ticks) > before + 2)
        assert page.batch_running() and page._batch_finishing
        assert "Saving tables" in page.status_text()
        release.set()
        _wait(page, lambda: not page.batch_running())
        assert (tmp_path / "out" / "batch_batch.json").exists()
    finally:
        timer.stop()
        release.set()
        _close(page)


def test_unwritable_preview_cache_does_not_change_fit_success_or_grow_ram(tmp_path, monkeypatch):
    page = _gisaxs_page(tmp_path / "data", bands=(204, 204))
    try:
        page.view_model.preview_cache_factory = lambda: TemporaryFitPreviewCache(max_frames=1)
        page.view_model.model_fitter = lambda *_args, **_kwargs: [_solution()]
        monkeypatch.setattr(np, "savez", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("temporary disk full")))
        page.run_batch(tmp_path / "out", BatchChoices(tables=False, fit="model", fit_model="sphere"))
        _wait(page, lambda: not page.batch_running())
        _done(page)
        run = page._batch_run
        assert not run.failed and all(frame["state"] == "ok" for frame in run.fit_frames)
        assert run.preview_cache.resident_frames == 1
        with pytest.raises(OSError, match="temporary disk full"):
            page.view_model.batch_preview_curves(run, run.fit_previews[0])
        page.series_fit_table.setCurrentCell(0, 0)
        _done(page)
        assert "Could not load this fit preview" in page.series_fit_note.text()
        assert page.series_fit_table.item(0, 1).text() == "Fitted"
    finally:
        _close(page)


def test_a_few_odd_rows_do_not_materialize_a_million_lazy_requests():
    from src.gimap.features.analyze.presentation.excluded_frames import ExcludedFrames

    touched = []

    class Requests:
        def __len__(self):
            return 1_000_000

        def __getitem__(self, index):
            touched.append(index)
            return index

    kept = ExcludedFrames(Requests(), [0, 123, 999_999])
    assert len(kept) == 999_997 and not touched
    assert kept[0] == 1 and kept[122] == 124 and kept[-1] == 999_998
    assert touched == [1, 124, 999_998]
