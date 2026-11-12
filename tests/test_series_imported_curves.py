"""Fixed table-column selections are distinct curves, never an inferred time series."""

import csv
import json
import threading

import numpy as np
import pytest

from src.gimap.features.fitting.application.curve_sources import sources_from_table
from src.gimap.features.fitting.application.curve_units import FROM_FILE
from src.gimap.features.fitting.application.models import InSituSourceFrame
from src.gimap.features.fitting.application.series_frames import file_stamp
from src.gimap.features.fitting.application.single_fit import evaluate
from tests.test_fit_series import Q_NM, _loaded, _model, _pages, _wait
from tests.test_series_loading_responsiveness import _until


def _table(path, radii=(4.8, 6.2), *, unit="angstrom", sigma=False):
    columns = [Q_NM / 10.0 if unit == "angstrom" else Q_NM]
    columns += [evaluate(_model(radius), Q_NM) for radius in radii]
    names = [f"q ({'1/A' if unit == 'angstrom' else '1/nm'})"] + [f"I{i + 1}" for i in range(len(radii))]
    if sigma:
        columns.append(np.full(Q_NM.shape, 2.0))
        names.append("sigma")
    np.savetxt(path, np.column_stack(columns), header="\t".join(names))
    return path


def _sources(series, path, layout="shared_q"):
    return sources_from_table(series.view_model.inspect_curve_table(path), layout)


def test_imported_columns_fit_independently_and_export_real_sources(tmp_path, monkeypatch):
    from src.gimap.features.fitting.presentation.single import series_page

    single, series = _pages()
    try:
        path = _table(tmp_path / "samples.txt")
        sources = _sources(series, path)
        starts, original = [], series_page.fit_frame

        def fitting(index, token, curve, start, settings, stop, predict):
            starts.append((start.get((0, "R")).value, settings.start, curve.sigma))
            return original(index, token, curve, start, settings, stop, predict)

        monkeypatch.setattr(series_page, "fit_frame", fitting)
        single.set_model(_model(5.2))
        assert series.open_imported_sources(sources, independent=True, q_unit="nm")
        _loaded(series)
        assert series.paths == [source.token for source in sources]
        assert series._short_name(0) != series._short_name(1)
        assert series._preview_curve(0).name == sources[0].display_name
        assert series._preview_curve(0).q == pytest.approx(Q_NM)  # header overrides the import fallback
        assert not np.array_equal(series._preview_curve(0).intensity, series._preview_curve(1).intensity)
        assert series._preview_curve(0).sigma is None and series._preview_curve(1).sigma is None
        assert series._start_choice() == "same" and not series.follow_check.isChecked()
        assert not series.skip_odd_check.isChecked() and not series.start_previous.isEnabled()
        assert not series.watch_check.isEnabled() and not series.pattern_edit.isEnabled()
        opened = []
        series.openFrameRequested.connect(lambda token, model: opened.append((token, model)))
        series._select(1)
        assert series.to_single_button.isVisible() and series.to_single_button.isEnabled()
        series.to_single_button.click()
        assert opened == [(sources[1].token, None)]
        opened.clear()
        series.watch_check.setChecked(True)
        assert not series.watch_check.isChecked()
        series._find_stages()
        assert series._series_stages is None and not series.stage_tasks.is_busy()
        series.start()
        _wait(series)
        assert not series.running and len(series.fits) == 2
        assert all(frame.ok and frame.result.weighting == "relative" for frame in series.fits.values())
        assert [frame.result.model.get((0, "R")).value for frame in series.fits.values()] == pytest.approx((4.8, 6.2), rel=.025)
        assert starts == [(5.2, "same", None), (5.2, "same", None)]
        assert file_stamp(sources[0].token) == file_stamp(sources[1].token) == file_stamp(str(path))
        saved = series.save_table(str(tmp_path / "fitted.csv"))
        with open(saved, encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert [row["source_file"] for row in rows] == [str(path)] * 2
        assert [row["intensity_column_1based"] for row in rows] == ["2", "3"]
        assert [row["q_column_1based"] for row in rows] == ["1", "1"]
        assert [row["sigma_column_1based"] for row in rows] == ["", ""]
        assert [row["sigma_source"] for row in rows] == ["none", "none"]
        record = json.loads((tmp_path / "fitted.json").read_text(encoding="utf-8"))
        assert record["listing"]["source_mode"] == "selection" and record["listing"]["independent"]
        for index, frame in enumerate(record["per_frame"]):
            assert frame["curve_source"] == {"path": str(path), "columns": sources[index].columns.to_dict(), "sigma_source": "none"}
            assert frame["q_unit"] == "angstrom" and frame["q_unit_from"] == FROM_FILE
        series._select(1)
        series._open_in_single()
        assert opened == [(sources[1].token, series.fits[1].result.model)]
    finally:
        series.dispose()
        single.dispose()


def test_fixed_selection_project_roundtrip_preserves_file_column_order_and_units(tmp_path):
    single, series = _pages()
    try:
        first = _table(tmp_path / "zeta.txt", unit="nm")
        second = _table(tmp_path / "alpha.dat", radii=(5.6,), sigma=True)
        sources = (*reversed(_sources(series, first)), *_sources(series, second, "single"))
        assert series.open_imported_sources(sources, independent=False, q_unit="nm")
        pending = json.loads(json.dumps(series.project_state()))
        assert pending["sources"][0]["columns"]["intensity"] == 2
        assert pending["first"] == 1 and pending["last"] == 0
        _loaded(series)
        assert series._start_choice() == "previous" and series.start_previous.isEnabled()
        assert series.follow_check.isChecked() and series._preview_curve(2).sigma is not None
        series.first_spin.setValue(2)
        series.last_spin.setValue(3)
        series._set_start("stages")
        state = json.loads(json.dumps(series.project_state()))
        series.close_series()
        assert series.watch_check.isEnabled() and series.pattern_edit.isEnabled()
        assert series.apply_project_state(state) == []
        assert series.project_state()["sources"] == state["sources"]
        _loaded(series)
        assert series.paths == [source.token for source in sources]
        assert series._listed() == [1, 2] and series._start_choice() == "stages"
        assert series._preview_unit() == "nm"
        assert series._preview_curve(1).q == pytest.approx(Q_NM)
        assert series.project_state()["sources"] == state["sources"]
    finally:
        series.dispose()
        single.dispose()


@pytest.mark.parametrize("changed", ["removed", "columns", "header"])
def test_restored_selection_reports_missing_or_changed_source_without_remapping(tmp_path, changed):
    single, series = _pages()
    try:
        path = _table(tmp_path / "selected.txt")
        sources = _sources(series, path)
        series.open_imported_sources(sources, independent=True)
        _loaded(series)
        state = json.loads(json.dumps(series.project_state()))
        series.close_series()
        if changed == "removed":
            path.unlink()
        elif changed == "columns":
            _table(path, radii=(4.8,))
        else:
            path.write_text(path.read_text().replace("I1\tI2", "I2\tI1"), encoding="utf-8")
        assert series.apply_project_state(state) == []
        _loaded(series)
        assert series.paths == [source.token for source in sources]
        assert len(series._read_errors) == 2
        assert all(series._frame_state(index)[0] == "failed" for index in (0, 1))
        assert "cannot be read" in series.status_label.text()
        assert not series.frame_plot.has_curves()
        assert not series.to_single_button.isEnabled()
        assert all(InSituSourceFrame.from_token(token).path == path for token in series._read_errors)
    finally:
        series.dispose()
        single.dispose()


def test_cancelled_import_cannot_replace_a_new_selection(tmp_path, monkeypatch):
    single, series = _pages()
    entered, release = threading.Event(), threading.Event()
    try:
        old_path, new_path = _table(tmp_path / "old.txt"), _table(tmp_path / "new.txt", radii=(5., 7.))
        old, new = _sources(series, old_path), _sources(series, new_path)
        original = series.view_model.inspect_curve_table

        def inspect(path):
            if path == old_path:
                entered.set()
                assert release.wait(20), "test did not release table inspection"
            return original(path)

        monkeypatch.setattr(series.view_model, "inspect_curve_table", inspect)
        series.open_imported_sources(old, independent=True)
        _until(entered.is_set)
        assert series.project_state()["sources"][0]["path"] == str(old_path)
        series.stop()
        assert not series.paths and not series.imported_mode()
        series.open_imported_sources(new, independent=True)
        _loaded(series)
        release.set()
        assert series.loading_tasks.wait(30)
        assert series.paths == [source.token for source in new]
        assert series._preview_curve(1).intensity == pytest.approx(evaluate(_model(7.), Q_NM))
        assert series.project_state()["sources"][0]["path"] == str(new_path)
    finally:
        release.set()
        series.dispose()
        single.dispose()


def test_independent_import_does_not_change_folder_defaults_or_legacy_project_settings(tmp_path):
    single, series = _pages()
    try:
        path = _table(tmp_path / "samples.txt")
        sources = _sources(series, path)
        series._set_start("stages")
        series.follow_check.setChecked(True)
        original = dict(series.preferences.get("fitting_series_page", {}))
        series.open_imported_sources(sources, independent=True)
        _loaded(series)
        series.set_series_method("fit1d")
        state = json.loads(json.dumps(series.project_state()))
        assert not series.follow_check.isEnabled() and not series.follow_check.isChecked()
        assert series.preferences.get("fitting_series_page", {}) == original
        series.close_series()
        assert series._start_choice() == "stages" and series.follow_check.isChecked()
        assert series.skip_odd_check.isChecked() and series.series_method() == "local"
        assert series.apply_project_state(state) == []
        _loaded(series)
        assert series.series_method() == "fit1d"
        assert series.preferences.get("fitting_series_page", {}) == original
        series.open_imported_sources(sources, independent=True)
        _loaded(series)
        assert series.apply_project_state({"folder": str(tmp_path), "pattern": "*.txt", "start": "previous", "follow": False}) == []
        _loaded(series)
        assert not series.imported_mode() and series.watch_check.isEnabled()
        assert series._start_choice() == "previous" and not series.follow_check.isChecked()
    finally:
        series.dispose()
        single.dispose()


@pytest.mark.parametrize("damaged", ["broken_token", "nested_token", "frame_token", "wrong_collection"])
def test_damaged_selection_project_reports_error_before_listing(tmp_path, damaged):
    single, series = _pages()
    try:
        path = _table(tmp_path / "selected.txt")
        sources = _sources(series, path)
        series.open_imported_sources(sources, independent=True)
        _loaded(series)
        state = json.loads(json.dumps(series.project_state()))
        if damaged == "wrong_collection":
            state["sources"] = 42
        else:
            state["sources"][0]["path"] = {
                "broken_token": str(path) + "::gimap-curve=broken",
                "nested_token": sources[0].token,
                "frame_token": str(path) + "::gimap-frame=1",
            }[damaged]
        notes = series.apply_project_state(state)
        assert len(notes) == 1 and "could not be restored" in notes[0]
        assert not series.paths and not series._listing_busy and not series.imported_mode()
        assert series.project_state().get("sources") is None
    finally:
        series.dispose()
        single.dispose()
