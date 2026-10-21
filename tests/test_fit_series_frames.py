"""In-situ series of detector frames: each frame is cut with Analyze's set-up (the same curve Analyze writes into
``<stem>_fit_input.dat``), in memory, and fitted as a curve; the record says how the frames were cut."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from tests.test_analyze_workspace import GALAXI, _done, _public_page, requires_public
from tests.test_fit_series import _pages, _wait


def _frames(tmp_path, count=3):
    folder = tmp_path / "beamtime" / "insitu"
    folder.mkdir(parents=True)
    for index in range(count):
        shutil.copy2(GALAXI, folder / f"galaxi_{index + 1:05d}.tif")
    return folder


@requires_public
def test_a_folder_of_frames_is_cut_as_in_analyze_and_fitted(tmp_path) -> None:
    from src.gimap.app.series_frames import SeriesFrames
    from src.gimap.features.analyze.application.export_record import fit_input_curve
    from src.gimap.features.fitting.application import LoadCurveRequest
    from src.gimap.features.fitting.application.single_fit import FitModel, new_component

    folder = _frames(tmp_path)
    analyze = _public_page()
    analyze.choose_mode("gisaxs")
    analyze.add_paths([str(folder / "galaxi_00001.tif")])  # Analyze set up on one frame of the series
    _done(analyze)
    expected = fit_input_curve(analyze.view_model.state.analysis)
    single, series = _pages(SeriesFrames(lambda: analyze.view_model))
    try:
        single.set_model(FitModel((new_component("sphere", radius=5.5, distance=55.0),)))
        assert series.open_series(folder)  # no curves here: the frames are listed, each to be cut
        assert series.pattern_edit.text() == "*.tif" and series.frames_mode() and len(series.paths) == 3
        assert "3 detector frames listed" in series.status_label.text()
        assert series.frames_card.isVisibleTo(series) and "GALAXI Pilatus 1M" in series.frames_label.text()
        assert "αi 0.463°" in series.frames_label.text()  # as the frame was analysed (from its profile)
        assert series.halves_source() == "analyze" and series.halves_note.text() == "as Analyze sends cuts to Fitting"
        before = sorted(path.name for path in folder.iterdir())
        outcome = series.load_series_curve(LoadCurveRequest(series.paths[1], "angstrom"))
        assert outcome.error is None, outcome.error
        curve = outcome.value
        keep = np.isfinite(expected.intensity)
        assert curve.q_source_unit == "angstrom" and np.allclose(curve.q, expected.x)
        assert np.allclose(curve.intensity[keep], expected.intensity[keep])
        assert curve.observation["source"] == "native_detector_columns" and curve.observation["halves"] == "mean"
        assert series.load_series_curve(LoadCurveRequest(series.paths[1], "angstrom")) is outcome  # cut once
        series.start()
        _wait(series, 300)
        frames = [series.fits[index] for index in sorted(series.fits)]
        assert len(frames) == 3 and all(frame.ok for frame in frames), [frame.error for frame in frames]
        assert frames[0].halves_used == "mean" and frames[0].points > 100
        assert sorted(path.name for path in folder.iterdir()) == before  # nothing written beside the data
        path = series.save_table(str(tmp_path / "series.csv"))
        record = json.loads(open(path.replace(".csv", ".json"), encoding="utf-8").read())
        cut = record["listing"]["cut_as_in_analyze"]
        assert record["listing"]["detector_frames"] is True and record["q_unit_of_files"] == "angstrom"
        assert cut["analyze_settings"]["profile_name"] == "GALAXI Pilatus 1M"
        assert cut["analyze_settings"]["format"] == "gimap-analyze-settings" and cut["halves"] == "mean"
        assert cut["set_up_on"]["path"].endswith("galaxi_00001.tif")
        assert record["settings"]["side_from"] == "analyze"
    finally:
        series.dispose()
        single.dispose()
        analyze.dispose()
        analyze.close()


@requires_public
def test_frames_wait_for_analyze_to_be_set_up(tmp_path) -> None:
    from src.gimap.app.series_frames import SeriesFrames
    from src.gimap.features.fitting.application.single_fit import FitModel, new_component

    folder = _frames(tmp_path, 2)
    analyze = _public_page()  # no frame analysed yet
    single, series = _pages(SeriesFrames(lambda: analyze.view_model))
    try:
        single.set_model(FitModel((new_component("sphere"),)))
        series.pattern_edit.setText("*.tif")
        assert series.open_series(folder) and len(series.paths) == 2
        assert "cannot be cut yet" in series.frames_label.text()
        series.start()
        assert not series.running and "open one frame of this series in Analyze" in series.status_label.text()
        analyze.add_paths([str(folder / "galaxi_00001.tif")])
        _done(analyze)
        series.take_setup_button.click()
        assert "GALAXI Pilatus 1M" in series.frames_label.text() and series.ready_to_cut()
    finally:
        series.dispose()
        single.dispose()
        analyze.dispose()
        analyze.close()


def test_frames_of_a_multi_frame_source_are_named_and_summed_as_analyze_sums() -> None:
    from pathlib import Path

    from src.gimap.app.series_frames import SeriesFrames
    from src.gimap.features.fitting.application import InSituSourceFrame
    from src.gimap.features.fitting.application.series_fit import file_name
    from src.gimap.features.fitting.application.series_frames import is_frame, is_frame_pattern, matching

    run = Path("/data/run.nxs")
    items = [(InSituSourceFrame(run, index).token, ()) for index in range(5)]
    assert items[0][0] == str(run) and items[2][0] == f"{run}::gimap-frame=2"
    assert all(is_frame(token) for token, _ in items) and not is_frame("/data/cut_fit_input.dat")
    assert is_frame_pattern("*.cbf") and is_frame_pattern("sample_*.NXS") and not is_frame_pattern("*_fit_input.dat")
    assert file_name(items[2][0], "/data") == "run.nxs · frame 3"
    assert matching(items, "run*.nxs") == items and matching(items, "*.cbf") == []
    summed = SeriesFrames.grouped(items, 2)  # Sum 2: frames 1+2, 3+4, 5
    assert [token for token, _ in summed] == [items[0][0], items[2][0], items[4][0]]
    assert summed[0][1] == ((run, 1),) and summed[1][1] == ((run, 3),) and summed[2][1] == ()
    assert SeriesFrames.grouped(items, 1) == items
