"""ALS 7.3.3 frames: Pilatus module gaps stored as 0, the ``.txt`` side file (αi, monitors), and a series that
switches between incidence angles and exposure times. All data here are synthetic; the side file follows the layout
of the public 7.3.3 files (``Motors:`` lines, ``Sample Alpha Stage``, ``Izero``, ``I1``)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.analyze.application import AnalysisRequest
from src.gimap.features.analyze.application.export_record import analysis_metadata
from src.gimap.features.analyze.application.reduction_inputs import frame_references, source_files
from src.gimap.features.analyze.application.series_angles import (
    ALL, default_incidence, incidence_groups, incidence_line, split_rows,
)
from src.gimap.features.analyze.application.series_map import RowLabel
from src.gimap.features.analyze.domain.normalisation import header_values
from src.gimap.shared.detector_io import load_detector_image
from src.gimap.shared.detector_io.module_gaps import layout_gaps, zero_gaps
from src.gimap.shared.detector_io.side_files import side_file_metadata
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from tests.test_analyze_workspace import _app, _context
from tests.test_assistant_calibration import CENTER, DISTANCE_M, PIXEL, SHAPE, WAVELENGTH
from tests.test_normalisation import _analyze, _save_edf
from tests.test_series_map import _ring_frame, _wait


def _side(image: Path, alpha: float, izero: float = 12000.0, *, motors: bool = True) -> Path:
    lines = ["12000.000", "8000.000", "3.4e5"]
    if motors:
        lines.append("Motors: 3")
    lines += ["Sample X Stage: -1.250000", f"Sample Alpha Stage: {alpha:.6f}", "Sample Phi Stage: 0.000000",
              "AIs: 3", "Beam Current: 500.100000", f"Izero: {izero:.6f}", "I1: 8000.000000",
              "Vertical Beam Position: NaN", "!0 ", "Alpha_scan_positions=", ""]
    path = image.with_suffix(".txt")
    path.write_text("\n".join(lines), encoding="ascii")
    return path


# -- module gaps -------------------------------------------------------------------------------------------------------


def test_the_zero_gaps_of_a_pilatus_2m_are_left_out_and_recorded(tmp_path: Path) -> None:
    name, gaps = layout_gaps((1679, 1475))
    assert name == "Pilatus 2M" and int(gaps.sum()) == 197365  # 7 bands of 17 rows, 2 of 7 columns
    assert layout_gaps((1475, 1679))[1].shape == (1475, 1679) and layout_gaps((400, 400)) == (None, None)
    counts = np.random.default_rng(3).poisson(0.3, (1679, 1475)).astype(np.int32)  # a short exposure: many zeros
    counts[gaps] = 0
    path = _save_edf(tmp_path / "film_2m_00001.edf", counts, {"count_time": "0.1"})
    image = load_detector_image(path)
    assert np.array_equal(image.mask, gaps) and image.metadata["module_gaps"]["pixels"] == 197365
    assert "module gaps" in image.metadata["mask_semantics"]
    # A zero count inside a module is a measurement.
    assert not image.mask[~gaps].any() and (counts[~gaps] == 0).any()
    # Gap rows that a file filled with values keep them; gap columns of zeros are still left out.
    filled = counts.copy()
    filled[gaps.all(axis=1), :] = 2
    filled[:, gaps.all(axis=0)] = 0
    left_out, record = zero_gaps(filled)
    assert record["pixels"] == int(gaps.all(axis=0).sum()) * 1679 and not left_out[gaps.all(axis=1), 0].any()
    # A floating-point frame (dark- or background-subtracted) is never touched: its 0 is data.
    assert zero_gaps(counts.astype(np.float32)) == (None, None)


def test_the_analysis_record_says_how_many_gap_pixels_were_left_out(tmp_path: Path) -> None:
    _name, gaps = layout_gaps((1679, 1475))
    counts = np.full((1679, 1475), 5, dtype=np.int32)
    counts[gaps] = 0
    path = _save_edf(tmp_path / "film_2m_00002.edf", counts, {"count_time": "1"})
    profile = InstrumentProfile("ALS 2M", DetectorGeometry(172e-6, 172e-6, 3.5, 685.0, 1650.0, 1.2398, 0.4),
                                None, (1679, 1475))
    frame = _analyze(profile)(AnalysisRequest(path=path, mode="gisaxs", profile_name="ALS 2M"))
    assert not np.asarray(frame.valid)[gaps].any()
    assert analysis_metadata(frame)["module_gaps_left_out"]["pixels"] == 197365


# -- the side file -----------------------------------------------------------------------------------------------------


def test_the_side_file_gives_the_incidence_angle_and_monitors(tmp_path: Path) -> None:
    path = _save_edf(tmp_path / "AuCNF_RUN1_2m_00007.edf", _ring_frame(100.0).astype(np.int32), {"count_time": "0.097"})
    side = _side(path, 0.25, izero=12766.5)
    metadata = load_detector_image(path).metadata
    assert metadata["incidence_deg"] == pytest.approx(0.25)
    assert metadata["incidence_source"] == f"side file {side.name}: Sample Alpha Stage"
    values = header_values(metadata)
    assert values["exposure_time_s"] == pytest.approx(0.097)
    assert values["monitors"] == {"Izero": pytest.approx(12766.5), "I1": 8000.0, "Beam Current": 500.1}
    # The side file is one of the files the frame was read from: recorded with its digest.
    assert source_files(path, metadata) == [str(path), str(side)]
    assert set(frame_references(path, metadata)) == {str(path), str(side)}
    # An angle the image header states is kept; a .txt without the motor list is not a side file.
    assert "incidence_deg" not in side_file_metadata(path, {"incidence_deg": 0.3})
    _side(path, 0.25, motors=False)
    assert side_file_metadata(path, {}) == {}


def test_each_frame_is_reduced_at_its_side_file_angle_unless_one_is_typed(tmp_path: Path) -> None:
    path = _save_edf(tmp_path / "AuCNF_RUN1_2m_00008.edf", _ring_frame(100.0).astype(np.int32), {"count_time": "1"})
    _side(path, 0.1)
    profile = InstrumentProfile("synthetic", DetectorGeometry(PIXEL, PIXEL, DISTANCE_M, CENTER[0], CENTER[1],
                                                              WAVELENGTH, 0.4), None, SHAPE)
    analyze = _analyze(profile)
    frame = analyze(AnalysisRequest(path=path, mode="giwaxs", profile_name="synthetic"))
    assert frame.geometry.incidence_deg == pytest.approx(0.1)
    record = analysis_metadata(frame)
    assert record["geometry_resolution"]["nominal_incidence_source"].startswith("side file ")
    assert {Path(item["file"]).suffix for item in record["input_sha256"]} >= {".edf", ".txt"}
    typed = analyze(AnalysisRequest(path=path, mode="giwaxs", profile_name="synthetic", incidence_deg=0.3))
    assert typed.geometry.incidence_deg == pytest.approx(0.3)


# -- one map per incidence angle ---------------------------------------------------------------------------------------


def _row(index: int, angle, exposure: float = 1.0):
    label = RowLabel(f"f{index}", source=f"f{index}.edf", frame_index=0, incidence_deg=angle, exposure_s=exposure)
    return (np.arange(3.0), np.ones(3), "q", label, (f"f{index}.edf", 0))


def test_a_map_holds_the_angle_of_most_frames_and_a_scan_stays_whole() -> None:
    spray = [_row(i, angle) for i, angle in enumerate([0.4, 0.4, 0.1, 0.25, 0.4, 0.399998, 0.6, 0.4])]
    assert [(angle, n) for angle, n in incidence_groups(spray)] == [(0.1, 1), (0.25, 1), (0.4, 5), (0.6, 1)]
    assert default_incidence(spray) == 0.4
    kept, aside = split_rows(spray, 0.4)
    assert len(kept) == 5 and len(aside) == 3
    assert incidence_line(0.4, aside) == ("frames at alpha_i = 0.4 deg only; 3 frames at other angles left out "
                                          "(0.1 deg: 1, 0.25 deg: 1, 0.6 deg: 1)")
    scan = [_row(i, 0.05 + 0.01 * i) for i in range(10)]  # every angle once: one map of the scan
    assert default_incidence(scan) == ALL and split_rows(scan, ALL) == (scan, [])
    assert default_incidence([_row(0, 0.2), _row(1, 0.3), _row(2, 0.3), _row(3, 0.2)]) == 0.2  # a tie: the first's
    assert default_incidence([_row(0, None), _row(1, None)]) == ALL  # transmission frames: no angle


def test_the_series_tab_shows_one_angle_and_offers_to_divide_by_exposure(tmp_path: Path) -> None:
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository

    _app()
    plan = [(0.4, 1.0), (0.4, 1.0), (0.1, 1.0), (0.4, 0.1), (0.25, 1.0), (0.4, 0.1), (0.4, 0.1)]
    paths = []
    for index, (angle, exposure) in enumerate(plan):
        counts = (_ring_frame(100.0 * index, seed=index) * exposure).astype(np.int32)
        path = _save_edf(tmp_path / f"spray_2m_{index:05d}.edf", counts, {"count_time": str(exposure)})
        _side(path, angle)
        paths.append(path)
    profile = InstrumentProfile("synthetic", DetectorGeometry(PIXEL, PIXEL, DISTANCE_M, CENTER[0], CENTER[1],
                                                              WAVELENGTH, 0.4), None, SHAPE)
    page = AnalyzePage(create_analyze_view_model(_context(InMemoryInstrumentProfileRepository([profile]))))
    try:
        page.set_mode_choice("giwaxs")
        page.add_paths([str(path) for path in paths])
        page.tasks.wait(60)
        _app().processEvents()
        page.series_build_button.click()
        _wait(page, lambda: page._series_map is not None and not page._series_queue and not page.batch_running()
              and len(page._series_rows) + len(page._series_aside) == len(plan))
        series = page._series_map
        assert series.rows == 5 and all(fact.incidence_deg == pytest.approx(0.4) for fact in series.facts)
        combo = page.series_angle_combo
        assert not combo.isHidden() and [combo.itemData(i) for i in range(combo.count())] == [0.1, 0.25, 0.4, ALL]
        assert "αi 0.4° only (2 frames at other angles" in page.series_title_label.text()
        assert "alpha_i = 0.4 deg only" in series.built_with["incidence"]
        # Another angle: its own map; all angles: every frame.
        page._series_angle_chosen(combo.findData(0.1))
        assert page._series_map.rows == 1 and str(page._series_map.refs[0][0]) == str(paths[2])
        page._series_angle_chosen(combo.findData(ALL))
        assert page._series_map.rows == len(plan) and not page._series_angle_words()
        page._series_angle_chosen(combo.findData(0.4))
        # Exposures of 1 s and 0.1 s, intensity not divided by them: a note, and the way to divide.
        note = page.series_exposure_label
        assert not note.isHidden() and "0.1, 1 s" in note.text()
        page.series_divide_by_exposure()
        assert page.view_model.state.corrections.normalisation.exposure == "header"
        _wait(page, lambda: not page.batch_running() and page._series_map is not None and page._series_map.rows == 5)
        assert note.isHidden() and page._series_angle_shown == 0.4
    finally:
        page.dispose()
        time.sleep(0.05)


# -- a watched folder reports a frame as soon as it is whole -----------------------------------------------------------


def test_an_edf_that_holds_what_its_header_announces_is_complete(tmp_path: Path) -> None:
    from src.gimap.features.analyze.infrastructure.adapters.frames import DetectorIoFrameSource
    from src.gimap.shared.detector_io.completeness import edf_written

    whole = _save_edf(tmp_path / "run_2m_00001.edf", np.ones((40, 30), dtype=np.int32), {"count_time": "0.1"})
    data = whole.read_bytes()
    half = tmp_path / "run_2m_00002.edf"
    half.write_bytes(data[: len(data) - 100])  # the detector is still writing it
    assert edf_written(whole) is True and edf_written(half) is False
    assert edf_written(tmp_path / "missing.edf") is False
    frames = DetectorIoFrameSource()
    assert frames.complete(whole)  # no side files in this folder: the image alone
    _side(whole, 0.4)
    third = tmp_path / "run_2m_00003.edf"
    third.write_bytes(data)
    assert frames.complete(whole) and not DetectorIoFrameSource().complete(third)  # its side file is still to come
    _side(third, 0.1)
    assert frames.complete(third)
    tif = tmp_path / "run_00001.tif"
    tif.write_bytes(b"II*\x00")
    assert not frames.complete(tif)  # other formats: two looks of the same size decide


def test_the_watch_reports_a_complete_frame_at_the_first_look(tmp_path: Path) -> None:
    from src.gimap.features.analyze.infrastructure.adapters.frames import DetectorIoFrameSource
    from src.gimap.features.analyze.presentation.folder_watch import FolderWatch

    frames = DetectorIoFrameSource()
    watch = FolderWatch(frames.expand, frames.settled_size, None, frames.complete)
    watch.start(tmp_path)
    whole = _save_edf(tmp_path / "run_2m_00001.edf", np.ones((40, 30), dtype=np.int32), {"count_time": "0.1"})
    data = whole.read_bytes()
    assert watch.poll() == [whole]  # whole at the first look
    half = tmp_path / "run_2m_00002.edf"
    half.write_bytes(data[:600])
    assert watch.poll() == []  # being written: not yet
    half.write_bytes(data)
    assert watch.poll() == [half]
    plain = FolderWatch(frames.expand, frames.settled_size)  # without the check: two looks of the same size
    plain.start(tmp_path)
    third = tmp_path / "run_2m_00003.edf"
    third.write_bytes(data)
    assert plain.poll() == [] and plain.poll() == [third]


# -- times of frames faster than their header stamps -------------------------------------------------------------------


def test_the_edf_date_is_a_time_stamp_and_the_file_time_is_kept(tmp_path: Path) -> None:
    path = _save_edf(tmp_path / "run_2m_00001.edf", np.ones((40, 30), dtype=np.int32),
                     {"count_time": "0.097", "Date": "Sun Sep  7 02:49:57 2014"})
    metadata = load_detector_image(path).metadata
    assert metadata["timestamp"] == "2014-09-07T02:49:57"
    assert metadata["file_time"].startswith(time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime)))


def test_frames_faster_than_whole_second_stamps_take_their_file_times() -> None:
    from src.gimap.features.analyze.application.series_map import series_times
    from src.gimap.features.analyze.domain.series_map import RowFacts

    def facts(files):
        stamps = [f"2027-01-20T14:00:{int(0.3 + 0.1 * i):02d}" for i in range(len(files))]  # 10 frames a second
        return [RowFacts(number=i + 1, source=f"f{i}.edf", timestamp=stamp, exposure_s=0.097, file_time=file)
                for i, (stamp, file) in enumerate(zip(stamps, files))]

    written = [f"2027-01-20T14:00:{0.35 + 0.1 * i:09.6f}" for i in range(12)]
    times, source = series_times(facts(written))
    assert source == "file_times" and np.allclose(times, 0.1 * np.arange(12))
    copied = [f"2027-03-02T09:{i:02d}:00.000000" for i in range(12)]  # copied later without their times
    times, source = series_times(facts(copied))
    assert source == "timestamps" and times[1] == 0.0  # the header's whole seconds, as stamped


# -- events of the run (spray pulses) on the map -----------------------------------------------------------------------


def test_an_events_file_names_its_time_column_and_its_events(tmp_path: Path) -> None:
    from src.gimap.features.analyze.application.series_events import read_events

    log = tmp_path / "spray_log.csv"
    log.write_text("pulse,utc,run_time_s\n2,2027-01-20T14:00:23.75+00:00,23.75\n1,2027-01-20T14:00:08.55+00:00,8.55\n"
                   "3,not a time,0\n", encoding="utf-8")
    events = read_events(log)
    assert [event.label for event in events] == ["1", "2"]  # in time order; a line without a time is left out
    assert events[1].seconds - events[0].seconds == pytest.approx(15.2)
    (tmp_path / "notes.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has no time column"):
        read_events(tmp_path / "notes.csv")


def test_the_series_map_marks_each_event_above_the_first_frame_after_it(tmp_path: Path) -> None:
    import os
    from datetime import datetime, timedelta

    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository

    _app()
    start = datetime(2027, 1, 20, 14, 0, 0)
    paths = []
    for index in range(6):
        moment = start + timedelta(seconds=index)
        path = _save_edf(tmp_path / f"run_2m_{index:05d}.edf", (_ring_frame(50.0 * index, seed=index)).astype(np.int32),
                         {"count_time": "1", "Date": moment.strftime("%a %b %d %H:%M:%S %Y")})
        written = (moment + timedelta(seconds=0.4)).timestamp()
        os.utime(path, (written, written))
        paths.append(path)
    log = tmp_path / "spray_log.csv"
    log.write_text("label,time\npulse 1,2027-01-20T14:00:02.5\nlater,2027-01-20T15:00:00\n", encoding="utf-8")
    profile = InstrumentProfile("synthetic", DetectorGeometry(PIXEL, PIXEL, DISTANCE_M, CENTER[0], CENTER[1],
                                                              WAVELENGTH, 0.2), None, SHAPE)
    page = AnalyzePage(create_analyze_view_model(_context(InMemoryInstrumentProfileRepository([profile]))))
    try:
        page.set_mode_choice("giwaxs")
        page.add_paths([str(path) for path in paths])
        page.tasks.wait(60)
        _app().processEvents()
        assert page.load_series_events(log)
        assert page.series_events_label.text() == "2 events of spray_log.csv: marked on the map once it is built"
        page.series_build_button.click()
        _wait(page, lambda: page._series_map is not None and page._series_map.rows == 6 and not page.batch_running())
        series = page._series_map
        assert series.times is not None and np.allclose(series.times, np.arange(6.0))
        assert page.series_events_label.text() == "1 of 2 events of spray_log.csv on the map"
        import pyqtgraph as pg

        lines = [item for item in page._series_event_items if isinstance(item, pg.InfiniteLine)]
        assert len(lines) == 1 and lines[0].value() == 3.0  # above row 4, the first frame after 14:00:02.5
        record = series.built_with["events"]["events"]
        assert record[0] == {"label": "pulse 1", "time": "2027-01-20T14:00:02.5", "row": 4,
                             "seconds_after_first_row": 2.5} and record[1]["row"] is None
        page.clear_series_events()
        assert page._series_event_items == [] and page.series_events_label.text() == "No events file"
    finally:
        page.dispose()
        time.sleep(0.05)


def test_a_look_at_a_folder_of_many_frames_asks_the_file_system_per_folder_not_per_frame(tmp_path: Path,
                                                                                      monkeypatch) -> None:
    """A run of the beamtime leaves tens of thousands of frames in a folder, and the watch looks every second or
    two: the frames already there are measured from the folder listing (with their side files), and their paths
    are not resolved one by one (a look at 20 000 frames took 23 s when each frame was asked for separately)."""
    from src.gimap.features.analyze.infrastructure.adapters.frames import DetectorIoFrameSource
    from src.gimap.features.analyze.presentation.folder_watch import FolderWatch

    data = _save_edf(tmp_path / "run_2m_00000.edf", np.ones((8, 6), dtype=np.int32), {"count_time": "0.1"}).read_bytes()
    for number in range(1, 300):
        (tmp_path / f"run_2m_{number:05d}.edf").write_bytes(data)
    for number in range(0, 300, 3):
        _side(tmp_path / f"run_2m_{number:05d}.edf", 0.4)
    frames = DetectorIoFrameSource()
    watch = FolderWatch(frames.expand, frames.settled_size, None, frames.complete)
    watch.start(tmp_path)
    calls = {"resolve": 0, "stat": 0}
    resolve, stat = Path.resolve, Path.stat

    def counted(name, real):
        def call(self, *args, **kwargs):
            calls[name] += 1
            return real(self, *args, **kwargs)
        return call

    monkeypatch.setattr(Path, "resolve", counted("resolve", resolve))
    monkeypatch.setattr(Path, "stat", counted("stat", stat))
    assert watch.poll() == []
    new = tmp_path / "run_2m_00300.edf"
    new.write_bytes(data)
    _side(new, 0.4)
    assert [path.name for path in watch.poll()] == ["run_2m_00300.edf"]  # whole, with its side file
    assert calls["resolve"] <= 4 and calls["stat"] <= 12, calls  # 300 frames: per folder and per new frame only
    # The size a look takes from the listing is the frame's with its side file, as asking the file system gives it.
    monkeypatch.setattr(Path, "stat", stat)
    listed = frames.settled_size(tmp_path.resolve() / "run_2m_00003.edf")
    assert listed == (tmp_path / "run_2m_00003.edf").stat().st_size + (tmp_path / "run_2m_00003.txt").stat().st_size
    assert frames.settled_size(tmp_path.resolve() / "run_2m_00004.edf") == len(data)
