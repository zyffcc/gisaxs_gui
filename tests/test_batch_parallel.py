"""Batch Export with several frames at once: the policy, worker processes, order, Stop, Pause, live map."""

from __future__ import annotations

import csv
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.analyze.application import (
    AnalysisRequest,
    BatchChoices,
    CutRegion,
    FrameJob,
    FrameTools,
    frames_at_once,
    run_frame_job,
)
from src.gimap.shared.geometry import InstrumentProfile
from tests.test_assistant_calibration import SHAPE, giwaxs_frame, save_tiff
from tests.test_giwaxs_workspace import GEOMETRY, _settle

LAMBDA_PIXELS = 4727 * 3142
GB = 1_000_000_000


# -- how many at once --------------------------------------------------------------------------


def test_frames_at_once_follows_the_speed_the_cores_and_the_memory() -> None:
    long = dict(cores=24, free_bytes=20 * GB, frame_pixels=LAMBDA_PIXELS, frames=403)
    assert frames_at_once("gentle", **long) == 1
    assert frames_at_once("balanced", **long) == 4
    assert frames_at_once("fast", **long) == 5  # 8 by the cores, 5 by half of 20 GB at 1.7 GB a frame
    assert frames_at_once("fast", **{**long, "free_bytes": 64 * GB}) == 8
    assert frames_at_once("balanced", **{**long, "free_bytes": 3 * GB}) == 1  # little memory: one at a time
    assert frames_at_once("balanced", **{**long, "cores": 4}) == 2
    assert frames_at_once("balanced", **{**long, "frames": 10}) == 4
    assert frames_at_once("balanced", **{**long, "frames": 3}) == 1  # 6 s of work: not worth the processes
    assert frames_at_once("fast", **{**long, "frames": 2}) == 1
    short = dict(cores=24, free_bytes=20 * GB, frame_pixels=400 * 400, frames=50)
    assert frames_at_once("fast", **short) == 1  # a short batch: starting processes would take longer
    assert frames_at_once("balanced", **{**long, "free_bytes": None}) == 4


# -- worker processes ----------------------------------------------------------------------------


@pytest.fixture
def frames_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "run_09"
    folder.mkdir()
    for index in range(6):
        save_tiff(folder / f"run_09_{index:03d}.tif", giwaxs_frame(seed=index) * (1.0 + 0.1 * index))
    return folder


def _jobs(folder: Path, out: Path, choices: BatchChoices) -> list[FrameJob]:
    paths = sorted(folder.glob("*.tif"))
    return [FrameJob(index=index, request=AnalysisRequest(path=path, mode="giwaxs", profile_name="synthetic"),
                     choices=choices, destination=out, live_key="radial") for index, path in enumerate(paths)]


def test_worker_processes_give_the_same_outcome_as_the_application(frames_folder: Path, tmp_path: Path) -> None:
    from src.gimap.features.analyze.application import AnalyzeFrame, ExportAnalysis
    from src.gimap.features.analyze.infrastructure.adapters import CsvCurveWriter, DetectorIoFrameSource
    from src.gimap.features.analyze.infrastructure.frame_workers import ProcessFrameWorkers
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository

    profile = InstrumentProfile("synthetic", GEOMETRY, None, SHAPE)
    choices = BatchChoices(tables=True, per_frame=True)
    here = FrameTools(AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([profile])),
                      ExportAnalysis(CsvCurveWriter()))
    expected = [run_frame_job(here, job) for job in _jobs(frames_folder, tmp_path / "here", choices)]

    workers = ProcessFrameWorkers(2, [profile])
    try:
        futures = [workers.submit(job) for job in _jobs(frames_folder, tmp_path / "workers", choices)]
        outcomes = [future.result(timeout=180) for future in futures]
    finally:
        workers.shutdown(wait=True)

    for mine, theirs in zip(expected, outcomes):
        assert (mine.index, mine.label) == (theirs.index, theirs.label)
        np.testing.assert_allclose(mine.live_row[1], theirs.live_row[1], rtol=1e-12, equal_nan=True)
        assert [curve.key for curve in mine.curves] == [curve.key for curve in theirs.curves]
    assert len(list((tmp_path / "workers" / "curves").glob("*.csv"))) == len(list((tmp_path / "here" / "curves").glob("*.csv")))


def test_workers_do_not_rerun_the_main_script() -> None:
    import types

    from src.gimap.features.analyze.infrastructure.frame_workers import WORKER_MAIN, _light_main

    fake = types.ModuleType("__main__")
    fake.__file__ = "main.py"
    fake.__spec__ = None
    saved = sys.modules["__main__"]
    sys.modules["__main__"] = fake
    try:
        with _light_main():
            assert fake.__spec__.name == WORKER_MAIN
        assert fake.__spec__ is None
    finally:
        sys.modules["__main__"] = saved


# -- the page: order, Stop, Pause, a worker lost, the live map ---------------------------------------


class ThreadWorkers:
    """Worker processes stood in for by threads, with a delay per frame (outcomes out of order)."""

    def __init__(self, page, count: int, delays=None, broken_at=None):
        self.page = page
        self.pool = ThreadPoolExecutor(count)
        self.delays = delays or {}
        self.broken_at = broken_at
        self.submitted: list[int] = []
        self.shut = False

    def submit(self, job) -> Future:
        self.submitted.append(job.index)
        if self.broken_at is not None and job.index == self.broken_at:
            future = Future()
            future.set_exception(BrokenProcessPool("a worker process ended"))
            return future

        def work():
            time.sleep(self.delays.get(job.index, 0.0))
            return self.page.view_model.run_job(job)

        return self.pool.submit(work)

    def shutdown(self, *, wait: bool = False) -> None:
        self.shut = True
        self.pool.shutdown(wait=wait, cancel_futures=True)


@pytest.fixture
def page(frames_folder: Path, monkeypatch):
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.bindings import batch_run
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from tests.test_analyze_workspace import _app, _context

    _app()
    profile = InstrumentProfile("synthetic", GEOMETRY, None, SHAPE)
    page = AnalyzePage(create_analyze_view_model(_context(InMemoryInstrumentProfileRepository([profile]))))
    page.resize(1400, 900)
    page.show()
    page.set_mode_choice("giwaxs")
    page.add_paths([str(frames_folder)])
    _settle(page, lambda: page.view_model.state.analysis is not None and page.view_model.state.analysis.reduction is not None)
    page.view_model.add_region(CutRegion("Ring q 1.100", (1.07, 1.13)))
    page.run_analysis()
    _settle(page, lambda: page.view_model.state.analysis.reduction.curve("region1_chi") is not None)
    monkeypatch.setattr(batch_run, "frames_at_once", lambda speed, **_kw: 1 if speed == "gentle" else 3)
    page.workers_made = []

    def workers(count, _profiles, **options):
        made = ThreadWorkers(page, count, **page.worker_options)
        page.workers_made.append(made)
        return made

    page.worker_options = {}
    page.view_model.frame_workers = workers
    yield page
    page.tasks.wait(30)
    page.dispose()
    page.close()


def _rows(path: Path) -> list[list[str]]:
    with path.open() as stream:
        return [row for row in csv.reader(line for line in stream if not line.startswith("#"))]


def _run(page, out: Path, speed: str, **choices) -> None:
    page.run_batch(out, BatchChoices(tables=True, speed=speed, fit="peaks", fit_start="previous", **choices), stem="run_09")
    _settle(page, lambda: not page.batch_running(), timeout_s=120)


def test_frames_arriving_out_of_order_are_folded_in_frame_order(page, tmp_path: Path) -> None:
    _run(page, tmp_path / "gentle", "gentle")
    assert not page.workers_made  # gentle: one at a time, in the application
    page.worker_options = {"delays": {0: 0.4, 1: 0.0, 2: 0.2, 3: 0.0, 4: 0.3, 5: 0.0}}
    _run(page, tmp_path / "parallel", "balanced")

    (workers,) = page.workers_made
    assert sorted(workers.submitted) == list(range(6)) and workers.shut
    for name in ("run_09_radial_frames.csv", "run_09_region1_frames.csv", "run_09_peak_fits.csv"):
        gentle, parallel = _rows(tmp_path / "gentle" / name), _rows(tmp_path / "parallel" / name)
        assert gentle == parallel, name  # the same numbers, the frames in the same order
    assert "Exported 6/6" in page.status_text()
    panel = page.batch_panel
    assert panel.isVisible() and not panel.open_button.isHidden() and panel.stop_button.isHidden()
    assert page._series_map is not None and page._series_map.rows == 6  # the live map holds every frame
    assert page.current_right() == "series"


def test_stop_keeps_what_is_done_and_writes_the_tables(page, tmp_path: Path) -> None:
    page.worker_options = {"delays": {0: 0.05, **{index: 1.0 for index in range(1, 6)}}}
    page.run_batch(tmp_path / "stopped", BatchChoices(tables=True, speed="balanced"), stem="run_09")
    _settle(page, lambda: page._batch_run is not None and len(page._batch_run.frames) >= 1, timeout_s=60)
    page.cancel_batch()
    assert "Stopping" in page.batch_panel.now_label.text() or not page.batch_running()
    _settle(page, lambda: not page.batch_running(), timeout_s=60)
    done = len(page._batch_run.frames)
    assert 1 <= done < 6
    assert len(_rows(tmp_path / "stopped" / "run_09_radial_frames.csv")[0]) == done + 1  # x and the frames done
    assert f"Exported {done}/{done}" in page.status_text()
    assert page.batch_panel.title_label.text() == "Batch Export stopped"


def test_pause_starts_no_new_frames_until_resumed(page, tmp_path: Path) -> None:
    page.worker_options = {"delays": {index: 0.2 for index in range(6)}}
    page.run_batch(tmp_path / "paused", BatchChoices(tables=True, speed="balanced"), stem="run_09")
    page.batch_panel.pause_button.setChecked(True)
    (workers,) = page.workers_made
    started = list(workers.submitted)
    _settle(page, lambda: not page._batch_inflight, timeout_s=60)
    time.sleep(0.3)
    _settle(page)
    assert workers.submitted == started and page.batch_running()  # nothing new while paused
    assert "Paused" in page.batch_panel.now_label.text()
    page.batch_panel.pause_button.setChecked(False)
    _settle(page, lambda: not page.batch_running(), timeout_s=60)
    assert len(page._batch_run.frames) == 6


def test_a_lost_worker_process_leaves_the_rest_to_the_application(page, tmp_path: Path) -> None:
    page.worker_options = {"broken_at": 2}
    _run(page, tmp_path / "broken", "balanced")
    assert len(page._batch_run.frames) == 6 and not page._batch_run.failures  # frame 3 reduced again, here
    assert "one frame at a time" in page.batch_panel.detail_label.text() or page._batch_capacity == 1
    assert _rows(tmp_path / "broken" / "run_09_radial_frames.csv")[0][1:] == [
        f"run_09_{index:03d}.tif" for index in range(6)]


def test_the_dialog_offers_the_speeds_this_computer_allows(page, monkeypatch) -> None:
    from src.gimap.features.analyze.presentation.bindings import batch_run

    page.view_model.system_resources = lambda: (24, 20 * GB)
    options = page.batch_speed_options(403)
    assert [key for key, _text in options] == ["gentle", "balanced", "fast"] and "3 frames at once" in options[1][1]
    monkeypatch.setattr(batch_run, "frames_at_once", lambda *_a, **_kw: 1)
    assert [key for key, _text in page.batch_speed_options(6)] == ["gentle"]  # a short batch
