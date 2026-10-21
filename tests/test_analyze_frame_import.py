"""Import registers lazy frame references; pixels are read only for the frame shown.

The million-frame NeXus below is 12 kB on disk, with only three allocated chunks.
Its logical 3.4 GB stack must never be converted into an array during import.
"""

from __future__ import annotations

import threading
import time
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, QTimer
from PyQt5.QtTest import QTest

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.presentation.frame_selection import FrameSelection
from src.gimap.shared.detector_io import forget_layout
from tests.test_analyze_workspace import _context, _done, _page

DATASET = "/entry/instrument/detector/data"
SHAPE = (24, 36)
MILLION = 1_000_000


@pytest.fixture(autouse=True)
def _clean_layout_and_language():
    forget_layout()
    apply_language("en")
    yield
    forget_layout()
    apply_language("en")


def _sparse(path: Path, count=MILLION) -> Path:
    with h5py.File(path, "w") as handle:
        data = handle.create_dataset(DATASET, shape=(count,) + SHAPE, dtype="i4", chunks=(1,) + SHAPE,
                                     compression="gzip", fillvalue=0)
        for index in (value for value in (0, 499_999, 999_999) if value < count):
            data[index] = np.full(SHAPE, index + 1, dtype=np.int32)
        exposure = handle.create_dataset("/entry/instrument/detector/count_time", data=0.25)
        exposure.attrs["units"] = "s"
    assert path.stat().st_size < 100_000
    return path.resolve()


def _pixel_reads(monkeypatch) -> list[int]:
    """Reject either form of an accidental full-stack read before it allocates memory."""
    reads: list[int] = []
    getitem, asarray = h5py.Dataset.__getitem__, h5py.Dataset.__array__

    def recorded(dataset, selection):
        if dataset.name == DATASET:
            first = selection[0] if isinstance(selection, tuple) and selection else selection
            assert isinstance(first, (int, np.integer)), f"Full detector stack read: {selection!r}"
            reads.append(int(first))
        return getitem(dataset, selection)

    def whole_array(dataset, *args, **kwargs):
        assert dataset.name != DATASET, "Detector stack converted into an array"
        return asarray(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recorded)
    monkeypatch.setattr(h5py.Dataset, "__array__", whole_array)
    return reads


def _until(condition, seconds=10):
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        QTest.qWait(10)
    assert condition(), "Import did not reach the expected state"


def _close(page):
    page.cancel_frame_import()
    page.tasks.wait(30)
    page.dispose()
    page.close()


def test_a_million_frames_are_listed_lazily_and_only_the_chosen_frame_is_read(tmp_path, monkeypatch):
    path = _sparse(tmp_path / "million.nxs")
    reads = _pixel_reads(monkeypatch)
    page = _page(_context())
    try:
        page.import_paths([path], interactive=False)
        _done(page)
        model = page.frame_list.model()
        assert model.rowCount() == MILLION
        assert model.ref(0) == (path, 0)
        assert model.ref(499_999) == (path, 499_999)
        assert model.ref(MILLION - 1) == (path, MILLION - 1)
        assert reads == [0]
        requests = page.view_model.batch_requests()
        assert not isinstance(requests, list) and len(requests) == MILLION
        assert requests[-1].frame_index == MILLION - 1
        assert reads == [0]  # even enumerating the endpoint does not read its pixels
        page.frame_list.setCurrentIndex(model.index(499_999, 0))
        _done(page)
        assert reads == [0, 499_999]
        analysis = page.view_model.state.analysis
        assert analysis.frame_index == 499_999 and np.all(analysis.data == 500_000)
    finally:
        _close(page)


def test_only_selected_frames_appear_and_navigation_never_reads_an_unimported_frame(tmp_path, monkeypatch):
    path = _sparse(tmp_path / "selected.nxs")
    reads = _pixel_reads(monkeypatch)
    page = _page(_context())
    selected = FrameSelection.explicit((999_999, 0, 499_999, 499_999))
    try:
        page.import_paths([path], interactive=False, selections={str(path): selected})
        _done(page)
        model = page.frame_list.model()
        assert model.rowCount() == 3
        assert [model.ref(row) for row in range(3)] == [(path, index) for index in (0, 499_999, 999_999)]
        assert [request.frame_index for request in page.view_model.batch_requests()] == [0, 499_999, 999_999]
        assert reads == [0]
        page._step_frame(1)
        _done(page)
        assert page.view_model.state.frame_index == 499_999 and reads == [0, 499_999]
        page._step_frame(1)
        _done(page)
        assert page.view_model.state.frame_index == 999_999 and reads == [0, 499_999, 999_999]
        page.frame_spin.setValue(2)  # this frame was never imported
        _done(page)
        assert page.view_model.state.frame_index == 999_999 and reads == [0, 499_999, 999_999]
        assert "import" in page.status_text().lower() or "choose" in page.status_text().lower()
        assert page.view_model.imported_frame_count() == 3
    finally:
        _close(page)


def test_a_selected_range_preserves_its_source_indices_without_loading_the_other_frames(tmp_path, monkeypatch):
    path = _sparse(tmp_path / "range.nxs")
    reads = _pixel_reads(monkeypatch)
    page = _page(_context())
    try:
        page.import_paths([path], interactive=False,
                          selections={str(path): FrameSelection.range(499_997, 500_001, every=2)})
        _done(page)
        model = page.frame_list.model()
        assert [model.ref(row)[1] for row in range(model.rowCount())] == [499_997, 499_999, 500_001]
        assert [request.frame_index for request in page.view_model.batch_requests()] == [499_997, 499_999, 500_001]
        assert reads == [499_997]
        numbers = page.view_model.frame_numbers()
        assert numbers[(str(path).casefold(), 499_999)] == 500_000
    finally:
        _close(page)


def test_a_single_image_and_selected_nexus_frames_share_one_unambiguous_list(tmp_path):
    from tests.test_assistant_calibration import save_tiff

    path = _sparse(tmp_path / "stack.nxs", count=3)
    image = save_tiff(tmp_path / "single.tif", np.full(SHAPE, 17, dtype=np.float32)).resolve()
    page = _page(_context())
    try:
        page.import_paths([image, path], interactive=False,
                          selections={str(path): FrameSelection.explicit((0, 2))})
        _done(page)
        model = page.frame_list.model()
        refs = [model.ref(row) for row in range(model.rowCount())]
        assert len(refs) == 3 and set(refs) == {(image, 0), (path, 0), (path, 2)}
        assert {(request.path, request.frame_index) for request in page.view_model.batch_requests()} == set(refs)
        row = refs.index((image, 0))
        page.frame_list.setCurrentIndex(model.index(row, 0))
        _done(page)
        assert page.view_model.state.analysis.path == image
        np.testing.assert_array_equal(page.view_model.state.analysis.data, np.full(SHAPE, 17))
    finally:
        _close(page)


def test_module_sets_and_master_chunks_are_listed_once_per_complete_detector_frame(tmp_path):
    from tests.test_analyze_nexus_sources import _master, _module

    first = _module(tmp_path / "modules" / "run_m01.nxs", 1, frames=3).resolve()
    second = _module(tmp_path / "modules" / "run_m02.nxs", 2, frames=3).resolve()
    master = _master(tmp_path / "master", "run", chunks=(2, 1)).resolve()
    paths = [second, first, master, *sorted(master.parent.glob("*_data_*.h5"))]
    page = _page(_context())
    try:
        page.import_paths(paths, interactive=False)
        _done(page)
        model = page.frame_list.model()
        refs = [model.ref(row) for row in range(model.rowCount())]
        assert len(refs) == 6 and len(set(refs)) == 6
        assert set(refs) == {(path, index) for path in (first, master) for index in range(3)}
        assert len(page.view_model.state.files) == 2
        assert {(request.path, request.frame_index) for request in page.view_model.batch_requests()} == set(refs)
        assert page.view_model.describe(first).kind == "module_set"
        assert page.view_model.describe(master).kind == "chunked_master"
    finally:
        _close(page)


def test_batch_processing_uses_only_imported_frames_and_keeps_original_frame_numbers(tmp_path, monkeypatch):
    from src.gimap.features.analyze.application import BatchChoices
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from src.gimap.shared.detector_io import load_detector_image
    from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile

    path = _sparse(tmp_path / "processing.nxs")
    geometry = DetectorGeometry(172e-6, 172e-6, 0.4, 18.0, 40.0, 1.0, 0.4)
    profiles = InMemoryInstrumentProfileRepository([InstrumentProfile("import stack", geometry, None, SHAPE)])
    page = _page(_context(profiles))
    reads = _pixel_reads(monkeypatch)
    selected = (0, 499_999, 999_999)
    try:
        page.set_mode_choice("gisaxs")
        page.import_paths([path], interactive=False, selections={str(path): FrameSelection.explicit(selected)})
        _done(page)
        page.run_batch(tmp_path / "out", BatchChoices(tables=False, per_frame=False, frames=True,
                                                     frame_format="hdf5", speed="gentle"), stem="selected")
        _until(lambda: not page.batch_running(), seconds=60)
        _done(page)
        record = json.loads((tmp_path / "out" / "selected_batch.json").read_text(encoding="utf-8"))
        assert record["requested"] == 3 and record["summary"]["exported"] == 3
        assert record["failed"] == [] and record["frame_numbers"] == [1, 500_000, 1_000_000]
        assert set(reads) == set(selected) and reads.count(499_999) == reads.count(999_999) == 1
        frames = sorted((tmp_path / "out" / "frames").glob("*.h5"))
        assert len(frames) == 3
        # Converted exports hold whole frames, not a sampled preview of the detector pixels.
        assert [int(load_detector_image(frame).data[0, 0]) for frame in frames] == [1, 500_000, 1_000_000]
    finally:
        _close(page)


def test_reimporting_another_subset_never_presents_the_old_map_as_the_current_import(tmp_path):
    from tests.test_analyze_nexus_sources import _ring_stack
    from tests.test_review_analyze_run_and_clear import _profiles
    from tests.test_series_cut import _build

    path = _ring_stack(tmp_path / "rings.nxs", 5).resolve()
    page = _page(_context(_profiles()))
    try:
        page.set_mode_choice("giwaxs")
        page.import_paths([path], interactive=False, selections={str(path): FrameSelection.explicit((0, 2, 4))})
        _done(page)
        _build(page, 3)
        assert page._series_map.numbers == (1, 3, 5)
        page.import_paths([path], interactive=False, selections={str(path): FrameSelection.explicit((1, 3))})
        _done(page)
        assert page.frame_list.model().rowCount() == 2
        if page._series_map is not None and page._series_map.numbers != (2, 4):
            assert not page.series_stale_label.isHidden(), "Old map shown after import scope changed without a notice"
    finally:
        _close(page)


@pytest.mark.parametrize("action", ["cancel", "clear"])
def test_a_slow_metadata_scan_keeps_the_window_responsive_and_late_results_are_discarded(tmp_path, monkeypatch, action):
    path = _sparse(tmp_path / "waiting.nxs")
    reads = _pixel_reads(monkeypatch)
    page = _page(_context())
    entered, release = threading.Event(), threading.Event()
    ticks: list[int] = []
    timer = QTimer(page)
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(len(ticks)))
    expand = page.view_model._frames.expand

    def held_scan(paths):
        entered.set()
        assert release.wait(15), "Test did not release the slow metadata scan"
        return expand(paths)

    monkeypatch.setattr(page.view_model._frames, "expand", held_scan)
    try:
        timer.start()
        page.import_paths([path], interactive=False)
        _until(entered.is_set)
        assert page.tasks.is_busy() and not page.import_cancel_button.isHidden()
        QTest.qWait(100)
        assert len(ticks) >= 3 and reads == []
        if action == "cancel":
            page.cancel_frame_import()
        else:
            page.clear_files()
        release.set()
        _done(page)
        assert page.frame_list.model().rowCount() == 0
        assert page.file_list.count() == 0 and page.view_model.state.files == []
        assert reads == [] and page.view_model.state.analysis is None
        assert page.import_cancel_button.isHidden()
    finally:
        timer.stop()
        release.set()
        _close(page)


def test_disposing_the_page_disconnects_frame_list_updates_before_its_widgets_are_destroyed(tmp_path, monkeypatch):
    page = _page(_context())
    errors = []
    try:
        page.import_paths([_sparse(tmp_path / "closing.nxs", count=3)], interactive=False)
        _done(page)
        monkeypatch.setattr(sys, "excepthook", lambda kind, error, trace: errors.append(error))
        page.dispose()
        page.file_list.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QTest.qWait(20)
        assert errors == [], "Frame-list reset called back after the underlying widget was destroyed"
    finally:
        page.close()
