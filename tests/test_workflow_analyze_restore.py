"""Reopening a project keeps the detector source and its original frame, not an old list row."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.gimap.shared.detector_io import forget_layout
from tests.test_analyze_nexus_sources import H, W, _master, _stack
from tests.test_analyze_workspace import _context, _done, _page
from tests.test_assistant_calibration import save_tiff


@pytest.fixture
def page_factory():
    forget_layout()
    pages = []

    def make():
        page = _page(_context())
        pages.append(page)
        return page

    yield make
    for page in reversed(pages):
        page.dispose()
        page.close()
    forget_layout()


def _saved(page):
    # Use exactly the JSON-compatible representation that Save Project writes.
    return json.loads(json.dumps(page.project_state()))


def _assert_frame(page, path: Path, frame: int, value: int):
    _done(page)
    state = page.view_model.state
    analysis = state.analysis
    assert page.view_model.current_path == path
    assert state.frame_index == frame
    assert analysis is not None and analysis.path == path and analysis.frame_index == frame
    np.testing.assert_array_equal(analysis.raw_data, np.full((H, W), value))
    np.testing.assert_array_equal(analysis.data, np.full((H, W), value))


def test_project_reopens_the_same_source_when_import_order_is_sorted(page_factory, tmp_path):
    first = _stack(tmp_path / "a_series.nxs", 10)
    second = _stack(tmp_path / "b_series.nxs", 10)
    page = page_factory()
    # Separate imports retain their arrival order; reopening expands all saved paths together.
    page.add_paths([second])
    _done(page)
    page.add_paths([first])
    _done(page)
    assert page.view_model.state.files == [second, first]
    assert page.show_frame(0, 7)
    _assert_frame(page, second, 7, 7)
    saved = _saved(page)

    reopened = page_factory()
    assert reopened.apply_project_state(saved) == []
    _assert_frame(reopened, second, 7, 7)


def test_project_preserves_the_observations_summed_across_single_frame_files(page_factory, tmp_path):
    first, second, third = [save_tiff(tmp_path / f"{name}.tif", np.full((H, W), value))
                            for name, value in (("a", 1), ("b", 10), ("c", 100))]
    page = page_factory()
    page.add_paths([second])
    _done(page)
    page.add_paths([first, third])
    _done(page)
    assert page.view_model.state.files == [second, first, third]
    page.view_model.set_sum_count(2)
    assert page.show_frame(0, 0)
    _assert_frame(page, second, 0, 11)
    assert page.view_model.state.analysis.summed_frames == ((first, 0),)
    saved = _saved(page)

    reopened = page_factory()
    assert reopened.apply_project_state(saved) == []
    # Keeping B selected is insufficient: sorting its neighbours would silently sum B+C (110).
    _assert_frame(reopened, second, 0, 11)
    assert reopened.view_model.state.analysis.summed_frames == ((first, 0),)
    assert reopened.view_model.state.files == [second, first, third]


def test_missing_earlier_source_does_not_shift_the_selected_observation(page_factory, tmp_path):
    first, second, third = [_stack(tmp_path / f"{name}_series.nxs", 10) for name in "abc"]
    page = page_factory()
    page.add_paths([first, second, third])
    _done(page)
    page.view_model.set_import_selection(second, [1, 4, 7])
    assert page.show_frame(1, 7)
    _assert_frame(page, second, 7, 7)
    saved = _saved(page)
    first.unlink()
    forget_layout()

    reopened = page_factory()
    assert reopened.apply_project_state(saved)
    _assert_frame(reopened, second, 7, 7)
    assert list(reopened.view_model.selected_frame_indices(second)) == [1, 4, 7]


def test_missing_selected_source_falls_back_to_the_first_imported_frame(page_factory, tmp_path):
    first, second, third = [_stack(tmp_path / f"{name}_series.nxs", 10) for name in "abc"]
    page = page_factory()
    page.add_paths([first, second, third])
    _done(page)
    page.view_model.set_import_selection(first, [2, 6])
    page.view_model.set_import_selection(second, [1, 7])
    page.view_model.set_import_selection(third, [3, 7])
    assert page.show_frame(1, 7)
    _assert_frame(page, second, 7, 7)
    saved = _saved(page)
    second.unlink()
    forget_layout()

    reopened = page_factory()
    assert reopened.apply_project_state(saved)
    _assert_frame(reopened, first, 2, 2)
    assert list(reopened.view_model.selected_frame_indices(first)) == [2, 6]


def test_reappearing_master_keeps_the_chunk_observation_and_import_scope(page_factory, tmp_path):
    folder = tmp_path / "detector"
    master = _master(folder, "scan", chunks=(3, 4, 2))
    held = master.rename(tmp_path / "held_master.h5")
    chunk = folder / "scan_data_000002.h5"
    forget_layout()
    page = page_factory()
    page.add_paths([chunk])
    _done(page)
    assert page.view_model.state.files == [chunk]
    page.view_model.set_import_selection(chunk, [1, 3])
    assert page.show_frame(0, 1)
    _assert_frame(page, chunk, 1, 2)
    saved = _saved(page)
    held.rename(master)
    forget_layout()

    reopened = page_factory()
    assert reopened.apply_project_state(saved) == []
    # The first chunk has three frames: local frame 1 becomes master frame 4.
    _assert_frame(reopened, master, 4, 2)
    assert list(reopened.view_model.selected_frame_indices(master)) == [4, 6]


@pytest.mark.parametrize("first_selection, expected", [
    (None, [0, 1, 2, 4, 6]),
    ([0, 2], [0, 2, 4, 6]),
], ids=["all-and-subset", "two-explicit-subsets"])
def test_reappearing_master_unions_every_saved_chunk_scope(page_factory, tmp_path, first_selection, expected):
    folder = tmp_path / "detector"
    master = _master(folder, "scan", chunks=(3, 4, 2))
    held = master.rename(tmp_path / "held_master.h5")
    chunks = [folder / f"scan_data_{number:06d}.h5" for number in (1, 2)]
    forget_layout()
    page = page_factory()
    page.add_paths(chunks)
    _done(page)
    assert page.view_model.state.files == chunks
    page.view_model.set_import_selection(chunks[0], first_selection)
    page.view_model.set_import_selection(chunks[1], [1, 3])
    assert page.show_frame(1, 1)
    _assert_frame(page, chunks[1], 1, 2)
    saved = _saved(page)
    if first_selection is None:
        assert str(chunks[0]) not in saved["import_selections"]  # All is deliberately implicit.
    held.rename(master)
    forget_layout()

    reopened = page_factory()
    assert reopened.apply_project_state(saved) == []
    _assert_frame(reopened, master, 4, 2)
    assert list(reopened.view_model.selected_frame_indices(master)) == expected
    # A chunk's All must not import the other, previously unopened chunks of the master.
    assert not any(index in reopened.view_model.selected_frame_indices(master) for index in (7, 8))


def test_direct_master_selection_takes_priority_over_retained_chunk_selection(page_factory, tmp_path):
    folder = tmp_path / "detector"
    master = _master(folder, "scan", chunks=(3, 4, 2))
    page = page_factory()
    page.add_paths([master])
    _done(page)
    page.view_model.set_import_selection(master, [7, 8])
    assert page.show_frame(0, 7)
    _assert_frame(page, master, 7, 3)
    saved = _saved(page)
    # A project can retain an alias record alongside an explicit canonical-source choice.
    chunk = folder / "scan_data_000002.h5"
    saved["import_selections"] = {str(chunk): {"indices": [1]}, **saved["import_selections"]}

    reopened = page_factory()
    assert reopened.apply_project_state(saved) == []
    _assert_frame(reopened, master, 7, 3)
    assert list(reopened.view_model.selected_frame_indices(master)) == [7, 8]
