"""Source selections and huge lazy sequences use metadata, preserve indices, and avoid duplicate work."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.gimap.features.analyze.presentation.frame_selection import FrameSelection
from tests.test_analyze_frames_and_center import _view_model


class MetadataFrames:
    def __init__(self, counts):
        self.counts, self.loads = dict(counts), []

    def frame_count(self, path):
        return self.counts[Path(path).name]

    def describe(self, path):
        return None

    def expand(self, paths):
        return list(map(Path, paths))

    def settled_size(self, path):
        return 1

    def load(self, path, frame_index=0):
        self.loads.append((path, frame_index))
        raise AssertionError("Selection and request construction must not read pixels.")


def _model(tmp_path, **counts):
    source = MetadataFrames(counts)
    model = _view_model(source)
    paths = [tmp_path / name for name in counts]
    model.add_paths(paths)
    return model, source, paths


def test_a_million_frame_source_is_a_range_and_lazy_request_sequence(tmp_path):
    model, source, (path,) = _model(tmp_path, **{"huge.nxs": 1_000_000})
    assert model.state.files == [path]
    assert isinstance(model.selected_frame_indices(path), range)
    frames = model.all_frames()
    assert len(frames) == model.listed_frames() == 1_000_000
    assert frames[0] == (path, 0) and frames[-1] == (path, 999_999)
    requests = model.batch_requests()
    assert not isinstance(requests, list)
    assert len(requests) == 1_000_000
    assert len(requests[::100_000]) == 10
    assert requests[::100_000][-1].frame_index == 900_000
    assert requests[-1].frame_index == 999_999
    assert len(model.frame_numbers().sources) == 1
    assert model.frame_numbers()[(str(path).casefold(), 999_999)] == 1_000_000
    assert source.loads == []


def test_imported_indices_apply_to_batch_and_preserve_physical_numbering(tmp_path):
    model, source, (first, second) = _model(tmp_path, **{"first.nxs": 10, "next.nxs": 6})
    model.set_import_selection(first, FrameSelection.explicit([8, 1, 1, 5]))
    model.set_import_selection(second, FrameSelection.range(2, 5, 2))
    refs = [(first, 1), (first, 5), (first, 8), (second, 2), (second, 4)]
    assert list(model.iter_imported_frames()) == list(model.all_frames()) == refs
    assert model.listed_frames() == model.imported_frame_count() == 5
    assert model.total_frame_count() == 16
    assert [(item.path, item.frame_index) for item in model.batch_requests()] == refs
    assert [model.frame_numbers()[(str(path).casefold(), index)] for path, index in refs] == [2, 6, 9, 13, 15]
    with model.only_frames(6, 13):
        assert [(item.path, item.frame_index) for item in model.batch_requests()] == refs[1:4]
    assert source.loads == []


def test_sparse_selections_do_not_sum_across_unimported_frames(tmp_path):
    model, _source, (path,) = _model(tmp_path, **{"scan.nxs": 20})
    model.set_sum_count(3)
    model.set_import_selection(path, [0, 1, 2, 4, 6, 7, 8, 9])
    requests = model.batch_requests()
    assert [(item.frame_index, tuple(index for _path, index in item.summed_frames)) for item in requests] == [
        (0, (1, 2)), (4, ()), (6, (7, 8)), (9, ())]
    assert [item.frames for item in model.batch_groups()] == [3, 1, 3, 1]
    assert model.batch_tail() == ((path, 9),)
    model.select(0)
    model.set_frame(1)
    assert model.summed_frames_for(path, 1) == ((path, 2),)
    model.set_import_selection(path, FrameSelection.range(0, 18, 2))
    assert len(model.batch_requests()) == 10
    assert all(not request.summed_frames for request in model.batch_requests())
    assert model.batch_group_summary()["short_count"] == 10
    assert model.batch_tail() == ()


def test_sum_across_single_frame_files_keeps_adjacent_original_files_only(tmp_path):
    model, _source, paths = _model(tmp_path, **{f"frame{index}.tif": 1 for index in range(5)})
    model.set_sum_count(3)
    assert [item.frames for item in model.batch_groups()] == [3, 2]
    model.set_import_selection(paths[2], [])
    requests = model.batch_requests()
    assert [(item.path, item.summed_frames) for item in requests] == [
        (paths[0], ((paths[1], 0),)), (paths[3], ((paths[4], 0),))]
    assert model.summed_frames_for(paths[0]) == ((paths[1], 0),)


def test_lazy_requests_freeze_scientific_settings(tmp_path):
    model, _source, (_path,) = _model(tmp_path, **{"scan.nxs": 10})
    model.set_incidence(0.3)
    model.set_horizontal_band(10, 12)
    requests = model.batch_requests()
    model.set_incidence(0.9)
    model.set_horizontal_band(20, 22)
    assert requests[-1].incidence_deg == 0.3
    assert requests[-1].gisaxs.horizontal_row == 11


def test_all_grows_but_closed_import_range_does_not(tmp_path):
    model, source, (path,) = _model(tmp_path, **{"growing.nxs": 12})
    source.counts[path.name] = 18
    model.frame_count(path)
    assert model.listed_frames() == 18
    model.set_import_selection(path, FrameSelection.range(3, 9, 2))
    source.counts[path.name] = 25
    model.frame_count(path)
    assert list(model.selected_frame_indices(path)) == [3, 5, 7, 9]
    assert model.listed_frames() == 4
    assert model._watch_groups([(path, index) for index in [7, 8, 9, 10, 20]])[0].frame_index == 7
    assert [request.frame_index for request in model._watch_groups([(path, 10), (path, 20)])] == []


def test_selection_round_trip_rejects_invalid_indices_and_remove_clear_forget_it(tmp_path):
    model, _source, paths = _model(tmp_path, **{"a.nxs": 12, "b.nxs": 6})
    model.set_import_selection(paths[0], FrameSelection.range(2, 10, 3))
    model.set_import_selection(paths[1], [4, 0, 4])
    record = model.export_import_selections()
    model.apply_import_selections({})
    assert model.listed_frames() == 18
    assert model.apply_import_selections(record) == []
    assert model.listed_frames() == 5
    with pytest.raises(ValueError):
        model.set_import_selection(paths[0], [12])
    with pytest.raises(ValueError):
        model.set_import_selection(paths[0], FrameSelection.range(-1, 2))
    assert len(model.apply_import_selections({str(paths[0]): {"first": 0, "last": 30}})) == 1
    model.set_import_selection(paths[0], [3])
    model.remove_file(0)
    assert str(paths[0]).casefold() not in model._import_selections
    model.clear_files()
    assert model._import_selections == {}


def test_worker_sources_commit_does_not_expand_twice(tmp_path, monkeypatch):
    model, source, paths = _model(tmp_path, **{"a.nxs": 3})
    model.clear_files()
    monkeypatch.setattr(source, "expand", lambda _paths: pytest.fail("already inspected sources must not expand again"))
    assert model.add_import_sources(paths) == paths
    assert model.add_import_sources(paths) == [] and model.already_listed == [0]


def test_chunk_selections_are_offset_and_unioned_when_master_arrives(tmp_path):
    from src.gimap.shared.detector_io import forget_layout
    from tests.test_analyze_nexus_sources import _master, _model as real_model

    master = _master(tmp_path, "scan", chunks=(3, 4, 2))
    held = master.rename(tmp_path.parent / (tmp_path.name + "_held_master.h5"))
    chunks = sorted(tmp_path.glob("scan_data*.h5"))
    forget_layout()
    model = real_model()
    model.add_paths(chunks[:2])
    assert model.state.files == chunks[:2]
    model.set_import_selection(chunks[0], [1])
    model.set_import_selection(chunks[1], FrameSelection.range(0, 2, 2))
    held.rename(master)
    forget_layout()
    model.add_paths([master])
    assert model.state.files == [master]
    assert list(model.selected_frame_indices(master)) == [1, 3, 5]
    assert [item.frame_index for item in model.batch_requests()] == [1, 3, 5]
    assert [model.frame_numbers()[(str(master).casefold(), item)] for item in [1, 3, 5]] == [2, 4, 6]
    assert set(model._import_selections) == {str(master).casefold()}


def test_module_canonical_path_change_preserves_selection(tmp_path):
    from src.gimap.shared.detector_io import forget_layout
    from tests.test_analyze_nexus_sources import _module, _model as real_model

    second = _module(tmp_path / "scan_m02.nxs", 2, frames=8)
    model = real_model()
    model.add_paths([second])
    model.set_import_selection(second, FrameSelection.range(2, 6, 2))
    first = _module(tmp_path / "scan_m01.nxs", 1, frames=8)
    forget_layout()
    model.add_paths([first])
    assert model.state.files == [first]
    assert list(model.selected_frame_indices(first)) == [2, 4, 6]
    assert len(model.batch_requests()) == 3


@pytest.mark.parametrize("ranges", [
    (range(0, 100, 2), range(1, 51, 2), range(2, 99, 3)),
    (range(13, 21, 3), range(18, 43, 4), range(8, 42, 3), range(16, 32)),
])
def test_overlapping_selections_union_once_in_increasing_physical_order(ranges):
    selected = FrameSelection(ranges)
    assert list(selected.indices(100)) == sorted(set().union(*ranges))
    huge = FrameSelection((range(1_000_000), range(0, 1_000_000, 2), range(500_000, 500_001)))
    assert huge.ranges == (range(1_000_000),)


def test_consecutive_sum_can_cross_two_compact_selection_ranges(tmp_path):
    model, _source, (path,) = _model(tmp_path, **{"scan.nxs": 20})
    model.set_import_selection(path, FrameSelection((range(0, 10, 2), range(9, 13))))
    model.set_sum_count(3)
    assert [(request.frame_index, tuple(index for _path, index in request.summed_frames))
            for request in model.batch_requests()] == [
                (0, ()), (2, ()), (4, ()), (6, ()), (8, (9, 10)), (11, (12,))]
    assert model.batch_group_index(path, 10) == 4
    assert model.batch_group_index(path, 12) == 5
    assert model.batch_group_index(path, 13) is None
    with model.only_frames(10, 13):
        assert model.batch_group_index(path, 8) is None
        assert model.batch_group_index(path, 9) == 0
        assert model.batch_group_index(path, 12) == 1
