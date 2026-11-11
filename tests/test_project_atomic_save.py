"""A failed save must leave both the last project and its Compare arrays usable."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from src.gimap.app import project
from src.gimap.features.compare.presentation.page import ComparePage


def _components(value=1.0):
    """Use Compare's real serializer with synthetic maps, without opening a window."""
    compare = SimpleNamespace(
        _q_range=None, _reading=0,
        shape_check=SimpleNamespace(isChecked=lambda: True),
        end_spin=SimpleNamespace(value=lambda: 3),
        series=[SimpleNamespace(
            paths=(), name="synthetic map", x=np.array([0.1, 0.2]),
            image=np.full((3, 2), value), labels=("frame 1", "frame 2", "frame 3"),
            x_label="q (Å⁻¹)", source="synthetic",
        )],
    )
    compare.project_state = MethodType(ComparePage.project_state, compare)
    empty = SimpleNamespace(project_state=lambda: {})
    return SimpleNamespace(
        analyze_page=empty,
        fitting_workspace=SimpleNamespace(fit_page=empty, series_page=empty),
        compare_page=compare,
    )


def _snapshot(path):
    record = project.read(path)
    sidecar = path.parent / record["compare"]["arrays_file"]
    return path.read_bytes(), sidecar, sidecar.read_bytes(), set(path.parent.iterdir())


def _unchanged(path, snapshot):
    content, sidecar, arrays, files = snapshot
    assert path.read_bytes() == content, "the previous project JSON was damaged"
    assert sidecar.read_bytes() == arrays, "the previous project now references different arrays"
    assert set(path.parent.iterdir()) == files, "a failed save left partial output behind"
    with np.load(sidecar) as stored:
        np.testing.assert_array_equal(stored["image_0"], np.ones((3, 2)))


def test_serialization_failure_keeps_previous_project_and_arrays(tmp_path):
    path = project.save(_components(), tmp_path / "sample.gimap")
    before = _snapshot(path)
    components = _components(9.0)
    components.sputter_page = SimpleNamespace(project_state=lambda: {"invalid": object()})
    with pytest.raises((TypeError, ValueError)):
        project.save(components, path)
    _unchanged(path, before)


def test_sidecar_write_failure_keeps_previous_project_and_arrays(tmp_path, monkeypatch):
    path = project.save(_components(), tmp_path / "sample.gimap")
    before = _snapshot(path)

    def interrupted(sidecar, **_arrays):
        Path(sidecar).write_bytes(b"partial array archive")
        raise OSError("synthetic sidecar write failure")

    monkeypatch.setattr(np, "savez_compressed", interrupted)
    with pytest.raises(OSError, match="sidecar write failure"):
        project.save(_components(9.0), path)
    _unchanged(path, before)


def test_json_write_failure_keeps_previous_project_and_arrays(tmp_path, monkeypatch):
    path = project.save(_components(), tmp_path / "sample.gimap")
    before = _snapshot(path)
    original_open = Path.open

    class InterruptedWrite:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.stream.close()

        def write(self, content):
            self.stream.write(content[:20])
            self.stream.flush()
            raise OSError("synthetic JSON write failure")

    def interrupted(file, mode="r", *args, **kwargs):
        stream = original_open(file, mode, *args, **kwargs)
        return InterruptedWrite(stream) if "w" in mode and "b" not in mode else stream

    monkeypatch.setattr(Path, "open", interrupted)
    with pytest.raises(OSError, match="JSON write failure"):
        project.save(_components(9.0), path)
    _unchanged(path, before)


def test_json_replace_failure_keeps_previous_project_and_arrays(tmp_path, monkeypatch):
    path = project.save(_components(), tmp_path / "sample.gimap")
    before = _snapshot(path)

    def refused(*_args, **_kwargs):
        raise PermissionError("synthetic JSON replace failure")

    monkeypatch.setattr(os, "replace", refused)
    with pytest.raises(OSError, match="JSON replace failure"):
        project.save(_components(9.0), path)
    _unchanged(path, before)


def test_repeated_save_commits_new_arrays_and_removes_previous_generation(tmp_path, monkeypatch):
    path = project.save(_components(), tmp_path / "sample.gimap")
    old_content, old_sidecar, old_arrays, _files = _snapshot(path)
    replace = os.replace
    commits = []

    def checked_replace(source, destination):
        assert Path(destination) == path
        assert path.read_bytes() == old_content
        assert old_sidecar.read_bytes() == old_arrays
        new_record = json.loads(Path(source).read_text(encoding="utf-8"))
        new_sidecar = path.parent / new_record["compare"]["arrays_file"]
        assert new_sidecar != old_sidecar
        with np.load(new_sidecar) as stored:
            np.testing.assert_array_equal(stored["image_0"], np.full((3, 2), 9.0))
        commits.append(new_sidecar)
        replace(source, destination)

    monkeypatch.setattr(os, "replace", checked_replace)
    assert project.save(_components(9.0), path, "compare") == path
    assert len(commits) == 1
    assert not old_sidecar.exists()
    assert set(tmp_path.iterdir()) == {path, commits[0]}
    assert project.read(path)["page"] == "compare"


@pytest.mark.parametrize("failed_flush", [1, 2], ids=["sidecar", "json"])
def test_flush_failure_keeps_previous_project_and_arrays(tmp_path, monkeypatch, failed_flush):
    path = project.save(_components(), tmp_path / "sample.gimap")
    before = _snapshot(path)
    flush = os.fsync
    calls = 0

    def interrupted(descriptor):
        nonlocal calls
        calls += 1
        if calls == failed_flush:
            raise OSError("synthetic flush failure")
        flush(descriptor)

    monkeypatch.setattr(os, "fsync", interrupted)
    with pytest.raises(OSError, match="flush failure"):
        project.save(_components(9.0), path)
    _unchanged(path, before)


def test_first_save_failure_leaves_no_project_or_partial_arrays(tmp_path, monkeypatch):
    def refused(*_args):
        raise PermissionError("synthetic replace failure")

    monkeypatch.setattr(os, "replace", refused)
    with pytest.raises(OSError, match="replace failure"):
        project.save(_components(), tmp_path / "first.gimap")
    assert list(tmp_path.iterdir()) == []


def test_legacy_project_remains_readable_and_is_upgraded_on_save(tmp_path):
    path = tmp_path / "sample.gimap"
    path.write_text(json.dumps(project.collect(_components(), path=path)), encoding="utf-8")
    legacy = tmp_path / "sample.compare.npz"
    assert project.read(path)["compare"]["arrays_file"] == legacy.name
    assert project.missing(project.read(path), path) == []
    project.save(_components(9.0), path)
    current = tmp_path / project.read(path)["compare"]["arrays_file"]
    assert current != legacy and current.exists() and not legacy.exists()
    with np.load(current) as stored:
        np.testing.assert_array_equal(stored["image_0"], np.full((3, 2), 9.0))


@pytest.mark.parametrize("external_reference", [False, True], ids=["save-as", "copied-project"])
def test_saving_a_copy_does_not_delete_the_other_projects_arrays(tmp_path, external_reference):
    original = project.save(_components(), tmp_path / "original.gimap")
    before = _snapshot(original)
    copied = tmp_path / "copied.gimap"
    if external_reference:
        copied.write_bytes(original.read_bytes())
    project.save(_components(9.0), copied)
    assert original.read_bytes() == before[0]
    assert before[1].read_bytes() == before[2]
    assert project.missing(project.read(original), original) == []
    assert project.missing(project.read(copied), copied) == []


def test_successful_save_without_maps_removes_only_its_previous_sidecar(tmp_path):
    path = project.save(_components(), tmp_path / "sample")
    assert path.suffix == ".gimap"
    unrelated = tmp_path / "other.compare.npz"
    unrelated.write_bytes(b"not part of this project")
    components = _components()
    components.compare_page.series.clear()
    project.save(components, path)
    assert "arrays_file" not in project.read(path)["compare"]
    assert set(tmp_path.iterdir()) == {path, unrelated}


def test_cleanup_failure_does_not_report_a_committed_save_as_failed(tmp_path, monkeypatch):
    path = project.save(_components(), tmp_path / "sample.gimap")
    old_sidecar = _snapshot(path)[1]
    unlink = Path.unlink

    def busy(file, *args, **kwargs):
        if file == old_sidecar:
            raise PermissionError("synthetic open archive")
        return unlink(file, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", busy)
    assert project.save(_components(9.0), path) == path
    current = tmp_path / project.read(path)["compare"]["arrays_file"]
    with np.load(current) as stored:
        np.testing.assert_array_equal(stored["image_0"], np.full((3, 2), 9.0))


@pytest.mark.parametrize("version", [None, True, [], {}, "1.5", 1.5, -1])
def test_invalid_project_version_is_reported_as_invalid_file(tmp_path, version):
    path = tmp_path / "invalid.gimap"
    path.write_text(json.dumps({"format": project.PROJECT_FORMAT, "version": version}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a GIMaP project"):
        project.read(path)


def test_future_project_version_is_still_reported_separately(tmp_path):
    path = tmp_path / "future.gimap"
    path.write_text(json.dumps({"format": project.PROJECT_FORMAT, "version": project.PROJECT_VERSION + 1}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="made by a newer GIMaP"):
        project.read(path)
