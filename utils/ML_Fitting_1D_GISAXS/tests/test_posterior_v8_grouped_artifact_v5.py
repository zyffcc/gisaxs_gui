from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import zipfile

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import (
    V5_CHECKED_ARRAY_ARTIFACT_SCHEMA,
    V5_CHECKED_ARRAY_ARTIFACT_VERSION,
    array_manifest,
    canonical_json,
    read_checked_array_artifact,
    write_checked_array_artifact,
)


def _payload(arrays):
    core = {
        "container_schema": V5_CHECKED_ARRAY_ARTIFACT_SCHEMA,
        "container_version": V5_CHECKED_ARRAY_ARTIFACT_VERSION,
        "dataset_schema": "test.grouped/v1",
        "arrays": array_manifest(arrays),
    }
    return {
        **core,
        "manifest_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }


def test_checked_artifact_is_deterministic_atomic_and_never_overwrites(tmp_path):
    arrays = {
        "observation__x": np.arange(24, dtype=np.float32).reshape(2, 4, 3),
        "candidate__recipe_index": np.asarray((0, 1, 1), dtype=np.int32),
    }
    manifest = _payload(arrays)
    first_path = tmp_path / "first.gvd5"
    second_path = tmp_path / "second.gvd5"

    first = write_checked_array_artifact(first_path, manifest=manifest, arrays=arrays)
    second = write_checked_array_artifact(second_path, manifest=manifest, arrays=arrays)
    assert first.artifact_sha256 == second.artifact_sha256
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.manifest_sha256 == manifest["manifest_sha256"]

    with pytest.raises(FileExistsError, match="overwrite"):
        write_checked_array_artifact(first_path, manifest=manifest, arrays=arrays)
    assert first_path.read_bytes() == second_path.read_bytes()


def test_checked_artifact_round_trips_without_pickle(tmp_path):
    arrays = {
        "clean__recipe_json": np.asarray(('{"a":1}', '{"a":2}'), dtype=np.str_),
        "observation__point_mask": np.asarray(((True, False), (True, True))),
    }
    manifest = _payload(arrays)
    path = tmp_path / "roundtrip.gvd5"
    written = write_checked_array_artifact(path, manifest=manifest, arrays=arrays)
    loaded_manifest, loaded, receipt = read_checked_array_artifact(path)

    assert loaded_manifest == manifest
    assert receipt.artifact_sha256 == written.artifact_sha256
    for name, expected in arrays.items():
        np.testing.assert_array_equal(loaded[name], expected)
        assert not loaded[name].flags.writeable


def test_checked_artifact_rejects_tampering_and_old_manifest(tmp_path):
    arrays = {"observation__x": np.ones((1, 2, 3), dtype=np.float32)}
    manifest = _payload(arrays)
    path = tmp_path / "valid.gvd5"
    write_checked_array_artifact(path, manifest=manifest, arrays=arrays)

    with zipfile.ZipFile(path, "r") as source:
        members = {name: source.read(name) for name in source.namelist()}
    changed = np.full((1, 2, 3), 2.0, dtype=np.float32)
    buffer = BytesIO()
    np.lib.format.write_array(buffer, changed, allow_pickle=False)
    members["arrays/observation__x.npy"] = buffer.getvalue()
    tampered = tmp_path / "tampered.gvd5"
    with zipfile.ZipFile(tampered, "w") as target:
        for name, encoded in members.items():
            target.writestr(name, encoded)
    with pytest.raises(ValueError, match="metadata or SHA-256"):
        read_checked_array_artifact(tampered)

    old = dict(manifest)
    old["container_schema"] = "gisaxs.posterior_v8.checked_array_artifact/v0"
    old_path = tmp_path / "old.gvd5"
    with zipfile.ZipFile(old_path, "w") as target:
        target.writestr("manifest.json", json.dumps(old))
        target.writestr("arrays/observation__x.npy", members["arrays/observation__x.npy"])
    with pytest.raises(ValueError, match="unsupported checked-array artifact schema"):
        read_checked_array_artifact(old_path)


def test_object_arrays_and_unsafe_names_are_rejected(tmp_path):
    with pytest.raises(TypeError, match="object/pickle"):
        array_manifest({"clean__bad": np.asarray([object()], dtype=object)})
    with pytest.raises(ValueError, match="unsafe"):
        array_manifest({"../escape": np.asarray((1,), dtype=np.int32)})
    with pytest.raises(FileNotFoundError, match="parent"):
        write_checked_array_artifact(
            tmp_path / "missing" / "x.gvd5",
            manifest=_payload({"x": np.asarray((1,), dtype=np.int32)}),
            arrays={"x": np.asarray((1,), dtype=np.int32)},
        )
