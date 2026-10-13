"""The release ships a verified runtime subset, without personal examples or broken entry points."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tools.list_fit1d_release_files import KIT, MANIFEST, REQUIRED, SCHEMA, release_files


def install_kit(root: Path) -> dict:
    home = root / KIT
    contents = {name: b"runtime fixture\n" for name in REQUIRED}
    contents["models/models.json"] = json.dumps({
        "default": "general", "models": {"general": {"dir": "general"}},
    }).encode()
    contents.update({
        "models/general/weights_numpy.npz": b"fixture weights",
        "models/general/model_config.json": b"{}",
        "models/general/MODEL_INFO.json": b"{}",
        "README_zh.md": b"Runtime instructions",
        "_env.bat": b"@echo off\r\n",
        "run_panel.bat": b"python -m gisaxs1d_kit.qt_panel\r\n",
        "run_panel.sh": b"#!/bin/sh\n",
        "examples/private_curve.npz": b"private example measurements",
        "expected.json": b"private reference values",
        "selftest_reference_run.json": b"private reference run",
        "selftest.py": b"opens private_curve.npz and expected.json",
        "selftest.bat": b"python selftest.py",
        "run_panel_zh.bat": b"opens private_curve.npz",
        "gisaxs1d_kit/core 3.py": b"old duplicate",
        "app/research/__pycache__/accidental.py": b"cache",
        "app/research/validation/private.py": b"private validation",
    })
    for name, data in contents.items():
        path = home / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {"schema": SCHEMA, "files": {
        name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
    }}
    (home / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_kit_absent_remains_optional(tmp_path):
    assert release_files(tmp_path) == {"files": [], "manifest": None}


def test_runtime_subset_includes_weights_and_revalidates_without_private_examples(tmp_path):
    install_kit(tmp_path)
    selected = release_files(tmp_path)
    paths = selected["files"]
    assert f"{KIT}/models/general/weights_numpy.npz" in paths
    assert f"{KIT}/tables_packed/ff_1_number_v2.fft" in paths
    assert f"{KIT}/app/research/fit1d/api.py" in paths
    assert f"{KIT}/README_zh.md" not in paths
    assert not any("examples/" in path or "selftest" in path or "expected" in path
                   or "run_panel_zh" in path or " 3." in path or "__pycache__" in path
                   or "/validation/" in path for path in paths)
    shipped = tmp_path / "shipped"
    for name in paths:
        destination = shipped / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name == f"{KIT}/{MANIFEST}":
            destination.write_text(json.dumps(selected["manifest"]), encoding="utf-8")
        else:
            shutil.copyfile(tmp_path / name, destination)
    assert release_files(shipped) == selected


@pytest.mark.parametrize("name", ["models/general/weights_numpy.npz",
                                 "tables_packed/ff_1_number_v2.fft", "gisaxs1d_kit/core.py"])
def test_changed_runtime_resources_stop_before_packaging(tmp_path, name):
    install_kit(tmp_path)
    (tmp_path / KIT / name).write_bytes(b"changed after packing")
    with pytest.raises(ValueError, match="checksum mismatch"):
        release_files(tmp_path)


def test_missing_runtime_resource_stops_before_packaging(tmp_path):
    install_kit(tmp_path)
    (tmp_path / KIT / "models/general/weights_numpy.npz").unlink()
    with pytest.raises(ValueError, match="missing or linked"):
        release_files(tmp_path)


def test_missing_manifest_does_not_silently_omit_an_installed_kit(tmp_path):
    (tmp_path / KIT).mkdir(parents=True)
    with pytest.raises(ValueError, match=MANIFEST):
        release_files(tmp_path)


def test_omitting_required_code_from_manifest_stops_packaging(tmp_path):
    manifest = install_kit(tmp_path)
    del manifest["files"]["gisaxs1d_kit/core.py"]
    (tmp_path / KIT / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        release_files(tmp_path)


@pytest.mark.parametrize("name", ["../private.py", "/private.py", "app\\research\\private.py", "."])
def test_manifest_paths_cannot_escape_the_kit(tmp_path, name):
    manifest = install_kit(tmp_path)
    manifest["files"][name] = "0" * 64
    (tmp_path / KIT / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid fit1d manifest path"):
        release_files(tmp_path)
