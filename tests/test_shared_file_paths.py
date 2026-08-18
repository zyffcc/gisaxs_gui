from __future__ import annotations

import os
import unicodedata

from src.gimap.shared.file_paths import normalize_path
from utils.path_utils import normalize_path as legacy_normalize_path


def test_normalize_path_accepts_unicode_file_uri(tmp_path):
    data_file = tmp_path / "衍射 data.cbf"
    data_file.write_bytes(b"fixture")

    assert normalize_path(data_file.as_uri()) == str(data_file)


def test_normalize_path_resolves_environment_and_unicode_form(tmp_path, monkeypatch):
    composed = tmp_path / "café.cbf"
    composed.write_bytes(b"fixture")
    decomposed_name = unicodedata.normalize("NFD", composed.name)
    monkeypatch.setenv("GIMAP_PATH_TEST_ROOT", str(tmp_path))

    resolved = normalize_path(f"$GIMAP_PATH_TEST_ROOT/{decomposed_name}")

    assert os.path.samefile(resolved, composed)


def test_legacy_path_entry_reexports_shared_implementation():
    assert legacy_normalize_path is normalize_path
