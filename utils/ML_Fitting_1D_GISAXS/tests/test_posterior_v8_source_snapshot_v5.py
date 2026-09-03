from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.package_source_snapshot_v5 import (
    ARCHIVE_ROOT,
    MANIFEST_NAME,
    SOURCE_SNAPSHOT_EXTRACTED_TREE_SCHEMA,
    SOURCE_SNAPSHOT_SCHEMA,
    build_source_snapshot,
    extract_source_snapshot,
    selected_source_files,
    verify_extracted_source_snapshot,
    verify_source_snapshot,
)


def _minimal_source(root: Path) -> Path:
    required = (
        "AGENTS.md",
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "utils/__init__.py",
        "src/gimap/example.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/package_source_snapshot_v5.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/study_protocol.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/launch_k1_phase_a_dag_v5.py",
        "utils/ML_Fitting_1D_GISAXS/tests/test_example.py",
        "docs/architecture/multisolution-inversion.md",
        "docs/research/multisolution-inversion-literature.md",
    )
    for index, relative in enumerate(required):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source-{index}\n", encoding="utf-8")
    (root / "utils/ML_Fitting_1D_GISAXS/Training/Cut_Data_unmasked.txt").parent.mkdir(
        parents=True, exist_ok=True
    )
    (root / "utils/ML_Fitting_1D_GISAXS/Training/Cut_Data_unmasked.txt").write_text(
        "private measurement\n", encoding="utf-8"
    )
    cache = root / "utils/ML_Fitting_1D_GISAXS/PosteriorV8/__pycache__/ignored.py"
    cache.parent.mkdir(parents=True)
    cache.write_text("ignored\n", encoding="utf-8")
    return root


def test_snapshot_includes_untracked_style_source_but_omits_data_and_caches(tmp_path):
    source = _minimal_source(tmp_path / "working-tree")
    names = {path.relative_to(source).as_posix() for path in selected_source_files(source)}

    assert "utils/ML_Fitting_1D_GISAXS/tests/test_example.py" in names
    assert "utils/ML_Fitting_1D_GISAXS/Training/Cut_Data_unmasked.txt" not in names
    assert not any("__pycache__" in name for name in names)


def test_snapshot_is_deterministic_and_verifies_every_manifested_file(tmp_path):
    source = _minimal_source(tmp_path / "working-tree")
    first = build_source_snapshot(source, tmp_path / "first.tar")
    second = build_source_snapshot(source, tmp_path / "second.tar")

    assert first["archive_sha256"] == second["archive_sha256"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["selected_file_count"] == second["selected_file_count"]
    assert (
        verify_source_snapshot(tmp_path / "first.tar", expected_sha256=first["archive_sha256"])[
            "verified"
        ]
        is True
    )
    with tarfile.open(tmp_path / "first.tar", "r") as archive:
        manifest = json.load(archive.extractfile(f"{ARCHIVE_ROOT}/{MANIFEST_NAME}"))
    assert manifest["schema_version"] == SOURCE_SNAPSHOT_SCHEMA
    assert all(
        entry["path"] != "utils/ML_Fitting_1D_GISAXS/Training/Cut_Data_unmasked.txt"
        for entry in manifest["entries"]
    )


def test_snapshot_is_exclusive_and_output_must_be_outside_source(tmp_path):
    source = _minimal_source(tmp_path / "working-tree")
    archive = tmp_path / "source.tar"
    build_source_snapshot(source, archive)
    with pytest.raises(FileExistsError, match="overwrite"):
        build_source_snapshot(source, archive)
    with pytest.raises(ValueError, match="outside"):
        build_source_snapshot(source, source / "source.tar")


def test_safe_extraction_binds_exact_read_only_tree_to_archive_and_manifest(tmp_path):
    source = _minimal_source(tmp_path / "working-tree")
    archive = tmp_path / "source.tar"
    built = build_source_snapshot(source, archive)
    extracted = tmp_path / "immutable-source"

    identity = extract_source_snapshot(
        archive,
        extracted,
        expected_sha256=built["archive_sha256"],
    )

    assert identity["schema_version"] == SOURCE_SNAPSHOT_EXTRACTED_TREE_SCHEMA
    assert identity["archive_sha256"] == built["archive_sha256"]
    assert identity["manifest_sha256"] == built["manifest_sha256"]
    assert identity["source_tree_file_count"] == built["selected_file_count"]
    assert identity["exact_manifest_file_set_verified"] is True
    assert identity["exact_manifest_directory_set_verified"] is True
    assert identity["read_only_tree_verified"] is True
    assert identity["symlink_free_lexical_paths_verified"] is True
    assert (extracted / MANIFEST_NAME).is_file()
    assert archive.stat().st_mode & 0o222 == 0
    assert extracted.stat().st_mode & 0o222 == 0
    assert all(path.stat().st_mode & 0o222 == 0 for path in extracted.rglob("*"))
    assert (
        verify_extracted_source_snapshot(
            archive,
            extracted,
            expected_archive_sha256=built["archive_sha256"],
            expected_manifest_sha256=built["manifest_sha256"],
            expected_source_tree_sha256=identity["source_tree_sha256"],
        )
        == identity
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        extract_source_snapshot(
            archive,
            extracted,
            expected_sha256=built["archive_sha256"],
        )


def test_extracted_verifier_rejects_writable_tree_extra_file_and_identity_drift(tmp_path):
    source = _minimal_source(tmp_path / "working-tree")
    archive = tmp_path / "source.tar"
    built = build_source_snapshot(source, archive)
    extracted = tmp_path / "immutable-source"
    identity = extract_source_snapshot(
        archive,
        extracted,
        expected_sha256=built["archive_sha256"],
    )

    nested = extracted / "utils"
    nested.chmod(0o755)
    with pytest.raises(ValueError, match="source directory must be read-only"):
        verify_extracted_source_snapshot(
            archive,
            extracted,
            expected_archive_sha256=built["archive_sha256"],
        )
    nested.chmod(0o555)

    selected = extracted / "src/gimap/example.py"
    selected.chmod(0o644)
    with pytest.raises(ValueError, match="source file must be read-only"):
        verify_extracted_source_snapshot(
            archive,
            extracted,
            expected_archive_sha256=built["archive_sha256"],
        )
    selected.chmod(0o444)

    extracted.chmod(0o755)
    extra = extracted / "unexpected.py"
    extra.write_text("unexpected\n", encoding="utf-8")
    extra.chmod(0o444)
    extracted.chmod(0o555)
    with pytest.raises(ValueError, match="file set does not exactly match"):
        verify_extracted_source_snapshot(
            archive,
            extracted,
            expected_archive_sha256=built["archive_sha256"],
        )

    with pytest.raises(ValueError, match="manifest SHA-256"):
        verify_extracted_source_snapshot(
            archive,
            extracted,
            expected_archive_sha256=built["archive_sha256"],
            expected_manifest_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="source-tree SHA-256"):
        verify_extracted_source_snapshot(
            archive,
            extracted,
            expected_archive_sha256=built["archive_sha256"],
            expected_source_tree_sha256="0" * 64,
        )
    assert identity["source_tree_sha256"] != "0" * 64


def test_extracted_verifier_rejects_missing_file_symlink_and_wrong_valid_archive(
    tmp_path,
):
    source = _minimal_source(tmp_path / "working-tree")
    archive = tmp_path / "source.tar"
    built = build_source_snapshot(source, archive)
    extracted = tmp_path / "immutable-source"
    extract_source_snapshot(
        archive,
        extracted,
        expected_sha256=built["archive_sha256"],
    )

    symlink_parent = extracted / "src"
    symlink_parent.chmod(0o755)
    linked = symlink_parent / "linked-source"
    linked.symlink_to(extracted / "docs", target_is_directory=True)
    symlink_parent.chmod(0o555)
    with pytest.raises(ValueError, match="symlink"):
        verify_extracted_source_snapshot(
            archive,
            extracted,
            expected_archive_sha256=built["archive_sha256"],
        )
    symlink_parent.chmod(0o755)
    linked.unlink()
    symlink_parent.chmod(0o555)

    wrong_source = _minimal_source(tmp_path / "wrong-working-tree")
    (wrong_source / "src/gimap/example.py").write_text(
        "different valid source\n", encoding="utf-8"
    )
    wrong_archive = tmp_path / "wrong-source.tar"
    wrong = build_source_snapshot(wrong_source, wrong_archive)
    with pytest.raises(ValueError, match="manifest does not match|identity mismatch"):
        verify_extracted_source_snapshot(
            wrong_archive,
            extracted,
            expected_archive_sha256=wrong["archive_sha256"],
        )

    missing = extracted / "src/gimap/example.py"
    missing.parent.chmod(0o755)
    missing.unlink()
    missing.parent.chmod(0o555)
    with pytest.raises(ValueError, match="file set does not exactly match"):
        verify_extracted_source_snapshot(
            archive,
            extracted,
            expected_archive_sha256=built["archive_sha256"],
        )


def test_verifier_rejects_archive_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, "w") as archive:
        payload = b"unsafe"
        info = tarfile.TarInfo(f"{ARCHIVE_ROOT}/../escape.py")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe archive member"):
        verify_source_snapshot(archive_path)


def test_verifier_rejects_unmanifested_regular_file(tmp_path):
    source = _minimal_source(tmp_path / "working-tree")
    original = tmp_path / "source.tar"
    build_source_snapshot(source, original)
    modified = tmp_path / "modified.tar"
    with tarfile.open(original, "r") as incoming, tarfile.open(modified, "w") as outgoing:
        for member in incoming.getmembers():
            stream = incoming.extractfile(member) if member.isfile() else None
            outgoing.addfile(member, stream)
        payload = b"surprise"
        info = tarfile.TarInfo(f"{ARCHIVE_ROOT}/extra.py")
        info.size = len(payload)
        outgoing.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unmanifested"):
        verify_source_snapshot(modified)


def test_verifier_rejects_unmanifested_directory_and_symlinked_path_component(tmp_path):
    source = _minimal_source(tmp_path / "working-tree")
    original = tmp_path / "source.tar"
    build_source_snapshot(source, original)
    modified = tmp_path / "modified.tar"
    with tarfile.open(original, "r") as incoming, tarfile.open(modified, "w") as outgoing:
        for member in incoming.getmembers():
            stream = incoming.extractfile(member) if member.isfile() else None
            outgoing.addfile(member, stream)
        directory = tarfile.TarInfo(f"{ARCHIVE_ROOT}/unexpected-empty/")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o555
        outgoing.addfile(directory)

    with pytest.raises(ValueError, match="directory set"):
        verify_source_snapshot(modified)

    alias = tmp_path / "archive-alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="traverse a symlink"):
        verify_source_snapshot(alias / original.name)
    with pytest.raises(ValueError, match="traverse a symlink"):
        extract_source_snapshot(
            original,
            alias / "aliased-extracted-source",
            expected_sha256=verify_source_snapshot(original)["archive_sha256"],
        )
    assert not (tmp_path / "aliased-extracted-source").exists()
