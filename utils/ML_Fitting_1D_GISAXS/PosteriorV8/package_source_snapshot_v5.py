"""Build and verify a deterministic V5.2 source-only Maxwell snapshot.

The working tree intentionally contains untracked research code, so a Git
archive is not a complete source record.  This packager walks an explicit
allow-list, rejects symlinks, omits data/model/cache artifacts, and embeds a
content-addressed per-file manifest in an uncompressed deterministic tar.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
from typing import Iterable, Sequence


SOURCE_SNAPSHOT_SCHEMA = "gisaxs.posterior_v8.source_snapshot/v2"
SOURCE_SNAPSHOT_VERSION = (
    "posterior_v8_v5_2_exact_manifest_read_only_extracted_source_binding_v2"
)
SOURCE_SNAPSHOT_EXTRACTED_TREE_SCHEMA = (
    "gisaxs.posterior_v8.extracted_source_snapshot_identity/v1"
)
SOURCE_SNAPSHOT_EXTRACTED_TREE_VERSION = (
    "posterior_v8_exact_archive_manifest_source_tree_read_only_binding_v1"
)
SOURCE_TREE_HASH_SEMANTICS = (
    "sha256(length_prefixed_utf8_relative_path||file_sha256)_over_"
    "lexicographically_sorted_manifest_entries"
)
ARCHIVE_ROOT = "snapshot"
MANIFEST_NAME = "SOURCE-MANIFEST.json"
_SHA256_LENGTH = 64
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH

_ROOT_FILES = (
    Path("AGENTS.md"),
    Path("pyproject.toml"),
    Path("requirements.txt"),
    Path("requirements-dev.txt"),
    Path("utils/__init__.py"),
)
_TREE_RULES = (
    (Path("src"), frozenset({".py", ".qss"})),
    (
        Path("utils/ML_Fitting_1D_GISAXS"),
        frozenset({".py", ".sbatch", ".sh", ".md"}),
    ),
    (Path("docs/architecture"), frozenset({".md"})),
    (Path("docs/research"), frozenset({".md"})),
)
_IGNORED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "checkpoints",
        "datasets",
        "models",
        "node_modules",
    }
)


def _selection_policy() -> dict[str, bool]:
    return {
        "working_tree_including_untracked_source": True,
        "git_tracked_files_only": False,
        "data_models_and_caches_included": False,
        "symlinks_allowed": False,
    }


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _strict_json_object(encoded: bytes, name: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate {name} field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(encoded, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _manifest_entries(manifest: dict[str, object]) -> tuple[dict[str, object], ...]:
    expected_fields = {
        "schema_version",
        "version",
        "archive_root",
        "selection_policy",
        "entries",
    }
    if set(manifest) != expected_fields:
        raise ValueError("source manifest fields are incomplete, extended, or unsupported")
    if (
        manifest["schema_version"] != SOURCE_SNAPSHOT_SCHEMA
        or manifest["version"] != SOURCE_SNAPSHOT_VERSION
        or manifest["archive_root"] != ARCHIVE_ROOT
        or manifest["selection_policy"] != _selection_policy()
    ):
        raise ValueError("source manifest identity or selection policy is unsupported")
    values = manifest["entries"]
    if not isinstance(values, list) or not values:
        raise ValueError("source manifest entries must be a non-empty list")
    entries: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for value in values:
        if not isinstance(value, dict) or set(value) != {"path", "sha256", "byte_count"}:
            raise ValueError("source manifest entry shape is invalid")
        raw_path = value["path"]
        if not isinstance(raw_path, str):
            raise ValueError("source manifest path must be a string")
        relative = PurePosixPath(raw_path)
        relative_text = relative.as_posix()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative_text in {".", MANIFEST_NAME}
            or relative_text != raw_path
        ):
            raise ValueError("source manifest contains an unsafe or non-canonical relative path")
        if relative_text in seen_paths:
            raise ValueError(f"duplicate source manifest path: {relative_text}")
        seen_paths.add(relative_text)
        byte_count = value["byte_count"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError("source manifest byte_count must be a non-negative integer")
        entries.append(
            {
                "path": relative_text,
                "sha256": _digest(value["sha256"], "source manifest file SHA-256"),
                "byte_count": byte_count,
            }
        )
    if [value["path"] for value in entries] != sorted(seen_paths):
        raise ValueError("source manifest entries must use deterministic path order")
    return tuple(entries)


def _source_tree_sha256(entries: Sequence[dict[str, object]]) -> str:
    digest = sha256()
    for entry in entries:
        relative = str(entry["path"]).encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(str(entry["sha256"])))
    return digest.hexdigest()


def _assert_no_symlink_components(path: Path, name: str) -> Path:
    lexical = Path(os.path.abspath(path))
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{name} must not traverse a symlink: {current}")
    return lexical


def _archive_directories(relative_files: Iterable[str]) -> set[str]:
    directories = {ARCHIVE_ROOT}
    for relative in (*relative_files, MANIFEST_NAME):
        parent = PurePosixPath(ARCHIVE_ROOT, relative).parent
        while parent.parts:
            directories.add(parent.as_posix())
            if parent.as_posix() == ARCHIVE_ROOT:
                break
            parent = parent.parent
    return directories


def _under_source(path: Path, source_root: Path, name: str) -> Path:
    lexical = Path(os.path.abspath(path))
    root = source_root.resolve(strict=True)
    try:
        lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} must be under source_root") from exc
    resolved = lexical.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} resolves outside source_root")
    return resolved


def _walk_selected_tree(
    source_root: Path,
    relative_root: Path,
    suffixes: frozenset[str],
) -> Iterable[Path]:
    tree_root = _under_source(source_root / relative_root, source_root, str(relative_root))
    if not tree_root.is_dir() or tree_root.is_symlink():
        raise ValueError(f"source tree must be a real directory: {relative_root}")
    for current, directory_names, file_names in os.walk(tree_root, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            child = current_path / name
            relative = child.relative_to(source_root)
            if name in _IGNORED_PARTS or name.startswith("."):
                continue
            if child.is_symlink():
                raise ValueError(f"source selection contains a symlink: {relative}")
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            path = current_path / name
            relative = path.relative_to(source_root)
            if name.startswith(".") or any(part in _IGNORED_PARTS for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"source selection contains a symlink: {relative}")
            if path.suffix in suffixes:
                yield path


def selected_source_files(source_root: Path) -> tuple[Path, ...]:
    """Return all source-only files in deterministic relative-path order."""

    root = source_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("source_root must be a real directory")
    selected: dict[str, Path] = {}
    for relative in _ROOT_FILES:
        path = _under_source(root / relative, root, str(relative))
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing required source file: {relative}")
        selected[relative.as_posix()] = path
    for relative_root, suffixes in _TREE_RULES:
        for path in _walk_selected_tree(root, relative_root, suffixes):
            relative = path.relative_to(root).as_posix()
            selected[relative] = path
    required = (
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/package_source_snapshot_v5.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/study_protocol.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/launch_k1_phase_a_dag_v5.py",
    )
    missing = [relative for relative in required if relative not in selected]
    if missing:
        raise FileNotFoundError(f"source selection is incomplete: {missing}")
    return tuple(selected[key] for key in sorted(selected))


def _manifest(source_root: Path, files: Sequence[Path]) -> tuple[dict[str, object], bytes]:
    entries = []
    for path in files:
        digest, byte_count = _sha256_file(path)
        entries.append(
            {
                "path": path.relative_to(source_root).as_posix(),
                "sha256": digest,
                "byte_count": byte_count,
            }
        )
    payload = {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA,
        "version": SOURCE_SNAPSHOT_VERSION,
        "archive_root": ARCHIVE_ROOT,
        "selection_policy": _selection_policy(),
        "entries": entries,
    }
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return payload, encoded


def _tar_info(name: str, size: int, *, mode: int = 0o444) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _directory_info(name: str) -> tarfile.TarInfo:
    info = _tar_info(name.rstrip("/") + "/", 0, mode=0o555)
    info.type = tarfile.DIRTYPE
    return info


def build_source_snapshot(source_root: Path, archive_path: Path) -> dict[str, object]:
    """Create an exclusive deterministic tar and return its verified identity."""

    root = _assert_no_symlink_components(source_root, "source_root").resolve(strict=True)
    output = _assert_no_symlink_components(archive_path, "archive output")
    if not output.parent.is_dir():
        raise FileNotFoundError("archive output parent must already exist")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite source snapshot: {output}")
    if output.parent.resolve().is_relative_to(root):
        raise ValueError("archive output must be outside source_root")
    files = selected_source_files(root)
    manifest, manifest_bytes = _manifest(root, files)
    relative_names = [path.relative_to(root).as_posix() for path in files]
    directories = _archive_directories(relative_names)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as archive:
            for directory in sorted(directories):
                archive.addfile(_directory_info(directory))
            manifest_info = _tar_info(f"{ARCHIVE_ROOT}/{MANIFEST_NAME}", len(manifest_bytes))
            archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
            for path, relative in zip(files, relative_names, strict=True):
                digest_before, size = _sha256_file(path)
                info = _tar_info(f"{ARCHIVE_ROOT}/{relative}", size)
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
                digest_after, size_after = _sha256_file(path)
                if (digest_after, size_after) != (digest_before, size):
                    raise RuntimeError(f"source changed while packaging: {relative}")
        temporary.chmod(0o444)
        try:
            os.link(temporary, output)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite source snapshot: {output}") from None
    finally:
        temporary.unlink(missing_ok=True)
    verified = verify_source_snapshot(output)
    if verified["manifest_sha256"] != _sha256_bytes(manifest_bytes):
        output.unlink(missing_ok=True)
        raise RuntimeError("written source snapshot failed manifest identity verification")
    return {
        **verified,
        "archive_path": str(output),
        "selected_file_count": len(manifest["entries"]),
    }


def _safe_member_name(name: str) -> PurePosixPath:
    value = PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError(f"unsafe archive member path: {name}")
    if value.parts[0] != ARCHIVE_ROOT:
        raise ValueError(f"archive member is outside {ARCHIVE_ROOT}: {name}")
    return value


def _verified_snapshot(
    archive_path: Path,
    *,
    expected_sha256: str | None,
    collect_contents: bool,
) -> tuple[dict[str, object], bytes, tuple[dict[str, object], ...], dict[str, bytes]]:
    lexical = _assert_no_symlink_components(archive_path, "source archive")
    archive_file = lexical.resolve(strict=True)
    if not archive_file.is_file():
        raise ValueError("source archive must be a regular file")
    expected = None if expected_sha256 is None else _digest(expected_sha256, "expected SHA-256")
    with archive_file.open("rb") as raw:
        archive_digest = sha256()
        archive_bytes = 0
        while chunk := raw.read(1024 * 1024):
            archive_digest.update(chunk)
            archive_bytes += len(chunk)
        archive_sha256 = archive_digest.hexdigest()
        if expected is not None and archive_sha256 != expected:
            raise ValueError("source archive SHA-256 does not match the expected digest")
        raw.seek(0)
        with tarfile.open(fileobj=raw, mode="r:") as archive:
            names: set[str] = set()
            regular: dict[str, tarfile.TarInfo] = {}
            directory_members: set[str] = set()
            for member in archive.getmembers():
                safe = _safe_member_name(member.name)
                normalized = safe.as_posix().rstrip("/")
                if normalized in names:
                    raise ValueError(f"duplicate archive member: {normalized}")
                names.add(normalized)
                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError(f"unsupported archive member type: {member.name}")
                if member.isfile():
                    regular[normalized] = member
                elif member.isdir():
                    directory_members.add(normalized)
                else:
                    raise ValueError(f"unsupported archive member type: {member.name}")
            manifest_key = f"{ARCHIVE_ROOT}/{MANIFEST_NAME}"
            if manifest_key not in regular:
                raise ValueError("source archive is missing its manifest")
            manifest_stream = archive.extractfile(regular[manifest_key])
            if manifest_stream is None:
                raise ValueError("source manifest is unreadable")
            manifest_bytes = manifest_stream.read()
            entries = _manifest_entries(
                _strict_json_object(manifest_bytes, "source manifest")
            )
            expected_members = {manifest_key}
            contents: dict[str, bytes] = {}
            for entry in entries:
                relative = str(entry["path"])
                member_key = f"{ARCHIVE_ROOT}/{relative}"
                expected_members.add(member_key)
                member = regular.get(member_key)
                if member is None:
                    raise ValueError(f"manifested source file is absent: {relative}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"manifested source file is unreadable: {relative}")
                content = stream.read()
                if (
                    len(content) != entry["byte_count"]
                    or _sha256_bytes(content) != entry["sha256"]
                ):
                    raise ValueError(f"manifested source file identity mismatch: {relative}")
                if collect_contents:
                    contents[relative] = content
            if set(regular) != expected_members:
                raise ValueError("source archive contains unmanifested regular files")
            expected_directories = _archive_directories(
                str(entry["path"]) for entry in entries
            )
            if directory_members != expected_directories:
                raise ValueError(
                    "source archive directory set does not exactly match its manifest"
                )
    identity = {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA,
        "version": SOURCE_SNAPSHOT_VERSION,
        "archive_sha256": archive_sha256,
        "archive_byte_count": archive_bytes,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "selected_file_count": len(entries),
        "verified": True,
    }
    return identity, manifest_bytes, entries, contents


def verify_source_snapshot(
    archive_path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    """Verify archive safety, exact manifest shape, and every file digest."""

    identity, _, _, _ = _verified_snapshot(
        archive_path,
        expected_sha256=expected_sha256,
        collect_contents=False,
    )
    return identity


def _assert_read_only(path: Path, name: str, *, directory: bool) -> None:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    if directory:
        if not path.is_dir():
            raise ValueError(f"{name} must be a directory")
    elif not path.is_file():
        raise ValueError(f"{name} must be a regular file")
    if stat.S_IMODE(path.stat().st_mode) & _WRITE_BITS or os.access(path, os.W_OK):
        raise ValueError(f"{name} must be read-only")


def _expected_directories(relative_files: set[str]) -> set[str]:
    result: set[str] = set()
    for relative in relative_files:
        parent = PurePosixPath(relative).parent
        while parent.parts:
            result.add(parent.as_posix())
            parent = parent.parent
    result.discard(".")
    return result


def verify_extracted_source_snapshot(
    archive_path: Path,
    source_root: Path,
    *,
    expected_archive_sha256: str,
    expected_manifest_sha256: str | None = None,
    expected_source_tree_sha256: str | None = None,
) -> dict[str, object]:
    """Bind one immutable extracted tree to one exact verified source archive."""

    archive_lexical = _assert_no_symlink_components(archive_path, "source archive")
    _assert_read_only(archive_lexical, "source archive", directory=False)
    archive, manifest_bytes, entries, _ = _verified_snapshot(
        archive_lexical,
        expected_sha256=expected_archive_sha256,
        collect_contents=False,
    )
    manifest_sha256 = str(archive["manifest_sha256"])
    if expected_manifest_sha256 is not None and manifest_sha256 != _digest(
        expected_manifest_sha256, "expected manifest SHA-256"
    ):
        raise ValueError("source manifest SHA-256 does not match the expected digest")
    source_tree_sha256 = _source_tree_sha256(entries)
    if expected_source_tree_sha256 is not None and source_tree_sha256 != _digest(
        expected_source_tree_sha256, "expected source-tree SHA-256"
    ):
        raise ValueError("source-tree SHA-256 does not match the expected digest")

    root_lexical = _assert_no_symlink_components(source_root, "source_root")
    root = root_lexical.resolve(strict=True)
    _assert_read_only(root, "source_root", directory=True)
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        _assert_read_only(current_path, "source directory", directory=True)
        for name in directory_names:
            directory = current_path / name
            _assert_read_only(directory, "source directory", directory=True)
            actual_directories.add(directory.relative_to(root).as_posix())
        for name in file_names:
            path = current_path / name
            _assert_read_only(path, "source file", directory=False)
            actual_files.add(path.relative_to(root).as_posix())

    manifested_files = {str(entry["path"]) for entry in entries}
    expected_files = manifested_files | {MANIFEST_NAME}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ValueError(
            "extracted source file set does not exactly match the archive manifest: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    expected_directories = _expected_directories(manifested_files)
    if actual_directories != expected_directories:
        missing = sorted(expected_directories - actual_directories)
        unexpected = sorted(actual_directories - expected_directories)
        raise ValueError(
            "extracted source directory set does not exactly match the archive manifest: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    if (root / MANIFEST_NAME).read_bytes() != manifest_bytes:
        raise ValueError("extracted source manifest does not match the archive manifest")
    for entry in entries:
        path = root / str(entry["path"])
        digest, byte_count = _sha256_file(path)
        if digest != entry["sha256"] or byte_count != entry["byte_count"]:
            raise ValueError(f"extracted source file identity mismatch: {entry['path']}")

    replay = verify_source_snapshot(
        archive_lexical,
        expected_sha256=str(archive["archive_sha256"]),
    )
    if replay != archive:
        raise RuntimeError("source archive changed during extracted-tree verification")
    return {
        "schema_version": SOURCE_SNAPSHOT_EXTRACTED_TREE_SCHEMA,
        "version": SOURCE_SNAPSHOT_EXTRACTED_TREE_VERSION,
        "archive_path": str(archive_lexical.resolve(strict=True)),
        "archive_sha256": archive["archive_sha256"],
        "archive_byte_count": archive["archive_byte_count"],
        "manifest_sha256": manifest_sha256,
        "selected_file_count": archive["selected_file_count"],
        "source_root": str(root),
        "source_tree_sha256": source_tree_sha256,
        "source_tree_file_count": len(entries),
        "source_tree_hash_semantics": SOURCE_TREE_HASH_SEMANTICS,
        "exact_manifest_file_set_verified": True,
        "exact_manifest_directory_set_verified": True,
        "symlink_free_lexical_paths_verified": True,
        "read_only_tree_verified": True,
        "verified": True,
    }


def _make_tree_writable_for_cleanup(path: Path) -> None:
    if not path.exists():
        return
    for current, directory_names, file_names in os.walk(path, topdown=False):
        current_path = Path(current)
        for name in file_names:
            (current_path / name).chmod(0o600)
        for name in directory_names:
            (current_path / name).chmod(0o700)
        current_path.chmod(0o700)


def extract_source_snapshot(
    archive_path: Path,
    source_root: Path,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    """Safely extract one verified archive into a new immutable source tree."""

    archive_lexical = _assert_no_symlink_components(archive_path, "source archive")
    _assert_read_only(archive_lexical, "source archive", directory=False)
    archive, manifest_bytes, entries, contents = _verified_snapshot(
        archive_lexical,
        expected_sha256=expected_sha256,
        collect_contents=True,
    )
    target = _assert_no_symlink_components(source_root, "source_root")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite extracted source snapshot: {target}")
    parent = target.parent.resolve(strict=True)
    temporary = Path(tempfile.mkdtemp(dir=parent, prefix=f".{target.name}.", suffix=".tmp"))
    published = False
    renamed = False
    try:
        files = {MANIFEST_NAME: manifest_bytes, **contents}
        for relative, content in files.items():
            destination = temporary / relative
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            destination.chmod(0o444)
        directories = [path for path in temporary.rglob("*") if path.is_dir()]
        for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
            directory.chmod(0o555)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to overwrite extracted source snapshot: {target}")
        # Keep only the private staging root writable until the atomic publish.
        # A 0555 directory cannot be renamed on macOS even when its parent is
        # writable; all descendants are already frozen before publication.
        temporary.rename(target)
        renamed = True
        target.chmod(0o555)
        identity = verify_extracted_source_snapshot(
            archive_lexical,
            target,
            expected_archive_sha256=str(archive["archive_sha256"]),
            expected_manifest_sha256=str(archive["manifest_sha256"]),
            expected_source_tree_sha256=_source_tree_sha256(entries),
        )
        published = True
        return identity
    finally:
        if not published:
            cleanup = target if renamed else temporary
            if cleanup.exists():
                _make_tree_writable_for_cleanup(cleanup)
                shutil.rmtree(cleanup)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source-root", required=True, type=Path)
    build.add_argument("--archive", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--archive", required=True, type=Path)
    verify.add_argument("--expected-sha256")
    extract = subparsers.add_parser("extract")
    extract.add_argument("--archive", required=True, type=Path)
    extract.add_argument("--source-root", required=True, type=Path)
    extract.add_argument("--expected-sha256", required=True)
    verify_extracted = subparsers.add_parser("verify-extracted")
    verify_extracted.add_argument("--archive", required=True, type=Path)
    verify_extracted.add_argument("--source-root", required=True, type=Path)
    verify_extracted.add_argument("--expected-archive-sha256", required=True)
    verify_extracted.add_argument("--expected-manifest-sha256")
    verify_extracted.add_argument("--expected-source-tree-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        result = build_source_snapshot(args.source_root, args.archive)
    elif args.command == "verify":
        result = verify_source_snapshot(
            args.archive,
            expected_sha256=args.expected_sha256,
        )
    elif args.command == "extract":
        result = extract_source_snapshot(
            args.archive,
            args.source_root,
            expected_sha256=args.expected_sha256,
        )
    else:
        result = verify_extracted_source_snapshot(
            args.archive,
            args.source_root,
            expected_archive_sha256=args.expected_archive_sha256,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_source_tree_sha256=args.expected_source_tree_sha256,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_ROOT",
    "MANIFEST_NAME",
    "SOURCE_SNAPSHOT_EXTRACTED_TREE_SCHEMA",
    "SOURCE_SNAPSHOT_EXTRACTED_TREE_VERSION",
    "SOURCE_SNAPSHOT_SCHEMA",
    "SOURCE_SNAPSHOT_VERSION",
    "SOURCE_TREE_HASH_SEMANTICS",
    "build_source_snapshot",
    "extract_source_snapshot",
    "main",
    "selected_source_files",
    "verify_extracted_source_snapshot",
    "verify_source_snapshot",
]
