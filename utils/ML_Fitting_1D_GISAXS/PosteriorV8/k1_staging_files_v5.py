"""Race-resistant file primitives for K1 job-local staging."""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Iterator
import zipfile

from .k1_training_chain_contract_v5 import canonical_json, digest


WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def lexical_no_symlinks(path: Path, name: str) -> Path:
    lexical = Path(os.path.abspath(path))
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{name} must not traverse a symlink: {current}")
    return lexical


def _status_identity(status: os.stat_result) -> tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_uid,
        status.st_gid,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
        status.st_nlink,
    )


@contextmanager
def stable_regular_stream(path: Path, name: str) -> Iterator[tuple[Path, object, os.stat_result]]:
    """Open one regular path and reject mutation or replacement around its read."""

    lexical = lexical_no_symlinks(path, name)
    resolved_before = lexical.resolve(strict=True)
    path_before = os.lstat(lexical)
    if not stat.S_ISREG(path_before.st_mode):
        raise ValueError(f"{name} must be a regular file")
    try:
        descriptor = os.open(
            lexical,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        raise ValueError(f"{name} changed or became a symlink before open") from exc
    stream = os.fdopen(descriptor, "rb")
    try:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or _status_identity(opened) != _status_identity(
            path_before
        ):
            raise RuntimeError(f"{name} identity changed while it was opened")
        yield resolved_before, stream, opened
    finally:
        try:
            after_fd = os.fstat(stream.fileno())
            lexical_no_symlinks(lexical, name)
            resolved_after = lexical.resolve(strict=True)
            after_path = os.lstat(lexical)
            if (
                _status_identity(after_fd) != _status_identity(path_before)
                or _status_identity(after_path) != _status_identity(path_before)
                or resolved_after != resolved_before
            ):
                raise RuntimeError(f"{name} changed or was replaced during verified read")
        except OSError as exc:
            raise RuntimeError(
                f"{name} changed or was replaced during verified read"
            ) from exc
        finally:
            stream.close()


def file_sha256(path: Path, name: str = "file") -> str:
    return str(regular_identity(path, name)["sha256"])


def read_regular_bytes(
    path: Path,
    name: str,
    *,
    maximum_bytes: int | None = None,
) -> bytes:
    with stable_regular_stream(path, name) as (_, stream, status):
        if maximum_bytes is None:
            raw = stream.read()
        else:
            raw = stream.read(maximum_bytes + 1)
            if len(raw) > maximum_bytes:
                raise ValueError(f"{name} is unexpectedly large")
        if len(raw) != status.st_size:
            raise RuntimeError(f"{name} byte count changed during verified read")
    return raw


def read_only_bytes_identity(
    path: Path,
    name: str,
    *,
    maximum_bytes: int | None = None,
) -> tuple[bytes, dict[str, object]]:
    """Read and identify one immutable single-link file through the same fd."""

    value = sha256()
    with stable_regular_stream(path, name) as (resolved, stream, status):
        if maximum_bytes is None:
            raw = stream.read()
        else:
            raw = stream.read(maximum_bytes + 1)
            if len(raw) > maximum_bytes:
                raise ValueError(f"{name} is unexpectedly large")
        value.update(raw)
        if len(raw) != status.st_size:
            raise RuntimeError(f"{name} byte count changed during verified read")
        mode = stat.S_IMODE(status.st_mode)
        if mode & WRITE_BITS:
            raise ValueError(f"{name} must be read-only")
        if status.st_nlink != 1:
            raise ValueError(f"{name} must have exactly one hard link")
        identity = {
            "path": str(resolved),
            "sha256": value.hexdigest(),
            "byte_count": status.st_size,
            "mode_octal": f"{mode:04o}",
            "device": status.st_dev,
            "inode": status.st_ino,
            "uid": status.st_uid,
            "gid": status.st_gid,
            "link_count": status.st_nlink,
            "mtime_ns": status.st_mtime_ns,
            "ctime_ns": status.st_ctime_ns,
            "regular_file": True,
            "read_only": True,
        }
    return raw, identity


def regular_identity(
    path: Path,
    name: str,
    *,
    require_read_only: bool = False,
    require_single_link: bool = False,
) -> dict[str, object]:
    value = sha256()
    byte_count = 0
    with stable_regular_stream(path, name) as (resolved, stream, status):
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
            byte_count += len(chunk)
    if byte_count != status.st_size:
        raise RuntimeError(f"{name} byte count changed during verified read")
    mode = stat.S_IMODE(status.st_mode)
    if require_read_only and mode & WRITE_BITS:
        raise ValueError(f"{name} must be read-only")
    if require_single_link and status.st_nlink != 1:
        raise ValueError(f"{name} must have exactly one hard link")
    return {
        "path": str(resolved),
        "sha256": value.hexdigest(),
        "byte_count": status.st_size,
        "mode_octal": f"{mode:04o}",
        "device": status.st_dev,
        "inode": status.st_ino,
        "uid": status.st_uid,
        "gid": status.st_gid,
        "link_count": status.st_nlink,
        "mtime_ns": status.st_mtime_ns,
        "ctime_ns": status.st_ctime_ns,
        "regular_file": True,
        "read_only": not bool(mode & WRITE_BITS),
    }


def read_only_identity(path: Path, name: str) -> dict[str, object]:
    return regular_identity(
        path,
        name,
        require_read_only=True,
        require_single_link=True,
    )


def read_only_json(
    path: Path,
    name: str,
    *,
    maximum_bytes: int = 32 * 1024 * 1024,
) -> tuple[dict[str, object], dict[str, object]]:
    """Parse strict JSON from the exact bytes covered by one immutable identity."""

    raw, identity = read_only_bytes_identity(
        path, name, maximum_bytes=maximum_bytes
    )
    return _strict_object(raw, name), identity


def _strict_object(raw: bytes, name: str) -> dict[str, object]:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one object")
    return value


def checked_json(
    path: Path, *, maximum_bytes: int = 32 * 1024 * 1024
) -> dict[str, object]:
    raw = read_regular_bytes(
        path,
        f"JSON artifact {path}",
        maximum_bytes=maximum_bytes,
    )
    return _strict_object(raw, f"JSON artifact {path}")


def checked_zip_manifest(path: Path) -> dict[str, object]:
    try:
        with stable_regular_stream(path, "checked artifact") as (_, stream, _), zipfile.ZipFile(stream, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("checked artifact contains duplicate archive members")
            member = archive.getinfo("manifest.json")
            if member.file_size > 32 * 1024 * 1024:
                raise ValueError("checked artifact manifest is unexpectedly large")
            value = _strict_object(archive.read(member), "checked artifact manifest")
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"checked artifact has no readable manifest: {path}") from exc
    core = dict(value)
    supplied = digest(core.pop("manifest_sha256", None), "artifact manifest SHA-256")
    if supplied != sha256(canonical_json(core).encode()).hexdigest():
        raise ValueError("checked artifact manifest SHA-256 does not reproduce")
    return value


def copy_regular_exclusive(
    source: Path, destination: Path, *, expected_sha256: str, name: str
) -> dict[str, object]:
    expected = digest(expected_sha256, f"{name} expected SHA-256")
    copied = sha256()
    byte_count = 0
    destination_created = False
    try:
        with stable_regular_stream(source, f"original {name}") as (
            source_resolved,
            source_stream,
            source_status,
        ):
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"refusing to overwrite staged {name}")
            destination_descriptor = os.open(
                destination,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o400,
            )
            destination_created = True
            with os.fdopen(destination_descriptor, "wb") as destination_stream:
                while chunk := source_stream.read(1024 * 1024):
                    destination_stream.write(chunk)
                    copied.update(chunk)
                    byte_count += len(chunk)
                destination_stream.flush()
                os.fsync(destination_stream.fileno())
                os.fchmod(destination_stream.fileno(), 0o400)
            if byte_count != source_status.st_size:
                raise RuntimeError(f"original {name} byte count changed during copy")
        local = read_only_identity(destination, f"staged {name}")
        if copied.hexdigest() != expected or local["sha256"] != expected:
            raise RuntimeError(f"staged {name} differs from the frozen identity")
        return {
            "original_path": str(source_resolved),
            "job_local_path": local["path"],
            "frozen_sha256": expected,
            "copy_stream_sha256": copied.hexdigest(),
            "post_copy_sha256": local["sha256"],
            "byte_count": byte_count,
            "mode_octal": local["mode_octal"],
            "regular_file": True,
            "read_only": True,
        }
    except Exception:
        if destination_created:
            destination.unlink(missing_ok=True)
        raise


def existing_under_root(path: Path, root: Path, name: str) -> Path:
    lexical = lexical_no_symlinks(path, name)
    resolved = lexical.resolve(strict=True)
    allowed = root.resolve(strict=True)
    if resolved == allowed or not resolved.is_relative_to(allowed):
        raise ValueError(f"{name} escaped {allowed}")
    return resolved


def freeze_staging_tree(root: Path) -> None:
    def remove_write_bits(path: Path) -> None:
        mode = stat.S_IMODE(path.stat().st_mode)
        frozen_mode = mode & ~WRITE_BITS
        if frozen_mode != mode:
            path.chmod(frozen_mode)

    for current, directory_names, file_names in os.walk(root, topdown=False):
        current_path = Path(current)
        for name in (*directory_names, *file_names):
            path = current_path / name
            if path.is_symlink():
                raise ValueError("job-private staging contains a symlink")
            remove_write_bits(path)
        remove_write_bits(current_path)


__all__ = [
    "WRITE_BITS",
    "checked_json",
    "checked_zip_manifest",
    "copy_regular_exclusive",
    "existing_under_root",
    "file_sha256",
    "freeze_staging_tree",
    "lexical_no_symlinks",
    "read_only_identity",
    "read_only_bytes_identity",
    "read_only_json",
    "read_regular_bytes",
    "regular_identity",
    "stable_regular_stream",
]
