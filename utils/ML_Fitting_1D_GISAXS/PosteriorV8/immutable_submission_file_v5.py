"""Strictly pin one immutable source file for audited scheduler submission."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import stat
from typing import Mapping

from .k1_phase_a_cross_platform_v5 import file_identity as strict_file_identity


_EXACT_IDENTITY_FIELDS = {
    "path",
    "sha256",
    "byte_count",
    "mode",
    "device",
    "inode",
    "mtime_ns",
    "ctime_ns",
    "nlink",
}


def _stat_tuple(status: os.stat_result) -> tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
        status.st_nlink,
    )


def _unlink_if_same_file(path: Path, file_key: tuple[int, int] | None) -> None:
    if file_key is None:
        return
    try:
        status = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (status.st_dev, status.st_ino) == file_key and stat.S_ISREG(status.st_mode):
        path.unlink()


def copy_immutable_submission_file(
    source: Path,
    target: Path,
    *,
    expected_source_identity: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    """Copy through no-follow fds to an exclusive, fsynced, 0400 target."""

    if not isinstance(expected_source_identity, Mapping) or set(
        expected_source_identity
    ) != _EXACT_IDENTITY_FIELDS:
        raise ValueError(f"{name} source identity is incomplete or unsupported")
    before = strict_file_identity(
        source, name=f"immutable source {name}", require_read_only=True
    )
    if before != dict(expected_source_identity):
        raise RuntimeError(f"immutable source {name} changed before pinned copy")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing preoccupied submission {name}: {target}")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise ValueError(f"submission {name} parent must be a real directory")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("pinned submission copies require O_NOFOLLOW support")
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    target_fd: int | None = None
    target_created = False
    target_file_key: tuple[int, int] | None = None
    try:
        source_before = os.fstat(source_fd)
        if not stat.S_ISREG(source_before.st_mode) or source_before.st_nlink != 1:
            raise RuntimeError(f"source {name} ceased to be a unique regular file")
        expected_stat = (
            int(before["device"]),
            int(before["inode"]),
            source_before.st_mode,
            int(before["byte_count"]),
            int(before["mtime_ns"]),
            int(before["ctime_ns"]),
            int(before["nlink"]),
        )
        if _stat_tuple(source_before) != expected_stat:
            raise RuntimeError(f"source {name} changed while its fd was opened")
        target_fd = os.open(
            target,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        target_created = True
        target_status = os.fstat(target_fd)
        target_file_key = (target_status.st_dev, target_status.st_ino)
        copied_sha = sha256()
        copied_bytes = 0
        while chunk := os.read(source_fd, 1024 * 1024):
            view = memoryview(chunk)
            while view:
                written = os.write(target_fd, view)
                view = view[written:]
            copied_sha.update(chunk)
            copied_bytes += len(chunk)
        source_after = os.fstat(source_fd)
        path_after = os.stat(source, follow_symlinks=False)
        if (
            _stat_tuple(source_before) != _stat_tuple(source_after)
            or _stat_tuple(source_after) != _stat_tuple(path_after)
            or not stat.S_ISREG(path_after.st_mode)
        ):
            raise RuntimeError(f"source {name} changed during pinned copy")
        if (
            copied_sha.hexdigest() != before["sha256"]
            or copied_bytes != before["byte_count"]
        ):
            raise RuntimeError(f"pinned {name} bytes do not match the verified source")
        os.fchmod(target_fd, 0o400)
        os.fsync(target_fd)
    except BaseException:
        if target_fd is not None:
            os.close(target_fd)
            target_fd = None
        if target_created:
            _unlink_if_same_file(target, target_file_key)
        raise
    finally:
        os.close(source_fd)
        if target_fd is not None:
            os.close(target_fd)
    identity = strict_file_identity(
        target, name=f"pinned submission {name}", require_read_only=True
    )
    if (
        identity["sha256"] != before["sha256"]
        or identity["byte_count"] != before["byte_count"]
        or identity["mode"] != 0o400
        or identity["nlink"] != 1
    ):
        _unlink_if_same_file(target, target_file_key)
        raise RuntimeError(f"pinned submission {name} final identity is invalid")
    parent_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return identity


__all__ = ["copy_immutable_submission_file"]
