"""Completion-last publication helpers for Phase-A v7."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Mapping, Sequence

from .k1_phase_a_capability_v7 import (
    V7PhaseAInputCapability,
    _claim_completion_capability,
)
from .k1_phase_a_contract_v7 import completion_payload, portable_identity
from .k1_staging_files_v5 import read_only_bytes_identity, read_only_identity


def publish_json_exclusive(path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise FileNotFoundError("Phase-A publication parent must be a real directory")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite Phase-A publication: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    target_created = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o400)
        raw, _ = read_only_bytes_identity(
            temporary, f"pending publication {path.name}"
        )
        if raw != encoded:
            raise RuntimeError("Phase-A pending JSON bytes changed")
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite Phase-A publication: {path}"
            ) from None
        target_created = True
        temporary.unlink()
        raw, identity = read_only_bytes_identity(path, f"published {path.name}")
        if raw != encoded:
            raise RuntimeError("Phase-A published JSON bytes changed")
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return identity
    except BaseException:
        if target_created:
            path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def freeze_tree(root: Path) -> None:
    for current, directory_names, file_names in os.walk(root, topdown=False):
        current_path = Path(current)
        for name in file_names:
            path = current_path / name
            if path.is_symlink():
                raise ValueError("Phase-A output tree contains a symlink")
            path.chmod(0o400)
            read_only_identity(path, f"Phase-A output {path.name}")
        for name in directory_names:
            directory = current_path / name
            if directory.is_symlink():
                raise ValueError("Phase-A output tree contains a symlink")
            directory.chmod(0o500)
        current_path.chmod(0o500)


def hidden_partial_directory(target: Path, job_id: str) -> Path:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite Phase-A output: {target}")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise FileNotFoundError("Phase-A output parent must be a real directory")
    partial = Path(
        tempfile.mkdtemp(
            dir=target.parent,
            prefix=f".{target.name}.partial-{job_id}-",
        )
    )
    partial.chmod(0o700)
    return partial


def publish_file_from_partial(partial: Path, target: Path) -> dict[str, object]:
    """Publish a complete sibling file without replacement, then require nlink one."""

    if partial.parent != target.parent:
        raise ValueError("Phase-A partial file must be a sibling of its target")
    partial.chmod(0o400)
    read_only_identity(partial, f"Phase-A partial {partial.name}")
    try:
        os.link(partial, target, follow_symlinks=False)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite Phase-A output: {target}") from None
    partial.unlink()
    identity = read_only_identity(target, f"Phase-A published {target.name}")
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return identity


def promote_directory(partial: Path, target: Path) -> None:
    """Publish a flat frozen result directory without replacing any path."""

    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite Phase-A output: {target}")
    if partial.parent != target.parent or partial.is_symlink() or not partial.is_dir():
        raise ValueError("Phase-A partial directory is not a sibling real directory")
    freeze_tree(partial)
    children = tuple(partial.iterdir())
    if any(child.is_symlink() or not child.is_file() for child in children):
        raise ValueError("Phase-A promoted model output must be a flat regular-file tree")
    # The files stay frozen while the private staging directory temporarily
    # regains owner-write permission so their directory entries can be moved.
    # Linux correctly rejects child.unlink() from the 0500 directory produced
    # by freeze_tree(), even though each child itself is read-only.
    partial.chmod(0o700)
    target.mkdir(mode=0o700, exist_ok=False)
    for child in children:
        destination = target / child.name
        try:
            os.link(child, destination, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite Phase-A output member: {destination}"
            ) from None
        child.unlink()
        read_only_identity(destination, f"Phase-A promoted {destination.name}")
    partial.chmod(0o700)
    partial.rmdir()
    target.chmod(0o500)
    if not target.is_dir() or target.is_symlink() or stat.S_IMODE(target.stat().st_mode) != 0o500:
        raise RuntimeError("Phase-A directory promotion failed identity checks")
    target_fd = os.open(target, os.O_RDONLY)
    try:
        os.fsync(target_fd)
    finally:
        os.close(target_fd)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def artifact_rows(values: Sequence[tuple[str, Path]]) -> list[dict[str, object]]:
    return [
        {
            "role": role,
            "path": str(path.resolve(strict=True)),
            "identity": portable_identity(read_only_identity(path, f"Phase-A {role}")),
        }
        for role, path in values
    ]


def publish_stage_completion(
    path: Path,
    launch_binding: Mapping[str, object],
    artifacts: Sequence[tuple[str, Path]],
    capability: V7PhaseAInputCapability,
) -> dict[str, object]:
    payload = completion_payload(
        launch_binding,
        artifact_rows(artifacts),
        _claim_completion_capability(capability),
    )
    publish_json_exclusive(path, payload)
    return payload


__all__ = [
    "artifact_rows",
    "freeze_tree",
    "hidden_partial_directory",
    "promote_directory",
    "publish_file_from_partial",
    "publish_json_exclusive",
    "publish_stage_completion",
]
