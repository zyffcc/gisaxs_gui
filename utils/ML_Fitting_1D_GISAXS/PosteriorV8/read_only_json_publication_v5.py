"""Exclusive atomic publication for canonical read-only JSON artifacts."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import stat
import tempfile
from typing import Mapping

from .grouped_artifact_v5 import canonical_json


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    if not isinstance(payload, Mapping):
        raise TypeError("canonical JSON payload must be a mapping")
    return canonical_json(dict(payload)).encode("utf-8")


def publish_read_only_canonical_json(
    path: str | os.PathLike[str], payload: Mapping[str, object]
) -> str:
    """Publish one immutable canonical object without replacing any path."""

    target = Path(path)
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise ValueError("artifact parent must be an existing non-symlink directory")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite artifact: {target}")
    encoded = canonical_json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o400, follow_symlinks=False)
        os.link(temporary, target, follow_symlinks=False)
        published = True
        temporary.unlink()
        metadata = target.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or metadata.st_nlink != 1
        ):
            raise RuntimeError("published JSON artifact is not uniquely linked and read-only")
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        if published:
            target.unlink(missing_ok=True)
        raise
    return sha256(encoded).hexdigest()


__all__ = ["canonical_json_bytes", "publish_read_only_canonical_json"]
