"""SHA-256 of the files a result was made from, so a record names its input exactly, not only by its path.

A digest is kept for (path, size, modification time): the frames of one source exported one by one hash the
file once. A file that cannot be read has no digest (``None``); the record then still names it by path."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, Optional

_CHUNK = 1 << 20
_known: dict[tuple[str, int, int], str] = {}


def sha256_of(path: str | Path) -> Optional[str]:
    try:
        resolved = os.path.realpath(os.fspath(path))
        stat = os.stat(resolved)
        key = (resolved, stat.st_size, stat.st_mtime_ns)
        if key not in _known:
            digest = hashlib.sha256()
            with open(resolved, "rb") as handle:
                for block in iter(lambda: handle.read(_CHUNK), b""):
                    digest.update(block)
            _known[key] = digest.hexdigest()
        return _known[key]
    except OSError:
        return None


def input_digests(paths: Iterable[str | Path]) -> list[dict[str, Optional[str]]]:
    """``[{"file": name, "sha256": hex}, …]`` for every distinct path, in order; ``file`` is the file name, so
    the record can be checked on another computer where the folder is different."""
    seen: list[str] = []
    for path in paths:
        text = os.fspath(path)
        if text and text not in seen:
            seen.append(text)
    return [{"file": Path(text).name, "sha256": sha256_of(text)} for text in seen]
