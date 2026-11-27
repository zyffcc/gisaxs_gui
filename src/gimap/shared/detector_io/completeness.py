"""Whether a detector file is completely written, from what the file itself says it holds.

A folder watch that only compares sizes between two looks waits at least one look interval for every frame. An EDF
file states its own length: its header (``{ … }``, padded to a multiple of 512 bytes) gives ``Size`` (bytes of the
image after the header), so a file whose length reaches header + ``Size`` holds its whole image and can be read at
the first look. Every other format, or an EDF without ``Size``, says ``None``: not known, the two looks decide.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_HEADER_MAX = 64 * 1024
_SIZE = re.compile(rb"^\s*Size\s*=\s*(\d+)\s*;", re.MULTILINE)


def edf_written(path: Path) -> Optional[bool]:
    """``True`` when the EDF at ``path`` holds as many bytes as its header announces, ``False`` when it holds fewer
    (still being written), ``None`` when the header cannot tell (no closing brace yet, no ``Size``, several images)."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(_HEADER_MAX)
            length = handle.seek(0, 2)
    except OSError:
        return False
    end = head.find(b"}")
    if not head.startswith(b"{") or end < 0:
        return None if len(head) >= _HEADER_MAX else False
    header_end = end + 1
    if head[header_end:header_end + 1] == b"\n":
        header_end += 1
    header = head[:end]
    if header.count(b"Size") == 0 or b"EDF_BlockID" in header and b"EDF_BinarySize" in header:
        return None
    sizes = _SIZE.findall(header)
    if len(sizes) != 1:
        return None
    return length >= header_end + int(sizes[0])


__all__ = ["edf_written"]
