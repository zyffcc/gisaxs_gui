"""Prediction 文件索引、范围和 stack 分组规则。"""

from __future__ import annotations

import re
from collections.abc import Sequence

FRAME_SUFFIXES = (".cbf", ".edf", ".tif", ".tiff")
"""The detector frames 2D Prediction reads (a NeXus/HDF5 series: Tools ▸ Format Converter first)."""


def extract_frame_index(file_name: str, suffixes: Sequence[str] = FRAME_SUFFIXES) -> int | None:
    """The frame number at the end of a file name (``run_00042.edf`` → 42); 1 for a frame without one."""
    name = str(file_name)
    suffix = next((s for s in suffixes if name.lower().endswith(s.lower())), None)
    if suffix is None:
        return None
    stem = name[: len(name) - len(suffix)]
    match = re.search(r"_(\d+)$", stem) or re.search(r"(\d+)$", stem)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return 1


def extract_cbf_index(file_name: str) -> int | None:
    return extract_frame_index(file_name, (".cbf",))


def parse_index_range(text: str) -> list[int]:
    match = re.match(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", str(text))
    if not match:
        return []
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else start
    if end < start:
        start, end = end, start
    return list(range(start, end + 1))


def build_complete_batches(paths: Sequence[str], batch_size: int) -> tuple[tuple[str, ...], ...]:
    size = max(1, int(batch_size))
    if size == 1:
        return tuple((str(path),) for path in paths)
    return tuple(
        tuple(str(path) for path in paths[index:index + size])
        for index in range(0, len(paths), size)
        if len(paths[index:index + size]) == size
    )
