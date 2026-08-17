"""Prediction 文件索引、范围和 stack 分组规则。"""

from __future__ import annotations

import re
from collections.abc import Sequence


def extract_cbf_index(file_name: str) -> int | None:
    match = re.search(r"_(\d+)(?=\.cbf$)", file_name, re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)(?=\.cbf$)", file_name, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return 1 if file_name.lower().endswith(".cbf") else None


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
