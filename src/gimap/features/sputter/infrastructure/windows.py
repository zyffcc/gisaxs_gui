"""Which raw frames a time window of a result summed, and the geometry the result was made with.

The window's row is looked up in the model's binned files (``binned/chunk_NNNNN.json``, by its id, the ``frame`` of the
result's tables), the scan folder and the request in ``gimap_request.json``, the model's own estimate of the beam
centre in ``intake/geometry.json`` (``db_col`` / ``db_row``: 0-based pixel centres, converted to GIMaP's pixel
coordinates with ``shared/geometry/conventions.py``) and the detector in the first frame's header. Nothing is
written; no pixel is read.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Optional

from src.gimap.shared.geometry.conventions import canonical_from_index_center

from ..application import WindowFrames
from ..domain import (FRAMES_MISSING, NO_SCAN, NO_WINDOWS, SCAN_MISSING, UNKNOWN_WINDOW, geometry_of, number,
                      window_frames)
from .scan_header import header_text
from .toolkit import REQUEST_FILE

CHUNK = re.compile(r"chunk_\d+\.json")
ESTIMATE_FILE = Path("intake") / "geometry.json"


def _json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _estimated_beam(folder: Path) -> Optional[tuple[float, float]]:
    """The model's own beam centre (GIMaP pixel coordinates), when it wrote one."""
    record = _json(folder / ESTIMATE_FILE)
    if not isinstance(record, dict):
        return None
    column, row = number(record.get("db_col")), number(record.get("db_row"))
    return None if column is None or row is None else canonical_from_index_center(column, row)


def _detector(frame: Path) -> str:
    """The detector a CBF header names (``# Detector: PILATUS 2M, S/N …`` → ``PILATUS 2M``), else ""."""
    try:
        found = re.search(r"^#\s*Detector:\s*([^,\r\n]+)", header_text(frame), re.IGNORECASE | re.MULTILINE)
    except OSError:
        return ""
    return found.group(1).strip() if found else ""


class BinnedWindowReader:
    def read(self, folder: Path, window: str) -> WindowFrames:
        folder, window = Path(folder), str(window)
        record = _json(folder / REQUEST_FILE)
        record = record if isinstance(record, dict) else {}
        request = record.get("request") if isinstance(record.get("request"), dict) else {}
        scan = record.get("scan_folder") or request.get("scan_folder")
        if not isinstance(scan, str) or not scan:
            raise FileNotFoundError(NO_SCAN.format(folder=folder))
        scan = Path(scan)
        if not scan.is_dir():
            raise FileNotFoundError(SCAN_MISSING.format(folder=scan))
        found = window_frames(self._row(folder, window), scan, geometry_of(request, _estimated_beam(folder)),
                              folder=folder)
        missing = [path for path in found.frames if not path.is_file()]
        if missing:
            raise FileNotFoundError(FRAMES_MISSING.format(missing=len(missing), count=found.count, window=window,
                                                          folder=scan, name=missing[0].name))
        return replace(found, detector=_detector(found.frames[0]))

    @staticmethod
    def _row(folder: Path, window: str) -> dict:
        binned = folder / "binned"
        chunks = sorted(path for path in binned.glob("chunk_*.json") if CHUNK.fullmatch(path.name)) \
            if binned.is_dir() else []
        if not chunks:
            raise FileNotFoundError(NO_WINDOWS.format(folder=folder))
        for chunk in chunks:
            data = _json(chunk)
            rows = data.get("rows") if isinstance(data, dict) else None
            for row in rows if isinstance(rows, list) else ():
                if isinstance(row, dict) and str(row.get("frame")) == window:
                    return row
        raise ValueError(UNKNOWN_WINDOW.format(window=window, folder=folder))


__all__ = ["BinnedWindowReader", "CHUNK", "ESTIMATE_FILE"]
