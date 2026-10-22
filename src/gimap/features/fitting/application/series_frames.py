"""Detector frames in In-situ series: which listed items are frames, and the curve of a frame as the cutter gives it.

The cutter is built by the composition root (``app/series_frames.py``: Analyze's set-up applied to the frame, the
same curve Analyze writes into ``<stem>_fit_input.dat``); here it is only a callable ``cut(recipe, token, summed)``
returning ``q`` (Å⁻¹), ``intensity``, ``sigma``, ``pixels`` and ``observation``. A frame is named by its
``InSituSourceFrame.token`` (the file, with ``::gimap-frame=N`` past the first frame of a source).
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Callable

from src.gimap.shared.detector_io.loading import DETECTOR_SUFFIXES

from ..domain import CurveData
from .curve_units import FROM_FRAME
from .models import CurveOperationResult, InSituSourceFrame
from .use_cases import _structured_file_error

FRAME_PATTERNS = tuple(f"*{suffix}" for suffix in DETECTOR_SUFFIXES)
"""The file patterns of detector frames (the pattern field of the Curves step takes these, or a narrower one)."""


def frame_path(token: str) -> Path:
    return InSituSourceFrame.from_token(token).path


def is_frame(token: str) -> bool:
    """Whether a listed item is a detector frame (to be cut), not a curve file."""
    return frame_path(token).suffix.lower() in DETECTOR_SUFFIXES


def is_frame_pattern(pattern: str) -> bool:
    """Whether the pattern of the Curves step names detector frames (``*.cbf``, ``sample_*.nxs`` …)."""
    return Path(str(pattern).strip()).suffix.lower() in DETECTOR_SUFFIXES


def matching(items, pattern: str) -> list:
    """The ``(token, summed)`` items whose file name matches ``pattern`` (``*`` for every frame)."""
    pattern = str(pattern).strip() or "*"
    return [item for item in items if fnmatch.fnmatch(frame_path(item[0]).name.casefold(), pattern.casefold())]


def frame_name(token: str) -> str:
    """``film_00012.cbf``, or ``run.nxs · frame 4`` for a frame past the first of its source (1-based)."""
    frame = InSituSourceFrame.from_token(token)
    return frame.path.name if frame.frame_index == 0 else f"{frame.path.name} · frame {frame.frame_index + 1}"


def frame_curve(cut: Callable, recipe, token: str, summed=()) -> CurveOperationResult:
    """The frame cut by ``cut`` as a curve (q in Å⁻¹), or the error that says why it could not be."""
    try:
        values = cut(recipe, token, summed)
        return CurveOperationResult(value=CurveData(
            q=values["q"], intensity=values["intensity"], error=values.get("sigma"), q_source_unit="angstrom",
            source_path=token, pixels=values.get("pixels"), observation=dict(values.get("observation") or {}),
            q_unit_from=FROM_FRAME))
    except Exception as exc:  # noqa: BLE001 - every failure of a frame is said with the frame, the series goes on
        return CurveOperationResult(error=_structured_file_error(frame_path(token), "read", exc))


__all__ = ["FRAME_PATTERNS", "frame_curve", "frame_name", "frame_path", "is_frame", "is_frame_pattern", "matching"]
