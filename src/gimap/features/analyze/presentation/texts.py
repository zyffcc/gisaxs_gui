"""Texts Analyze shows that the application and the domain make in English, in the interface language.

The reduction, the geometry resolution and the settings file keep their messages and curve titles in
English (they go into exported records, and the assistant reads them). What the page shows of them is
translated here: an exact text through ``tr``, a message made from a known template by recognising the
template and filling the Chinese one with the same values. Numbers, units, symbols and names stay as
they are. Text that matches nothing is shown as it came.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from src.gimap.app.presentation.i18n import tr, trf

from ..application import RECTANGLE
from ..application.geometry_resolution import CENTER_NOT_USED, CENTER_NOT_USED_HEADER

NO_PROFILE = "No instrument profile for {detector} ({shape}). Calibrate once or enter the geometry to get q curves."
PROFILE_SHAPE = "Profile “{name}” is for {profile_shape} frames; this frame is {shape}."
EMPTY_CUT = "{title}: no valid pixels in the cut band."
EMPTY_SECTOR = "{title}: no valid pixels in this sector."
"""Messages of a frame (``FrameAnalysis.messages``: ``use_cases.py``, ``domain/gisaxs.py``, ``domain/giwaxs.py``)."""
PROFILE_ADDED = "Instrument profile “{name}” added from the settings file."
PROFILE_MISSING = "Instrument profile “{name}” is not here: the geometry is matched automatically."
MASK_FILE_MISSING = "Mask file not found, left out: {path}"
BACKGROUND_MISSING = "Background frame not found, left out: {path}"
"""Notes of a settings file applied (``batch_model.apply_settings``)."""
REGION_Q_EMPTY = "Region {name}: the q range is empty."
REGION_CHI_EMPTY = "Region {name}: the χ range is empty."
"""A region set by hand that holds nothing (``domain/regions.py``; the name comes quoted)."""


def curve_title(title: str) -> str:
    """A curve's title in the interface language: its name translated, its values after the comma kept
    (“Box I(qz), q∥ = 0.100–0.200 Å⁻¹” → “框 I(qz), q∥ = 0.100–0.200 Å⁻¹”)."""
    head, comma, values = str(title).partition(", ")
    return tr(head) + comma + values


def _same(text: str) -> str:
    return text


TEMPLATES: tuple[tuple[str, dict[str, Callable[[str], str]]], ...] = (
    (NO_PROFILE, {"detector": tr}),  # “this detector” when the file names none
    (PROFILE_SHAPE, {}),
    (CENTER_NOT_USED, {}),
    (CENTER_NOT_USED_HEADER, {}),
    (EMPTY_CUT, {"title": curve_title}),
    (EMPTY_SECTOR, {"title": curve_title}),
    (PROFILE_ADDED, {}),
    (PROFILE_MISSING, {}),
    (MASK_FILE_MISSING, {}),
    (BACKGROUND_MISSING, {}),
    (REGION_Q_EMPTY, {}),
    (REGION_CHI_EMPTY, {}),
)
"""English templates of the messages made elsewhere, with what to do with each value (default: keep it)."""


def _pattern(template: str) -> re.Pattern:
    parts = re.split(r"\{(\w+)\}", template)
    pattern = "".join(re.escape(part) if index % 2 == 0 else f"(?P<{part}>.+?)" for index, part in enumerate(parts))
    return re.compile(pattern, re.DOTALL)


_PATTERNS = tuple((template, _pattern(template), values) for template, values in TEMPLATES)


def message_text(message: str) -> str:
    """``message`` (made in English by the application or the domain) in the interface language."""
    message = str(message)
    exact = tr(message)
    if exact != message:
        return exact
    for template, pattern, convert in _PATTERNS:
        found = pattern.fullmatch(message)
        if found is not None:
            values = {name: convert.get(name, _same)(value) for name, value in found.groupdict().items()}
            return trf(template, **values)
    return message


def mask_text(shape) -> str:
    """A drawn mask as the Mask step lists it (``MaskShape.describe`` in the interface language)."""
    xs = [x for x, _y in shape.points]
    ys = [y for _x, y in shape.points]
    x, y = f"{min(xs):.0f}–{max(xs):.0f}", f"{min(ys):.0f}–{max(ys):.0f}"
    if shape.kind == RECTANGLE:
        return trf("Rectangle: x {x}, y {y}", x=x, y=y)
    return trf("Polygon ({n} points): x {x}, y {y}", n=len(shape.points), x=x, y=y)


def detector_text(analysis) -> Optional[str]:
    """The name the frame gives its detector, else the name of the instrument profile it matched (``None``:
    neither: say nothing rather than “Detector: Detector”)."""
    name = getattr(analysis, "detector_name", None)
    if name:
        return str(name)
    resolution = getattr(analysis, "resolution", None)
    profile = getattr(resolution, "profile", None) if resolution is not None else None
    return str(profile.name) if profile is not None and getattr(profile, "name", None) else None


__all__ = [
    "BACKGROUND_MISSING", "EMPTY_CUT", "EMPTY_SECTOR", "MASK_FILE_MISSING", "NO_PROFILE", "PROFILE_ADDED",
    "PROFILE_MISSING", "PROFILE_SHAPE", "REGION_CHI_EMPTY", "REGION_Q_EMPTY", "TEMPLATES", "curve_title", "detector_text", "mask_text", "message_text",
]
