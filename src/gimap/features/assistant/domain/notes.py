"""Values a person wrote in free-text notes: the incidence angle, the X-ray energy, the pixel size.

Only an unambiguous value is returned: when the notes name two different
incidence angles (or energies), neither is taken and the caller has to ask.
"""

from __future__ import annotations

import re
from typing import Optional

_NUMBER = r"(\d+(?:\.\d+)?)"
_INCIDENCE = re.compile(
    r"(?:α\s*_?\s*i|alpha[\s_-]?i|\bai\b|incidence(?:\s+angle)?|incident\s+angle|掠?入射角)"
    r"\s*(?:of|is|was|=|:|≈|~)?\s*" + _NUMBER + r"\s*(?:°|deg(?:rees?)?|度)?",
    re.IGNORECASE,
)
_ENERGY = re.compile(_NUMBER + r"\s*kev\b", re.IGNORECASE)
_MICRONS = r"\s*(?:µm|μm|um|micron(?:s|meters?)?)"
_PIXEL = re.compile(
    r"pixel(?:\s*size)?\s*(?:of|is|=|:)?\s*" + _NUMBER + _MICRONS + r"|" + _NUMBER + _MICRONS + r"\s*pixels?",
    re.IGNORECASE,
)


def _unique(values: list[float], tolerance: float) -> Optional[float]:
    if not values:
        return None
    first = values[0]
    return first if all(abs(value - first) <= tolerance for value in values) else None


def incidence_from_notes(text: str) -> Optional[float]:
    """The incidence angle αi (degrees) the notes give, when they give exactly one (0–5°)."""
    values = [float(match.group(1)) for match in _INCIDENCE.finditer(text or "")]
    return _unique([value for value in values if 0.0 < value < 5.0], 1e-6)


def energy_from_notes(text: str) -> Optional[float]:
    """The X-ray energy (keV) the notes give, when they give exactly one (1–200 keV)."""
    values = [float(match.group(1)) for match in _ENERGY.finditer(text or "")]
    return _unique([value for value in values if 1.0 <= value <= 200.0], 1e-3)


def pixel_size_from_notes(text: str) -> Optional[float]:
    """The detector pixel size (µm) the notes give, when they give exactly one (10–1000 µm)."""
    values = [float(match.group(1) or match.group(2)) for match in _PIXEL.finditer(text or "")]
    return _unique([value for value in values if 10.0 <= value <= 1000.0], 1e-6)


__all__ = ["energy_from_notes", "incidence_from_notes", "pixel_size_from_notes"]
