"""Values a person wrote in free-text notes: the incidence angle, the X-ray energy, the pixel size, the calibrant.

Only an unambiguous value is returned: when the notes name two different
incidence angles (or energies, or calibrants), neither is taken and the caller
has to ask.
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

_BEFORE, _AFTER = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"
"""A calibrant's name stands alone: not inside another word or number (Chinese text around it is fine)."""
_LAB6, _CEO2 = r"lab[6₆]", r"ceo[2₂]"
_JOINED = r"[\s_]*(?:\+|/|&|-|and|und|with|与|和|加)?[\s_]*"
"""How a mixture is written; a comma lists separate calibrants ("AgBH, LaB6, CeO2")."""
_MIXTURE = re.compile(_BEFORE + f"(?:{_LAB6}{_JOINED}{_CEO2}|{_CEO2}{_JOINED}{_LAB6})" + _AFTER, re.IGNORECASE)
_CALIBRANTS = {
    "agbh": re.compile(
        _BEFORE + r"(?:ag[\s_-]?beh?|agbh|ag[\s_-]?behenate|silver[\s_-]+behenate|behenate)" + _AFTER + r"|山嵛酸银",
        re.IGNORECASE,
    ),
    "lab6": re.compile(_BEFORE + rf"(?:{_LAB6}|lanthanum[\s_-]+hexaboride|hexaboride)" + _AFTER + r"|六硼化镧", re.IGNORECASE),
    "ceo2": re.compile(_BEFORE + rf"(?:{_CEO2}|cerium[\s_-]+(?:di)?oxide|ceria)" + _AFTER + r"|氧化铈", re.IGNORECASE),
}
"""The calibrants GIMaP's calibration can fit, as people write them (AgBH, AgBe, silver behenate, LaB₆, ceria …)."""


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


def calibrants_in_notes(text: str) -> list[str]:
    """Every calibrant the notes name (``agbh``, ``lab6``, ``ceo2``, ``lab6_ceo2``), in the order first named.

    LaB6 and CeO2 written together ("LaB6+CeO2", "CeO2/LaB6", "LaB6 and CeO2") are the mixed calibrant;
    named apart they are two calibrants. A name inside a path counts like one in a sentence.
    """
    text = text or ""
    first: dict[str, int] = {}
    for match in _MIXTURE.finditer(text):
        first.setdefault("lab6_ceo2", match.start())
    rest = _MIXTURE.sub(lambda match: " " * len(match.group(0)), text)
    for key, pattern in _CALIBRANTS.items():
        match = pattern.search(rest)
        if match is not None:
            first[key] = match.start()
    return sorted(first, key=first.__getitem__)


def standard_from_notes(text: str) -> Optional[str]:
    """The calibrant the notes name, when they name exactly one (else ``None``: none, or several)."""
    named = calibrants_in_notes(text)
    return named[0] if len(named) == 1 else None


__all__ = ["calibrants_in_notes", "energy_from_notes", "incidence_from_notes", "pixel_size_from_notes", "standard_from_notes"]
