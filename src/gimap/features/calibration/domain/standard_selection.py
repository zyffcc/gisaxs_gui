"""Calibration standard lookup and source-name detection rules."""

from __future__ import annotations

from pathlib import Path

from .models import CalibrationStandard
from .scientific_kernel import STANDARDS, available_standards


STANDARD_SOURCE_ALIASES = {
    "agbh": ("agbh", "ag_behenate", "silver_behenate"),
    "lab6": ("lab6", "lanthanum_hexaboride"),
    "ceo2": ("ceo2", "cerium_dioxide"),
}


def standard_options() -> tuple[CalibrationStandard, ...]:
    return available_standards()


def detect_standard_keys(source_path: str | Path) -> tuple[str, ...]:
    source_name = str(source_path).lower()
    return tuple(
        key
        for key, aliases in STANDARD_SOURCE_ALIASES.items()
        if any(alias in source_name for alias in aliases)
    )


def standard_display_name(key: str) -> str:
    standard = STANDARDS.get(key)
    return standard.display_name if standard else key


def standard_q_values(key: str) -> tuple[float, ...]:
    resolved_key = "agbh" if key == "auto" else key
    standard = STANDARDS.get(resolved_key)
    return standard.q_values_inv_angstrom if standard else ()
