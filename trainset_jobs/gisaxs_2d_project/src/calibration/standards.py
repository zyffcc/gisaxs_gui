from __future__ import annotations

import math

from .models import CalibrationStandard


def _cubic_q(a_angstrom: float, squared_indices: tuple[int, ...]) -> tuple[float, ...]:
    return tuple(2.0 * math.pi * math.sqrt(value) / a_angstrom for value in squared_indices)


# Ag behenate uses the accepted 58.38 A lamellar spacing. LaB6 and CeO2
# values are generated transparently from their room-temperature cubic cells.
AGBH_Q1 = 2.0 * math.pi / 58.38

STANDARDS: dict[str, CalibrationStandard] = {
    "agbh": CalibrationStandard(
        key="agbh",
        display_name="Silver behenate (AgBH)",
        q_values_inv_angstrom=tuple(AGBH_Q1 * order for order in range(1, 21)),
        relative_intensities=tuple(1.0 / order for order in range(1, 21)),
        notes="Lamellar orders calculated from d001 = 58.38 A.",
    ),
    "lab6": CalibrationStandard(
        key="lab6",
        display_name="Lanthanum hexaboride (LaB6)",
        q_values_inv_angstrom=_cubic_q(4.1569, (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22)),
        notes="Cubic LaB6, a = 4.1569 A; unique low-order powder lines.",
    ),
    "ceo2": CalibrationStandard(
        key="ceo2",
        display_name="Cerium dioxide (CeO2)",
        q_values_inv_angstrom=_cubic_q(5.41165, (3, 4, 8, 11, 12, 16, 19, 20, 24, 27, 32, 35, 36, 40, 43, 44, 48)),
        notes="Fluorite CeO2, a = 5.41165 A; allowed all-odd/all-even reflections.",
    ),
}


def get_standard(key: str) -> CalibrationStandard:
    try:
        return STANDARDS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown calibration standard: {key}") from exc


def available_standards() -> tuple[CalibrationStandard, ...]:
    return tuple(STANDARDS.values())
