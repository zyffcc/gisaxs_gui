"""pyFAI detector orientation for flat, regularly spaced pixels."""

from __future__ import annotations

import re

INVALID_ORIENTATION = "The .poni detector orientation must be 1, 2, 3 or 4."
MISSING_SHAPE = "A flipped .poni detector orientation needs the detector dimensions. Supply max_shape or open the frame."


def orientation_number(value) -> int:
    """Integer pyFAI orientation, including numbers written as text in a .poni."""
    if isinstance(value, bool) or str(value).strip() not in ("1", "2", "3", "4"):
        raise ValueError(INVALID_ORIENTATION)
    return int(value)


def oriented_position(x, y, orientation, shape):
    """Convert displayed corner coordinates to native pyFAI coordinates, or vice versa."""
    if orientation == 3:
        return x, y
    if shape is None:
        raise ValueError(MISSING_SHAPE)
    return (shape[1] - x if orientation in (1, 4) else x,
            shape[0] - y if orientation in (1, 2) else y)


# Shapes of the detector models recognised by the automatic calibration search.
KNOWN_DETECTORS = {
    "pilatus100k": (195, 487), "pilatus200k": (407, 487), "pilatus300k": (619, 487), "pilatus300kw": (195, 1475),
    "pilatus1m": (1043, 981), "pilatus2m": (1679, 1475), "pilatus6m": (2527, 2463),
    "eiger500k": (512, 1030), "eiger1m": (1065, 1030), "eiger4m": (2167, 2070), "eiger9m": (3269, 3110),
    "eiger16m": (4371, 4150), "eiger2500k": (512, 1028), "eiger21m": (1062, 1028), "eiger24m": (2162, 2068),
    "eiger29m": (3262, 3108), "eiger216m": (4362, 4148),
}


def detector_shape(name) -> tuple[int, int] | None:
    key = re.sub(r"[^a-z0-9]", "", str(name or "").lower()).replace("cdte", "")
    return KNOWN_DETECTORS.get(key)
