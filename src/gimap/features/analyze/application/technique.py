"""Which technique Auto takes for a frame: GISAXS, GIWAXS, or SAXS/WAXS in transmission.

GISAXS or GIWAXS follows from how far the detector reaches in 2θ (``domain/classification``). A frame is taken as
transmission instead when no incidence angle is known (none given, and the profile's is 0) and either its file name
names a calibration standard (silver behenate, LaB6, CeO2) or, at small angles (where the rule says GISAXS), it shows
at least three complete, sharp rings about the direct beam (``domain/ring_frame``: rings that run through the directions the detector covers, also below the beam,
which a sample surface would shadow). An incidence angle above 0 always means grazing incidence. The ring test looks
at every pixel once; its answer is kept for the frame and beam centre it was made for.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.gimap.shared.geometry import DetectorGeometry

from ..domain import GISAXS, GIWAXS, TRANSMISSION, calibrant_name, classify_measurement, rings_about_beam

BY_NAME = "Taken as SAXS/WAXS in transmission: the file name names a calibration standard and no incidence angle is known."
BY_RINGS = ("Taken as SAXS/WAXS in transmission: {n} complete rings about the direct beam and no incidence angle is "
            "known.")
TECHNIQUE_NAMES = {GISAXS: "GISAXS", GIWAXS: "GIWAXS", TRANSMISSION: "SAXS/WAXS (transmission)"}


class TechniqueGuess:
    """``detect(path, frame, data, valid, geometry, incidence_given)`` → (technique, message or None)."""

    def __init__(self) -> None:
        self._key: Optional[tuple] = None
        self._rings = 0

    def complete_rings(self, key: tuple, data: np.ndarray, valid: np.ndarray, geometry: DetectorGeometry) -> int:
        if key != self._key:
            found = rings_about_beam(data, valid, geometry.beam_center_x_px, geometry.beam_center_y_px)
            self._key, self._rings = key, found.complete_rings if found.is_ring_frame else 0
        return self._rings

    def detect(
        self, path, frame: tuple, data: np.ndarray, valid: np.ndarray, geometry: DetectorGeometry,
        incidence_given: Optional[float],
    ) -> tuple[str, Optional[str]]:
        shape = (int(data.shape[0]), int(data.shape[1]))
        by_angle = classify_measurement(shape, geometry)
        grazing = bool(incidence_given) or bool(geometry.incidence_deg)
        if grazing:
            return by_angle, None
        if calibrant_name(path):
            return TRANSMISSION, BY_NAME
        if by_angle != GISAXS:  # rings that run round the beam within what a wide-angle detector covers: GIWAXS too
            return by_angle, None
        key = (str(path), frame, shape, round(geometry.beam_center_x_px, 3), round(geometry.beam_center_y_px, 3))
        rings = self.complete_rings(key, data, valid, geometry)
        if rings:
            return TRANSMISSION, BY_RINGS.format(n=rings)
        return by_angle, None


__all__ = ["BY_NAME", "BY_RINGS", "TECHNIQUE_NAMES", "TechniqueGuess"]
