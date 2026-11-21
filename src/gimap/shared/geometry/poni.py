"""pyFAI ``.poni`` files: read one into GIMaP's geometry, and write one from it (PONI version 2.1).

A ``.poni`` describes the detector as pyFAI does (:class:`~.detector_tilt.DetectorTilt`): the distance ``L`` from the
sample to the PONI (where the detector normal through the sample meets the detector), the PONI itself (``poni1``
down the rows, ``poni2`` along the columns, metres from the corner of pixel (0, 0)), the rotations rot1–rot3 and the
pixel sizes. GIMaP's :class:`~.detector_geometry.DetectorGeometry` keeps the direct beam and the distance to it
instead, with the same rotations. The two describe the same detector (pyFAI's ``Geometry.getFit2D`` conversion)::

    x = (poni2 − L tan rot1) / pixel2,   y = (poni1 + L tan rot2 / cos rot1) / pixel1,   D = L / (cos rot1 cos rot2)

and back::

    L = D cos rot1 cos rot2,   poni2 = x pixel2 + L tan rot1,   poni1 = y pixel1 − L tan rot2 / cos rot1.

``x, y`` are GIMaP's pixel-corner coordinates (the centre of the first pixel is 0.5), which are pyFAI's. Reading and
writing are each other's inverse to rounding (``repr`` of the floats is written).

Pure Python: no Qt, no feature imports.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Optional

from .detector_geometry import DetectorGeometry
from .detector_tilt import DetectorTilt

DETECTOR_PIXELS_M = (
    ("pilatus", 172e-6), ("eiger", 75e-6), ("perkin", 200e-6), ("lambda", 55e-6), ("mar3450", 100e-6),
    ("mar345", 100e-6), ("jungfrau", 75e-6),
)
"""Pixel sizes of detectors a ``.poni`` file may name instead of listing ``pixel1``/``pixel2``."""
MISSING_VALUES = "The .poni file lacks Distance, Poni1, Poni2 or the pixel size."
PONI_VERSION = "2.1"


def _float(value) -> Optional[float]:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class PoniFile:
    """What a ``.poni`` states: pyFAI's description of the detector (``tilt``, also when every rotation is 0), the
    wavelength (Å; ``None`` when the file has none), the detector's name and, when its ``Detector_config`` gives it,
    the frame shape ``(rows, columns)`` (``max_shape``)."""

    tilt: DetectorTilt
    wavelength_angstrom: Optional[float]
    detector: str = ""
    shape: Optional[tuple[int, int]] = None

    @property
    def tilted(self) -> bool:
        return bool(self.tilt.rot1_rad or self.tilt.rot2_rad or self.tilt.rot3_rad)

    def geometry(self, *, wavelength_angstrom: Optional[float] = None, incidence_deg: float = 0.0) -> DetectorGeometry:
        """GIMaP's geometry of this file (its direct beam, the distance to it and its rotations). ``wavelength_angstrom``
        is needed when the file states none."""
        wavelength = self.wavelength_angstrom if self.wavelength_angstrom else wavelength_angstrom
        if not wavelength:
            raise ValueError("The .poni file states no wavelength: enter the wavelength or the energy.")
        tilt = self.tilt
        x, y = tilt.direct_beam_px()
        return DetectorGeometry(tilt.pixel2_m, tilt.pixel1_m, tilt.beam_distance_m, x, y, float(wavelength),
                                incidence_deg, tilt.rot1_rad, tilt.rot2_rad, tilt.rot3_rad)


def poni_values(text: str) -> dict[str, str]:
    """The ``key: value`` lines of a ``.poni`` (keys in lower case; comments left out)."""
    values: dict[str, str] = {}
    for line in str(text).splitlines():
        if line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip().lower()] = value.strip()
    return values


def _config(values: dict[str, str]) -> dict:
    text = values.get("detector_config")
    if not text:
        return {}
    try:
        config = json.loads(text)
    except ValueError:
        config = {}
        for name in ("pixel1", "pixel2"):
            found = re.search(rf'"{name}"\s*:\s*([0-9.eE+-]+)', text)
            if found:
                config[name] = found.group(1)
    return config if isinstance(config, dict) else {}


def read_poni(text: str) -> PoniFile:
    """A ``.poni`` file's geometry (``ValueError`` when Distance, Poni1, Poni2 or the pixel size is missing)."""
    values = poni_values(text)
    config = _config(values)
    pixel1 = _float(values.get("pixelsize1")) or _float(config.get("pixel1"))
    pixel2 = _float(values.get("pixelsize2")) or _float(config.get("pixel2"))
    if not pixel1 or not pixel2:
        name = values.get("detector", "").lower()
        known = next((size for key, size in DETECTOR_PIXELS_M if key in name), None)
        pixel1, pixel2 = pixel1 or known, pixel2 or known
    distance = _float(values.get("distance"))
    poni1, poni2 = _float(values.get("poni1")), _float(values.get("poni2"))
    if distance is None or poni1 is None or poni2 is None or not pixel1 or not pixel2:
        raise ValueError(MISSING_VALUES)
    rotations = [_float(values.get(name)) or 0.0 for name in ("rot1", "rot2", "rot3")]
    tilt = DetectorTilt(distance, poni1, poni2, *rotations, pixel1, pixel2)
    wavelength = _float(values.get("wavelength"))
    shape = None
    try:
        rows, columns = (int(value) for value in config.get("max_shape") or ())
        shape = (rows, columns) if rows > 0 and columns > 0 else None
    except (TypeError, ValueError):
        shape = None
    return PoniFile(tilt, wavelength * 1e10 if wavelength else None, values.get("detector", ""), shape)


def tilt_from_geometry(geometry: DetectorGeometry) -> DetectorTilt:
    """pyFAI's description of ``geometry`` (the inverse of :meth:`PoniFile.geometry`)."""
    c1, c2 = math.cos(geometry.rot1_rad), math.cos(geometry.rot2_rad)
    distance = geometry.distance_m * c1 * c2
    poni2 = geometry.beam_center_x_px * geometry.pixel_size_x_m + distance * math.tan(geometry.rot1_rad)
    poni1 = geometry.beam_center_y_px * geometry.pixel_size_y_m - distance * math.tan(geometry.rot2_rad) / c1
    return DetectorTilt(distance, poni1, poni2, geometry.rot1_rad, geometry.rot2_rad, geometry.rot3_rad,
                        geometry.pixel_size_y_m, geometry.pixel_size_x_m)


def format_poni(
    geometry: DetectorGeometry, *, shape: Optional[tuple[int, int]] = None, detector: str = "Detector",
    comments: tuple[str, ...] = (),
) -> str:
    """The text of a ``.poni`` (version 2.1, pyFAI's generic ``Detector`` with its pixel sizes and, when known, the
    frame shape) for ``geometry``. The incidence angle and a critical angle are no part of a ``.poni``."""
    tilt = tilt_from_geometry(geometry)
    config: dict = {"pixel1": tilt.pixel1_m, "pixel2": tilt.pixel2_m}
    if shape is not None:
        config["max_shape"] = [int(shape[0]), int(shape[1])]
    config["orientation"] = 3
    lines = [f"# {comment}" for comment in comments]
    lines += [
        "# Nota: C-Order, 1 refers to the Y axis, 2 to the X axis",
        f"poni_version: {PONI_VERSION}",
        f"Detector: {detector or 'Detector'}",
        f"Detector_config: {json.dumps(config)}",
        f"Distance: {tilt.distance_m!r}",
        f"Poni1: {tilt.poni1_m!r}",
        f"Poni2: {tilt.poni2_m!r}",
        f"Rot1: {tilt.rot1_rad!r}",
        f"Rot2: {tilt.rot2_rad!r}",
        f"Rot3: {tilt.rot3_rad!r}",
        f"Wavelength: {geometry.wavelength_angstrom * 1e-10!r}",
    ]
    return "\n".join(lines) + "\n"


__all__ = ["DETECTOR_PIXELS_M", "MISSING_VALUES", "PoniFile", "format_poni", "poni_values", "read_poni",
           "tilt_from_geometry"]
