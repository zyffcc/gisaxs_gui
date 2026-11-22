"""The person's GISAXS cuts as Analyze places, shows and keeps them (the cuts themselves: ``domain/gisaxs_cuts.py``).

* **Units.** A cut is stored in q (Å⁻¹); it can be typed and shown in q, in exit angles or in detector pixels. Each
  axis converts on its own, by an exact and invertible rule: q_z ↔ α_f (α_f depends on q_z alone) ↔ the row on the
  beam-centre column; q_y ↔ 2θ_f on the sample horizon (α_f = 0, q_y = k sin 2θ_f) ↔ the column on the horizon line.
  A pixel or angle typed for a point away from that column or line is therefore read on it (the cut stays exactly the
  q region it names; the detector view draws it where it lies).
* **Drawing.** ``detector_outlines`` and ``q_outlines`` give the border of a cut on the detector (q mapped back to
  pixels, ``shared.geometry.q_inverse``) and on the q map; ``band_outline_q`` the band of the automatic cuts on the q
  map.
* **Keeping.** ``cut_setup`` names the detector and its geometry a set of cuts was made on; ``setup_changes`` says what
  another frame has different, so the cuts are kept or dropped by the person's choice, never silently.
* **Files.** ``cut_to_dict`` / ``cuts_from_dicts``: the cuts in a settings file and in the session memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from src.gimap.shared.geometry import DetectorGeometry
from src.gimap.shared.geometry.q_inverse import (
    alpha_f_of_qz,
    positions_of_q,
    q_of_positions,
    q_per_pixel,
    qy_on_horizon,
    qy_of_two_theta_f,
    qz_of_alpha_f,
    two_theta_f_of_qy,
)

from ..domain import GisaxsCut, cut_from_record, cut_key, cut_outlines, cut_record, cut_rect
from ..domain.gisaxs_cuts import ALONG_QY, ALONG_QZ, BOX, CUT_KINDS, CUT_TEMPLATES, HORIZONTAL, LINE, SECTOR, VERTICAL
from ..domain.vertical_band import SPECULAR_EXCLUSION_PX

UNIT_Q, UNIT_ANGLE, UNIT_PIXEL = "q", "angle", "pixel"
UNITS = (UNIT_Q, UNIT_ANGLE, UNIT_PIXEL)
AXIS_SYMBOLS = {
    ("qz", UNIT_Q): ("qz", "Å⁻¹"), ("qz", UNIT_ANGLE): ("αf", "°"), ("qz", UNIT_PIXEL): ("y", "px"),
    ("qy", UNIT_Q): ("qy", "Å⁻¹"), ("qy", UNIT_ANGLE): ("2θf", "°"), ("qy", UNIT_PIXEL): ("x", "px"),
}
"""The symbol and unit of each axis in each unit (figure content: not translated)."""
WIDTH_SYMBOLS = {UNIT_Q: "Å⁻¹", UNIT_ANGLE: "°", UNIT_PIXEL: "px"}
DECIMALS = {UNIT_Q: 4, UNIT_ANGLE: 3, UNIT_PIXEL: 1}
HORIZONTAL_HALF_PX, VERTICAL_HALF_PX, LINE_HALF_PX, BOX_HALF_PX = 2.5, 5.0, 2.0, 10.0
"""Half widths of a cut placed by one click, in pixels near the beam (as the automatic bands: 2.5 rows, 5 columns)."""
SECTOR_HALF_DEG = 5.0


# -- units ------------------------------------------------------------------------------------


def _vacuum(geometry: DetectorGeometry) -> DetectorGeometry:
    """q_y and the horizon do not depend on refraction: the geometry without it, for the q_y axis."""
    return geometry.with_critical_angle(0.0) if geometry.refraction else geometry


def _column_qz(geometry: DetectorGeometry, rows: np.ndarray) -> np.ndarray:
    return q_of_positions(geometry, np.full(rows.shape, geometry.beam_center_x_px), rows)[1]


def _solve(function, start: np.ndarray, target: np.ndarray, tilted: bool) -> np.ndarray:
    """``start`` refined so that ``function`` gives ``target`` (Newton; an untilted detector is exact already)."""
    position = np.array(start, dtype=np.float64)
    if not tilted:
        return position
    for _ in range(6):
        here = function(position)
        slope = (function(position + 0.01) - here) / 0.01
        with np.errstate(invalid="ignore", divide="ignore"):
            position = position - np.where(slope != 0, (here - target) / slope, 0.0)
    return position


def axis_to_unit(geometry: DetectorGeometry, unit: str, axis: str, values) -> np.ndarray:
    """q values (Å⁻¹) of an axis (``qy`` or ``qz``) in ``unit`` (module docstring: one exact rule per axis)."""
    values = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if unit == UNIT_Q:
        return values
    if axis == "qz":
        if unit == UNIT_ANGLE:
            return alpha_f_of_qz(geometry, values)
        rows = positions_of_q(geometry, np.zeros_like(values), values)[1]
        return _solve(lambda y: _column_qz(geometry, y), rows, values, geometry.grazing_requires_projection)
    flat = _vacuum(geometry)
    if unit == UNIT_ANGLE:
        return two_theta_f_of_qy(flat, values)
    horizon_qz = flat.wavevector_inv_angstrom * math.sin(flat.incidence_rad)
    return positions_of_q(flat, values, np.full_like(values, horizon_qz))[0]


def axis_from_unit(geometry: DetectorGeometry, unit: str, axis: str, values) -> np.ndarray:
    """``axis_to_unit`` backwards: values typed in ``unit`` as q (Å⁻¹)."""
    values = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if unit == UNIT_Q:
        return values
    if axis == "qz":
        return qz_of_alpha_f(geometry, values) if unit == UNIT_ANGLE else _column_qz(geometry, values)
    return qy_of_two_theta_f(_vacuum(geometry), values) if unit == UNIT_ANGLE else qy_on_horizon(geometry, values)


def range_to_unit(geometry: DetectorGeometry, unit: str, axis: str, q_range) -> tuple[float, float]:
    low, high = sorted(float(value) for value in axis_to_unit(geometry, unit, axis, q_range))
    return low, high


def range_from_unit(geometry: DetectorGeometry, unit: str, axis: str, values) -> tuple[float, float]:
    low, high = sorted(float(value) for value in axis_from_unit(geometry, unit, axis, values))
    return low, high


def width_to_unit(geometry: DetectorGeometry, unit: str, width: float) -> float:
    """A half width across a line (Å⁻¹) in ``unit``: pixels of ``q_per_pixel``, degrees of k·angle (small angles)."""
    if unit == UNIT_PIXEL:
        return float(width) / q_per_pixel(geometry)
    if unit == UNIT_ANGLE:
        return math.degrees(float(width) / geometry.wavevector_inv_angstrom)
    return float(width)


def width_from_unit(geometry: DetectorGeometry, unit: str, width: float) -> float:
    if unit == UNIT_PIXEL:
        return float(width) * q_per_pixel(geometry)
    if unit == UNIT_ANGLE:
        return math.radians(float(width)) * geometry.wavevector_inv_angstrom
    return float(width)


def span_text(geometry: DetectorGeometry, unit: str, axis: str, q_range) -> str:
    low, high = range_to_unit(geometry, unit, axis, q_range)
    symbol, suffix = AXIS_SYMBOLS[(axis, unit)]
    digits = DECIMALS[unit]
    return f"{symbol} {0.5 * (low + high):.{digits}f} ± {0.5 * (high - low):.{digits}f} {suffix}"


def describe_cut(cut: GisaxsCut, geometry: Optional[DetectorGeometry], unit: str) -> str:
    """Where a cut is, in one short line in ``unit`` (q when there is no geometry)."""
    unit = unit if geometry is not None else UNIT_Q
    if cut.kind == SECTOR:
        low, high = cut.chi_range
        radial = "" if cut.q_range is None else f", qyz {cut.q_range[0]:.4f}–{cut.q_range[1]:.4f} Å⁻¹"
        return f"χ {0.5 * (low + high):.1f} ± {0.5 * (high - low):.1f}°{radial}"
    if cut.kind == LINE:
        digits = DECIMALS[unit]
        start = [axis_to_unit(geometry, unit, axis, value)[0] if geometry else value
                 for axis, value in zip(("qy", "qz"), cut.start)]
        end = [axis_to_unit(geometry, unit, axis, value)[0] if geometry else value for axis, value in zip(("qy", "qz"), cut.end)]
        width = width_to_unit(geometry, unit, cut.half_width) if geometry else cut.half_width
        return (f"({start[0]:.{digits}f}, {start[1]:.{digits}f}) → ({end[0]:.{digits}f}, {end[1]:.{digits}f})"
                f" ± {width:.{digits}f} {WIDTH_SYMBOLS[unit]}")
    convert = (lambda axis, q_range: span_text(geometry, unit, axis, q_range)) if geometry else (
        lambda axis, q_range: _span_q(axis, q_range))
    order = ("qz", "qy") if cut.kind == HORIZONTAL else ("qy", "qz")
    return " · ".join(convert(axis, getattr(cut, axis)) for axis in order if getattr(cut, axis) is not None)


def _span_q(axis: str, q_range) -> str:
    low, high = q_range
    return f"{axis} {0.5 * (low + high):.4f} ± {0.5 * (high - low):.4f} Å⁻¹"


# -- the frame's q range and outlines ---------------------------------------------------------------


def q_extent(analysis) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    """``((qy_min, qy_max), (qz_min, qz_max))`` of a GISAXS frame above the horizon: its q map's, else from a grid of
    detector positions (``None`` without a geometry)."""
    geometry = analysis.geometry
    if geometry is None:
        return None
    reduction = analysis.reduction
    rsm = reduction.reciprocal_space_map if reduction is not None else None
    if rsm is not None:
        return tuple(rsm.q_parallel_range), tuple(rsm.qz_range)
    rows, columns = analysis.shape
    x_values = np.linspace(0.0, columns, 41)
    if geometry.grazing_requires_projection:
        x, y = np.meshgrid(x_values, np.linspace(0.0, rows, 41))
        # Include the horizon itself in addition to the coarse interior grid.
        horizon = np.broadcast_to(geometry.horizon_row(x_values), x_values.shape)
        x, y = np.r_[x.ravel(), x_values], np.r_[y.ravel(), horizon]
        a, b, c = geometry.horizon_plane()
        use = (y >= 0) & (y <= rows) & (a * (x - geometry.beam_center_x_px) + b * (y - geometry.beam_center_y_px) >= c - 1e-15)
        x, y = x[use], y[use]
    else:
        x, y = np.meshgrid(x_values, np.linspace(0.0, min(rows, max(0.0, geometry.horizon_row())), 41))
    qy, qz = q_of_positions(geometry, x, y)
    finite = np.isfinite(qy) & np.isfinite(qz)
    if not finite.any():
        return None
    return (float(qy[finite].min()), float(qy[finite].max())), (float(qz[finite].min()), float(qz[finite].max()))


def q_at(geometry: DetectorGeometry, x: float, y: float) -> tuple[float, float]:
    """``(q_y, q_z)`` at one continuous detector position."""
    qy, qz = q_of_positions(geometry, np.array([float(x)]), np.array([float(y)]))
    return float(qy[0]), float(qz[0])


def _distinct(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A polyline without repeated points (a zero-length step draws a stray line across a pyqtgraph view)."""
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size > 1:
        moved = np.r_[True, (np.diff(x) != 0) | (np.diff(y) != 0)]
        x, y = x[moved], y[moved]
    return x, y


def q_outlines(cut: GisaxsCut, extent) -> list[tuple[np.ndarray, np.ndarray]]:
    return [_distinct(*line) for line in cut_outlines(cut, extent)]


def detector_outlines(cut: GisaxsCut, geometry: DetectorGeometry, extent) -> list[tuple[np.ndarray, np.ndarray]]:
    """The border of a cut on the detector (continuous pixel coordinates; points no ray reaches are left out)."""
    lines = []
    for qy, qz in cut_outlines(cut, extent):
        x, y = _distinct(*positions_of_q(geometry, qy, qz))
        if x.size >= 2:
            lines.append((x, y))
    return lines


def band_outline_q(geometry: DetectorGeometry, rows, columns, *, points: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """The border of a band of whole pixels (``rows``, ``columns``: half-open ranges) in q, for the q map."""
    (top, bottom), (left, right) = (float(rows[0]), float(rows[1])), (float(columns[0]), float(columns[1]))
    across, down = np.linspace(left, right, points), np.linspace(top, bottom, points)
    x = np.concatenate([across, np.full(points, right), across[::-1], np.full(points, left)])
    y = np.concatenate([np.full(points, top), down, np.full(points, bottom), down[::-1]])
    return _distinct(*q_of_positions(geometry, x, y))


# -- cuts placed by a click, a drag or the Add menu -------------------------------------------------


def _half_q(geometry: DetectorGeometry, axis: str, centre: float, pixels: float) -> float:
    position = axis_to_unit(geometry, UNIT_PIXEL, axis, centre)[0]
    if not math.isfinite(position):
        return pixels * q_per_pixel(geometry)
    low, high = range_from_unit(geometry, UNIT_PIXEL, axis, (position - pixels, position + pixels))
    return 0.5 * (high - low)


def cut_at(kind: str, geometry: DetectorGeometry, qy: float, qz: float, *, name: str, extent=None) -> GisaxsCut:
    """A cut of ``kind`` through one point (q_y, q_z), as wide as the automatic band of its kind."""
    if kind == HORIZONTAL:
        half = _half_q(geometry, "qz", qz, HORIZONTAL_HALF_PX)
        return GisaxsCut(name, HORIZONTAL, qz=(qz - half, qz + half))
    if kind == VERTICAL:
        half = _half_q(geometry, "qy", qy, VERTICAL_HALF_PX)
        return GisaxsCut(name, VERTICAL, qy=(qy - half, qy + half))
    if kind == SECTOR:
        chi = math.degrees(math.atan2(qy, qz))
        return GisaxsCut(name, SECTOR, chi_range=(chi - SECTOR_HALF_DEG, chi + SECTOR_HALF_DEG))
    step = q_per_pixel(geometry)
    if kind == LINE:
        (low, high) = extent[0] if extent is not None else (qy - 50 * step, qy + 50 * step)
        return GisaxsCut(name, LINE, start=(low, qz), end=(high, qz), half_width=LINE_HALF_PX * step)
    half = BOX_HALF_PX * step
    return GisaxsCut(name, BOX, qy=(qy - half, qy + half), qz=(qz - half, qz + half))


def cut_between(kind: str, geometry: DetectorGeometry, first, second, *, name: str) -> GisaxsCut:
    """A cut of ``kind`` from two points (q_y, q_z): a box's corners, a line's ends, a band's edges."""
    (y0, z0), (y1, z1) = first, second
    if kind == LINE:
        return GisaxsCut(name, LINE, start=(y0, z0), end=(y1, z1), half_width=LINE_HALF_PX * q_per_pixel(geometry))
    if kind == HORIZONTAL:
        return GisaxsCut(name, HORIZONTAL, qz=(z0, z1))
    if kind == VERTICAL:
        return GisaxsCut(name, VERTICAL, qy=(y0, y1))
    if kind == SECTOR:
        chi = sorted(math.degrees(math.atan2(y, z)) for y, z in (first, second))
        radii = sorted(math.hypot(y, z) for y, z in (first, second))
        return GisaxsCut(name, SECTOR, q_range=tuple(radii), chi_range=tuple(chi))
    return GisaxsCut(name, BOX, qy=(y0, y1), qz=(z0, z1), along=ALONG_QY)


def _vertical_q_band(geometry: DetectorGeometry, q_range, name: str) -> GisaxsCut:
    """An editable band wholly on one measured side of qy = 0, including a one-sided crop."""
    step = geometry.wavevector_inv_angstrom * geometry.pixel_size_x_m / geometry.distance_m
    gap, width = SPECULAR_EXCLUSION_PX * step, 2 * VERTICAL_HALF_PX * step
    low, high = q_range
    if low == high:
        # A one-column crop has no span in qy. Give its one measured q a finite band.
        if low == 0:
            return GisaxsCut(name, VERTICAL, qy=(gap, gap + width))
        half = min(width, abs(low)) * 0.5
        return GisaxsCut(name, VERTICAL, qy=(low - half, low + half))
    sides = [(1, max(low, 0.0), high), (-1, max(-high, 0.0), -low)]
    for sign, near, far in sides:
        start = max(gap, near)
        if start + width <= far:
            return GisaxsCut(name, VERTICAL, qy=(sign * start, sign * (start + width)))
    # A narrow crop: take the outermost available band, keeping at least half the side off the rod.
    sign, near, far = max(sides, key=lambda side: side[2] - side[1])
    return GisaxsCut(name, VERTICAL, qy=(sign * max(near, 0.5 * far, far - width), sign * far))


def new_cut(kind: str, geometry: DetectorGeometry, extent, *, name: str, yoneda_qz=None) -> GisaxsCut:
    """A cut from the Add menu, where it is easy to see and then to type: a horizontal band at the specular
    reflection (α_f = α_i), a vertical band beside the rod, a box and a line in the
    middle of the frame, a sector at χ = 45°."""
    (qy_low, qy_high), (qz_low, qz_high) = extent
    if kind == HORIZONTAL:
        specular = float(qz_of_alpha_f(geometry, geometry.incidence_deg))
        qz = specular if qz_low < specular < qz_high else 0.5 * (qz_low + qz_high)
        return cut_at(HORIZONTAL, geometry, 0.0, qz, name=name)
    if kind == VERTICAL:
        return _vertical_q_band(geometry, (qy_low, qy_high), name)
    if kind == SECTOR:
        return GisaxsCut(name, SECTOR, chi_range=(45.0 - SECTOR_HALF_DEG, 45.0 + SECTOR_HALF_DEG))
    qy_mid, qz_mid = 0.5 * (qy_low + qy_high), 0.5 * (qz_low + qz_high)
    if kind == LINE:
        start = (0.0, float(yoneda_qz) if yoneda_qz is not None else qz_low)
        return cut_between(LINE, geometry, start, (0.6 * qy_high, 0.6 * qz_high), name=name)
    quarter_y, quarter_z = 0.125 * (qy_high - qy_low), 0.125 * (qz_high - qz_low)
    return GisaxsCut(name, BOX, qy=(qy_mid - quarter_y, qy_mid + quarter_y), qz=(qz_mid - quarter_z, qz_mid + quarter_z))


def copy_name(name: str, taken: Iterable[str]) -> str:
    """``name (2)``, ``name (3)`` …: the first not taken."""
    taken = set(taken)
    base = name.rsplit(" (", 1)[0] if name.endswith(")") and " (" in name else name
    number = 2
    while f"{base} ({number})" in taken:
        number += 1
    return f"{base} ({number})"


def renamed(cut: GisaxsCut, name: str) -> GisaxsCut:
    return replace(cut, name=str(name).strip() or cut.name)


# -- the set-up the cuts were made on ------------------------------------------------------------


@dataclass(frozen=True)
class CutSetup:
    """What a set of cuts was made on: the detector, the frame size and the detector geometry (αi is the sample's:
    the cuts are in q, so another αi only moves them on the detector)."""

    detector: str
    shape: tuple[int, int]
    pixel_um: tuple[float, float]
    distance_mm: float
    wavelength_angstrom: float
    center_px: tuple[float, float]
    tilt_deg: tuple[float, float, float]
    orientation: int = 3


def cut_setup(analysis) -> Optional[CutSetup]:
    geometry = getattr(analysis, "geometry", None)
    if geometry is None:
        return None
    return CutSetup(
        str(analysis.detector_name or ""), tuple(int(value) for value in analysis.shape),
        (geometry.pixel_size_x_m * 1e6, geometry.pixel_size_y_m * 1e6), geometry.distance_m * 1e3,
        geometry.wavelength_angstrom, (geometry.beam_center_x_px, geometry.beam_center_y_px),
        tuple(math.degrees(value) for value in (geometry.rot1_rad, geometry.rot2_rad, geometry.rot3_rad)),
        geometry.orientation,
    )


def setup_changes(before: Optional[CutSetup], after: Optional[CutSetup]) -> list[str]:
    """What differs between two set-ups (English words, translated where shown); ``[]``: the same, or unknown."""
    if before is None or after is None:
        return []
    changes = []
    if before.detector != after.detector:
        changes.append("detector")
    if before.orientation != after.orientation:
        changes.append("detector orientation")
    if before.shape != after.shape:
        changes.append("frame size")
    if max(abs(a - b) for a, b in zip(before.pixel_um, after.pixel_um)) > 0.05:
        changes.append("pixel size")
    if abs(before.distance_mm - after.distance_mm) > 0.5:
        changes.append("distance")
    if abs(before.wavelength_angstrom - after.wavelength_angstrom) > 1e-4 * before.wavelength_angstrom:
        changes.append("wavelength")
    if math.hypot(*(a - b for a, b in zip(before.center_px, after.center_px))) > 1.0:
        changes.append("beam centre")
    if max(abs(a - b) for a, b in zip(before.tilt_deg, after.tilt_deg)) > 0.01:
        changes.append("detector tilt")
    return changes


# -- files ------------------------------------------------------------------------------------


def cut_to_dict(cut: GisaxsCut) -> dict[str, Any]:
    record = cut_record(cut)
    record.pop("q_unit", None)
    return {"name": record.pop("gisaxs_cut"), **record}


def cuts_from_dicts(items: Optional[Sequence[Mapping[str, Any]]]) -> list[GisaxsCut]:
    """``GisaxsCut`` s from dicts (``ValueError`` names the first bad one)."""
    cuts = []
    for index, item in enumerate(items or ()):
        try:
            cuts.append(cut_from_record(item))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Cut {index + 1} is not valid: {exc}") from exc
    return cuts


__all__ = [
    "ALONG_QY", "ALONG_QZ", "BOX", "CUT_KINDS", "CUT_TEMPLATES", "GisaxsCut", "HORIZONTAL", "LINE", "SECTOR", "VERTICAL", "cut_key",
    "cut_rect",
    "AXIS_SYMBOLS", "CutSetup", "DECIMALS", "UNITS", "UNIT_ANGLE", "UNIT_PIXEL", "UNIT_Q", "WIDTH_SYMBOLS",
    "axis_from_unit", "axis_to_unit", "band_outline_q", "q_at", "span_text", "copy_name", "cut_at", "cut_between", "cut_setup",
    "cut_to_dict", "cuts_from_dicts", "describe_cut", "detector_outlines", "new_cut", "q_extent", "q_outlines",
    "range_from_unit", "range_to_unit", "renamed", "setup_changes", "width_from_unit", "width_to_unit",
]
