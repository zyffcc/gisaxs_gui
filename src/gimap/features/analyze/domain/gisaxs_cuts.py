"""Cuts of a GISAXS pattern a person defines: bands, boxes, sectors and lines, all in q.

Every cut is defined in the q_y–q_z plane of the grazing-incidence map (Å⁻¹, ``gisaxs_maps``: q_y signed, q_z along
the surface normal), so it means the same on another frame of the same geometry and is drawn alike on the detector
and on the q map. A pixel belongs to a cut when the q of its centre lies inside: whole pixels, never split. Only pixels
above the sample horizon count (with refraction, above α_c), as in the q map.

Kinds and the curve each gives:

* ``horizontal``: a band q_z ∈ [z0, z1] (limited in q_y when ``qy`` is given) → I(q_y), one point per detector column:
  the mean of the column's pixels in the band at the mean q_y of those pixels — the sampling of the automatic Yoneda
  cut (``gisaxs.horizontal_cut``), which keeps its own band of whole rows.
* ``vertical``: a band q_y ∈ [y0, y1] (limited in q_z when ``qz`` is given) → I(q_z), one point per detector row.
* ``box``: q_y ∈ [y0, y1] × q_z ∈ [z0, z1] → I(q_y) per column or I(q_z) per row (``along``).
* ``sector``: q_yz = √(q_y² + q_z²) ∈ [r0, r1] (every q_yz when ``q_range`` is not given) and χ = atan2(q_y, q_z) ∈
  [c0, c1] (0° along q_z, ±90° in the sample plane, the sign of q_y; the origin is the direct beam) → I(q_yz).
* ``line``: within ``half_width`` (Å⁻¹, perpendicular) of the segment from ``start`` to ``end`` (each (q_y, q_z)) → I(s),
  s the distance from ``start`` along the segment.

Sector and line curves are binned in steps of ``q_per_pixel`` (the q of one pixel near the direct beam) and sparse
bins are joined (``merge_sparse_bins``). Errors follow the frame's model (``BinnedMean``, ``native_profile``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from src.gimap.shared.geometry import DetectorGeometry
from src.gimap.shared.geometry.q_inverse import alpha_f_of_qz, q_of_positions, two_theta_f_of_qy

from .binning import BinnedMean, native_profile
from .curve_errors import PixelErrors, pixel_errors
from .models import Curve

HORIZONTAL, VERTICAL, BOX, SECTOR, LINE = "horizontal", "vertical", "box", "sector", "line"
CUT_KINDS = (HORIZONTAL, VERTICAL, BOX, SECTOR, LINE)
ALONG_QY, ALONG_QZ = "qy", "qz"
CUT_KEY = "cut"
"""Curve keys of the person's GISAXS cuts: ``cut1``, ``cut2`` …"""
AXIS_LABELS = {ALONG_QY: "qy (Å⁻¹)", ALONG_QZ: "qz (Å⁻¹)", SECTOR: "qyz (Å⁻¹)", LINE: "s (Å⁻¹)"}


EMPTY_RANGE = "Cut “{name}”: the {what} range is empty."
UNKNOWN_KIND = "Cut “{name}”: unknown kind {kind}."
NEEDS_QZ = "Cut “{name}”: a horizontal band needs its qz range."
NEEDS_QY = "Cut “{name}”: a vertical band needs its qy range."
NEEDS_BOX = "Cut “{name}”: a box needs its qy and qz ranges."
BOX_ALONG = "Cut “{name}”: a box is cut along qy or qz."
LINE_ENDS = "Cut “{name}”: a line needs a start and an end point (qy, qz)."
LINE_LENGTH = "Cut “{name}”: the line has no length."
LINE_WIDTH = "Cut “{name}”: the half width of a line must be above 0."
NO_PIXELS = "Cut “{name}”: no valid pixels above the horizon in it."
CUT_TEMPLATES = (EMPTY_RANGE, UNKNOWN_KIND, NEEDS_QZ, NEEDS_QY, NEEDS_BOX, BOX_ALONG, LINE_ENDS, LINE_LENGTH, LINE_WIDTH,
                 NO_PIXELS)
"""The messages of a cut that cannot be made or holds nothing (English: the presentation translates them)."""


def _range(name: str, what: str, values) -> Optional[tuple[float, float]]:
    if values is None:
        return None
    low, high = sorted(float(value) for value in values)
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        raise ValueError(EMPTY_RANGE.format(name=name, what=what))
    return low, high


def _point(name: str, values) -> tuple[float, float]:
    try:
        y, z = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(LINE_ENDS.format(name=name)) from exc
    if not (math.isfinite(y) and math.isfinite(z)):
        raise ValueError(LINE_ENDS.format(name=name))
    return y, z


@dataclass(frozen=True)
class GisaxsCut:
    """One cut of a GISAXS frame, in q (Å⁻¹) and degrees (see the module docstring for each ``kind``)."""

    name: str
    kind: str
    qy: Optional[tuple[float, float]] = None
    qz: Optional[tuple[float, float]] = None
    along: str = ALONG_QY
    q_range: Optional[tuple[float, float]] = None
    chi_range: tuple[float, float] = (-5.0, 5.0)
    start: Optional[tuple[float, float]] = None
    end: Optional[tuple[float, float]] = None
    half_width: float = 0.0

    def __post_init__(self) -> None:
        name, kind = str(self.name), str(self.kind)
        if kind not in CUT_KINDS:
            raise ValueError(UNKNOWN_KIND.format(name=name, kind=kind))
        set_ = lambda key, value: object.__setattr__(self, key, value)  # noqa: E731
        set_("name", name)
        set_("qy", _range(name, "qy", self.qy))
        set_("qz", _range(name, "qz", self.qz))
        if kind == HORIZONTAL and self.qz is None:
            raise ValueError(NEEDS_QZ.format(name=name))
        if kind == VERTICAL and self.qy is None:
            raise ValueError(NEEDS_QY.format(name=name))
        if kind == BOX and (self.qy is None or self.qz is None):
            raise ValueError(NEEDS_BOX.format(name=name))
        along = {HORIZONTAL: ALONG_QY, VERTICAL: ALONG_QZ}.get(kind, self.along)
        if along not in (ALONG_QY, ALONG_QZ):
            raise ValueError(BOX_ALONG.format(name=name))
        set_("along", along)
        if kind == SECTOR:
            low, high = sorted(float(value) for value in self.chi_range)
            low, high = max(-90.0, low), min(90.0, high)
            if high <= low:
                raise ValueError(EMPTY_RANGE.format(name=name, what="χ"))
            set_("chi_range", (low, high))
            q_range = _range(name, "q", self.q_range)
            if q_range is not None:
                q_range = (max(0.0, q_range[0]), q_range[1])
            set_("q_range", q_range)
        if kind == LINE:
            start, end = _point(name, self.start), _point(name, self.end)
            if math.hypot(end[0] - start[0], end[1] - start[1]) <= 0:
                raise ValueError(LINE_LENGTH.format(name=name))
            width = float(self.half_width)
            if not (math.isfinite(width) and width > 0):
                raise ValueError(LINE_WIDTH.format(name=name))
            set_("start", start)
            set_("end", end)
            set_("half_width", width)

    @property
    def axis(self) -> str:
        """What the curve is against: ``qy``, ``qz``, ``sector`` (q_yz) or ``line`` (s)."""
        if self.kind in (SECTOR, LINE):
            return self.kind
        return self.along

    @property
    def x_label(self) -> str:
        return AXIS_LABELS[self.axis]


def cut_key(index: int) -> str:
    return f"{CUT_KEY}{index + 1}"


def _line_frame(cut: GisaxsCut):
    (y0, z0), (y1, z1) = cut.start, cut.end
    length = math.hypot(y1 - y0, z1 - z0)
    uy, uz = (y1 - y0) / length, (z1 - z0) / length
    return y0, z0, uy, uz, length


def cut_mask(cut: GisaxsCut, maps, usable: np.ndarray) -> np.ndarray:
    """The pixels of a cut (``maps``: ``GisaxsMaps``; ``usable``: valid and above the horizon)."""
    qy, qz = maps.qy, maps.qz
    inside = np.array(usable, dtype=bool, copy=True)
    if cut.kind in (HORIZONTAL, VERTICAL, BOX):
        if cut.qy is not None:
            inside &= (qy >= cut.qy[0]) & (qy <= cut.qy[1])
        if cut.qz is not None:
            inside &= (qz >= cut.qz[0]) & (qz <= cut.qz[1])
        return inside
    if cut.kind == SECTOR:
        chi = np.degrees(np.arctan2(qy, qz))
        inside &= (chi >= cut.chi_range[0]) & (chi <= cut.chi_range[1])
        if cut.q_range is not None:
            radius = np.hypot(qy, qz)
            inside &= (radius >= cut.q_range[0]) & (radius <= cut.q_range[1])
        return inside
    y0, z0, uy, uz, length = _line_frame(cut)
    along = (qy - y0) * uy + (qz - z0) * uz
    across = (qz - z0) * uy - (qy - y0) * uz
    return inside & (along >= 0) & (along <= length) & (np.abs(across) <= cut.half_width)


def cut_record(cut: GisaxsCut) -> dict[str, Any]:
    """The definition of a cut in q, as written into its curve's region (the CSV header, the JSON record)."""
    record: dict[str, Any] = {"gisaxs_cut": cut.name, "kind": cut.kind, "q_unit": "1/A"}
    if cut.kind in (HORIZONTAL, VERTICAL, BOX):
        record.update(qy=None if cut.qy is None else list(cut.qy), qz=None if cut.qz is None else list(cut.qz),
                      along=cut.along)
    elif cut.kind == SECTOR:
        record.update(q_yz=None if cut.q_range is None else list(cut.q_range), chi_deg=list(cut.chi_range))
    else:
        record.update(start=list(cut.start), end=list(cut.end), half_width=cut.half_width)
    return record


def cut_from_record(record: Mapping[str, Any]) -> GisaxsCut:
    """The cut a region record describes (``cut_record``; extra keys are ignored)."""
    kind = str(record.get("kind"))
    name = str(record.get("gisaxs_cut") or record.get("name") or "Cut")
    pair = lambda key: None if record.get(key) is None else tuple(float(value) for value in record[key])  # noqa: E731
    if kind == SECTOR:
        return GisaxsCut(name, kind, q_range=pair("q_yz"),
                         chi_range=pair("chi_deg") or (-5.0, 5.0))
    if kind == LINE:
        return GisaxsCut(name, kind, start=pair("start"), end=pair("end"),
                         half_width=float(record.get("half_width") or 0.0))
    return GisaxsCut(name, kind, qy=pair("qy"), qz=pair("qz"),
                     along=str(record.get("along") or ALONG_QY))


def _angle_record(cut: GisaxsCut, geometry: Optional[DetectorGeometry]) -> dict[str, Any]:
    """The q ranges of a band or box also as exit angles: α_f of q_z (exact), 2θ_f of q_y on the horizon."""
    if geometry is None or cut.kind not in (HORIZONTAL, VERTICAL, BOX):
        return {}
    record = {}
    if cut.qz is not None:
        record["alpha_f_deg"] = [float(value) for value in alpha_f_of_qz(geometry, np.array(cut.qz))]
    if cut.qy is not None:
        record["two_theta_f_deg"] = [float(value) for value in two_theta_f_of_qy(geometry, np.array(cut.qy))]
    return record


def _title(cut: GisaxsCut) -> str:
    if cut.kind == SECTOR:
        chi = f"χ {cut.chi_range[0]:g}–{cut.chi_range[1]:g}°"
        q = "" if cut.q_range is None else f", qyz {cut.q_range[0]:.4f}–{cut.q_range[1]:.4f} Å⁻¹"
        return f"{cut.name}: I(qyz), {chi}{q}"
    if cut.kind == LINE:
        (y0, z0), (y1, z1) = cut.start, cut.end
        return f"{cut.name}: I(s), ({y0:.4f}, {z0:.4f}) → ({y1:.4f}, {z1:.4f}) ± {cut.half_width:.4f} Å⁻¹"
    parts = []
    if cut.qz is not None and cut.along == ALONG_QY:
        parts.append(f"qz {cut.qz[0]:.4f}–{cut.qz[1]:.4f} Å⁻¹")
    if cut.qy is not None and (cut.along == ALONG_QZ or cut.kind == BOX):
        parts.append(f"qy {cut.qy[0]:.4f}–{cut.qy[1]:.4f} Å⁻¹")
    if cut.kind == BOX and cut.along == ALONG_QZ and cut.qz is not None:
        parts.append(f"qz {cut.qz[0]:.4f}–{cut.qz[1]:.4f} Å⁻¹")
    return f"{cut.name}: I({cut.along}), " + ", ".join(parts)


def cut_curve(
    cut: GisaxsCut, key: str, maps, values: np.ndarray, usable: np.ndarray, *, step: float,
    counts: bool | np.ndarray | PixelErrors = True, geometry: Optional[DetectorGeometry] = None,
) -> Curve:
    """The curve of one cut (module docstring). ``counts``: as ``gisaxs.reduce_gisaxs``; ``step``: the bin width of
    sector and line curves (Å⁻¹)."""
    inside = cut_mask(cut, maps, usable)
    record = {**cut_record(cut), **_angle_record(cut, geometry), "pixels_in_cut": int(inside.sum())}
    title, label = _title(cut), cut.x_label
    empty = np.array([], dtype=np.float64)
    if not inside.any():
        return Curve(key, title, empty, empty, empty, np.array([], dtype=np.int64), label, region=record)
    rows, columns = np.nonzero(inside.any(axis=1))[0], np.nonzero(inside.any(axis=0))[0]
    window = (slice(int(rows[0]), int(rows[-1]) + 1), slice(int(columns[0]), int(columns[-1]) + 1))
    record["pixel_rows"] = [int(rows[0]), int(rows[-1]) + 1]
    record["pixel_columns"] = [int(columns[0]), int(columns[-1]) + 1]
    errors = pixel_errors(counts).subset(window)
    if cut.axis in (ALONG_QY, ALONG_QZ):
        x_map = maps.qy if cut.axis == ALONG_QY else maps.qz
        x, mean, sigma, pixels = native_profile(
            values[window], inside[window], x_map[window], axis=0 if cut.axis == ALONG_QY else 1,
            errors=errors,
        )
        return Curve(key, title, x, mean, sigma, pixels, label, region=record)
    qy = maps.qy[window][inside[window]].astype(np.float64)
    qz = maps.qz[window][inside[window]].astype(np.float64)
    if cut.kind == SECTOR:
        coordinate = np.hypot(qy, qz)
        low, high = cut.q_range or (float(coordinate.min()), float(coordinate.max()))
    else:
        y0, z0, uy, uz, length = _line_frame(cut)
        coordinate = (qy - y0) * uy + (qz - z0) * uz
        low, high = 0.0, length
    bins = max(1, int(math.ceil((high - low) / max(float(step), 1e-12))))
    accumulator = BinnedMean.linear(low, high, bins)
    accumulator.add(coordinate, values[window][inside[window]], errors=errors.subset(inside[window]))
    x, mean, sigma, pixels = accumulator.result(counts=errors, merge=True)
    return Curve(key, title, x, mean, sigma, pixels, label, region=record)


def _segment(a, b, points: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, points)
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def _closed(corners, points: int) -> tuple[np.ndarray, np.ndarray]:
    """A closed polygon, each edge sampled in proportion to its length (a short edge: its two ends only)."""
    edges = list(zip(corners, corners[1:] + corners[:1]))
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in edges]
    longest = max(lengths) or 1.0
    ys, zs = [], []
    for (first, second), length in zip(edges, lengths):
        y, z = _segment(first, second, max(2, int(round(points * length / longest))))
        ys.append(y[:-1])
        zs.append(z[:-1])
    ys.append(np.array([corners[0][0]]))
    zs.append(np.array([corners[0][1]]))
    return np.concatenate(ys), np.concatenate(zs)


def cut_outlines(cut: GisaxsCut, extent, *, points: int = 48) -> list[tuple[np.ndarray, np.ndarray]]:
    """Closed polylines ``(q_y, q_z)`` of a cut's border (each edge sampled, so it can be mapped onto the detector,
    where it curves). ``extent``: ``((qy_min, qy_max), (qz_min, qz_max))`` of the frame, for an open band."""
    (qy_low, qy_high), (qz_low, qz_high) = extent
    if cut.kind in (HORIZONTAL, VERTICAL, BOX):
        (y0, y1), (z0, z1) = cut.qy or (qy_low, qy_high), cut.qz or (qz_low, qz_high)
        return [_closed([(y0, z0), (y1, z0), (y1, z1), (y0, z1)], points)]
    if cut.kind == SECTOR:
        corners = [(qy_low, qz_low), (qy_low, qz_high), (qy_high, qz_low), (qy_high, qz_high)]
        r0, r1 = cut.q_range or (0.0, max(math.hypot(y, z) for y, z in corners))
        chi = np.radians(np.linspace(cut.chi_range[0], cut.chi_range[1], 2 * points))
        outer = (r1 * np.sin(chi), r1 * np.cos(chi))
        inner_chi = chi[::-1] if r0 > 0 else chi[:1]  # (from the direct beam: one point)
        inner = (r0 * np.sin(inner_chi), r0 * np.cos(inner_chi))
        y = np.concatenate([outer[0], inner[0], outer[0][:1]])
        z = np.concatenate([outer[1], inner[1], outer[1][:1]])
        return [(y, z)]
    y0, z0, uy, uz, length = _line_frame(cut)
    ny, nz = -uz * cut.half_width, uy * cut.half_width
    (sy, sz), (ey, ez) = cut.start, cut.end
    strip = _closed([(sy + ny, sz + nz), (ey + ny, ez + nz), (ey - ny, ez - nz), (sy - ny, sz - nz)], points)
    return [strip, _segment(cut.start, cut.end, points)]


def cut_rect(cut: GisaxsCut, extent) -> Optional[tuple[float, float, float, float]]:
    """``(qy0, qz0, qy1, qz1)`` of a band or box (an open band reaches the frame's ``extent``); ``None`` otherwise."""
    if cut.kind not in (HORIZONTAL, VERTICAL, BOX):
        return None
    (y0, y1), (z0, z1) = cut.qy or extent[0], cut.qz or extent[1]
    return y0, z0, y1, z1


def row_band_q(geometry: DetectorGeometry, start: float, stop: float) -> dict[str, list[float]]:
    """The q_z and α_f of a band of rows' edges ``start``, ``stop`` (continuous rows) on the beam-centre column: where
    the automatic horizontal cut lies in q (its pixels are whole rows, so off the column the q_z of an edge changes
    a little)."""
    _qy, qz = q_of_positions(geometry, np.full(2, geometry.beam_center_x_px), np.array([stop, start], dtype=float))
    alpha = alpha_f_of_qz(geometry, qz)
    return {"qz_edges": [float(value) for value in qz], "alpha_f_edges_deg": [float(value) for value in alpha]}


def column_band_q(geometry: DetectorGeometry, start: float, stop: float) -> dict[str, list[float]]:
    """The q_y and 2θ_f of a band of columns' edges on the sample horizon (α_f = 0): where the automatic vertical cut
    lies in q."""
    qy, _qz = q_of_positions(geometry, np.array([start, stop], dtype=float), np.full(2, geometry.horizon_row()))
    angle = two_theta_f_of_qy(geometry, qy)
    return {"qy_edges": [float(value) for value in qy], "two_theta_f_edges_deg": [float(value) for value in angle]}


__all__ = [
    "ALONG_QY", "ALONG_QZ", "AXIS_LABELS", "BOX", "CUT_KEY", "CUT_KINDS", "CUT_TEMPLATES", "NO_PIXELS", "GisaxsCut", "HORIZONTAL", "LINE", "SECTOR",
    "VERTICAL", "column_band_q", "cut_curve", "cut_from_record", "cut_key", "cut_mask", "cut_outlines", "cut_record",
    "cut_rect", "row_band_q",
]
