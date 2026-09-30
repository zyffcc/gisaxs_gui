"""Automatic GISAXS reduction: Yoneda band, horizontal I(qy) and vertical I(qz) cuts.

All positions use the canonical pixel frame of
:class:`~src.gimap.shared.geometry.DetectorGeometry` (pixel ``(i, j)`` covers
``[j, j+1] x [i, i+1]``, row 0 at the top) and q comes from the exact
grazing-incidence model, in Å⁻¹.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.gimap.shared.geometry import DetectorGeometry
from src.gimap.shared.geometry.q_mapping import grazing_q_region

from .binning import binned_mean_2d, native_profile
from .models import GISAXS, Curve, GisaxsCutSettings, ReciprocalSpaceMap, Reduction, YonedaEstimate

YONEDA_SEARCH_MAX_DEG = 1.5
"""Highest exit angle searched for the Yoneda band (above any common αc)."""
HORIZON_TOLERANCE_DEG = 0.2
"""The search also extends this far below the computed horizon, because a
slightly wrong αi or beam centre moves the real horizon by tens of pixels."""
SPECULAR_EXCLUSION_PX = 15
"""Half width around the beam-centre column skipped (specular rod, beam stop)."""
SIDE_BAND_WIDTH_PX = 100
MAP_SHAPE = (600, 600)
MAP_BLOCK_ROWS = 256


@dataclass(frozen=True)
class GisaxsMaps:
    """Per-pixel qy and qz (float32) and the pixels above the sample horizon."""

    qy: np.ndarray
    qz: np.ndarray
    above_horizon: np.ndarray


def gisaxs_maps(shape: tuple[int, int], geometry: DetectorGeometry) -> GisaxsMaps:
    """Exact qy, qz of every pixel, computed in row blocks (no full float64 temporary)."""
    rows, columns = int(shape[0]), int(shape[1])
    qy = np.empty((rows, columns), dtype=np.float32)
    qz = np.empty_like(qy)
    for start in range(0, rows, MAP_BLOCK_ROWS):
        stop = min(rows, start + MAP_BLOCK_ROWS)
        block = grazing_q_region(geometry, (start, stop), (0, columns))
        qy[start:stop] = block.qy
        qz[start:stop] = block.qz
    horizon_qz = geometry.wavevector_inv_angstrom * math.sin(geometry.incidence_rad)
    return GisaxsMaps(qy, qz, qz >= np.float32(horizon_qz))


def gisaxs_q_map(image: np.ndarray, valid: np.ndarray, maps: GisaxsMaps) -> ReciprocalSpaceMap | None:
    """The intensity above the horizon on a regular qy–qz grid (mean per cell, NaN where empty)."""
    usable = np.asarray(valid, dtype=bool) & maps.above_horizon
    if not usable.any():
        return None
    qy, qz = maps.qy[usable], maps.qz[usable]
    y_range = (float(qy.min()), float(qy.max()))
    z_range = (float(qz.min()), float(qz.max()))
    grid = binned_mean_2d(
        qy, qz, np.asarray(image)[usable],
        x_range=(y_range[0], y_range[1] + 1e-12), y_range=(z_range[0], z_range[1] + 1e-12), shape=MAP_SHAPE,
    )
    return ReciprocalSpaceMap(np.flipud(grid), y_range, z_range, x_label="qy (Å⁻¹)")


def _row_bounds(center: float, half: float, rows: int) -> tuple[int, int]:
    start = int(math.floor(center - half))
    stop = int(math.ceil(center + half))
    start, stop = max(0, start), min(rows, stop)
    return start, max(start, stop)


def exit_angle_deg_at_row(geometry: DetectorGeometry, row: float) -> float:
    """Exit angle on the beam-centre column at a continuous row coordinate."""
    height_m = (geometry.beam_center_y_px - float(row)) * geometry.pixel_size_y_m
    return math.degrees(math.atan2(height_m, geometry.distance_m)) - geometry.incidence_deg


def _nan_smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values
    finite = np.isfinite(values)
    kernel = np.ones(int(width))
    sums = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    counts = np.convolve(finite.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def locate_yoneda(
    image: np.ndarray,
    valid: np.ndarray,
    geometry: DetectorGeometry,
    *,
    search_max_deg: float = YONEDA_SEARCH_MAX_DEG,
    exclusion_px: int = SPECULAR_EXCLUSION_PX,
    side_width_px: int = SIDE_BAND_WIDTH_PX,
    smooth_rows: int = 3,
) -> YonedaEstimate | None:
    """Brightest row between the sample horizon and ``search_max_deg`` exit angle.

    The profile is the per-row median of two column bands beside the specular
    rod, so the direct beam, the specular reflection, a rod-shaped beam stop
    and isolated hot pixels (module edges) do not compete with the diffuse
    Yoneda band.  The window reaches ``HORIZON_TOLERANCE_DEG`` below the
    computed horizon.  Returns ``None`` when the window lies outside the frame
    or holds no valid pixels.
    """
    rows, columns = image.shape
    center_column = int(math.floor(geometry.beam_center_x_px))
    left = (
        max(0, center_column - exclusion_px - side_width_px),
        max(0, center_column - exclusion_px),
    )
    right = (
        min(columns, center_column + 1 + exclusion_px),
        min(columns, center_column + 1 + exclusion_px + side_width_px),
    )
    band_columns = np.r_[left[0] : left[1], right[0] : right[1]]
    if band_columns.size == 0:
        return None
    try:
        top = geometry.row_for_exit_angle(search_max_deg)
    except ValueError:
        top = 0.0
    try:
        bottom = geometry.row_for_exit_angle(-HORIZON_TOLERANCE_DEG)
    except ValueError:
        bottom = geometry.horizon_row()
    start = max(0, int(math.floor(min(top, bottom))))
    stop = min(rows, int(math.ceil(max(top, bottom))))
    if stop - start < 3:
        return None
    window = image[start:stop][:, band_columns].astype(np.float64)
    window_valid = valid[start:stop][:, band_columns]
    window = np.where(window_valid, window, np.nan)
    # Median only over rows with data: no all-NaN warnings, which matters because
    # reductions run on worker threads where catch_warnings() is not thread safe.
    profile = np.full(window.shape[0], np.nan)
    has_data = window_valid.any(axis=1)
    if has_data.any():
        profile[has_data] = np.nanmedian(window[has_data], axis=1)
    smoothed = _nan_smooth(profile, smooth_rows)
    if not np.any(np.isfinite(smoothed)):
        return None
    row = start + int(np.nanargmax(smoothed)) + 0.5
    return YonedaEstimate(
        row=row,
        alpha_f_deg=exit_angle_deg_at_row(geometry, row),
        search_rows=(start, stop),
        side_columns=(left, right),
    )


def horizontal_cut(
    image: np.ndarray,
    valid: np.ndarray,
    geometry: DetectorGeometry,
    *,
    center_row: float,
    half_height_px: float,
    counts: bool = True,
) -> Curve:
    """I(qy) over a band of rows: one point per detector column (native sampling)."""
    rows, columns = image.shape
    start, stop = _row_bounds(center_row, half_height_px, rows)
    region = {"rows": (start, stop), "columns": (0, columns), "center_row": float(center_row)}
    if stop <= start:
        return _empty_curve("horizontal", "Horizontal cut I(qy)", "qy (Å⁻¹)", region)
    q = grazing_q_region(geometry, (start, stop), (0, columns))
    x, mean, sigma, pixels = native_profile(image[start:stop], valid[start:stop], q.qy, axis=0, counts_model=counts)
    if x.size == 0:
        return _empty_curve("horizontal", "Horizontal cut I(qy)", "qy (Å⁻¹)", region)
    return Curve(
        "horizontal", "Horizontal cut I(qy)", x, mean, sigma, pixels, "qy (Å⁻¹)", region=region
    )


def vertical_cut(
    image: np.ndarray,
    valid: np.ndarray,
    geometry: DetectorGeometry,
    *,
    center_column: float,
    half_width_px: float,
    counts: bool = True,
) -> Curve:
    """I(qz) over a band of columns above the sample horizon: one point per detector row."""
    rows, columns = image.shape
    start = max(0, int(math.floor(center_column - half_width_px)))
    stop = min(columns, int(math.ceil(center_column + half_width_px)))
    horizon = geometry.horizon_row()
    last_row = int(min(rows, max(0, math.floor(horizon))))
    region = {
        "rows": (0, last_row),
        "columns": (start, max(start, stop)),
        "center_column": float(center_column),
    }
    if stop <= start or last_row <= 0:
        return _empty_curve("vertical", "Vertical cut I(qz)", "qz (Å⁻¹)", region)
    q = grazing_q_region(geometry, (0, last_row), (start, stop))
    x, mean, sigma, pixels = native_profile(
        image[:last_row, start:stop], valid[:last_row, start:stop], q.qz, axis=1, counts_model=counts
    )
    if x.size == 0:
        return _empty_curve("vertical", "Vertical cut I(qz)", "qz (Å⁻¹)", region)
    return Curve(
        "vertical", "Vertical cut I(qz)", x, mean, sigma, pixels, "qz (Å⁻¹)", region=region
    )


def _empty_curve(key: str, title: str, x_label: str, region: dict) -> Curve:
    empty = np.array([], dtype=np.float64)
    return Curve(key, title, empty, empty, empty, np.array([], dtype=np.int64), x_label, region=region)


def reduce_gisaxs(
    image: np.ndarray,
    valid: np.ndarray,
    geometry: DetectorGeometry,
    settings: GisaxsCutSettings | None = None,
    *,
    q_map: ReciprocalSpaceMap | None = None,
    counts: bool = True,
) -> Reduction:
    """Yoneda-band horizontal cut and beam-centre vertical cut, no user input needed.

    ``q_map`` (``gisaxs_q_map``) is passed through: it depends on the frame, not on the cuts.
    """
    settings = settings or GisaxsCutSettings()
    warnings_out: list[str] = []
    rows, columns = image.shape
    if geometry.incidence_deg <= 0.0:
        warnings_out.append(
            "Incidence angle is 0°: the sample horizon equals the beam centre row. "
            "Set αi in the geometry for correct qz."
        )
    if not (0.0 <= geometry.beam_center_x_px <= columns):
        warnings_out.append("The beam centre column lies outside the frame.")

    yoneda = locate_yoneda(image, valid, geometry)
    if settings.horizontal_row is not None:
        horizontal_row = float(settings.horizontal_row)
        horizontal_source = "manual"
    elif yoneda is not None:
        horizontal_row = yoneda.row
        horizontal_source = "yoneda"
    else:
        horizontal_row = min(max(geometry.horizon_row() - 10.0, 0.5), rows - 0.5)
        horizontal_source = "horizon"
        warnings_out.append("No Yoneda band found; the horizontal cut sits just above the horizon.")

    vertical_column = (
        float(settings.vertical_column)
        if settings.vertical_column is not None
        else float(geometry.beam_center_x_px)
    )
    curves = (
        horizontal_cut(
            image,
            valid,
            geometry,
            center_row=horizontal_row,
            half_height_px=settings.horizontal_half_height_px,
            counts=counts,
        ),
        vertical_cut(
            image,
            valid,
            geometry,
            center_column=vertical_column,
            half_width_px=settings.vertical_half_width_px,
            counts=counts,
        ),
    )
    for curve in curves:
        if curve.is_empty:
            warnings_out.append(f"{curve.title}: no valid pixels in the cut band.")
    markers = {
        "beam_center": (geometry.beam_center_x_px, geometry.beam_center_y_px),
        "horizon_row": geometry.horizon_row(),
        "horizontal_band": (
            horizontal_row - settings.horizontal_half_height_px,
            horizontal_row + settings.horizontal_half_height_px,
        ),
        "horizontal_source": horizontal_source,
        "vertical_band": (
            vertical_column - settings.vertical_half_width_px,
            vertical_column + settings.vertical_half_width_px,
        ),
        "yoneda": yoneda,
    }
    return Reduction(GISAXS, curves, markers, tuple(warnings_out), reciprocal_space_map=q_map)


__all__ = [
    "GisaxsMaps",
    "exit_angle_deg_at_row",
    "gisaxs_maps",
    "gisaxs_q_map",
    "horizontal_cut",
    "locate_yoneda",
    "reduce_gisaxs",
    "vertical_cut",
]
