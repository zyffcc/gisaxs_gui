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
from src.gimap.shared.geometry.q_inverse import q_per_pixel
from src.gimap.shared.geometry.q_mapping import grazing_q_region

from .binning import binned_mean_2d, map_grid_shape, native_profile
from .curve_errors import PixelErrors, pixel_errors
from .gisaxs_cuts import NO_PIXELS, column_band_q, cut_curve, cut_key, row_band_q
from .models import GISAXS, Curve, GisaxsCutSettings, ReciprocalSpaceMap, Reduction, YonedaEstimate
from .vertical_band import LIMITED_CLEARANCE, SPECULAR_EXCLUSION_PX, default_vertical_column

YONEDA_SEARCH_MAX_DEG = 1.5
"""Highest exit angle searched for the Yoneda band (above any common αc)."""
HORIZON_TOLERANCE_DEG = 0.2
"""The search also extends this far below the computed horizon, because a
slightly wrong αi or beam centre moves the real horizon by tens of pixels."""
SIDE_BAND_WIDTH_PX = 100
MAP_SHAPE = (600, 600)
MAP_BLOCK_ROWS = 256
ROW_BAND_WARNING = ("The detector horizon changes by more than half a row across this frame. "
                    "The Yoneda search and automatic horizontal cut use detector rows. "
                    "Add a horizontal band in q to select a constant exit angle.")


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
    above = np.empty((rows, columns), dtype=bool) if geometry.refraction or geometry.requires_projection else None
    for start in range(0, rows, MAP_BLOCK_ROWS):
        stop = min(rows, start + MAP_BLOCK_ROWS)
        block = grazing_q_region(geometry, (start, stop), (0, columns))
        qy[start:stop] = block.qy
        qz[start:stop] = block.qz
        if above is not None:  # refraction: α_f ≥ α_c (below it the film's q_z tells no angle)
            above[start:stop] = block.sin_exit >= math.sin(geometry.critical_rad)
    if above is None:
        horizon_qz = geometry.wavevector_inv_angstrom * math.sin(geometry.incidence_rad)
        above = qz >= np.float32(horizon_qz)
    return GisaxsMaps(qy, qz, above)


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
        x_range=(y_range[0], y_range[1] + 1e-12), y_range=(z_range[0], z_range[1] + 1e-12),
        shape=map_grid_shape(maps.qy, maps.qz, usable, y_range[1] - y_range[0], z_range[1] - z_range[0], MAP_SHAPE),
    )
    return ReciprocalSpaceMap(np.flipud(grid), y_range, z_range, x_label="qy (Å⁻¹)")


def _row_bounds(center: float, half: float, rows: int) -> tuple[int, int]:
    start = int(math.floor(center - half))
    stop = int(math.ceil(center + half))
    start, stop = max(0, start), min(rows, stop)
    return start, max(start, stop)


def exit_angle_deg_at_row(geometry: DetectorGeometry, row: float) -> float:
    """Exit angle on the beam-centre column at a continuous row coordinate."""
    if geometry.requires_projection:
        return math.degrees(geometry.exit_angle_at(geometry.beam_center_x_px, float(row)))
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
    left = (min(columns, max(0, center_column - exclusion_px - side_width_px)),
            min(columns, max(0, center_column - exclusion_px)))
    right = (min(columns, max(0, center_column + 1 + exclusion_px)),
             min(columns, max(0, center_column + 1 + exclusion_px + side_width_px)))
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
    if not math.isfinite(bottom):
        return None
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
    counts: bool | np.ndarray | PixelErrors = True,
) -> Curve:
    """I(qy) over a band of rows: one point per detector column (native sampling). ``counts``: as
    ``reduce_gisaxs``."""
    rows, columns = image.shape
    start, stop = _row_bounds(center_row, half_height_px, rows)
    region = {"rows": (start, stop), "columns": (0, columns), "center_row": float(center_row)}
    if stop <= start:
        return _empty_curve("horizontal", "Horizontal cut I(qy)", "qy (Å⁻¹)", region)
    q = grazing_q_region(geometry, (start, stop), (0, columns))
    x, mean, sigma, pixels = native_profile(
        image[start:stop], valid[start:stop], q.qy, axis=0,
        errors=pixel_errors(counts).subset(slice(start, stop)),
    )
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
    counts: bool | np.ndarray | PixelErrors = True,
) -> Curve:
    """I(qz) over a band of columns above the sample horizon (with refraction, above the row of α_c): one point
    per detector row."""
    rows, columns = image.shape
    start = max(0, int(math.floor(center_column - half_width_px)))
    stop = min(columns, int(math.ceil(center_column + half_width_px)))
    first_row, last_row = 0, rows
    if not geometry.tilted:  # retain the native whole-row boundary after flipping the pixel axes
        horizon = geometry.row_for_exit_angle(geometry.critical_angle_deg)
        if geometry.orientation in (1, 2):
            first_row = min(rows, max(0, math.ceil(horizon)))
        else:
            last_row = min(rows, max(0, math.floor(horizon)))
    region = {
        "rows": (first_row, last_row),
        "columns": (start, max(start, stop)),
        "center_column": float(center_column),
    }
    if stop <= start or last_row <= first_row:
        return _empty_curve("vertical", "Vertical cut I(qz)", "qz (Å⁻¹)", region)
    band = np.s_[first_row:last_row, start:stop]
    q = grazing_q_region(geometry, (first_row, last_row), (start, stop))
    use = valid[band]
    if geometry.tilted:
        above = q.sin_exit >= math.sin(geometry.critical_rad)
        use = use & above
        occupied = np.flatnonzero(above.any(axis=1))
        region["rows"] = (int(occupied[0]), int(occupied[-1] + 1)) if occupied.size else (0, 0)
        region["per_pixel_exit_angle_min_deg"] = geometry.critical_angle_deg
    x, mean, sigma, pixels = native_profile(
        image[band], use, q.qz, axis=1, errors=pixel_errors(counts).subset(band),
    )
    if x.size == 0:
        return _empty_curve("vertical", "Vertical cut I(qz)", "qz (Å⁻¹)", region)
    return Curve(
        "vertical", "Vertical cut I(qz)", x, mean, sigma, pixels, "qz (Å⁻¹)", region=region
    )


def _empty_curve(key: str, title: str, x_label: str, region: dict) -> Curve:
    empty = np.array([], dtype=np.float64)
    return Curve(key, title, empty, empty, empty, np.array([], dtype=np.int64), x_label, region=region)


def _with_q_edges(curves, geometry: DetectorGeometry) -> tuple[Curve, ...]:
    """The automatic cuts with where their bands lie in q and in exit angle (``row_band_q``, ``column_band_q``), for
    the record of an export: the pixels and the numbers stay as they are."""
    horizontal, vertical = curves
    rows, columns = horizontal.region["rows"], vertical.region["columns"]
    extra_h = row_band_q(geometry, *rows) if rows[1] > rows[0] else {}
    extra_v = column_band_q(geometry, *columns) if columns[1] > columns[0] else {}
    return (
        Curve(horizontal.key, horizontal.title, horizontal.x, horizontal.intensity, horizontal.sigma, horizontal.pixels,
              horizontal.x_label, horizontal.y_label, {**horizontal.region, **extra_h}),
        Curve(vertical.key, vertical.title, vertical.x, vertical.intensity, vertical.sigma, vertical.pixels,
              vertical.x_label, vertical.y_label, {**vertical.region, **extra_v}),
    )


def reduce_gisaxs(
    image: np.ndarray,
    valid: np.ndarray,
    geometry: DetectorGeometry,
    settings: GisaxsCutSettings | None = None,
    *,
    q_map: ReciprocalSpaceMap | None = None,
    counts: bool | np.ndarray | PixelErrors = True,
    maps: GisaxsMaps | None = None,
) -> Reduction:
    """Yoneda-band horizontal cut and off-rod vertical cut, then the person's cuts
    (``settings.regions``, ``gisaxs_cuts``; they need the frame's ``maps``), keys ``cut1``, ``cut2`` …

    ``counts`` takes the frame's PixelErrors, a counting flag or a per-pixel
    scale for Poisson counts. The same error model applies to every cut.

    ``q_map`` (``gisaxs_q_map``) is passed through: it depends on the frame, not on the cuts.
    """
    settings = settings or GisaxsCutSettings()
    warnings_out: list[str] = []
    rows, columns = image.shape
    if geometry.incidence_deg <= 0.0:
        warnings_out.append("Set αi in the geometry for correct qz." if geometry.requires_projection else
            "Incidence angle is 0°: the sample horizon equals the beam centre row. "
            "Set αi in the geometry for correct qz."
        )
    if not (0.0 <= geometry.beam_center_x_px <= columns):
        warnings_out.append("The beam centre column lies outside the frame.")

    horizon = geometry.horizon_row()
    if geometry.requires_projection:
        a, b, _ = geometry.horizon_plane()
        if abs(a) * columns > abs(b) * 0.5:
            warnings_out.append(ROW_BAND_WARNING)
    yoneda = locate_yoneda(image, valid, geometry)
    if settings.horizontal_row is not None:
        horizontal_row = float(settings.horizontal_row)
        horizontal_source = "manual"
    elif yoneda is not None:
        horizontal_row = yoneda.row
        horizontal_source = "yoneda"
    else:
        if math.isfinite(horizon):
            above_sign = 1 if geometry.horizon_plane()[1] > 0 else -1
            horizontal_row = min(max(horizon + above_sign * 10.0, 0.5), rows - 0.5)
        else:
            horizontal_row = rows / 2
        horizontal_source = "horizon"
        warnings_out.append("No Yoneda band found; the horizontal cut sits just above the horizon." if math.isfinite(horizon)
                            else "The horizon has no unique row at the beam column. The horizontal cut uses the middle row.")

    if settings.vertical_column is None:
        vertical_column, clearance = default_vertical_column(geometry, image.shape, settings.vertical_half_width_px)
        if not clearance:
            warnings_out.append(LIMITED_CLEARANCE)
    else:
        vertical_column = float(settings.vertical_column)
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
    curves = _with_q_edges(curves, geometry)
    for curve in curves:
        if curve.is_empty:
            warnings_out.append(f"{curve.title}: no valid pixels in the cut band.")
    if settings.regions:
        if maps is None:
            maps = gisaxs_maps(image.shape, geometry)
        usable = np.asarray(valid, dtype=bool) & maps.above_horizon
        step = q_per_pixel(geometry)
        for index, cut in enumerate(settings.regions):
            curve = cut_curve(cut, cut_key(index), maps, image, usable, step=step, counts=counts, geometry=geometry)
            if curve.is_empty:
                warnings_out.append(NO_PIXELS.format(name=cut.name))
            curves = (*curves, curve)
    markers = {
        "beam_center": (geometry.beam_center_x_px, geometry.beam_center_y_px),
        "horizon_row": horizon if math.isfinite(horizon) else None,
        "horizontal_band": (
            horizontal_row - settings.horizontal_half_height_px,
            horizontal_row + settings.horizontal_half_height_px,
        ),
        "horizontal_source": horizontal_source,
        "vertical_source": "automatic" if settings.vertical_column is None else "manual",
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
