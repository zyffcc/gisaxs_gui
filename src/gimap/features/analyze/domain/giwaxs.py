"""Automatic GIWAXS reduction: I(q), in-/out-of-plane sectors, I(χ) and a q∥–qz map.

Polar angle convention: ``χ = atan2(q∥, qz)`` in degrees, so χ = 0° points
along the surface normal (out of plane) and χ = ±90° lies in the sample
plane; the sign follows q∥ (and therefore qy).  Pixels below the sample
horizon (αf < 0, the substrate shadow) are excluded from every profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.gimap.shared.geometry import DetectorGeometry
from src.gimap.shared.geometry.q_mapping import grazing_q_region

from .binning import BinnedMean, binned_mean_2d, counts_and_scale, merge_sparse_bins
from .regions import region_curves, region_key
from .models import (
    GIWAXS,
    X_AXIS_TWO_THETA,
    Curve,
    GiwaxsSettings,
    QBox,
    ReciprocalSpaceMap,
    Reduction,
    Sector,
)

BLOCK_ROWS = 256
MAX_RADIAL_BINS = 3000
CHI_BIN_DEG = 1.0
MAP_SHAPE = (800, 800)


@dataclass(frozen=True)
class GiwaxsMaps:
    """Per-pixel q quantities in float32 (the frame can have > 10⁷ pixels)."""

    q: np.ndarray
    q_parallel: np.ndarray
    qz: np.ndarray
    chi_deg: np.ndarray
    above_horizon: np.ndarray
    q_step: float = 0.0
    """q spanned by one pixel where the pixels are coarsest (99th percentile of |∇q|, Å⁻¹)."""


def giwaxs_maps(shape: tuple[int, int], geometry: DetectorGeometry) -> GiwaxsMaps:
    """Exact q maps computed in row blocks so no full float64 temporary exists."""
    rows, columns = int(shape[0]), int(shape[1])
    q = np.empty((rows, columns), dtype=np.float32)
    q_parallel = np.empty_like(q)
    qz = np.empty_like(q)
    chi = np.empty_like(q)
    # αf >= 0  <=>  qz >= k·sin(αi) in the exact model.
    horizon_qz = geometry.wavevector_inv_angstrom * math.sin(geometry.incidence_rad)
    for start in range(0, rows, BLOCK_ROWS):
        stop = min(rows, start + BLOCK_ROWS)
        block = grazing_q_region(geometry, (start, stop), (0, columns))
        q[start:stop] = block.q
        q_parallel[start:stop] = block.q_parallel
        qz[start:stop] = block.qz
        chi[start:stop] = np.degrees(np.arctan2(block.q_parallel, block.qz))
    return GiwaxsMaps(q, q_parallel, qz, chi, qz >= np.float32(horizon_qz), _pixel_q_step(q))


def _pixel_q_step(q: np.ndarray, stride: int = 4) -> float:
    """The largest q change from one pixel to the next (99th percentile, on every ``stride``-th pixel)."""
    sample = np.asarray(q[::stride, ::stride], dtype=np.float64)
    if sample.shape[0] < 2 or sample.shape[1] < 2:
        return 0.0
    rows, columns = np.gradient(sample)
    step = np.hypot(rows, columns) / stride
    finite = step[np.isfinite(step)]
    return float(np.percentile(finite, 99)) if finite.size else 0.0


def _radial_bins(maps: GiwaxsMaps, usable: np.ndarray, q_low: float, q_high: float) -> int:
    """No bin narrower than a pixel: bins finer than the detector's q sampling alternate between more
    and fewer pixels (a comb of noise). Without a pixel step, one bin per pixel of the diagonal."""
    if maps.q_step > 0:
        return int(min(MAX_RADIAL_BINS, max(100, math.ceil((q_high - q_low) / maps.q_step))))
    rows, columns = usable.shape
    return int(min(MAX_RADIAL_BINS, max(100, math.hypot(rows, columns))))


def two_theta_deg(q: np.ndarray, wavelength_angstrom: float) -> np.ndarray:
    """Scattering angle 2θ (degrees) of momentum transfer q (Å⁻¹): q = 4π sin θ / λ."""
    ratio = np.clip(np.asarray(q, dtype=np.float64) * wavelength_angstrom / (4.0 * math.pi), -1.0, 1.0)
    return np.degrees(2.0 * np.arcsin(ratio))


def q_from_two_theta(two_theta: np.ndarray, wavelength_angstrom: float) -> np.ndarray:
    return 4.0 * math.pi * np.sin(np.radians(np.asarray(two_theta, dtype=np.float64)) / 2.0) / wavelength_angstrom


def _in_two_theta(curve: Curve, wavelength_angstrom: float) -> Curve:
    return Curve(
        curve.key, curve.title.replace("I(q)", "I(2θ)"), two_theta_deg(curve.x, wavelength_angstrom),
        curve.intensity, curve.sigma, curve.pixels, "2θ (°)", curve.y_label, dict(curve.region),
    )


def _sector_curves(sector: Sector, maps, values, usable, *, low, high, bins, counts: bool = True) -> list[Curve]:
    chi_low, chi_high = sorted((float(sector.chi_min_deg), float(sector.chi_max_deg)))
    in_chi = usable & (maps.chi_deg >= chi_low) & (maps.chi_deg <= chi_high)
    q_low = low if sector.q_min is None else max(low, float(sector.q_min))
    q_high = high if sector.q_max is None else min(high, float(sector.q_max))
    region = {"chi_deg": (chi_low, chi_high), "q": (q_low, q_high)}
    curves = [
        _profile(
            "sector", f"Sector I(q), χ = {chi_low:g}…{chi_high:g}°", "q (Å⁻¹)", maps.q, values,
            in_chi & (maps.q >= q_low) & (maps.q <= q_high),
            low=q_low, high=max(q_high, q_low + 1e-9), bins=bins, region=region, counts=counts,
        )
    ]
    if sector.q_min is not None or sector.q_max is not None:
        in_q = usable & (maps.q >= q_low) & (maps.q <= q_high)
        curves.append(
            _profile(
                "sector_chi", f"Sector I(χ), q = {q_low:.3f}–{q_high:.3f} Å⁻¹", "χ (°)",
                maps.chi_deg, values, in_q & (maps.chi_deg >= chi_low) & (maps.chi_deg <= chi_high),
                low=chi_low, high=max(chi_high, chi_low + 1e-9),
                bins=max(1, int(round((chi_high - chi_low) / CHI_BIN_DEG))), region=region, merge=False, counts=counts,
            )
        )
    return curves


def _box_curves(box: QBox, maps, values, usable, *, counts: bool = True) -> list[Curve]:
    par_low, par_high = sorted(float(value) for value in box.q_parallel)
    z_low, z_high = sorted(float(value) for value in box.qz)
    inside = (
        usable
        & (maps.q_parallel >= par_low) & (maps.q_parallel <= par_high)
        & (maps.qz >= z_low) & (maps.qz <= z_high)
    )
    region = {"q_parallel": (par_low, par_high), "qz": (z_low, z_high)}
    bins = max(20, min(MAX_RADIAL_BINS, int(math.sqrt(max(1, int(inside.sum()))))))
    q_inside = maps.q[inside]
    q_low, q_high = (float(q_inside.min()), float(q_inside.max())) if q_inside.size else (0.0, 1.0)
    return [
        _profile(
            "box_q", "q box I(q)", "q (Å⁻¹)", maps.q, values, inside,
            low=q_low, high=max(q_high, q_low + 1e-9), bins=bins, region=region, counts=counts,
        ),
        _profile(
            "box_qz", f"Box I(qz), q∥ = {par_low:.3f}–{par_high:.3f} Å⁻¹", "qz (Å⁻¹)",
            maps.qz, values, inside, low=z_low, high=max(z_high, z_low + 1e-9), bins=bins, region=region,
            counts=counts,
        ),
        _profile(
            "box_qpar", f"Box I(q∥), qz = {z_low:.3f}–{z_high:.3f} Å⁻¹", "q∥ (Å⁻¹)",
            maps.q_parallel, values, inside,
            low=par_low, high=max(par_high, par_low + 1e-9), bins=bins, region=region, counts=counts,
        ),
    ]


def _profile(
    key, title, x_label, x_values, image, mask, *, low, high, bins, region, merge: bool = True, counts: bool = True,
) -> Curve:
    """Mean per bin; sparse bins joined (``merge_sparse_bins``) except for I(χ), whose 1° bins the
    orientation analysis relies on (an empty or thin χ bin is how the missing wedge shows)."""
    accumulator = BinnedMean.linear(low, high, bins)
    poisson, scale = counts_and_scale(counts, mask)
    accumulator.add(x_values[mask], image[mask], scale)
    result = accumulator.result(counts=poisson)
    x, mean, sigma, pixels = merge_sparse_bins(*result) if merge else result
    return Curve(key, title, x, mean, sigma, pixels, x_label, region=region)


def strongest_ring(curve: Curve, *, skip_fraction: float = 0.05) -> tuple[float, float] | None:
    """q window ``(low, high)`` around the most prominent peak of I(q).

    The low-q end (direct beam tail) is skipped; prominence is measured on
    log intensity against a running median so a sloping background does not
    win, and one-bin spikes (hot pixels) are removed first and peaks narrower
    than three bins ignored.  The window is ±max(3 bins, 1.5 % of q).
    """
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks

    if curve.is_empty or len(curve.x) < 16:
        return None
    x, y = curve.x, curve.intensity
    start = int(len(x) * skip_fraction)
    x, y = x[start:], y[start:]
    positive = y > 0
    if positive.sum() < 16:
        return None
    x, log_y = x[positive], np.log(y[positive])
    despiked = median_filter(log_y, size=3, mode="nearest")
    baseline = median_filter(despiked, size=max(5, len(log_y) // 10) | 1, mode="nearest")
    residual = despiked - baseline
    peaks, properties = find_peaks(residual, prominence=0.05, width=3)
    if peaks.size == 0:
        return None
    best = peaks[int(np.argmax(properties["prominences"]))]
    step = float(np.median(np.diff(x))) if len(x) > 1 else 0.0
    half = max(3.0 * step, 0.015 * float(x[best]))
    return float(x[best] - half), float(x[best] + half)


def reduce_giwaxs(
    image: np.ndarray,
    valid: np.ndarray,
    geometry: DetectorGeometry,
    settings: GiwaxsSettings | None = None,
    *,
    maps: GiwaxsMaps | None = None,
    with_map: bool = True,
    counts: bool = True,
) -> Reduction:
    """The GIWAXS curves; the q∥–qz map too unless ``with_map`` is ``False``.

    ``counts``: the frame holds photon counts (Poisson errors); ``False`` for dark- or
    background-subtracted frames, whose errors come from the scatter of the pixels in each bin; an
    array: photon counts multiplied per pixel by it (intensity corrections, ``intensity.py``), so a
    pixel's Poisson variance is that factor times its value.
    """
    settings = settings or GiwaxsSettings()
    maps = maps or giwaxs_maps(image.shape, geometry)
    usable = valid & maps.above_horizon
    warnings_out: list[str] = []
    if not usable.any():
        return Reduction(GIWAXS, (), {}, ("No valid pixels above the sample horizon.",))
    values = image
    q_low = float(maps.q[usable].min())
    q_high = float(maps.q[usable].max())
    bins = int(settings.bins) if settings.bins else _radial_bins(maps, usable, q_low, q_high)
    radial = _profile(
        "radial", "I(q)", "q (Å⁻¹)", maps.q, values, usable,
        low=q_low, high=q_high, bins=bins, region={"sector": "full"}, counts=counts,
    )
    absolute_chi = np.abs(maps.chi_deg)
    in_plane_limit = 90.0 - float(settings.in_plane_half_width_deg)
    out_limit = float(settings.out_of_plane_half_width_deg)
    in_plane = _profile(
        "in_plane", "In-plane sector I(q)", "q (Å⁻¹)", maps.q, values,
        usable & (absolute_chi >= in_plane_limit),
        low=q_low, high=q_high, bins=bins, region={"chi_deg": (in_plane_limit, 90.0)}, counts=counts,
    )
    out_of_plane = _profile(
        "out_of_plane", "Out-of-plane sector I(q)", "q (Å⁻¹)", maps.q, values,
        usable & (absolute_chi <= out_limit),
        low=q_low, high=q_high, bins=bins, region={"chi_deg": (0.0, out_limit)}, counts=counts,
    )
    window = settings.chi_q_window or strongest_ring(radial)
    curves = [radial, in_plane, out_of_plane]
    if window is None:
        warnings_out.append("No diffraction ring found for I(χ); drag a q window to choose one.")
    else:
        low, high = sorted(float(value) for value in window)
        ring = usable & (maps.q >= low) & (maps.q <= high)
        curves.append(
            _profile(
                "azimuthal", f"I(χ), q = {low:.3f}–{high:.3f} Å⁻¹", "χ (°)", maps.chi_deg, values,
                ring, low=-90.0, high=90.0, bins=int(round(180.0 / CHI_BIN_DEG)),
                region={"q_window": (low, high)}, merge=False, counts=counts,
            )
        )
    if settings.sector is not None:
        curves.extend(_sector_curves(settings.sector, maps, values, usable, low=q_low, high=q_high, bins=bins, counts=counts))
    if settings.box is not None:
        curves.extend(_box_curves(settings.box, maps, values, usable, counts=counts))
    for index, region in enumerate(settings.regions):
        curves.extend(region_curves(
            region, region_key(index), maps, values, usable, q_low=q_low, q_high=q_high, bins=bins, counts=counts,
        ))
    for curve in curves:
        if curve.is_empty:
            warnings_out.append(f"{curve.title}: no valid pixels in this sector.")
    if settings.x_axis == X_AXIS_TWO_THETA:
        radial_keys = {"radial", "in_plane", "out_of_plane", "sector", "box_q"} | {
            region_key(index) for index in range(len(settings.regions))
        }
        curves = [
            _in_two_theta(curve, geometry.wavelength_angstrom) if curve.key in radial_keys else curve
            for curve in curves
        ]
    rsm = None
    if with_map:
        q_par = maps.q_parallel[usable]
        qz = maps.qz[usable]
        grid = binned_mean_2d(
            q_par, qz, image[usable],
            x_range=(float(q_par.min()), float(q_par.max()) + 1e-12),
            y_range=(float(qz.min()), float(qz.max()) + 1e-12),
            shape=MAP_SHAPE,
        )
        rsm = ReciprocalSpaceMap(
            image=np.flipud(grid),
            q_parallel_range=(float(q_par.min()), float(q_par.max())),
            qz_range=(float(qz.min()), float(qz.max())),
        )
    markers = {
        "beam_center": (geometry.beam_center_x_px, geometry.beam_center_y_px),
        "horizon_row": geometry.horizon_row(),
        "chi_q_window": window,
    }
    return Reduction(GIWAXS, tuple(curves), markers, tuple(warnings_out), rsm)


__all__ = [
    "GiwaxsMaps",
    "giwaxs_maps",
    "q_from_two_theta",
    "reduce_giwaxs",
    "strongest_ring",
    "two_theta_deg",
]
