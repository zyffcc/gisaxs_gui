"""Bragg peaks of an I(q) profile: position, d-spacing, width, height and significance.

The background under the peaks is a SNIP baseline; each candidate is then
fitted with a Gaussian on a local linear background, weighted by the Poisson
errors.  A peak must rise ``min_snr`` standard errors *and* ``min_relative_height``
above the background (many pixels per bin make tiny ripples statistically
"significant").  When nothing qualifies, the search says why, with the
strongest feature it saw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .profiles import Profile, estimate_background

DEFAULT_BACKGROUND_WINDOW = 0.15
"""Å⁻¹: half width of the widest SNIP step; broader features count as background."""
MIN_POINTS = 12
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
GAUSS_AREA = math.sqrt(math.pi / (4.0 * math.log(2.0)))
"""Gaussian area = height × FWHM × GAUSS_AREA."""
SERIES = {
    "lamellar (1:2:3:4)": (1.0, 2.0, 3.0, 4.0, 5.0),
    "hexagonal (1:√3:2:√7:3)": (1.0, math.sqrt(3.0), 2.0, math.sqrt(7.0), 3.0),
}
SERIES_TOLERANCE = 0.02
BROAD_RELATIVE_WIDTH = 0.15
"""A peak wider than this fraction of its q is a halo (amorphous or liquid-like order)."""
SPIKE_HEIGHT_OVER_BACKGROUND = 5.0
"""A resolution-limited peak this far above its background is a spike, not diffraction."""


@dataclass(frozen=True)
class Peak:
    q: float
    q_err: float
    d: float
    """Lattice spacing 2π/q in Å."""
    d_err: float
    fwhm: float
    fwhm_err: float
    height: float
    """Counts per pixel above the local background."""
    area: float
    background: float
    snr: float
    reduced_chi2: Optional[float]
    flags: tuple[str, ...] = ()
    window: tuple = ()
    """``(q_low, q_high)`` of the points the Gaussian was fitted to (empty when the fit failed)."""
    slope: float = 0.0
    """Slope of the local linear background (per Å⁻¹, about the centre)."""


@dataclass(frozen=True)
class PeakSearch:
    peaks: tuple[Peak, ...]
    x_range: tuple[float, float]
    points: int
    step: float
    background_window: float
    min_snr: float
    min_relative_height: float
    strongest: Optional[dict]
    """The most significant bump seen, qualifying or not: q, snr, relative_height."""
    reason: str
    """Why no peak was reported ("" when there are peaks)."""
    series: tuple[str, ...]
    baseline_x: np.ndarray
    baseline_y: np.ndarray
    data_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    data_sigma: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """The measured profile on ``baseline_x`` (to draw a fit)."""

    def background_at(self, q: float) -> Optional[float]:
        if self.baseline_x.size == 0:
            return None
        return float(np.interp(q, self.baseline_x, self.baseline_y))


def _gaussian_line(x, height, center, fwhm, b0, b1):
    width = fwhm / FWHM_PER_SIGMA
    return height * np.exp(-0.5 * ((x - center) / width) ** 2) + b0 + b1 * (x - center)


def _local_snr(residual: np.ndarray, sigma: np.ndarray, index: int, half_points: int = 1) -> float:
    """Mean residual over ±``half_points`` bins divided by its standard error."""
    half_points = max(1, int(half_points))
    start, stop = max(0, index - half_points), min(residual.size, index + half_points + 1)
    center = float(residual[start:stop].mean())
    error = float(np.sqrt((sigma[start:stop] ** 2).sum()) / (stop - start))
    return center / error if error > 0 else 0.0


def _fit_peak(data: Profile, index: int, width_points: float, baseline: np.ndarray):
    """Gaussian + linear background around ``index``; ``None`` when the fit is not usable."""
    from scipy.optimize import curve_fit

    step = data.step()
    guess_fwhm = max(width_points * step, 2.0 * step)
    half = max(3.0 * guess_fwhm, 5.0 * step)
    x0 = float(data.x[index])
    window = (data.x >= x0 - half) & (data.x <= x0 + half)
    if int(window.sum()) < 7:
        return None
    x, y, sigma = data.x[window], data.y[window], data.sigma[window]
    start = [max(float(data.y[index] - baseline[index]), 1e-12), x0, guess_fwhm, float(baseline[index]), 0.0]
    bounds = ([0.0, x0 - half, 0.5 * step, -np.inf, -np.inf], [np.inf, x0 + half, 2.0 * half, np.inf, np.inf])
    try:
        params, covariance = curve_fit(
            _gaussian_line, x, y, p0=start, sigma=sigma, absolute_sigma=True, bounds=bounds, maxfev=5000
        )
    except (RuntimeError, ValueError):
        return None
    if not np.all(np.isfinite(params)) or not np.all(np.isfinite(covariance)):
        return None
    height, center, fwhm, b0, b1 = (float(value) for value in params)
    if not (x[0] < center < x[-1]) or fwhm >= 2.0 * half * 0.999 or height <= 0:
        return None
    dof = max(1, x.size - 5)
    chi2 = float((((y - _gaussian_line(x, *params)) / sigma) ** 2).sum() / dof)
    errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None)) * max(1.0, math.sqrt(chi2))
    return height, center, fwhm, b0, errors, chi2, b1, (float(x[0]), float(x[-1]))


def _same_peak(center: float, fwhm: float, snr: float, other: "Peak") -> bool:
    """Whether a new fit converged on ``other``: close, and not a significant sharp peak on a halo."""
    if abs(center - other.q) >= 0.5 * max(fwhm, other.fwhm):
        return False
    # Crystalline reflections sit on amorphous halos: much narrower (or wider) and significant is another peak.
    distinct_width = max(fwhm, other.fwhm) > 3.0 * min(fwhm, other.fwhm)
    return not (distinct_width and snr >= 5.0 and other.snr >= 5.0)


def caveat(peak: "Peak") -> str:
    """Why a peak should not be analysed as a crystalline reflection ("" when it can be)."""
    if "spike" in peak.flags:
        return (
            "a spike one or two bins wide far above the background: hot pixels, a module edge or a "
            "zinger rather than diffraction (check whether the calibration image shows it too)"
        )
    if "fit_failed" in peak.flags:
        return "its shape could not be fitted: look at the image at this q (module gap, detector edge, overlap)"
    if "weak" in peak.flags:
        return "weak (3–5σ): tentative, may be noise"
    if "broad" in peak.flags:
        return "a broad halo (amorphous or liquid-like order), not a crystalline reflection"
    return ""


def _series_hints(qs: list[float]) -> tuple[str, ...]:
    hints: list[str] = []
    for base in qs[:3]:
        for name, ratios in SERIES.items():
            orders = [base]
            for ratio in ratios[1:]:
                target = base * ratio
                match = [q for q in qs if abs(q - target) <= SERIES_TOLERANCE * target]
                if match:
                    orders.append(min(match, key=lambda q: abs(q - target)))
            if len(orders) >= 3:
                listed = ", ".join(f"{q:.4g}" for q in orders)
                hints.append(f"q = {listed} Å⁻¹ follow a {name} series (within 2 %) from {base:.4g} Å⁻¹")
    return tuple(dict.fromkeys(hints))


def find_peaks(
    profile: Profile,
    *,
    x_range: Optional[tuple[float, float]] = None,
    min_snr: float = 3.0,
    min_relative_height: float = 0.02,
    background_window: Optional[float] = None,
    max_peaks: int = 12,
) -> PeakSearch:
    """Peaks of ``profile`` (x = q in Å⁻¹) inside ``x_range``, sorted by q."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks as scipy_peaks

    data = profile.measured(x_range)
    window_q = float(background_window) if background_window else DEFAULT_BACKGROUND_WINDOW
    if x_range is not None:
        span = tuple(sorted(float(value) for value in x_range))
    elif data.size:
        span = (float(data.x[0]), float(data.x[-1]))
    else:
        span = (float("nan"), float("nan"))
    empty = np.zeros(0)
    if data.size < MIN_POINTS:
        return PeakSearch(
            (), span, data.size, data.step(), window_q, min_snr, min_relative_height, None,
            f"Only {data.size} measured bins in q = {span[0]:.4g}–{span[1]:.4g} Å⁻¹; "
            f"at least {MIN_POINTS} are needed.",
            (), empty, empty,
        )
    step = data.step()
    window_points = int(np.clip(round(window_q / step), 3, max(3, data.size // 3)))
    baseline = estimate_background(data.y, data.sigma, window_points)
    residual = data.y - baseline
    smooth = gaussian_filter1d(residual, 1.0)
    indices, properties = scipy_peaks(smooth, prominence=0.0, width=1.0)
    candidates, strongest = [], None
    for position, index in enumerate(indices):
        width_points = float(properties["widths"][position])
        snr = _local_snr(residual, data.sigma, int(index), round(width_points / 2))
        relative = float(smooth[index] / max(abs(baseline[index]), 1e-12))
        feature = {"q": float(data.x[index]), "snr": snr, "relative_height": relative}
        if strongest is None or snr > strongest["snr"]:
            strongest = feature
        if snr >= min_snr and relative >= min_relative_height and width_points >= 2.0:
            candidates.append((float(properties["prominences"][position]), int(index), width_points, snr))
    candidates.sort(reverse=True)
    peaks: list[Peak] = []
    for _prominence, index, width_points, candidate_snr in candidates[: int(max_peaks)]:
        fitted = _fit_peak(data, index, width_points, baseline)
        flags: list[str] = []
        if fitted is None:
            if candidate_snr < 2.0 * min_snr:
                continue  # an unfittable, marginal bump is not reported as a peak
            height, center, fwhm = float(residual[index]), float(data.x[index]), width_points * step
            background, chi2 = float(baseline[index]), None
            errors = np.full(5, np.nan)
            slope, window = 0.0, ()
            snr = candidate_snr
            flags.append("fit_failed")
        else:
            height, center, fwhm, background, errors, chi2, slope, window = fitted
            snr = height / errors[0] if errors[0] > 0 else candidate_snr
            if snr < min_snr or height < min_relative_height * max(abs(background), 1e-12):
                continue
        if any(_same_peak(center, fwhm, snr, other) for other in peaks):
            continue
        d = 2.0 * math.pi / center
        peaks.append(Peak(
            q=center, q_err=float(errors[1]), d=d, d_err=float(2.0 * math.pi * errors[1] / center ** 2),
            fwhm=fwhm, fwhm_err=float(errors[2]), height=height, area=height * fwhm * GAUSS_AREA,
            background=background, snr=float(snr), reduced_chi2=chi2, flags=tuple(flags), window=window, slope=slope,
        ))
    peaks.sort(key=lambda peak: peak.q)
    finished: list[Peak] = []
    for peak in peaks:
        flags = list(peak.flags)
        if peak.snr < 5.0:
            flags.append("weak")
        if peak.q - 1.5 * peak.fwhm < span[0] or peak.q + 1.5 * peak.fwhm > span[1]:
            flags.append("at_edge")
        if peak.fwhm < 3.0 * step:
            flags.append("resolution_limited")
            if peak.height > SPIKE_HEIGHT_OVER_BACKGROUND * max(abs(peak.background), 1e-12):
                flags.append("spike")
        if peak.fwhm > BROAD_RELATIVE_WIDTH * peak.q:
            flags.append("broad")
        if any(other is not peak and abs(peak.q - other.q) < 0.75 * (peak.fwhm + other.fwhm) for other in peaks):
            flags.append("overlap")
        finished.append(Peak(**{**peak.__dict__, "flags": tuple(flags)}))
    reason = ""
    if not finished:
        if strongest is None:
            reason = f"The profile has no local maximum in q = {span[0]:.4g}–{span[1]:.4g} Å⁻¹."
        else:
            reason = (
                f"No peak rises {min_snr:g}σ and {min_relative_height:.0%} above the background in "
                f"q = {span[0]:.4g}–{span[1]:.4g} Å⁻¹ (strongest feature: {strongest['snr']:.1f}σ, "
                f"{strongest['relative_height']:.1%} at q = {strongest['q']:.4g} Å⁻¹)."
            )
    return PeakSearch(
        tuple(finished), span, data.size, step, window_q, min_snr, min_relative_height,
        # Series from halos, spikes or failed fits would suggest an order the data do not show.
        strongest, reason, _series_hints([peak.q for peak in finished if not caveat(peak)]), data.x, baseline,
        data_y=data.y, data_sigma=data.sigma,
    )


__all__ = ["BROAD_RELATIVE_WIDTH", "DEFAULT_BACKGROUND_WINDOW", "GAUSS_AREA", "Peak", "PeakSearch", "caveat", "find_peaks"]
