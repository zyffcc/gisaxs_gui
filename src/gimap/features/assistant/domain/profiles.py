"""Reduced 1D profiles as the assistant sees them, and the helpers every metric shares.

A profile is one Analyze curve: ``x`` (q in Å⁻¹, or χ in degrees), ``y`` the
mean counts per pixel of each bin, ``sigma`` its Poisson standard error and
``pixels`` the number of pixels averaged.  Bins without pixels are not
measurements and are dropped before any metric sees them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Profile:
    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    pixels: np.ndarray

    @classmethod
    def of(cls, x, y, sigma=None, pixels=None) -> "Profile":
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        sigma = (
            np.sqrt(np.clip(np.abs(y), 1.0, None))
            if sigma is None
            else np.asarray(sigma, dtype=np.float64).reshape(-1)
        )
        pixels = np.ones_like(x) if pixels is None else np.asarray(pixels, dtype=np.float64).reshape(-1)
        if not x.size == y.size == sigma.size == pixels.size:
            raise ValueError("Profile arrays differ in length")
        return cls(x, y, sigma, pixels)

    def measured(self, x_range: tuple[float, float] | None = None) -> "Profile":
        """Finite bins with pixels and a positive error, inside ``x_range``, sorted by x."""
        keep = (
            np.isfinite(self.x) & np.isfinite(self.y) & np.isfinite(self.sigma)
            & (self.pixels > 0) & (self.sigma > 0)
        )
        if x_range is not None:
            low, high = sorted(float(value) for value in x_range)
            keep &= (self.x >= low) & (self.x <= high)
        order = np.argsort(self.x[keep], kind="stable")
        return Profile(*(values[keep][order] for values in (self.x, self.y, self.sigma, self.pixels)))

    @property
    def size(self) -> int:
        return int(self.x.size)

    def step(self) -> float:
        return float(np.median(np.diff(self.x))) if self.size > 1 else 0.0


def snip_background(y: np.ndarray, window: int) -> np.ndarray:
    """Background under peaks by SNIP clipping (decreasing window) on log-log-sqrt values.

    ``window`` is the half width in points of the widest clipping step; it should
    exceed the half width of the broadest peak that must stay a peak.
    """
    values = np.clip(np.asarray(y, dtype=np.float64), 0.0, None)
    v = np.log(np.log(np.sqrt(values + 1.0) + 1.0) + 1.0)
    window = int(max(1, min(window, (v.size - 1) // 2)))
    for p in range(window, 0, -1):
        middle = v[p:-p]
        average = 0.5 * (v[:-2 * p] + v[2 * p:])
        v[p:-p] = np.minimum(middle, average)
    return (np.exp(np.exp(v) - 1.0) - 1.0) ** 2 - 1.0


def estimate_background(y: np.ndarray, sigma: np.ndarray, window: int) -> np.ndarray:
    """Background of a noisy profile: SNIP on a smoothed copy, then lifted onto the noise.

    SNIP follows the lower edge of the noise, which would turn every noise bump
    into a "peak".  Smoothing first (2 bins) and then shifting the baseline by
    the median residual of the peak-free bins (|residual| < 3σ) removes that bias.
    """
    from scipy.ndimage import gaussian_filter1d

    y = np.asarray(y, dtype=np.float64)
    baseline = snip_background(gaussian_filter1d(y, 2.0, mode="nearest"), window)
    residual = y - baseline
    quiet = np.abs(residual) < 3.0 * np.asarray(sigma, dtype=np.float64)
    if quiet.sum() >= max(5, y.size // 5):
        baseline = baseline + float(np.median(residual[quiet]))
    return baseline


def downsample(profile: Profile, max_points: int) -> Profile:
    """At most ``max_points`` bins: neighbours averaged with their pixel counts as weights."""
    if profile.size <= max_points:
        return profile
    edges = np.linspace(0, profile.size, int(max_points) + 1).astype(int)
    xs, ys, sigmas, pixels = [], [], [], []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        w = profile.pixels[start:stop]
        total = float(w.sum())
        if total <= 0:
            continue
        xs.append(float((profile.x[start:stop] * w).sum() / total))
        ys.append(float((profile.y[start:stop] * w).sum() / total))
        sigmas.append(float(np.sqrt(((profile.sigma[start:stop] * w) ** 2).sum()) / total))
        pixels.append(total)
    return Profile(*(np.asarray(values) for values in (xs, ys, sigmas, pixels)))


def significant(value: float, digits: int = 4) -> float:
    """``value`` rounded to ``digits`` significant digits (for compact reports)."""
    value = float(value)
    if value == 0.0 or not np.isfinite(value):
        return value
    return float(f"{value:.{digits}g}")


__all__ = ["Profile", "downsample", "estimate_background", "significant", "snip_background"]
