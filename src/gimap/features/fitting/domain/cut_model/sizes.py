"""Size distributions of one population: quadrature nodes and number weights, and their moments.

The original three laws (saved fits and Predict):

* ``number_gauss``: a Gaussian of the particle count in R, mean ``mean`` and standard deviation ``rel · mean``,
  cut at max(0, mean(1 − 4 rel)) and mean(1 + 4 rel) (fit1d's sizes and those of ``tools/benchmarks/yoneda_truth``).
  The cut is part of the definition: weighted by V² ∝ R⁶ the upper cut removes about 1.5 % of the intensity at
  rel = 0.35 and more above.
* ``lognormal``: a log-normal of the particle count whose (untruncated) mean is ``mean`` and relative standard
  deviation ``rel``: σ_ln² = ln(1 + rel²), median = mean / √(1 + rel²); in ln R from 4 σ_ln below its median to
  4 σ_ln above the median of its V²-weighted form (``_lognormal_span``), so effectively not truncated.
* ``intensity_gauss``: a Gaussian of the scattered intensity at q = 0 in R (each node weighted by count × V²), the
  sizes of GIMaP's earlier Fitting for spheres and random cylinders; by number it is that Gaussian / V².

``mean`` and ``rel`` are the parameters of the law, not the moments of the truncated distribution; ``moments``
gives those, by number and by V² (intensity).

The rule is composite Gauss–Legendre with ``PANEL_NODES`` nodes per panel. A form factor squared oscillates in R
with period π/q, so the number of panels follows the number of such periods across the distribution at the
largest q (at least ``MIN_PANELS``): the quadrature converges with q instead of aliasing, as a fixed rectangle rule
does at large q·σ_R.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import manual_sizes

LAWS = ("number_gauss", "lognormal", "intensity_gauss", *manual_sizes.LAWS)
SPAN = 4.0
PANEL_NODES = 8
MIN_PANELS = 2
MAX_PANELS = 4096
PANELS_PER_PERIOD = 1.0
"""Panels per period π/q of |F|² across the distribution (8 nodes each): ≤ 1e-6 relative error in the tests."""
FLOOR = 1e-9
"""Smallest node, in units of the mean: a Gaussian wider than 1/4 is cut at 0, where V = 0."""


@lru_cache(maxsize=64)
def _legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(int(n))
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


def panel_counts(q_max: float, extent) -> np.ndarray:
    """Panels for distributions spanning ``extent`` (nm, one per row) up to ``q_max``, rounded up to a power of 2 so
    that a batch falls into few groups of equal rules."""
    periods = np.abs(q_max) * np.atleast_1d(np.asarray(extent, float)) / math.pi
    need = np.clip(np.ceil(PANELS_PER_PERIOD * periods), MIN_PANELS, MAX_PANELS)
    return np.minimum(2 ** np.ceil(np.log2(need)), MAX_PANELS).astype(int)


def panel_count(q_max: float, extent) -> int:
    """Panels for a distribution spanning ``extent`` (nm, scalar or array: the largest is used) up to ``q_max``."""
    return int(np.max(panel_counts(q_max, extent)))


def _composite(lo: np.ndarray, hi: np.ndarray, panels: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes (B, n) and plain quadrature weights (B, n) of a composite Gauss–Legendre rule on [lo, hi] per row."""
    x, w = _legendre(PANEL_NODES)
    edges = lo[:, None] + (hi - lo)[:, None] * np.linspace(0.0, 1.0, panels + 1)[None, :]
    half = 0.5 * (edges[:, 1:] - edges[:, :-1])
    nodes = (edges[:, :-1, None] + half[:, :, None] * (x[None, None, :] + 1.0)).reshape(lo.size, -1)
    weights = (half[:, :, None] * w[None, None, :]).reshape(lo.size, -1)
    return nodes, weights


def _arrays(mean, rel) -> tuple[np.ndarray, np.ndarray]:
    mean = np.atleast_1d(np.asarray(mean, float))
    rel = np.broadcast_to(np.atleast_1d(np.asarray(rel, float)), mean.shape).astype(float)
    if np.any(mean <= 0) or np.any(rel < 0):
        raise ValueError("A size distribution needs a mean > 0 and a relative width >= 0")
    return mean, rel


def extent(law: str, mean, rel, power: float = 6.0) -> np.ndarray:
    """Width (nm) of the truncated distribution of every row (``power`` as in ``nodes``)."""
    mean, rel = _arrays(mean, rel)
    if law == "lognormal_median":
        lo, hi = manual_sizes.lognormal_span(mean, rel, power)
        return np.exp(hi) - np.exp(lo)
    if law == "schulz":
        lo, hi = manual_sizes.gamma_span(mean, rel, power)
        return hi - lo
    if law == "lognormal":
        lo, hi = _lognormal_span(mean, rel, power)
        return np.exp(hi) - np.exp(lo)
    return mean * (1.0 + SPAN * rel) - mean * np.maximum(1.0 - SPAN * rel, 0.0)


def upper(law: str, mean, rel, power: float = 6.0) -> np.ndarray:
    """The largest size (nm) of the truncated distribution of every row."""
    mean, rel = _arrays(mean, rel)
    if law == "lognormal_median":
        return np.exp(manual_sizes.lognormal_span(mean, rel, power)[1])
    if law == "schulz":
        return manual_sizes.gamma_span(mean, rel, power)[1]
    if law == "lognormal":
        return np.exp(_lognormal_span(mean, rel, power)[1])
    return mean * (1.0 + SPAN * rel)


def characteristic(law: str, mean, rel, power: float = 6.0) -> np.ndarray:
    """A size (nm) above which a row's intensity-weighted distribution has little weight: the upper cut of a Gaussian,
    and two widths above the median of a log-normal weighted by V² ∝ R^power (the scale of the oscillations that
    survive the size average, for the grids of a random cylinder)."""
    mean, rel = _arrays(mean, rel)
    if law == "lognormal_median":
        return mean * np.exp(power * rel**2 + 2.0 * rel)
    if law == "schulz":
        factor = 1.0 + power * rel**2
        return mean * (factor + 3.0 * rel * np.sqrt(factor))
    if law == "lognormal":
        s = np.sqrt(np.log1p(rel**2))
        return np.exp(np.log(mean) - 0.5 * s**2 + power * s**2 + 2.0 * s)
    return mean * (1.0 + SPAN * rel)


def _lognormal_span(mean, rel, power):
    """ln R from 4 σ_ln below the median by number to 4 σ_ln above the median weighted by V² ∝ R^power (that one
    is log-normal too, its median times exp(power σ_ln²)): the largest particles, which dominate the intensity of a
    broad distribution, are not cut off."""
    s = np.sqrt(np.log1p(rel**2))
    centre = np.log(mean) - 0.5 * s**2
    return centre - SPAN * s, centre + power * s**2 + SPAN * s


def nodes(law: str, mean, rel, *, power: float, panels: int) -> tuple[np.ndarray, np.ndarray]:
    """Radii (B, n) and number weights (B, n), each row summing to 1.

    ``power``: V² ∝ R^power at the other sizes fixed (6 for a sphere, 4 for a cylinder's radius, 2 for its height);
    used by ``intensity_gauss`` only, to turn its intensity weights into number weights. A width of 0 gives every
    node the mean (one size)."""
    if law not in LAWS:
        raise ValueError(f"Unknown size law {law!r} (one of {', '.join(LAWS)})")
    mean, rel = _arrays(mean, rel)
    narrow = rel <= 1e-12
    safe = np.where(narrow, 1.0, rel)
    if law == "lognormal_median":
        u, w = _composite(*manual_sizes.lognormal_span(mean, safe, power), max(panels, 8))
        radii = np.exp(u)
        density = np.exp(-0.5 * ((u - np.log(mean)[:, None]) / safe[:, None]) ** 2)
    elif law == "schulz":
        radii, w = _composite(*manual_sizes.gamma_span(mean, safe, power), max(panels, 8))
        density = manual_sizes.gamma_weights(radii, mean, safe)
    elif law == "lognormal":
        s = np.sqrt(np.log1p(safe**2))
        centre = np.log(mean) - 0.5 * s**2
        u, w = _composite(*_lognormal_span(mean, safe, power), panels)
        radii = np.exp(u)
        density = np.exp(-0.5 * ((u - centre[:, None]) / s[:, None]) ** 2)
    else:
        lo = mean * np.maximum(1.0 - SPAN * safe, FLOOR)
        radii, w = _composite(lo, mean * (1.0 + SPAN * safe), panels)
        density = np.exp(-0.5 * ((radii - mean[:, None]) / (safe * mean)[:, None]) ** 2)
    weights = w * density
    if law == "intensity_gauss":
        # by number: intensity / V²; scaled by the mean first so that small nodes do not overflow
        weights = weights / (radii / mean[:, None]) ** power
    weights = weights / weights.sum(axis=1, keepdims=True)
    if np.any(narrow):
        radii = np.where(narrow[:, None], mean[:, None], radii)
        weights = np.where(narrow[:, None], 1.0 / radii.shape[1], weights)
    return radii, weights


def moments(radii: np.ndarray, weights: np.ndarray, power: float) -> dict:
    """Mean and relative standard deviation of the radius by number and by V² (∝ R^power) of one row each."""
    out = {}
    for name, w in (("number", weights), ("intensity", weights * radii**power)):
        w = w / w.sum(axis=-1, keepdims=True)
        m = np.sum(w * radii, axis=-1)
        sd = np.sqrt(np.maximum(np.sum(w * (radii - m[..., None]) ** 2, axis=-1), 0.0))
        out[name] = (m, sd / m)
    return out


TAIL_SHARE = 0.1
"""By number, an intensity Gaussian is reported only when its count peaks inside the grid and its smallest node
holds less than this share of the largest (else its mean is set by where the grid is cut: σ_R/R > 1/√24 ≈ 0.2 for
spheres)."""


def size_report(law: str, mean: float, rel: float, power: float) -> dict:
    """The moments of one population's sizes, by number and by intensity, on a fine fixed rule.

    ``number`` is ``None`` when the number distribution is not defined (``TAIL_SHARE``)."""
    if law in ("lognormal_median", "schulz"):
        return manual_sizes.moments(law, mean, rel, power)
    radii, weights = nodes(law, mean, rel, power=power, panels=64)
    m = moments(radii, weights, power)
    number = (float(m["number"][0][0]), float(m["number"][1][0]))
    if law == "intensity_gauss" and rel > 0:
        w = weights[0] / weights[0].max()
        if int(np.argmax(w)) == 0 or w[0] >= TAIL_SHARE:
            number = None
    return {"number": number, "intensity": (float(m["intensity"][0][0]), float(m["intensity"][1][0]))}


def lognormal_moments(mean: float, rel: float, power: float) -> dict:
    """Exact moments of the untruncated log-normal: weighted by R^p it is log-normal with the same width and its
    median times exp(p σ_ln²), so the V²-weighted mean is mean·(1 + rel²)^p."""
    return {"number": (mean, rel), "intensity": (mean * (1.0 + rel**2) ** power, rel)}


__all__ = ["LAWS", "SPAN", "extent", "lognormal_moments", "moments", "nodes", "panel_count", "panel_counts",
           "characteristic", "size_report", "upper"]
