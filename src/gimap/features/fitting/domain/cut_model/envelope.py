"""Size averages at large q·σ: each form factor squared replaced by its average over one oscillation.

A form factor squared oscillates in R with period π/q. Averaged over a size distribution whose weight varies on a scale
σ (nm), the oscillating part is damped by about e^{−2(qσ)²}; from qσ ≥ ``DAMPING`` = 6 on it is below 10⁻³⁰ of the rest,
and the average of |F|² is the average of its envelope, which needs no node per oscillation. Between q_cut = 6/σ and
``BLEND`` · q_cut the average goes over smoothly (a smoothstep in q) from the resolved one to the envelope's, so the
model stays a smooth function of R and σ; the quadrature only has to resolve ``BLEND`` · q_cut, not the largest q of the
curve. The envelopes are exact functions:

* sphere, f(x) = 3 j₁(x)/x with j₁ = Re h₁ (h₁ = j₁ + i y₁, the spherical Hankel function):
  f(a) f(b) → (9/(2ab)) Re[h₁(a) h₁(b)*]; for a = b, 9 (1 + x²)/(2x⁶). The pair form keeps the slow phase between two
  DWBA paths of a sphere;
* disc, (2J₁(x)/x)² → 2 (J₁(x)² + Y₁(x)²)/x²;
* height factor, (sin(x/2)/(x/2))² → 2/x².

A node keeps its exact form factor below x = ``X_MIN`` (less than one oscillation). The coherent amplitude <VA> of the
decoupling approximation is damped by e^{−(qσ)²/2} ≤ e^{−18} there and fades to 0 with the same step. What the
envelope leaves out is the ringing of the hard cut of the distribution (a Gaussian by number at R(1 + 4σR/R), weighted by
R⁶): an artefact of the cut, below 10⁻³ of the curve for σR/R ≤ 0.35 and up to 2 · 10⁻² at σR/R = 0.5
(``tests/test_cut_model.py`` compares with the fully resolved average). A Gaussian by intensity keeps much weight at
small R, where a node's exact form factor and its envelope differ; it is always fully resolved (no q_cut).
"""

from __future__ import annotations

import numpy as np
from scipy.special import j1, y1

DAMPING = 6.0
BLEND = 2.0
X_MIN = 6.0


def damping_scale(law: str, mean, rel, power: float) -> np.ndarray:
    """σ (nm) of every row: the scale over which its intensity-weighted size distribution varies."""
    mean = np.atleast_1d(np.asarray(mean, float))
    rel = np.broadcast_to(np.atleast_1d(np.asarray(rel, float)), mean.shape)
    if law == "lognormal":
        s = np.sqrt(np.log1p(rel**2))
        median_by_intensity = mean / np.sqrt(1.0 + rel**2) * np.exp(power * s**2)
        return median_by_intensity * np.exp(-3.0 * s) * s  # where the intensity weight starts to matter
    return mean * rel


def q_cut(law: str, mean, rel, power: float) -> np.ndarray:
    """The q above which a row's size average goes over to its envelope's (∞ for one size and for sizes by
    intensity)."""
    if law == "intensity_gauss":
        return np.full(np.atleast_1d(np.asarray(mean, float)).shape, np.inf)
    scale = damping_scale(law, mean, rel, power)
    with np.errstate(divide="ignore"):
        return np.where(scale > 0, DAMPING / np.maximum(scale, 1e-300), np.inf)


def blend(q, cut):
    """0 below ``cut``, 1 above ``BLEND`` · ``cut``, a smoothstep 3t² − 2t³ between (rows of ``cut``, columns of q)."""
    q = np.asarray(q, float)[None, :]
    cut = np.asarray(cut, float)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.clip((q - cut) / ((BLEND - 1.0) * cut), 0.0, 1.0)
    t = np.where(np.isfinite(cut), t, 0.0)
    return t * t * (3.0 - 2.0 * t)


def sphere_hankel(x):
    """h₁(x) = j₁(x) + i y₁(x) = (sin x/x² − cos x/x) − i (cos x/x² + sin x/x)."""
    x = np.asarray(x, float)
    s, c = np.sin(x), np.cos(x)
    return (s / x**2 - c / x) - 1j * (c / x**2 + s / x)


def sphere_pair(a, b):
    """The envelope of f(a) f(b), f(x) = 3 j₁(x)/x."""
    return 4.5 * np.real(sphere_hankel(a) * np.conj(sphere_hankel(b))) / (a * b)


def sphere_square(x):
    """The envelope of f(x)²: 9 (1 + x²)/(2x⁶)."""
    x = np.asarray(x, float)
    return 4.5 * (1.0 + x * x) / x**6


def disc_square(x):
    """The envelope of (2 J₁(x)/x)²: 2 (J₁² + Y₁²)/x²."""
    x = np.asarray(x, float)
    return 2.0 * (j1(x) ** 2 + y1(x) ** 2) / x**2


def sinc_square(x):
    """The envelope of (sin(x/2)/(x/2))²: 2/x²."""
    x = np.asarray(x, float)
    return 2.0 / x**2


__all__ = ["BLEND", "DAMPING", "X_MIN", "blend", "damping_scale", "disc_square", "q_cut", "sinc_square", "sphere_hankel", "sphere_pair",
           "sphere_square"]
