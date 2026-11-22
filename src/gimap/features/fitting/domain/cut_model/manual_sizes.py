"""Size laws of Refine and Search. The original law keys keep Predict's tabulated model and saved fits reproducible."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaincinv, gammainccinv

LAWS = ("gaussian", "lognormal_median", "schulz")
LABELS = {"gaussian": "Gaussian, by number", "lognormal_median": "Log-normal, by number (median)",
          "schulz": "Schulz (Gamma), by number", "intensity_gauss": "Gaussian, by intensity",
          "number_gauss": "Gaussian, by number (tabulated)", "lognormal": "Log-normal, by number (mean)"}
LOGNORMAL_PARAMETERS = {
    "R": ("Median R", "Median radius of the log-normal distribution"),
    "sigma_R": ("σ(ln R)", "Standard deviation of ln R"),
    "h": ("Median h", "Median height of the log-normal distribution"),
    "sigma_h": ("σ(ln h)", "Standard deviation of ln h"),
}
TAIL = 1e-10
"""Cut at most this probability from each tail, including from the distribution weighted by size**power."""


def lognormal_span(median, s, power):
    # Weighting by R**power translates ln R by power*s². Each omitted Gaussian tail beyond 6.5σ is < 5e-11.
    centre = np.log(median)
    return centre - 6.5 * s, centre + (power * s + 6.5) * s


def gamma_span(mean, cv, power):
    # Weighting by R**power raises the shape by power and keeps the scale.
    safe = np.maximum(cv, 1e-10)
    shape = 1.0 / safe**2
    scale = mean * safe**2
    lo = gammaincinv(shape, TAIL) * scale
    hi = gammainccinv(shape + power, TAIL) * scale
    return np.where(cv == 0, mean, lo), np.where(cv == 0, mean, hi)


def gamma_weights(radius, mean, cv):
    # Subtract the log-density at the mean. Evaluating log1p(t)-t as a series near zero also handles narrow laws.
    t = (radius - mean[:, None]) / mean[:, None]
    logarithm = np.log(radius / mean[:, None])
    difference = logarithm - t
    small = np.abs(t) < 1e-4
    x = t[small]
    difference[small] = x**2 * (-0.5 + x * (1.0 / 3.0 + x * (-0.25 + x / 5.0)))
    return np.exp(difference / cv[:, None]**2 - logarithm)


def moments(law, size, width, power):
    if law == "lognormal_median":
        cv = float(np.sqrt(np.expm1(width**2)))
        return {"number": (size * np.exp(0.5 * width**2), cv),
                "intensity": (size * np.exp((power + 0.5) * width**2), cv)}
    # A Gamma distribution weighted by R**p has mean (k+p)θ and relative standard deviation 1/√(k+p).
    factor = 1.0 + power * width**2
    return {"number": (size, width), "intensity": (size * factor, width / np.sqrt(factor))}
