"""One diffraction peak fitted in a q window: a peak shape on a straight background.

``fit_peak`` fits ``area · shape(x; centre, FWHM, η) + b0 + b1 · (x − centre₀)`` to the points of
``[low, high]`` by weighted least squares (the points' σ as weights, ``scipy.optimize.curve_fit``
with ``absolute_sigma``; the errors are scaled by √χ²ᵣ when the scatter is larger than σ says).
Shapes are normalised to unit area, so ``area`` is the integrated intensity above the background:

* ``gaussian`` — exp(−4 ln2 (x − c)² / w²), the instrument and small-crystallite limit;
* ``lorentzian`` — 1 / (1 + 4 (x − c)² / w²);
* ``pseudo_voigt`` — η · Lorentzian + (1 − η) · Gaussian with the same FWHM (η fitted in [0, 1]).

``start`` gives the starting values (e.g. the previous frame's result, for a series that changes
slowly); without it they come from the window itself: the largest point above a straight line
through the window's ends, and its half-maximum width. The centre stays inside the window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

GAUSSIAN, LORENTZIAN, PSEUDO_VOIGT = "gaussian", "lorentzian", "pseudo_voigt"
PROFILES = (GAUSSIAN, LORENTZIAN, PSEUDO_VOIGT)
MIN_POINTS = 7
SIGNIFICANCE = 3.0
"""A peak whose area is less than this many times its error is not reported as found."""
_LN2 = math.log(2.0)


@dataclass(frozen=True)
class PeakFit:
    ok: bool
    message: str = ""
    profile: str = PSEUDO_VOIGT
    center: float = math.nan
    center_err: float = math.nan
    fwhm: float = math.nan
    fwhm_err: float = math.nan
    area: float = math.nan
    area_err: float = math.nan
    eta: float = math.nan
    """Lorentzian fraction of a pseudo-Voigt (1 for a Lorentzian, 0 for a Gaussian)."""
    height: float = math.nan
    background: tuple[float, float] = (math.nan, math.nan)
    """``(b0, b1)``: the background at the start centre and its slope."""
    chi2_red: float = math.nan
    points: int = 0
    x: np.ndarray = field(default_factory=lambda: np.empty(0))
    y: np.ndarray = field(default_factory=lambda: np.empty(0))
    """The data fitted (the points of the window)."""
    fitted: np.ndarray = field(default_factory=lambda: np.empty(0))
    """The fitted curve on ``x``."""

    def as_start(self) -> dict:
        return {"center": self.center, "fwhm": self.fwhm, "area": self.area, "eta": self.eta}


def _gauss(x, center, fwhm):
    return math.sqrt(4.0 * _LN2 / math.pi) / fwhm * np.exp(-4.0 * _LN2 * (x - center) ** 2 / fwhm**2)


def _lorentz(x, center, fwhm):
    return (2.0 / (math.pi * fwhm)) / (1.0 + 4.0 * (x - center) ** 2 / fwhm**2)


def shape(x, center: float, fwhm: float, eta: float, profile: str) -> np.ndarray:
    """The unit-area peak shape at ``x``."""
    x = np.asarray(x, dtype=float)
    if profile == GAUSSIAN:
        return _gauss(x, center, fwhm)
    if profile == LORENTZIAN:
        return _lorentz(x, center, fwhm)
    return eta * _lorentz(x, center, fwhm) + (1.0 - eta) * _gauss(x, center, fwhm)


def _guess(x: np.ndarray, y: np.ndarray) -> dict:
    """Starting values from the window: the top above the straight line through its ends."""
    edge = max(1, min(3, x.size // 6))
    x0, y0 = float(np.mean(x[:edge])), float(np.mean(y[:edge]))
    x1, y1 = float(np.mean(x[-edge:])), float(np.mean(y[-edge:]))
    slope = (y1 - y0) / (x1 - x0) if x1 > x0 else 0.0
    net = y - (y0 + slope * (x - x0))
    top = int(np.argmax(net))
    height = max(float(net[top]), 1e-12)
    above = np.flatnonzero(net >= 0.5 * height)
    width = float(x[above.max()] - x[above.min()]) if above.size > 1 else 0.0
    step = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    fwhm = max(width, 2.0 * step)
    return {"center": float(x[top]), "fwhm": fwhm, "area": height * fwhm * 1.064, "eta": 0.5,
            "b0": float(y0 + slope * (x[top] - x0)), "b1": slope}


def fit_peak(
    x, y, sigma=None, window: Optional[tuple[float, float]] = None, *,
    profile: str = PSEUDO_VOIGT, start: Optional[dict] = None,
) -> PeakFit:
    """Fit one peak in ``window`` (see the module docstring); ``ok`` is ``False`` with a reason when it cannot."""
    from scipy.optimize import curve_fit

    if profile not in PROFILES:
        raise ValueError(f"Unknown peak shape {profile!r}.")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = np.ones_like(y) if sigma is None else np.asarray(sigma, dtype=float)
    low, high = window if window is not None else (float(np.nanmin(x)), float(np.nanmax(x)))
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(s) & (x >= low) & (x <= high)
    if keep.sum() < MIN_POINTS:
        return PeakFit(False, f"only {int(keep.sum())} points in the window", profile)
    order = np.argsort(x[keep])
    x, y, s = x[keep][order], y[keep][order], s[keep][order]
    s = np.where(s > 0, s, np.nanmedian(s[s > 0]) if (s > 0).any() else 1.0)
    guess = _guess(x, y)
    if start:
        for key in ("center", "fwhm", "area", "eta"):
            value = start.get(key)
            if value is not None and np.isfinite(value):
                guess[key] = float(value)
    step = float(np.median(np.diff(x)))
    span = float(x[-1] - x[0])
    guess["center"] = float(np.clip(guess["center"], x[0], x[-1]))
    guess["fwhm"] = float(np.clip(guess["fwhm"], step, span))
    guess["eta"] = float(np.clip(guess["eta"], 0.0, 1.0))
    reference = guess["center"]
    scale = max(float(np.nanmax(np.abs(y))), 1e-12)

    def model(xs, center, fwhm, area, eta, b0, b1):
        return area * shape(xs, center, fwhm, eta, profile) + b0 + b1 * (xs - reference)

    p0 = [guess["center"], guess["fwhm"], max(guess["area"], 1e-12), guess["eta"], guess["b0"], guess["b1"]]
    lower = [x[0], 0.5 * step, 0.0, 0.0, -np.inf, -np.inf]
    upper = [x[-1], 2.0 * span, np.inf, 1.0, np.inf, np.inf]
    fixed_eta = {GAUSSIAN: 0.0, LORENTZIAN: 1.0}.get(profile)
    if fixed_eta is not None:  # the mixing is not a parameter of the pure shapes
        p0[3], lower[3], upper[3] = fixed_eta, fixed_eta - 1e-9, fixed_eta + 1e-9
    try:
        popt, pcov = curve_fit(model, x, y, p0=p0, sigma=s, absolute_sigma=True, bounds=(lower, upper), maxfev=20000,
                               x_scale=[step, step, max(guess["area"], 1e-12), 1.0, scale, scale / max(span, 1e-12)])
    except (RuntimeError, ValueError) as exc:
        return PeakFit(False, f"the fit did not converge ({exc})", profile, points=int(x.size))
    fitted = model(x, *popt)
    dof = max(1, x.size - (5 if fixed_eta is not None else 6))
    chi2_red = float(np.sum(((y - fitted) / s) ** 2) / dof)
    errors = np.sqrt(np.clip(np.diag(pcov), 0.0, None)) * math.sqrt(max(1.0, chi2_red))
    center, fwhm, area, eta, b0, b1 = (float(value) for value in popt)
    height = float(area * shape(np.array([center]), center, fwhm, eta, profile)[0])
    message = ""
    if area <= 0:
        message = "no peak above the background"
    elif np.isfinite(errors[2]) and area < SIGNIFICANCE * errors[2]:
        message = f"no significant peak (area < {SIGNIFICANCE:g} × its error)"
    elif fwhm >= 0.9 * span:
        message = "the peak is as wide as the window: widen the window"
    return PeakFit(
        ok=not message, message=message, profile=profile, center=center, center_err=float(errors[0]),
        fwhm=fwhm, fwhm_err=float(errors[1]), area=area, area_err=float(errors[2]), eta=eta, height=height,
        background=(b0, b1), chi2_red=chi2_red, points=int(x.size), x=x, y=y, fitted=fitted,
    )


__all__ = ["GAUSSIAN", "LORENTZIAN", "PROFILES", "PSEUDO_VOIGT", "PeakFit", "fit_peak", "shape"]
