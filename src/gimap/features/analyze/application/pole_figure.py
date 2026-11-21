"""Use case: the pole figure of the ring in Analyze's I(χ) window and the fit of its orientation distribution.

The ring is the I(χ) window of the frame (the purple band over I(q); ``set_chi_window``). The background under it is
the one given (the automatic analysis passes its SNIP baseline at the ring), else the local linear background of a
Gaussian fitted to I(q) over three times the window (``peak_fit.fit_peak``), else the mean of I(q) just outside the
window. The profile, the models and the errors are those of ``domain/crystal/pole_figure.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..domain.crystal import GAUSSIAN, MAIER_SAUPE, MODELS, PoleFit, fit_pole_figure, fold_profile
from ..domain.crystal.pole_figure import CHI_MAX
from ..domain.giwaxs import q_from_two_theta
from ..domain.peak_fit import GAUSSIAN as GAUSSIAN_PEAK
from ..domain.peak_fit import fit_peak
from .models import FrameAnalysis

FOLD_EDGE_DEG = 1.0
"""A free direction that ends this close to 0° or 90° is fixed there and fitted again (the fold makes its error
undefined: the profile does not change to first order when the direction moves off a fold)."""
BACKGROUND_GIVEN, BACKGROUND_PEAK, BACKGROUND_FLANKS, BACKGROUND_NONE = (
    "given", "local linear background of a Gaussian fit to I(q)", "mean of I(q) beside the window", "none")


@dataclass(frozen=True)
class PoleFigure:
    fit: PoleFit
    window: tuple[float, float]
    background: float
    background_source: str
    lorentz: bool
    chi0_request: Optional[float]
    chi_max: float = CHI_MAX

    @property
    def q_center(self) -> float:
        return 0.5 * (self.window[0] + self.window[1])

    def record(self) -> dict:
        return {
            **self.fit.record(),
            "ring": {"q_window": list(self.window), "q": self.q_center},
            "background": {"value": self.background, "source": self.background_source},
            "lorentz_corrected_input": self.lorentz,
            "direction_requested_deg": self.chi0_request, "fitted_up_to_chi_deg": self.chi_max,
        }


def _radial_in_q(analysis: FrameAnalysis):
    curve = analysis.reduction.curve("radial") if analysis.reduction is not None else None
    if curve is None or curve.is_empty:
        return None
    x = np.asarray(curve.x, dtype=float)
    if str(curve.x_label).startswith("2θ") and analysis.geometry is not None:
        x = q_from_two_theta(x, analysis.geometry.wavelength_angstrom)
    return x, np.asarray(curve.intensity, dtype=float), np.asarray(curve.sigma, dtype=float)


def ring_background(analysis: FrameAnalysis, window: tuple[float, float]) -> tuple[float, str]:
    """The background under the ring of ``window`` from I(q) (see the module docstring)."""
    radial = _radial_in_q(analysis)
    if radial is None:
        return 0.0, BACKGROUND_NONE
    x, y, sigma = radial
    low, high = sorted(float(value) for value in window)
    centre, half = 0.5 * (low + high), 0.5 * (high - low)
    fit = fit_peak(x, y, sigma, (centre - 3.0 * half, centre + 3.0 * half), profile=GAUSSIAN_PEAK)
    if fit.ok and np.isfinite(fit.background[0]) and low <= fit.center <= high:
        return float(fit.background[0]), BACKGROUND_PEAK  # b0: the straight background at the peak's start centre
    beside = np.isfinite(y) & (((x >= low - half) & (x < low)) | ((x > high) & (x <= high + half)))
    if beside.any():
        return float(np.nanmean(y[beside])), BACKGROUND_FLANKS
    return 0.0, BACKGROUND_NONE


def pole_figure(analysis: Optional[FrameAnalysis], *, model: str = GAUSSIAN, chi0_deg: Optional[float] = 0.0,
                background: Optional[float] = None, fraction_iso: Optional[float] = None,
                chi_max: float = CHI_MAX) -> PoleFigure:
    """The pole figure of the I(χ) ring of ``analysis`` and its fit (ValueError without a GIWAXS ring)."""
    if analysis is None or analysis.reduction is None:
        raise ValueError("Open a GIWAXS frame with a geometry first.")
    curve = analysis.reduction.curve("azimuthal")
    window = analysis.reduction.markers.get("chi_q_window")
    if curve is None or curve.is_empty or window is None:
        raise ValueError("No ring is selected for I(χ): drag the purple band over I(q) onto a ring.")
    window = tuple(sorted(float(value) for value in window))
    if background is None:
        value, source = ring_background(analysis, window)
    else:
        value, source = float(background), BACKGROUND_GIVEN
    lorentz = bool(getattr(analysis.corrections, "lorentz", False))
    profile = fold_profile(curve.x, curve.intensity, curve.sigma, value, lorentz=lorentz)
    fit = fit_pole_figure(profile, model, chi0_deg=chi0_deg, fraction_iso=fraction_iso, chi_max=chi_max)
    if fit.ok and chi0_deg is None:
        direction = fit.values.get("chi0_deg", 45.0)
        for edge in (0.0, 90.0):
            if abs(direction - edge) < FOLD_EDGE_DEG:
                refit = fit_pole_figure(profile, model, chi0_deg=edge, fraction_iso=fraction_iso, chi_max=chi_max)
                if refit.ok:
                    note = f"the direction fitted at the fold ({edge:g}°) was fixed there"
                    fit = PoleFit(refit.ok, refit.model, "; ".join(m for m in (refit.message, note) if m),
                                  refit.profile, refit.values, refit.errors, refit.fixed, refit.chi2_red, refit.points,
                                  refit.curve_chi, refit.curve_value)
    return PoleFigure(fit, window, value, source, lorentz, chi0_deg, float(chi_max))


def pole_figure_csv(result: PoleFigure, header: Optional[dict] = None) -> str:
    """The folded profile and the fitted curve as a table; the source, settings and fit in ``#`` lines."""
    fit, profile = result.fit, result.fit.profile
    lines = ["# GIMaP pole figure (Lorentz-corrected crystallites per degree of chi, background subtracted)"]
    for key, value in (header or {}).items():
        lines.append(f"# {key}: {value}")
    lines.append("# pole figure: " + json.dumps(_plain(result.record()), ensure_ascii=False, sort_keys=True))
    lines.append("# columns: chi_deg  N  sigma  measured  model")
    if profile is None:
        return "\n".join(lines) + "\n"
    model = np.interp(profile.chi, fit.curve_chi, fit.curve_value) if fit.ok and fit.curve_chi.size else \
        np.full(profile.chi.shape, np.nan)
    for chi, value, sigma, measured, fitted in zip(profile.chi, profile.value, profile.sigma, profile.measured, model):
        lines.append(f"{chi:.2f}  {_number(value)}  {_number(sigma)}  {int(bool(measured))}  {_number(fitted)}")
    return "\n".join(lines) + "\n"


def _number(value) -> str:
    return "nan" if value is None or not np.isfinite(value) else f"{float(value):.10g}"


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


__all__ = ["CHI_MAX", "GAUSSIAN", "MAIER_SAUPE", "MODELS", "PoleFigure", "pole_figure", "pole_figure_csv", "ring_background"]
