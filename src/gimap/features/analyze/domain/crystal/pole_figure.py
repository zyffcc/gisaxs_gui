"""The pole figure of one ring of a fibre-textured film, and a fit of its orientation distribution.

**Profile.** I(χ) of the ring (1° bins over [−90°, 90°], Analyze's azimuthal curve) minus the background b under the
ring is folded onto |χ| in 90 bins of 1° by inverse-variance weighting (as for the Hermans factor, Physics and
Methods 7.7). Bins without pixels, and when b > 0 bins below 0.2 b and their neighbours (a shadow), are unmeasured.
The folded values are made Lorentz-corrected crystallite counts per degree of χ:

    N_k = Ī_k sin χ_k          (an I(χ) as measured: 1/L ≈ (q/k) sin χ, (5.12))
    N_k = Ī_k                  (an I(χ) already divided by the Lorentz factor; b is then spread as b sin χ / ⟨sin χ⟩)

**Fit.** N(χ) = A · N_model(χ) with N_model of ``pole_models`` averaged over each 1° bin, by weighted least squares
on the measured bins only (``scipy.optimize.least_squares``, bounds as below). The missing wedge and other
unmeasured bins do not enter the fit; the oriented fraction 1 − φ and the Hermans factor of the fitted
distribution include them as the model fills them. Errors: the covariance (JᵀJ)⁻¹ scaled by max(1, χ²_red) (as
(7.12)); the errors of derived quantities (oriented fraction, FWHM of a Maier–Saupe fit, f, the 2D order) by
linear propagation through the numerical Jacobian. They are statistical: the background b, the model and the
window are taken as exact.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .pole_models import GAUSSIAN, MAIER_SAUPE, MODELS, binned, crystallites_per_radian, fwhm_deg, hermans, order_2d

BINS = 90
SHADOW = 0.2
"""A bin below this fraction of the ring background is shadowed (as in the Hermans factor, 7.7)."""
MIN_BINS = 8
FWHM_BOUNDS = (0.3, 150.0)
M_BOUNDS = (-60.0, 60000.0)
CHI_MAX = 85.0
"""Bins above this χ are not fitted by default: next to the sample horizon absorption in the film, the transmission
function or a shadow of the sample lower the intensity (the bins of Δ_h in 7.7); the model fills them."""
START_WIDTHS = (4.0, 15.0, 45.0)
"""Further starting FWHMs (°) of the fit."""
START_DIRECTIONS = (20.0, 45.0, 70.0)
"""Further starting directions χ0 (°) when the direction is fitted."""


@dataclass(frozen=True)
class PoleProfile:
    chi: np.ndarray
    """Bin centres, 0.5° … 89.5°."""
    value: np.ndarray
    """N(χ): background-subtracted, Lorentz-corrected intensity per bin (NaN unmeasured)."""
    sigma: np.ndarray
    measured: np.ndarray
    background: float
    lorentz_input: bool
    """The I(χ) was already divided by the Lorentz factor."""

    @property
    def weighted_coverage(self) -> float:
        """Σ_measured sin χ / Σ_all sin χ (7.24)."""
        weight = np.sin(np.radians(self.chi))
        return float(weight[self.measured].sum() / weight.sum())

    @property
    def coverage(self) -> float:
        return float(self.measured.mean())

    def herman_measured(self) -> Optional[float]:
        """f from the measured bins alone, ⟨cos²χ⟩ = Σ max(N, 0) cos²χ / Σ max(N, 0) (the sums of (7.25))."""
        weight = np.clip(np.where(self.measured, self.value, 0.0), 0.0, None)
        if not weight.sum() > 0:
            return None
        cos2 = np.cos(np.radians(self.chi)) ** 2
        return float(0.5 * (3.0 * (weight * cos2).sum() / weight.sum() - 1.0))


@dataclass(frozen=True)
class PoleFit:
    ok: bool
    model: str
    message: str = ""
    profile: Optional[PoleProfile] = None
    values: dict = field(default_factory=dict)
    """name → value: scale, fraction_iso, width (FWHM ° or m), chi0_deg, oriented_fraction, fwhm_deg, herman,
    herman_oriented, order_2d."""
    errors: dict = field(default_factory=dict)
    fixed: tuple[str, ...] = ()
    chi2_red: float = math.nan
    points: int = 0
    curve_chi: np.ndarray = field(default_factory=lambda: np.empty(0))
    curve_value: np.ndarray = field(default_factory=lambda: np.empty(0))
    """The fitted N(χ) on a 0.25° grid over [0°, 90°] (per degree, as the profile)."""

    def record(self) -> dict:
        profile = self.profile
        return {
            "model": self.model, "ok": self.ok, "message": self.message,
            "values": dict(self.values), "errors": dict(self.errors), "fixed": list(self.fixed),
            "reduced_chi2": self.chi2_red, "points": self.points,
            "coverage": None if profile is None else profile.coverage,
            "weighted_coverage": None if profile is None else profile.weighted_coverage,
            "herman_measured": None if profile is None else profile.herman_measured(),
            "background": None if profile is None else profile.background,
            "lorentz_input": None if profile is None else profile.lorentz_input,
        }


def fold_profile(chi_deg, intensity, sigma, background: float = 0.0, *, lorentz: bool = False) -> PoleProfile:
    """The folded, background-subtracted, Lorentz-corrected profile of a signed I(χ) (see the module docstring)."""
    chi = np.asarray(chi_deg, dtype=float)
    value = np.asarray(intensity, dtype=float)
    error = np.asarray(sigma, dtype=float)
    finite = np.isfinite(chi) & np.isfinite(value) & np.isfinite(error) & (error > 0) & (np.abs(chi) <= 90.0)
    b = float(background or 0.0)
    sine = np.abs(np.sin(np.radians(chi)))
    if lorentz and b and finite.any():
        local = b * sine / max(float(sine[finite].mean()), 1e-12)
    else:
        local = np.full(chi.shape, b)
    shadow = np.zeros(chi.shape, dtype=bool)
    if b > 0:
        low = finite & (value < SHADOW * local)
        shadow = low | np.roll(low, 1) | np.roll(low, -1)
    keep = finite & ~shadow
    index = np.clip(np.floor(np.abs(chi)).astype(int), 0, BINS - 1)
    weights = np.zeros(BINS)
    sums = np.zeros(BINS)
    for k, net, err in zip(index[keep], (value - local)[keep], error[keep]):
        weights[k] += 1.0 / err ** 2
        sums[k] += net / err ** 2
    measured = weights > 0
    centres = np.arange(BINS) + 0.5
    mean = np.where(measured, sums / np.where(measured, weights, 1.0), np.nan)
    spread = np.where(measured, 1.0 / np.sqrt(np.where(measured, weights, 1.0)), np.nan)
    if not lorentz:
        factor = np.sin(np.radians(centres))
        mean, spread = mean * factor, spread * factor
    return PoleProfile(centres, mean, spread, measured, b, bool(lorentz))


def _parameters(model: str, values: dict) -> tuple[float, float, float, float]:
    return values["scale"], values["fraction_iso"], values["width"], values["chi0_deg"]


def _derived(model: str, scale: float, fraction: float, width: float, chi0: float) -> dict:
    oriented = 1.0 - fraction
    f_total = hermans(model, fraction, width, chi0)
    return {
        "oriented_fraction": oriented, "fwhm_deg": fwhm_deg(model, width), "herman": f_total,
        "herman_oriented": hermans(model, 0.0, width, chi0), "order_2d": order_2d(model, fraction, width, chi0),
    }


def _start(profile: PoleProfile, model: str, chi0: Optional[float], width: Optional[float]) -> dict:
    """Starting values: the scale from the measured sum, φ from the lowest pole densities, the direction from the
    highest one, the FWHM from the bins above half of it (or ``width``)."""
    measured = profile.measured & np.isfinite(profile.value)
    chi, value = profile.chi[measured], profile.value[measured]
    scale = max(float(np.clip(value, 0.0, None).sum()) / max(profile.coverage, 0.1), 1e-12)
    density = value / np.maximum(np.sin(np.radians(chi)), 1e-3)
    floor = float(np.percentile(density, 20)) if len(density) else 0.0
    fraction = float(np.clip(floor / (scale * math.radians(1.0)), 0.02, 0.95))
    top = int(np.argmax(density)) if len(density) else 0
    centre = float(chi[top]) if chi0 is None and len(chi) else float(chi0 or 0.0)
    if width is None:
        above = int(np.sum(density > 0.5 * (density[top] + floor))) if len(density) else 10
        folded = centre < 10.0 or centre > 80.0  # half of the peak lies beyond the fold
        width = float(np.clip(above * (2.0 if folded else 1.0), 2.0, 60.0))
    return {"scale": scale, "fraction_iso": fraction, "width": _spread(model, width), "chi0_deg": centre}


def _spread(model: str, fwhm: float) -> float:
    """The width parameter of ``model`` for a FWHM in degrees: the FWHM itself, or m ≈ 4 ln 2 / FWHM² (radians)."""
    return float(fwhm) if model == GAUSSIAN else 4.0 * math.log(2.0) / math.radians(max(float(fwhm), 0.5)) ** 2


def fit_pole_figure(profile: PoleProfile, model: str = GAUSSIAN, *, chi0_deg: Optional[float] = 0.0,
                    width: Optional[float] = None, fraction_iso: Optional[float] = None,
                    chi_max: float = CHI_MAX) -> PoleFit:
    """Fit ``model`` to ``profile``; ``chi0_deg`` ``None`` fits the direction, a number fixes it; ``fraction_iso``
    a number fixes φ (0: no isotropic part). ``width`` is a starting FWHM in degrees. Only bins with χ ≤ ``chi_max``
    are fitted (``CHI_MAX``). Several starts are tried (``START_WIDTHS``, the start φ and 0.5) and the lowest χ²
    is kept: a single start can end in a minimum with φ at 0 and a very broad peak."""
    from scipy.optimize import least_squares

    if model not in MODELS:
        raise ValueError(f"Unknown orientation model {model!r}.")
    measured = profile.measured & np.isfinite(profile.value) & np.isfinite(profile.sigma) & (profile.sigma > 0)
    measured &= profile.chi <= float(chi_max)
    if measured.sum() < MIN_BINS:
        return PoleFit(False, model, f"only {int(measured.sum())} measured χ bins (at least {MIN_BINS})", profile)
    if not np.nanmax(profile.value[measured] / profile.sigma[measured]) >= 3.0:
        return PoleFit(False, model, "no χ bin of the ring is 3 σ above the background", profile)
    centres = profile.chi[measured]
    low, high = centres - 0.5, centres + 0.5
    data, error = profile.value[measured], profile.sigma[measured]
    start = _start(profile, model, chi0_deg, width)
    if fraction_iso is not None:
        start["fraction_iso"] = float(fraction_iso)
    names = ["scale", "fraction_iso", "width", "chi0_deg"]
    fixed = tuple(name for name, value in (("chi0_deg", chi0_deg), ("fraction_iso", fraction_iso)) if value is not None)
    free = [name for name in names if name not in fixed]
    width_bounds = FWHM_BOUNDS if model == GAUSSIAN else M_BOUNDS
    bounds = {"scale": (0.0, np.inf), "fraction_iso": (0.0, 1.0), "width": width_bounds, "chi0_deg": (0.0, 90.0)}
    per_degree = math.radians(1.0)  # N per radian → per degree, the unit of the profile

    def values_of(vector) -> dict:
        values = dict(start)
        values.update(zip(free, vector))
        return values

    def model_of(values: dict) -> np.ndarray:
        scale, fraction, spread, chi0 = _parameters(model, values)
        return scale * per_degree * binned(low, high, model, fraction, spread, chi0)

    def residuals(vector) -> np.ndarray:
        return (model_of(values_of(vector)) - data) / error

    lower = np.array([bounds[name][0] for name in free])
    upper = np.array([bounds[name][1] for name in free])
    margin = 1e-6 * (np.abs(lower) + 1.0)
    solution = None
    widths = [start["width"]] + [_spread(model, value) for value in START_WIDTHS if width is None]
    fractions = sorted({start["fraction_iso"], 0.5}) if fraction_iso is None else [start["fraction_iso"]]
    centres = [start["chi0_deg"]] + (list(START_DIRECTIONS) if chi0_deg is None else [])
    for start_width, start_fraction, start_centre in itertools.product(widths, fractions, centres):
        trial = dict(start, width=start_width, fraction_iso=start_fraction, chi0_deg=start_centre)
        x0 = np.array([trial[name] for name in free], dtype=float)
        x0 = np.minimum(np.maximum(x0, lower + margin), np.where(np.isfinite(upper), upper - margin, np.inf))
        try:
            found = least_squares(residuals, x0, bounds=(lower, upper), x_scale="jac", max_nfev=4000)
        except Exception:  # noqa: BLE001 - a start that fails is skipped; no start at all is reported below
            continue
        if solution is None or found.cost < solution.cost:
            solution = found
    if solution is None:
        return PoleFit(False, model, "the fit failed from every start", profile)
    dof = max(1, len(data) - len(free))
    chi2_red = float(np.sum(solution.fun ** 2) / dof)
    values = values_of(solution.x)
    try:
        covariance = np.linalg.pinv(solution.jac.T @ solution.jac) * max(1.0, chi2_red)
    except np.linalg.LinAlgError:
        covariance = np.full((len(free), len(free)), np.nan)
    errors = {name: float(math.sqrt(max(covariance[i, i], 0.0))) for i, name in enumerate(free)}
    errors.update({name: 0.0 for name in fixed})
    derived = _derived(model, *_parameters(model, values))

    def derived_vector(vector) -> np.ndarray:
        return np.array(list(_derived(model, *_parameters(model, values_of(vector))).values()), dtype=float)

    jacobian = np.zeros((len(derived), len(free)))
    for i, name in enumerate(free):
        step = 1e-4 * max(abs(solution.x[i]), 1e-3)
        up, down = solution.x.copy(), solution.x.copy()
        up[i] = min(up[i] + step, upper[i])
        down[i] = max(down[i] - step, lower[i])
        if up[i] > down[i]:
            jacobian[:, i] = (derived_vector(up) - derived_vector(down)) / (up[i] - down[i])
    derived_cov = jacobian @ covariance @ jacobian.T
    for i, name in enumerate(derived):
        errors[name] = float(math.sqrt(max(derived_cov[i, i], 0.0))) if np.isfinite(derived_cov[i, i]) else math.nan
    values.update(derived)
    grid = np.linspace(0.0, 90.0, 361)
    curve = values["scale"] * per_degree * crystallites_per_radian(
        np.radians(grid), model, values["fraction_iso"], values["width"], values["chi0_deg"])
    message = "" if solution.success else str(solution.message)
    at_bound = [name for i, name in enumerate(free) if name != "scale"
                and (abs(solution.x[i] - lower[i]) < 1e-6 * (abs(lower[i]) + 1) or abs(solution.x[i] - upper[i]) < 1e-6 * (abs(upper[i]) + 1))]
    if at_bound:
        message = (message + "; " if message else "") + "at its bound: " + ", ".join(at_bound)
    return PoleFit(True, model, message, profile, {k: float(v) for k, v in values.items()}, errors, fixed,
                   chi2_red, int(len(data)), grid, curve)


__all__ = ["BINS", "GAUSSIAN", "MAIER_SAUPE", "PoleFit", "PoleProfile", "fit_pole_figure", "fold_profile"]
