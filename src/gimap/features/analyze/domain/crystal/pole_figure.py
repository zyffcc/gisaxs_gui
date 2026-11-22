"""The pole figure of one ring of a fibre-textured film, and a fit of its orientation distribution.

**Profile.** I(χ) of the ring (1° bins over [−90°, 90°], Analyze's azimuthal curve) minus the background b under the
ring is folded onto |χ| in 90 bins of 1° by inverse-variance weighting (as for the Hermans factor, Physics and
Methods 7.7). Bins without pixels, and when b > 0 bins below 0.2 b and their neighbours (a shadow), are unmeasured.
The input is a binned mean pixel intensity before the detector Lorentz factor. The number per unit
polar angle is proportional to the folded density times sin χ.

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
from functools import cached_property, lru_cache
from typing import Optional

import numpy as np

from .....shared.ring_profile import fold_ring, measured_hermans
from .pole_models import GAUSSIAN, MAIER_SAUPE, MODELS, binned, crystallites_per_radian, fwhm_deg, hermans, order_2d

BINS = 90
MIN_BINS = 8
FWHM_BOUNDS = (0.3, 150.0)
M_BOUNDS = (-60.0, 60000.0)
CHI_MAX = 90.0
"""The input already has the exit-angle exclusion. An explicit polar-angle limit can narrow it further."""
START_WIDTHS = (4.0, 15.0, 45.0)
"""Further starting FWHMs (°) of the fit."""
START_DIRECTIONS = (20.0, 45.0, 70.0)
"""Further starting directions χ0 (°) when the direction is fitted."""


@dataclass(frozen=True)
class PoleProfile:
    chi: np.ndarray
    """Bin centres, 0.5° … 89.5°."""
    value: np.ndarray
    """N(χ): background-subtracted density times sin χ (NaN unmeasured)."""
    sigma: np.ndarray
    measured: np.ndarray
    background: float
    lorentz_input: bool
    """Legacy record field, always false for a binned orientation profile."""

    @property
    def weighted_coverage(self) -> float:
        """Σ_measured sin χ / Σ_all sin χ (7.24)."""
        weight = np.sin(np.radians(self.chi))
        return float(weight[self.measured].sum() / weight.sum())

    @property
    def coverage(self) -> float:
        return float(self.measured.mean())

    def herman_measured(self) -> Optional[float]:
        """f from the measured bins. Negative net counts are retained after background subtraction."""
        return measured_hermans(self.chi[self.measured], self.value[self.measured])[0]



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
    at_bound: tuple[str, ...] = ()
    """Free parameters (not the scale) that ended at a bound: what they stand for is not determined by the fit."""

    converged: bool = False

    chi_max: float = CHI_MAX

    @property
    def measured_bins(self):
        if self.profile is None:
            return np.zeros(0, dtype=bool)
        return self.profile.measured & (self.profile.chi <= self.chi_max)

    @property
    def weighted_coverage(self):
        if self.profile is None:
            return None
        weight = np.sin(np.radians(self.profile.chi))
        return float(weight[self.measured_bins].sum() / weight.sum())

    @cached_property
    def measured_moment(self):
        if self.profile is None:
            return None, None, None
        p = self.profile
        keep = self.measured_bins
        return measured_hermans(p.chi[keep], p.value[keep], p.sigma[keep])

    @property
    def quality_reasons(self) -> tuple[str, ...]:
        reasons = []
        if not self.ok or not self.converged:
            reasons.append("fit did not converge")
        if self.chi2_red > 5.:
            reasons.append("reduced χ² exceeds 5")
        if self.values.get("fwhm_deg", 0.) > 100.:
            reasons.append("FWHM exceeds 100°")
        if self.ok and any(not math.isfinite(self.errors.get(name, 0.))
                           for name in ("scale", "fraction_iso", "width", "chi0_deg")):
            reasons.append("fit parameters are not identifiable")
        labels = {"fraction_iso": "isotropic fraction at a bound", "width": "width at a bound",
                  "chi0_deg": "direction at a bound"}
        reasons.extend(labels[name] for name in self.at_bound if name in labels)
        return tuple(reasons)

    def record(self) -> dict:
        profile = self.profile
        return {
            "model": self.model, "ok": self.ok, "message": self.message,
            "values": dict(self.values), "errors": dict(self.errors), "fixed": list(self.fixed),
            "at_bound": list(self.at_bound), "converged": self.converged, "quality_reasons": list(self.quality_reasons),
            "reduced_chi2": self.chi2_red, "points": self.points,
            "coverage": None if profile is None else float(self.measured_bins.mean()),
            "weighted_coverage": self.weighted_coverage,
            "herman_measured": self.measured_moment[0], "herman_measured_err": self.measured_moment[1],
            "background": None if profile is None else profile.background,
            "lorentz_input": None if profile is None else profile.lorentz_input,
        }


def fold_profile(chi_deg, intensity, sigma, background: float = 0.0) -> PoleProfile:
    """Crystallites per bin from mean pixel intensity before the detector Lorentz correction."""
    b = float(background)
    centres, mean, spread, measured, _shadow = fold_ring(chi_deg, intensity, sigma, b)
    factor = np.sin(np.radians(centres))
    value = np.where(measured, mean * factor, np.nan)
    error = np.where(measured, spread * factor, np.nan)
    return PoleProfile(centres, value, error, measured, b, False)


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
    return float(fwhm) if model != MAIER_SAUPE else 4.0 * math.log(2.0) / math.radians(max(float(fwhm), 0.5)) ** 2


def fit_pole_figure(profile: PoleProfile, model: str = GAUSSIAN, *, chi0_deg: Optional[float] = 0.0,
                    width: Optional[float] = None, fraction_iso: Optional[float] = None,
                    chi_max: float = CHI_MAX, start_values: Optional[dict] = None) -> PoleFit:
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
        return PoleFit(False, model, f"only {int(measured.sum())} measured χ bins (at least {MIN_BINS})", profile, chi_max=chi_max)
    if not np.nanmax(profile.value[measured] / profile.sigma[measured]) >= 3.0:
        return PoleFit(False, model, "no χ bin of the ring is 3 σ above the background", profile, chi_max=chi_max)
    centres = profile.chi[measured]
    low, high = centres - 0.5, centres + 0.5
    data, error = profile.value[measured], profile.sigma[measured]
    start = _start(profile, model, chi0_deg, width)
    if fraction_iso is not None:
        start["fraction_iso"] = float(fraction_iso)
    names = ["scale", "fraction_iso", "width", "chi0_deg"]
    fixed = tuple(name for name, value in (("chi0_deg", chi0_deg), ("fraction_iso", fraction_iso)) if value is not None)
    free = [name for name in names if name not in fixed]
    width_bounds = M_BOUNDS if model == MAIER_SAUPE else FWHM_BOUNDS
    bounds = {"scale": (0.0, np.inf), "fraction_iso": (0.0, 1.0), "width": width_bounds, "chi0_deg": (0.0, 90.0)}
    per_degree = math.radians(1.0)  # N per radian → per degree, the unit of the profile

    def values_of(vector) -> dict:
        values = dict(start)
        values.update(zip(free, vector))
        return values

    # The bin-integrated shape depends only on width and direction. Reuse it
    # for scale/fraction steps of the numerical Jacobian, especially for ψ quadrature.
    isotropic = per_degree * binned(low, high, GAUSSIAN, 1., 15., 0.)

    @lru_cache(maxsize=12)
    def oriented(spread, chi0):
        return per_degree * binned(low, high, model, 0., spread, chi0)

    def model_of(values: dict) -> np.ndarray:
        scale, fraction, spread, chi0 = _parameters(model, values)
        if model == GAUSSIAN:
            return scale * per_degree * binned(low, high, model, fraction, spread, chi0)
        return scale * ((1. - fraction) * oriented(spread, chi0) + fraction * isotropic)

    def residuals(vector) -> np.ndarray:
        return (model_of(values_of(vector)) - data) / error

    lower = np.array([bounds[name][0] for name in free])
    upper = np.array([bounds[name][1] for name in free])
    margin = 1e-6 * (np.abs(lower) + 1.0)
    solution = None
    widths = [start["width"]] + [_spread(model, value) for value in START_WIDTHS if width is None]
    fractions = sorted({start["fraction_iso"], 0.5}) if fraction_iso is None else [start["fraction_iso"]]
    if model != GAUSSIAN:
        fractions = [start["fraction_iso"]]
    if model == MAIER_SAUPE:
        widths.append(-20.)  # A girdle perpendicular to the director has m < 0.
    centres = [start["chi0_deg"]] + (list(START_DIRECTIONS) if chi0_deg is None else [])
    trials = (dict(start, width=w, fraction_iso=f, chi0_deg=c) for w, f, c in itertools.product(widths, fractions, centres))
    if start_values is not None:
        trials = [dict(start, **{name: start_values[name] for name in free})]
    for trial in trials:
        x0 = np.array([trial[name] for name in free], dtype=float)
        x0 = np.minimum(np.maximum(x0, lower + margin), np.where(np.isfinite(upper), upper - margin, np.inf))
        try:
            found = least_squares(residuals, x0, bounds=(lower, upper), x_scale="jac", max_nfev=4000)
        except Exception:  # noqa: BLE001 - a start that fails is skipped; no start at all is reported below
            continue
        if solution is None or (found.success, -found.cost) > (solution.success, -solution.cost):
            solution = found
    if solution is None:
        return PoleFit(False, model, "the fit failed from every start", profile, chi_max=chi_max)
    dof = max(1, len(data) - len(free))
    chi2_red = float(np.sum(solution.fun ** 2) / dof)
    values = values_of(solution.x)
    try:
        # Scale columns before inversion. The count scale and the fraction can
        # differ by many orders of magnitude. Forming JᵀJ first can discard
        # the scale direction and underestimate the error of the wedge fill.
        lengths = np.linalg.norm(solution.jac, axis=0)
        if not np.all(lengths > 0):
            raise np.linalg.LinAlgError("undetermined fit parameter")
        scaled = solution.jac / lengths
        if np.linalg.matrix_rank(scaled) < len(free):
            raise np.linalg.LinAlgError("undetermined fit parameter")
        inverse = np.linalg.pinv(scaled) / lengths[:, None]
        covariance = (inverse @ inverse.T) * max(1.0, chi2_red)
    except np.linalg.LinAlgError:
        covariance = np.full((len(free), len(free)), np.nan)
    errors = {name: float(math.sqrt(max(covariance[i, i], 0.0))) for i, name in enumerate(free)}
    errors.update({name: 0.0 for name in fixed})
    # Fill only the contiguous low-χ gap. Keep measured counts and leave the
    # horizon and internal detector gaps out. The derivative below propagates
    # the fitted parameter covariance through just this extrapolated contribution.
    first = int(np.flatnonzero(measured)[0])
    wedge = profile.chi[:first]
    p2 = (3. * np.cos(np.radians(profile.chi))**2 - 1.) / 2.
    measured_counts = profile.value[measured]
    measured_sum = measured_counts.sum()
    measured_moment = np.dot(measured_counts, p2[measured])

    def all_derived(values):
        scale, fraction, width, chi0 = _parameters(model, values)
        result = _derived(model, scale, fraction, width, chi0)
        extra = scale * per_degree * binned(wedge - 0.5, wedge + 0.5, model, fraction, width, chi0)
        total = measured_sum + extra.sum()
        result["herman_wedge_fill"] = (measured_moment + np.dot(extra, p2[:first])) / total if total > 0. else math.nan
        return result

    derived = all_derived(values)

    def derived_vector(vector) -> np.ndarray:
        return np.array(list(all_derived(values_of(vector)).values()), dtype=float)

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
                and (solution.active_mask[i] != 0 or abs(solution.x[i] - lower[i]) < 1e-6 * (abs(lower[i]) + 1)
                     or abs(solution.x[i] - upper[i]) < 1e-6 * (abs(upper[i]) + 1))]
    if at_bound:
        message = (message + "; " if message else "") + "at its bound: " + ", ".join(at_bound)
    return PoleFit(True, model, message, profile, {k: float(v) for k, v in values.items()}, errors, fixed,
                   chi2_red, int(len(data)), grid, curve, tuple(at_bound), bool(solution.success), chi_max)


__all__ = ["BINS", "GAUSSIAN", "MAIER_SAUPE", "PoleFit", "PoleProfile", "fit_pole_figure", "fold_profile"]
