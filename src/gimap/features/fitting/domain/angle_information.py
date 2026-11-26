"""How much a measurement at another incidence angle would add to what is known about a fitted cut.

For a fitted horizontal cut (populations, settings, the q of its points and how many detector pixels each point
averages), the counts expected at incidence angle α_i, exposure t, at the cut's exit angle (the Yoneda row of the
substrate in ``dwba`` mode) are

    μ_j(θ; α_i) = n_j · t / t_0 · A(α_i) · [ (1 − b) I_j^top(θ; α_i) + b · B(α_i) · I_j^in(θ; α_i) ],

* ``I^top``: the DWBA model of particles on the surface (``cut_model``, four paths);
* ``I^in``, ``B``: particles buried in a film of refractive index 1 − δ_f + iβ_f (the cellulose film a spray lands on),
  a fraction ``b`` of them, their number falling off with depth as exp(−z/ζ). They see the wave transmitted into the
  film, |t(α_i)|² |t(α_f)|², damped over the intensity penetration depths Λ(α) = 1 / (2k Im √(sin²α − 2δ_f + 2iβ_f)):

      B = |t_i|² |t_f|² / (1 + ζ (1/Λ_i + 1/Λ_f)),

  and their form factor at the vacuum q_z (Born; refraction of q_z in the film is left out). Below the film's
  critical angle Λ is a few nm and the buried particles fall dark; above it Λ is microns and they all scatter — the
  depth contrast of several incidence angles;
* ``n_j`` the pixels of point j, ``t_0`` the exposure the fit was made at, ``A(α_i)`` the illuminated footprint
  relative to the fitted angle, ``min(L, h / sin α_i) / min(L, h / sin α_0)`` for a beam of height h on a sample of
  length L.

The Fisher information of the parameters θ (relative units, ∂/∂ln θ), with each point's variance the Poisson count
plus a model-error floor f (Physics and Methods 9.20: beyond about 1/f² counts the model, not the photons, limits
the error),

    F_ab = Σ_j (∂μ_j/∂ln θ_a)(∂μ_j/∂ln θ_b) / (μ_j + f² μ_j²),

adds over measurements. What a candidate angle adds to the measurements made so far is the gain
log det(F_made + F_α) − log det(F_made); the Cramér–Rao bounds after it are the square roots of the diagonal of
(F_made + F_α)⁻¹ — the relative errors a joint fit of all of them could reach at best, the other parameters free.

Several candidate solutions (the modes Predict found, fits that describe the frame equally well) are told apart by
the expected χ² between their counts at an angle, Σ_j (μ_j^a − μ_j^b)² / (μ_j + f² μ_j²), μ their mean.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from . import fit_model as fm
from .cut_model import Population, Settings, evaluate

STEP = 1e-3
"""Relative step of the central differences (absolute for a parameter at 0)."""
MODEL_FLOOR = 0.05
"""Default model-error floor f per point (a relative accuracy of the model of 5 %)."""
RIDGE = 1e-9
"""Added to the diagonal of F before determinants and inverses (a parameter no measurement fixes stays finite)."""
PARAMETERS = ("radius", "radius_spread", "height", "distance", "distance_spread")
BURIAL = "burial"


@dataclass(frozen=True)
class Burial:
    """A fraction of the particles buried in a film, their number falling off with depth as exp(−z/ζ)."""

    film_delta: float
    film_beta: float
    fraction: float = 0.2
    depth_nm: float = 50.0


@dataclass(frozen=True)
class Footprint:
    """The beam's footprint on the sample: height of the beam (mm) and length of the sample along it (mm)."""

    beam_height_mm: float = 0.03
    sample_length_mm: float = 20.0

    def length(self, alpha_deg: float) -> float:
        return min(self.sample_length_mm, self.beam_height_mm / math.sin(math.radians(alpha_deg)))


@dataclass(frozen=True)
class FittedCut:
    """A fitted horizontal cut: the model and the points it was fitted at."""

    populations: tuple[Population, ...]
    settings: Settings
    """With its ``geometry`` (``dwba``: substrate δ, β; exit angle ``None``: the substrate's Yoneda row)."""
    q: np.ndarray
    """nm⁻¹, the points of the cut."""
    pixels: np.ndarray
    """Detector pixels averaged into each point (the fitted intensity is per pixel)."""
    exposure_s: float = 1.0
    burial: Optional[Burial] = None
    parameters: tuple[tuple[object, str], ...] = ()
    """``(population index or "burial", field)`` whose bounds are wanted; empty: all the cut uses."""

    def wanted(self) -> tuple[tuple[object, str], ...]:
        if self.parameters:
            return self.parameters
        chosen: list[tuple[object, str]] = []
        for index, population in enumerate(self.populations):
            for name in PARAMETERS:
                if name == "height" and population.family == "sphere":
                    continue
                if name in ("distance", "distance_spread") and not population.distance > 0:
                    continue
                if name.endswith("spread") and not getattr(population, name) > 0:
                    continue
                chosen.append((index, name))
        if self.burial is not None:
            chosen += [(BURIAL, "fraction"), (BURIAL, "depth_nm")]
        return tuple(chosen)

    def value(self, target, name: str) -> float:
        return float(getattr(self.burial if target == BURIAL else self.populations[int(target)], name))

    def with_value(self, target, name: str, value: float) -> "FittedCut":
        if target == BURIAL:
            return replace(self, burial=replace(self.burial, **{name: value}))
        populations = list(self.populations)
        populations[int(target)] = replace(populations[int(target)], **{name: value})
        return replace(self, populations=tuple(populations))

    # -- what ``expected_counts`` asks of a cut ---------------------------------------------------------------------

    @property
    def reference_deg(self) -> float:
        return float(self.settings.geometry.alpha_i_deg)

    @property
    def exit_deg(self) -> float:
        return float(self.settings.geometry.exit_angle_deg)

    @property
    def k(self) -> float:
        return float(self.settings.geometry.k)

    @property
    def background(self) -> float:
        return float(self.settings.background)

    def top(self, alpha_deg: float) -> np.ndarray:
        settings = replace(self.settings, geometry=replace(self.settings.geometry, alpha_i_deg=float(alpha_deg)))
        return evaluate(list(self.populations), settings, self.q)

    def inside(self, alpha_deg: float) -> np.ndarray:
        geometry = replace(self.settings.geometry, alpha_i_deg=float(alpha_deg), mode="born")
        born = replace(self.settings, geometry=geometry, background=0.0, tail_amplitude=0.0, diffuse_amplitude=0.0)
        return evaluate(list(self.populations), born, self.q)


@dataclass(frozen=True)
class ModelCut:
    """A horizontal cut fitted in Fitting (``fit_model.FitModel``): the same questions as ``FittedCut``. The cut is taken
    at the substrate's critical angle when the model has a substrate (the Yoneda row at every incidence angle), else at
    the fitted exit angle."""

    model: "fm.FitModel"
    q: np.ndarray
    pixels: np.ndarray
    exposure_s: float = 1.0
    burial: Optional[Burial] = None
    parameters: tuple[tuple[object, str], ...] = ()
    """``(component index or "burial", key)``; empty: every free shape and interference parameter."""

    def _global(self, key: str) -> float:
        return float(self.model.globals[key].value)

    def wanted(self) -> tuple[tuple[object, str], ...]:
        if self.parameters:
            return self.parameters
        chosen: list[tuple[object, str]] = [
            (index, key) for index, component in enumerate(self.model.components) for key in component.keys()
            if key != "Int" and component.params[key].free and component.params[key].tie is None
            and float(component.params[key].value) != 0.0]
        if self.burial is not None:
            chosen += [(BURIAL, "fraction"), (BURIAL, "depth_nm")]
        return tuple(chosen)

    def value(self, target, name: str) -> float:
        if target == BURIAL:
            return float(getattr(self.burial, name))
        return float(self.model.get((target, name)).value)

    def with_value(self, target, name: str, value: float) -> "ModelCut":
        if target == BURIAL:
            return replace(self, burial=replace(self.burial, **{name: value}))
        return replace(self, model=self.model.with_values({(target, name): value}))

    @property
    def reference_deg(self) -> float:
        return self._global("alpha_i")

    @property
    def exit_deg(self) -> float:
        substrate = self.model.substrate
        if substrate is not None:
            return float(substrate.critical_angle_deg(self._global("wavelength")))
        return self._global("alpha_f")

    @property
    def k(self) -> float:
        return 2.0 * math.pi / self._global("wavelength")

    @property
    def background(self) -> float:
        return self._global("background")

    def top(self, alpha_deg: float) -> np.ndarray:
        model = self.model.with_values({("globals", "alpha_i"): float(alpha_deg), ("globals", "alpha_f"): self.exit_deg})
        return fm.evaluate(model, self.q)

    def inside(self, alpha_deg: float) -> np.ndarray:
        model = replace(self.model, substrate=None).with_values({
            ("globals", "alpha_i"): float(alpha_deg), ("globals", "alpha_f"): self.exit_deg,
            ("globals", "refl_amp"): 0.0, ("globals", "background"): 0.0, ("globals", "res_amplitude"): 0.0,
            ("globals", "diffuse_amplitude"): 0.0})
        return fm.evaluate(model, self.q)


@dataclass(frozen=True)
class AngleScore:
    alpha_i_deg: float
    counts: float
    """Expected counts of the whole cut at this angle."""
    gain: float
    """log det(F_made + F_α) − log det(F_made): the information the angle adds (nats); with nothing made yet, log det F_α."""
    bounds: dict = field(default_factory=dict)
    """``"target:field"`` → relative Cramér–Rao bound σ/θ after adding this angle to the measurements made."""
    separation: float = float("nan")
    """Expected χ² between the closest pair of candidate solutions at this angle (NaN for one candidate)."""


# -- the expected counts -------------------------------------------------------------------------------------------


def _transmission(alpha_deg: float, delta: float, beta: float) -> complex:
    sine = math.sin(math.radians(alpha_deg))
    root = cmath.sqrt(sine * sine - 2.0 * delta + 2j * beta)
    return 2.0 * sine / (sine + root)


def penetration_nm(alpha_deg: float, delta: float, beta: float, k: float) -> float:
    """Intensity penetration depth Λ (nm) of the wave in a medium 1 − δ + iβ at grazing angle α (k in nm⁻¹)."""
    sine = math.sin(math.radians(alpha_deg))
    imaginary = cmath.sqrt(sine * sine - 2.0 * delta + 2j * beta).imag
    return math.inf if imaginary <= 0 else 1.0 / (2.0 * k * imaginary)


def burial_factor(burial: Burial, alpha_i: float, alpha_f: float, k: float) -> float:
    """B of the module text: how bright a buried particle is relative to one in vacuum at these angles."""
    t2 = abs(_transmission(alpha_i, burial.film_delta, burial.film_beta)) ** 2 * \
        abs(_transmission(alpha_f, burial.film_delta, burial.film_beta)) ** 2
    damping = sum(0.0 if not math.isfinite(depth) else 1.0 / depth
                  for depth in (penetration_nm(alpha_i, burial.film_delta, burial.film_beta, k),
                                penetration_nm(alpha_f, burial.film_delta, burial.film_beta, k)))
    return t2 / (1.0 + burial.depth_nm * damping)


def expected_counts(cut, alpha_deg: float, exposure_s: float, footprint: Footprint = Footprint()) -> np.ndarray:
    """μ_j at incidence ``alpha_deg`` for ``exposure_s`` (the module text) of a ``FittedCut`` or ``ModelCut``."""
    reference = cut.reference_deg
    area = footprint.length(alpha_deg) / footprint.length(reference) if reference > 0 else 1.0
    top = cut.top(alpha_deg)
    if cut.burial is not None and cut.burial.fraction > 0:
        factor = burial_factor(cut.burial, alpha_deg, cut.exit_deg, cut.k)
        particles = top - cut.background
        top = cut.background + (1.0 - cut.burial.fraction) * particles + cut.burial.fraction * factor * cut.inside(alpha_deg)
    return np.clip(top, 0.0, None) * np.asarray(cut.pixels, float) * (exposure_s / cut.exposure_s) * area


# -- information ------------------------------------------------------------------------------------------------------


def _variance(mu: np.ndarray, floor: float) -> np.ndarray:
    return np.maximum(mu + (floor * mu) ** 2, 1e-12)


def fisher(cut: FittedCut, alpha_deg: float, exposure_s: float, *, floor: float = MODEL_FLOOR,
           footprint: Footprint = Footprint()) -> tuple[np.ndarray, tuple[tuple[object, str], ...]]:
    """``(F, parameters)``: the Fisher matrix of one exposure in relative units (∂/∂ln θ)."""
    wanted = cut.wanted()
    mu = expected_counts(cut, alpha_deg, exposure_s, footprint)
    columns = []
    for target, name in wanted:
        value = cut.value(target, name)
        step = STEP * abs(value) if value else STEP
        plus = expected_counts(cut.with_value(target, name, value + step), alpha_deg, exposure_s, footprint)
        minus = expected_counts(cut.with_value(target, name, value - step), alpha_deg, exposure_s, footprint)
        columns.append((plus - minus) / (2.0 * step) * (value if value else 1.0))
    jacobian = np.stack(columns, axis=1) if columns else np.zeros((mu.size, 0))
    return (jacobian.T / _variance(mu, floor)) @ jacobian, wanted


def separation(candidates: Sequence[FittedCut], alpha_deg: float, exposure_s: float, *, floor: float = MODEL_FLOOR,
               footprint: Footprint = Footprint()) -> float:
    """Expected χ² between the closest pair of candidates at this angle (NaN for fewer than two)."""
    if len(candidates) < 2:
        return float("nan")
    counts = [expected_counts(cut, alpha_deg, exposure_s, footprint) for cut in candidates]
    best = math.inf
    for a in range(len(counts)):
        for b in range(a + 1, len(counts)):
            mean = 0.5 * (counts[a] + counts[b])
            best = min(best, float(np.sum((counts[a] - counts[b]) ** 2 / _variance(mean, floor))))
    return best


def _log_det(matrix: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(matrix + RIDGE * np.eye(len(matrix)))
    return float(value) if sign > 0 else float("-inf")


def score_angles(candidates: Sequence[FittedCut], angles: Sequence[float], exposure_s: float, *,
                 made: Sequence[tuple[float, float]] = (), floor: float = MODEL_FLOOR,
                 footprint: Footprint = Footprint()) -> list[AngleScore]:
    """What an exposure ``exposure_s`` at each of ``angles`` adds to the measurements ``made`` (``(angle, exposure)``),
    for the first candidate (the fit kept); with several candidates also how well each angle tells them apart."""
    cut = candidates[0]
    known = None
    for alpha, exposure in made:
        matrix, wanted = fisher(cut, alpha, exposure, floor=floor, footprint=footprint)
        known = matrix if known is None else known + matrix
    scores = []
    for alpha in angles:
        matrix, wanted = fisher(cut, alpha, exposure_s, floor=floor, footprint=footprint)
        total = matrix if known is None else known + matrix
        gain = _log_det(total) - (_log_det(known) if known is not None else 0.0)
        try:
            covariance = np.linalg.inv(total + RIDGE * np.eye(len(total)))
            bounds = {f"{target}:{name}": float(math.sqrt(max(covariance[i, i], 0.0)))
                      for i, (target, name) in enumerate(wanted)}
        except np.linalg.LinAlgError:
            bounds = {f"{target}:{name}": math.inf for target, name in wanted}
        counts = float(np.sum(expected_counts(cut, alpha, exposure_s, footprint)))
        scores.append(AngleScore(float(alpha), counts, float(gain), bounds,
                                 separation(candidates, alpha, exposure_s, floor=floor, footprint=footprint)))
    return scores


__all__ = ["AngleScore", "BURIAL", "Burial", "FittedCut", "Footprint", "MODEL_FLOOR", "ModelCut", "PARAMETERS", "STEP",
           "burial_factor", "expected_counts", "fisher", "penetration_nm", "score_angles", "separation"]
