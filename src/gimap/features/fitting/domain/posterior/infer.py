"""The posterior of a 1D cut's parameters from the proposal network, corrected by importance sampling (numpy).

For a measured cut and each composition considered (given, or the network's most probable ones):

1. N samples z ~ q(z | curve, composition) from the network (``flow.Network``);
2. per sample the cut_model terms on the measured points — every population T_j (with the sample's resolution), the
   beam tail and a constant — and their amplitudes by non-negative least squares; the likelihood is Gaussian in the
   counts' σ, with the smooth multiplicative model error of amplitude τ (``encoding.error_basis``) integrated out
   exactly (Woodbury) and the amplitudes integrated out under uniform priors on [0, ``AMPLITUDE_RANGE`` × the curve's
   maximum] (each column is 1 at its maximum): Laplace for the positive ones, a half Gaussian for those at 0;
3. weights w = L π(z)/q(z), the effective sample size (Σw)²/Σw² and the evidence Z = mean w (Dax et al. 2023).

The beam tail's width is at most ``TAIL_WIDTH_MAX`` × the window's q_min and its exponent in ``TAIL_EXPONENT``, as in
``encoding``; the weights of wider samples are 0.

The compositions' posterior is ∝ Z (each composition considered equally likely beforehand). The weighted samples are
grouped into modes (``modes``): sizes and distances apart by more than ``MODE_SPACING`` (relative) or spreads by more
than ``MODE_SPREAD`` start a mode of their own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import nnls

from ..cut_model import model as cut
from ..cut_model import resolution
from ..cut_model.geometry import Geometry
from ..substrates import Substrate
from . import encoding as E
from . import substrate_level

AMPLITUDE_RANGE = 10.0
TAIL_WIDTH_MAX = E.TAIL_WIDTH_MAX
MODE_SPACING, MODE_SPREAD = 0.1, 0.05
MAX_MODES = 4
PROPOSAL_WIDTHS = (1.0,)
"""The flow's base normal widths, equal shares of the samples (``flow.Network.sample``): above 1 the proposals reach
further from the network's most probable values. Half of 1000 samples at width 2 found more modes of curves that
the moves of ``refine`` did not, but lost more elsewhere (Physics and Methods 9.13), so the network's own width is
used."""
PER_COUNT = 1
"""When the head chooses the compositions: at least this many of every number of populations among those allowed
(``candidates``), besides its ``shortlist`` most probable."""
SIMPLE = ("sphere", "vertical_cylinder")
"""One population of each of these families is always among the candidates (``candidates``), with or without a
paracrystal as the head's most probable single population: they are fast to evaluate, and their evidence decides."""
EXPLORE_ESS = 50.0
"""The effective samples of the tempered weights from which the ``explore`` modes of a composition are found: on a
bright curve the weights sit on one or two samples, and modes of slightly lower likelihood would never be refined."""
SHARP = 0.25
"""A sample's resolution σ_q below this fraction of the smallest spacing of the measured points is not applied."""
TABLE_POINTS = 256
"""Logarithmic points (besides the measured ones) on which the terms of the other samples are convolved."""


@dataclass(frozen=True)
class Cut:
    """A measured cut in the network's window: |q| (nm⁻¹, sorted), I, σ, the vacuum q₊ = k_i + k_f of the Born term
    (0: the effective interface) and, when both angles are known, q₋ = k_i − k_f. With a known substrate, the
    measured wavelength and both angles set the four scattering paths. Otherwise the two-path model fits r."""
    q: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray
    qz: float = 0.0
    q_max: float = 1.0
    ratio: float = 1.0
    dropped: int = 0
    """Points below the window (q < q_max / RATIO_MAX), not used."""
    qz_minus: Optional[float] = None
    substrate: Optional[Substrate] = None
    dwba: Optional[Geometry] = None
    upright_heights: tuple = ()

    @property
    def level(self) -> int:
        return 4 if self.substrate is not None else 3 if self.reflected else 2 if self.qz > 0 else 1

    def substrate_record(self) -> dict:
        if self.substrate is None:
            return {}
        geometry = self.dwba
        return dict(substrate=self.substrate.to_dict(), substrate_label=self.substrate.label,
                    wavelength_nm=geometry.wavelength_nm, alpha_i_deg=geometry.alpha_i_deg,
                    alpha_f_deg=geometry.alpha_f_deg, delta=geometry.delta, beta=geometry.beta)

    def upright_height(self, slot: int):
        return next(((h, s) for n, h, s in self.upright_heights if n == slot + 1), None)

    @property
    def reflected(self) -> bool:
        """Two paths (q₋ known and q₊ > 0)."""
        return self.substrate is None and self.qz_minus is not None and self.qz > 0

    @classmethod
    def prepare(cls, q, intensity, sigma=None, qz: float = 0.0, qz_minus: Optional[float] = None, *,
                substrate=None, grazing=None, upright_heights=None) -> "Cut":
        """With a substrate, derive both vertical wave vectors from grazing (wavelength in nm, angles in degrees)."""
        q = np.abs(np.asarray(q, float)).ravel()
        y = np.asarray(intensity, float).ravel()
        s = None if sigma is None else np.asarray(sigma, float).ravel()
        keep = np.isfinite(q) & np.isfinite(y) & (q > 0)
        if s is not None:
            keep &= np.isfinite(s) & (s > 0)
        q, y = q[keep], y[keep]
        s = s[keep] if s is not None else _noise(q, y)
        order = np.argsort(q)
        q, y, s = q[order], y[order], s[order]
        q_max, ratio = E.window(q)
        inside = q >= q_max / ratio * (1 - 1e-12)
        if substrate is not None:
            substrate, geometry = substrate_level.resolve(substrate, grazing)
            return cls(q[inside], y[inside], s[inside], geometry.qz_born, q_max, ratio, int(np.sum(~inside)),
                       geometry.k_i - geometry.k_f, substrate, geometry, substrate_level.heights_by_slot(upright_heights))
        if qz_minus is not None and not abs(float(qz_minus)) < float(qz):
            raise ValueError("k_i − k_f must lie between −(k_i + k_f) and k_i + k_f")
        return cls(q[inside], y[inside], s[inside], float(qz), q_max, ratio, int(np.sum(~inside)),
                   None if qz_minus is None or qz <= 0 else float(qz_minus))

    def angles(self) -> tuple[float, float]:
        """Measured (α_i, α_f) for DWBA, otherwise angles at λ = 0.1 nm giving the cut's k_i ± k_f."""
        if self.dwba is not None:
            return self.dwba.alpha_i_deg, self.dwba.alpha_f_deg
        k = 2 * math.pi / 0.1
        minus = self.qz_minus if self.reflected else 0.0
        return (math.degrees(math.asin(0.5 * (self.qz + minus) / k)),
                math.degrees(math.asin(0.5 * (self.qz - minus) / k)))

    def geometry(self, reflection: complex = 0j) -> Geometry:
        """Select the interface, Born, two-path or fixed four-term model. Only level 3 takes a free reflection."""
        if self.dwba is not None:
            return self.dwba
        if self.qz <= 0:
            return Geometry()
        alpha_i, alpha_f = self.angles()
        if not self.reflected:
            return Geometry("born", 0.1, alpha_i, alpha_f)
        return Geometry("two_path", 0.1, alpha_i, alpha_f, abs(reflection), math.atan2(reflection.imag, reflection.real))


def _noise(q, y) -> np.ndarray:
    """σ of a cut given without one: a relative error from the scatter of ln I about its local trend (at least 1 %)."""
    order = np.argsort(q)
    log_y = np.log(np.maximum(y[order], 1e-300))
    second = log_y[2:] - 2 * log_y[1:-1] + log_y[:-2]
    relative = max(0.01, float(np.median(np.abs(second))) / (0.6745 * math.sqrt(6))) if second.size else 0.05
    return relative * np.maximum(np.abs(y), 1e-300)


def physical(values, q_max: float) -> np.ndarray:
    """Values of ``encoding`` (dimensionless) in nm and nm⁻¹: R, h, D, w and σ_q scaled by q_max (copy)."""
    out = np.array(values, float)
    for slot in range(E.SLOTS):
        j = slot * E.PER_SLOT
        out[..., j] = out[..., j] / q_max
        out[..., j + 2] = 2.0 * out[..., j] * out[..., j + 2]          # h (nm) from h/2R
        out[..., j + 4] = 2.0 * out[..., j] * out[..., j + 4]          # D (nm) from D/2R
    g = E.GLOBAL.start
    out[..., g] = out[..., g] * q_max
    out[..., g + 2] = out[..., g + 2] * q_max
    return out


def dimensionless(values, q_max: float) -> np.ndarray:
    """The inverse of ``physical``."""
    out = np.array(values, float)
    for slot in range(E.SLOTS):
        j = slot * E.PER_SLOT
        out[..., j + 2] = out[..., j + 2] / (2.0 * out[..., j])
        out[..., j + 4] = out[..., j + 4] / (2.0 * out[..., j])
        out[..., j] = out[..., j] * q_max
    g = E.GLOBAL.start
    out[..., g] = out[..., g] / q_max
    out[..., g + 2] = out[..., g + 2] / q_max
    return out


PHYSICAL_NAMES = tuple(n.replace("h_over_2R", "h").replace("D_over_2R", "D") for n in E.NAMES)


class Likelihood:
    """log p(cut | values, composition) with the amplitudes and the model error integrated out."""

    def __init__(self, data: Cut, composition: int, law: str = "number_gauss", coupling: str = "lma"):
        self.data, self.composition, self.law, self.coupling = data, int(composition), law, coupling
        self.types = E.composition_types(composition)
        self.height_fractions = substrate_level.height_fractions(data, self.types, law, coupling)
        q = data.q
        self.position = np.log(q / (data.q_max / data.ratio)) / np.log(data.ratio)
        self.basis = E.error_basis(self.position)  # (8, n)
        self.log_norm = -float(np.sum(np.log(data.sigma))) - 0.5 * q.size * math.log(2 * math.pi)
        steps = np.diff(np.unique(q))
        self.spacing = float(steps.min()) if steps.size else 0.0

    def columns(self, values) -> np.ndarray:
        """(B, K + 2, n) of physical values (B, DIMS): each population's T on the measured points with the sample's
        resolution (normalised to its maximum), the beam tail, 1."""
        v = np.atleast_2d(np.asarray(values, float))
        rows, n, k = v.shape[0], self.data.q.size, len(self.types)
        g = E.GLOBAL.start
        out = np.empty((rows, k + 2, n))
        geometry = self.data.geometry()
        sigma = v[:, g + 2]
        sharp = sigma <= SHARP * self.spacing  # the resolution does not matter between the points: no convolution
        groups = [(np.flatnonzero(sharp), self.data.q), (np.flatnonzero(~sharp), None)]
        for slot, (family, structure) in enumerate(self.types):
            j = slot * E.PER_SLOT
            for rows_, grid in groups:
                if not rows_.size:
                    continue
                if grid is None:
                    grid = resolution.table_grid(self.data.q, float(np.max(sigma[rows_])), TABLE_POINTS)
                kwargs = dict(law=self.law, coupling=self.coupling)
                if family == "sphere" and self.data.reflected:  # every sample its own r
                    terms = np.stack([cut.population_terms(
                        family, grid, radius=v[b, j], radius_spread=v[b, j + 1],
                        distance=v[b, j + 4] if structure else 0.0, distance_spread=v[b, j + 5] if structure else 0.0,
                        geometry=self.data.geometry(complex(v[b, g + 4], v[b, g + 5])), **kwargs)[0] for b in rows_])
                elif slot in self.height_fractions:
                    terms = substrate_level.upright_da_terms(
                        grid, v[rows_, j], v[rows_, j + 1], v[rows_, j + 4], v[rows_, j + 5], self.law,
                        geometry, self.height_fractions[slot])
                else:
                    terms = cut.population_terms(
                        family, grid, radius=v[rows_, j], radius_spread=v[rows_, j + 1],
                        height=v[rows_, j + 2] if family == "cylinder" else 10.0,
                        height_spread=v[rows_, j + 3] if family == "cylinder" else 0.0,
                        distance=v[rows_, j + 4] if structure else 0.0,
                        distance_spread=v[rows_, j + 5] if structure else 0.0, geometry=geometry, **kwargs)
                if grid is self.data.q:
                    out[rows_, slot] = terms
                else:
                    for i, b in enumerate(rows_):
                        out[b, slot] = resolution.tabulated(self.data.q, grid, terms[i], float(sigma[b]))[0]
            out[:, slot] /= np.maximum(out[:, slot].max(axis=1, keepdims=True), 1e-300)
        for b in range(rows):
            out[b, k] = cut.tail(self.data.q, float(v[b, g]), float(v[b, g + 1]))
        out[:, k + 1] = 1.0
        return out

    def model_curves(self, values):
        """(columns (B, K + 2, n), amplitudes (B, K + 2) by non-negative least squares in σ, model curves (B, n))."""
        x = self.columns(values)
        y, s = self.data.intensity, self.data.sigma
        scaled = x / s[None, None, :]
        amplitudes = np.zeros(x.shape[:2])
        for b in range(x.shape[0]):
            norms = np.maximum(np.linalg.norm(scaled[b], axis=1), 1e-300)
            solution, _ = nnls((scaled[b] / norms[:, None]).T, y / s)
            amplitudes[b] = solution / norms
        return x, amplitudes, np.einsum("bm,bmn->bn", amplitudes, x)

    def __call__(self, values) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(log marginal likelihood, amplitudes (B, K + 2), χ² at the amplitudes) of physical values (B, DIMS)."""
        values = np.atleast_2d(values)
        x, amplitudes, model = self.model_curves(values)
        y, s = self.data.intensity, self.data.sigma
        m = x.shape[1]
        scaled = x / s[None, None, :]
        r = (y[None, :] - model) / s[None, :]
        chi2 = np.sum(r * r, axis=1)
        tau = values[:, E.GLOBAL.start + 3]
        u = self.basis[None, :, :] * (model / s[None, :])[:, None, :]  # (B, 8, n)
        gram = np.einsum("bkn,bln->bkl", u, u)
        inner = np.eye(E.ERROR_BASIS)[None] / (tau[:, None, None] ** 2) + gram
        projected = np.einsum("bkn,bn->bk", u, r)
        solved = np.linalg.solve(inner, projected[..., None])[..., 0]
        _sign, log_det = np.linalg.slogdet(np.eye(E.ERROR_BASIS)[None] + tau[:, None, None] ** 2 * gram)
        log_like = self.log_norm - 0.5 * (chi2 - np.sum(projected * solved, axis=1)) - 0.5 * log_det
        # the amplitudes, uniform on [0, top]: Laplace at the positive ones, a half Gaussian at those at 0
        top = AMPLITUDE_RANGE * float(np.max(np.abs(y)))
        positive = amplitudes > 0
        full = np.einsum("bmn,bln->bml", scaled, scaled)
        hessian = np.where(positive[:, :, None] & positive[:, None, :], full, 0.0)
        hessian += np.eye(m)[None] * (~positive)[:, :, None]
        _sign, log_det_h = np.linalg.slogdet(hessian)
        diagonal = np.maximum(np.einsum("bmm->bm", full), 1e-300)
        at_zero = np.where(positive, 0.0, 0.5 * math.log(0.5 * math.pi) - 0.5 * np.log(diagonal))
        log_marginal = (log_like - m * math.log(top) + 0.5 * positive.sum(axis=1) * math.log(2 * math.pi)
                        - 0.5 * log_det_h + np.sum(at_zero, axis=1))
        wide = values[:, E.GLOBAL.start] > TAIL_WIDTH_MAX * self.data.q_max / self.data.ratio
        return np.where(wide, -np.inf, log_marginal), amplitudes, chi2



@dataclass
class Composition:
    index: int
    name: str
    samples: np.ndarray
    """Physical values of the samples (N, DIMS), NaN where inactive."""
    z: np.ndarray
    log_weights: np.ndarray
    amplitudes: np.ndarray
    chi2: np.ndarray
    log_evidence: float
    ess: float
    head_log_prob: float
    probability: float = 0.0
    modes: list = field(default_factory=list)
    explore: list = field(default_factory=list)
    """Modes of the tempered weights (``tempered``): more starting points for least squares."""

    @property
    def weights(self) -> np.ndarray:
        w = np.exp(self.log_weights - np.max(self.log_weights))
        return w / w.sum()


@dataclass
class Mode:
    composition: int
    weight: float
    """Share of the composition's posterior mass."""
    best: np.ndarray
    """Physical values of the sample with the highest weight in the mode."""
    amplitudes: np.ndarray
    chi2_reduced: float
    median: np.ndarray
    low68: np.ndarray
    high68: np.ndarray
    low95: np.ndarray
    high95: np.ndarray
    members: int


def _weighted_quantiles(values, weights, levels) -> np.ndarray:
    order = np.argsort(values, axis=0)
    out = np.empty((len(levels), values.shape[1]))
    for d in range(values.shape[1]):
        v, w = values[order[:, d], d], weights[order[:, d]]
        cum = np.cumsum(w) - 0.5 * w
        cum /= w.sum()
        out[:, d] = np.interp(levels, cum, v)
    return out


def _distance(a, b, types) -> np.ndarray:
    """Max-norm distance of samples ``a`` (N, DIMS) from one sample ``b`` in units of MODE_SPACING / MODE_SPREAD."""
    worst = np.zeros(a.shape[0])
    for slot, (family, structure) in enumerate(types):
        j = slot * E.PER_SLOT
        worst = np.maximum(worst, np.abs(np.log(a[:, j] / b[j])) / MODE_SPACING)
        worst = np.maximum(worst, np.abs(a[:, j + 1] - b[j + 1]) / MODE_SPREAD)
        if family == "cylinder":
            worst = np.maximum(worst, np.abs(np.log(a[:, j + 2] / b[j + 2])) / MODE_SPACING)
        if structure:
            worst = np.maximum(worst, np.abs(np.log(a[:, j + 4] / b[j + 4])) / MODE_SPACING)
            worst = np.maximum(worst, np.abs(a[:, j + 5] - b[j + 5]) / MODE_SPREAD)
    return worst


def tempered(log_weights, ess: float) -> np.ndarray:
    """The normalised weights raised to the largest power β ≤ 1 that gives ``ess`` effective samples."""
    shifted = np.asarray(log_weights, float) - np.max(log_weights)
    shifted = np.where(np.isfinite(shifted), shifted, -np.inf)

    def weights(beta):
        w = np.exp(beta * shifted)
        return w / w.sum()

    def effective(beta):
        w = weights(beta)
        return 1.0 / np.sum(w * w)

    low, high = 0.0, 1.0
    if effective(1.0) < ess:
        for _ in range(40):
            middle = 0.5 * (low + high)
            low, high = (middle, high) if effective(middle) >= ess else (low, middle)
        return weights(low)
    return weights(1.0)


def modes(result: Composition, points: int, weights: Optional[np.ndarray] = None) -> list[Mode]:
    """The weighted samples grouped around the heaviest ones (at most MAX_MODES, each ≥ 1 % of the mass); ``weights``
    instead of the composition's own (e.g. tempered)."""
    weights = result.weights if weights is None else weights
    types = E.composition_types(result.index)
    order = np.argsort(-weights)
    seeds = []
    for i in order:
        if weights[i] < 1e-4 * weights[order[0]] or len(seeds) >= MAX_MODES:
            break
        if not seeds or np.all(_distance(result.samples[seeds], result.samples[i], types) > 1.0):
            seeds.append(i)
    distances = np.stack([_distance(result.samples, result.samples[s], types) for s in seeds])
    owner = np.argmin(distances, axis=0)
    out = []
    active = E.active(result.index)[0]
    for m, seed in enumerate(seeds):
        members = owner == m
        mass = float(weights[members].sum())
        if mass < 0.01:
            continue
        quantiles = np.full((5, E.DIMS), np.nan)
        quantiles[:, active] = _weighted_quantiles(result.samples[members][:, active], weights[members],
                                                   [0.5, 0.16, 0.84, 0.025, 0.975])
        out.append(Mode(result.index, mass, result.samples[seed], result.amplitudes[seed],
                        float(result.chi2[seed]) / max(1, points - int(active.sum()) - len(types) - 2), *quantiles,
                        int(members.sum())))
    total = sum(m.weight for m in out)
    for m in out:
        m.weight /= total
    return sorted(out, key=lambda m: -m.weight)


def sample_composition(network, h, data: Cut, composition: int, count: int, rng, law: str = "number_gauss",
                       coupling: str = "lma", head_log_prob: float = 0.0) -> Composition:
    """Importance-sampled posterior of one composition."""
    z, log_q = network.sample(network.context(h, composition), composition, count, rng, widths=PROPOSAL_WIDTHS)
    values = E.from_z(z, np.full(count, data.ratio), composition)
    samples = physical(values, data.q_max)
    likelihood = Likelihood(data, composition, law, coupling)
    log_like, amplitudes, chi2 = likelihood(samples)
    log_w = log_like + E.log_prior(z, composition) - log_q
    log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
    top = np.max(log_w)
    w = np.exp(log_w - top)
    ess = float(w.sum() ** 2 / np.sum(w * w))
    log_evidence = float(top + np.log(np.mean(w)))
    return Composition(composition, E.composition_name(composition), samples, z, log_w, amplitudes, chi2,
                       log_evidence, ess, head_log_prob)


def conditions(data: Cut) -> np.ndarray:
    """``encoding.conditions`` of a cut (1, CONDITIONS)."""
    return E.conditions([data.ratio], [data.qz / data.q_max],
                        [data.qz_minus / data.q_max if data.reflected or data.substrate is not None else np.nan])


def candidates(network, h, allowed: Optional[Sequence[int]] = None, count: int = 4) -> list[tuple[int, float]]:
    """The ``count`` most probable compositions under the head among ``allowed`` (default all), and the
    ``PER_COUNT`` most probable of every number of populations among them: (index, log p), most probable first."""
    log_probs = network.composition_log_probs(h)[0]
    pool = np.arange(len(E.COMPOSITIONS)) if allowed is None else np.asarray(list(allowed), int)
    ranked = pool[np.argsort(-log_probs[pool])]
    best = list(ranked[:count])
    for k in sorted({len(E.COMPOSITIONS[c]) for c in ranked}):
        best += [c for c in ranked if len(E.COMPOSITIONS[c]) == k and c not in best][:max(
            0, PER_COUNT - sum(len(E.COMPOSITIONS[c]) == k for c in best))]
    # The simplest models always compete: on measured cuts the head favours two populations (a beam tail or a
    # roughness reads as one more), and a single sphere population that the evidence would prefer was not tried.
    single = [c for c in ranked if len(E.COMPOSITIONS[c]) == 1]
    if single and SIMPLE:
        structure = E.composition_types(single[0])[0][1]  # with or without a paracrystal, as the head's best single
        for family in SIMPLE:
            simple = E.composition_index([(family, structure)])
            if simple in pool and simple not in best:
                best.append(simple)
    best.sort(key=lambda c: -log_probs[c])
    return [(int(c), float(log_probs[c])) for c in best]


def infer(network, data: Cut, compositions: Optional[Sequence[int]] = None, *, allowed=None, shortlist: int = 4,
          samples: int = 2000, cylinder_samples: int = 300, seed: int = 0, law: str = "number_gauss",
          coupling: str = "lma", progress=None) -> list[Composition]:
    """The posterior of each composition (given, or the head's ``shortlist`` among ``allowed``), most probable
    first; ``probability`` from the evidences, ``modes`` of each. ``progress``: given ``(done, total)`` before the
    first composition is sampled and after each (it changes nothing that is computed)."""
    rng = np.random.default_rng(seed)
    binned = E.bin_curve(data.q, data.intensity, data.sigma, data.q_max, data.ratio)
    h = network.summary(E.features(*(a[None] for a in binned)), conditions(data))
    if compositions is None:
        chosen = candidates(network, h, allowed, shortlist)
    else:
        log_probs = network.composition_log_probs(h)[0]
        chosen = [(int(c), float(log_probs[c])) for c in compositions]
    out = []
    if progress is not None:
        progress(0, len(chosen))
    for c, head in chosen:
        has_cylinder = any(f == "cylinder" for f, _s in E.composition_types(c))
        out.append(sample_composition(network, h, data, c, cylinder_samples if has_cylinder else samples, rng, law,
                                      coupling, head))
        if progress is not None:
            progress(len(out), len(chosen))
    evidences = np.array([r.log_evidence for r in out])
    probabilities = np.exp(evidences - evidences.max())
    probabilities /= probabilities.sum()
    for r, p in zip(out, probabilities):
        r.probability = float(p)
        r.modes = modes(r, data.q.size)
        r.explore = modes(r, data.q.size, tempered(r.log_weights, EXPLORE_ESS))
    return sorted(out, key=lambda r: -r.probability)


__all__ = ["AMPLITUDE_RANGE", "EXPLORE_ESS", "PROPOSAL_WIDTHS", "SIMPLE", "TAIL_WIDTH_MAX", "Composition", "Cut", "Likelihood",
           "Mode", "PHYSICAL_NAMES", "candidates", "conditions", "dimensionless", "infer", "modes", "physical",
           "sample_composition", "tempered"]
