"""The posterior around a least-squares solution: importance sampling with a Student-t proposal at the optimum
(numpy).

For a measured cut the likelihood is far narrower than any proposal amortised over all curves, so the network's own
samples carry the weight on one or two of them (``infer``). Here the proposal is local: at a refined solution z* (in the
network's coordinates z), the Gauss–Newton curvature of the σ-weighted residuals (their Jacobian J by finite
differences, the amplitudes re-solved at every point). Coordinates the curve constrains (JᵀJ on the diagonal above
``INFORMED``) get a multivariate t with ``DOF`` degrees of freedom and scale (JᵀJ + P)⁻¹, P = ``PRIOR_PRECISION``·I,
inflated by ``INFLATE``, centred at z* (within ±``CENTRE_MAX``); each of the others — a beam tail that is not there, a
resolution at the bottom of its range, the model error's τ, which the residuals do not see — from a piecewise-constant
density over ``CELLS`` cells of its unit range, set by the likelihood along that coordinate at the cells' centres (the
others at z*), with a floor of ``FLOOR`` of the largest. Later rounds adapt a t over all coordinates together to
the samples so far (``local_posterior``). The samples are weighted by L·π/q with the same likelihood as ``infer``
(the model error and the amplitudes integrated out): the intervals, the effective sample size and the evidence of
the mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

from . import encoding as E
from .infer import Cut, Likelihood, dimensionless, physical

PRIOR_PRECISION = 0.3
INFLATE = 1.4
INFORMED = 1.0
CENTRE_MAX = 6.0
CELLS, FLOOR = 9, 0.02
DOF = 5
ROUNDS, ESS_TARGET = 8, 200
ADAPT = 2
"""Effective samples per coordinate of the tempered weights that give a later round its mean and covariance."""
Z_MAX, RIDGE = 10.0, 1e-4
"""|z| of a sample in those moments at most ``Z_MAX``; ``RIDGE`` added to the covariance's diagonal."""
STEP = 1e-3


@dataclass
class Local:
    """The importance-sampled posterior of one mode."""
    composition: int
    samples: np.ndarray
    """Physical values (N, DIMS), NaN where inactive."""
    log_weights: np.ndarray
    amplitudes: np.ndarray
    chi2: np.ndarray
    log_evidence: float
    ess: float
    centre: np.ndarray
    """Physical values at the optimum."""

    @property
    def weights(self) -> np.ndarray:
        w = np.exp(self.log_weights - np.max(self.log_weights))
        return w / w.sum()

    def tempered(self, ess_min: float) -> np.ndarray:
        """Normalised weights w^β with the largest β ≤ 1 whose effective sample size is at least ``ess_min``.

        With few effective samples the weighted quantiles sit on those few and the intervals come out too narrow;
        tempering moves them towards the proposal, which is wider than the posterior by construction (a t with
        ``DOF`` degrees of freedom, scale ×``INFLATE``)."""
        return _temper(self.log_weights, ess_min)

    def quantiles(self, levels, ess_min: float = 0.0) -> np.ndarray:
        """Weighted quantiles (len(levels), DIMS) of the active values; NaN elsewhere. ``ess_min``: the weights are
        tempered (``tempered``) when fewer samples than that are effective."""
        out = np.full((len(levels), E.DIMS), np.nan)
        active = E.active(self.composition)[0]
        w = self.tempered(ess_min) if ess_min > 0 else self.weights
        for d in np.flatnonzero(active):
            order = np.argsort(self.samples[:, d])
            cum = np.cumsum(w[order]) - 0.5 * w[order]
            out[:, d] = np.interp(levels, cum / w.sum(), self.samples[order, d])
        return out


def _temper(log_weights, ess_min: float) -> np.ndarray:
    """Normalised weights w^β, β the largest ≤ 1 that gives at least ``ess_min`` effective samples (β = 0, equal
    weights, when there are no more finite ones than that)."""
    log_weights = np.asarray(log_weights, float)
    finite = np.isfinite(log_weights)
    log_w = np.where(finite, log_weights - np.max(log_weights), -np.inf)

    def weights(beta):
        w = np.where(finite, np.exp(beta * np.where(finite, log_w, 0.0)), 0.0)
        return w / w.sum()

    def ess(w):
        return 1.0 / np.sum(w * w)

    if ess(weights(1.0)) >= ess_min or finite.sum() <= ess_min:
        return weights(1.0) if ess(weights(1.0)) >= ess_min else weights(0.0)
    low, high = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (low + high)
        low, high = (mid, high) if ess(weights(mid)) >= ess_min else (low, mid)
    return weights(low)


def _residuals(likelihood: Likelihood, z, data: Cut, composition: int) -> np.ndarray:
    values = physical(E.from_z(z, np.full(z.shape[0], data.ratio), composition), data.q_max)
    _x, _a, model = likelihood.model_curves(values)
    return (data.intensity[None, :] - model) / data.sigma[None, :]


def curvature(likelihood: Likelihood, z_star, data: Cut, composition: int, dims) -> np.ndarray:
    """JᵀJ of the σ-weighted residuals over the coordinates ``dims`` at z* (forward differences in z)."""
    rows = np.repeat(np.asarray(z_star, float)[None, :], len(dims) + 1, axis=0)
    for i, d in enumerate(dims):
        rows[i + 1, d] += STEP
    r = _residuals(likelihood, rows, data, composition)
    jacobian = (r[1:] - r[:1]) / STEP  # (len(dims), n)
    return jacobian @ jacobian.T


def _scan(likelihood: Likelihood, z_star, data: Cut, composition: int, dims) -> list:
    """Per coordinate of ``dims``, the probabilities of ``CELLS`` cells of its unit range from the likelihood at their
    centres (the others at z*), at least ``FLOOR`` of the largest."""
    centres = (np.arange(CELLS) + 0.5) / CELLS
    rows = np.repeat(np.asarray(z_star, float)[None, :], CELLS * len(dims), axis=0)
    for i, d in enumerate(dims):
        rows[i * CELLS:(i + 1) * CELLS, d] = np.log(centres) - np.log1p(-centres)
    values = physical(E.from_z(rows, np.full(rows.shape[0], data.ratio), composition), data.q_max)
    log_like = likelihood(values)[0].reshape(len(dims), CELLS)
    out = []
    for row in log_like:
        p = np.exp(row - np.max(row)) if np.any(np.isfinite(row)) else np.ones(CELLS)
        p = np.maximum(np.nan_to_num(p), FLOOR)
        out.append(p / p.sum())
    return out


def _t_log_density(x, scale_inverse, log_det_scale, dof):
    d = x.shape[1]
    quad = np.sum((x @ scale_inverse.T) ** 2, axis=1)
    return (gammaln(0.5 * (dof + d)) - gammaln(0.5 * dof) - 0.5 * d * math.log(dof * math.pi) - log_det_scale
            - 0.5 * (dof + d) * np.log1p(quad / dof))


def _scan_log_density(z, others, scan) -> np.ndarray:
    """log density (in z) of the scanned coordinates ``others`` under their piecewise-constant densities ``scan``."""
    out = np.zeros(z.shape[0])
    for d, cell_p in zip(others, scan):
        v = np.clip(0.5 * (1.0 + np.tanh(0.5 * z[:, d])), 1e-9, 1 - 1e-9)
        cell = np.minimum((v * CELLS).astype(int), CELLS - 1)
        out += np.log(cell_p[cell] * CELLS) + np.log(v * (1 - v))
    return out


def _t_sample(rng, mean, covariance, count: int) -> np.ndarray:
    root = _root(covariance)
    return mean[None, :] + (rng.standard_normal((count, mean.size)) @ root.T) / np.sqrt(
        rng.chisquare(DOF, count) / DOF)[:, None]


def _t_log(x, mean, covariance) -> np.ndarray:
    root = _root(covariance)
    return _t_log_density(x - mean[None, :], np.linalg.inv(root), float(np.sum(np.log(np.diag(root)))), DOF)


def local_posterior(data: Cut, composition: int, centre, tau: float, rng: np.random.Generator, count: int = 600,
                    law: str = "number_gauss", coupling: str = "lma", rounds: int = ROUNDS) -> Local:
    """The posterior of the mode at the physical values ``centre`` (DIMS,; its model error τ given separately).

    Adaptive (population Monte Carlo), at most ``rounds`` rounds of ``count`` samples, until ``ESS_TARGET`` of them
    are effective. The first round is the proposal at the optimum (the module's docstring). Each later one is a t
    over all coordinates together, the scanned ones and τ included, with the mean and the covariance (inflated by
    ``INFLATE``) of all samples so far under their weights tempered to ``ADAPT`` effective samples per coordinate:
    it follows a posterior that lies away from the optimum or along a direction the first proposal has no width in
    (a spread at its bound that trades against the radius, a curve whose model error moves the optimum). All
    samples are weighted against the mixture of the rounds' proposals."""
    likelihood = Likelihood(data, composition, law, coupling)
    active = E.active(composition)[0]
    centre = np.array(centre, float)
    centre[E.GLOBAL.start + 3] = tau
    unit = np.clip(E.to_unit(dimensionless(centre[None, :], data.q_max), [data.ratio], composition), 1e-6, 1 - 1e-6)
    z_star = np.where(active, np.log(unit) - np.log1p(-unit), 0.0)[0]
    tau_dim = E.GLOBAL.start + 3
    dims = [d for d in np.flatnonzero(active) if d != tau_dim]
    every = dims + [tau_dim]
    gram = curvature(likelihood, z_star, data, composition, dims)
    # A curve the model does not describe within σ (χ²/N > 1): its model error widens the likelihood beyond the
    # curvature of the σ-weighted residuals, so the proposal is widened by √(χ²/N), as least squares scales its errors.
    residual = _residuals(likelihood, z_star[None, :], data, composition)[0]
    gram = gram / max(1.0, float(residual @ residual) / max(1, residual.size - len(dims)))
    informed = [i for i, d in enumerate(dims) if gram[i, i] > INFORMED]
    chosen = [dims[i] for i in informed]
    others = [d for i, d in enumerate(dims) if i not in informed] + [tau_dim]
    scan = _scan(likelihood, z_star, data, composition, others)
    edges = np.linspace(0.0, 1.0, CELLS + 1)
    first_mean = first = None
    if informed:
        precision = gram[np.ix_(informed, informed)] + PRIOR_PRECISION * np.eye(len(informed))
        first = np.linalg.inv(precision) * INFLATE**2
        first_mean = np.clip(z_star[chosen], -CENTRE_MAX, CENTRE_MAX)

    def first_log_q(z):
        out = _scan_log_density(z, others, scan)
        return out + _t_log(z[:, chosen], first_mean, first) if informed else out

    later = []  # (mean, covariance) over ``every``
    z_all, log_like_all, amplitudes_all, chi2_all = [], [], [], []
    for round_ in range(max(1, rounds)):
        z = np.zeros((count, E.DIMS))
        if later:
            z[:, every] = _t_sample(rng, *later[-1], count)
        else:
            for d, cell_p in zip(others, scan):
                cell = rng.choice(CELLS, count, p=cell_p)
                v = np.clip(edges[cell] + rng.uniform(size=count) / CELLS, 1e-9, 1 - 1e-9)
                z[:, d] = np.log(v) - np.log1p(-v)
            if informed:
                z[:, chosen] = _t_sample(rng, first_mean, first, count)
        samples = physical(E.from_z(z, np.full(count, data.ratio), composition), data.q_max)
        log_like, amplitudes, chi2 = likelihood(samples)
        z_all.append(z)
        log_like_all.append(log_like)
        amplitudes_all.append(amplitudes)
        chi2_all.append(chi2)
        z = np.concatenate(z_all)
        parts = np.array([first_log_q(z)] + [_t_log(z[:, every], m, c) for m, c in later])
        top = np.max(parts, axis=0)
        log_q = top + np.log(np.mean(np.exp(parts - top), axis=0))
        log_w = np.concatenate(log_like_all) + E.log_prior(z, composition) - log_q
        log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
        w = np.exp(log_w - np.max(log_w))
        ess = float(w.sum() ** 2 / np.sum(w * w))
        if ess >= ESS_TARGET or round_ >= rounds - 1:
            break
        w = _temper(log_w, ADAPT * len(every))
        moved = np.clip(z[:, every], -Z_MAX, Z_MAX)
        mean = w @ moved
        spread = (moved - mean).T @ ((moved - mean) * w[:, None])
        later.append((mean, 0.5 * (spread + spread.T) * INFLATE**2 + RIDGE * np.eye(len(every))))
    top = np.max(log_w)
    samples = physical(E.from_z(z, np.full(z.shape[0], data.ratio), composition), data.q_max)
    return Local(composition, samples, log_w, np.concatenate(amplitudes_all), np.concatenate(chi2_all),
                 float(top + np.log(np.mean(np.exp(log_w - top)))), ess,
                 physical(E.from_z(z_star[None, :], [data.ratio], composition), data.q_max)[0])


def _root(covariance):
    return np.linalg.cholesky(0.5 * (covariance + covariance.T) + 1e-12 * np.eye(covariance.shape[0]))


__all__ = ["DOF", "ESS_TARGET", "INFLATE", "INFORMED", "Local", "PRIOR_PRECISION", "ROUNDS", "curvature",
           "local_posterior"]
