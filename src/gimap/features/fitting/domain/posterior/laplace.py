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
others at z*), with a floor of ``FLOOR`` of the largest. The samples are weighted by L·π/q with the same likelihood as ``infer`` (the model error and the
amplitudes integrated out): the intervals, the effective sample size and the evidence of the mode.
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
ROUNDS, ESS_TARGET = 3, 200
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

    def quantiles(self, levels) -> np.ndarray:
        """Weighted quantiles (len(levels), DIMS) of the active values; NaN elsewhere."""
        out = np.full((len(levels), E.DIMS), np.nan)
        active = E.active(self.composition)[0]
        w = self.weights
        for d in np.flatnonzero(active):
            order = np.argsort(self.samples[:, d])
            cum = np.cumsum(w[order]) - 0.5 * w[order]
            out[:, d] = np.interp(levels, cum / w.sum(), self.samples[order, d])
        return out


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


def local_posterior(data: Cut, composition: int, centre, tau: float, rng: np.random.Generator, count: int = 600,
                    law: str = "number_gauss", coupling: str = "lma") -> Local:
    """The posterior of the mode at the physical values ``centre`` (DIMS,; its model error τ given separately).

    Adaptive (population Monte Carlo): while the effective sample size is below ``ESS_TARGET``, up to ``ROUNDS``
    rounds of ``count`` more samples from a t at the weighted mean of the informed coordinates, with half their
    weighted covariance and half the first scale; all samples are weighted against the mixture of the rounds'
    proposals (the uninformed coordinates keep their scan densities)."""
    likelihood = Likelihood(data, composition, law, coupling)
    active = E.active(composition)[0]
    centre = np.array(centre, float)
    centre[E.GLOBAL.start + 3] = tau
    unit = np.clip(E.to_unit(dimensionless(centre[None, :], data.q_max), [data.ratio], composition), 1e-6, 1 - 1e-6)
    z_star = np.where(active, np.log(unit) - np.log1p(-unit), 0.0)[0]
    tau_dim = E.GLOBAL.start + 3
    dims = [d for d in np.flatnonzero(active) if d != tau_dim]
    gram = curvature(likelihood, z_star, data, composition, dims)
    informed = [i for i, d in enumerate(dims) if gram[i, i] > INFORMED]
    chosen = [dims[i] for i in informed]
    others = [d for i, d in enumerate(dims) if i not in informed] + [tau_dim]
    scan = _scan(likelihood, z_star, data, composition, others)
    edges = np.linspace(0.0, 1.0, CELLS + 1)
    first = None
    if informed:
        precision = gram[np.ix_(informed, informed)] + PRIOR_PRECISION * np.eye(len(informed))
        first = np.linalg.inv(precision) * INFLATE**2
    proposals = [(np.clip(z_star[chosen], -CENTRE_MAX, CENTRE_MAX), first)] if informed else [(None, None)]
    z_all, log_like_all, amplitudes_all, chi2_all, log_other_all = [], [], [], [], []
    for round_ in range(ROUNDS):
        mean, covariance = proposals[-1]
        z = np.zeros((count, E.DIMS))
        log_other = np.zeros(count)
        for d, cell_p in zip(others, scan):
            cell = rng.choice(CELLS, count, p=cell_p)
            v = np.clip(edges[cell] + rng.uniform(size=count) / CELLS, 1e-9, 1 - 1e-9)
            z[:, d] = np.log(v) - np.log1p(-v)
            log_other += np.log(cell_p[cell] * CELLS) + np.log(v * (1 - v))  # density in v, then in z
        if informed:
            root = _root(covariance)
            z[:, chosen] = mean[None, :] + (rng.standard_normal((count, len(chosen))) @ root.T) / np.sqrt(
                rng.chisquare(DOF, count) / DOF)[:, None]
        samples = physical(E.from_z(z, np.full(count, data.ratio), composition), data.q_max)
        log_like, amplitudes, chi2 = likelihood(samples)
        z_all.append(z)
        log_like_all.append(log_like)
        amplitudes_all.append(amplitudes)
        chi2_all.append(chi2)
        log_other_all.append(log_other)
        z = np.concatenate(z_all)
        log_q = np.concatenate(log_other_all)
        if informed:
            parts = [_t_log_density(z[:, chosen] - m[None, :], np.linalg.inv(_root(c)),
                                    float(np.sum(np.log(np.diag(_root(c))))), DOF) for m, c in proposals]
            top = np.max(parts, axis=0)
            log_q = log_q + top + np.log(np.mean(np.exp(np.array(parts) - top), axis=0))
        log_w = np.concatenate(log_like_all) + E.log_prior(z, composition) - log_q
        log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
        w = np.exp(log_w - np.max(log_w))
        ess = float(w.sum() ** 2 / np.sum(w * w))
        if ess >= ESS_TARGET or not informed or round_ == ROUNDS - 1:
            break
        w = w / w.sum()
        mean = np.sum(w[:, None] * z[:, chosen], axis=0)
        spread = (z[:, chosen] - mean).T @ ((z[:, chosen] - mean) * w[:, None])
        proposals.append((mean, 0.5 * (spread + spread.T) * INFLATE**2 + 0.5 * first))
    top = np.max(log_w)
    samples = physical(E.from_z(z, np.full(z.shape[0], data.ratio), composition), data.q_max)
    return Local(composition, samples, log_w, np.concatenate(amplitudes_all), np.concatenate(chi2_all),
                 float(top + np.log(np.mean(np.exp(log_w - top)))), ess,
                 physical(E.from_z(z_star[None, :], [data.ratio], composition), data.q_max)[0])


def _root(covariance):
    return np.linalg.cholesky(0.5 * (covariance + covariance.T) + 1e-12 * np.eye(covariance.shape[0]))


__all__ = ["DOF", "ESS_TARGET", "INFLATE", "INFORMED", "Local", "PRIOR_PRECISION", "ROUNDS", "curvature",
           "local_posterior"]
