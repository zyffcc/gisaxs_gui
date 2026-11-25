"""Predict's evidence with the scales integrated after the model error, against both integrated together.

Predict's likelihood of a cut (``infer.Likelihood``) at shape values θ is that of

    y = (X a) (1 + B c) + ε,   ε ~ N(0, S²),   c ~ N(0, τ² I₈),   a uniform on [0, top]^M,

X the M columns (each population's term, the beam tail, the constant), B the 8 cubic B-splines of the model error.
The B-splines sum to 1, so a constant in B c is the same as a common factor of all the scales a. Predict solves a* by
non-negative least squares, integrates c exactly at a* (Woodbury), and then integrates a by Laplace at a* with the
Hessian XᵀS⁻²X of χ²/2 alone (a half Gaussian for a scale at 0). Predict pools the proposals of every retained
start to estimate the integral of that likelihood over θ once per composition (``laplace.pool_composition``).

Here the same model is integrated over a and c together. For any a the c integral is Gaussian and exact,

    f(a) = log N(y; X a, S² + τ² diag(X a) Bᵀ B diag(X a)),

and the joint likelihood is L(θ) = top^-M ∫_[0,top]^M exp f(a) da, by adaptive importance sampling in
t = a·√diag(XᵀS⁻²X). The first proposal is a mixture of two multivariate t (5 degrees of freedom) at the maximum â of
f on a ≥ 0, with the covariance of −∇²f there inflated ×1.2 (48 %) and ×3 (32 %), and a product of piecewise-constant
densities from f along each coordinate through â (20 %, ``LineScan``: it reaches a scale the model error leaves
almost free, or one at 0 whose density falls exponentially). Four pilot rounds of N/4 draws then each add a t with
the weighted mean and covariance of the pilot draws so far (half of the next draws); the final N draws alone give the
estimate; draws outside the box weigh 0. A second run (other draws, ×1.0 and ×2 at â, the adapted t ×1.5 instead
of ×1.2) checks it (``converged``: within 0.1 and at least 20 effective samples each), and at the refined point a
third run with four times the draws. f at a* with Predict's Laplace terms gives back infer.Likelihood to rounding
(``reproduced``).

The cuts are one-population cases of the first benchmark set (recovery_cases.draw(index, 20261116) and its curve,
both halves, 'model' and 'measured'), the cut's q_z given (Predict's default 'born' cut). For each, Predict
(refine.predict, default settings, its σ with the 5 % model-error floor in the posterior) runs with each of four
compositions: the truth (its family with a paracrystal), the same family without a paracrystal, the other family
with a paracrystal (three columns each), and two populations of the true family with paracrystals (four columns).
For an independent check, the full composition target is sampled again from the most probable solution alone
(``local_posterior``, Predict's sample count and rounds, the model error starting at 0.02), and Δ = log L_joint −
log L_Predict is computed at:
  refined      the solution's values, τ at the posterior median of τ
  tau/2, tau*2 the same with τ halved and doubled (within τ's prior range 0.003–0.15)
  posterior k  POINTS draws of the local posterior (systematic resampling of its weights; POSTERIOR_SAMPLES
               draws per joint integral, SAMPLES at the three points above)
Because Z_joint = Z_Predict · E_post[exp Δ] (the posterior of Predict's likelihood), the posterior draws give the
estimated correction in log evidence, log mean exp Δ, and the change in each log Bayes factor against the truth.
The correction uses these independent single-start draws. predict_log_evidence records Predict's pooled estimate.
Each Δ is split into the gain of moving a from a* to â (f(â) − f(a*)), the volume (½ log det XᵀS⁻²X − ½ log det
−∇²f, over the scales above 0) and the rest (the joint integral against the Laplace of f at â).

usage: python tools/benchmarks/evidence_integration_check.py [--indices 0,1,4,5,8,13] [--points 40]
           [--samples 65536] [--posterior-samples 8192] [--network PATH] [--out evidence_integration_check]
Writes tools/benchmarks/NAME.json (every solution, its three named points in full, the posterior draws as rows) and
every point in full to tools/benchmarks/per_cut/NAME.json.gz; about 25 minutes on one core.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import gammaln, logsumexp  # noqa: E402

HERE = Path(__file__).resolve().parent
GIMAP = HERE.parents[1]
for _p in (str(HERE), str(GIMAP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import recovery_cases as fr  # noqa: E402
from src.gimap.features.fitting.domain.posterior import encoding as E  # noqa: E402
from src.gimap.features.fitting.domain.posterior import infer, refine  # noqa: E402
from src.gimap.features.fitting.domain.posterior.flow import Network  # noqa: E402
from src.gimap.features.fitting.domain.posterior.laplace import local_posterior  # noqa: E402

SEED0 = 20261116
INDICES = (0, 1, 4, 5, 8, 13)
NETWORK = GIMAP / "src/gimap/features/fitting/domain/posterior/data/flow_v1"
FAMILY = {"sphere": "sphere", "vc": "vertical_cylinder"}
OTHER = {"sphere": "vertical_cylinder", "vertical_cylinder": "sphere"}
TAU = E.ERROR_DIM
TAU_RANGE = (0.003, 0.15)
TAU_START = 0.02
"""The model error at which ``local_posterior`` starts (Predict's own when the network's mode has none)."""
POINTS = 40
SAMPLES = 2 ** 16
POSTERIOR_SAMPLES = 2 ** 13
"""Draws of each joint integral at the refined point and its two values of τ, and at each posterior draw (whose
correction's uncertainty comes from the number of posterior draws, not from these)."""
LONG = 4
"""The refined point's third run has this many times ``SAMPLES``."""
DOF = 5
MAIN = ((0.6, 1.2), (0.4, 3.0))
CHECK = ((0.5, 1.0), (0.5, 2.0))
"""(share of the samples, inflation of the covariance) of the proposal's two t components."""
ADAPT_ESS, ADAPT_ROUNDS = 20.0, 4
ADAPT_INFLATE = (1.2, 1.5)
"""The adapted t's inflation in the main run and in the check."""
CONVERGED = 0.1
"""A point's joint integral counts as converged when the main and the check run agree within this (and both have at
least ``ADAPT_ESS`` effective samples)."""
SCAN_SHARE, SCAN_CELLS, SCAN_FLOOR, SCAN_LOG, SCAN_DROP = 0.2, 400, 1e-4, 5.0, 50.0
"""The line scan's share of the draws, its cells per coordinate, the floor of its density, the t̂ above which its
cells are even in ln t, and the fall of f that ends its range (``LineScan``)."""
STEP_GRADIENT, STEP_HESSIAN = 1e-4, 1e-2
"""Finite-difference steps in t = a·√diag(XᵀS⁻²X), where χ²/2 has unit curvature."""
CHUNK = 8192


class Amplitudes:
    """f(a) = log p(y | a, θ, τ) of a cut at fixed shape values θ, the model error's spline coefficients integrated
    out exactly at every a (``infer.Likelihood.__call__`` does this at a* only)."""

    def __init__(self, likelihood: infer.Likelihood, theta):
        self.theta = np.asarray(theta, float)
        data = likelihood.data
        self.x = likelihood.columns(self.theta[None])[0]  # (M, n)
        self.y, self.s = data.intensity, data.sigma
        self.basis = likelihood.basis  # (8, n)
        k = self.basis.shape[0]
        self.pairs = (self.basis[:, None, :] * self.basis[None, :, :]).reshape(k * k, -1).T  # (n, 64)
        self.tau = float(self.theta[TAU])
        self.top = infer.AMPLITUDE_RANGE * float(np.max(np.abs(self.y)))
        self.log_norm = likelihood.log_norm
        scaled = self.x / self.s
        self.gram = scaled @ scaled.T  # XᵀS⁻²X: the Hessian of χ²/2 in a
        self.scale = 1.0 / np.sqrt(np.maximum(np.diag(self.gram), 1e-300))  # a = scale · t

    @property
    def columns(self) -> int:
        return self.x.shape[0]

    def __call__(self, a) -> np.ndarray:
        a = np.atleast_2d(np.asarray(a, float))
        k = self.basis.shape[0]
        out = np.empty(a.shape[0])
        eye = np.eye(k) / self.tau ** 2
        for start in range(0, a.shape[0], CHUNK):
            block = a[start:start + CHUNK]
            model = block @ self.x
            w = model / self.s
            r = (self.y - model) / self.s
            chi2 = np.einsum("bn,bn->b", r, r)
            gram = ((w * w) @ self.pairs).reshape(-1, k, k)
            projected = (w * r) @ self.basis.T
            inner = gram + eye
            solved = np.linalg.solve(inner, projected[..., None])[..., 0]
            _sign, log_det = np.linalg.slogdet(inner)
            log_det = log_det + 2 * k * math.log(self.tau)  # log det(I + τ² G) = 2k ln τ + log det(G + I/τ²)
            explained = np.einsum("bk,bk->b", projected, solved)
            out[start:start + block.shape[0]] = self.log_norm - 0.5 * (chi2 - explained) - 0.5 * log_det
        return out

    def in_t(self, t) -> np.ndarray:
        return self(np.atleast_2d(t) * self.scale)


def half_gaussian(diagonal) -> np.ndarray:
    """log ∫₀^∞ exp(−d a²/2) da."""
    return 0.5 * math.log(0.5 * math.pi) - 0.5 * np.log(np.maximum(diagonal, 1e-300))


def laplace(value: float, hessian, positive, top: float) -> float:
    """value + log of the Laplace volume over the scales above 0 (Hessian ``hessian``), half Gaussians at the others,
    under the uniform prior on [0, top]^M: the formula of ``infer.Likelihood.__call__``."""
    m = len(positive)
    block = np.where(positive[:, None] & positive[None, :], hessian, 0.0) + np.diag((~positive).astype(float))
    sign, log_det = np.linalg.slogdet(block)
    if sign <= 0:
        return float("nan")
    zero = np.where(positive, 0.0, half_gaussian(np.diag(hessian)))
    return float(value - m * math.log(top) + 0.5 * positive.sum() * math.log(2 * math.pi) - 0.5 * log_det
                 + zero.sum())


def maximum(f: Amplitudes, a_star) -> tuple[np.ndarray, int]:
    """â = argmax f on [0, top]^M from a*, in t (bounded L-BFGS with central-difference gradients): (t̂, iterations)."""
    m = f.columns
    steps = np.vstack([np.eye(m), -np.eye(m)]) * STEP_GRADIENT

    def value_and_gradient(t):
        values = f.in_t(np.vstack([t[None, :], t[None, :] + steps]))
        gradient = (values[1:m + 1] - values[m + 1:]) / (2 * STEP_GRADIENT)
        return -values[0], -gradient

    start = np.asarray(a_star, float) / f.scale
    bounds = [(0.0, f.top / sc) for sc in f.scale]
    result = minimize(value_and_gradient, start, jac=True, method="L-BFGS-B", bounds=bounds,
                      options=dict(ftol=1e-15, gtol=1e-9, maxiter=1000))
    t_hat = np.clip(result.x, 0.0, None)
    if -result.fun < f.in_t(start)[0]:  # never worse than the start
        t_hat = start
    return t_hat, int(result.nit)


def hessian_t(f: Amplitudes, t) -> np.ndarray:
    """−∇²f in t at t (central differences; f is smooth across a = 0, only the prior stops there)."""
    m = f.columns
    h = STEP_HESSIAN
    rows, keys = [], []
    for i in range(m):
        for j in range(i, m):
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                point = t.copy()
                point[i] += si * h
                point[j] += sj * h
                rows.append(point)
                keys.append((i, j, si * sj))
    values = f.in_t(np.array(rows))
    out = np.zeros((m, m))
    for (i, j, sign), v in zip(keys, values):
        out[i, j] += sign * v / (4 * h * h)
    out = out + np.triu(out, 1).T
    return -out


class StudentT:
    """A multivariate t (``DOF`` degrees of freedom) in t."""

    def __init__(self, mean, covariance):
        self.mean = np.asarray(mean, float)
        self.root = np.linalg.cholesky(0.5 * (covariance + covariance.T))
        d = self.mean.size
        self.constant = (gammaln(0.5 * (DOF + d)) - gammaln(0.5 * DOF) - 0.5 * d * math.log(DOF * math.pi)
                         - float(np.sum(np.log(np.diag(self.root)))))

    def sample(self, count: int, rng) -> np.ndarray:
        z = rng.standard_normal((count, self.mean.size)) @ self.root.T
        return self.mean[None, :] + z / np.sqrt(rng.chisquare(DOF, count) / DOF)[:, None]

    def log_density(self, t) -> np.ndarray:
        z = np.linalg.solve(self.root, (t - self.mean).T).T
        return self.constant - 0.5 * (DOF + self.mean.size) * np.log1p(np.sum(z * z, axis=1) / DOF)


class LineScan:
    """A product of piecewise-constant densities, one per coordinate of t, each from f along that coordinate
    through t̂ (the others at t̂), at least ``SCAN_FLOOR`` of its largest cell's density: it reaches a scale the
    model error leaves almost free and a scale at 0 whose density falls exponentially, where the curvature at t̂
    says little. A coordinate whose t̂ is above ``SCAN_LOG`` gets cells even in ln t from t̂·e⁻⁸ up to the prior's
    top (and one cell from 0), the others even cells from 0 to where f has fallen by ``SCAN_DROP``."""

    def __init__(self, f: Amplitudes, t_hat):
        self.edges, self.probabilities = [], []
        upper = f.top / f.scale
        for k in range(f.columns):
            if t_hat[k] > SCAN_LOG:
                edges = np.concatenate([[0.0], np.geomspace(t_hat[k] * math.exp(-8.0), upper[k], SCAN_CELLS)])
            else:
                reach = max(1.0, 2.0 * t_hat[k])
                while reach < upper[k]:
                    line = np.repeat(t_hat[None, :], 2, axis=0)
                    line[1, k] = reach
                    values = f.in_t(line)
                    if values[0] - values[1] > SCAN_DROP:
                        break
                    reach *= 2.0
                edges = np.linspace(0.0, min(reach, upper[k]), SCAN_CELLS + 1)
            centres = np.repeat(t_hat[None, :], edges.size - 1, axis=0)
            centres[:, k] = 0.5 * (edges[1:] + edges[:-1])
            log_density = f.in_t(centres)
            density = np.exp(log_density - np.max(log_density))
            density = np.maximum(density, SCAN_FLOOR)
            mass = density * np.diff(edges)
            self.edges.append(edges)
            self.probabilities.append(mass / mass.sum())

    def sample(self, count: int, rng) -> np.ndarray:
        out = np.empty((count, len(self.edges)))
        for k, (edges, p) in enumerate(zip(self.edges, self.probabilities)):
            cell = rng.choice(p.size, count, p=p)
            out[:, k] = edges[cell] + rng.uniform(size=count) * (edges[cell + 1] - edges[cell])
        return out

    def log_density(self, t) -> np.ndarray:
        out = np.zeros(t.shape[0])
        for k, (edges, p) in enumerate(zip(self.edges, self.probabilities)):
            cell = np.searchsorted(edges, t[:, k], side="right") - 1
            inside = (cell >= 0) & (cell < p.size)
            safe = np.clip(cell, 0, p.size - 1)
            out += np.where(inside, np.log(p[safe] / np.diff(edges)[safe]), -np.inf)
        return out


def _weigh(f: Amplitudes, mixture, count: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """``count`` draws t of a mixture ((share, component) each; all draws weighed against the whole mixture) and
    their log weights top^-M exp f(a(t)) |da/dt| / q(t) (−∞ outside the prior's box)."""
    shares = np.array([share for share, _c in mixture])
    t = np.vstack([c.sample(n, rng) for (_s, c), n in zip(mixture, rng.multinomial(count, shares / shares.sum()))])
    with np.errstate(divide="ignore"):
        log_q = logsumexp(np.stack([math.log(share / shares.sum()) + c.log_density(t) for share, c in mixture]),
                          axis=0)
    inside = np.all(t >= 0, axis=1) & np.all(t <= (f.top / f.scale)[None, :], axis=1)
    log_w = np.full(count, -np.inf)
    jacobian = float(np.sum(np.log(f.scale)))
    log_w[inside] = f.in_t(t[inside]) + jacobian - f.columns * math.log(f.top) - log_q[inside]
    return t, log_w


def _estimate(log_w) -> dict:
    top = float(np.max(log_w))
    w = np.exp(log_w - top)
    mean = float(w.mean())
    return dict(log_z=top + math.log(mean), se=float(w.std(ddof=1) / (mean * math.sqrt(w.size))),
                ess=float(w.sum() ** 2 / np.sum(w * w)), inside=float(np.mean(np.isfinite(log_w))), samples=int(w.size))


def _tempered(log_w, ess: float) -> np.ndarray:
    """Normalised weights w^β, β the largest ≤ 1 that leaves at least ``ess`` effective samples."""
    finite = np.isfinite(log_w)
    shifted = np.where(finite, log_w - np.max(log_w[finite]), 0.0)

    def weights(beta):
        w = np.where(finite, np.exp(beta * shifted), 0.0)
        return w / w.sum()

    low, high = 0.0, 1.0
    if 1.0 / np.sum(weights(1.0) ** 2) >= ess:
        return weights(1.0)
    for _ in range(40):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if 1.0 / np.sum(weights(middle) ** 2) >= ess else (low, middle)
    return weights(low)


def integrate(f: Amplitudes, t_hat, covariance, scan: LineScan, components, count: int, rng,
              adapt: bool = False, inflate: float = 1.2) -> dict:
    """log top^-M ∫_[0,top]^M exp f(a) da by importance sampling in t. The proposal: t distributions at t̂ with the
    covariance ``covariance`` inflated (``components``: (share, inflation) each; the rest of the draws from the line
    scan ``scan``, share ``SCAN_SHARE``). ``adapt``: first ``ADAPT_ROUNDS`` pilots of ``count``/4 draws (population
    Monte Carlo); after each, a t with the weighted mean and covariance of all pilot draws so far (their weights
    tempered to at least ``ADAPT_ESS`` effective samples, inflated ×``inflate``) takes half of the next
    draws, and of the final ``count`` draws, which alone give the estimate."""
    base = [(share * (1 - SCAN_SHARE), StudentT(t_hat, covariance * inflation ** 2))
            for share, inflation in components] + [(SCAN_SHARE, scan)]
    mixture, pool_t, pool_w = base, [], []
    for _round in range(ADAPT_ROUNDS if adapt else 0):
        t, log_w = _weigh(f, mixture, count // 4, rng)
        pool_t.append(t)
        pool_w.append(log_w)
        t, log_w = np.vstack(pool_t), np.concatenate(pool_w)
        if not np.any(np.isfinite(log_w)):
            continue
        w = _tempered(log_w, ADAPT_ESS)
        mean = w @ t
        spread = (t - mean).T @ ((t - mean) * w[:, None])
        spread = 0.5 * (spread + spread.T) + 1e-9 * np.trace(spread) * np.eye(f.columns)
        mixture = [(0.5 * share, c) for share, c in base] + [(0.5, StudentT(mean, spread * inflate ** 2))]
    out = _estimate(_weigh(f, mixture, count, rng)[1])
    out["adapted"] = mixture is not base
    return out


def joint(likelihood: infer.Likelihood, theta, rng, samples: int, long: bool = False) -> dict:
    """Predict's log likelihood of θ and the joint integral over the scales and the model error's coefficients."""
    theta = np.asarray(theta, float)
    program, amplitudes, chi2 = likelihood(theta[None])
    program = float(program[0])
    a_star = amplitudes[0]
    f = Amplitudes(likelihood, theta)
    positive = a_star > 0
    f_star = float(f(a_star[None])[0])
    rebuilt = laplace(f_star, f.gram, positive, f.top)
    t_hat, iterations = maximum(f, a_star)
    a_hat = t_hat * f.scale
    f_hat = float(f.in_t(t_hat)[0])
    h_t = hessian_t(f, t_hat)
    eigen = np.linalg.eigvalsh(h_t)
    regularised = bool(eigen.min() <= 1e-9 * max(eigen.max(), 1e-300))
    if regularised:  # f of a composition far from the curve is flat or not concave at â: keep the proposal proper
        h_t = h_t + (abs(eigen.min()) + 1e-6 * max(eigen.max(), 1.0)) * np.eye(f.columns)
    covariance = np.linalg.inv(h_t)
    h_a = h_t / np.outer(f.scale, f.scale)
    on = a_hat > 0
    laplace_joint = laplace(f_hat, h_a, on, f.top)
    scan = LineScan(f, t_hat)
    main = integrate(f, t_hat, covariance, scan, MAIN, samples, rng, adapt=True, inflate=ADAPT_INFLATE[0])
    check = integrate(f, t_hat, covariance, scan, CHECK, samples, rng, adapt=True, inflate=ADAPT_INFLATE[1])
    out = dict(
        tau=f.tau, columns=f.columns, positive=int(positive.sum()), chi2=float(chi2[0]),
        a_star=a_star.tolist(), a_hat=a_hat.tolist(), maximiser_iterations=iterations,
        program=program, reproduced=rebuilt - program,
        joint=main["log_z"], joint_se=main["se"], joint_ess=main["ess"], joint_inside=main["inside"],
        check=check["log_z"], check_se=check["se"], check_ess=check["ess"], adapted=main["adapted"],
        converged=bool(abs(main["log_z"] - check["log_z"]) < CONVERGED
                       and min(main["ess"], check["ess"]) >= ADAPT_ESS),
        laplace_joint=laplace_joint, delta=main["log_z"] - program,
        parts=dict(maximum=f_hat - f_star,
                   volume=float(0.5 * np.linalg.slogdet(f.gram[np.ix_(positive, positive)])[1]
                                - 0.5 * np.linalg.slogdet(h_a[np.ix_(on, on)])[1]) if positive.any() and on.any()
                   else None,
                   laplace_vs_program=laplace_joint - program, rest=main["log_z"] - laplace_joint),
        scale_direction=dict(chi2_curvature=float(a_star @ f.gram @ a_star),
                             spline_constant=float(8.0 / f.tau ** 2)),
        hessian_regularised=regularised)
    if long:
        extra = integrate(f, t_hat, covariance, scan, MAIN, LONG * samples, rng, adapt=True,
                          inflate=ADAPT_INFLATE[0])
        out.update(long=extra["log_z"], long_se=extra["se"], long_ess=extra["ess"])
    return out


def weighted_median(values, weights) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order]) - 0.5 * weights[order]
    return float(np.interp(0.5, cum / weights.sum(), values[order]))


def parameter_points(local, centre, count: int, rng) -> list[tuple[str, np.ndarray]]:
    w = local.weights
    tau = weighted_median(local.samples[:, TAU], w)
    refined = np.array(centre, float)
    refined[TAU] = tau
    out = [("refined", refined)]
    for label, factor in (("tau/2", 0.5), ("tau*2", 2.0)):
        moved = refined.copy()
        moved[TAU] = float(np.clip(tau * factor, *TAU_RANGE))
        out.append((label, moved))
    cum = np.cumsum(w)
    picks = np.searchsorted(cum / cum[-1], (rng.uniform() + np.arange(count)) / count)
    out += [(f"posterior {k + 1}", local.samples[min(i, len(w) - 1)].copy()) for k, i in enumerate(picks)]
    return out


def compositions_of(case) -> list[tuple[str, int]]:
    family = FAMILY[case["populations"][0]["shape"]]
    return [("truth", E.composition_index([(family, True)])),
            ("no paracrystal", E.composition_index([(family, False)])),
            ("other family", E.composition_index([(OTHER[family], True)])),
            ("two populations", E.composition_index([(family, True), (family, True)]))]


def run_cut(network, index: int, points: int, samples: int, posterior_samples: int) -> dict:
    case = fr.draw(index, SEED0, set_name="legacy-first")
    intensity, sigma = fr.curve(case)
    qz = float(fr.GEOM.qz_vacuum)
    counts = infer.Cut.prepare(fr.Q, intensity, sigma, qz)
    data = replace(counts, sigma=np.hypot(counts.sigma, refine.MODEL_ERROR_FLOOR * np.abs(counts.intensity)))
    record = dict(index=index, composition=case["composition"], level=case["level"],
                  truth=dict(R=case["populations"][0]["R"], sR=case["populations"][0]["sR"],
                             D=case["populations"][0]["D"], s_nn=case["populations"][0]["s_nn"],
                             peak_counts=case["peak_counts"], background=case["background"]),
                  points_measured=int(data.q.size), compositions=[])
    for role, composition in compositions_of(case):
        tick = time.perf_counter()
        entry = dict(role=role, index=composition, name=E.composition_name(composition))
        solutions = refine.predict(network, fr.Q, intensity, sigma, qz=qz, compositions=[composition], seed=index)
        best = solutions[0]
        diagnostics = best.extras["diagnostics"]
        entry.update(predict_log_evidence=diagnostics.get("composition_log_evidence"),
                     predict_ess=diagnostics.get("composition_ess"),
                     chi2_reduced=float(best.chi2_reduced), flags=list(best.flags),
                     compared_by=best.extras.get("compared_by"), solutions=len(solutions))
        centre = refine.model_values(best.model, composition)
        if centre is None:
            entry["skipped"] = "the refined radii left the composition's order"
            record["compositions"].append(entry)
            continue
        rng = np.random.default_rng(1000 * index + composition)
        local = local_posterior(data, composition, centre, TAU_START, rng, refine.LOCAL_SAMPLES,
                                rounds=refine.LOCAL_ROUNDS)
        entry.update(local_log_evidence=float(local.log_evidence), local_ess=float(local.ess),
                     local_samples=int(local.samples.shape[0]),
                     tau_posterior=dict(zip(("median", "low68", "high68"),
                                            map(float, local.quantiles([0.5, 0.16, 0.84])[:, TAU]))))
        likelihood = infer.Likelihood(data, composition)
        entry["points"] = []
        for label, theta in parameter_points(local, centre, points, rng):
            row = joint(likelihood, theta, rng, posterior_samples if label.startswith("posterior") else samples,
                        long=label == "refined")
            row["label"] = label
            entry["points"].append(row)
        posterior = np.array([p["delta"] for p in entry["points"] if p["label"].startswith("posterior")])
        correction = float(logsumexp(posterior) - math.log(posterior.size))
        ratio = np.exp(posterior - posterior.max())
        log_tau = np.log([p["tau"] for p in entry["points"]])
        slope, intercept = np.polyfit(log_tau, [p["delta"] for p in entry["points"]], 1)
        entry.update(delta_posterior=dict(mean=float(posterior.mean()), sd=float(posterior.std(ddof=1)),
                                          min=float(posterior.min()), max=float(posterior.max())),
                     correction=correction,
                     correction_se=float(ratio.std(ddof=1) / (ratio.mean() * math.sqrt(ratio.size))),
                     corrected_log_evidence=entry["local_log_evidence"] + correction,
                     delta_vs_log_tau=dict(slope=float(slope), intercept=float(intercept)),
                     converged_points=int(sum(p["converged"] for p in entry["points"])),
                     distinct_posterior_points=int(len({round(p["program"], 9) for p in entry["points"][3:]})),
                     seconds=time.perf_counter() - tick)
        record["compositions"].append(entry)
        print(f"  cut {index} {role:15s} {entry['name']:28s} logZ {entry['local_log_evidence']:.2f} "
              f"Δ refined {entry['points'][0]['delta']:+.3f}  posterior {posterior.mean():+.3f}±{posterior.std():.3f} "
              f"correction {correction:+.3f}  converged {entry['converged_points']}/{len(entry['points'])} "
              f"({entry['seconds']:.0f} s)", flush=True)
    truth = next(c for c in record["compositions"] if c["role"] == "truth")
    for c in record["compositions"]:
        if "correction" not in c or c is truth:
            continue
        program = c["local_log_evidence"] - truth["local_log_evidence"]
        corrected = c["corrected_log_evidence"] - truth["corrected_log_evidence"]
        c["against_truth"] = dict(log_bayes_program=program,
                                  log_bayes_predict=c["predict_log_evidence"] - truth["predict_log_evidence"],
                                  log_bayes_joint=corrected, shift=corrected - program,
                                  ranking_changed=bool(np.sign(program) != np.sign(corrected)))
    return record


def summary(cuts) -> dict:
    entries = [c for cut in cuts for c in cut["compositions"] if "points" in c]
    points = [p for c in entries for p in c["points"]]
    good = [c for c in entries if c["converged_points"] == len(c["points"])]
    by_role = {}
    for c in entries:
        by_role.setdefault(c["role"], []).append(c)
    out = dict(
        reproduced_max_abs=float(max(abs(p["reproduced"]) for p in points)),
        solutions=len(entries), solutions_converged=len(good),
        not_converged=[dict(cut=cut["index"], role=c["role"], chi2_reduced=c["chi2_reduced"],
                            converged_points=c["converged_points"], points=len(c["points"]))
                       for cut in cuts for c in cut["compositions"]
                       if "points" in c and c["converged_points"] < len(c["points"])],
        convergence_of_converged=dict(
            max_abs_main_minus_check=float(max(abs(p["joint"] - p["check"]) for c in good for p in c["points"])),
            max_se=float(max(max(p["joint_se"], p["check_se"]) for c in good for p in c["points"])),
            min_ess=float(min(min(p["joint_ess"], p["check_ess"]) for c in good for p in c["points"])),
            max_abs_main_minus_long=float(max(abs(p["joint"] - p["long"]) for c in good for p in c["points"]
                                              if "long" in p))) if good else None,
        delta_converged=dict(min=float(min(p["delta"] for c in good for p in c["points"])),
                             max=float(max(p["delta"] for c in good for p in c["points"]))) if good else None,
        tau_dependence=[dict(cut=cut["index"], role=c["role"],
                             **{p["label"]: dict(tau=p["tau"], delta=p["delta"]) for p in c["points"][:3]})
                        for cut in cuts for c in cut["compositions"] if "points" in c],
        by_role={role: dict(columns=items[0]["points"][0]["columns"], solutions=len(items),
                            converged=int(sum(any(c is g for g in good) for c in items)),
                            chi2_reduced=dict(min=float(min(c["chi2_reduced"] for c in items)),
                                              max=float(max(c["chi2_reduced"] for c in items))),
                            tau_median=dict(min=float(min(c["tau_posterior"]["median"] for c in items)),
                                            max=float(max(c["tau_posterior"]["median"] for c in items))),
                            within_solution_sd=dict(median=float(np.median([c["delta_posterior"]["sd"] for c in items])),
                                                    max=float(max(c["delta_posterior"]["sd"] for c in items))),
                            correction=dict(min=float(min(c["correction"] for c in items)),
                                            max=float(max(c["correction"] for c in items)),
                                            median=float(np.median([c["correction"] for c in items]))),
                            delta_refined=dict(min=float(min(c["points"][0]["delta"] for c in items)),
                                               max=float(max(c["points"][0]["delta"] for c in items))))
                 for role, items in by_role.items()},
        bayes_shift=[dict(cut=cut["index"], level=cut["level"], against=c["role"], **c["against_truth"])
                     for cut in cuts for c in cut["compositions"] if "against_truth" in c])
    shifts = [b["shift"] for b in out["bayes_shift"]]
    out["bayes_shift_abs_max"] = float(max(abs(s) for s in shifts)) if shifts else None
    out["rankings_changed"] = int(sum(b["ranking_changed"] for b in out["bayes_shift"]))
    return out


POSTERIOR_FIELDS = ("tau", "program", "joint", "joint_se", "check", "delta", "converged", "positive")
PARTS = ("maximum", "volume", "rest")


def _rounded(value, digits: int = 8):
    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    if isinstance(value, dict):
        return {k: _rounded(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v, digits) for v in value]
    return value


def _slim(cuts) -> list:
    """The cuts with each solution's posterior draws as rows of ``POSTERIOR_FIELDS`` and ``PARTS``."""
    out = []
    for cut in cuts:
        entries = []
        for c in cut["compositions"]:
            entry = dict(c)
            if "points" in c:
                draws = [q for q in c["points"] if q["label"].startswith("posterior")]
                entry["points"] = [q for q in c["points"] if not q["label"].startswith("posterior")]
                entry["posterior_points"] = dict(
                    fields=POSTERIOR_FIELDS + tuple(f"parts.{k}" for k in PARTS),
                    rows=[[q[f] for f in POSTERIOR_FIELDS] + [q["parts"][k] for k in PARTS] for q in draws])
            entries.append(entry)
        out.append(dict(cut, compositions=entries))
    return out


def write(result: dict, out: str) -> None:
    """NAME.json (``_slim``, 8 significant digits) and per_cut/NAME.json.gz (everything)."""
    (HERE / "per_cut").mkdir(exist_ok=True)
    (HERE / "per_cut" / f"{out}.json.gz").write_bytes(
        gzip.compress(json.dumps(result, indent=1).encode("utf-8"), mtime=0))
    slim = dict(result, cuts=_slim(result["cuts"]), every_point=f"per_cut/{out}.json.gz")
    (HERE / f"{out}.json").write_text(json.dumps(_rounded(slim), indent=1), encoding="utf-8")


def _code_state() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=GIMAP, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    folder = "src/gimap/features/fitting/domain"
    return {"commit": git("rev-parse", "--short", "HEAD"), "fitting_tree": git("rev-parse", f"HEAD:{folder}"),
            "fitting_code_modified": bool(git("status", "--porcelain", "--", folder))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--indices", default=",".join(map(str, INDICES)))
    ap.add_argument("--points", type=int, default=POINTS)
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--posterior-samples", type=int, default=POSTERIOR_SAMPLES)
    ap.add_argument("--network", default=str(NETWORK))
    ap.add_argument("--out", default="evidence_integration_check")
    a = ap.parse_args()
    tick = time.perf_counter()
    network = Network.load(a.network)
    cuts = []
    for index in (int(i) for i in a.indices.split(",")):
        case = fr.draw(index, SEED0, set_name="legacy-first")
        if len(case["populations"]) != 1:
            raise SystemExit(f"cut {index} has {len(case['populations'])} populations; one-population cuts only")
        print(f"cut {index}: {case['composition']} {case['level']}", flush=True)
        cuts.append(run_cut(network, index, a.points, a.samples, a.posterior_samples))
    result = dict(
        description=__doc__.split("\n\n")[0],
        settings=dict(seed0=SEED0, indices=[c["index"] for c in cuts], points=a.points, samples=a.samples,
                      posterior_samples=a.posterior_samples, scan=dict(share=SCAN_SHARE, cells=SCAN_CELLS,
                                                                       floor=SCAN_FLOOR, log_above=SCAN_LOG,
                                                                       drop=SCAN_DROP),
                      long_samples=LONG * a.samples, dof=DOF, main=MAIN, check=CHECK, adapt_inflate=ADAPT_INFLATE,
                      adapt_rounds=ADAPT_ROUNDS, adapt_ess=ADAPT_ESS, tau_start=TAU_START,
                      model_error_floor=refine.MODEL_ERROR_FLOOR, amplitude_range=infer.AMPLITUDE_RANGE,
                      local_samples=refine.LOCAL_SAMPLES, local_rounds=refine.LOCAL_ROUNDS, cut="born"),
        network=Path(a.network).name, network_sha256=network.meta.get("sha256"), code=_code_state(),
        summary=summary(cuts), cuts=cuts, seconds=time.perf_counter() - tick)
    write(result, a.out)
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
