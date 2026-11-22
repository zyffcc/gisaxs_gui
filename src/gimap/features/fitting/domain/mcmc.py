"""A short Markov chain Monte Carlo run on a posterior, and how far it can be trusted (Physics and Methods 9.19).

The sampler is an affine-invariant ensemble of W walkers in d dimensions (emcee's scheme, Foreman-Mackey et al. 2013):
each half of the ensemble moves at once against the other half, every step by one of two moves:

* with probability ``DE_SHARE`` the differential-evolution move of DE-MC (ter Braak 2006; the proposal of DREAM,
  Vrugt et al. 2009), y = x_k + γ (x_a − x_b), x_a ≠ x_b two walkers of the other half, γ = 2.38/√(2d) (1 + 10⁻⁵ N(0,1)),
  accepted with probability min(1, p(y)/p(x_k));
* otherwise the stretch move of Goodman & Weare (2010), y = x_j + Z (x_k − x_j), Z ~ g(z) ∝ 1/√z on [1/a, a]
  (a = ``STRETCH``), accepted with probability min(1, Z^(d−1) p(y)/p(x_k)).

Both leave p invariant and are unaffected by linear correlations and scales between the coordinates.

Diagnostics of a run (the first ``BURN`` of the steps are discarded):

* the integrated autocorrelation time τ of each coordinate, in steps: the walkers' normalised autocorrelation functions
  averaged, summed up to Sokal's automatic window M ≥ 5 τ(M) (as emcee); the run holds about W·n/τ independent
  samples; emcee advises n ≥ 50 τ;
* split R̂ of each coordinate on the independent ensembles' mean traces, splitting each trace in time;
* the acceptance fraction (0.2–0.5 is usual for the stretch move).

``converged`` asks for both: n ≥ ``TAU_FACTOR`` τ for every coordinate and R̂ < ``RHAT_LIMIT``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

STRETCH = 2.0
DE_SHARE = 0.8
BURN = 0.5
TAU_FACTOR = 50.0
RHAT_LIMIT = 1.05
WINDOW_C = 5.0


@dataclass
class Chain:
    positions: np.ndarray
    """(steps, ensembles, W, d), or (steps, W, d) for a single ensemble."""
    log_prob: np.ndarray
    """One log probability per saved walker position."""
    accepted: np.ndarray
    """Moves accepted per walker."""
    stopped: bool = False

    @property
    def steps(self) -> int:
        return int(self.positions.shape[0])

    def kept(self, burn: float = BURN) -> np.ndarray:
        """Positions after discarding the first ``burn`` fraction of steps."""
        return self.positions[int(math.floor(burn * self.steps)):]

    @property
    def acceptance(self) -> float:
        return float(np.mean(self.accepted) / max(1, self.steps))


def walkers_for(dimensions: int, at_least: int = 16) -> int:
    """An even number of walkers, at least ``at_least`` and four per dimension (and more than 2d, which the stretch
    move needs)."""
    count = max(at_least, 4 * dimensions, 2 * dimensions + 2)
    return count + (count % 2)


def sample(log_prob: Callable[[np.ndarray], np.ndarray], start: np.ndarray, steps: int, rng: np.random.Generator, *,
           progress: Optional[Callable[[int, int], None]] = None, stop: Optional[Callable[[], bool]] = None,
           stretch: float = STRETCH) -> Chain:
    """Sample from one or more independent ensembles, started at (W, d) or (ensembles, W, d).

    All ensembles complete each step before progress is reported, so a stopped run has equally long traces.
    """
    x = np.array(start, float)
    ensembles = x if x.ndim == 3 else x[None, ...]
    streams = ([np.random.default_rng(int(seed)) for seed in rng.integers(2**63 - 1, size=len(ensembles))]
               if x.ndim == 3 else [rng])
    runs = [_ensemble(log_prob, group, stream, stretch) for group, stream in zip(ensembles, streams)]
    positions = np.empty((steps, *x.shape))
    log_probs = np.empty((steps, *x.shape[:-1]))
    accepted = np.zeros(ensembles.shape[:-1])
    done, stopped = 0, False
    for step in range(steps):
        if stop is not None and stop():
            stopped = True
            break
        for index, run in enumerate(runs):
            points, lp, accepted[index] = next(run)
            dest = (step, index) if x.ndim == 3 else step
            positions[dest], log_probs[dest] = points, lp
        done = step + 1
        if progress is not None:
            progress(done, steps)
    return Chain(positions[:done], log_probs[:done], accepted if x.ndim == 3 else accepted[0], stopped)


def _ensemble(log_prob, x, rng, stretch):
    """One independent trajectory. Partners for every proposal belong to this ensemble."""
    walkers, dims = x.shape
    if walkers % 2 or walkers <= dims or walkers < 4:
        raise ValueError("The ensemble needs an even number of walkers, more than the dimensions")
    lp = np.asarray(log_prob(x), float)
    if not np.all(np.isfinite(lp)):
        raise ValueError("Every walker must start where the posterior is above zero")
    accepted = np.zeros(walkers)
    halves = (np.arange(0, walkers // 2), np.arange(walkers // 2, walkers))
    while True:
        evolve = rng.uniform() < DE_SHARE
        for moving, other in (halves, halves[::-1]):
            if evolve:
                pick = np.array([rng.choice(other, size=2, replace=False) for _ in moving])
                gamma = 2.38 / math.sqrt(2.0 * dims) * (1.0 + 1e-5 * rng.standard_normal(moving.size))
                proposal = x[moving] + gamma[:, None] * (x[pick[:, 0]] - x[pick[:, 1]])
                factor = 0.0
            else:
                partners = x[rng.choice(other, size=moving.size)]
                z = ((stretch - 1.0) * rng.uniform(size=moving.size) + 1.0) ** 2 / stretch
                proposal = partners + z[:, None] * (x[moving] - partners)
                factor = (dims - 1) * np.log(z)
            new = np.asarray(log_prob(proposal), float)
            log_ratio = factor + new - lp[moving]
            take = np.isfinite(new) & (np.log(rng.uniform(size=moving.size)) < log_ratio)
            x[moving[take]] = proposal[take]
            lp[moving[take]] = new[take]
            accepted[moving[take]] += 1
        yield x, lp, accepted


def _autocorrelation(series: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation function of each column of ``series`` (n, W) (FFT), averaged over the columns."""
    n = series.shape[0]
    centred = series - series.mean(axis=0)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centred, n=size, axis=0)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), n=size, axis=0)[:n]
    variance = acf[0]
    good = variance > 0
    if not np.any(good):
        return np.zeros(n)
    return np.mean(acf[:, good] / variance[good], axis=1)


def autocorrelation_time(series: np.ndarray, c: float = WINDOW_C) -> float:
    """Integrated autocorrelation time (steps) of one coordinate, ``series`` (n, W) (module docstring)."""
    series = np.asarray(series, float)
    if series.shape[0] < 4:
        return math.nan
    rho = _autocorrelation(series)
    taus = 2.0 * np.cumsum(rho) - 1.0
    window = np.arange(taus.size) < c * taus
    m = int(np.argmin(window)) if not np.all(window) else taus.size - 1
    return float(max(taus[m], 1.0))


def split_rhat(series: np.ndarray) -> float:
    """Split R̂ of (steps, independent ensemble traces), splitting each column in time."""
    series = np.asarray(series, float)
    half = series.shape[0] // 2
    if half < 2:
        return math.nan
    chains = np.concatenate([series[:half], series[-half:]], axis=1)
    within = float(np.mean(np.var(chains, axis=0, ddof=1)))
    between = half * float(np.var(np.mean(chains, axis=0), ddof=1))
    if within <= 0:
        return math.inf if between > 0 else math.nan
    pooled = (half - 1) / half * within + between / half
    return float(math.sqrt(pooled / within))


@dataclass(frozen=True)
class Diagnostics:
    tau: tuple
    """Autocorrelation time (steps) per coordinate."""
    rhat: tuple
    kept_steps: int
    walkers: int
    acceptance: float
    ensembles: int = 1

    @property
    def effective_samples(self) -> float:
        finite = [t for t in self.tau if math.isfinite(t)]
        return self.ensembles * self.walkers * self.kept_steps / max(finite) if finite else math.nan

    @property
    def converged(self) -> bool:
        if not self.tau or any(not math.isfinite(t) for t in self.tau) or any(not math.isfinite(r) for r in self.rhat):
            return False
        return self.ensembles >= 2 and self.kept_steps >= TAU_FACTOR * max(self.tau) and max(self.rhat) < RHAT_LIMIT


def diagnose(chain: Chain, burn: float = BURN, coordinates: Optional[np.ndarray] = None) -> Diagnostics:
    """τ from individual walkers' autocorrelations and R̂ from each ensemble's mean over its walkers at each step."""
    kept = (chain.positions if coordinates is None else np.asarray(coordinates, float))[int(math.floor(burn * chain.steps)):]
    if kept.ndim == 3:
        kept = kept[:, None, :, :]
    steps, ensembles, walkers, dims = kept.shape
    traces = kept.mean(axis=2)
    taus = tuple(autocorrelation_time(kept[..., k].reshape(steps, ensembles * walkers)) for k in range(dims))
    rhats = tuple(split_rhat(traces[..., k]) for k in range(dims))
    return Diagnostics(taus, rhats, steps, walkers, chain.acceptance, ensembles)


def quantiles(samples: np.ndarray, levels=(0.5, 0.16, 0.84, 0.025, 0.975)) -> np.ndarray:
    """(len(levels), k) quantiles of samples (n, k), NaN-free columns only (others NaN)."""
    samples = np.asarray(samples, float)
    out = np.full((len(levels), samples.shape[1]), np.nan)
    for k in range(samples.shape[1]):
        column = samples[:, k][np.isfinite(samples[:, k])]
        if column.size:
            out[:, k] = np.quantile(column, levels)
    return out


__all__ = ["BURN", "Chain", "DE_SHARE", "Diagnostics", "RHAT_LIMIT", "STRETCH", "TAU_FACTOR", "autocorrelation_time", "diagnose",
           "quantiles", "sample", "split_rhat", "walkers_for"]
