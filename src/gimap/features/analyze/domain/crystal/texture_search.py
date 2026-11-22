"""Weighted simultaneous least squares for one or two candidate fibre axes.

At a fixed mosaic width, isotropic fraction and axis mixing fraction the ring
amplitudes are linear. For free family weights they are the nonnegative family
amplitudes; their sum is the scale of that ring. Profiling them analytically
leaves only two or three shared parameters for nonlinear optimisation.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np
from scipy.optimize import least_squares, nnls

from .texture_kernel import LOG_WIDTHS

_START_INDEX = np.arange(0, len(LOG_WIDTHS), 7)
_LOG_STEP = LOG_WIDTHS[1] - LOG_WIDTHS[0]
BIC_TIE = 1e-4


class RingData:
    def __init__(self, ring, families, library, mask):
        self.ring, self.families, self.library, self.mask = ring, families, library, mask
        error = ring.profile.sigma[mask]
        self.y = ring.profile.value[mask] / error
        self.iso = library.isotropic[mask] / error
        # Spline coefficients: order, knot, axis, family, measured bin.
        self.by_family = np.ascontiguousarray(library.spline.c[..., mask] / error)
        self.fixed = (library.fractions @ self.by_family)[..., None, :]
        # CubicSpline's last coefficient is the value at each left knot.
        last = library.spline(LOG_WIDTHS[-1])[..., mask] / error
        fixed_grid = np.concatenate((self.fixed[-1, :, :, 0, :], (library.fractions @ last)[None]))
        self.starts = fixed_grid[_START_INDEX]
        self.norm2 = float(np.dot(self.y, self.y))


def _coefficients(shapes, y):
    """Positive weighted least-squares scale(s) for a ring at given shared parameters."""
    if len(shapes) == 1:
        v = shapes[0]
        return np.array([max(0., np.dot(v, y) / max(np.dot(v, v), 1e-300))])
    if len(shapes) == 2:
        gram = shapes @ shapes.T
        rhs = shapes @ y
        a, b, c = gram[0, 0], gram[0, 1], gram[1, 1]
        det = a*c - b*b
        if det > 1e-13 * a*c:
            s = np.array([c*rhs[0] - b*rhs[1], a*rhs[1] - b*rhs[0]]) / det
            if np.all(s >= 0.):
                return s
        diagonal = np.maximum(gram.diagonal(), 1e-300)
        column = np.argmax(np.maximum(0., rhs)**2 / diagonal)
        s = np.zeros(2)
        s[column] = max(0., rhs[column] / diagonal[column])
        return s
    from scipy.optimize import lsq_linear
    try:
        return nnls(shapes.T, y)[0]
    except (RuntimeError, np.linalg.LinAlgError):
        return lsq_linear(shapes.T, y, bounds=(0., np.inf)).x


def _seed(rings, indices):
    """Resolve competing width minima on a grid with independent ring fractions for this start."""
    costs = np.zeros(len(_START_INDEX))
    params = np.zeros((len(_START_INDEX), 2))
    for ring in rings:
        columns = np.stack([np.broadcast_to(ring.iso, ring.starts[:, 0].shape),
                            *(ring.starts[:, i] for i in indices)], axis=2)
        # A cheap unconstrained solve provides seeds, only. Clipped estimates
        # are used to score the start, then the shared constrained fit follows.
        amplitudes = np.maximum((np.linalg.pinv(columns) @ ring.y), 0.)
        residuals = (columns @ amplitudes[..., None]).squeeze(-1) - ring.y
        costs += np.sum(residuals**2, axis=1)
        params[:, 0] += ring.norm2 * amplitudes[:, 0] / np.maximum(amplitudes.sum(axis=1), 1e-100)
        params[:, 1] += ring.norm2 * amplitudes[:, 1] / np.maximum(amplitudes[:, 1:].sum(axis=1), 1e-100)
    i = np.argmin(costs)
    fractions = np.clip(params[i] / max(sum(r.norm2 for r in rings), 1e-100), 0.04, 0.96)
    return [LOG_WIDTHS[_START_INDEX[i]], *fractions[:len(indices)]]


class Hypothesis:
    def __init__(self, rings, indices, free):
        self.rings, self.indices, self.free = rings, indices, free
        self.curves = [tuple((r.by_family if free else r.fixed)[:, :, i] for i in indices) for r in rings]
        shared = len(indices) + 1 if indices else 0
        self.parameters = shared + sum(len(r.families) if free else 1 for r in rings)
        self.points = sum(len(r.y) for r in rings)

    def shapes(self, x, index):
        w = x[0]
        knot = min(max(int((w - LOG_WIDTHS[0]) / _LOG_STEP), 0), len(LOG_WIDTHS) - 2)
        d = w - LOG_WIDTHS[knot]

        def evaluate(curve):
            a, b, c, e = curve[:, knot]
            return ((a*d + b)*d + c)*d + e

        curves = self.curves[index]
        oriented = evaluate(curves[0])
        if len(curves) == 2:
            oriented = x[2] * oriented + (1. - x[2]) * evaluate(curves[1])
        return (1. - x[1])*oriented + x[1]*self.rings[index].iso

    def residual(self, x):
        blocks = []
        for i, ring in enumerate(self.rings):
            shapes = self.shapes(x, i)
            coeff = _coefficients(shapes, ring.y)
            blocks.append(coeff @ shapes - ring.y)
        return np.concatenate(blocks)

    def fit(self, starts):
        lower = [LOG_WIDTHS[0]] + [0.] * len(self.indices)
        upper = [LOG_WIDTHS[-1]] + [1.] * len(self.indices)
        best = None
        for start in starts:
            solution = least_squares(self.residual, start, bounds=(lower, upper), ftol=2e-7, max_nfev=120)
            if solution.success and (best is None or solution.cost < best.cost):
                best = solution
        if best is None:
            return None
        return Trial(tuple(self.indices), self.free, best.x, 2. * best.cost,
                     2. * best.cost + self.parameters * math.log(self.points))


@dataclass(frozen=True)
class Trial:
    indices: tuple
    free: bool
    x: np.ndarray
    chi2: float
    bic: float


def search(rings, n_axes, check_cancel):
    points = sum(len(r.y) for r in rings)
    null_chi2 = 0.
    for ring in rings:
        fit = _coefficients(ring.iso[None], ring.y) * ring.iso
        null_chi2 += np.sum((fit - ring.y)**2)
    null = Trial((), False, np.array([]), null_chi2, null_chi2 + len(rings)*math.log(points))
    trials = {(): null}
    free_count = sum(len(r.families)-1 for r in rings)
    for size in (1, 2):
        for indices in itertools.combinations(range(n_axes), size):
            if check_cancel is not None:
                check_cancel()
            seed = _seed(rings, indices)
            fixed = Hypothesis(rings, indices, False).fit([seed])
            if fixed is None:
                continue
            chosen = fixed
            if free_count and points > len(rings) + free_count + size + 1:
                problem = Hypothesis(rings, indices, True)
                starts = [fixed.x]
                if abs(fixed.x[0] - math.log(15.)) > 0.7:
                    starts.append([math.log(15.), 0.4, *([0.5] if size == 2 else [])])
                free = problem.fit(starts)
                if free is not None and free.bic < fixed.bic:
                    chosen = free
            # A pair that loses to a constituent cannot support a second axis.
            if size == 2 and any(trials.get((i,), null).bic <= chosen.bic for i in indices):
                continue
            trials[indices] = chosen
    # Distinct directions can give exactly the same polar angles on the
    # observed rings. Suppress optimiser roundoff when naming such competitors.
    ranked = sorted(trials.values(), key=lambda trial: trial.bic)
    result = []
    while ranked:
        tied = [t for t in ranked if t.bic <= ranked[0].bic + BIC_TIE]
        result.extend(sorted(tied, key=lambda trial: trial.indices))
        ranked = ranked[len(tied):]
    return result
