"""Weighted simultaneous least squares for one or two candidate fibre axes.

At fixed mosaic widths, isotropic fraction and mixing fraction the ring
amplitudes are linear. For free family weights they are the nonnegative family
amplitudes; their sum is the scale of that ring. Profiling them analytically
leaves only two or four shared parameters for nonlinear optimisation.
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
    """Grid both widths independently. Ring-specific amplitudes only initialise the joint fit."""
    widths = np.arange(len(_START_INDEX))
    if len(indices) == 2:
        first, second = np.meshgrid(widths, widths, indexing="ij")
        select = first < second if indices[0] == indices[1] else np.ones(first.shape, dtype=bool)
        grid = [first[select], second[select]]
    else:
        grid = [widths]
    costs = np.zeros(len(grid[0]))
    fractions = np.zeros((len(costs), 2))
    for ring in rings:
        columns = np.stack([np.broadcast_to(ring.iso, (len(costs), len(ring.y))),
                            *(ring.starts[g, i] for g, i in zip(grid, indices))], axis=1)
        gram = columns @ columns.transpose(0, 2, 1)
        rhs = columns @ ring.y
        # Scale the small Gram matrices before the pseudoinverse. Narrow poles
        # can lie almost entirely in a gap. Collinear columns are allowed here.
        norms = np.maximum(np.sqrt(gram.diagonal(axis1=1, axis2=2)), 1e-100)
        normal = gram / (norms[:, :, None] * norms[:, None, :])
        amplitudes = (np.linalg.pinv(normal, rcond=1e-10) @ (rhs / norms)[..., None])[..., 0] / norms
        amplitudes = np.maximum(amplitudes, 0.)
        residual = (amplitudes[:, None, :] @ columns)[:, 0] - ring.y
        costs += np.sum(residual**2, axis=1)
        fractions[:, 0] += ring.norm2 * amplitudes[:, 0] / np.maximum(amplitudes.sum(axis=1), 1e-100)
        fractions[:, 1] += ring.norm2 * amplitudes[:, 1] / np.maximum(amplitudes[:, 1:].sum(axis=1), 1e-100)
    i = np.argmin(costs)
    phi, share = np.clip(fractions[i] / max(sum(r.norm2 for r in rings), 1e-100), 0.04, 0.96)
    log_widths = [LOG_WIDTHS[_START_INDEX[g[i]]] for g in grid]
    return [log_widths[0], phi] if len(indices) == 1 else [log_widths[0], phi, log_widths[1], share]


def _spline(curve, log_width):
    knot = min(max(int((log_width - LOG_WIDTHS[0]) / _LOG_STEP), 0), len(LOG_WIDTHS) - 2)
    d = log_width - LOG_WIDTHS[knot]
    a, b, c, e = curve[:, knot]
    return ((a*d + b)*d + c)*d + e, (3*a*d + 2*b)*d + c


class Hypothesis:
    def __init__(self, rings, indices, free):
        self.rings, self.indices, self.free = rings, indices, free
        self.curves = [tuple((r.by_family if free else r.fixed)[:, :, i] for i in indices) for r in rings]
        self.parameters = 2*len(indices) + sum(len(r.families) if free else 1 for r in rings)
        self.points = sum(len(r.y) for r in rings)

    def _model(self, x, index):
        curves = self.curves[index]
        first, d_first = _spline(curves[0], x[0])
        iso, phi = self.rings[index].iso, x[1]
        if len(curves) == 1:
            return (1. - phi)*first + phi*iso, [(1. - phi)*d_first, iso - first]
        second, d_second = _spline(curves[1], x[2])
        share = x[3]
        oriented = share*first + (1. - share)*second
        return ((1. - phi)*oriented + phi*iso,
                [(1. - phi)*share*d_first, iso - oriented, (1. - phi)*(1. - share)*d_second,
                 (1. - phi)*(first - second)])

    def shapes(self, x, index):
        return self._model(x, index)[0]

    def fit(self, starts):
        """Optimise with the variable-projection Jacobian on the active NNLS face."""
        last_x = None
        jacobian = None

        def evaluate(x):
            nonlocal last_x, jacobian
            blocks, jac = [], []
            for i, ring in enumerate(self.rings):
                shapes, gradients = self._model(x, i)
                coeff = _coefficients(shapes, ring.y)
                residual = coeff @ shapes - ring.y
                derivatives = np.asarray(gradients)
                change = (coeff @ derivatives).T
                # Differentiate A.T @ (A c - y) = 0 over the locally nonzero coefficients.
                active = coeff > 0.
                if active.any():
                    rows = shapes[active]
                    inverse = (rows / max(np.dot(rows[0], rows[0]), 1e-300) if len(rows) == 1
                               else np.linalg.pinv(rows.T))
                    change -= rows.T @ (inverse @ change) + inverse.T @ (derivatives[:, active] @ residual).T
                jac.append(change)
                blocks.append(residual)
            last_x = x.copy()
            jacobian = np.concatenate(jac)
            return np.concatenate(blocks)

        def derivative(x):
            if not np.array_equal(last_x, x):
                evaluate(x)
            return jacobian

        lower = [LOG_WIDTHS[0], 0.] * len(self.indices)
        upper = [LOG_WIDTHS[-1], 1.] * len(self.indices)
        best = None
        for start in starts:
            solution = least_squares(evaluate, start, jac=derivative, bounds=(lower, upper), ftol=2e-7, max_nfev=120)
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


def _prediction(rings, trial):
    problem = Hypothesis(rings, trial.indices, trial.free)
    prediction = []
    for i, ring in enumerate(rings):
        shapes = problem.shapes(trial.x, i)
        prediction.append(_coefficients(shapes, ring.y) @ shapes)
    return prediction


def _observable(rings, mixed, alternatives):
    """Resolve each simpler prediction in at least one measured ring.

    For independent bin errors, the standardised separation of two entire profiles
    along their difference is ||(N1 - N2)/sigma||_2. Coherent small differences
    across several bins can resolve an axis even if each is below one bin error.
    """
    predictions = _prediction(rings, mixed)
    for trial in alternatives:
        rival = _prediction(rings, trial)
        if all(np.linalg.norm(p - r) <= 1. for p, r in zip(predictions, rival)):
            return False
    return True


def multiple_poles(rings):
    """A pole and its antipode measure just one crystal direction, at any diffraction order."""
    directions = np.concatenate([family.directions for ring in rings for family in ring.families])
    return np.any(abs(directions @ directions[0]) < 1. - 1e-9)


def search(rings, n_axes, check_cancel):
    points = sum(len(r.y) for r in rings)
    null_chi2 = 0.
    for ring in rings:
        fit = _coefficients(ring.iso[None], ring.y) * ring.iso
        null_chi2 += np.sum((fit - ring.y)**2)
    null = Trial((), False, np.array([]), null_chi2, null_chi2 + len(rings)*math.log(points))
    trials = {(): null}
    free_count = sum(len(r.families)-1 for r in rings)
    axes = range(n_axes)
    pairs = itertools.combinations(axes, 2) if multiple_poles(rings) else ()
    for indices in itertools.chain(((a,) for a in axes), ((a, a) for a in axes), pairs):
        if check_cancel is not None:
            check_cancel()
        seed = _seed(rings, indices)
        size = len(indices)
        fixed = Hypothesis(rings, indices, False).fit([seed])
        if fixed is None:
            continue
        chosen = fixed
        if free_count and points > len(rings) + free_count + 2*size:
            starts = [fixed.x]
            alternate = [math.log(15.), 0.4] if size == 1 else [math.log(8.), 0.4, math.log(36.), 0.5]
            if max(abs(fixed.x[::2] - alternate[::2])) > 0.7:
                starts.append(alternate)
            free = Hypothesis(rings, indices, True).fit(starts)
            if free is not None and free.bic < fixed.bic:
                chosen = free
        if size == 2:
            rivals = [trials.get((i,), null) for i in set(indices)]
            if any(rival.bic <= chosen.bic for rival in rivals):
                continue
            if indices[0] != indices[1]:
                rivals.extend(trials[i, i] for i in indices if (i, i) in trials)
                if not _observable(rings, chosen, [t for t in rivals if t.indices]):
                    continue
        trials[indices] = chosen
    # Resolve roundoff ties deterministically while reporting the actual BIC difference.
    ranked = sorted(trials.values(), key=lambda trial: trial.bic)
    result = []
    while ranked:
        tied = [t for t in ranked if t.bic <= ranked[0].bic + BIC_TIE]
        result.extend(sorted(tied, key=lambda trial: trial.indices))
        ranked = ranked[len(tied):]
    return result
