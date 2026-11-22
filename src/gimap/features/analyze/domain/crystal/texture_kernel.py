"""Spherical Gaussian pole figures for all reflections in an extraction window.

The spherical addition theorem gives the azimuthal average of P_l(cos γ) as
P_l(cos χ) P_l(cos ψ). This separates the crystal angles from the mosaic width
and allows many candidate crystal orientations to use the same bin integrals.
Only even l survive the antipodal symmetry.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import roots_legendre

from .pole_models import SIGMA_PER_FWHM
from .reflections import _canonical_key, hkl_text


WIDTH_BOUNDS = (2., 120.)
LOG_WIDTHS = np.linspace(*np.log(WIDTH_BOUNDS), 85)
BIN_SAMPLES = 12
# At 2° FWHM, l=576 is over eight Gaussian standard deviations in frequency.
MAX_DEGREE = 576


def _even_legendre(x):
    """P_0, P_2, … for every x, using the three-term recurrence."""
    x = np.asarray(x)
    previous, current = np.ones_like(x), x
    yield previous
    for n in range(2, MAX_DEGREE + 1):
        previous, current = current, ((2*n - 1)*x*current - (n - 1)*previous) / n
        if n % 2 == 0:
            yield current


@lru_cache(maxsize=1)
def _mosaic_moments():
    """Normalised Legendre coefficients of the antipodal spherical Gaussian.

    Integrate about one pole over the entire sphere (the other gives the same
    even moments). Use z=γ/s for narrow widths and discard only z>12 tails.
    For wide Gaussians the high-degree oscillations require more quadrature
    nodes, even though their true coefficients are very small.
    """
    x, weights = roots_legendre(500)
    sigma = np.exp(LOG_WIDTHS)[:, None] * math.pi / 180. * SIGMA_PER_FWHM
    z = (x + 1.) * 0.5 * np.minimum(12., math.pi / sigma)
    theta = z * sigma
    measure = weights * np.sin(theta) * np.exp(-0.5*z*z)
    measure /= measure.sum(axis=1, keepdims=True)
    coefficients = [(2*l + 1) * np.sum(measure * p, axis=1)
                    for l, p in zip(range(0, MAX_DEGREE + 1, 2), _even_legendre(np.cos(theta)))]
    return np.stack(coefficients, axis=1)


@lru_cache(maxsize=4)
def _bin_moments(centres):
    # The same twelve midpoint samples over a degree as pole_models.binned.
    samples = (np.arange(BIN_SAMPLES) + 0.5) / BIN_SAMPLES - 0.5
    chi = np.radians(np.asarray(centres)[:, None] + samples)
    weight = np.radians(1.) * np.sin(chi)
    return np.array([np.mean(p * weight, axis=-1) for p in _even_legendre(np.cos(chi))])


def _rotations(phase):
    matrices = [np.eye(3, dtype=int), *(op.rotation for op in phase.operations)]
    return np.unique(matrices, axis=0)


def _orbit(hkl, rotations):
    orbit = np.asarray(hkl) @ rotations
    return {tuple(int(x) for x in row) for row in np.concatenate([orbit, -orbit])}


def candidate_axes(phase):
    """Primitive reciprocal directions in the index cube, modulo crystal symmetry and antipodes."""
    remaining = {h for h in itertools.product(range(-2, 3), repeat=3) if math.gcd(*h) == 1}
    axes = []
    rotations = _rotations(phase)
    while remaining:
        equivalent = remaining.intersection(_orbit(next(iter(remaining)), rotations))
        axes.append(min(equivalent, key=_canonical_key))
        remaining.difference_update(equivalent)
    return sorted(axes, key=lambda h: (sum(abs(v) for v in h), _canonical_key(h)))


@dataclass(frozen=True)
class Family:
    label: str
    multiplicity: int
    q: float
    strength: float
    directions: np.ndarray
    fractions: np.ndarray


def ring_families(phase, found, low, high):
    """Group individual generated reflections, retaining every pole and its |F|²."""
    rotations = _rotations(phase)
    inside = np.flatnonzero((found.q >= low) & (found.q <= high))
    by_hkl = {tuple(found.hkl[i]): int(i) for i in inside}
    families = []
    while by_hkl:
        orbit = _orbit(next(iter(by_hkl)), rotations)
        members = sorted(orbit.intersection(by_hkl))
        indices = [by_hkl.pop(hkl) for hkl in members]
        representative = min(members, key=_canonical_key)
        f2, q = found.f2[indices], found.q[indices]
        families.append(Family(hkl_text(representative), len(indices), float(q.mean()), float(f2.sum()),
                               found.q_vectors[indices] / q[:, None], f2 / f2.sum()))
    return sorted(families, key=lambda family: family.q)


class PoleLibrary:
    """Bin-averaged probabilities, indexed by axis and family, as a function of log FWHM."""

    def __init__(self, phase, axes, families, centres):
        normals = np.array([phase.cell.plane_normal(hkl) for hkl in axes])
        bins = _bin_moments(tuple(centres))
        self.isotropic = bins[0]
        spectra = []
        for family in families:
            cosines = np.clip(normals @ family.directions.T, -1., 1.)
            spectra.append(np.stack([p @ family.fractions for p in _even_legendre(cosines)], axis=1))
        family_moments = np.stack(spectra, axis=1)
        # Each product below sums over Legendre degrees, one set of probabilities
        # per width, candidate axis, family and polar bin. Interpolate in log w.
        grid = np.stack([(family_moments * moments) @ bins for moments in _mosaic_moments()])
        self.spline = CubicSpline(LOG_WIDTHS, grid, axis=0)
        self.fractions = np.array([f.strength for f in families])
        self.fractions /= self.fractions.sum()
