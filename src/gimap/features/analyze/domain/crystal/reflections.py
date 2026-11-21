"""The allowed reflections of a phase up to q_max, their structure factors, and how they group.

``reflections`` lists every (h, k, l) ≠ 0 with |q| ≤ q_max that the symmetry allows (``systematically_absent``).
With atom sites, the intensity of each is the squared structure factor

    F(h) = Σ_j o_j f0_j(|q|) exp(−B_j s²) exp(2πi h · x_j),    s = |q| / 4π           (sum over the unit cell)

with o the occupancy, f0 the scattering factor (``scattering.py``) and B the isotropic displacement; a reflection
whose |F|² is below 10⁻⁶ of the largest is not listed (absent by the special positions of the atoms). Without
atoms every allowed reflection has unit intensity.

Two groupings:

* ``powder_rings`` (an isotropic sample): one ring per |q|; its intensity is Σ |F|² over the reflections on it
  (the multiplicity m times |F|² when they are equivalent);
* ``fibre_spots`` (crystallites at random about the surface normal n): one spot per (|q_∥|, q_z) with
  q_z = q · n and q_∥ = √(|q|² − q_z²); spots with q_z < 0 are those of the crystallites turned upside down and
  coincide with a spot of q_z > 0, so only q_z ≥ 0 is listed (an in-plane spot counts both h and −h).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .cell import UnitCell
from .cif import AtomSite
from .scattering import coefficients_of, f0
from .symmetry import Operation, expand_sites, systematically_absent

MAX_INDICES = 3_000_000
"""Index triples searched at most (a q_max too large for a big cell is refused rather than run for minutes)."""
WEAK = 1e-6
"""Reflections with |F|² below this fraction of the strongest are not listed."""


@dataclass(frozen=True)
class Phase:
    name: str
    cell: UnitCell
    operations: tuple[Operation, ...]
    atoms: tuple[AtomSite, ...] = ()


@dataclass(frozen=True)
class Reflections:
    hkl: np.ndarray
    """(n, 3) integers."""
    q_vectors: np.ndarray
    """(n, 3) Cartesian q in the cell frame (Å⁻¹)."""
    q: np.ndarray
    f2: np.ndarray
    """|F|² (electrons²) with atoms, 1 without."""
    with_atoms: bool
    approximated: tuple[str, ...] = ()
    """Elements whose f0 was scaled from a neighbour (``scattering.coefficients_of``)."""
    extinct_by_atoms: int = 0
    """Reflections allowed by the symmetry but with |F|² ≈ 0 for these atoms."""


@dataclass(frozen=True)
class Group:
    """A powder ring or a fibre spot: reflections at the same position."""

    q: float
    q_parallel: float
    qz: float
    hkl: tuple[tuple[int, int, int], ...]
    """Every reflection at the position."""
    labels: tuple[str, ...]
    """One representative per family (different |F|² or not related by the symmetry), strongest first."""
    multiplicity: int
    intensity: float
    """Σ |F|² over the reflections at the position."""
    extra: dict = field(default_factory=dict)

    @property
    def d(self) -> float:
        return 2.0 * math.pi / self.q

    @property
    def chi_deg(self) -> float:
        """Polar angle from the surface normal, atan2(q_∥, q_z), in degrees (0° out of plane, 90° in plane)."""
        return math.degrees(math.atan2(self.q_parallel, self.qz))

    @property
    def mean_f2(self) -> float:
        return self.intensity / self.multiplicity if self.multiplicity else 0.0

    @property
    def label(self) -> str:
        return ", ".join(self.labels)


def hkl_text(hkl) -> str:
    """``110`` when every index is a single non-negative digit, else ``1 -1 0``."""
    values = [int(value) for value in hkl]
    if all(0 <= value <= 9 for value in values):
        return "".join(str(value) for value in values)
    return " ".join(str(value) for value in values)


def _index_grid(limits: tuple[int, int, int]) -> np.ndarray:
    h_max, k_max, l_max = limits
    count = (2 * h_max + 1) * (2 * k_max + 1) * (2 * l_max + 1)
    if count > MAX_INDICES:
        raise ValueError(
            f"q_max is too large for this cell ({count} index triples to search); lower q_max."
        )
    grid = np.stack(np.meshgrid(
        np.arange(-h_max, h_max + 1), np.arange(-k_max, k_max + 1), np.arange(-l_max, l_max + 1), indexing="ij",
    ), axis=-1).reshape(-1, 3)
    return grid[np.any(grid != 0, axis=1)]


def structure_factor_squared(phase: Phase, hkl: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """|F|² of each reflection and the elements whose scattering factor was approximated."""
    amplitude = np.zeros(len(hkl), dtype=complex)
    approximated = set()
    s2 = (q / (4.0 * math.pi)) ** 2
    images = expand_sites([atom.xyz for atom in phase.atoms], phase.operations)
    for atom, positions in zip(phase.atoms, images):
        if coefficients_of(atom.element)[2]:
            approximated.add(atom.element)
        factor = atom.occupancy * f0(atom.element, q) * np.exp(-atom.b_iso * s2)
        phases = np.exp(2j * math.pi * (hkl @ positions.T))
        amplitude += factor * phases.sum(axis=1)
    return np.abs(amplitude) ** 2, tuple(sorted(approximated))


def reflections(phase: Phase, q_max: float) -> Reflections:
    """The reflections of ``phase`` with 0 < |q| ≤ ``q_max`` (see the module docstring)."""
    if not q_max > 0:
        raise ValueError("q_max must be positive.")
    grid = _index_grid(phase.cell.index_limits(q_max))
    vectors = phase.cell.q_vectors(grid)
    q = np.linalg.norm(vectors, axis=1)
    keep = q <= q_max * (1.0 + 1e-12)
    grid, vectors, q = grid[keep], vectors[keep], q[keep]
    allowed = ~systematically_absent(grid, phase.operations)
    grid, vectors, q = grid[allowed], vectors[allowed], q[allowed]
    if not phase.atoms or not len(grid):
        return Reflections(grid, vectors, q, np.ones(len(grid)), False)
    f2, approximated = structure_factor_squared(phase, grid, q)
    strong = f2 > WEAK * float(f2.max()) if len(f2) and f2.max() > 0 else np.zeros(len(f2), dtype=bool)
    return Reflections(grid[strong], vectors[strong], q[strong], f2[strong], True, approximated,
                       int(len(f2) - strong.sum()))


def _canonical_key(hkl) -> tuple:
    values = tuple(int(value) for value in hkl)
    return (sum(value < 0 for value in values), tuple(-value for value in values))


def _families(members: np.ndarray, f2: np.ndarray, operations) -> list[tuple[tuple[int, int, int], float, int]]:
    """``(representative, Σ|F|², count)`` of each family among ``members``: reflections with the same |F|² related by
    a rotation of the symmetry or by −1 (Friedel). Without rotations (a cell entered with only its centring) the
    point group is not known: reflections whose indices are the same up to order and sign count as one family."""
    rotations = [operation.rotation for operation in operations] or [np.eye(3, dtype=int)]
    by_permutation = all(np.array_equal(rotation, np.eye(3, dtype=int)) for rotation in rotations)
    remaining = list(range(len(members)))
    families = []
    while remaining:
        first = remaining.pop(0)
        orbit = {tuple(int(v) for v in sign * (members[first] @ rotation))
                 for rotation in rotations for sign in (1, -1)}
        pattern = tuple(sorted(abs(int(v)) for v in members[first]))
        chosen = [first]
        for other in list(remaining):
            same_f2 = abs(f2[other] - f2[first]) <= 1e-6 * max(abs(f2[first]), 1e-30)
            related = (tuple(sorted(abs(int(v)) for v in members[other])) == pattern if by_permutation
                       else tuple(int(v) for v in members[other]) in orbit)
            if same_f2 and related:
                chosen.append(other)
                remaining.remove(other)
        representative = min((tuple(int(v) for v in members[index]) for index in chosen), key=_canonical_key)
        families.append((representative, float(f2[chosen].sum()), len(chosen)))
    families.sort(key=lambda family: (-family[1] / family[2], _canonical_key(family[0])))
    return families


def _clusters(keys: np.ndarray, tolerance: float) -> list[np.ndarray]:
    """Indices grouped by equal rows of ``keys`` (each column within ``tolerance``)."""
    order = np.lexsort(keys.T[::-1])
    groups: list[list[int]] = []
    for index in order:
        if groups and np.all(np.abs(keys[index] - keys[groups[-1][0]]) <= tolerance):
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    return [np.array(group) for group in groups]


def powder_rings(found: Reflections, operations=()) -> list[Group]:
    """One ring per |q| (within 10⁻⁶ Å⁻¹), in order of q."""
    rings = []
    for members in _clusters(np.round(found.q, 6).reshape(-1, 1), 1.5e-6):
        hkl = found.hkl[members]
        families = _families(hkl, found.f2[members], operations)
        rings.append(Group(
            float(found.q[members].mean()), 0.0, 0.0, tuple(tuple(int(v) for v in row) for row in hkl),
            tuple(hkl_text(family[0]) for family in families), len(members), float(found.f2[members].sum()),
        ))
    rings.sort(key=lambda ring: ring.q)
    return rings


def fibre_spots(found: Reflections, normal, operations=()) -> list[Group]:
    """One spot per (|q_∥|, q_z ≥ 0) for crystallites at random about ``normal`` (unit vector, cell frame)."""
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    qz = found.q_vectors @ normal
    q_parallel = np.sqrt(np.maximum(found.q ** 2 - qz ** 2, 0.0))
    up = qz >= -1e-9
    indices = np.flatnonzero(up)
    keys = np.column_stack([np.round(q_parallel[indices], 6), np.round(np.abs(qz[indices]), 6)])
    spots = []
    for cluster in _clusters(keys, 1.5e-6) if len(indices) else []:
        members = indices[cluster]
        hkl = found.hkl[members]
        families = _families(hkl, found.f2[members], operations)
        spots.append(Group(
            float(found.q[members].mean()), float(q_parallel[members].mean()), float(np.abs(qz[members]).mean()),
            tuple(tuple(int(v) for v in row) for row in hkl), tuple(hkl_text(family[0]) for family in families),
            len(members), float(found.f2[members].sum()),
        ))
    spots.sort(key=lambda spot: (spot.q, spot.qz))
    return spots


__all__ = [
    "Group", "MAX_INDICES", "Phase", "Reflections", "fibre_spots", "hkl_text", "powder_rings", "reflections",
    "structure_factor_squared",
]
