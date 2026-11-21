"""Symmetry operations written as in a CIF (``-x+1/2, y, -z``) and what they imply.

An operation maps fractional coordinates ``x' = R x + t``. ``parse_operation`` reads the usual notations
(``x-y``, ``1/2+x``, ``-X+0.5``). Two uses:

* ``expand_sites``: the atoms of the unit cell from the asymmetric unit (images closer than 10⁻³ in each
  fractional coordinate, modulo 1, count once);
* ``systematically_absent``: the exact extinction rule. The reflection **h** is absent when an operation leaves it
  unchanged (``h R = h``) while ``h · t`` is not an integer: the contributions of the two images then cancel for
  every atom position. This needs no atoms, so a cell with its symmetry gives the allowed reflections alone.

``centring_operations`` adds the lattice translations of a centring letter (P, A, B, C, I, F; R in the hexagonal
setting, obverse) for a cell entered without a CIF.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

IDENTITY = "x,y,z"
_TERM = re.compile(r"([+-]?)([0-9.]*(?:/[0-9.]+)?)\*?([xyz]?)")
CENTRINGS = {
    "P": ((0.0, 0.0, 0.0),),
    "A": ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5)),
    "B": ((0.0, 0.0, 0.0), (0.5, 0.0, 0.5)),
    "C": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0)),
    "I": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
    "F": ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)),
    "R": ((0.0, 0.0, 0.0), (2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0)),
}
"""Lattice translations of each centring (R: hexagonal axes, obverse setting)."""


@dataclass(frozen=True)
class Operation:
    rotation: np.ndarray
    """3 × 3 integer matrix R (rows: x', y', z')."""
    translation: np.ndarray
    """t, fractional."""
    text: str = IDENTITY

    def apply(self, xyz) -> np.ndarray:
        return np.asarray(xyz, dtype=float) @ self.rotation.T + self.translation


def _number(text: str) -> float:
    if "/" in text:
        top, bottom = text.split("/", 1)
        return float(Fraction(top or "1")) / float(Fraction(bottom))
    return float(text)


def parse_operation(text: str) -> Operation:
    """``Operation`` of a CIF operation such as ``-x+1/2,y,-z+1/2`` (ValueError when it cannot be read)."""
    cleaned = str(text).strip().strip("'").strip('"').replace(" ", "").lower()
    parts = cleaned.split(",")
    if len(parts) != 3:
        raise ValueError(f"A symmetry operation has three parts separated by commas: {text!r}.")
    rotation = np.zeros((3, 3))
    translation = np.zeros(3)
    for row, part in enumerate(parts):
        if not part:
            raise ValueError(f"Empty part in the symmetry operation {text!r}.")
        position = 0
        while position < len(part):
            match = _TERM.match(part, position)
            if match is None or match.end() == position:
                raise ValueError(f"Cannot read {part!r} in the symmetry operation {text!r}.")
            sign, number, axis = match.groups()
            factor = -1.0 if sign == "-" else 1.0
            if axis:
                rotation[row, "xyz".index(axis)] += factor * (_number(number) if number else 1.0)
            elif number:
                translation[row] += factor * _number(number)
            else:
                raise ValueError(f"Cannot read {part!r} in the symmetry operation {text!r}.")
            position = match.end()
    if not np.allclose(rotation, np.round(rotation)) or abs(abs(np.linalg.det(rotation)) - 1.0) > 1e-6:
        raise ValueError(f"{text!r} is not a crystallographic symmetry operation.")
    return Operation(np.round(rotation).astype(int), translation, cleaned)


def parse_operations(texts) -> tuple[Operation, ...]:
    """The operations of a list of strings; the identity alone when the list is empty."""
    operations = tuple(parse_operation(text) for text in texts if str(text).strip())
    return _unique(operations) if operations else (parse_operation(IDENTITY),)


def centring_operations(letter: str, operations=None) -> tuple[Operation, ...]:
    """``operations`` (default: the identity) combined with the lattice translations of ``letter``."""
    letter = (letter or "P").strip().upper()[:1]
    if letter not in CENTRINGS:
        raise ValueError(f"Unknown lattice centring {letter!r} (P, A, B, C, I, F or R).")
    base = tuple(operations) if operations else (parse_operation(IDENTITY),)
    combined = []
    for shift in CENTRINGS[letter]:
        for operation in base:
            combined.append(Operation(operation.rotation, (operation.translation + np.asarray(shift)) % 1.0,
                                      operation.text))
    return _unique(combined)


def _wrapped(delta) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    return delta - np.round(delta)


def _unique(operations) -> tuple[Operation, ...]:
    kept: list[Operation] = []
    for operation in operations:
        duplicate = any(
            np.array_equal(operation.rotation, other.rotation)
            and np.allclose(_wrapped(operation.translation - other.translation), 0.0, atol=1e-6)
            for other in kept
        )
        if not duplicate:
            kept.append(operation)
    return tuple(kept)


def expand_sites(fractional, operations, tolerance: float = 1e-3) -> list[np.ndarray]:
    """For each asymmetric position, its distinct images in the unit cell (fractional, in [0, 1))."""
    images = []
    for position in np.atleast_2d(np.asarray(fractional, dtype=float)):
        found: list[np.ndarray] = []
        for operation in operations:
            image = operation.apply(position) % 1.0
            if not any(np.all(np.abs(_wrapped(image - other)) < tolerance) for other in found):
                found.append(image)
        images.append(np.array(found))
    return images


def systematically_absent(hkl, operations) -> np.ndarray:
    """``True`` for each row ``(h, k, l)`` that the operations extinguish (see the module docstring)."""
    hkl = np.atleast_2d(np.asarray(hkl, dtype=int))
    absent = np.zeros(len(hkl), dtype=bool)
    for operation in operations:
        if not np.any(np.abs(_wrapped(operation.translation)) > 1e-9):
            continue
        fixed = np.all(hkl @ operation.rotation == hkl, axis=1)
        phase = hkl @ operation.translation
        absent |= fixed & (np.abs(phase - np.round(phase)) > 1e-6)
    return absent


__all__ = [
    "CENTRINGS", "IDENTITY", "Operation", "centring_operations", "expand_sites", "parse_operation",
    "parse_operations", "systematically_absent",
]
