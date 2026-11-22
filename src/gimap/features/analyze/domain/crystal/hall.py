"""Space-group operations from a Hall symbol (Hall 1981, *International Tables* B, 1.4 appendix).

A Hall symbol names the generators of a space group explicitly, origin included, so the group follows without
tables: ``-P 2ybc`` is P2₁/c (inversion at the origin, a twofold axis along b with the translation (0, ½, ½)).
Read here:

* the lattice: ``P A B C I R F``; a leading ``-`` adds the inversion at the origin;
* up to four matrix symbols ``[-]N[screw][axis][translations]``: N = 1, 2, 3, 4, 6; the screw digit k gives
  k/N of a lattice period along the axis; axes ``x y z``, ``'`` and ``"`` (the face diagonals a−b and a+b of the
  axis before), ``*`` (a+b+c); translations ``a b c n u v w d`` (½ a, ½ b, ½ c, ½(a+b+c), ¼ a, ¼ b, ¼ c,
  ¼(a+b+c)). Default axes: the first symbol z; the second, a twofold, x after 2 or 4 and a−b after 3 or 6; a
  threefold in third place a+b+c;
* an origin shift ``(vx vy vz)`` in twelfths of the periods: every operation (R, t) becomes (R, t + v − R v).

The group is the closure of the generators and the lattice translations, translations modulo whole periods, in
twelfths (exact). ``operation_text`` writes an operation as a CIF does (``-y+1/2,x,z+1/4``)."""

from __future__ import annotations

import re
from fractions import Fraction

import numpy as np

from .symmetry import Operation

TWELFTHS = 12
LATTICE_TRANSLATIONS = {
    "P": (), "A": ((0, 6, 6),), "B": ((6, 0, 6),), "C": ((6, 6, 0),), "I": ((6, 6, 6),),
    "R": ((8, 4, 4), (4, 8, 8)), "F": ((0, 6, 6), (6, 0, 6), (6, 6, 0)),
}
"""Centring translations (twelfths); R in the hexagonal setting, obverse."""
TRANSLATIONS = {"a": (6, 0, 0), "b": (0, 6, 0), "c": (0, 0, 6), "n": (6, 6, 6),
                "u": (3, 0, 0), "v": (0, 3, 0), "w": (0, 0, 3), "d": (3, 3, 3)}
_ROTATIONS = {
    ("z", 2): ((-1, 0, 0), (0, -1, 0), (0, 0, 1)), ("z", 3): ((0, -1, 0), (1, -1, 0), (0, 0, 1)),
    ("z", 4): ((0, -1, 0), (1, 0, 0), (0, 0, 1)), ("z", 6): ((1, -1, 0), (1, 0, 0), (0, 0, 1)),
    ("x", 2): ((1, 0, 0), (0, -1, 0), (0, 0, -1)), ("x", 3): ((1, 0, 0), (0, 0, -1), (0, 1, -1)),
    ("x", 4): ((1, 0, 0), (0, 0, -1), (0, 1, 0)), ("x", 6): ((1, 0, 0), (0, 1, -1), (0, 1, 0)),
    ("y", 2): ((-1, 0, 0), (0, 1, 0), (0, 0, -1)), ("y", 3): ((-1, 0, 1), (0, 1, 0), (-1, 0, 0)),
    ("y", 4): ((0, 0, 1), (0, 1, 0), (-1, 0, 0)), ("y", 6): ((0, 0, 1), (0, 1, 0), (-1, 0, 1)),
    ("'z", 2): ((0, -1, 0), (-1, 0, 0), (0, 0, -1)), ('"z', 2): ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    ("'x", 2): ((-1, 0, 0), (0, 0, -1), (0, -1, 0)), ('"x', 2): ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ("'y", 2): ((0, 0, -1), (0, -1, 0), (-1, 0, 0)), ('"y', 2): ((0, 0, 1), (0, -1, 0), (1, 0, 0)),
    ("*", 3): ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
}
"""Rotation matrices of Hall's table (rows x', y', z'); ``'z``: the axis ' after an axis along z."""
_AXIS_VECTOR = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
_MATRIX = re.compile(r"^(-?)([12346])([1-5]?)([xyz'\"*]?)([abcnuvwd]*)$")
_SHIFT = re.compile(r"\(([^)]*)\)")
MAX_ORDER = 192
"""The largest number of general positions of a space group (Fm-3m and its kin, conventional cell)."""


def _key(operation) -> tuple:
    rotation, translation = operation
    return tuple(int(value) for value in rotation.ravel()) + tuple(int(value) for value in translation % TWELFTHS)


def _closure(generators: list) -> list:
    """Every product of the generators, translations modulo one period (twelfths), identity first."""
    identity = (np.eye(3, dtype=int), np.zeros(3, dtype=int))
    found = {_key(identity): identity}
    queue = [identity]
    while queue:
        rotation, translation = queue.pop()
        for g_rotation, g_translation in generators:
            product = (g_rotation @ rotation, (g_rotation @ translation + g_translation) % TWELFTHS)
            name = _key(product)
            if name not in found:
                found[name] = product
                queue.append(product)
                if len(found) > MAX_ORDER:
                    raise ValueError("The Hall symbol does not give a crystallographic space group.")
    first = _key(identity)
    return [found[first]] + [found[name] for name in sorted(found) if name != first]


def _axis(position: int, order: int, given: str, previous_order: int) -> str:
    if given:
        return given
    if position == 0 or order == 1:
        return "z"
    if position == 1 and order == 2:
        return "x" if previous_order in (2, 4) else "'"
    if position == 2 and order == 3:
        return "*"
    return ""


def hall_operations(symbol: str) -> tuple[Operation, ...]:
    """The operations of the space group of the Hall ``symbol`` (ValueError when it cannot be read)."""
    text = str(symbol).strip()
    if text[:1] in ("'", '"') and text[-1:] == text[:1]:
        text = text[1:-1].strip()
    shift = np.zeros(3, dtype=int)
    match = _SHIFT.search(text)
    if match:
        parts = match.group(1).replace(",", " ").split()
        if len(parts) != 3 or not all(re.fullmatch(r"-?\d+", part) for part in parts):
            raise ValueError(f"Cannot read the origin shift of the Hall symbol {symbol!r}.")
        shift = np.array([int(part) for part in parts], dtype=int)
        text = text[:match.start()] + text[match.end():]
    tokens = text.split()
    if not tokens:
        raise ValueError("Empty Hall symbol.")
    letter = tokens[0].lstrip("-").upper()
    if letter not in LATTICE_TRANSLATIONS or len(tokens[0].lstrip("-")) != 1:
        raise ValueError(f"Unknown lattice {tokens[0]!r} in the Hall symbol {symbol!r}.")
    generators = [(np.eye(3, dtype=int), np.array(vector, dtype=int)) for vector in LATTICE_TRANSLATIONS[letter]]
    if tokens[0].startswith("-"):
        generators.append((-np.eye(3, dtype=int), np.zeros(3, dtype=int)))
    previous_order, previous_axis = 0, "z"
    for position, token in enumerate(tokens[1:]):
        part = _MATRIX.match(token)
        if part is None:
            raise ValueError(f"Cannot read {token!r} in the Hall symbol {symbol!r}.")
        improper, order_text, screw, given, letters = part.groups()
        order = int(order_text)
        axis = _axis(position, order, given, previous_order)
        lookup = (axis + previous_axis, order) if axis in ("'", '"') else (axis, order)
        if order == 1:
            rotation = np.eye(3, dtype=int)
        elif lookup in _ROTATIONS:
            rotation = np.array(_ROTATIONS[lookup], dtype=int)
        else:
            raise ValueError(f"No axis for {token!r} in the Hall symbol {symbol!r}.")
        translation = np.zeros(3, dtype=int)
        if screw:
            if axis not in _AXIS_VECTOR:
                raise ValueError(f"A screw along {axis!r} is not read here ({symbol!r}).")
            translation += np.array(_AXIS_VECTOR[axis]) * (int(screw) * TWELFTHS // order)
        for name in letters:
            translation += np.array(TRANSLATIONS[name])
        generators.append((-rotation if improper else rotation, translation % TWELFTHS))
        previous_order = order
        if axis in _AXIS_VECTOR:
            previous_axis = axis
    operations = []
    for rotation, translation in _closure(generators):
        moved = (translation + shift - rotation @ shift) % TWELFTHS / TWELFTHS
        operations.append(Operation(rotation, moved, operation_text(rotation, moved)))
    return tuple(operations)


def operation_text(rotation, translation) -> str:
    """``-y+1/2,x,z+1/4``: an operation as a CIF writes it."""
    rows = []
    for row, shift in zip(np.asarray(rotation, dtype=int), np.asarray(translation, dtype=float)):
        terms = ""
        for coefficient, name in zip(row, "xyz"):
            if coefficient:
                sign = "-" if coefficient < 0 else ("+" if terms else "")
                terms += sign + ("" if abs(coefficient) == 1 else str(abs(coefficient))) + name
        fraction = Fraction(float(shift) % 1.0).limit_denominator(2 * TWELFTHS)
        if fraction:
            terms += ("+" if terms else "") + f"{fraction.numerator}/{fraction.denominator}"
        rows.append(terms or "0")
    return ",".join(rows)


__all__ = ["LATTICE_TRANSLATIONS", "MAX_ORDER", "TRANSLATIONS", "hall_operations", "operation_text"]
