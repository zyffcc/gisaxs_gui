"""The symmetry operations of a space group named by its Hermann–Mauguin symbol, its number or its Hall symbol.

A CIF that lists no operations (``_space_group_symop_operation_xyz``) but names its space group gets them here,
from ``space_group_table.py`` through ``hall.py``. The symbol is read as CIFs and papers write it: with or
without spaces (``I 4/m c m``, ``I4/mcm``), subscripts as ``2_1`` or ``2(1)``, the short monoclinic symbol
(``P 21/c`` for ``P 1 21/c 1``), the old cubic ``Fm3m`` for ``Fm-3m``, the old ``Cmca`` for ``Cmce``, an
orthorhombic P group in any of its six axis settings (``Pbnm``), and qualifiers ``:1``/``:2``, ``S``/``Z``
(origin choice 1 or 2) and ``:H``/``:R`` (hexagonal or rhombohedral axes).

Where the symbol leaves a choice open, it is made and said:

* origin choice 1 or 2 (24 groups): the one that puts the atoms of the file on fewer sites (atoms are found on
  special positions of the right origin; on the wrong one they spread over general positions), choice 2 when
  that is a tie or there are no atoms;
* hexagonal or rhombohedral axes of an R group: from the cell (γ = 120°, or a = b = c with α = β = γ).

The cell must have the shape of the crystal system and setting (a monoclinic group with unique axis b: α = γ =
90°; a tetragonal one: a = b, all 90°; …); otherwise the operations would be wrong for it and ``SpaceGroupError``
says why. So does a symbol that cannot be read or is not in the table."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np

from .hall import hall_operations, operation_text
from .space_group_table import OLD_NAMES, SPACE_GROUPS
from .symmetry import Operation, expand_sites

UNREADABLE = "The space-group symbol “{symbol}” cannot be read."
UNKNOWN = "The space group “{symbol}” is not in GIMaP's table (a setting it does not know)."
CELL_MISFIT = "The cell does not have the shape the space group {symbol} needs."
DISAGREE = "The space-group symbol {symbol} and number {number} disagree."
NO_NUMBER = "There is no space group number {number}."
ORIGIN_TAKEN = ("The file does not say which origin choice of {symbol} its coordinates use: choice {choice} was "
                "taken (it puts the atoms on fewer sites).")
LENGTH_TOLERANCE = 2e-3
"""Relative difference of cell lengths that the symmetry makes equal (a = b of a tetragonal cell)."""
ANGLE_TOLERANCE = 0.2
"""Degrees: angles that the symmetry fixes (90°, 120°, α = β = γ)."""


class SpaceGroupError(ValueError):
    """The symbol, the number or the cell do not give the operations; ``template`` and ``values`` say why."""

    def __init__(self, template: str, **values):
        super().__init__(template.format(**values))
        self.template = template
        self.values = values


@dataclass(frozen=True)
class SpaceGroup:
    number: int
    symbol: str
    """As in the table, with a qualifier where it has one (``F d -3 m :2``) or the setting (``P b n m``)."""
    hall: str
    operations: tuple[Operation, ...]
    note: str = ""
    """A choice the symbol left open and how it was made (``ORIGIN_TAKEN``), empty otherwise."""

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(operation.text for operation in self.operations)


def system_of(number: int) -> str:
    for last, name in ((2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"), (142, "tetragonal"),
                       (167, "trigonal"), (194, "hexagonal"), (230, "cubic")):
        if number <= last:
            return name
    raise SpaceGroupError(NO_NUMBER, number=number)


def compact(symbol: str) -> str:
    """The symbol without spaces, subscripts or qualifier, lattice letter upper case (``P21/c``)."""
    text = str(symbol).strip().strip("'\"")
    text = text.replace("−", "-").replace("–", "-").replace("̅", "").replace("_", "")
    text = re.sub(r"(\d)\((\d)\)", r"\1\2", text)
    text = re.sub(r"\s+", "", text)
    return text[:1].upper() + text[1:].lower() if text else ""


def _split_qualifier(symbol: str) -> tuple[str, str]:
    """``(symbol, qualifier)``: ``1``/``2`` (origin choice), ``H``/``R`` (axes) or ``""``."""
    text = str(symbol).strip().strip("'\"")
    found = re.search(r"\(?\s*origin\s*(?:choice)?\s*([12])\s*\)?\s*$", text, re.IGNORECASE)
    if found:
        return text[:found.start()].strip(), found.group(1)
    found = re.search(r"\s*:\s*([12HRhr])\s*$", text) or re.search(r"\s+([SZHRshr])\s*$", text)
    if found:
        letter = found.group(1).upper()
        return text[:found.start()].strip(), {"S": "1", "Z": "2"}.get(letter, letter)
    return text, ""


def _symbol_of_orthorhombic_p(operations: Sequence[Operation]) -> str:
    """The Hermann–Mauguin symbol of an orthorhombic P group in the axes of its operations."""
    parts = []
    for axis in range(3):
        direction = np.zeros(3, dtype=int)
        direction[axis] = 1
        twofold = np.diag([1 if index == axis else -1 for index in range(3)])
        plane = axis_symbol = None
        for operation in operations:
            if np.array_equal(operation.rotation, -twofold):
                glide = (operation.translation + operation.rotation @ operation.translation) / 2.0 % 1.0
                halves = [index for index in range(3) if index != axis and abs(glide[index] - 0.5) < 1e-6]
                quarters = [index for index in range(3) if index != axis and abs(glide[index] % 0.5 - 0.25) < 1e-6]
                plane = "d" if quarters else "n" if len(halves) == 2 else "abc"[halves[0]] if halves else "m"
            elif np.array_equal(operation.rotation, twofold):
                axis_symbol = "21" if abs(operation.translation[axis] % 1.0 - 0.5) < 1e-6 else "2"
        parts.append(plane or axis_symbol or "?")
    return "P " + " ".join(parts)


def _transformed(operations: Sequence[Operation], basis: np.ndarray) -> tuple[Operation, ...]:
    """The operations in the axes ``basis`` (columns: the new axes in the old ones)."""
    inverse = np.round(np.linalg.inv(basis)).astype(int)
    moved = []
    for operation in operations:
        rotation = inverse @ operation.rotation @ basis
        translation = (inverse @ operation.translation) % 1.0
        moved.append(Operation(rotation, translation, operation_text(rotation, translation)))
    return tuple(moved)


@lru_cache(maxsize=1)
def _index() -> dict[str, list[tuple[int, str, str, Optional[np.ndarray]]]]:
    """Compact symbol → ``[(number, symbol, hall, basis or None)]`` (several: origin choices, H and R axes)."""
    index: dict[str, list] = {}

    def add(key, entry):
        bucket = index.setdefault(compact(key), [])
        if not any(item[1] == entry[1] for item in bucket):
            bucket.append(entry)

    for number, symbol, hall in SPACE_GROUPS:
        base, qualifier = _split_qualifier(symbol)
        add(base, (number, symbol, hall, None))
        parts = base.split()
        if 3 <= number <= 15 and parts[1] == "1" and parts[3] == "1":
            add(f"{parts[0]} {parts[2]}", (number, symbol, hall, None))  # P 21/c for P 1 21/c 1
    for old, new in OLD_NAMES.items():
        base, _ = _split_qualifier(old)
        for entry in index.get(compact(_split_qualifier(new)[0]), []):
            add(base, entry)
    for number, symbol, hall in SPACE_GROUPS:  # the other axis settings of the orthorhombic P groups
        if not 16 <= number <= 74 or not symbol.startswith("P"):
            continue
        standard = hall_operations(hall)
        for order in itertools.permutations(range(3)):
            basis = np.zeros((3, 3), dtype=int)
            for column, row in enumerate(order):
                basis[row, column] = 1
            if round(np.linalg.det(basis)) < 0:
                basis[:, 2] *= -1
            if order == (0, 1, 2):
                continue
            setting = _symbol_of_orthorhombic_p(_transformed(standard, basis))
            qualifier = _split_qualifier(symbol)[1]
            add(setting, (number, setting + (f" :{qualifier}" if qualifier else ""), hall, basis))
    return index


def _operations(entry) -> tuple[Operation, ...]:
    number, _symbol, hall, basis = entry
    operations = hall_operations(hall)
    return operations if basis is None else _transformed(operations, basis)


def cell_fits(number: int, symbol: str, cell: Sequence[float]) -> bool:
    """Whether the cell ``(a, b, c, α, β, γ)`` has the shape of the crystal system and setting of ``symbol``."""
    a, b, c, alpha, beta, gamma = (float(value) for value in cell)
    same = lambda x, y: abs(x - y) <= LENGTH_TOLERANCE * max(abs(x), abs(y))  # noqa: E731
    angle = lambda x, y: abs(x - y) <= ANGLE_TOLERANCE  # noqa: E731
    system = system_of(number)
    if system == "triclinic":
        return True
    if system == "monoclinic":
        return angle(alpha, 90) and angle(gamma, 90)
    if system == "orthorhombic":
        return angle(alpha, 90) and angle(beta, 90) and angle(gamma, 90)
    if system == "tetragonal":
        return same(a, b) and angle(alpha, 90) and angle(beta, 90) and angle(gamma, 90)
    if symbol.endswith(":R"):
        return same(a, b) and same(b, c) and angle(alpha, beta) and angle(beta, gamma)
    if system in ("trigonal", "hexagonal"):
        return same(a, b) and angle(alpha, 90) and angle(beta, 90) and angle(gamma, 120)
    return same(a, b) and same(b, c) and angle(alpha, 90) and angle(beta, 90) and angle(gamma, 90)


def _site_count(operations, sites) -> int:
    return sum(len(images) for images in expand_sites(sites, operations)) if len(sites) else 0


def _choose(entries, qualifier: str, cell, sites) -> tuple[tuple, str]:
    """The entry of the qualifier, else the one the cell or the atoms point to (and the note saying so)."""
    if len(entries) == 1:
        return entries[0], ""
    if qualifier:
        for entry in entries:
            if entry[1].endswith(f":{qualifier}"):
                return entry, ""
    if all(entry[1].endswith((":H", ":R")) for entry in entries):
        axes = [entry for entry in entries if cell is not None and cell_fits(entry[0], entry[1], cell)]
        return (axes or entries)[0], ""
    counts = [(_site_count(_operations(entry), sites), 0 if entry[1].endswith(":2") else 1, entry) for entry in entries]
    counts.sort(key=lambda item: (item[0], item[1]))
    chosen = counts[0][2]
    base = _split_qualifier(chosen[1])[0]
    return chosen, ORIGIN_TAKEN.format(symbol=base, choice=chosen[1][-1])


def space_group(symbol: str = "", *, number: Optional[int] = None, hall: str = "", cell=None,
                sites: Sequence = ()) -> SpaceGroup:
    """The operations of the space group named (see the module docstring); ``SpaceGroupError`` when they cannot
    be found for this cell. ``sites``: fractional positions of the atoms (for the origin choice)."""
    sites = np.asarray(sites, dtype=float).reshape(-1, 3) if len(sites) else np.zeros((0, 3))
    if hall.strip():
        operations = hall_operations(hall)
        return SpaceGroup(int(number or 0), symbol.strip() or hall.strip(), hall.strip(), operations)
    entries, qualifier = [], ""
    if symbol.strip():
        base, qualifier = _split_qualifier(symbol)
        key = compact(base)
        if not re.fullmatch(r"[PABCIFR][0-9a-z/\-]+", key):
            raise SpaceGroupError(UNREADABLE, symbol=symbol.strip())
        entries = _index().get(key) or _index().get(re.sub(r"^([PIF][mnad])3", r"\1-3", key)) or []
        if not entries:
            raise SpaceGroupError(UNKNOWN, symbol=symbol.strip())
        if number and all(entry[0] != number for entry in entries):
            raise SpaceGroupError(DISAGREE, symbol=symbol.strip(), number=number)
    elif number:
        system_of(int(number))
        entries = [(n, s, h, None) for n, s, h in SPACE_GROUPS if n == int(number)]
        entries = [entry for entry in entries if entry[1] == entries[0][1] or _split_qualifier(entry[1])[1]]
    else:
        raise SpaceGroupError(UNREADABLE, symbol="")
    entry, note = _choose(entries, qualifier, cell, sites)
    if cell is not None and not cell_fits(entry[0], entry[1], cell):
        raise SpaceGroupError(CELL_MISFIT, symbol=entry[1])
    return SpaceGroup(entry[0], entry[1], entry[2], _operations(entry), note)


__all__ = [
    "ANGLE_TOLERANCE", "CELL_MISFIT", "DISAGREE", "LENGTH_TOLERANCE", "NO_NUMBER", "ORIGIN_TAKEN", "SpaceGroup",
    "SpaceGroupError", "UNKNOWN", "UNREADABLE", "cell_fits", "compact", "space_group", "system_of",
]
