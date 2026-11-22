"""A small reader of crystallographic information files (CIF 1.1): the cell, the symmetry and the atom sites.

Read from the first data block that gives a cell:

* ``_cell_length_a/b/c`` (Å) and ``_cell_angle_alpha/beta/gamma`` (°); standard uncertainties in brackets,
  ``8.874(2)``, are dropped;
* the symmetry operations of ``_space_group_symop_operation_xyz`` or ``_symmetry_equiv_pos_as_xyz`` (also with
  a dot, ``_space_group_symop.operation_xyz``), the Hermann–Mauguin and Hall symbols and the number of the space
  group;
* the atom sites: ``_atom_site_label``, ``_atom_site_type_symbol``, ``_atom_site_fract_x/y/z``,
  ``_atom_site_occupancy`` (default 1), ``_atom_site_U_iso_or_equiv`` or ``_atom_site_B_iso_or_equiv``
  (B = 8π² U; default 0);
* the stoichiometric formula ``_chemical_formula_sum`` and formula units per cell ``_cell_formula_units_Z``
  for the film optical constants, separately from the sites used to calculate reflection intensities.

Without operations they are generated from the Hall symbol, the Hermann–Mauguin symbol or the number
(``space_groups.py``; ``symmetry`` is then ``symbol`` and ``notes`` says so). When that is not possible (a symbol
that cannot be read or is not in the table, a cell of another shape, no space group named), the phase gives
positions only (``symmetry`` ``none``): every hkl of the cell, without intensities or extinctions, and
``warnings`` says why. The name is ``_chemical_name_common``, ``_chemical_name_mineral`` or the data block's name;
the sum formula (``formula``) only when there is nothing else. Quoted values, semicolon text fields and comments are
understood; everything else in the file (bonds, anisotropic displacements, refinement details) is ignored.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .cell import UnitCell
from .scattering import element_of
from .space_groups import SpaceGroupError, space_group

SYMOP_TAGS = ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz")
SPACE_GROUP_TAGS = ("_space_group_name_h-m_alt", "_symmetry_space_group_name_h-m", "_space_group_name_h-m")
HALL_TAGS = ("_space_group_name_hall", "_symmetry_space_group_name_hall")
NUMBER_TAGS = ("_space_group_it_number", "_symmetry_int_tables_number")
NAME_TAGS = ("_chemical_name_common", "_chemical_name_mineral")
FORMULA_TAGS = ("_chemical_formula_sum", "_chemical_formula_structural", "_chemical_formula_moiety")
FROM_SYMBOL = "{count} symmetry operations generated from the space group {symbol} (No. {number}): the file lists none."
"""A note of a phase whose operations come from its space group."""
NO_SPACE_GROUP = "The file names no space group."
POSITIONS_ONLY = ("No symmetry operations: positions only. Every hkl of the cell is listed, without intensities or "
                  "extinctions, and rings assigned to this phase are tentative.")
"""The warnings of a phase without operations (after the reason: ``space_groups.SpaceGroupError``)."""
GENERIC_BLOCKS = ("global", "i", "1", "cif", "data", "phase", "structure", "unnamed", "unknown")
"""Data block names that say nothing about the phase (``data_global``, ``data_I``)."""
_ESD = re.compile(r"\(\d+\)$")


@dataclass(frozen=True)
class AtomSite:
    label: str
    element: str
    xyz: tuple[float, float, float]
    occupancy: float = 1.0
    b_iso: float = 0.0
    """Isotropic displacement B (Å²); exp(−B s²) multiplies the scattering factor."""


@dataclass(frozen=True)
class CifStructure:
    name: str
    cell: UnitCell
    operations: tuple[str, ...]
    space_group: str = ""
    atoms: tuple[AtomSite, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)
    symmetry: str = "file"
    """Where the operations come from: ``file``, ``symbol`` (generated from the space group) or ``none``
    (positions only)."""
    notes: tuple[str, ...] = ()
    """What was derived and how (``FROM_SYMBOL``, ``space_groups.ORIGIN_TAKEN``)."""
    formula: str = ""
    space_group_number: int = 0
    formula_sum: str = ""
    formula_units_z: Optional[float] = None


def _tokens(text: str) -> list[str]:
    """CIF tokens: tags, values (quotes removed), ``loop_`` and ``data_…``; a semicolon field is one token."""
    tokens: list[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(";"):
            field_lines = [line[1:]]
            index += 1
            while index < len(lines) and not lines[index].startswith(";"):
                field_lines.append(lines[index])
                index += 1
            tokens.append("\n".join(field_lines).strip())
            index += 1
            continue
        position, length = 0, len(line)
        while position < length:
            char = line[position]
            if char.isspace():
                position += 1
            elif char == "#":
                break
            elif char in "'\"":
                end = position + 1
                while end < length and not (line[end] == char and (end + 1 == length or line[end + 1].isspace())):
                    end += 1
                tokens.append(line[position + 1:end])
                position = end + 1
            else:
                end = position
                while end < length and not line[end].isspace():
                    end += 1
                tokens.append(line[position:end])
                position = end
        index += 1
    return tokens


def _blocks(tokens: list[str]) -> list[tuple[str, dict[str, list[str]]]]:
    """``(block name, {tag: values})`` of every data block; a single item is a list of one value."""
    blocks: list[tuple[str, dict[str, list[str]]]] = []
    items: Optional[dict[str, list[str]]] = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        lowered = token.lower()
        if lowered.startswith("data_"):
            items = {}
            blocks.append((token[5:], items))
            index += 1
        elif items is None:
            index += 1
        elif lowered == "loop_":
            index += 1
            tags = []
            while index < len(tokens) and tokens[index].startswith("_"):
                tags.append(_tag(tokens[index]))
                index += 1
            values = []
            while index < len(tokens) and not tokens[index].startswith("_") and tokens[index].lower() != "loop_" \
                    and not tokens[index].lower().startswith("data_"):
                values.append(tokens[index])
                index += 1
            for column, tag in enumerate(tags):
                items[tag] = values[column::len(tags)] if tags else []
        elif token.startswith("_"):
            value = tokens[index + 1] if index + 1 < len(tokens) else ""
            items[_tag(token)] = [value]
            index += 2
        else:
            index += 1
    return blocks


def _tag(token: str) -> str:
    return token.lower().replace(".", "_")


def number(value: str) -> Optional[float]:
    """A CIF number without its standard uncertainty; ``None`` for ``?``, ``.`` or text."""
    text = _ESD.sub("", str(value).strip())
    if text in ("", "?", "."):
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _first(items: dict[str, list[str]], tags) -> str:
    for tag in tags:
        values = items.get(tag)
        if values and values[0] not in ("?", "."):
            return values[0].strip()
    return ""


def _atoms(items: dict[str, list[str]], warnings: list[str]) -> tuple[AtomSite, ...]:
    labels = items.get("_atom_site_label") or []
    xs, ys, zs = (items.get(f"_atom_site_fract_{axis}") or [] for axis in "xyz")
    types = items.get("_atom_site_type_symbol") or labels
    occupancies = items.get("_atom_site_occupancy") or []
    u_values = items.get("_atom_site_u_iso_or_equiv") or []
    b_values = items.get("_atom_site_b_iso_or_equiv") or []
    sites = []
    for row in range(min(len(labels), len(xs), len(ys), len(zs))):
        position = tuple(number(values[row]) for values in (xs, ys, zs))
        if any(value is None for value in position):
            warnings.append(f"Atom {labels[row]} has no fractional coordinates and is left out.")
            continue
        try:
            element = element_of(types[row] if row < len(types) else labels[row])
        except ValueError:
            warnings.append(f"Atom {labels[row]}: unknown element, left out.")
            continue
        occupancy = number(occupancies[row]) if row < len(occupancies) else None
        b_iso = number(b_values[row]) if row < len(b_values) else None
        if b_iso is None:
            u_iso = number(u_values[row]) if row < len(u_values) else None
            b_iso = 8.0 * math.pi ** 2 * u_iso if u_iso is not None else 0.0
        sites.append(AtomSite(labels[row], element, position, 1.0 if occupancy is None else occupancy, max(b_iso, 0.0)))
    return tuple(sites)


def parse_cif(text: str) -> CifStructure:
    """The structure of the first data block with a cell (ValueError without one)."""
    blocks = _blocks(_tokens(text))
    for block_name, items in blocks:
        lengths = [number(_first(items, (f"_cell_length_{axis}",))) for axis in "abc"]
        if any(value is None for value in lengths):
            continue
        angles = [number(_first(items, (f"_cell_angle_{angle}",))) for angle in ("alpha", "beta", "gamma")]
        cell = UnitCell(*lengths, *(90.0 if value is None else value for value in angles))
        warnings: list[str] = []
        operations: tuple[str, ...] = ()
        for tag in SYMOP_TAGS:
            if items.get(tag):
                operations = tuple(value for value in items[tag] if value not in ("?", "."))
                break
        symbol = _first(items, SPACE_GROUP_TAGS)
        number_ = number_of(_first(items, NUMBER_TAGS))
        atoms = _atoms(items, warnings)
        symmetry, notes = "file", []
        if not operations:
            symmetry, operations, symbol, number_ = _derived(symbol, number_, _first(items, HALL_TAGS), cell, atoms,
                                                             warnings, notes)
        formula = " ".join(_first(items, FORMULA_TAGS).split())
        name = _first(items, NAME_TAGS) or _block_name(block_name) or formula or "phase"
        return CifStructure(" ".join(name.split()), cell, operations, symbol, atoms, tuple(warnings), symmetry,
                            tuple(notes), formula, number_, _first(items, ("_chemical_formula_sum",)),
                            number(_first(items, ("_cell_formula_units_z",))))
    raise ValueError("No data block with a unit cell (_cell_length_a, _b, _c) in this file.")


def number_of(value: str) -> int:
    """The space-group number of a CIF value (0 when it is none)."""
    found = number(value)
    return int(found) if found is not None and found == int(found) and 1 <= found <= 230 else 0


def _block_name(name: str) -> str:
    """The data block's name, unless it says nothing (``data_global``)."""
    text = str(name).strip()
    return "" if not text or text.casefold() in GENERIC_BLOCKS else text


def _derived(symbol: str, number_: int, hall: str, cell: UnitCell, atoms, warnings: list, notes: list):
    """``(symmetry, operations, symbol, number)`` from the space group named, when the file lists no operations."""
    sites = [atom.xyz for atom in atoms]
    try:
        try:
            group = space_group(symbol, number=number_ or None, hall=hall, cell=cell.as_tuple(), sites=sites)
        except SpaceGroupError:
            raise
        except ValueError:  # an unreadable Hall symbol: the Hermann–Mauguin symbol or the number may still do
            group = space_group(symbol, number=number_ or None, cell=cell.as_tuple(), sites=sites)
    except SpaceGroupError as error:
        warnings.extend([str(error) if symbol or number_ or hall else NO_SPACE_GROUP, POSITIONS_ONLY])
        return "none", (), symbol, number_
    notes.append(FROM_SYMBOL.format(count=len(group.operations), symbol=group.symbol, number=group.number))
    if group.note:
        notes.append(group.note)
    return "symbol", group.texts, symbol or group.symbol, group.number or number_


def read_cif(path) -> CifStructure:
    return parse_cif(Path(path).read_text(encoding="utf-8", errors="replace"))


__all__ = ["AtomSite", "CifStructure", "FROM_SYMBOL", "NO_SPACE_GROUP", "POSITIONS_ONLY", "number", "number_of",
           "parse_cif", "read_cif"]
