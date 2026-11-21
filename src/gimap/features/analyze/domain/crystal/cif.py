"""A small reader of crystallographic information files (CIF 1.1): the cell, the symmetry and the atom sites.

Read from the first data block that gives a cell:

* ``_cell_length_a/b/c`` (Å) and ``_cell_angle_alpha/beta/gamma`` (°); standard uncertainties in brackets,
  ``8.874(2)``, are dropped;
* the symmetry operations of ``_space_group_symop_operation_xyz`` or ``_symmetry_equiv_pos_as_xyz`` (also with
  a dot, ``_space_group_symop.operation_xyz``), and the Hermann–Mauguin symbol;
* the atom sites: ``_atom_site_label``, ``_atom_site_type_symbol``, ``_atom_site_fract_x/y/z``,
  ``_atom_site_occupancy`` (default 1), ``_atom_site_U_iso_or_equiv`` or ``_atom_site_B_iso_or_equiv``
  (B = 8π² U; default 0).

Without operations only the lattice centring of the Hermann–Mauguin symbol is applied (its first letter), and the
reader says so (``warnings``). Quoted values, semicolon text fields and comments are understood; everything else
in the file (bonds, anisotropic displacements, refinement details) is ignored.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .cell import UnitCell
from .scattering import element_of

SYMOP_TAGS = ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz")
SPACE_GROUP_TAGS = ("_space_group_name_h-m_alt", "_symmetry_space_group_name_h-m", "_space_group_name_h-m")
NAME_TAGS = ("_chemical_name_common", "_chemical_name_mineral", "_chemical_formula_sum", "_chemical_name_systematic")
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
        space_group = _first(items, SPACE_GROUP_TAGS)
        if not operations:
            warnings.append(
                "The file lists no symmetry operations: only the atoms as listed are used"
                + (f", with the lattice centring of {space_group}." if space_group else " (P1).")
            )
        name = _first(items, NAME_TAGS) or block_name or "phase"
        return CifStructure(" ".join(name.split()), cell, operations, space_group, _atoms(items, warnings), tuple(warnings))
    raise ValueError("No data block with a unit cell (_cell_length_a, _b, _c) in this file.")


def read_cif(path) -> CifStructure:
    return parse_cif(Path(path).read_text(encoding="utf-8", errors="replace"))


__all__ = ["AtomSite", "CifStructure", "number", "parse_cif", "read_cif"]
