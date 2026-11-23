"""Space groups from their symbols: the table of Hall symbols (``domain/crystal/space_group_table.py``), the
operations generated from it (``hall.py``), the lookup of a symbol as CIFs write it (``space_groups.py``) and a
CIF that names its space group without listing the operations (``cif.py``).

Every entry is checked against what its Hermann–Mauguin symbol says independently of the Hall symbol: the number
of general positions (order of the point group × lattice points per cell) and every glide plane and screw axis
the symbol names (the intrinsic translation of an operation of the group along that direction). The extinctions
of the groups of perovskites, lead iodide, the calibrants and common thin-film materials are checked by hand."""

from __future__ import annotations

import itertools
import re

import numpy as np
import pytest

from src.gimap.features.analyze.application.indexing import IndexingSettings, PhaseSpec, assign_peaks, phase_overlay
from src.gimap.features.analyze.domain.crystal import Phase, parse_cif, parse_operations, powder_rings, reflections
from src.gimap.features.analyze.domain.crystal.cif import POSITIONS_ONLY
from src.gimap.features.analyze.domain.crystal.hall import hall_operations
from src.gimap.features.analyze.domain.crystal.space_group_table import SPACE_GROUPS
from src.gimap.features.analyze.domain.crystal.space_groups import SpaceGroupError, space_group
from src.gimap.features.analyze.domain.crystal.symmetry import expand_sites, systematically_absent
from tests.test_crystal_indexing import MAPI_CIF

POINT_GROUP_ORDER = {
    "1": 1, "-1": 2, "2": 2, "m": 2, "2/m": 4, "222": 4, "mm2": 4, "mmm": 8, "4": 4, "-4": 4, "4/m": 8, "422": 8,
    "4mm": 8, "-42m": 8, "-4m2": 8, "4/mmm": 16, "3": 3, "-3": 6, "32": 6, "3m": 6, "-3m": 12, "6": 6, "-6": 6,
    "6/m": 12, "622": 12, "6mm": 12, "-6m2": 12, "-62m": 12, "6/mmm": 24, "23": 12, "m-3": 24, "432": 24,
    "-43m": 24, "m-3m": 48,
}
LATTICE_POINTS = {"P": 1, "A": 2, "B": 2, "C": 2, "I": 2, "F": 4, "R": 3}
SHIFTS = [np.array(vector) for vector in itertools.product((-1, 0, 1), repeat=3)]


def general_positions(symbol: str) -> int:
    """Order of the point group of the symbol (screws → rotations, glides → mirrors) × lattice points."""
    base, _, qualifier = symbol.partition(":")
    lattice, *parts = base.split()
    point = []
    for part in parts:
        part = re.sub(r"^(-?)([2346])[1-5]", r"\1\2", part)
        part = re.sub(r"/[abcnde]$", "/m", part)
        point.append("m" if re.fullmatch(r"[abcnde]", part) else part)
    if len(point) == 3 and point[0] == point[2] == "1":  # monoclinic full symbol
        point = [point[1]]
    name = "".join(point)
    name = {"321": "32", "312": "32", "3m1": "3m", "31m": "3m", "-3m1": "-3m", "-31m": "-3m"}.get(name, name)
    return POINT_GROUP_ORDER[name] * (1 if qualifier.strip() == "R" else LATTICE_POINTS[lattice])


def _directions(number: int):
    if number <= 74:
        return [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if number <= 194:
        return [(0, 0, 1), (1, 0, 0), (1, -1, 0)]
    return [(0, 0, 1), (1, 1, 1), (1, -1, 0)]


def _order(rotation) -> int:
    power = np.eye(3, dtype=int)
    for n in range(1, 7):
        power = rotation @ power
        if np.array_equal(power, np.eye(3, dtype=int)):
            return n
    return 0


def _counterclockwise(rotation, direction) -> bool:
    for vector in np.eye(3, dtype=int):
        if np.linalg.matrix_rank(np.vstack([vector, direction])) == 2:
            return np.linalg.det(np.vstack([vector, rotation @ vector, direction]).astype(float)) > 0
    return False


def _same(a, b) -> bool:
    difference = (np.asarray(a, float) - np.asarray(b, float)) % 1.0
    return bool(np.all(np.minimum(difference, 1.0 - difference) < 1e-6))


def _glides(letter: str, direction) -> list:
    axial = direction in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    in_plane = [index for index in range(3) if direction[index] == 0] if axial else [0, 1, 2]
    if letter == "m":
        return [(0, 0, 0)]
    if letter in "abc":
        return [tuple(0.5 if index == "abc".index(letter) else 0 for index in range(3))]
    if letter == "n":
        return [tuple(0 if direction[index] else 0.5 for index in range(3))] if axial else [(0.5, 0.5, 0.5)]
    if letter == "d":
        found = []
        for first, second in itertools.product((0.25, 0.75), repeat=2):
            vector = [0.0, 0.0, 0.0]
            if axial:
                vector[in_plane[0]], vector[in_plane[1]] = first, second
            else:
                vector = [first, first, second]
            found.append(tuple(vector))
        return found
    return [tuple(0.5 if index == axis else 0 for index in range(3)) for axis in in_plane]  # e: two of these


def symbol_mismatches(number: int, symbol: str, operations) -> list[str]:
    """The axes and planes of ``symbol`` that the operations do not have (empty: all there)."""
    base, _, qualifier = symbol.partition(":")
    if number <= 2 or qualifier.strip() == "R":
        return []
    missing = []
    for part, direction in zip(base.split()[1:], _directions(number)):
        if part == "1":
            continue
        axis, _, plane = part.partition("/")
        if re.fullmatch(r"[abcnmde]", axis):
            axis, plane = "", axis
        vector = np.array(direction)
        if axis:
            improper, n, screw = re.fullmatch(r"(-?)([12346])([1-5]?)", axis).groups()
            n, screw = int(n), int(screw or 0)
            found = False
            for operation in operations:
                rotation = -operation.rotation if improper else operation.rotation
                if (round(np.linalg.det(operation.rotation)) < 0) != bool(improper):
                    continue
                if not np.array_equal(rotation @ vector, vector) or _order(rotation) != n:
                    continue
                if n > 2 and not _counterclockwise(rotation, vector):
                    continue
                if improper:
                    found = True
                    break
                for shift in SHIFTS:
                    translation, power, total = operation.translation + shift, np.eye(3), np.zeros(3)
                    for _ in range(n):
                        total += power @ translation
                        power = rotation @ power
                    if _same(total / n, screw / n * vector):
                        found = True
                if found:
                    break
            if not found:
                missing.append(f"{axis} along {direction}")
        if plane:
            wanted, seen = _glides(plane, direction), set()
            for operation in operations:
                if round(np.linalg.det(operation.rotation)) > 0:
                    continue
                if not np.array_equal(-operation.rotation @ vector, vector) or _order(-operation.rotation) != 2:
                    continue
                for shift in SHIFTS:
                    translation = operation.translation + shift
                    glide = (translation + operation.rotation @ translation) / 2.0
                    seen.update(target for target in wanted if _same(glide, target))
            if len(seen) < (2 if plane == "e" else 1):
                missing.append(f"{plane} perpendicular to {direction}")
    return missing


# -- the table -------------------------------------------------------------------------------------------------

def test_every_space_group_type_is_in_the_table_with_its_number_of_general_positions() -> None:
    assert {number for number, _symbol, _hall in SPACE_GROUPS} == set(range(1, 231))
    for number, symbol, hall in SPACE_GROUPS:
        assert len(hall_operations(hall)) == general_positions(symbol), (number, symbol, hall)


def test_every_entry_has_the_glides_and_screws_its_symbol_names() -> None:
    for number, symbol, hall in SPACE_GROUPS:
        assert symbol_mismatches(number, symbol, hall_operations(hall)) == [], (number, symbol, hall)
    # the check is not blind: a neighbour's Hall symbol fails it
    assert symbol_mismatches(186, "P 63 m c", hall_operations("P 6c -2")) != []  # P63cm
    assert symbol_mismatches(76, "P 41", hall_operations("P 4cw")) != []  # P43
    assert symbol_mismatches(140, "I 4/m c m", hall_operations("-I 4 2")) != []  # I4/mmm


def test_the_two_origin_choices_put_the_inversion_centre_where_international_tables_do() -> None:
    inversion = lambda op: np.array_equal(op.rotation, -np.eye(3)) and _same(op.translation, 0)  # noqa: E731
    for number, symbol, hall in SPACE_GROUPS:
        if symbol.endswith(":2"):
            assert any(inversion(op) for op in hall_operations(hall)), symbol
        elif symbol.endswith(":1"):
            assert not any(inversion(op) for op in hall_operations(hall)), symbol
    # silicon: 8a is (0, 0, 0) with origin choice 1 and (1/8, 1/8, 1/8) with choice 2, eight atoms per cell
    assert len(expand_sites([(0, 0, 0)], space_group("F d -3 m :1").operations)[0]) == 8
    assert len(expand_sites([(0.125, 0.125, 0.125)], space_group("F d -3 m :2").operations)[0]) == 8


KNOWN = {  # symbol: (general positions, absent, present)
    "P -1": (2, [], [(1, 0, 0), (1, 1, 1)]),
    "P 21/c": (4, [(1, 0, 1), (0, 1, 0), (0, 0, 1)], [(1, 0, 2), (0, 2, 0), (1, 1, 1)]),
    "C 2/c": (8, [(1, 0, 0), (0, 1, 1), (2, 0, 1)], [(2, 0, 2), (1, 1, 1), (0, 2, 0)]),
    "P n m a": (8, [(0, 1, 0), (0, 2, 1), (1, 1, 0)], [(0, 1, 1), (2, 1, 0), (1, 0, 1)]),
    "I 4/m c m": (32, [(1, 0, 1), (1, 0, 3), (1, 0, 0), (0, 0, 1), (1, 1, 1)],
                  [(1, 1, 0), (0, 0, 2), (2, 1, 1), (2, 0, 2), (1, 1, 2), (2, 2, 0)]),
    "P m -3 m": (48, [], [(1, 0, 0), (1, 1, 0), (1, 1, 1)]),
    "F m -3 m": (192, [(1, 0, 0), (1, 1, 0), (2, 1, 0)], [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1)]),
    "F d -3 m": (192, [(2, 0, 0), (4, 2, 0), (1, 1, 0)], [(1, 1, 1), (2, 2, 0), (3, 1, 1), (4, 0, 0), (2, 2, 2)]),
    "I a -3 d": (96, [(1, 1, 0), (2, 0, 0), (3, 1, 0), (2, 2, 2)], [(2, 1, 1), (2, 2, 0), (3, 2, 1), (4, 0, 0)]),
    "P 63 m c": (12, [(0, 0, 1), (1, 1, 1)], [(1, 0, 0), (0, 0, 2), (1, 0, 1), (1, 1, 2)]),
    "P -3 m 1": (12, [], [(0, 0, 1), (1, 0, 0), (1, 0, 1)]),
    "R -3 m": (36, [(1, 0, 0), (0, 0, 1), (1, 1, 1)], [(0, 0, 3), (1, 0, 1), (0, 1, 2), (1, 1, 0)]),
    "P 42/m n m": (16, [(1, 0, 0), (0, 0, 1), (0, 1, 2)], [(1, 1, 0), (1, 0, 1), (2, 0, 0), (2, 1, 1)]),
    "I 41/a m d": (32, [(1, 0, 0), (0, 0, 2), (1, 1, 0), (1, 1, 1)], [(1, 0, 1), (0, 0, 4), (2, 0, 0), (1, 1, 2)]),
    "P 4 m m": (8, [], [(1, 0, 0), (0, 0, 1), (1, 0, 1)]),
}
"""Reflection conditions of International Tables A for the groups of thin-film materials: I4/mcm (tetragonal
MAPbI3), Pm-3m and Pnma (cubic and orthorhombic perovskites; LaB6), P-3m1 (PbI2), Fm-3m (Au, Ag, Cu, CeO2),
Fd-3m (Si), Ia-3d (garnets), P63mc (ZnO), P42/mnm and I41/amd (rutile, anatase), R-3m (Bi2Se3 and layered
halides), P21/c, P-1 and C2/c (organic semiconductors), P4mm (tetragonal BaTiO3)."""


@pytest.mark.parametrize("symbol", sorted(KNOWN))
def test_the_extinctions_of_the_groups_of_thin_film_materials(symbol: str) -> None:
    count, absent, present = KNOWN[symbol]
    operations = space_group(symbol).operations
    assert len(operations) == count
    assert systematically_absent(absent, operations).all() if absent else True
    assert not systematically_absent(present, operations).any()


# -- the lookup ------------------------------------------------------------------------------------------------

CUBIC = (5.431, 5.431, 5.431, 90, 90, 90)


def test_symbols_are_read_as_cifs_and_papers_write_them() -> None:
    tetragonal = (8.874, 8.874, 12.671, 90, 90, 90)
    monoclinic = (7.2, 5.9, 9.8, 90, 104.5, 90)
    assert space_group("I 4/m c m", cell=tetragonal).symbol == space_group("I4/mcm").symbol == "I 4/m c m"
    for written in ("P 21/c", "P2_1/c", "P2(1)/c", "P 1 21/c 1", "p21/C"):
        assert space_group(written, cell=monoclinic).symbol == "P 1 21/c 1", written
    assert space_group("P 21/n", cell=monoclinic).symbol == "P 1 21/n 1"
    assert space_group("Fm3m").number == 225 and space_group("Ia3d").number == 230  # the old cubic −3
    assert space_group("Cmca").symbol == "C m c e"  # the old symbol of No. 64
    pbnm = space_group("Pbnm", cell=(5.47, 5.64, 7.82, 90, 90, 90))  # orthorhombic perovskites, often so given
    assert pbnm.number == 62 and len(pbnm.operations) == 8 and pbnm.symbol == "P b n m"
    assert systematically_absent([(0, 1, 1), (1, 0, 2)], pbnm.operations).all()  # 0kl: k = 2n; h0l: h + l = 2n
    assert not systematically_absent([(1, 1, 0), (0, 0, 2), (1, 0, 1), (0, 2, 1)], pbnm.operations).any()
    assert space_group("R-3m", cell=(4.14, 4.14, 28.6, 90, 90, 120)).symbol == "R -3 m :H"
    assert space_group("R-3m", cell=(9.8, 9.8, 9.8, 24.2, 24.2, 24.2)).symbol == "R -3 m :R"
    assert space_group("F d -3 m S").symbol == "F d -3 m :1" and space_group("Fd-3m Z").symbol == "F d -3 m :2"
    assert space_group("F d -3 m", sites=[(0, 0, 0)]).symbol == "F d -3 m :1"  # fewer sites: the right origin
    assert space_group("F d -3 m", sites=[(0.125, 0.125, 0.125)]).symbol == "F d -3 m :2"
    assert "choice 1 was taken" in space_group("Fd-3m", sites=[(0, 0, 0)]).note
    assert space_group(number=140).symbol == "I 4/m c m" and space_group(number=14).symbol == "P 1 21/c 1"


@pytest.mark.parametrize("symbol, cell, says", [
    ("Q 9", None, "cannot be read"),
    ("P 1 1 21/b", (5, 6, 7, 90, 90, 104), "not in GIMaP's table"),  # unique axis c: a setting not tabulated
    ("P 21/c", (5, 6, 7, 90, 90, 104), "does not have the shape"),  # b-unique operations on a c-unique cell
    ("P 4/m m m", (4.0, 4.1, 5.0, 90, 90, 90), "does not have the shape"),
])
def test_a_symbol_that_does_not_give_the_operations_says_why(symbol, cell, says) -> None:
    with pytest.raises(SpaceGroupError, match=re.escape(says)):
        space_group(symbol, cell=cell)
    with pytest.raises(SpaceGroupError, match="disagree"):
        space_group("P n m a", number=14)


# -- CIFs without operations -----------------------------------------------------------------------------------

def _without_operations(text: str, *, symbol: str = "") -> str:
    lines = text.splitlines()
    start = lines.index("loop_")
    end = next(index for index in range(start + 1, len(lines)) if lines[index] == "loop_")
    kept = lines[:start] + lines[end:]
    kept = [line for line in kept if not line.startswith("_chemical_name_common")]
    kept.insert(2, "_chemical_formula_sum 'C H6 I3 N Pb'")
    if symbol:
        kept = [f"_symmetry_space_group_name_H-M '{symbol}'" if line.startswith("_symmetry_space_group_name_H-M")
                else line for line in kept]
    return "\n".join(kept)


def _rings(structure) -> dict:
    phase = Phase(structure.name, structure.cell, parse_operations(structure.operations), structure.atoms)
    rings = powder_rings(reflections(phase, 2.1), phase.operations)
    strongest = max(ring.intensity for ring in rings)
    return {ring.label: round(100 * ring.intensity / strongest, 3) for ring in rings}


def test_a_cif_naming_only_its_space_group_gives_the_reflections_of_the_full_cif() -> None:
    full = parse_cif(MAPI_CIF)
    short = parse_cif(_without_operations(MAPI_CIF))  # as copied from a paper: I 4/m c m and the atoms
    assert short.symmetry == "symbol" and len(short.operations) == 32 and not short.warnings
    assert short.notes[0] == ("32 symmetry operations generated from the space group I 4/m c m (No. 140): the file "
                              "lists none.")
    assert _rings(short) == _rings(full)
    assert "101" not in _rings(short) and "103" not in _rings(short)  # forbidden by the c glide of I4/mcm
    # the name is the data block's, not the sum formula (which is kept for the tooltip)
    assert (short.name, short.formula) == ("mapbi3", "C H6 I3 N Pb")
    assert full.name == "MAPbI3 (tetragonal)"


def test_a_cif_whose_symmetry_cannot_be_found_gives_positions_only_and_tentative_assignments(tmp_path) -> None:
    text = _without_operations(MAPI_CIF, symbol="P 1 1 21/b")
    structure = parse_cif(text)
    assert structure.symmetry == "none" and structure.operations == ()
    assert structure.warnings == ("The space group “P 1 1 21/b” is not in GIMaP's table (a setting it does not know).",
                                  POSITIONS_ONLY)
    unreadable = parse_cif(_without_operations(MAPI_CIF, symbol="I 4/m c m (unusual)"))
    assert unreadable.warnings == ("The space-group symbol “I 4/m c m (unusual)” cannot be read.", POSITIONS_ONLY)
    unnamed = parse_cif(_without_operations(MAPI_CIF).replace("_symmetry_space_group_name_H-M", "_x_unused"))
    assert unnamed.warnings == ("The file names no space group.", POSITIONS_ONLY) and unnamed.symmetry == "none"
    path = tmp_path / "loose.cif"
    path.write_text(text, encoding="utf-8")
    from src.gimap.features.analyze.application.indexing import phase_from_cif

    spec = phase_from_cif(path)
    assert spec.positions_only and spec.record()["positions_only"] and spec.record()["warnings"][-1] == POSITIONS_ONLY
    overlay = phase_overlay(spec, IndexingSettings(q_max=1.7), None)
    labels = {label for ring in overlay.rings for label in ring.labels}
    assert {"101", "110", "002"} <= labels  # every hkl: no extinctions without the symmetry
    assert not overlay.with_atoms and {ring.intensity / ring.multiplicity for ring in overlay.rings} == {1.0}
    found = assign_peaks([{"q": 1.0013, "q_err": 0.0005, "fwhm": 0.01}], [spec])
    assert found["assignments"][0]["assigned"] and found["assignments"][0]["tentative"]
    good = phase_from_cif(_write(tmp_path / "good.cif", _without_operations(MAPI_CIF)))
    assert not assign_peaks([{"q": 1.0013, "q_err": 0.0005, "fwhm": 0.01}], [good])["assignments"][0]["tentative"]


def _write(path, text: str):
    path.write_text(text, encoding="utf-8")
    return path


def test_the_symmetry_column_keeps_the_warning_of_a_phase_without_operations(tmp_path) -> None:
    from src.gimap.features.analyze.application.indexing import phase_from_cif
    from src.gimap.features.analyze.presentation.indexing_texts import phase_notes, symmetry_text

    loose = phase_from_cif(_write(tmp_path / "loose.cif", _without_operations(MAPI_CIF, symbol="P 1 1 21/b")))
    assert symmetry_text(loose).startswith("⚠ P 1 1 21/b, positions only")
    assert POSITIONS_ONLY in phase_notes(loose)
    derived = phase_from_cif(_write(tmp_path / "derived.cif", _without_operations(MAPI_CIF)))
    assert symmetry_text(derived) == ("I 4/m c m, 32 operations from the space group (none in the CIF); 3 atom sites: "
                                      "|F|² from the atoms")
    # the test CIF lists Pb and I sites only, so the sum-formula check (C H6 I3 N Pb) warns first
    assert "\n32 symmetry operations generated from the space group I 4/m c m" in phase_notes(derived)
    assert isinstance(PhaseSpec("typed", CUBIC).phase(), Phase)  # a typed cell keeps its centring
