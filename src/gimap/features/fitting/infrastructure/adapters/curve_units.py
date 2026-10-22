"""The unit of q a curve file states in its header (read by 1D Predict before it reads the numbers).

Where GIMaP writes it, and other programs write it too:

* a phrase ``q in <unit>`` in a comment line — Analyze's curve for Fitting begins
  ``# GIMaP Analyze fit input (q in 1/A)``, its curve tables ``# GIMaP Analyze export (q in 1/A, angles in deg)``,
  the model curves of a batch fit say ``q in 1/nm``;
* the name of the first column with its unit — the header row (the last line of words before the numbers:
  ``q (1/A),I,sigma,pixels``; Fitting's tables ``q_nm^-1,…``), else a ``# columns:`` comment line
  (``# columns: qy (1/A)  I  sigma  pixels``, what ``_fit_input.dat`` has). The header row wins over that line, as
  in Compare's reader of curve files; without either, the last comment line before the numbers is taken for the
  names (``# q (1/nm)  I  err``, as many programs write them).

Any usual spelling of the two units counts: ``1/A``, ``A^-1``, ``Å⁻¹``, ``1/Å``, ``1/angstrom``, ``A-1``;
``1/nm``, ``nm^-1``, ``nm⁻¹``, ``nm-1``. Nothing else in the header does: a sample called ``A-1`` on the
``# source:`` line states nothing, and a file whose first column is an angle (χ, 2θ) states no unit of q, whatever
its comment says. A header that states both units is refused (``BOTH_UNITS``), not guessed.

The units are named as Fitting names the unit of a curve's q (``CurveData.q_source_unit``): ``"angstrom"``, ``"nm"``.
"""

from __future__ import annotations

import re
from itertools import islice
from pathlib import Path
from typing import Iterable, Optional

ANGSTROM, NM = "angstrom", "nm"
BOTH_UNITS = "The header states q both in 1/A and in 1/nm; correct it to one unit."
"""Why a file's q cannot be read: its header contradicts itself (English: the record and the window's log)."""
HEAD_LINES = 200
"""Lines of a file looked at for its header; the numbers end the header earlier."""
COMMENTS = ("#", "%", ";", "//")
"""What begins a comment line (as for the curve reader of the Single analysis page)."""

_ANGSTROM = r"(?:Å|A|Ang|angstroms?)"
_INVERSE = r"(?:\^\s*\(?\s*-\s*1\s*\)?|⁻¹|-1|\*\*\s*-1)"
_PER_ANGSTROM = re.compile(rf"1\s*/\s*{_ANGSTROM}(?![A-Za-z])|(?<![A-Za-z]){_ANGSTROM}\s*{_INVERSE}", re.IGNORECASE)
_PER_NM = re.compile(rf"1\s*/\s*nm(?![A-Za-z])|(?<![A-Za-z])nm\s*{_INVERSE}", re.IGNORECASE)
_Q_IN = re.compile(r"(?<![A-Za-z])\|?q\|?\s+in\s+([^\s,;)\]]+)", re.IGNORECASE)
"""``q in 1/A`` (also ``|q| in 1/nm``): the unit is the word after ``in``."""
_IS_Q = re.compile(r"\s*\|?q", re.IGNORECASE)
"""A column of q: q, Q, |q|, qy, qz, qxy, q_par …"""
_OTHER_AXIS = re.compile(r"chi|χ|theta|θ|deg|°", re.IGNORECASE)
"""A first column of angles: the file states no unit of q for it."""
_CELLS = re.compile(r"[\s,;]+")


def _units_in(text: str) -> set[str]:
    found = set()
    if _PER_ANGSTROM.search(text):
        found.add(ANGSTROM)
    if _PER_NM.search(text):
        found.add(NM)
    return found


def _first_cell(text: str) -> str:
    """The name of the first column of a header row or a ``columns`` line, whatever parts the columns: commas,
    semicolons, tabs, two or more spaces (``q (1/A)  I  sigma``) or single ones (``q (1/A) I sigma``)."""
    text = text.strip()
    for separator in (",", ";", "\t"):
        if separator in text:
            return text.split(separator)[0].strip()
    parts = re.split(r"\s{2,}", text)
    if len(parts) > 1:
        return parts[0]
    words = text.split()
    if len(words) > 1 and words[1][:1] in "([":  # “q (1/A) I”: the unit belongs to the name
        return " ".join(words[:2])
    return words[0] if words else ""


def _is_numbers(text: str) -> bool:
    """A row of data: its first two cells are numbers."""
    cells = [cell for cell in _CELLS.split(text) if cell]
    if len(cells) < 2:
        return False
    try:
        float(cells[0]), float(cells[1])
    except ValueError:
        return False
    return True


def stated_q_unit(lines: Iterable[str]) -> Optional[str]:
    """``"angstrom"``, ``"nm"`` or ``None`` (no unit stated) from the lines of a file up to its first row of numbers
    (module docstring); ``ValueError(BOTH_UNITS)`` when they state both."""
    header, columns, comment, found = "", "", "", set()
    for line in lines:
        text = line.strip().lstrip("﻿")
        if not text:
            continue
        if text.startswith(COMMENTS):
            comment = text.lstrip("#%;/").strip()
            for match in _Q_IN.finditer(comment):
                found |= _units_in(match.group(1))
            if comment.lower().startswith("columns"):
                columns = comment[len("columns"):].lstrip(":").strip()
            continue
        if _is_numbers(text):
            break
        header = text  # the last line of words before the numbers
    cell = _first_cell(header) or _first_cell(columns)
    if _OTHER_AXIS.search(cell):  # χ or 2θ: Analyze's tables of them still say “q in 1/A”, for their other curves
        return None
    cell = cell or _first_cell(comment)  # no header row: the names may be in the last comment (“# q (1/nm)  I”)
    if _IS_Q.match(cell):
        found |= _units_in(cell)
    if len(found) > 1:
        raise ValueError(BOTH_UNITS)
    return next(iter(found), None)


def file_q_unit(path) -> Optional[str]:
    """The unit of q the file at ``path`` states (``stated_q_unit``), from its first ``HEAD_LINES`` lines.
    ``OSError`` when it cannot be read."""
    with Path(path).open(encoding="utf-8-sig", errors="replace") as stream:
        return stated_q_unit(islice(stream, HEAD_LINES))


__all__ = ["ANGSTROM", "BOTH_UNITS", "NM", "file_q_unit", "stated_q_unit"]
