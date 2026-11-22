"""Which columns of a CSV curve are q, I and σ — by the names of its header, never by their place alone.

GIMaP writes curves as CSV in several layouts: the command line's cuts (``# qy (A^-1), I, sigma`` as the last comment
line), Analyze's curve tables (``q,I,sigma,pixels``), and Fitting's own Save ▸ Data and Fit (``q_nm^-1,q_A^-1,I,sigma,
model,residual,…``), where the second column is q again, in Å⁻¹. ``header_columns`` takes

* q: the column named q (q, Q, |q|, qy, qz, qxy …); several only when they are one q in different units, stated in
  their names (Fitting's ``q_nm^-1`` and ``q_A^-1``): then the first;
* I: the one column named I or intensity;
* σ: the one column named sigma, σ, err, dI … (none: no σ).

Columns of other names (model, residual, the terms, pixels) are not read. A table whose header does not decide —
two intensities, two q of different axes, no header and more than three columns — is ``COLUMNS_TO_CHOOSE``: the page
shows the table and asks. A CSV without a header of two or three columns is read as q, I [, σ], like a .dat file.

The unit of q is the one the chosen column's name states (``q_nm^-1`` → nm⁻¹), else for a first column what the
header says of it (``curve_units``), else none (the setting for files without one).
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Optional

from ...application.curve_sources import ColumnMapping, CurveTablePreview
from ...application.models import COLUMNS_TO_CHOOSE
from src.gimap.shared.nxcansas import q_unit
from .curve_tables import _column_kind, inspect_curve_table
from .curve_units import _units_in, file_q_unit

POSITIONAL_COLUMNS = 3
"""A CSV without a header is read by place (q, I, σ) up to this many columns."""
_UNIT_TEXT = re.compile(r"\([^)]*\)|\[[^]]*\]|[_ ]?(?:nm|a|å|ang(?:strom)?)\s*(?:\^\s*\(?-1\)?|-1|⁻¹)?$", re.IGNORECASE)


def column_unit(name: Optional[str]) -> Optional[str]:
    """``"angstrom"`` / ``"nm"`` when the column's name states one unit of q, else ``None``."""
    if not name:
        return None
    units = _units_in(str(name))
    return next(iter(units)) if len(units) == 1 else None


def _base(name: str) -> str:
    """The name without its unit (``q_nm^-1`` and ``q_A^-1`` → ``q``; ``qy (A^-1)`` → ``qy``)."""
    text = str(name).strip()
    previous = None
    while previous != text:
        previous, text = text, _UNIT_TEXT.sub("", text).strip()
    return text.lower()


def header_columns(preview: CurveTablePreview) -> Optional[ColumnMapping]:
    """q, I and σ of a table by the names of its header (module docstring); ``None`` when they do not decide."""
    if not preview.has_header:
        return None
    names = preview.column_names
    kinds = [_column_kind(name) for name in names]
    q = [index for index, kind in enumerate(kinds) if kind == "q"]
    intensity = [index for index, kind in enumerate(kinds) if kind == "intensity"]
    sigma = [index for index, kind in enumerate(kinds) if kind == "sigma"]
    if not q or len(intensity) != 1 or len(sigma) > 1:
        return None
    if len(q) > 1:
        units = [column_unit(names[index]) for index in q]
        if len({_base(names[index]) for index in q}) != 1 or None in units or len(set(units)) != len(units):
            return None  # q of different axes (qy, qz), or the same name twice: not to be guessed
    if len(preview.resolution_columns) > 1:
        return None
    dq = preview.resolution_columns[0] if preview.resolution_columns else None
    return ColumnMapping(q[0], intensity[0], sigma[0] if sigma else None, tuple(names), preview.column_count, dq)


def curve_columns(path: Path, selection: Optional[ColumnMapping] = None) -> Optional[ColumnMapping]:
    """Use named columns for CSV and for any table with named dQ. Preserve positional reads of older text files."""
    preview = inspect_curve_table(Path(path))
    if selection is not None:
        if selection.dq is None and len(preview.resolution_columns) == 1:
            return replace(selection, dq=preview.resolution_columns[0])
        return selection
    if path.suffix.lower() != ".csv" and not preview.resolution_columns:
        return None
    found = header_columns(preview)
    if found is not None:
        return found
    if not preview.has_header and preview.column_count <= POSITIONAL_COLUMNS:
        return None
    raise ValueError(COLUMNS_TO_CHOOSE)


def resolution_unit(name: str) -> Optional[str]:
    """The stated inverse-length unit of a dQ column. Bare names use the unit of q."""
    unit_text = re.sub(r"^(?:dq|sigma[_ ]?q|σ[_ ]?q)", "", name.strip(), flags=re.IGNORECASE)
    unit_text = unit_text.strip(" _")
    if unit_text.startswith(("(", "[")) and unit_text.endswith((")", "]")):
        unit_text = unit_text[1:-1].strip()
    if not unit_text:
        return None
    unit = q_unit(unit_text)
    if unit is None:
        raise ValueError("The unit of dQ must be an inverse length.")
    return unit


def column_q_unit(path: Path, columns: ColumnMapping) -> Optional[str]:
    """The unit of q of the chosen q column: what its name states, else (the first column) what the file's header
    says of its q (``curve_units.file_q_unit``), else ``None``."""
    name = columns.column_names[columns.q] if columns.column_names else None
    stated = column_unit(name)
    if stated is not None:
        return stated
    return file_q_unit(path) if columns.q == 0 else None


__all__ = ["POSITIONAL_COLUMNS", "column_q_unit", "column_unit", "curve_columns", "header_columns", "resolution_unit"]
