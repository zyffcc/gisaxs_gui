"""Inspect text tables and read an explicitly selected curve without mixing rows or columns."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...application.curve_sources import ColumnMapping, CurveTablePreview, validate_columns
from .curve_units import COMMENTS, file_q_unit
from .scattering_curve_files import NO_DATA_ROWS, TOO_FEW_POINTS, XYData, _optional_float, _split, _to_float


@dataclass(frozen=True)
class _Table:
    preview: CurveTablePreview
    rows: tuple[tuple[str, ...], ...]
    delimiter: str
    first_data_line: int


def _delimiter(text: str) -> str:
    if "\t" in text:
        return "\t"
    if ";" in text:
        return ";"
    if "," in text and (" " not in text or text.count(",") >= text.count(" ")):
        return ","
    return "whitespace"


def _is_data(cells: list[str], delimiter: str) -> bool:
    if len(cells) < 2:
        return False
    try:
        _to_float(cells[0], delimiter != ",")
    except ValueError:
        return False
    # An empty first intensity must not hide later, valid intensity columns.
    for cell in cells[1:]:
        try:
            _to_float(cell, delimiter != ",")
            return True
        except ValueError:
            continue
    return False


def _header_cells(text: str, delimiter: str) -> tuple[str, ...]:
    if not text:
        return ()
    if delimiter != "whitespace" and delimiter in text:
        return tuple(cell.strip() for cell in text.split(delimiter))
    # A parenthesized unit belongs to its column name, including single-spaced headers.
    return tuple(re.findall(r"[^\s]+(?:\s+\([^)]*\)|\s+\[[^]]*\])?", text.strip()))


def _column_kind(name: str) -> str:
    text = re.sub(r"\([^)]*\)|\[[^]]*\]", "", name).strip().lower()
    if re.fullmatch(r"(?:dq|sigma[_ ]?q|σ[_ ]?q)(?:[_ ]?(?:nm|a|å|m).*)?", text):
        return "dq"
    if re.fullmatch(r"\|?q\|?(?:[xyz]|xy|zxy|_?par|_?perp)?(?:[_ ]?(?:nm|a|å).*)?", text):
        return "q"
    if re.fullmatch(r"(?:(?:i|intensity)[ _-])?(?:sigma|σ|err(?:or)?|di|d_i|std(?:dev)?|uncertainty)"
                    r"(?:[ _-]?\d+|[ _-].+)?", text):
        return "sigma"
    if re.fullmatch(r"(?:i|intensity)(?:[ _-]?\d+|[ _-].+)?", text):
        return "intensity"
    if text in {"pixels", "pixel_count", "npixels"}:
        return "pixels"
    return "unknown"


def _table_from_lines(path: Path, lines, *, preview_only: bool) -> _Table:
    header, columns, column_comment = "", "", ""
    analyze = False
    first, delimiter = None, "whitespace"
    for index, line in enumerate(lines):
        text = line.strip().lstrip("\ufeff")
        if not text:
            continue
        if text.startswith(COMMENTS):
            comment = text.lstrip("#%;/").strip()
            analyze |= comment.startswith("GIMaP Analyze fit input") or comment.startswith("GIMaP Analyze export")
            if comment.lower().startswith("columns:"):
                columns = comment.split(":", 1)[1].strip()
            candidate_names = _header_cells(comment, _delimiter(comment))
            candidate_kinds = tuple(map(_column_kind, candidate_names))
            if len(candidate_names) >= 2 and (candidate_kinds[0] == "q" or {"q", "dq", "intensity"} <= set(candidate_kinds)):
                column_comment = comment
            continue
        candidate = _delimiter(text)
        if _is_data(_split(text, candidate), candidate):
            first, delimiter = index, candidate
            break
        try:
            _to_float(_split(text, candidate)[0], candidate != ",")
        except ValueError:
            pass
        else:
            continue  # a row with no readable intensities is not a new column header
        header = text
    if first is None:
        raise ValueError(NO_DATA_ROWS)
    collected = [tuple(cell.strip() for cell in _split(text, delimiter))]
    for line in lines:
        text = line.strip().lstrip("\ufeff")
        if not text or text.startswith(COMMENTS):
            continue
        collected.append(tuple(cell.strip() for cell in _split(text, delimiter)))
        if preview_only and len(collected) >= 10:
            break
    rows = tuple(collected)
    count = max(len(row) for row in rows if _is_data(row, delimiter))
    names = _header_cells(header or columns, delimiter)
    if names and _column_kind(names[0]) != "q" and len(names) != count:
        names = _header_cells(columns, delimiter)
    if not names:
        candidate = _header_cells(column_comment, delimiter)
        if candidate:
            names = candidate
    has_header = bool(names)
    # A named header defines the schema. Short data rows are kept for per-curve validation.
    if names:
        count = len(names)
    else:
        names = tuple(f"Column {index + 1}" for index in range(count))
    kinds = tuple(map(_column_kind, names)) if has_header else ()
    analyze |= len(kinds) >= 4 and kinds[:4] == ("q", "intensity", "sigma", "pixels")
    if (analyze or count == 2 or (len(kinds) >= 3 and kinds[:3] == ("q", "intensity", "sigma"))
            or ("dq" in kinds and kinds.count("intensity") == 1)):
        suggested = "single"
    elif count >= 3 and kinds and kinds[0] == "q" and all(kind == "intensity" for kind in kinds[1:]):
        suggested = "shared_q"
    else:
        suggested = "ambiguous"
    uncertainty = tuple(index for index, kind in enumerate(kinds) if kind == "sigma")
    intensity = tuple(index for index, kind in enumerate(kinds) if kind == "intensity")
    resolution = tuple(index for index, kind in enumerate(kinds) if kind == "dq")
    preview = CurveTablePreview(path, names, rows[:10], count, suggested, analyze, file_q_unit(path), has_header,
                                uncertainty, intensity, resolution_columns=resolution,
                                q_columns=tuple(index for index, kind in enumerate(kinds) if kind == "q"))
    return _Table(preview, rows, delimiter, first + 1)


def _read_table(path: Path, *, preview_only: bool = False) -> _Table:
    # Do not read the file into a string for a preview: large shared-q tables can contain millions of rows.
    # utf-8-sig also accepts ordinary UTF-8; Latin-1 retains the legacy reader's encoding fallback.
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with path.open(encoding=encoding) as stream:
                return _table_from_lines(path, iter(stream), preview_only=preview_only)
        except UnicodeDecodeError:
            if encoding == "latin-1":
                raise
    raise ValueError(NO_DATA_ROWS)


def inspect_curve_table(path: Path) -> CurveTablePreview:
    return _read_table(path, preview_only=True).preview


def load_table_curve(path: Path, columns: ColumnMapping, *, keep_missing_sigma: bool = False) -> XYData:
    table = _read_table(path)
    preview = table.preview
    validate_columns(preview, columns)
    decimal_comma = table.delimiter != ","
    valid, bad_lines = [], 0
    for row in table.rows:
        if table.delimiter == "whitespace" and len(row) < preview.column_count:
            # Whitespace cannot mark an empty cell: guessing which column disappeared would mix curves.
            bad_lines += 1
            continue
        try:
            values = (_to_float(row[columns.q], decimal_comma), _to_float(row[columns.intensity], decimal_comma))
        except (IndexError, ValueError):
            bad_lines += 1
            continue
        if not all(np.isfinite(value) for value in values):
            bad_lines += 1
            continue
        for index in (columns.sigma, columns.dq):
            if index is not None:
                values += (_optional_float(row, index, decimal_comma),)
        if columns.sigma is not None and not keep_missing_sigma and not np.isfinite(values[2]):
            bad_lines += 1
            continue
        valid.append(values)
    if len(valid) < 2:
        raise ValueError(TOO_FEW_POINTS)
    array = np.asarray(valid, dtype=float)
    error = array[:, 2] if columns.sigma is not None else None
    return XYData(array[:, 0], array[:, 1], error, {
        "path": str(path), "data_start_line": table.first_data_line, "bad_lines": bad_lines,
        "has_error": error is not None, "rows_left_out": bad_lines,
        "dq_column_name": preview.column_names[columns.dq] if columns.dq is not None and preview.has_header else "",
        "delimiter": {"\t": "tab", ",": "comma", ";": "semicolon"}.get(table.delimiter, "whitespace"),
    }, dq=None if columns.dq is None else array[:, -1])
