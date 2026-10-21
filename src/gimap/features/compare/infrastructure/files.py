"""Curve files in, tables and records out."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from ..application.sources import NO_TABLE, NOT_NUMBERS

NM_UNIT =re.compile(r"nm\s*(\^?-1|⁻¹|-1)|1\s*/\s*nm", re.IGNORECASE)
COLUMNS = "columns"
"""A comment line that names the columns begins with this word (``# columns: q (1/A)  I  sigma``)."""
CURVE = "curve:"
"""… and one that says which curve the file holds, with this (``# curve: In-plane sector I(q)``)."""
TABLE_TITLE = "): "
"""What a table of a batch says before its curve (``# GIMaP Analyze batch export (GIMaP Analyze 1): I(q)``)."""
HEAD_LINES = 12
"""Comment lines at the top of a file that are looked at for its title."""


def first_cell(text: str) -> str:
    """The name of the first column in a header row or a ``columns`` line, whatever parts the columns: commas,
    semicolons, tabs, two or more spaces (``q (1/A)  I  sigma``) or single ones (``q_(1/A) I sigma``)."""
    text = text.strip()
    for separator in (",", ";", "\t"):
        if separator in text:
            return text.split(separator)[0].strip()
    parts = re.split(r"\s{2,}", text)
    if len(parts) > 1:
        return parts[0]
    words = text.split()
    if len(words) > 1 and words[1].startswith("("):  # “q (1/A) I”: the unit belongs to the name
        return " ".join(words[:2])
    return words[0] if words else ""


def axis_label(name: str) -> str:
    """The x axis a column is, by its name as Analyze writes it (plain or with symbols) or as others do:
    ``chi (deg)`` → ``χ (°)``, ``|chi| (deg)`` → ``|χ| (°)``, ``2theta (deg)`` → ``2θ (°)``, ``qy (1/A)`` →
    ``qy (Å⁻¹)`` …; anything else is q."""
    text = name.lower().replace("_", " ")
    if "theta" in text or "θ" in text:
        return "2θ (°)"
    if "chi" in text or "χ" in text:
        return "|χ| (°)" if "|" in text else "χ (°)"
    for symbol in ("qy", "qz"):
        if symbol in text:
            return f"{symbol} (Å⁻¹)"
    if any(mark in text for mark in ("q par", "q∥", "q||", "qpar", "qxy")):
        return "q∥ (Å⁻¹)"
    return "q (Å⁻¹)"


def _columns_line(comment: str) -> str:
    """What a comment line says about the columns (``# columns: …``), else ``""``. A line that merely holds
    the word — Analyze's ``# region: {"rows": …, "columns": …}`` — says nothing about them."""
    text = comment.lstrip("#").strip()
    return text[len(COLUMNS):].lstrip(":").strip() if text.lower().startswith(COLUMNS) else ""


def _axis(header: str, columns: str) -> tuple[str, float]:
    """``(x label, factor that brings x to Å⁻¹)`` from the header row (it wins) and the ``columns`` line."""
    label = axis_label(first_cell(header) or first_cell(columns))
    in_nm = label.endswith("(Å⁻¹)") and NM_UNIT.search(f"{header} {columns}") is not None
    return label, 0.1 if in_nm else 1.0


class TextCurveReader:
    """Two or more columns (x, I, …), ``#`` comments, comma, tab or space separated; a header row is
    allowed. The x axis is read from the header row — the last line of words before the numbers —, else
    from a ``# columns:`` line: q, qy, qz, χ, 2θ (``axis_label``). q in nm⁻¹ (said there) is converted
    to Å⁻¹."""

    def read(self, path: Path) -> tuple[np.ndarray, np.ndarray, str]:
        text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        header, columns, rows = "", "", []
        for line in text:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if not rows:
                    columns = _columns_line(stripped) or columns
                continue
            # Cells between commas, semicolons, tabs and spaces (a regular expression gives the same cells and
            # takes three times as long over the lines of a thousand files).
            cells = [cell for cell in stripped.replace(",", " ").replace(";", " ").replace("\t", " ").split(" ") if cell]
            try:
                values = [float(cell) for cell in cells[:2]]
            except ValueError:
                if not rows:
                    header = stripped  # the header row
                continue
            if len(values) == 2:
                rows.append(values)
        if len(rows) < 3:
            raise ValueError("fewer than three points of two numbers")
        data = np.asarray(rows, dtype=float)
        label, factor = _axis(header, columns)
        return data[:, 0] * factor if factor != 1.0 else data[:, 0], data[:, 1], label

    @staticmethod
    def _table_rows(path: Path) -> list[list[str]]:
        """The rows of a table as cells, comment lines left out: comma- or tab-separated (cells may hold
        spaces, a comma-separated one may be quoted), else separated by white space."""
        lines = [line for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        if not lines:
            return []
        separator = "," if "," in lines[0] else "\t" if "\t" in lines[0] else None
        return list(csv.reader(lines, delimiter=separator)) if separator else [line.split() for line in lines]

    def read_table(self, path: Path) -> tuple[np.ndarray, np.ndarray, tuple, str]:
        """A table with x in its first column and one column per frame (Analyze's ``<batch>_<curve>_frames``):
        ``(x, intensities (frames × points), the frames' names, x label)``; an empty cell is no data (NaN)."""
        rows = self._table_rows(Path(path))
        if len(rows) < 4 or len(rows[0]) < 2:
            raise ValueError(NO_TABLE)
        header, width = rows[0], len(rows[0])
        try:
            data = np.asarray([[float(cell) if cell.strip() else np.nan for cell in (row + [""] * width)[:width]]
                               for row in rows[1:]], dtype=float)
        except ValueError as exc:
            raise ValueError(NOT_NUMBERS.format(error=exc)) from exc
        label, factor = _axis(header[0], "")
        kept = np.isfinite(data[:, 0])
        return data[kept, 0] * factor, data[kept, 1:].T, tuple(name.strip() for name in header[1:]), label

    def table_frames(self, path: Path) -> int:
        """How many frames a table holds (its columns after the first)."""
        with Path(path).open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if line.strip() and not line.lstrip().startswith("#"):
                    cells = next(csv.reader([line])) if "," in line else line.split("\t") if "\t" in line else line.split()
                    return max(0, len(cells) - 1)
        return 0

    def title(self, path: Path) -> str:
        """Which curve a file holds, as its first comment lines say it (``# curve: …``; a table of a batch
        names it after its own title); ``""`` when they do not."""
        try:
            with Path(path).open(encoding="utf-8", errors="replace") as stream:
                for _ in range(HEAD_LINES):
                    line = stream.readline()
                    if not line.startswith("#"):
                        break
                    text = line.lstrip("#").strip()
                    if text.lower().startswith(CURVE):
                        return text[len(CURVE):].strip()
                    if TABLE_TITLE in text and "batch export" in text:
                        return text.split(TABLE_TITLE, 1)[1].strip()
        except OSError:
            pass
        return ""


class CsvTableWriter:
    def write_table(self, path: Path, comments: Sequence[str], header: Sequence[str], rows) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            for line in comments:
                stream.write(f"# {line}\n")
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def write_record(self, path: Path, record: dict) -> Path:
        path = Path(path)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


__all__ = ["CsvTableWriter", "TextCurveReader", "axis_label", "first_cell"]
