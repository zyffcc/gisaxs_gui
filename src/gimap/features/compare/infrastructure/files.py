"""Curve files in, tables and records out."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

NM_UNIT = re.compile(r"nm\s*(\^?-1|⁻¹|-1)|1\s*/\s*nm", re.IGNORECASE)


class TextCurveReader:
    """Two or more columns (x, I, …), ``#`` comments, comma, tab or space separated; a header line is
    allowed. x in nm⁻¹ (said in a header or ``# columns:`` line) is converted to Å⁻¹."""

    def read(self, path: Path) -> tuple[np.ndarray, np.ndarray, str]:
        text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        described, rows = "", []
        for line in text:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if "columns" in stripped.lower() or not described:
                    described = stripped if "columns" in stripped.lower() else described
                continue
            cells = [cell for cell in re.split(r"[,\t; ]+", stripped) if cell]
            try:
                values = [float(cell) for cell in cells[:2]]
            except ValueError:
                described = described or stripped  # a header line
                continue
            if len(values) == 2:
                rows.append(values)
        if len(rows) < 3:
            raise ValueError("fewer than three points of two numbers")
        data = np.asarray(rows, dtype=float)
        x, intensity = data[:, 0], data[:, 1]
        if NM_UNIT.search(described):
            x = x / 10.0
        lower = described.lower()
        name = "qy" if "qy" in lower else "qz" if "qz" in lower else "q"
        return x, intensity, f"{name} (Å⁻¹)"


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


__all__ = ["CsvTableWriter", "TextCurveReader"]
