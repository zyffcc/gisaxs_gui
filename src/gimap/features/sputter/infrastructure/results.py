"""Read a toolkit result folder: ``structure*.csv`` and ``structure*_summary.json``.

``physical`` (``structure_physical.csv``) comes first: when it is there it is the default explanation. The
scan a result was predicted from is read from ``gimap_request.json`` (written by GIMaP before the model runs).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from typing import Optional

from ..application import NO_TABLE, StructureResult
from ..domain import parse_row
from .toolkit import REQUEST_FILE

VARIANTS = {"physical": ("structure_physical.csv", "structure_physical_summary.json"),
            "window": ("structure.csv", "structure_summary.json"),
            "selected": ("structure_selected.csv", "structure_selected_summary.json"),
            "sequence": ("structure_sequence.csv", "structure_sequence_summary.json")}


class StructureFolderReader:
    def available(self, folder: Path) -> list[str]:
        """Variants present in a result folder, in display order (the default first)."""
        return [v for v, (table, _) in VARIANTS.items() if (Path(folder) / table).is_file()]

    def read(self, folder: Path, variant: str = "physical") -> StructureResult:
        folder = Path(folder)
        table_name, summary_name = VARIANTS[variant]
        table = folder / table_name
        if not table.is_file():
            raise FileNotFoundError(NO_TABLE.format(folder=folder, table=table_name))
        with table.open(encoding="utf-8", newline="") as f:
            rows = [parse_row(r) for r in csv.DictReader(f)]
        summary_file = folder / summary_name
        summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.is_file() else {}
        rows.sort(key=lambda r: (r.get("time_s") is None, r.get("time_s") or 0.))
        return StructureResult(folder, rows, summary, variant, scan_of(folder))


def scan_of(folder: Path) -> Optional[Path]:
    """The scan folder a result was predicted from, as it was chosen (None when the folder does not say)."""
    try:
        record = json.loads((Path(folder) / REQUEST_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    request = record.get("request") if isinstance(record.get("request"), dict) else {}
    scan = record.get("scan_folder") or request.get("scan_folder")
    return Path(scan) if isinstance(scan, str) and scan else None


__all__ = ["StructureFolderReader", "VARIANTS", "scan_of"]
