"""Read a toolkit result folder: ``structure.csv`` and ``structure_summary.json``."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..domain import StructureResult, parse_row


VARIANTS = {"window": ("structure.csv", "structure_summary.json"),
            "selected": ("structure_selected.csv", "structure_selected_summary.json"),
            "sequence": ("structure_sequence.csv", "structure_sequence_summary.json")}


class StructureFolderReader:
    def available(self, folder: Path) -> list[str]:
        """Variants present in a result folder, in display order."""
        return [v for v, (table, _) in VARIANTS.items() if (Path(folder) / table).is_file()]

    def read(self, folder: Path, variant: str = "window") -> StructureResult:
        folder = Path(folder)
        table_name, summary_name = VARIANTS[variant]
        table = folder / table_name
        if not table.is_file():
            raise FileNotFoundError(f"{folder} has no {table_name} (run Analyze first, or choose a toolkit output folder).")
        with table.open(encoding="utf-8", newline="") as f:
            rows = [parse_row(r) for r in csv.DictReader(f)]
        summary_file = folder / summary_name
        summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.is_file() else {}
        rows.sort(key=lambda r: (r.get("time_s") is None, r.get("time_s") or 0.))
        return StructureResult(folder, rows, summary, variant)


__all__ = ["StructureFolderReader"]
