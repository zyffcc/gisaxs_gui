"""Read a toolkit result folder: ``structure*.csv`` and ``structure*_summary.json``.

``physical`` and ``literature`` both come from ``structure_physical.csv``. The network's physical path is always
the default explanation (first); the literature method (R from the nominal thickness and 2π/q*) is a check of it,
offered second when the summary says it has values (a deposition rate was given), never the default. The scan a
result was predicted from is read from ``gimap_request.json`` (written by GIMaP before the model runs).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from typing import Optional

from ..application import NO_TABLE, StructureResult
from ..domain import literature_available, literature_rows, parse_row
from .toolkit import REQUEST_FILE

VARIANTS = {"physical": ("structure_physical.csv", "structure_physical_summary.json"),
            "literature": ("structure_physical.csv", "structure_physical_summary.json"),
            "window": ("structure.csv", "structure_summary.json"),
            "selected": ("structure_selected.csv", "structure_selected_summary.json"),
            "sequence": ("structure_sequence.csv", "structure_sequence_summary.json")}


def _summary(path: Path) -> dict:
    """The summary for choosing the variants: an unreadable one offers no literature method (``read`` says why)."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


class StructureFolderReader:
    def available(self, folder: Path) -> list[str]:
        """Variants present in a result folder, in display order: the physical path (the default), the literature
        method right after it when it has values (a check of the network), then the per-window choice, the learned
        selector and the time-consistent path."""
        folder = Path(folder)
        present = [v for v, (table, _) in VARIANTS.items() if v != "literature" and (folder / table).is_file()]
        if "physical" in present and literature_available(_summary(folder / VARIANTS["physical"][1])):
            present.insert(present.index("physical") + 1, "literature")
        return present

    def read(self, folder: Path, variant: str = "physical") -> StructureResult:
        folder = Path(folder)
        table_name, summary_name = VARIANTS[variant]
        table = folder / table_name
        if not table.is_file():
            raise FileNotFoundError(NO_TABLE.format(folder=folder, table=table_name))
        with table.open(encoding="utf-8", newline="") as f:
            rows = [parse_row(r) for r in csv.DictReader(f)]
        summary_file = folder / summary_name  # a corrupt summary is reported, not read as empty
        summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.is_file() else {}
        rows.sort(key=lambda r: (r.get("time_s") is None, r.get("time_s") or 0.))
        if variant == "literature":
            rows = literature_rows(rows, summary)
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
