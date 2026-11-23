"""Compare Scans: several Analyze results side by side (``domain/compare.py``).

``CompareService`` lists the results analysed under a results root (newest first), opens a result folder as a compared
scan (its result through ``ClientService``, the user's choice of solution per window, and the model-free readings of
the package's report through ``CompareFiles``), and writes the exported tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol

from ..domain.compare import ComparedScan, compare_csv, default_name, growth_csv, report_readings

MAX_LISTED = 40
"""Results listed from a results root (the newest)."""
EXPORT_FILES = ("compare_curves.csv", "compare_growth.csv", "compare.png", "compare.pdf")
"""The files Export… writes (tables, then the figure)."""


class CompareFiles(Protocol):
    def analysed(self, root: Path, limit: int) -> list:
        """Folders under ``root`` holding a complete Analyze result, newest first."""

    def report_text(self, path: Optional[Path]) -> str:
        """The text of a result's ``report.csv`` ("" when it is missing or unreadable)."""

    def write_text(self, path: Path, text: str) -> Path: ...


class CompareService:
    def __init__(self, client, files: CompareFiles):
        self.client = client
        """``ClientService``: opens a result and its choices."""
        self.files = files

    def analysed(self, roots) -> list:
        """Result folders under each root (newest first per root, each folder once)."""
        seen, out = set(), []
        for root in roots:
            if root is None:
                continue
            for folder in self.files.analysed(Path(root), MAX_LISTED):
                key = str(Path(folder).resolve())
                if key not in seen:
                    seen.add(key)
                    out.append(Path(folder))
        return out

    def load(self, folder, name: str = "") -> ComparedScan:
        """The result in (or under) ``folder`` as a compared scan; raises ``OSError`` / ``ValueError`` like
        ``ClientService.open``."""
        result = self.client.open(Path(folder))
        try:
            choices = self.client.choices(result)
        except (OSError, ValueError):
            choices = {}
        readings = report_readings(self.files.report_text(result.file("report_csv")))
        return ComparedScan(name or default_name(result), result, readings, choices)

    @staticmethod
    def tables(scans, axis: str) -> tuple:
        """(text of the curves table, text of the growth table)."""
        return compare_csv(scans, axis), growth_csv(scans)

    def write_tables(self, scans, axis: str, folder) -> list:
        curves, growth = self.tables(scans, axis)
        folder = Path(folder)
        return [self.files.write_text(folder / EXPORT_FILES[0], curves),
                self.files.write_text(folder / EXPORT_FILES[1], growth)]


__all__ = ["CompareFiles", "CompareService", "EXPORT_FILES", "MAX_LISTED"]
