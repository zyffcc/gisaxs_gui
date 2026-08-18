"""Local JSON/CSV adapter for Classification sessions and exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path


class LocalClassificationArtifactRepository:
    def save_session(self, path: Path, values: dict) -> None:
        Path(path).write_text(
            json.dumps(values, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_session(self, path: Path) -> dict:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Classification session must contain a JSON object.")
        return values

    def export_csv(self, path: Path, columns: tuple[str, ...], rows: tuple[tuple, ...]) -> None:
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
