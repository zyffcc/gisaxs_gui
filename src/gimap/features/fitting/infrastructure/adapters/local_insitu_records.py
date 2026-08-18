"""Local JSONL/CSV adapter for the existing in-situ record format."""

from __future__ import annotations

import csv
import json
from pathlib import Path


BASE_COLUMNS = (
    "file_index",
    "file_name",
    "timestamp",
    "load_status",
    "cut_status",
    "fit_status",
    "chi_square",
    "fitted_parameters",
    "error_message",
)


class LocalInSituRecordRepository:
    def cache_directory(self) -> Path:
        return Path.cwd() / ".gimap_cache"

    def session_path(self) -> Path:
        return self.cache_directory() / "insitu_current_session.jsonl"

    def ensure_directory(self) -> Path:
        directory = self.cache_directory()
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def reset(self) -> None:
        self.ensure_directory()
        self.session_path().write_text("", encoding="utf-8")

    def append(self, record) -> None:
        self.ensure_directory()
        with self.session_path().open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(dict(record), ensure_ascii=False, default=str) + "\n"
            )

    def load(self) -> list[dict[str, object]]:
        path = self.session_path()
        if not path.is_file():
            return []
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = line.strip()
                if value:
                    rows.append(json.loads(value))
        return rows

    def export_csv(self, path: Path, rows) -> Path:
        target = Path(path)
        extras = []
        for row in rows:
            for key in row:
                if key not in BASE_COLUMNS and key not in extras:
                    extras.append(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(BASE_COLUMNS) + extras,
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        return target
