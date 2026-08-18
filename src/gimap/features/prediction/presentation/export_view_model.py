"""Prediction export commands and presentation-neutral error mapping."""

from __future__ import annotations

from pathlib import Path


class PredictionExportViewModel:
    def __init__(self, *, jsonl, ascii, array, on_error):
        self._jsonl = jsonl
        self._ascii = ascii
        self._array = array
        self._on_error = on_error

    def _run(self, command, unavailable_message, *args):
        if command is None:
            self._on_error(unavailable_message)
            return None
        try:
            return command.execute(*args)
        except Exception as exc:
            self._on_error(str(exc))
            return None

    def export_jsonl(self, items, export_path: Path, timestamp: str):
        return self._run(
            self._jsonl,
            "JSONL export is unavailable",
            tuple(items),
            Path(export_path),
            timestamp,
        )

    def export_ascii(self, items, export_path: Path, timestamp: str):
        return self._run(
            self._ascii,
            "ASCII export is unavailable",
            tuple(items),
            Path(export_path),
            timestamp,
        )

    def export_array(self, request):
        return self._run(self._array, "Array export is unavailable", request)
