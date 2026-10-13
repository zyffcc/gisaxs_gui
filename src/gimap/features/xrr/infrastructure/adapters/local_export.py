"""CSV and JSON-record export adapters for extracted XRR curves."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ...application import XrrExtractionResult


class LocalXrrCurveExportAdapter:
    def export(self, path: Path, result: XrrExtractionResult) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "index",
                    "source",
                    "frame",
                    "theta_deg",
                    "qz_inv_angstrom",
                    "intensity",
                    "roi_center_x_px",
                    "roi_center_y_px",
                    "valid_pixels",
                )
            )
            for point in result.points:
                writer.writerow(
                    (
                        point.sequence_index,
                        point.source_name,
                        point.frame_index,
                        point.theta_deg,
                        point.qz_inv_angstrom,
                        "" if point.intensity is None else point.intensity,
                        point.roi_center_x_px,
                        point.roi_center_y_px,
                        point.valid_pixels,
                    )
                )


class LocalXrrRecordAdapter:
    """Writes the JSON settings record next to an exported curve."""

    def write(self, path: Path, record: dict[str, Any]) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = ["LocalXrrCurveExportAdapter", "LocalXrrRecordAdapter"]
