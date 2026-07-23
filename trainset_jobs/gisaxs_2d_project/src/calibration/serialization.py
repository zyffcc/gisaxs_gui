from __future__ import annotations

import json
from pathlib import Path

from .models import CalibrationResult


def save_calibration(result: CalibrationResult, path: str | Path) -> None:
    target = Path(path)
    target.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_calibration(path: str | Path) -> CalibrationResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "gimap-geometry-calibration":
        raise ValueError("This file is not a GIMaP geometry calibration.")
    if int(payload.get("format_version", 0)) != 1:
        raise ValueError("Unsupported geometry calibration file version.")
    return CalibrationResult.from_dict(payload)
