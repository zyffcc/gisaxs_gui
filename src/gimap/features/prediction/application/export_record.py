"""The JSON record written beside a 2D Prediction export: what produced the exported figure and data.

The record names the files written, the input frame(s) and stack, the prediction module and model, the
framework and runtime, and the preprocessing steps the model input went through (their settings, not
their intermediate images), so an exported prediction can be traced back and reproduced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .ports import PredictionExportRepository

RECORD_SCHEMA = "gimap.prediction-export/1"
_SMALL_ARRAY = 16  # at most this many values are written out; a larger array is recorded by its shape


@dataclass(frozen=True)
class PredictionRecordRequest:
    path: Path
    """The record file (``<stem>.json``, beside the exported image)."""
    outputs: tuple[Path, ...] = ()
    """The files this export wrote (image, data)."""
    kind: str = "prediction"
    """``prediction`` (the model's result) or ``input preview`` (the detector frame as displayed)."""
    shown: str = ""
    """The result view that was exported (``hr``, ``curve``, ``steps``, …)."""
    mode: str = "single_file"
    input_files: tuple[str, ...] = ()
    stack: int = 1
    module: Mapping[str, Any] = field(default_factory=dict)
    """Prediction module: name, id, version, file."""
    model_path: str = ""
    framework: str = ""
    runtime_name: str = ""
    runtime_version: str = ""
    preprocess_entry: str = ""
    preprocess_steps: tuple[str, ...] = ()
    """The steps the module configures."""
    applied_steps: tuple[Mapping[str, Any], ...] = ()
    """The steps of the run that produced the result, with their settings (intermediate images left out)."""
    display: Mapping[str, Any] = field(default_factory=dict)
    """Display settings of an exported view (colour map, limits, log scale): not applied to the data."""


def _plain(value: Any) -> Any:
    """``value`` as JSON: numbers, text, lists and mappings; a large array by its shape and type."""
    if isinstance(value, np.ndarray):
        if value.size <= _SMALL_ARRAY:
            return [_plain(item) for item in value.tolist()]
        return {"array_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def applied_step_record(step: Mapping[str, Any]) -> dict[str, Any]:
    """One preprocessing step of a run: its name and settings; the snapshot image only by its shape."""
    record = {}
    for key, value in step.items():
        if key == "image" and isinstance(value, np.ndarray):
            record["image_shape"] = list(value.shape)
            continue
        record[str(key)] = _plain(value)
    return record


class ExportPredictionRecord:
    """Write the JSON record of one Prediction export through the export repository."""

    def __init__(self, repository: PredictionExportRepository):
        self._repository = repository

    @staticmethod
    def record(request: PredictionRecordRequest) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema": RECORD_SCHEMA,
            "written": datetime.now().isoformat(timespec="seconds"),
            "kind": request.kind,
            "outputs": [Path(path).name for path in request.outputs],
            "input": {
                "mode": request.mode,
                "files": [str(path) for path in request.input_files],
                "stack": int(request.stack),
            },
        }
        if request.kind == "prediction":
            record["shown"] = request.shown
            record["module"] = _plain(dict(request.module))
            record["model"] = {
                "path": request.model_path,
                "framework": request.framework,
                "runtime": request.runtime_name,
                "runtime_version": request.runtime_version,
            }
            record["preprocessing"] = {
                "entry": request.preprocess_entry,
                "steps": list(request.preprocess_steps),
                "applied": [applied_step_record(step) for step in request.applied_steps],
            }
        if request.display:
            record["display"] = _plain(dict(request.display))
        return record

    def execute(self, request: PredictionRecordRequest) -> Path:
        text = json.dumps(self.record(request), indent=2, ensure_ascii=False)
        return self._repository.write_text(Path(request.path), text + "\n")


__all__ = ["ExportPredictionRecord", "PredictionRecordRequest", "RECORD_SCHEMA", "applied_step_record"]
