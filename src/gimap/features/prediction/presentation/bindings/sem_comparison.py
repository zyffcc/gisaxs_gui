"""Hand an immutable, source-labelled radius prediction to the application shell."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np


class SemComparisonMixin:
    def _capture_sem_prediction(self, outputs) -> None:
        self._sem_snapshot = None
        self._sem_snapshot_context = None
        image = getattr(self, "_current_image", None)
        module = self._typed_prediction_module()
        axes = outputs.get("distribution_axes", {})
        row = axes.get("row", {}) if isinstance(axes, dict) else {}
        if module is None or image is None or row.get("key") != "r" or row.get("unit") != "nm":
            return
        radius = outputs.get("r_x")
        probability = outputs.get("r")
        if radius is None or probability is None:
            return
        radius = np.array(radius, dtype=float, copy=True).reshape(-1)
        probability = np.array(probability, dtype=float, copy=True).reshape(-1)
        if (radius.size < 2 or radius.shape != probability.shape
                or not np.all(np.isfinite(radius)) or not np.all(np.isfinite(probability))
                or np.any(radius < 0) or np.any(np.diff(radius) <= 0)
                or np.any(probability < 0) or probability.sum() <= 0):
            return
        model_path = str(self.current_parameters.get("module_model_path") or "")
        files = list(getattr(self, "_current_input_files", ()) or ())
        if not files and getattr(self, "_current_image_path", None):
            files = [self._current_image_path]
        runtime = getattr(self, "_latest_runtime", None)
        self._sem_snapshot = {
            "radius_nm": radius,
            "probability": probability,
            "source": ", ".join(Path(path).name for path in files),
            "metadata": {
                "input_files": files,
                "stack": int(getattr(self, "_current_input_stack", 1) or 1),
                "module_id": module.id,
                "module_name": module.name,
                "module_file": str(module.yaml_path or ""),
                "module_version": module.version,
                "model_path": model_path,
                "radius_axis": deepcopy(row),
                "preprocessing_steps": list(module.preprocess.steps),
                "preprocessing_settings": deepcopy(dict(module.preprocess.params)),
                "runtime_name": str(getattr(runtime, "runtime_name", "")),
                "runtime_version": str(getattr(runtime, "runtime_version", "")),
            },
        }
        self._sem_snapshot_context = (image, deepcopy(module), model_path, outputs)

    def _sem_prediction_for_comparison(self):
        context = getattr(self, "_sem_snapshot_context", None)
        snapshot = getattr(self, "_sem_snapshot", None)
        if context is None or snapshot is None:
            return None
        image, module, model_path, outputs = context
        try:
            valid = (
                image is self._current_image
                and bool(module == self._typed_prediction_module())
                and model_path == str(self.current_parameters.get("module_model_path") or "")
                and outputs is self.prediction_results
            )
        except (TypeError, ValueError, AttributeError):
            valid = False
        return deepcopy(snapshot) if valid else None

    def _on_sem_comparison_clicked(self) -> None:
        self.sem_comparison_requested.emit(self._sem_prediction_for_comparison())
