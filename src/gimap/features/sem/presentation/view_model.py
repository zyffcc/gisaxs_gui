"""SEM commands and UI state; worker commands never commit mutable UI state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event

import numpy as np

from ..application import SemPrediction


@dataclass
class SemViewState:
    source_path: str = ""
    source_kind: str = ""
    image: object = None
    measurements: object = None
    comparison: object = None
    prediction: object = None
    columns: tuple[str, ...] = ()
    running: bool = False
    error_message: str = ""


class SemViewModel:
    def __init__(self, service):
        self.service = service
        self.state = SemViewState()
        self._cancel_event = None

    def set_prediction(self, radius_nm, probability=None, source="", metadata=None):
        """Freeze the current result so later prediction edits cannot alter a comparison."""
        if radius_nm is None:
            self.state.prediction = None
        else:
            from copy import deepcopy

            self.state.prediction = SemPrediction(
                radius_nm=np.array(radius_nm, dtype=float, copy=True),
                probability=np.array(probability, dtype=float, copy=True),
                source=str(source), metadata=deepcopy(metadata or {}),
            )
        self.state.comparison = None

    def select_source(self, path, kind):
        self.state.source_path = str(Path(path))
        self.state.source_kind = kind
        self.state.image = None
        self.state.measurements = None
        self.state.comparison = None
        self.state.columns = ()
        self.state.error_message = ""

    def invalidate_measurements(self):
        self.state.measurements = None
        self.state.comparison = None

    def load_image(self, path):
        return self.service.load_image(path)

    def inspect_csv(self, path, *, skip_rows=0, delimiter=","):
        return self.service.inspect_csv(path, skip_rows=skip_rows, delimiter=delimiter)

    def instance_defaults(self):
        defaults = {"python_executable": "", "model_directory": "", "available": False,
                    "message": "Local instance detection is unavailable."}
        callback = getattr(self.service, "instance_defaults", None)
        if callback is not None:
            try:
                defaults.update(dict(callback()))
            except Exception as exc:
                defaults["message"] = str(exc)
        return defaults

    def instance_status(self, settings):
        callback = getattr(self.service, "instance_status", None)
        if callback is None:
            return {"available": False, "message": "This SEM service does not support local instance detection."}
        try:
            return dict(callback(settings))
        except Exception as exc:
            return {"available": False, "message": str(exc)}

    def extraction_command(self, settings, options):
        import inspect

        self.cancel()
        event = Event()
        self._cancel_event = event
        image = self.state.image
        prediction = self.state.prediction
        service = self.service
        signature = inspect.signature(service.extract)
        supports_token = "cancelled" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )

        def run():
            kwargs = {"cancelled": event.is_set} if supports_token else {}
            measurements = service.extract(image, settings, **kwargs)
            comparison = service.compare(measurements, prediction, options) if len(measurements.radius_nm) else None
            return measurements, comparison

        return run

    def cancel(self):
        if self._cancel_event is not None:
            self._cancel_event.set()

    def csv_command(self, settings, options):
        source = self.state.source_path
        prediction = self.state.prediction
        service = self.service

        def run():
            measurements = service.import_csv(source, settings)
            comparison = service.compare(measurements, prediction, options) if len(measurements.radius_nm) else None
            return measurements, comparison

        return run

    def comparison_command(self, options):
        measurements, prediction = self.state.measurements, self.state.prediction
        service = self.service
        return lambda: service.compare(measurements, prediction, options)

    def export_command(self, out_directory):
        measurements, comparison = self.state.measurements, self.state.comparison
        service = self.service
        return lambda: service.export(measurements, comparison, Path(out_directory))


__all__ = ["SemViewModel", "SemViewState"]
