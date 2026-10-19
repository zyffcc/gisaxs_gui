"""SEM workflows remain usable independently of prediction or TensorFlow."""

from concurrent.futures import CancelledError
from pathlib import Path

from ..domain import (
    SemComparisonSettings, SemCsvSettings, SemExtractionSettings,
    compare_distributions, extract_particles, measure_labels,
)
from .ports import SemCsvReader, SemExporter, SemImageReader, SemInstanceSegmenter


class SemAnalysisService:
    def __init__(self, image_reader: SemImageReader, csv_reader: SemCsvReader, exporter: SemExporter,
                 instance_segmenter: SemInstanceSegmenter | None = None):
        self._image_reader = image_reader
        self._csv_reader = csv_reader
        self._exporter = exporter
        self._instance_segmenter = instance_segmenter

    def load_image(self, path):
        return self._image_reader.read(Path(path))

    def inspect_csv(self, path, skip_rows=0, delimiter=","):
        return self._csv_reader.inspect(Path(path), skip_rows=skip_rows, delimiter=delimiter)

    def instance_defaults(self):
        if self._instance_segmenter is None:
            return {"python_executable": "", "model_directory": ""}
        return dict(self._instance_segmenter.defaults())

    def instance_status(self, settings: SemExtractionSettings):
        if self._instance_segmenter is None:
            return {"available": False,
                    "message": "The selected SEM method requires an instance-segmentation adapter"}
        return dict(self._instance_segmenter.status(settings))

    def extract(self, image, settings: SemExtractionSettings, *, cancelled=None):
        if cancelled is not None and cancelled():
            raise CancelledError("SEM extraction cancelled")
        if settings.method == "threshold":
            return extract_particles(image, settings, cancelled=cancelled)
        if settings.method not in {"stardist", "labels"}:
            raise ValueError("SEM extraction method must be threshold, stardist or labels")
        if self._instance_segmenter is None:
            raise ValueError("The selected SEM method requires an instance-segmentation adapter")
        segmentation = self._instance_segmenter.segment(image, settings, cancelled=cancelled)
        return measure_labels(image, segmentation.labels, settings, segmentation.metadata, cancelled=cancelled)

    def import_csv(self, path, settings: SemCsvSettings):
        return self._csv_reader.read(Path(path), settings)

    def compare(self, measurements, prediction=None, options=None):
        return compare_distributions(measurements, prediction, options or SemComparisonSettings())

    def export(self, measurements, comparison, out_dir):
        return self._exporter.export(measurements, comparison, Path(out_dir))
