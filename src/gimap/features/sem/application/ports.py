"""File access and figure export ports for SEM analysis."""

from pathlib import Path
from typing import Protocol

from ..domain import (
    SemComparison, SemCsvSettings, SemExtractionSettings, SemImage,
    SemInstanceSegmentation, SemMeasurements,
)


class SemImageReader(Protocol):
    def read(self, path: Path) -> SemImage: ...


class SemCsvReader(Protocol):
    def inspect(self, path: Path, skip_rows: int = 0, delimiter: str = ",") -> tuple[str, ...]: ...
    def read(self, path: Path, settings: SemCsvSettings) -> SemMeasurements: ...


class SemExporter(Protocol):
    def export(self, measurements: SemMeasurements, comparison: SemComparison | None,
               out_dir: Path) -> tuple[Path, ...]: ...


class SemInstanceSegmenter(Protocol):
    def defaults(self) -> dict: ...
    def status(self, settings: SemExtractionSettings) -> dict: ...
    def segment(self, image: SemImage, settings: SemExtractionSettings, *,
                cancelled=None) -> SemInstanceSegmentation: ...
