"""Format Converter 可持久化的轻量 UI state。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OutputPreviewState:
    example: str
    image_count: int
    file_count: int
    estimated_bytes: int
    dtype_warning: str


@dataclass(frozen=True)
class ConversionReviewState:
    input_summary: str
    image_count: int
    output_files: int
    estimated_bytes: int
    destination: str
    naming: str
    is_large_output: bool


@dataclass
class FormatConverterState:
    destination: str = ""
    output_format: str = "TIFF"

    def snapshot(self) -> dict:
        return {
            "destination": self.destination,
            "output_format": self.output_format,
        }

    def restore(self, state: dict) -> None:
        self.destination = str(state.get("destination", ""))
        self.output_format = str(state.get("output_format", "TIFF"))
