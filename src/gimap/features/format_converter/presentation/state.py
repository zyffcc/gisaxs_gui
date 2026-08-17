"""Format Converter 可持久化的轻量 UI state。"""

from dataclasses import dataclass


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
