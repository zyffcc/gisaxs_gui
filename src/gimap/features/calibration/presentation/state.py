"""Calibration 可持久化的轻量 UI state。"""

from dataclasses import dataclass


@dataclass
class CalibrationState:
    last_image_path: str = ""
    last_result_source: str = ""

    def snapshot(self) -> dict:
        return {
            "last_image_path": self.last_image_path,
            "last_result_source": self.last_result_source,
        }

    def restore(self, state: dict) -> None:
        self.last_image_path = str(state.get("last_image_path", ""))
        self.last_result_source = str(state.get("last_result_source", ""))
