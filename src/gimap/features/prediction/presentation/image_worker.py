"""旧 Qt controller 使用的 image loading bridge。"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal


class PredictionImageLoader(QThread):
    image_loaded = pyqtSignal(object, str)
    progress_updated = pyqtSignal(int, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, view_model, parent=None):
        super().__init__(parent)
        self._view_model = view_model
        self._path = None
        self._count = 1

    def load_image(self, file_path: str, stack_count: int = 1) -> None:
        self._path = Path(file_path)
        self._count = max(1, int(stack_count))
        self.start()

    def run(self) -> None:
        if self._path is None:
            self.error_occurred.emit("No prediction image path was provided")
            return
        self.progress_updated.emit(5, f"Loading {self._path.name}")
        loaded = self._view_model.load_stack(self._path, self._count)
        if loaded is None:
            self.error_occurred.emit(
                self._view_model.state.error_message or "Prediction image loading failed"
            )
            return
        self.progress_updated.emit(100, "Image loading completed")
        self.image_loaded.emit(loaded.image, str(self._path))
