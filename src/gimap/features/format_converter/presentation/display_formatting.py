"""Display Formatting for Format Converter."""

from __future__ import annotations


import numpy as np

from PyQt5.QtCore import Qt

from PyQt5.QtGui import QImage, QPixmap

INPUT_FILTER = (
    "Detector images (*.nxs *.cbf *.tif *.tiff);;NXS (*.nxs);;CBF (*.cbf);;TIFF (*.tif *.tiff)"
)


def _human_bytes(value: int) -> str:
    number = float(max(0, value))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if number < 1024.0 or unit == "TB":
            return f"{number:.1f} {unit}" if unit != "B" else f"{int(number)} B"
        number /= 1024.0
    return f"{number:.1f} TB"


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _array_pixmap(data: np.ndarray, width: int = 210, height: int = 155) -> QPixmap:
    array = np.asarray(data, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size:
        low, high = np.percentile(finite, (1, 99))
        if high <= low:
            high = low + 1.0
        image = np.clip((np.nan_to_num(array, nan=low) - low) / (high - low), 0, 1)
    else:
        image = np.zeros(array.shape, dtype=np.float32)
    gray = np.ascontiguousarray(np.rint(image * 255).astype(np.uint8))
    qimage = QImage(
        gray.data, gray.shape[1], gray.shape[0], gray.strides[0], QImage.Format_Grayscale8
    ).copy()
    return QPixmap.fromImage(qimage).scaled(
        width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )
