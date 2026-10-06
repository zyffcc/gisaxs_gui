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


_VIRIDIS_LUT: np.ndarray | None = None


def _viridis_lut() -> np.ndarray:
    """256 RGB colours of viridis (the colour map of the detector images elsewhere)."""
    global _VIRIDIS_LUT
    if _VIRIDIS_LUT is None:
        from matplotlib import colormaps

        colours = colormaps["viridis"](np.linspace(0.0, 1.0, 256))[:, :3]
        _VIRIDIS_LUT = np.ascontiguousarray(np.rint(colours * 255.0).astype(np.uint8))
    return _VIRIDIS_LUT


def display_levels(data: np.ndarray) -> np.ndarray:
    """Display only: log(1 + I) of the values clipped at 0, scaled to 0–1 between the 0.5th and
    99.9th percentiles; NaN and negative pixels (gaps, flagged) show as the lowest level."""
    array = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(array)
    logged = np.log1p(np.clip(np.where(finite, array, 0.0), 0.0, None))
    values = logged[finite]
    if not values.size:
        return np.zeros(array.shape, dtype=np.float32)
    low, high = np.percentile(values, (0.5, 99.9))
    if high <= low:
        high = low + 1.0
    return np.clip((logged - low) / (high - low), 0.0, 1.0)


def _array_pixmap(data: np.ndarray, width: int = 210, height: int = 155) -> QPixmap:
    """A thumbnail of a frame: log scale, viridis (the values themselves are never changed)."""
    array = np.asarray(data)
    if array.ndim > 2:
        array = array.reshape(array.shape[-2:]) if np.prod(array.shape[:-2]) == 1 else array[0]
    # About two image pixels per screen pixel is enough for a thumbnail.
    stride = max(1, int(np.ceil(max(array.shape) / (2.0 * max(width, height, 1)))))
    levels = display_levels(array[::stride, ::stride])
    rgb = np.ascontiguousarray(_viridis_lut()[np.rint(levels * 255.0).astype(np.uint8)])
    qimage = QImage(
        rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888
    ).copy()
    return QPixmap.fromImage(qimage).scaled(
        width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )
