"""Matplotlib image viewer for the WAXS workspace."""

from __future__ import annotations

import time


from pathlib import Path

from typing import Optional

import numpy as np

from matplotlib import colormaps

from PyQt5.QtCore import pyqtSignal

from PyQt5.QtWidgets import (
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure


from .file_types import SUPPORTED_EXTENSIONS


class ScatteringImageViewer(QWidget):
    fileDropped = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None, *, view_model=None):
        super().__init__(parent)
        if view_model is None:
            raise ValueError("ScatteringImageViewer requires WaxsViewModel")
        self.view_model = view_model
        self.setAcceptDrops(True)
        self.figure = Figure(figsize=(6, 5), constrained_layout=False)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = None
        self.cax = None
        self.colorbar = None
        self._preview_cache_key = None
        self._preview_cache_array: Optional[np.ndarray] = None
        self._preview_cache_extent: tuple[float, float, float, float] | None = None
        self._reset_image_axes()
        self._placeholder()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toolbar, 0)
        layout.addWidget(self.canvas, 1)

    def _placeholder(self) -> None:
        self._reset_image_axes()
        self.ax.clear()
        self.cax.clear()
        self.cax.set_axis_off()
        self.ax.text(
            0.5,
            0.5,
            "Open a .nxs, .tif, or .tiff file",
            ha="center",
            va="center",
            color="#64748b",
            transform=self.ax.transAxes,
        )
        self.ax.set_axis_off()

    def show_image(
        self,
        image: np.ndarray,
        *,
        log_scale: bool,
        colormap: str,
        auto_scale: bool,
        vmin: float,
        vmax: float,
        mask_min: float,
        mask_max: float,
        flip_vertical: bool,
        title: str,
        extent: tuple[float, float, float, float] | None = None,
        xlabel: str = "X (pixel)",
        ylabel: str = "Y (pixel)",
    ) -> None:
        render_start = time.perf_counter()
        raw = np.asarray(image)
        preview_source, preview_extent = self._preview_image(raw, extent)
        preview = self.view_model.prepare_display(
            preview_source,
            log_scale=log_scale,
            mask_min=mask_min,
            mask_max=mask_max,
            flip_vertical=flip_vertical,
        )
        preview = np.ascontiguousarray(preview)
        if auto_scale:
            limits_start = time.perf_counter()
            limits = self.view_model.estimate_display_limits(
                raw,
                log_scale=log_scale,
                mask_min=mask_min,
                mask_max=mask_max,
            )
            limits_time = time.perf_counter() - limits_start
            if limits is not None:
                vmin, vmax = limits
        else:
            limits_time = 0.0

        self._ensure_image_axes()
        self.ax.clear()
        self.cax.clear()
        cmap = colormaps.get_cmap(colormap).copy()
        cmap.set_bad(cmap(0.0))
        artist = self.ax.imshow(
            preview,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            extent=preview_extent,
        )
        self.ax.set_aspect("equal", adjustable="box", anchor="C")
        self.ax.set_anchor("C")
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.colorbar = self.figure.colorbar(artist, cax=self.cax)
        render_time = time.perf_counter() - render_start
        self._log_display_debug(raw, preview, limits_time, render_time)

    def _reset_image_axes(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_axes([0.07, 0.08, 0.78, 0.86])
        self.cax = self.figure.add_axes([0.88, 0.08, 0.025, 0.86])
        self.colorbar = None

    def _ensure_image_axes(self) -> None:
        if self.ax is None or self.cax is None:
            self._reset_image_axes()

    def _preview_image(
        self,
        image: np.ndarray,
        extent: tuple[float, float, float, float] | None,
    ) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
        height, width = image.shape[:2]
        canvas_w = max(64, int(self.canvas.width()))
        canvas_h = max(64, int(self.canvas.height()))
        max_preview_pixels = 1_000_000
        stride = max(
            1,
            int(np.ceil(width / max(1, canvas_w))),
            int(np.ceil(height / max(1, canvas_h))),
            int(np.ceil(np.sqrt(max(1, image.size) / max_preview_pixels))),
        )
        preview = image[::stride, ::stride]
        if extent is None:
            preview_extent = (0.0, float(width), float(height), 0.0)
        else:
            preview_extent = extent
        self._preview_cache_key = (id(image), image.shape, str(image.dtype), stride, extent)
        self._preview_cache_array = preview
        self._preview_cache_extent = preview_extent
        return preview, preview_extent

    @staticmethod
    @staticmethod
    def _array_mb(arr: np.ndarray) -> float:
        return float(np.asarray(arr).nbytes) / (1024.0 * 1024.0)

    def _log_display_debug(
        self,
        raw: np.ndarray,
        preview: np.ndarray,
        limits_time: float,
        render_time: float,
    ) -> None:
        print(
            "[WAXS display] "
            f"raw shape={raw.shape} dtype={raw.dtype} MB={self._array_mb(raw):.2f}; "
            f"preview shape={preview.shape} dtype={preview.dtype} MB={self._array_mb(preview):.2f}; "
            f"display_limits={limits_time:.3f}s; render_preview={render_time:.3f}s"
        )

    def display_limits(
        self,
        image: np.ndarray,
        *,
        log_scale: bool,
        mask_min: float,
        mask_max: float,
        flip_vertical: bool,
    ) -> tuple[float, float] | None:
        del flip_vertical
        limits_start = time.perf_counter()
        limits = self.view_model.estimate_display_limits(
            image,
            log_scale=log_scale,
            mask_min=mask_min,
            mask_max=mask_max,
        )
        print(f"[WAXS display] display_limits={time.perf_counter() - limits_start:.3f}s")
        return limits

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
            if Path(path).suffix.lower() in SUPPORTED_EXTENSIONS:
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event) -> None:
        path = event.mimeData().urls()[0].toLocalFile()
        if path:
            self.fileDropped.emit(self.view_model.normalize_path(path))
