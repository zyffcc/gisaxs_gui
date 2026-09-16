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
from matplotlib.transforms import Bbox


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
        self._image_artists = []
        self._mesh_coordinates = None
        self._overlay_artists = []
        self._placeholder_artist = None
        self._placeholder()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toolbar, 0)
        layout.addWidget(self.canvas, 1)

    def _placeholder(self) -> None:
        self._ensure_image_axes()
        self.clear_overlays()
        for artist in self._image_artists:
            artist.set_visible(False)
        if self._placeholder_artist is not None:
            self._placeholder_artist.remove()
        self.cax.set_axis_off()
        self._placeholder_artist = self.ax.text(
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
        q_coordinates: tuple[np.ndarray, np.ndarray] | None = None,
        no_data_color: str = "white",
    ) -> None:
        render_start = time.perf_counter()
        raw = np.asarray(image)
        preview_source, preview_extent, stride = self._preview_image(raw, extent)
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
        self.clear_overlays()
        if self._placeholder_artist is not None:
            self._placeholder_artist.remove()
            self._placeholder_artist = None
        self.ax.set_axis_on()
        self.cax.set_axis_on()
        cmap = colormaps.get_cmap(colormap).copy()
        # Keep masked/no-data detector cells visually distinct from measured
        # intensities instead of extending the low end of the colormap into them.
        cmap.set_bad(no_data_color)
        self.ax.set_facecolor(no_data_color)
        mesh_coordinates = None
        if q_coordinates is not None:
            horizontal_q, qz = q_coordinates
            horizontal_preview = np.asarray(horizontal_q)[::stride, ::stride]
            qz_preview = np.asarray(qz)[::stride, ::stride]
            if flip_vertical:
                horizontal_preview = np.flipud(horizontal_preview)
                qz_preview = np.flipud(qz_preview)
            mesh_coordinates = (horizontal_preview, qz_preview)
        same_geometry = (
            (mesh_coordinates is None and self._mesh_coordinates is None)
            or (mesh_coordinates is not None and self._mesh_coordinates is not None
                and all(np.array_equal(a, b, equal_nan=True)
                        for a, b in zip(mesh_coordinates, self._mesh_coordinates)))
        )
        if not same_geometry:
            for artist in self._image_artists:
                artist.remove()
            self._image_artists = []
            self.ax.dataLim.set(Bbox.null())
        if mesh_coordinates is None:
            if not self._image_artists:
                self.ax.set_autoscale_on(True)
                self._image_artists = [self.ax.imshow(
                    preview, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                    aspect="equal", extent=preview_extent,
                )]
            else:
                artist = self._image_artists[0]
                old_extent = tuple(artist.get_extent())
                artist.set_data(preview)
                if old_extent != tuple(preview_extent):
                    artist.set_extent(preview_extent)
                    self.ax.set_xlim(preview_extent[:2])
                    self.ax.set_ylim(preview_extent[2:])
        else:
            horizontal_preview, qz_preview = mesh_coordinates
            branches = self._signed_q_branch_slices(horizontal_preview)
            if not self._image_artists:
                self.ax.set_autoscale_on(True)
                for branch in branches:
                    self._image_artists.append(self.ax.pcolormesh(
                        horizontal_preview[:, branch], qz_preview[:, branch],
                        preview[:, branch], shading="nearest", cmap=cmap,
                        vmin=vmin, vmax=vmax, rasterized=True,
                    ))
                self.ax.autoscale_view()
                self.ax.set_xlim(sorted(self.ax.get_xlim()))
                self.ax.set_ylim(sorted(self.ax.get_ylim()))
            else:
                for artist, branch in zip(self._image_artists, branches):
                    artist.set_array(preview[:, branch].ravel())
        if not same_geometry:
            self._mesh_coordinates = (
                tuple(a.copy() for a in mesh_coordinates) if mesh_coordinates is not None else None
            )
        for artist in self._image_artists:
            artist.set_visible(True)
            with artist.callbacks.blocked():
                artist.set_cmap(cmap)
                artist.set_clim(vmin, vmax)
        artist = self._image_artists[0]
        self.ax.set_autoscale_on(False)
        self.ax.set_aspect("equal", adjustable="box", anchor="C")
        self.ax.set_anchor("C")
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if self.colorbar is None:
            self.colorbar = self.figure.colorbar(artist, cax=self.cax)
        else:
            previous = self.colorbar.mappable
            if previous is not artist:
                previous.callbacks.disconnect(previous.colorbar_cid)
                previous.colorbar = None
                previous.colorbar_cid = None
                artist.colorbar = self.colorbar
                artist.colorbar_cid = artist.callbacks.connect("changed", self.colorbar.update_normal)
            self.colorbar.update_normal(artist)
        self.canvas.draw_idle()
        window = getattr(self, "interactive_window", None)
        if window is not None and window.isVisible():
            if q_coordinates is not None:
                window.set_unavailable("Q-space is available in the main detector view. Switch to Pixel to inspect here.")
            else:
                window.set_frame(
                    preview, intensity=np.flipud(raw) if flip_vertical else raw,
                    extent=preview_extent, origin="upper", levels=(vmin, vmax),
                    colormap=colormap, log_scale=log_scale, no_data_color=no_data_color,
                )
        render_time = time.perf_counter() - render_start
        self._log_display_debug(raw, preview, limits_time, render_time)

    def clear_overlays(self) -> None:
        """Remove only registered decorations, leaving selectors and image artists alive."""
        for artist in self._overlay_artists:
            if artist.axes is not None:
                artist.remove()
        self._overlay_artists.clear()

    def _reset_image_axes(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_axes([0.09, 0.14, 0.75, 0.76])
        self.cax = self.figure.add_axes([0.88, 0.14, 0.025, 0.76])
        self.colorbar = None
        self._image_artists = []
        self._mesh_coordinates = None
        self._overlay_artists = []
        self._placeholder_artist = None

    def _ensure_image_axes(self) -> None:
        if self.ax is None or self.cax is None:
            self._reset_image_axes()

    def _preview_image(
        self,
        image: np.ndarray,
        extent: tuple[float, float, float, float] | None,
    ) -> tuple[np.ndarray, tuple[float, float, float, float] | None, int]:
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
        preview = image[::stride, ::stride].copy()
        preview = self._detector_coverage_preview(image, preview, stride)
        if extent is None:
            preview_extent = (0.0, float(width), float(height), 0.0)
        else:
            preview_extent = extent
        self._preview_cache_key = (id(image), image.shape, str(image.dtype), stride, extent)
        self._preview_cache_array = preview
        self._preview_cache_extent = preview_extent
        return preview, preview_extent, stride

    @classmethod
    def _detector_coverage_preview(
        cls,
        image: np.ndarray,
        sampled: np.ndarray,
        stride: int,
    ) -> np.ndarray:
        """Interpolate interior NaNs while retaining unsupported detector area."""
        height, width = image.shape[:2]
        finite = np.isfinite(image)
        pad_y = (-height) % stride
        pad_x = (-width) % stride
        padded = np.pad(finite, ((0, pad_y), (0, pad_x)), constant_values=False)
        sampled_support = padded.reshape(
            padded.shape[0] // stride,
            stride,
            padded.shape[1] // stride,
            stride,
        ).any(axis=(1, 3))

        # A point belongs to the detector image only when measured pixels bracket
        # it in both directions. This preserves the pixel-space blank canvas and,
        # after coordinate projection, the high-q central coverage hole.
        row_envelope = np.maximum.accumulate(
            sampled_support, axis=1
        ) & np.maximum.accumulate(sampled_support[:, ::-1], axis=1)[:, ::-1]
        column_envelope = np.maximum.accumulate(
            sampled_support, axis=0
        ) & np.maximum.accumulate(sampled_support[::-1, :], axis=0)[::-1, :]
        detector_coverage = row_envelope & column_envelope

        result = np.asarray(sampled, dtype=float).copy()
        cls._interpolate_nan_lines(result, axis=1)
        cls._interpolate_nan_lines(result, axis=0)
        result[~detector_coverage] = np.nan
        return result

    @staticmethod
    def _interpolate_nan_lines(values: np.ndarray, *, axis: int) -> None:
        oriented = values if axis == 1 else values.T
        positions = np.arange(oriented.shape[1])
        for line in oriented:
            finite = np.isfinite(line)
            if np.count_nonzero(finite) >= 2:
                missing = ~finite
                line[missing] = np.interp(positions[missing], positions[finite], line[finite])

    @staticmethod
    def _signed_q_branch_slices(horizontal_q: np.ndarray) -> tuple[slice, ...]:
        """Split negative/positive signed-Qr branches at their discontinuity."""
        columns = np.nanmedian(np.asarray(horizontal_q, dtype=float), axis=0)
        negative = np.flatnonzero(columns < 0.0)
        positive = np.flatnonzero(columns > 0.0)
        branches: list[slice] = []
        if negative.size >= 2:
            branches.append(slice(int(negative[0]), int(negative[-1]) + 1))
        if positive.size >= 2:
            branches.append(slice(int(positive[0]), int(positive[-1]) + 1))
        if branches:
            return tuple(branches)
        return (slice(0, horizontal_q.shape[1]),)

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
            f"display_limits={limits_time:.3f}s; update_preview={render_time:.3f}s (paint deferred)"
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
