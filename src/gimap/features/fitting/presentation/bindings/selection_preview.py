"""Selection Preview for fitting presentation."""

from __future__ import annotations

import time

import numpy as np


from PyQt5.QtWidgets import (
    QGraphicsScene,
)


from ..binding_primitives import (
    is_matplotlib_available,
)


class SelectionPreviewMixin:
    """Own selection preview behavior."""

    def _try_update_cached_preview(self, image_data, selection_info=None):
        try:
            if not is_matplotlib_available():
                return False
            t_total = time.perf_counter()
            graphics_view = self.ui.gisaxsInputGraphicsView
            processed_data, _ = self._prepare_image_data_for_display(image_data)
            processed_data = np.flipud(processed_data)
            preview_data, _ = self._downsample_for_preview(processed_data)
            show_q_axis = self._should_show_q_axis()
            extent, q_ok = self._preview_extent(image_data.shape, show_q_axis)
            if show_q_axis and not q_ok:
                show_q_axis = False
                extent, _ = self._preview_extent(image_data.shape, False)
            finite_values = processed_data[np.isfinite(processed_data)]
            if finite_values.size == 0:
                raise ValueError("No finite detector pixels remain after masking")
            vmin = self._current_vmin if self._current_vmin is not None else np.min(finite_values)
            vmax = self._current_vmax if self._current_vmax is not None else np.max(finite_values)
            shape_changed = self._preview_shape != image_data.shape
            needs_create = (
                self._figure_cache is None
                or self._canvas_cache is None
                or self._preview_ax is None
                or self._preview_image_artist is None
                or shape_changed
            )
            mode_changed = shape_changed or self._preview_show_q_axis != show_q_axis

            if needs_create:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

                img_height, img_width = image_data.shape
                aspect_ratio = img_width / img_height
                base_size = 6
                fig_width = max(base_size if aspect_ratio > 1 else base_size * aspect_ratio, 3)
                fig_height = max(base_size / aspect_ratio if aspect_ratio > 1 else base_size, 2.5)
                self._figure_cache = Figure(figsize=(fig_width, fig_height), dpi=72)
                self._canvas_cache = FigureCanvas(self._figure_cache)
                self._preview_ax = self._figure_cache.add_subplot(111)
                if self._graphics_scene is None:
                    self._graphics_scene = QGraphicsScene()
                    graphics_view.setScene(self._graphics_scene)
                else:
                    self._graphics_scene.clear()
                self._preview_proxy_widget = self._graphics_scene.addWidget(self._canvas_cache)

            ax = self._preview_ax
            if needs_create or mode_changed:
                ax.clear()
                self._preview_selection_artists = []
                self._preview_image_artist = ax.imshow(
                    preview_data,
                    cmap=self._image_colormap,
                    aspect="equal",
                    origin="lower",
                    interpolation="nearest",
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent,
                )
                if show_q_axis:
                    ax.set_xlabel(r"$q_y$ (nm$^{-1}$)")
                    ax.set_ylabel(r"$q_z$ (nm$^{-1}$)")
                    ax.autoscale()
                else:
                    ax.set_xlabel("Pixels (Horizontal)")
                    ax.set_ylabel("Pixels (Vertical)")
                    ax.axis("off")
                    ax.set_xlim(-0.5, image_data.shape[1] - 0.5)
                    ax.set_ylim(-0.5, image_data.shape[0] - 0.5)
                self._figure_cache.tight_layout(pad=0.05)
            else:
                self._preview_image_artist.set_data(preview_data)
                self._preview_image_artist.set_extent(extent)
                self._preview_image_artist.set_clim(vmin, vmax)
                self._preview_image_artist.set_cmap(self._image_colormap)

            self._draw_preview_selection(ax, selection_info)
            render_start = time.perf_counter()
            self._canvas_cache.draw()
            print(
                f"[Timing] Matplotlib rendering: {(time.perf_counter() - render_start) * 1000:.2f} ms (Detector Preview)"
            )
            if self._preview_proxy_widget is not None:
                self._fit_view_to_item(graphics_view, self._preview_proxy_widget, keep_aspect=True)
            self._preview_shape = image_data.shape
            self._preview_show_q_axis = show_q_axis
            print(f"[Timing] preview rendering: {(time.perf_counter() - t_total) * 1000:.2f} ms")
            return True
        except Exception as e:
            self.status_updated.emit(f"Preview cache update failed: {str(e)}")
            return False

    def _update_graphics_view_with_selection(self, image_data, selection_info=None):
        """GraphicsView"""
        try:
            self._expand_right_card("detectorPreviewCard")
            if not is_matplotlib_available():
                self.status_updated.emit("matplotlib not available for image display")
                return

            graphics_view = self.ui.gisaxsInputGraphicsView
            if self._try_update_cached_preview(image_data, selection_info):
                return

            if self._graphics_scene is None:
                self._graphics_scene = QGraphicsScene()
                graphics_view.setScene(self._graphics_scene)
            else:
                self._graphics_scene.clear()

            img_height, img_width = image_data.shape
            aspect_ratio = img_width / img_height

            base_size = 6
            if aspect_ratio > 1:
                fig_width = base_size
                fig_height = base_size / aspect_ratio
            else:
                fig_height = base_size
                fig_width = base_size * aspect_ratio

            fig_width = max(fig_width, 3)
            fig_height = max(fig_height, 2.5)

            try:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            except Exception:
                return
            fig = Figure(figsize=(fig_width, fig_height), dpi=72)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            processed_data, is_log = self._prepare_image_data_for_display(image_data)

            processed_data = np.flipud(processed_data)

            show_q_axis = self._should_show_q_axis()

            finite_values = processed_data[np.isfinite(processed_data)]
            if finite_values.size == 0:
                raise ValueError("No finite detector pixels remain after masking")
            vmin = self._current_vmin if self._current_vmin is not None else np.min(finite_values)
            vmax = self._current_vmax if self._current_vmax is not None else np.max(finite_values)

            if show_q_axis:
                try:
                    qy_mesh, qz_mesh = self._get_cached_q_meshgrids()

                    if qy_mesh is not None and qz_mesh is not None:
                        qy_min, qy_max = qy_mesh.min(), qy_mesh.max()
                        qz_min, qz_max = qz_mesh.min(), qz_mesh.max()
                        q_extent = [qy_min, qy_max, qz_min, qz_max]

                        im = ax.imshow(
                            processed_data,
                            cmap=self._image_colormap,
                            aspect="equal",
                            origin="lower",
                            interpolation="nearest",
                            vmin=vmin,
                            vmax=vmax,
                            extent=q_extent,
                        )

                        ax.set_xlabel(r"$q_y$ (nm$^{-1}$)")
                        ax.set_ylabel(r"$q_z$ (nm$^{-1}$)")
                    else:
                        show_q_axis = False
                except Exception as e:
                    pass
                    show_q_axis = False

            if not show_q_axis:
                im = ax.imshow(
                    processed_data,
                    cmap=self._image_colormap,
                    aspect="equal",
                    origin="lower",
                    interpolation="nearest",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_xlabel("Pixels (Horizontal)")
                ax.set_ylabel("Pixels (Vertical)")

            if selection_info:
                bounds = selection_info.get("bounds", {})
                x_min = bounds.get("x_min", 0)
                x_max = bounds.get("x_max", 0)
                y_min = bounds.get("y_min", 0)
                y_max = bounds.get("y_max", 0)

                if self._show_cut_region:
                    from matplotlib.patches import Rectangle

                    selection_rect = Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                        alpha=0.8,
                    )
                    ax.add_patch(selection_rect)

            self._draw_detector_center_on_axis(ax)

            if not show_q_axis:
                ax.axis("off")

            fig.tight_layout(pad=0.05)

            if show_q_axis:
                ax.autoscale()
            else:
                ax.set_xlim(-0.5, processed_data.shape[1] - 0.5)
                ax.set_ylim(-0.5, processed_data.shape[0] - 0.5)

            canvas.draw()

            proxy_widget = self._graphics_scene.addWidget(canvas)

            self._fit_view_to_item(graphics_view, proxy_widget, keep_aspect=True)

            mode_text = "Log" if self._is_log_mode_enabled() else "Linear"
            coord_mode = "Q-space" if show_q_axis else "Pixel coordinates"
            selection_text = " with selection" if selection_info else ""
            self.status_updated.emit(
                f"{mode_text} image displayed ({coord_mode}){selection_text} (Double-click to open independent window)"
            )

        except Exception as e:
            self.status_updated.emit(f"Display error: {str(e)}")
