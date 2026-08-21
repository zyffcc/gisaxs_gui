"""Independent Image Rendering behavior."""

from __future__ import annotations

import time


import numpy as np

from PyQt5.QtCore import QTimer


from .scientific_commands import (
    _scientific_commands,
)
from .detector_render_lod import (
    choose_detector_render_level,
    detector_render_cell_budget,
)
from ..application import DetectorQGrid, normalize_horizontal_q_axis


class IndependentImageRenderingMixin:
    """Own independent image rendering behavior."""

    def update_image(self, image_data, vmin=None, vmax=None, use_log=True):
        """No description."""
        try:
            t_total_update = time.perf_counter()
            current_shape = image_data.shape
            shape_changed = self.last_image_shape is None or self.last_image_shape != current_shape
            show_q_axis = self._should_show_q_axis()
            render_level = None
            if show_q_axis:
                budget = detector_render_cell_budget(
                    self.canvas.width(),
                    self.canvas.height(),
                    minimum_cells=30_000,
                    maximum_cells=220_000,
                )
                render_level = choose_detector_render_level(
                    current_shape,
                    max_cells=budget,
                )
            render_step = render_level.stride if render_level is not None else 1

            self.current_image_shape = current_shape

            if shape_changed:
                self.current_xlim = None
                self.current_ylim = None
                self.last_image_shape = current_shape

            saved_xlim = self.current_xlim
            saved_ylim = self.current_ylim
            preserve_view = not shape_changed and saved_xlim is not None and saved_ylim is not None
            can_reuse_artist = (
                self.current_image is not None
                and not shape_changed
                and self._last_use_log == bool(use_log)
                and self._last_show_q_axis == show_q_axis
                and self._last_horizontal_q_axis == self._horizontal_q_axis()
                and self._last_q_cache_key == self._q_cache_key
                and self._last_render_step == render_step
            )
            if can_reuse_artist:
                t_total = time.perf_counter()
                if use_log:
                    t0 = time.perf_counter()
                    safe_data = np.where(
                        np.isfinite(image_data),
                        np.maximum(image_data, 0.001),
                        np.nan,
                    )
                    processed_data = np.log(safe_data, dtype=np.float32)
                    print(
                        f"[Timing] log transform: {(time.perf_counter() - t0) * 1000:.2f} ms (independent window)"
                    )
                else:
                    processed_data = image_data.astype(np.float32)
                if vmin is None or vmax is None:
                    t0 = time.perf_counter()
                    finite_values = processed_data[np.isfinite(processed_data)]
                    auto_vmin = np.percentile(finite_values, 1)
                    auto_vmax = np.percentile(finite_values, 99)
                    vmin = vmin if vmin is not None else auto_vmin
                    vmax = vmax if vmax is not None else auto_vmax
                    print(
                        f"[Timing] autoscale calculation: {(time.perf_counter() - t0) * 1000:.2f} ms (independent window)"
                    )
                processed_data = np.flipud(processed_data)
                if show_q_axis:
                    self.current_image.set_array(render_level.sample(processed_data).ravel())
                else:
                    self.current_image.set_data(processed_data)
                self.current_image.set_clim(vmin, vmax)
                self.current_image.set_cmap(self.colormap)
                if self.colorbar is not None:
                    try:
                        self.colorbar.update_normal(self.current_image)
                    except Exception:
                        pass
                if preserve_view:
                    self.ax.set_xlim(saved_xlim)
                    self.ax.set_ylim(saved_ylim)
                self._redraw_parameter_selection()
                render_start = time.perf_counter()
                self.canvas.draw()
                print(
                    f"[Timing] Matplotlib rendering: {(time.perf_counter() - render_start) * 1000:.2f} ms (independent window)"
                )
                print(
                    f"[Timing] independent window rendering: {(time.perf_counter() - t_total) * 1000:.2f} ms"
                )
                return

            try:
                xlim_cid = None
                ylim_cid = None

                try:
                    for cid, func in self.ax.callbacks.callbacks["xlim_changed"].items():
                        if func.func == self._on_xlim_changed:
                            xlim_cid = cid
                            break
                    for cid, func in self.ax.callbacks.callbacks["ylim_changed"].items():
                        if func.func == self._on_ylim_changed:
                            ylim_cid = cid
                            break

                    if xlim_cid is not None:
                        self.ax.callbacks.disconnect(xlim_cid)
                    if ylim_cid is not None:
                        self.ax.callbacks.disconnect(ylim_cid)

                except (AttributeError, KeyError):
                    try:
                        self.ax.callbacks.disconnect("xlim_changed", self._on_xlim_changed)
                        self.ax.callbacks.disconnect("ylim_changed", self._on_ylim_changed)
                    except TypeError:
                        pass

            except Exception:
                pass

            if self.colorbar is not None:
                try:
                    self.colorbar.remove()
                except Exception:
                    pass
                finally:
                    self.colorbar = None

            self.ax.clear()

            if use_log:
                t0 = time.perf_counter()
                safe_data = np.where(
                    np.isfinite(image_data),
                    np.maximum(image_data, 0.001),
                    np.nan,
                )
                processed_data = np.log(safe_data, dtype=np.float32)
                print(f"[Timing] log transform: {(time.perf_counter() - t0) * 1000:.2f} ms")
                scale_text = "Log Scale"
                colorbar_label = "Log Intensity"
            else:
                processed_data = image_data.astype(np.float32)
                scale_text = "Linear Scale"
                colorbar_label = "Intensity"

            if vmin is None or vmax is None:
                t0 = time.perf_counter()
                finite_values = processed_data[np.isfinite(processed_data)]
                auto_vmin = np.percentile(finite_values, 1)
                auto_vmax = np.percentile(finite_values, 99)
                vmin = vmin if vmin is not None else auto_vmin
                vmax = vmax if vmax is not None else auto_vmax
                print(
                    f"[Timing] autoscale calculation: {(time.perf_counter() - t0) * 1000:.2f} ms (independent window)"
                )

            processed_data = np.flipud(processed_data)

            if show_q_axis:
                self._get_q_axis_extent(image_data.shape)
                qy_mesh, qz_mesh = self._get_display_q_meshgrids()

                if qy_mesh is not None and qz_mesh is not None:
                    render_data = render_level.sample(processed_data)
                    render_qy = render_level.sample(qy_mesh)
                    render_qz = render_level.sample(qz_mesh)
                    self.current_image = self.ax.pcolormesh(
                        render_qy,
                        render_qz,
                        render_data,
                        cmap=self.colormap,
                        shading="nearest",
                        vmin=vmin,
                        vmax=vmax,
                        rasterized=True,
                    )
                else:
                    self.current_image = self.ax.imshow(
                        processed_data,
                        cmap=self.colormap,
                        aspect="equal",
                        origin="lower",
                        interpolation="nearest",
                        vmin=vmin,
                        vmax=vmax,
                    )

                self.ax.set_xlabel(self._horizontal_q_label())
                self.ax.set_ylabel(r"$q_z$ (nm$^{-1}$)")
            else:
                self.current_image = self.ax.imshow(
                    processed_data,
                    cmap=self.colormap,
                    aspect="equal",
                    origin="lower",
                    interpolation="nearest",
                    vmin=vmin,
                    vmax=vmax,
                )
                self.ax.set_xlabel("Pixels (Horizontal)")
                self.ax.set_ylabel("Pixels (Vertical)")

            coord_info = "Q-space" if show_q_axis else "Pixel coordinates"
            self.ax.set_title(
                f"GISAXS Image ({scale_text}) - {image_data.shape[1]} x {image_data.shape[0]} ({coord_info})\n"
                f"Vmin: {vmin:.3f}, Vmax: {vmax:.3f}"
            )

            if show_q_axis:
                self.ax.set_aspect("equal")
            else:
                self.ax.set_aspect("equal")

            try:
                self.colorbar = self.figure.colorbar(self.current_image, ax=self.ax)
                self.colorbar.set_label(colorbar_label)
            except Exception:
                self.colorbar = None

            self.figure.tight_layout()

            if preserve_view:
                self.ax.set_xlim(saved_xlim)
                self.ax.set_ylim(saved_ylim)
                self.current_xlim = saved_xlim
                self.current_ylim = saved_ylim
            else:
                if show_q_axis:
                    self.ax.autoscale()

                else:
                    self.ax.set_xlim(-0.5, processed_data.shape[1] - 0.5)
                    self.ax.set_ylim(-0.5, processed_data.shape[0] - 0.5)

                self.current_xlim = self.ax.get_xlim()
                self.current_ylim = self.ax.get_ylim()

            self._redraw_parameter_selection()

            try:
                self.ax.callbacks.connect("xlim_changed", self._on_xlim_changed)
                self.ax.callbacks.connect("ylim_changed", self._on_ylim_changed)
            except Exception:
                pass

            self._last_use_log = bool(use_log)
            self._last_show_q_axis = show_q_axis
            self._last_horizontal_q_axis = self._horizontal_q_axis()
            self._last_q_cache_key = self._q_cache_key
            self._last_render_step = render_step
            render_start = time.perf_counter()
            self.canvas.draw()
            print(
                f"[Timing] Matplotlib rendering: {(time.perf_counter() - render_start) * 1000:.2f} ms (independent window)"
            )
            print(
                f"[Timing] independent window rendering: {(time.perf_counter() - t_total_update) * 1000:.2f} ms"
            )

            if preserve_view:
                # 函数说明：实现 final 视图 check 相关逻辑。
                def final_view_check():
                    current_xlim_after_draw = self.ax.get_xlim()
                    current_ylim_after_draw = self.ax.get_ylim()
                    if (
                        abs(current_xlim_after_draw[0] - saved_xlim[0]) > 0.01
                        or abs(current_xlim_after_draw[1] - saved_xlim[1]) > 0.01
                        or abs(current_ylim_after_draw[0] - saved_ylim[0]) > 0.01
                        or abs(current_ylim_after_draw[1] - saved_ylim[1]) > 0.01
                    ):
                        self.ax.set_xlim(saved_xlim)
                        self.ax.set_ylim(saved_ylim)
                        self.current_xlim = saved_xlim
                        self.current_ylim = saved_ylim
                        self.canvas.draw_idle()

                QTimer.singleShot(50, final_view_check)

        except Exception as e:
            pass

    def _convert_q_to_pixel_coordinates(self, center_qy, center_qz, width_q, height_q):
        """Map a q point/region to the nearest detector cells."""
        try:
            grid = self._detector_q_grid()
            if grid is None:
                return {"center_x": 0, "center_y": 0, "width": 1, "height": 1}
            point = grid.nearest_point(
                center_qy,
                center_qz,
                self._horizontal_q_axis(),
            )
            region = grid.snap_region(
                center_qy - width_q / 2.0,
                center_qy + width_q / 2.0,
                center_qz - height_q / 2.0,
                center_qz + height_q / 2.0,
                self._horizontal_q_axis(),
            )
            return {
                "center_x": point.column,
                "center_y": grid.qz.shape[0] - 1 - point.row,
                "width": region.column_max - region.column_min + 1,
                "height": region.row_max - region.row_min + 1,
            }
        except Exception:
            return {"center_x": 0, "center_y": 0, "width": 1, "height": 1}

    def _update_cutline_labels_units(self):
        """No description."""
        try:
            show_q_axis = self._should_show_q_axis()

            if show_q_axis:
                horizontal_name = self._horizontal_q_axis()
                vertical_label = "qz center (nm^-1)"
                horizontal_label = f"{horizontal_name} center (nm^-1)"
                vertical_size_label = "qz span (nm^-1)"
                horizontal_size_label = f"{horizontal_name} span (nm^-1)"
            else:
                vertical_label = "Vertical (pixel)"
                horizontal_label = "Parallel (pixel)"
                vertical_size_label = "Vertical (pixel)"
                horizontal_size_label = "Parallel (pixel)"

            if hasattr(self.ui, "gisaxsInputCenterVerticalLabel"):
                self.ui.gisaxsInputCenterVerticalLabel.setText(vertical_label)

            if hasattr(self.ui, "gisaxsInputCenterParallelLabel"):
                self.ui.gisaxsInputCenterParallelLabel.setText(horizontal_label)

            if hasattr(self.ui, "gisaxsInputCutLineVerticalLabel"):
                self.ui.gisaxsInputCutLineVerticalLabel.setText(vertical_size_label)

            if hasattr(self.ui, "gisaxsInputCutLineParallelLabel"):
                self.ui.gisaxsInputCutLineParallelLabel.setText(horizontal_size_label)

        except Exception:
            pass

    def _should_show_q_axis(self):
        """Return the q-axis flag from the shared detector display state."""
        return bool(getattr(self, "show_q_axis", False))

    def _horizontal_q_axis(self):
        return normalize_horizontal_q_axis(getattr(self, "horizontal_q_axis", "qy"))

    def _horizontal_q_label(self):
        return r"$q_r$ (nm$^{-1}$)" if self._horizontal_q_axis() == "qr" else r"$q_y$ (nm$^{-1}$)"

    def _get_q_axis_extent(self, image_shape):
        """Return the display extent for the Q-space axes."""
        try:
            height, width = image_shape
            pixel_size_x = self.fitting_view_model.get_setting(
                "fitting", "detector.pixel_size_x", 172.0
            )  # micrometers
            pixel_size_y = self.fitting_view_model.get_setting(
                "fitting", "detector.pixel_size_y", 172.0
            )  # micrometers
            beam_center_x = self.fitting_view_model.get_setting(
                "fitting", "detector.beam_center_x", width / 2.0
            )
            beam_center_y = self.fitting_view_model.get_setting(
                "fitting", "detector.beam_center_y", height / 2.0
            )
            distance = self.fitting_view_model.get_setting(
                "fitting", "detector.distance", 2565.0
            )  # mm
            theta_in_deg = self.fitting_view_model.get_setting("beam", "grazing_angle", 0.4)
            wavelength = self.fitting_view_model.get_setting("beam", "wavelength", 0.1045)  # nm

            # Q-axis calculation parameters

            cache_key = (
                height,
                width,
                float(pixel_size_x),
                float(pixel_size_y),
                float(beam_center_x),
                float(beam_center_y),
                float(distance),
                float(theta_in_deg),
                float(wavelength),
            )

            grids_missing = any(
                grid is None for grid in (self._qy_mesh, self._qz_mesh, self._qr_mesh)
            )
            if self._q_cache_key != cache_key or grids_missing:
                self._q_detector = _scientific_commands(self).q_space.create_detector(
                    image_shape=(height, width),
                    pixel_size_x=pixel_size_x,
                    pixel_size_y=pixel_size_y,
                    beam_center_x=beam_center_x,
                    beam_center_y=beam_center_y,
                    distance=distance,
                    theta_in_deg=theta_in_deg,
                    wavelength=wavelength,
                    crop_params=None,
                )

                (
                    self._qy_mesh,
                    self._qz_mesh,
                    self._qr_mesh,
                ) = self._q_detector.get_q_coordinate_meshgrids()
                self._q_cache_key = cache_key

            horizontal_mesh, qz_mesh = self._get_cached_q_meshgrids()
            return [
                float(np.nanmin(horizontal_mesh)),
                float(np.nanmax(horizontal_mesh)),
                float(np.nanmin(qz_mesh)),
                float(np.nanmax(qz_mesh)),
            ]

        except Exception:
            height, width = image_shape
            return [-0.5, width - 0.5, -0.5, height - 0.5]

    def seed_q_grid_cache(self, cache_key, qy_mesh, qz_mesh, qr_mesh) -> None:
        """Reuse the owning fitting view's immutable full-resolution q grids."""

        if cache_key is None or any(grid is None for grid in (qy_mesh, qz_mesh, qr_mesh)):
            self._q_cache_key = None
            self._qy_mesh = None
            self._qz_mesh = None
            self._qr_mesh = None
            return
        DetectorQGrid(qy_mesh, qz_mesh, qr_mesh)
        self._q_cache_key = cache_key
        self._qy_mesh = qy_mesh
        self._qz_mesh = qz_mesh
        self._qr_mesh = qr_mesh

    def _get_cached_q_meshgrids(self):
        """Return active horizontal-q and qz grids in analysis-array order."""
        grid = self._detector_q_grid()
        return grid.meshes(self._horizontal_q_axis()) if grid is not None else (None, None)

    def _get_display_q_meshgrids(self):
        grid = self._detector_q_grid()
        return grid.display_meshes(self._horizontal_q_axis()) if grid is not None else (None, None)

    def _detector_q_grid(self):
        if self._qy_mesh is None or self._qz_mesh is None or self._qr_mesh is None:
            shape = getattr(self, "current_image_shape", None) or self.last_image_shape
            if shape is not None:
                self._get_q_axis_extent(shape)
        if self._qy_mesh is None or self._qz_mesh is None or self._qr_mesh is None:
            return None
        try:
            return DetectorQGrid(self._qy_mesh, self._qz_mesh, self._qr_mesh)
        except ValueError:
            return None
