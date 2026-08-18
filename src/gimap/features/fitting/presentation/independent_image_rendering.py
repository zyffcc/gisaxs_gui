"""Independent Image Rendering behavior."""

from __future__ import annotations

import time


import numpy as np

from PyQt5.QtCore import QTimer


from .scientific_commands import (
    _scientific_commands,
)


class IndependentImageRenderingMixin:
    """Own independent image rendering behavior."""

    def update_image(self, image_data, vmin=None, vmax=None, use_log=True):
        """No description."""
        try:
            t_total_update = time.perf_counter()
            current_shape = image_data.shape
            shape_changed = self.last_image_shape is None or self.last_image_shape != current_shape

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
                and self._last_show_q_axis == self._should_show_q_axis()
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

            show_q_axis = self._should_show_q_axis()

            if show_q_axis:
                extent = self._get_q_axis_extent(image_data.shape)

                qy_mesh, qz_mesh = self._get_cached_q_meshgrids()

                if qy_mesh is not None and qz_mesh is not None:
                    qy_min, qy_max = qy_mesh.min(), qy_mesh.max()
                    qz_min, qz_max = qz_mesh.min(), qz_mesh.max()
                    q_extent = [qy_min, qy_max, qz_min, qz_max]

                    self.current_image = self.ax.imshow(
                        processed_data,
                        cmap=self.colormap,
                        aspect="equal",
                        origin="lower",
                        interpolation="nearest",
                        vmin=vmin,
                        vmax=vmax,
                        extent=q_extent,
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
                        extent=extent,
                    )

                self.ax.set_xlabel(r"$q_y$ (nm$^{-1}$)")
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
        """No description."""
        try:
            qy_mesh, qz_mesh = self._get_cached_q_meshgrids()

            if qy_mesh is None or qz_mesh is None:
                return {"center_x": 0, "center_y": 0, "width": 100, "height": 100}

            if hasattr(self, "current_stack_data") and self.current_stack_data is not None:
                img_height, img_width = self.current_stack_data.shape
            else:
                img_height, img_width = qy_mesh.shape

            qy_diff = np.abs(qy_mesh - center_qy)
            qz_diff = np.abs(qz_mesh - center_qz)
            combined_diff = qy_diff + qz_diff
            center_idx = np.unravel_index(np.argmin(combined_diff), qy_mesh.shape)
            center_pixel_y, center_pixel_x = center_idx

            qy_range = qy_mesh.max() - qy_mesh.min()
            qz_range = qz_mesh.max() - qz_mesh.min()
            pixel_x_range = img_width
            pixel_y_range = img_height

            qy_to_pixel_ratio = pixel_x_range / qy_range
            qz_to_pixel_ratio = pixel_y_range / qz_range

            width_pixel = width_q * qy_to_pixel_ratio
            height_pixel = height_q * qz_to_pixel_ratio

            result = {
                "center_x": int(center_pixel_x),
                "center_y": int(center_pixel_y),
                "width": int(width_pixel),
                "height": int(height_pixel),
            }

            return result

        except Exception as e:
            return {"center_x": 0, "center_y": 0, "width": 100, "height": 100}

    def _update_cutline_labels_units(self):
        """No description."""
        try:
            show_q_axis = self._should_show_q_axis()

            if show_q_axis:
                unit_suffix = " (nm^-1)"
            else:
                unit_suffix = " (pixel)"

            if hasattr(self.ui, "gisaxsInputCenterVerticalLabel"):
                self.ui.gisaxsInputCenterVerticalLabel.setText(f"Vertical.{unit_suffix}")

            if hasattr(self.ui, "gisaxsInputCenterParallelLabel"):
                self.ui.gisaxsInputCenterParallelLabel.setText(f"Parallel.{unit_suffix}")

            if hasattr(self.ui, "gisaxsInputCutLineVerticalLabel"):
                self.ui.gisaxsInputCutLineVerticalLabel.setText(f"Vertical.{unit_suffix}")

            if hasattr(self.ui, "gisaxsInputCutLineParallelLabel"):
                self.ui.gisaxsInputCutLineParallelLabel.setText(f"Parallel.{unit_suffix}")

        except Exception:
            pass

    def _should_show_q_axis(self):
        """No description."""
        try:
            return self.fitting_view_model.get_setting("fitting", "detector.show_q_axis", False)
        except Exception:
            return False

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

            cache_key = f"{width}x{height}_{pixel_size_x}_{pixel_size_y}_{beam_center_x}_{beam_center_y}_{distance}_{theta_in_deg}_{wavelength}"

            if self._q_cache_key != cache_key or self._q_detector is None:
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

                self._qy_mesh, self._qz_mesh = self._q_detector.get_qy_qz_meshgrids()
                self._q_cache_key = cache_key

            _, _, extent = _scientific_commands(self).q_space.axis_labels_and_extent(
                self._q_detector
            )
            return extent

        except Exception:
            height, width = image_shape
            return [-0.5, width - 0.5, -0.5, height - 0.5]

    def _get_cached_q_meshgrids(self):
        """No description."""
        return self._qy_mesh, self._qz_mesh
