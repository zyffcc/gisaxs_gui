"""Independent Image Selection behavior."""

from __future__ import annotations


class IndependentImageSelectionMixin:
    """Own independent image selection behavior."""

    def update_parameter_selection(self, center_v, center_p, cutline_v, cutline_p):
        """No description."""
        if center_v == 0 and center_p == 0 and cutline_v == 0 and cutline_p == 0:
            self.clear_parameter_selection()
            return

        x_start = center_p - cutline_p / 2
        x_end = center_p + cutline_p / 2
        y_start = center_v - cutline_v / 2
        y_end = center_v + cutline_v / 2

        self.set_parameter_selection(
            {
                "bounds": {
                    "x_min": x_start,
                    "x_max": x_end,
                    "y_min": y_start,
                    "y_max": y_end,
                },
                "pixel_center_x": center_p,
                "pixel_center_y": center_v,
                "pixel_width": cutline_p,
                "pixel_height": cutline_v,
                "is_q_space": False,
                "is_parameter_based": True,
            }
        )

    def set_parameter_selection(self, selection_info):
        """No description."""
        self.parameter_selection_info = dict(selection_info) if selection_info else None
        self._redraw_parameter_selection()
        if self.canvas is not None:
            self.canvas.draw_idle()

    def _redraw_parameter_selection(self):
        """No description."""
        try:
            if self.parameter_selection is not None:
                try:
                    self.parameter_selection.remove()
                except Exception:
                    pass
                finally:
                    self.parameter_selection = None
            if self.parameter_selection_center is not None:
                try:
                    self.parameter_selection_center.remove()
                except Exception:
                    pass
                finally:
                    self.parameter_selection_center = None
            picked_marker = getattr(self, "_picked_center_marker", None)
            if picked_marker is not None:
                try:
                    picked_marker.remove()
                except Exception:
                    pass
                finally:
                    self._picked_center_marker = None

            if self.ax is None:
                return

            if self.parameter_selection_info and self.show_cut_region:
                bounds = self.parameter_selection_info.get("bounds", {})
                x_min = bounds.get("x_min", 0)
                x_max = bounds.get("x_max", 0)
                y_min = bounds.get("y_min", 0)
                y_max = bounds.get("y_max", 0)
                if x_min != x_max and y_min != y_max:
                    from matplotlib.patches import Rectangle

                    self.parameter_selection = Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                        alpha=0.85,
                    )
                    self.ax.add_patch(self.parameter_selection)
            if self.show_center:
                center = (
                    getattr(self, "_drag_current_center", None)
                    or self._get_detector_center_display_coordinates()
                )
                if center is None:
                    return
                center_lines = self.ax.plot(
                    float(center[0]),
                    float(center[1]),
                    marker="+",
                    color="cyan",
                    markersize=14,
                    markeredgewidth=2.5,
                )
                self._picked_center_marker = center_lines[0] if center_lines else None
                self.parameter_selection_center = self._picked_center_marker
        except Exception:
            pass

    def clear_parameter_selection(self):
        """No description."""
        self.parameter_selection_info = None
        if self.parameter_selection is not None:
            try:
                self.parameter_selection.remove()
            except Exception:
                pass
            finally:
                self.parameter_selection = None
        if self.parameter_selection_center is not None:
            try:
                self.parameter_selection_center.remove()
            except Exception:
                pass
            finally:
                self.parameter_selection_center = None
        self._redraw_parameter_selection()
        self.canvas.draw()

    def closeEvent(self, event):
        """No description."""
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            finally:
                self.colorbar = None

        try:
            self.figure.clear()
        except Exception:
            pass

        super().closeEvent(event)
