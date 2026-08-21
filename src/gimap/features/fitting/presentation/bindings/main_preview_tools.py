"""Direct manipulation tools for the embedded detector preview."""

from __future__ import annotations

from PyQt5.QtCore import Qt

from ..detector_data_access import analysis_image_for


class MainPreviewToolsMixin:
    """Translate preview clicks/drags into existing center and region commands."""

    def _ensure_main_preview_interaction(self) -> None:
        canvas = getattr(self, "_canvas_cache", None)
        if canvas is None or getattr(self, "_main_preview_event_canvas", None) is canvas:
            return
        self._main_preview_event_canvas = canvas
        canvas.mpl_connect("button_press_event", self._on_main_preview_press)
        canvas.mpl_connect("motion_notify_event", self._on_main_preview_motion)
        canvas.mpl_connect("button_release_event", self._on_main_preview_release)
        canvas.mpl_connect("key_press_event", self._on_main_preview_key)

    def _set_main_preview_tool(self, tool: str | None) -> None:
        if tool and analysis_image_for(self) is None:
            self.status_updated.emit("Import and show a detector image before selecting on it")
            tool = None
        self._main_preview_tool = tool
        self._main_preview_selection_start = None
        self._remove_main_preview_drag_artist()
        pick = getattr(self.ui, "fittingPickCenterButton", None)
        region = getattr(self.ui, "fittingSelectRegionButton", None)
        for button, checked in ((pick, tool == "center"), (region, tool == "region")):
            if button is not None:
                button.blockSignals(True)
                button.setChecked(checked)
                button.blockSignals(False)
        canvas = getattr(self, "_canvas_cache", None)
        if canvas is not None:
            canvas.setCursor(Qt.CrossCursor if tool else Qt.ArrowCursor)
            canvas.setFocus()
        hint = getattr(self.ui, "fittingDetectorToolHint", None)
        if hint is not None:
            hint.setText(
                {
                    "center": "Click the Yoneda / detector center · Esc cancels",
                    "region": "Drag a rectangle for the cut region · Esc cancels",
                }.get(tool, "Esc cancels a selection tool")
            )
        if tool:
            self.status_updated.emit(
                "Pick center: click one point in the preview"
                if tool == "center"
                else "Select cut region: drag a rectangle in the preview"
            )

    def _toggle_main_center_tool(self, checked: bool) -> None:
        self._set_main_preview_tool("center" if checked else None)

    def _toggle_main_region_tool(self, checked: bool) -> None:
        self._set_main_preview_tool("region" if checked else None)

    def _on_main_preview_key(self, event) -> None:
        if str(getattr(event, "key", "")).lower() in {"escape", "esc"}:
            self._set_main_preview_tool(None)

    def _on_main_preview_press(self, event) -> None:
        if event.inaxes != getattr(self, "_preview_ax", None):
            return
        if event.xdata is None or event.ydata is None or event.button != 1:
            return
        if getattr(self, "_main_preview_tool", None) == "center":
            x_value = float(event.xdata)
            y_value = float(event.ydata)
            show_q_axis = bool(self._should_show_q_axis())
            payload = {"is_q_space": show_q_axis, "x": x_value, "y": y_value}
            if show_q_axis:
                point = self._snap_q_point(x_value, y_value)
                if point is None:
                    self.status_updated.emit("Q-space grid is not available")
                    return
                x_value, y_value = point.horizontal_q, point.qz
                image_height = analysis_image_for(self).shape[0]
                pixels = {
                    "center_x": point.column,
                    "center_y": image_height - 1 - point.row,
                }
                payload.update(x=x_value, y=y_value)
                payload.update(
                    center_qy=x_value,
                    center_qz=y_value,
                    beam_center_x=float(pixels.get("center_x", 0.0)),
                    beam_center_y=float(pixels.get("center_y", 0.0)),
                    horizontal_q_axis=self._horizontal_q_axis(),
                )
            else:
                payload.update(beam_center_x=x_value, beam_center_y=y_value)
            self._on_detector_center_picked(payload)
            self._set_main_preview_tool(None)
        elif getattr(self, "_main_preview_tool", None) == "region":
            self._main_preview_selection_start = (float(event.xdata), float(event.ydata))

    def _on_main_preview_motion(self, event) -> None:
        start = getattr(self, "_main_preview_selection_start", None)
        if (
            getattr(self, "_main_preview_tool", None) != "region"
            or start is None
            or event.inaxes != getattr(self, "_preview_ax", None)
            or event.xdata is None
            or event.ydata is None
        ):
            return
        self._remove_main_preview_drag_artist()
        from matplotlib.patches import Rectangle

        x0, y0 = start
        x1, y1 = float(event.xdata), float(event.ydata)
        self._main_preview_drag_artist = Rectangle(
            (min(x0, x1), min(y0, y1)),
            abs(x1 - x0),
            abs(y1 - y0),
            linewidth=2,
            edgecolor="#ef4444",
            facecolor="#ef4444",
            alpha=0.15,
        )
        self._preview_ax.add_patch(self._main_preview_drag_artist)
        self._canvas_cache.draw_idle()

    def _on_main_preview_release(self, event) -> None:
        start = getattr(self, "_main_preview_selection_start", None)
        if getattr(self, "_main_preview_tool", None) != "region" or start is None:
            return
        self._main_preview_selection_start = None
        self._remove_main_preview_drag_artist()
        if (
            event.inaxes != getattr(self, "_preview_ax", None)
            or event.xdata is None
            or event.ydata is None
            or event.button != 1
        ):
            return
        x0, y0 = start
        x1, y1 = float(event.xdata), float(event.ydata)
        width, height = abs(x1 - x0), abs(y1 - y0)
        show_q_axis = bool(self._should_show_q_axis())
        if width <= (0.001 if show_q_axis else 5) or height <= (
            0.001 if show_q_axis else 5
        ):
            self.status_updated.emit("Selection is too small; drag a larger rectangle")
            return
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        bounds = {
            "x_min": min(x0, x1),
            "x_max": max(x0, x1),
            "y_min": min(y0, y1),
            "y_max": max(y0, y1),
        }
        if show_q_axis:
            region = self._snap_q_region(
                min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
            )
            if region is None:
                self.status_updated.emit("Q-space grid is not available")
                return
            image_height = analysis_image_for(self).shape[0]
            selection = {
                "center_x": region.center_horizontal,
                "center_y": region.center_qz,
                "width": region.width,
                "height": region.height,
                "pixel_center_x": (region.column_min + region.column_max) / 2.0,
                "pixel_center_y": image_height
                - 1
                - ((region.row_min + region.row_max) / 2.0),
                "pixel_width": region.column_max - region.column_min + 1,
                "pixel_height": region.row_max - region.row_min + 1,
                "pixel_row_min": region.row_min,
                "pixel_row_max": region.row_max,
                "pixel_column_min": region.column_min,
                "pixel_column_max": region.column_max,
                "is_q_space": True,
                "horizontal_q_axis": self._horizontal_q_axis(),
                "bounds": {
                    "x_min": region.horizontal_min,
                    "x_max": region.horizontal_max,
                    "y_min": region.qz_min,
                    "y_max": region.qz_max,
                },
            }
        else:
            selection = {
                "center_x": center_x,
                "center_y": center_y,
                "width": width,
                "height": height,
                "pixel_center_x": int(center_x),
                "pixel_center_y": int(center_y),
                "pixel_width": int(width),
                "pixel_height": int(height),
                "is_q_space": False,
                "bounds": bounds,
            }
        show_region = getattr(self.ui, "gisaxsInputShowCutRegionCheckBox", None)
        if show_region is not None:
            show_region.setChecked(True)
        self._on_region_selected(selection)

    def _remove_main_preview_drag_artist(self) -> None:
        artist = getattr(self, "_main_preview_drag_artist", None)
        if artist is not None:
            try:
                artist.remove()
            except (ValueError, RuntimeError):
                pass
        self._main_preview_drag_artist = None
        canvas = getattr(self, "_canvas_cache", None)
        if canvas is not None:
            canvas.draw_idle()

    def _reset_main_detector_view(self) -> None:
        self._set_main_preview_tool(None)
        self._refresh_image_display()


__all__ = ["MainPreviewToolsMixin"]
