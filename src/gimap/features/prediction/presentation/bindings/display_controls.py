"""Display Controls coordination for Prediction."""

from __future__ import annotations


from PyQt5.QtCore import Qt, QRectF


class DisplayControlsMixin:
    """Own display controls presentation behavior."""

    def _zoom_in(self) -> None:
        self._view_zoom_steps += 1
        self._apply_zoom()

    def _zoom_out(self) -> None:
        self._view_zoom_steps -= 1
        self._apply_zoom()

    def _zoom_reset(self) -> None:
        self._view_zoom_steps = 0
        self._apply_zoom(reset=True)

    def _predict_zoom_in(self) -> None:
        self._predict_zoom_steps += 1
        self._apply_predict_zoom()

    def _predict_zoom_out(self) -> None:
        self._predict_zoom_steps -= 1
        self._apply_predict_zoom()

    def _predict_zoom_reset(self) -> None:
        self._predict_zoom_steps = 0
        self._apply_predict_zoom(reset=True)

    def _apply_zoom(self, reset: bool = False) -> None:
        view = getattr(self.ui, "gisaxsImageGraphicsView", None)
        if view is None or self._current_pixmap is None:
            return
        view.resetTransform()
        if reset:
            view.fitInView(QRectF(self._current_pixmap.rect()), Qt.KeepAspectRatio)
            return
        factor = 1.15**self._view_zoom_steps
        view.scale(factor, factor)

    def _apply_predict_zoom(self, reset: bool = False) -> None:
        view = getattr(self.ui, "predict2dGraphicsView", None)
        if view is None or self._predict_pixmap is None:
            return
        view.resetTransform()
        if reset:
            view.fitInView(QRectF(self._predict_pixmap.rect()), Qt.KeepAspectRatio)
            return
        factor = 1.15**self._predict_zoom_steps
        view.scale(factor, factor)

    def _on_auto_scale_toggled(self) -> None:
        if self._ui_updating:
            return
        auto = getattr(self.ui, "gisaxsImageAutoScaleCheckBox", None)
        checked = bool(auto.isChecked()) if auto else True
        self.current_parameters["auto_scale"] = checked
        self._persist_parameters()
        if checked:
            self._update_image_display()

    def _on_auto_scale_reset(self) -> None:
        self.current_parameters["auto_scale"] = True
        self._set_checkbox("gisaxsImageAutoScaleCheckBox", True)
        self._persist_parameters()
        self._update_image_display()

    def _on_vmin_changed(self) -> None:
        if self._ui_updating:
            return
        value = self._get_double_spin_value("gisaxsImageVminValue")
        if value is None:
            return
        self.current_parameters["auto_scale"] = False
        self._set_checkbox("gisaxsImageAutoScaleCheckBox", False)
        self.current_parameters["vmin"] = value
        self._persist_parameters()
        self._update_image_display()

    def _on_vmax_changed(self) -> None:
        if self._ui_updating:
            return
        value = self._get_double_spin_value("gisaxsImageVmaxValue")
        if value is None:
            return
        self.current_parameters["auto_scale"] = False
        self._set_checkbox("gisaxsImageAutoScaleCheckBox", False)
        self.current_parameters["vmax"] = value
        self._persist_parameters()
        self._update_image_display()

    def _on_colormap_changed(self, text: str) -> None:
        if self._ui_updating:
            return
        self.current_parameters["colormap"] = text or self._DEFAULT_COLORMAPS[0]
        self._update_image_display()
        self._rerender_predict_view()

    def _on_predict_auto_scale_toggled(self) -> None:
        if self._ui_updating:
            return
        cb = getattr(self.ui, "predict2dAutoScaleCheckBox", None)
        checked = bool(cb.isChecked()) if cb else True
        self.current_parameters["predict_auto_scale"] = checked
        self._persist_parameters()
        self._rerender_predict_view()

    def _on_predict_auto_scale_reset(self) -> None:
        self.current_parameters["predict_auto_scale"] = True
        self._set_checkbox("predict2dAutoScaleCheckBox", True)
        self._persist_parameters()
        self._rerender_predict_view()

    def _on_predict_vmin_changed(self) -> None:
        if self._ui_updating:
            return
        value = self._get_double_spin_value("predict2dVminValue")
        if value is None:
            return
        self.current_parameters["predict_auto_scale"] = False
        self._set_checkbox("predict2dAutoScaleCheckBox", False)
        self.current_parameters["predict_vmin"] = value
        self._persist_parameters()
        self._rerender_predict_view()

    def _on_predict_vmax_changed(self) -> None:
        if self._ui_updating:
            return
        value = self._get_double_spin_value("predict2dVmaxValue")
        if value is None:
            return
        self.current_parameters["predict_auto_scale"] = False
        self._set_checkbox("predict2dAutoScaleCheckBox", False)
        self.current_parameters["predict_vmax"] = value
        self._persist_parameters()
        self._rerender_predict_view()
