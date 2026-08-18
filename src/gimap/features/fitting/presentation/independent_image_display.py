"""Independent Image Display behavior."""

from __future__ import annotations


from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (
    QAction,
    QComboBox,
    QLabel,
)


from .scientific_commands import (
    GISAXS_IMAGE_COLORMAPS,
)


class IndependentImageDisplayMixin:
    """Own independent image display behavior."""

    def _load_display_options(self):
        try:
            self.show_cut_region = bool(
                self.fitting_view_model.get_setting("fitting", "gisaxs_input.show_cut_region", True)
            )
            self.show_center = bool(
                self.fitting_view_model.get_setting("fitting", "gisaxs_input.show_center", True)
            )
            cmap = self.fitting_view_model.get_setting(
                "fitting", "gisaxs_input.colormap", "viridis"
            )
            self.colormap = cmap if cmap in GISAXS_IMAGE_COLORMAPS else "viridis"
        except Exception:
            pass

    def _save_display_options(self):
        try:
            self.fitting_view_model.set_setting(
                "fitting", "gisaxs_input.show_cut_region", bool(self.show_cut_region)
            )
            self.fitting_view_model.set_setting(
                "fitting", "gisaxs_input.show_center", bool(self.show_center)
            )
            self.fitting_view_model.set_setting("fitting", "gisaxs_input.colormap", self.colormap)
            self.fitting_view_model.save_settings()
        except Exception:
            pass

    def _setup_pick_center_action(self):
        try:
            if self.toolbar is None:
                return
            self.pick_center_action = QAction("Pick Center", self)
            self.pick_center_action.setCheckable(True)
            self.pick_center_action.setToolTip("Click the detector image to set Beam Center X/Y")
            self.pick_center_action.triggered.connect(self._toggle_pick_center_mode)
            self.toolbar.addSeparator()
            self.toolbar.addAction(self.pick_center_action)
        except Exception:
            self.pick_center_action = None

    def _setup_display_option_widgets(self):
        try:
            if self.toolbar is None:
                return
            self.show_cut_action = QAction("Cut Region", self)
            self.show_cut_action.setCheckable(True)
            self.show_cut_action.setChecked(bool(self.show_cut_region))
            self.show_cut_action.setToolTip("Show or hide the Cut Region overlay")
            self.show_cut_action.toggled.connect(self._on_show_cut_region_toggled)

            self.show_center_action = QAction("Center", self)
            self.show_center_action.setCheckable(True)
            self.show_center_action.setChecked(bool(self.show_center))
            self.show_center_action.setToolTip("Show or hide the center cross")
            self.show_center_action.toggled.connect(self._on_show_center_toggled)

            self.cmap_combo = QComboBox(self.toolbar)
            self.cmap_combo.addItems(list(GISAXS_IMAGE_COLORMAPS))
            idx = self.cmap_combo.findText(self.colormap)
            self.cmap_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.cmap_combo.setMinimumWidth(112)
            self.cmap_combo.setToolTip("Image color map")
            self.cmap_combo.currentTextChanged.connect(self._on_colormap_changed)

            self.toolbar.addSeparator()
            self.toolbar.addAction(self.show_cut_action)
            self.toolbar.addAction(self.show_center_action)
            self.toolbar.addWidget(QLabel("Color:", self.toolbar))
            self.toolbar.addWidget(self.cmap_combo)
        except Exception:
            pass

    def set_display_options(
        self, show_cut_region=None, show_center=None, colormap=None, emit=False
    ):
        try:
            if show_cut_region is not None:
                self.show_cut_region = bool(show_cut_region)
                action = getattr(self, "show_cut_action", None)
                if action is not None:
                    action.blockSignals(True)
                    action.setChecked(self.show_cut_region)
                    action.blockSignals(False)
            if show_center is not None:
                self.show_center = bool(show_center)
                action = getattr(self, "show_center_action", None)
                if action is not None:
                    action.blockSignals(True)
                    action.setChecked(self.show_center)
                    action.blockSignals(False)
            if colormap:
                self.colormap = colormap if colormap in GISAXS_IMAGE_COLORMAPS else "viridis"
                combo = getattr(self, "cmap_combo", None)
                if combo is not None:
                    combo.blockSignals(True)
                    idx = combo.findText(self.colormap)
                    combo.setCurrentIndex(idx if idx >= 0 else 0)
                    combo.blockSignals(False)
                if self.current_image is not None:
                    self.current_image.set_cmap(self.colormap)
                    if self.colorbar is not None:
                        self.colorbar.update_normal(self.current_image)
            self._redraw_parameter_selection()
            if self.canvas is not None:
                self.canvas.draw_idle()
            if emit:
                self._emit_display_options_changed()
        except Exception:
            pass

    def _emit_display_options_changed(self):
        options = {
            "show_cut_region": bool(self.show_cut_region),
            "show_center": bool(self.show_center),
            "colormap": self.colormap,
        }
        self._save_display_options()
        self.display_options_changed.emit(options)

    def _on_show_cut_region_toggled(self, checked: bool):
        self._set_cut_region_visible(checked, emit=True)

    def _on_show_center_toggled(self, checked: bool):
        self._set_center_visible(checked, emit=True)

    def _set_cut_region_visible(self, visible: bool, emit: bool = False):
        try:
            self.show_cut_region = bool(visible)
            action = getattr(self, "show_cut_action", None)
            if action is not None:
                action.blockSignals(True)
                action.setChecked(self.show_cut_region)
                action.blockSignals(False)
            self._redraw_parameter_selection()
            if self.canvas is not None:
                self.canvas.draw_idle()
            if emit:
                self._emit_display_options_changed()
        except Exception:
            pass

    def _set_center_visible(self, visible: bool, emit: bool = False):
        try:
            self.show_center = bool(visible)
            action = getattr(self, "show_center_action", None)
            if action is not None:
                action.blockSignals(True)
                action.setChecked(self.show_center)
                action.blockSignals(False)
            self._redraw_parameter_selection()
            if self.canvas is not None:
                self.canvas.draw_idle()
            if emit:
                self._emit_display_options_changed()
        except Exception:
            pass

    def _on_colormap_changed(self, text: str):
        self.colormap = text if text in GISAXS_IMAGE_COLORMAPS else "viridis"
        if self.current_image is not None:
            try:
                self.current_image.set_cmap(self.colormap)
                if self.colorbar is not None:
                    self.colorbar.update_normal(self.current_image)
            except Exception:
                pass
        if self.canvas is not None:
            self.canvas.draw_idle()
        self._emit_display_options_changed()

    def _toggle_pick_center_mode(self, checked: bool):
        try:
            if checked:
                self.pick_center_mode = True
                if self.selection_mode:
                    self._exit_selection_mode()
                if self.canvas is not None:
                    self.canvas.setCursor(Qt.CrossCursor)
                    self.canvas.setFocus()
                self.setWindowTitle(
                    "GIMaP Image Viewer - Pick Center Mode (click image, Esc to cancel)"
                )
                self.status_updated.emit(
                    "Pick Center mode: click one point on the image to set Detector Beam Center X/Y"
                )
            else:
                self._exit_pick_center_mode()
        except Exception:
            pass

    def _exit_pick_center_mode(self):
        try:
            self.pick_center_mode = False
            if self.pick_center_action is not None:
                self.pick_center_action.blockSignals(True)
                self.pick_center_action.setChecked(False)
                self.pick_center_action.blockSignals(False)
            if self.canvas is not None:
                self.canvas.unsetCursor()
            if not self.selection_mode:
                self.setWindowTitle(self.DEFAULT_TITLE)
        except Exception:
            pass
