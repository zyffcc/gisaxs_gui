"""Image Display Options for fitting presentation."""

from __future__ import annotations

import numpy as np

from PyQt5.QtCore import QSettings

from ..binding_primitives import (
    GISAXS_IMAGE_COLORMAPS,
    _scientific_commands,
)
from ..detector_data_access import analysis_image_for


class ImageDisplayOptionsMixin:
    """Own image display options behavior."""

    def _load_image_display_options(self):
        try:
            settings = QSettings()
            self._show_cut_region = bool(
                self.fitting_view_model.get_setting("fitting", "gisaxs_input.show_cut_region", True)
            )
            self._show_center = bool(
                self.fitting_view_model.get_setting("fitting", "gisaxs_input.show_center", True)
            )
            cmap = self.fitting_view_model.get_setting(
                "fitting", "gisaxs_input.colormap", "viridis"
            )
            self._image_colormap = cmap if cmap in GISAXS_IMAGE_COLORMAPS else "viridis"
            self._flip_ud = settings.value(
                "fitting/gisaxs_input/flip_ud",
                bool(self.fitting_view_model.get_setting("fitting", "gisaxs_input.flip_ud", False)),
                type=bool,
            )
            self._threshold_mask_enabled = settings.value(
                "fitting/gisaxs_input/threshold_mask_enabled",
                bool(
                    self.fitting_view_model.get_setting(
                        "fitting", "gisaxs_input.threshold_mask_enabled", False
                    )
                ),
                type=bool,
            )
            self._threshold_mask_min = float(
                settings.value(
                    "fitting/gisaxs_input/threshold_mask_min",
                    self.fitting_view_model.get_setting(
                        "fitting", "gisaxs_input.threshold_mask_min", -1e12
                    ),
                )
            )
            self._threshold_mask_max = float(
                settings.value(
                    "fitting/gisaxs_input/threshold_mask_max",
                    self.fitting_view_model.get_setting(
                        "fitting", "gisaxs_input.threshold_mask_max", 1e12
                    ),
                )
            )
            self._mirror_fill_detector_gaps = settings.value(
                "fitting/gisaxs_input/mirror_fill_detector_gaps",
                bool(
                    self.fitting_view_model.get_setting(
                        "fitting", "gisaxs_input.mirror_fill_detector_gaps", False
                    )
                ),
                type=bool,
            )
            self._mirror_gap_margin_px = int(
                settings.value(
                    "fitting/gisaxs_input/mirror_gap_margin_px",
                    self.fitting_view_model.get_setting(
                        "fitting", "gisaxs_input.mirror_gap_margin_px", 0
                    ),
                    type=int,
                )
            )
            self._mirror_gap_margin_px = int(np.clip(self._mirror_gap_margin_px, 0, 20))
        except Exception:
            self._show_cut_region = True
            self._show_center = True
            self._image_colormap = "viridis"
            self._flip_ud = False
            self._threshold_mask_enabled = False
            self._threshold_mask_min = -1e12
            self._threshold_mask_max = 1e12
            self._mirror_fill_detector_gaps = False
            self._mirror_gap_margin_px = 0
            try:
                settings = QSettings()
                self._mirror_fill_detector_gaps = settings.value(
                    "fitting/gisaxs_input/mirror_fill_detector_gaps",
                    False,
                    type=bool,
                )
                self._mirror_gap_margin_px = int(
                    settings.value("fitting/gisaxs_input/mirror_gap_margin_px", 0, type=int)
                )
                self._mirror_gap_margin_px = int(np.clip(self._mirror_gap_margin_px, 0, 20))
            except Exception:
                pass

    def _save_image_display_options(self):
        try:
            settings = QSettings()
            settings.setValue("fitting/gisaxs_input/flip_ud", bool(self._flip_ud))
            settings.setValue(
                "fitting/gisaxs_input/threshold_mask_enabled", bool(self._threshold_mask_enabled)
            )
            settings.setValue(
                "fitting/gisaxs_input/threshold_mask_min", float(self._threshold_mask_min)
            )
            settings.setValue(
                "fitting/gisaxs_input/threshold_mask_max", float(self._threshold_mask_max)
            )
            settings.setValue(
                "fitting/gisaxs_input/mirror_fill_detector_gaps",
                bool(self._mirror_fill_detector_gaps),
            )
            settings.setValue(
                "fitting/gisaxs_input/mirror_gap_margin_px", int(self._mirror_gap_margin_px)
            )
            self.fitting_view_model.set_setting(
                "fitting", "gisaxs_input.show_cut_region", bool(self._show_cut_region)
            )
            self.fitting_view_model.set_setting(
                "fitting", "gisaxs_input.show_center", bool(self._show_center)
            )
            self.fitting_view_model.set_setting(
                "fitting", "gisaxs_input.colormap", self._image_colormap
            )
            self.fitting_view_model.set_setting(
                "fitting", "gisaxs_input.flip_ud", bool(self._flip_ud)
            )
            self.fitting_view_model.set_setting(
                "fitting", "gisaxs_input.threshold_mask_enabled", bool(self._threshold_mask_enabled)
            )
            self.fitting_view_model.set_setting(
                "fitting", "gisaxs_input.threshold_mask_min", float(self._threshold_mask_min)
            )
            self.fitting_view_model.set_setting(
                "fitting", "gisaxs_input.threshold_mask_max", float(self._threshold_mask_max)
            )
            self.fitting_view_model.set_setting(
                "fitting",
                "gisaxs_input.mirror_fill_detector_gaps",
                bool(self._mirror_fill_detector_gaps),
            )
            self.fitting_view_model.set_setting(
                "fitting",
                "gisaxs_input.mirror_gap_margin_px",
                int(self._mirror_gap_margin_px),
            )
            self.fitting_view_model.save_settings()
        except Exception:
            pass

    def _initialize_image_display_option_widgets(self):
        try:
            combo = getattr(self.ui, "gisaxsInputColormapCombo", None)
            if combo is not None:
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(list(GISAXS_IMAGE_COLORMAPS))
                idx = combo.findText(self._image_colormap)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
                combo.blockSignals(False)
                combo.currentTextChanged.connect(self._on_main_colormap_changed)
            cut_cb = getattr(self.ui, "gisaxsInputShowCutRegionCheckBox", None)
            if cut_cb is not None:
                cut_cb.blockSignals(True)
                cut_cb.setChecked(bool(self._show_cut_region))
                cut_cb.blockSignals(False)
                cut_cb.toggled.connect(self._on_main_show_cut_region_toggled)
            center_cb = getattr(self.ui, "gisaxsInputShowCenterCheckBox", None)
            if center_cb is not None:
                center_cb.blockSignals(True)
                center_cb.setChecked(bool(self._show_center))
                center_cb.blockSignals(False)
                center_cb.toggled.connect(self._on_main_show_center_toggled)
            flip_cb = getattr(self.ui, "gisaxsInputFlipUdCheckBox", None)
            if flip_cb is not None:
                flip_cb.blockSignals(True)
                flip_cb.setChecked(bool(self._flip_ud))
                flip_cb.blockSignals(False)
                flip_cb.toggled.connect(self._on_main_flip_ud_toggled)
            threshold_cb = getattr(self.ui, "gisaxsInputThresholdMaskCheckBox", None)
            if threshold_cb is not None:
                threshold_cb.blockSignals(True)
                threshold_cb.setChecked(bool(self._threshold_mask_enabled))
                threshold_cb.blockSignals(False)
                threshold_cb.toggled.connect(self._on_threshold_mask_toggled)
            threshold_min = getattr(self.ui, "gisaxsInputThresholdMinSpinBox", None)
            if threshold_min is not None:
                threshold_min.blockSignals(True)
                threshold_min.setValue(float(self._threshold_mask_min))
                threshold_min.blockSignals(False)
                threshold_min.editingFinished.connect(self._on_threshold_limits_committed)
            threshold_max = getattr(self.ui, "gisaxsInputThresholdMaxSpinBox", None)
            if threshold_max is not None:
                threshold_max.blockSignals(True)
                threshold_max.setValue(float(self._threshold_mask_max))
                threshold_max.blockSignals(False)
                threshold_max.editingFinished.connect(self._on_threshold_limits_committed)
            self._set_threshold_mask_controls_enabled(bool(self._threshold_mask_enabled))
            mirror_cb = getattr(self.ui, "gisaxsInputMirrorGapFillCheckBox", None)
            if mirror_cb is not None:
                mirror_cb.blockSignals(True)
                mirror_cb.setChecked(bool(self._mirror_fill_detector_gaps))
                mirror_cb.blockSignals(False)
                mirror_cb.toggled.connect(self._on_main_mirror_gap_fill_toggled)
            margin_spin = getattr(self.ui, "gisaxsInputMirrorGapMarginSpinBox", None)
            if margin_spin is not None:
                margin_spin.blockSignals(True)
                margin_spin.setValue(int(self._mirror_gap_margin_px))
                margin_spin.setEnabled(bool(self._mirror_fill_detector_gaps))
                margin_spin.blockSignals(False)
                margin_spin.valueChanged.connect(self._on_main_mirror_gap_margin_changed)
            margin_label = getattr(self.ui, "gisaxsInputMirrorGapMarginLabel", None)
            if margin_label is not None:
                margin_label.setEnabled(bool(self._mirror_fill_detector_gaps))
            margin_unit = getattr(self.ui, "gisaxsInputMirrorGapMarginUnitLabel", None)
            if margin_unit is not None:
                margin_unit.setEnabled(bool(self._mirror_fill_detector_gaps))
        except Exception:
            pass

    def _sync_image_display_option_widgets(self):
        try:
            self._syncing_image_display_options = True
            combo = getattr(self.ui, "gisaxsInputColormapCombo", None)
            if combo is not None:
                combo.blockSignals(True)
                idx = combo.findText(self._image_colormap)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
                combo.blockSignals(False)
            cut_cb = getattr(self.ui, "gisaxsInputShowCutRegionCheckBox", None)
            if cut_cb is not None:
                cut_cb.blockSignals(True)
                cut_cb.setChecked(bool(self._show_cut_region))
                cut_cb.blockSignals(False)
            center_cb = getattr(self.ui, "gisaxsInputShowCenterCheckBox", None)
            if center_cb is not None:
                center_cb.blockSignals(True)
                center_cb.setChecked(bool(self._show_center))
                center_cb.blockSignals(False)
            flip_cb = getattr(self.ui, "gisaxsInputFlipUdCheckBox", None)
            if flip_cb is not None:
                flip_cb.blockSignals(True)
                flip_cb.setChecked(bool(self._flip_ud))
                flip_cb.blockSignals(False)
            threshold_cb = getattr(self.ui, "gisaxsInputThresholdMaskCheckBox", None)
            if threshold_cb is not None:
                threshold_cb.blockSignals(True)
                threshold_cb.setChecked(bool(self._threshold_mask_enabled))
                threshold_cb.blockSignals(False)
            threshold_min = getattr(self.ui, "gisaxsInputThresholdMinSpinBox", None)
            if threshold_min is not None:
                threshold_min.blockSignals(True)
                threshold_min.setValue(float(self._threshold_mask_min))
                threshold_min.blockSignals(False)
            threshold_max = getattr(self.ui, "gisaxsInputThresholdMaxSpinBox", None)
            if threshold_max is not None:
                threshold_max.blockSignals(True)
                threshold_max.setValue(float(self._threshold_mask_max))
                threshold_max.blockSignals(False)
            self._set_threshold_mask_controls_enabled(bool(self._threshold_mask_enabled))
            mirror_cb = getattr(self.ui, "gisaxsInputMirrorGapFillCheckBox", None)
            if mirror_cb is not None:
                mirror_cb.blockSignals(True)
                mirror_cb.setChecked(bool(self._mirror_fill_detector_gaps))
                mirror_cb.blockSignals(False)
            margin_spin = getattr(self.ui, "gisaxsInputMirrorGapMarginSpinBox", None)
            if margin_spin is not None:
                margin_spin.blockSignals(True)
                margin_spin.setValue(int(self._mirror_gap_margin_px))
                margin_spin.setEnabled(bool(self._mirror_fill_detector_gaps))
                margin_spin.blockSignals(False)
            margin_label = getattr(self.ui, "gisaxsInputMirrorGapMarginLabel", None)
            if margin_label is not None:
                margin_label.setEnabled(bool(self._mirror_fill_detector_gaps))
            margin_unit = getattr(self.ui, "gisaxsInputMirrorGapMarginUnitLabel", None)
            if margin_unit is not None:
                margin_unit.setEnabled(bool(self._mirror_fill_detector_gaps))
        finally:
            self._syncing_image_display_options = False

    def _apply_image_display_options(self, *, refresh=True, sync_window=True):
        self._save_image_display_options()
        self._refresh_current_parameter_selection_from_ui()
        self._sync_image_display_option_widgets()
        state = self._current_detector_display_state()
        if sync_window and self.independent_window is not None:
            try:
                self.independent_window.set_detector_display_state(state)
            except Exception:
                pass
        if refresh and analysis_image_for(self) is not None:
            try:
                if self._is_auto_scale_enabled():
                    display_image = self._get_current_display_image()
                    if display_image is not None:
                        self._handle_color_scale(display_image)
            except Exception:
                pass
            self._refresh_image_display()

    def _on_main_show_cut_region_toggled(self, checked: bool):
        if self._syncing_image_display_options:
            return
        self._show_cut_region = bool(checked)
        self._apply_image_display_options()

    def _on_main_show_center_toggled(self, checked: bool):
        if self._syncing_image_display_options:
            return
        self._show_center = bool(checked)
        self._apply_image_display_options()

    def _on_main_flip_ud_toggled(self, checked: bool):
        if self._syncing_image_display_options:
            return
        self._flip_ud = bool(checked)
        self._save_image_display_options()
        self._reapply_input_image_options()

    def _set_threshold_mask_controls_enabled(self, enabled: bool):
        for name in (
            "gisaxsInputThresholdMinLabel",
            "gisaxsInputThresholdMinSpinBox",
            "gisaxsInputThresholdMaxLabel",
            "gisaxsInputThresholdMaxSpinBox",
        ):
            widget = getattr(self.ui, name, None)
            if widget is not None:
                widget.setEnabled(bool(enabled))

    def _on_threshold_mask_toggled(self, checked: bool):
        if self._syncing_image_display_options:
            return
        self._threshold_mask_enabled = bool(checked)
        self._set_threshold_mask_controls_enabled(self._threshold_mask_enabled)
        self._save_image_display_options()
        self._reapply_input_image_options()

    def _on_threshold_limits_committed(self):
        if self._syncing_image_display_options:
            return
        lower_widget = getattr(self.ui, "gisaxsInputThresholdMinSpinBox", None)
        upper_widget = getattr(self.ui, "gisaxsInputThresholdMaxSpinBox", None)
        if lower_widget is None or upper_widget is None:
            return
        lower = float(lower_widget.value())
        upper = float(upper_widget.value())
        self._threshold_mask_min, self._threshold_mask_max = sorted((lower, upper))
        self._save_image_display_options()
        self._sync_image_display_option_widgets()
        if self._threshold_mask_enabled:
            self._reapply_input_image_options()

    def _on_main_colormap_changed(self, text: str):
        if self._syncing_image_display_options:
            return
        self._image_colormap = text if text in GISAXS_IMAGE_COLORMAPS else "viridis"
        self._apply_image_display_options()

    def _on_main_mirror_gap_fill_toggled(self, checked: bool):
        if self._syncing_image_display_options:
            return
        self._mirror_fill_detector_gaps = bool(checked)
        self._image_display_cache.clear()
        self._save_image_display_options()
        self._reapply_input_image_options()

    def _on_main_mirror_gap_margin_changed(self, value: int):
        if self._syncing_image_display_options:
            return
        self._mirror_gap_margin_px = int(np.clip(int(value), 0, 20))
        self._image_display_cache.clear()
        self._save_image_display_options()
        if self._mirror_fill_detector_gaps:
            self._reapply_input_image_options()

    def _get_mirror_gap_fill_center_x(self):
        try:
            image_data = self.current_raw_image
            if image_data is None:
                return None
            _, width = image_data.shape
            return float(
                self.fitting_view_model.get_setting(
                    "fitting", "detector.beam_center_x", width / 2.0
                )
            )
        except Exception:
            return None

    def _reapply_input_image_options(self, refresh=True):
        raw = self.current_raw_image
        if raw is None:
            return
        next_revision = int(getattr(self, "_analysis_revision", 0)) + 1
        mirror_enabled = bool(getattr(self, "_mirror_fill_detector_gaps", False))
        try:
            stack_count = max(1, int(self.current_parameters.get("stack_count", 1)))
        except Exception:
            stack_count = 1
        try:
            detector_image = _scientific_commands(self).image.prepare(
                raw,
                revision=next_revision,
                flip_ud=bool(getattr(self, "_flip_ud", False)),
                threshold_enabled=bool(getattr(self, "_threshold_mask_enabled", False)),
                threshold_min=float(getattr(self, "_threshold_mask_min", -1e12)),
                threshold_max=float(getattr(self, "_threshold_mask_max", 1e12)),
                mirror_fill_gaps=mirror_enabled,
                mirror_center_x=(
                    self._get_mirror_gap_fill_center_x() if mirror_enabled else None
                ),
                mirror_gap_margin_px=int(getattr(self, "_mirror_gap_margin_px", 0)),
                mirror_gap_value=-float(stack_count),
            )
        except Exception as exc:
            message = f"Image preprocessing failed: {exc}"
            emitter = getattr(getattr(self, "status_updated", None), "emit", None)
            if callable(emitter):
                emitter(message)
            self._last_mirror_fill_status = message
            return

        self.current_detector_image = detector_image
        self.current_raw_image = detector_image.raw_image
        self.current_analysis_image = detector_image.analysis_image
        self.current_stack_data = detector_image.analysis_image  # legacy read-only alias
        self.data = detector_image.analysis_image  # legacy read-only alias
        self._analysis_revision = detector_image.revision
        processed = detector_image.analysis_image
        self.summed_data = processed if stack_count > 1 else None
        self._image_display_cache.clear()
        self._last_mirror_fill_count = detector_image.mirror_filled_gap_pixels
        if mirror_enabled:
            message = (
                "Mirror gap fill applied to analysis data: "
                f"margin={detector_image.preprocessing.mirror_gap_margin_px} px, "
                f"replaced {detector_image.mirror_replaced_pixels} pixels"
            )
            if message != getattr(self, "_last_mirror_fill_status", ""):
                emitter = getattr(getattr(self, "status_updated", None), "emit", None)
                if callable(emitter):
                    emitter(message)
            self._last_mirror_fill_status = message
        else:
            self._last_mirror_fill_status = ""
        view_model = getattr(self, "fitting_view_model", None)
        accept_revision = getattr(view_model, "accept_analysis_revision", None)
        if callable(accept_revision):
            accept_revision(detector_image.revision)
            sync_workflow = getattr(self, "_sync_fitting_workflow", None)
            if callable(sync_workflow):
                sync_workflow()
        if self._threshold_mask_enabled:
            masked_count = int(np.count_nonzero(~np.isfinite(processed)))
            self.status_updated.emit(
                f"Threshold mask applied: {masked_count} pixel(s) excluded "
                f"outside [{self._threshold_mask_min:.6g}, {self._threshold_mask_max:.6g}]"
            )
        if refresh:
            display_image = self._get_current_display_image()
            if display_image is not None and self._is_auto_scale_enabled():
                self._handle_color_scale(display_image)
            self._refresh_image_display()

    def _get_current_display_image(self):
        # Display-only state is applied by the renderer. Scientific preprocessing
        # has already produced the one canonical analysis array.
        return analysis_image_for(self)

    def _on_independent_display_options_changed(self, options: dict):
        """Compatibility bridge for older independent image windows."""
        try:
            self._show_cut_region = bool(options.get("show_cut_region", self._show_cut_region))
            self._show_center = bool(options.get("show_center", self._show_center))
            cmap = options.get("colormap", self._image_colormap)
            self._image_colormap = cmap if cmap in GISAXS_IMAGE_COLORMAPS else "viridis"
            self._apply_image_display_options(sync_window=False)
        except Exception:
            pass

    def _persist_cut_region_parameters(
        self, center_parallel, center_vertical, cutline_parallel, cutline_vertical
    ):
        """Persist Cut Region geometry even when values are changed programmatically."""
        try:
            values = {
                "center_parallel": float(center_parallel),
                "center_vertical": float(center_vertical),
                "cutline_parallel": float(cutline_parallel),
                "cutline_vertical": float(cutline_vertical),
            }
            for key, value in values.items():
                self.fitting_view_model.set_setting("fitting", f"gisaxs_input.{key}", value)
            self.fitting_view_model.save_settings()
        except Exception:
            try:
                for key, value in {
                    "center_parallel": float(center_parallel),
                    "center_vertical": float(center_vertical),
                    "cutline_parallel": float(cutline_parallel),
                    "cutline_vertical": float(cutline_vertical),
                }.items():
                    self.fitting_view_model.set_setting("fitting", f"gisaxs_input.{key}", value)
                self.fitting_view_model.save_settings()
            except Exception:
                pass

    def _connect_parameter_widgets(self):
        """No description."""

        self._setup_particle_connections()
