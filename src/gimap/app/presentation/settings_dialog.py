"""Application-level display settings dialog for adaptive layouts."""

from __future__ import annotations

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QMessageBox,
)

from src.gimap.app.ports import UserPreferencesRepository
from src.gimap.app.presentation.responsive_layout import (
    LAYOUT_TARGETS,
    PROFILES,
    apply_density_profile,
    apply_window_profile,
    current_profile,
    install_adaptive_window_profile,
    parse_resolution,
    screen_metrics,
)
from src.gimap.app.presentation.style_loader import apply_main_window_styles

from .views import SettingsDialogView


class SettingsDialog(QDialog, SettingsDialogView):
    """Modern Display settings dialog."""

    PROFILE_LABELS = {
        "compact": "Compact",
        "normal": "Normal",
        "wide": "Wide",
    }
    def __init__(self, parent=None, *, preferences: UserPreferencesRepository):
        super().__init__(parent)
        self.parent_window = parent
        self.preferences = preferences
        self.setupUi(self)
        self.setModal(True)
        self._bind_form()
        self.load_settings()
        self._clamp_to_screen()
        install_adaptive_window_profile(
            self,
            self._on_dialog_screen_changed,
            apply_window_minimum=False,
            preferences=self.preferences,
        )

    def _bind_form(self) -> None:
        self.ui_scale_slider.valueChanged.connect(self._on_ui_scale_changed)
        self.adaptive_layout_cb.toggled.connect(self._on_adaptive_toggled)
        for key in ("auto", "compact", "normal", "wide", "custom"):
            self.layout_target_combo.addItem(LAYOUT_TARGETS[key].label, key)
        self.layout_target_combo.currentIndexChanged.connect(self._on_layout_target_changed)
        for value in (
            "1280x720",
            "1366x768",
            "1920x1080",
            "1920x1200",
            "2560x1440",
            "2560x1600",
            "2560x1660",
            "3440x1440",
            "3840x2160",
        ):
            self.custom_target_combo.addItem(value)
        self.custom_target_combo.editTextChanged.connect(self._refresh_summary)
        self.custom_target_combo.currentIndexChanged.connect(self._refresh_summary)
        self.auto_fit_cb.toggled.connect(self._refresh_summary)
        self.override_combo.addItem("Auto", "auto")
        for key, label in self.PROFILE_LABELS.items():
            self.override_combo.addItem(label, key)
        self.override_combo.currentIndexChanged.connect(self._refresh_summary)
        self.apply_button.clicked.connect(self.apply_settings)
        self.cancel_button.clicked.connect(self.reject)
        self.ok_button.clicked.connect(self.accept_settings)

    def _clamp_to_screen(self) -> None:
        metrics = screen_metrics(
            self.parent_window or self,
            preferences=self.preferences,
        )
        available = metrics.logical_geometry
        target_width = min(max(500, self.sizeHint().width()), max(500, int(available.width() * 0.92)))
        target_height = min(max(420, self.sizeHint().height()), max(420, int(available.height() * 0.80)))
        self.setMaximumHeight(max(420, int(available.height() * 0.80)))
        self.resize(target_width, target_height)

    def load_settings(self) -> None:
        self.auto_dpi_cb.setChecked(self.preferences.get("auto_detect_monitor_dpi", True))
        adaptive = self.preferences.get("adaptive_layout_enabled", True)
        self.adaptive_layout_cb.setChecked(adaptive)
        mode = self.preferences.get("responsive_layout_mode", "auto")
        if mode == "standard":
            mode = "normal"
        elif mode == "spacious":
            mode = "wide"
        index = self.override_combo.findData(mode)
        self.override_combo.setCurrentIndex(index if index >= 0 else 0)
        target_mode = self.preferences.get("layout_target_mode", "")
        if not target_mode:
            target_mode = "custom" if self.preferences.get("manual_screen_resolution", "auto") != "auto" else "auto"
        target_index = self.layout_target_combo.findData(target_mode)
        self.layout_target_combo.setCurrentIndex(target_index if target_index >= 0 else 0)
        custom_target = self.preferences.get(
            "layout_target_custom",
            self.preferences.get("manual_screen_resolution", "1920x1080"),
        )
        if custom_target and custom_target != "auto":
            self.custom_target_combo.setEditText(str(custom_target))
        self.auto_fit_cb.setChecked(self.preferences.get("auto_fit_layout_target", True))
        self.ui_scale_slider.setValue(int(self.preferences.get("visual_font_scale", 100)))
        self._on_ui_scale_changed(self.ui_scale_slider.value())
        self._on_adaptive_toggled(adaptive)
        self._on_layout_target_changed()
        self._refresh_summary()

    def _on_adaptive_toggled(self, enabled: bool) -> None:
        self.override_combo.setEnabled(not enabled)
        if enabled:
            auto_index = self.override_combo.findData("auto")
            self.override_combo.setCurrentIndex(auto_index)
        self._refresh_summary()

    def _on_ui_scale_changed(self, value: int) -> None:
        self.ui_scale_label.setText(f"{value}%")
        self._refresh_summary()

    def _on_layout_target_changed(self, *_args) -> None:
        is_custom = self.layout_target_combo.currentData() == "custom"
        self.custom_target_combo.setEnabled(is_custom)
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        profile = self._selected_profile()
        metrics = screen_metrics(
            self.parent_window or self,
            preferences=self.preferences,
        )
        logical = metrics.logical_geometry
        physical = metrics.estimated_physical_size
        target = self._selected_target_resolution()
        actual_text = f"{logical.width()} x {logical.height()}"
        target_text = "Auto" if target is None else f"{target.width()} x {target.height()}"

        self.screen_name_label.setText(metrics.name)
        self.logical_size_label.setText(f"{logical.width()} x {logical.height()}")
        self.dpr_label.setText(f"{metrics.device_pixel_ratio:.2f}")
        self.physical_size_label.setText(f"{physical.width()} x {physical.height()}")
        scaling = max(metrics.device_pixel_ratio, metrics.dpi_scale)
        self.scaling_label.setText(f"{int(round(scaling * 100))}%")
        self.actual_screen_label.setText(actual_text)
        self.layout_target_label.setText(target_text)
        self.effective_scale_label.setText(f"{self._selected_effective_scale()}%")
        self.profile_label.setText(profile.label)
        warning = self._selected_target_warning()
        self.warning_label.setText(warning)
        self.warning_label.setVisible(bool(warning))

    def _selected_profile(self):
        mode = self.override_combo.currentData()
        if not self.adaptive_layout_cb.isChecked() and mode in PROFILES:
            return PROFILES[mode]
        target = self._selected_target_resolution()
        if target is not None:
            from src.gimap.app.presentation.responsive_layout import profile_key_for_geometry
            from PyQt5.QtCore import QRect
            return PROFILES[profile_key_for_geometry(QRect(0, 0, target.width(), target.height()))]
        return current_profile(self, preferences=self.preferences)

    def _selected_target_resolution(self):
        mode = self.layout_target_combo.currentData()
        if mode in ("compact", "normal", "wide"):
            return LAYOUT_TARGETS[mode].resolution
        if mode == "custom":
            return parse_resolution(self.custom_target_combo.currentText())
        return None

    def _selected_target_text(self) -> str:
        target = self._selected_target_resolution()
        if target is None:
            return "auto"
        return f"{target.width()}x{target.height()}"

    def _selected_effective_scale(self) -> int:
        target = self._selected_target_resolution()
        if target is None or not self.auto_fit_cb.isChecked():
            return int(self.ui_scale_slider.value())
        metrics = screen_metrics(
            self.parent_window or self,
            preferences=self.preferences,
        )
        logical = metrics.logical_geometry
        fit_scale = min(
            logical.width() / max(1, target.width()),
            logical.height() / max(1, target.height()),
        )
        auto_fit_scale = max(0.75, min(1.0, fit_scale))
        fitted_scale = int(round(int(self.ui_scale_slider.value()) * auto_fit_scale))
        return max(75, min(140, fitted_scale))

    def _selected_target_warning(self) -> str:
        target = self._selected_target_resolution()
        if target is None:
            return ""
        logical = screen_metrics(
            self.parent_window or self,
            preferences=self.preferences,
        ).logical_geometry
        if target.width() > logical.width() or target.height() > logical.height():
            return (
                "The selected layout target is larger than the current screen. "
                "GIMaP will use a smaller UI scale and scrolling to keep the interface usable."
            )
        return ""

    def _apply_font(self) -> None:
        if self.parent_window is None:
            return
        scale = self._selected_effective_scale()
        base_size = 9.0
        font = QFont(self.parent_window.font())
        font.setPointSizeF(max(4.0, base_size * scale / 100.0))
        app = QApplication.instance()
        if app is not None:
            app.setFont(font)
        apply_main_window_styles(self.parent_window)

    def _save_settings(self) -> None:
        adaptive = self.adaptive_layout_cb.isChecked()
        mode = "auto" if adaptive else self.override_combo.currentData()
        self.preferences.set("auto_detect_monitor_dpi", self.auto_dpi_cb.isChecked())
        self.preferences.set("adaptive_layout_enabled", adaptive)
        self.preferences.set("responsive_layout_mode", mode)
        target_mode = str(self.layout_target_combo.currentData() or "auto")
        self.preferences.set("layout_target_mode", target_mode)
        self.preferences.set("layout_target_custom", self.custom_target_combo.currentText().strip())
        self.preferences.set("auto_fit_layout_target", self.auto_fit_cb.isChecked())
        self.preferences.set("manual_screen_resolution", self._selected_target_text())
        self.preferences.set("enable_adaptive_scaling", True)
        self.preferences.set("visual_font_scale", self.ui_scale_slider.value())
        self.preferences.save()

    def apply_settings(self) -> None:
        self._save_settings()
        profile = current_profile(
            self.parent_window or self,
            preferences=self.preferences,
        )
        self._apply_font()
        if self.parent_window is not None:
            apply_window_profile(
                self.parent_window,
                profile,
                resize_window=False,
                preferences=self.preferences,
            )
            if hasattr(self.parent_window, "components"):
                self.parent_window.components.apply_responsive_profile(profile)
            apply_main_window_styles(self.parent_window)
            apply_density_profile(
                self.parent_window,
                profile,
                preferences=self.preferences,
            )
        apply_main_window_styles(self)
        apply_density_profile(
            self,
            current_profile(self, preferences=self.preferences),
            preferences=self.preferences,
        )
        self._refresh_summary()
        QMessageBox.information(self, "Settings Applied", "Display settings have been applied.")

    def accept_settings(self) -> None:
        self.apply_settings()
        self.accept()

    def _on_dialog_screen_changed(self, profile, screen) -> None:
        self._refresh_summary()
