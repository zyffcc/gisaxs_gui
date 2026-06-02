"""Display settings dialog for adaptive monitor-aware layout."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.user_settings import user_settings
from ui.responsive_layout import (
    LAYOUT_TARGETS,
    PROFILES,
    apply_density_profile,
    apply_window_profile,
    current_profile,
    install_adaptive_window_profile,
    parse_resolution,
    screen_metrics,
)
from ui.style_loader import apply_main_window_styles


class SettingsDialog(QDialog):
    """Modern Display settings dialog."""

    PROFILE_LABELS = {
        "compact": "Compact",
        "normal": "Normal",
        "wide": "Wide",
    }
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Display Settings")
        self.setModal(True)
        self.setMinimumSize(500, 420)
        self._build_ui()
        self.load_settings()
        self._clamp_to_screen()
        install_adaptive_window_profile(self, self._on_dialog_screen_changed, apply_window_minimum=False)

    def _build_ui(self) -> None:
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(12, 12, 12, 12)
        dialog_layout.setSpacing(10)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setMinimumSize(0, 0)
        self.scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        content = QWidget(self.scroll_area)
        content.setObjectName("displaySettingsContent")
        content.setMinimumSize(0, 0)
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        root = QVBoxLayout(content)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(12)

        self.summary_card = QFrame(self)
        self.summary_card.setProperty("card", True)
        self.summary_card.setMinimumSize(0, 0)
        self.summary_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        summary_layout = QFormLayout(self.summary_card)
        summary_layout.setContentsMargins(14, 12, 14, 12)
        summary_layout.setHorizontalSpacing(14)
        summary_layout.setVerticalSpacing(8)

        title = QLabel("Display Settings", self.summary_card)
        title.setProperty("cardTitle", True)
        summary_layout.addRow(title)

        self.screen_name_label = QLabel(self.summary_card)
        self.logical_size_label = QLabel(self.summary_card)
        self.dpr_label = QLabel(self.summary_card)
        self.physical_size_label = QLabel(self.summary_card)
        self.scaling_label = QLabel(self.summary_card)
        self.actual_screen_label = QLabel(self.summary_card)
        self.layout_target_label = QLabel(self.summary_card)
        self.effective_scale_label = QLabel(self.summary_card)
        self.profile_label = QLabel(self.summary_card)
        self.warning_label = QLabel(self.summary_card)
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #b45309;")
        for label in (
            self.screen_name_label,
            self.logical_size_label,
            self.dpr_label,
            self.physical_size_label,
            self.scaling_label,
            self.actual_screen_label,
            self.layout_target_label,
            self.effective_scale_label,
            self.profile_label,
        ):
            label.setProperty("cardBody", True)
        summary_layout.addRow("Current screen:", self.screen_name_label)
        summary_layout.addRow("Logical size:", self.logical_size_label)
        summary_layout.addRow("Device pixel ratio:", self.dpr_label)
        summary_layout.addRow("Estimated physical size:", self.physical_size_label)
        summary_layout.addRow("Windows scaling estimate:", self.scaling_label)
        summary_layout.addRow("Actual screen:", self.actual_screen_label)
        summary_layout.addRow("Layout target:", self.layout_target_label)
        summary_layout.addRow("Effective UI scale:", self.effective_scale_label)
        summary_layout.addRow("Current profile:", self.profile_label)
        summary_layout.addRow("", self.warning_label)
        root.addWidget(self.summary_card)

        scale_group = QGroupBox("UI Scale", self)
        scale_group.setMinimumSize(0, 0)
        scale_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        scale_form = QFormLayout(scale_group)
        scale_form.setHorizontalSpacing(14)
        scale_form.setVerticalSpacing(10)

        scale_row = QHBoxLayout()
        self.ui_scale_slider = QSlider(Qt.Horizontal, scale_group)
        self.ui_scale_slider.setRange(40, 140)
        self.ui_scale_slider.setSingleStep(5)
        self.ui_scale_slider.setPageStep(10)
        self.ui_scale_slider.setTickPosition(QSlider.TicksBelow)
        self.ui_scale_slider.setTickInterval(10)
        self.ui_scale_label = QLabel(scale_group)
        self.ui_scale_label.setMinimumWidth(48)
        self.ui_scale_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.ui_scale_slider.valueChanged.connect(self._on_ui_scale_changed)
        scale_row.addWidget(self.ui_scale_slider, 1)
        scale_row.addWidget(self.ui_scale_label)
        scale_form.addRow("UI Scale:", scale_row)
        root.addWidget(scale_group)

        layout_group = QGroupBox("Adaptive Layout", self)
        layout_group.setMinimumSize(0, 0)
        layout_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout_form = QFormLayout(layout_group)
        layout_form.setHorizontalSpacing(14)
        layout_form.setVerticalSpacing(10)

        self.auto_dpi_cb = QCheckBox("Auto detect monitor DPI", layout_group)
        self.adaptive_layout_cb = QCheckBox("Adaptive layout", layout_group)
        self.adaptive_layout_cb.toggled.connect(self._on_adaptive_toggled)
        layout_form.addRow("", self.auto_dpi_cb)
        layout_form.addRow("", self.adaptive_layout_cb)

        self.layout_target_combo = QComboBox(layout_group)
        for key in ("auto", "compact", "normal", "wide", "custom"):
            self.layout_target_combo.addItem(LAYOUT_TARGETS[key].label, key)
        self.layout_target_combo.currentIndexChanged.connect(self._on_layout_target_changed)
        layout_form.addRow("Layout target:", self.layout_target_combo)

        self.custom_target_combo = QComboBox(layout_group)
        self.custom_target_combo.setEditable(True)
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
        layout_form.addRow("Custom target:", self.custom_target_combo)

        self.auto_fit_cb = QCheckBox("Auto-fit layout target to current screen", layout_group)
        self.auto_fit_cb.toggled.connect(self._refresh_summary)
        layout_form.addRow("", self.auto_fit_cb)

        self.override_combo = QComboBox(layout_group)
        self.override_combo.addItem("Auto", "auto")
        for key, label in self.PROFILE_LABELS.items():
            self.override_combo.addItem(label, key)
        self.override_combo.currentIndexChanged.connect(self._refresh_summary)
        layout_form.addRow("Advanced override:", self.override_combo)
        root.addWidget(layout_group)

        root.addStretch(1)
        self.scroll_area.setWidget(content)
        dialog_layout.addWidget(self.scroll_area, 1)

        button_row = QHBoxLayout()
        self.apply_button = QPushButton("Apply", self)
        self.apply_button.clicked.connect(self.apply_settings)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        self.ok_button = QPushButton("OK", self)
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept_settings)
        button_row.addWidget(self.apply_button)
        button_row.addStretch(1)
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.ok_button)
        dialog_layout.addLayout(button_row, 0)

        for button in (self.apply_button, self.cancel_button, self.ok_button):
            button.setMinimumHeight(32)
            button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def _clamp_to_screen(self) -> None:
        metrics = screen_metrics(self.parent_window or self)
        available = metrics.logical_geometry
        target_width = min(max(500, self.sizeHint().width()), max(500, int(available.width() * 0.92)))
        target_height = min(max(420, self.sizeHint().height()), max(420, int(available.height() * 0.80)))
        self.setMaximumHeight(max(420, int(available.height() * 0.80)))
        self.resize(target_width, target_height)

    def load_settings(self) -> None:
        self.auto_dpi_cb.setChecked(user_settings.get("auto_detect_monitor_dpi", True))
        adaptive = user_settings.get("adaptive_layout_enabled", True)
        self.adaptive_layout_cb.setChecked(adaptive)
        mode = user_settings.get("responsive_layout_mode", "auto")
        if mode == "standard":
            mode = "normal"
        elif mode == "spacious":
            mode = "wide"
        index = self.override_combo.findData(mode)
        self.override_combo.setCurrentIndex(index if index >= 0 else 0)
        target_mode = user_settings.get("layout_target_mode", "")
        if not target_mode:
            target_mode = "custom" if user_settings.get("manual_screen_resolution", "auto") != "auto" else "auto"
        target_index = self.layout_target_combo.findData(target_mode)
        self.layout_target_combo.setCurrentIndex(target_index if target_index >= 0 else 0)
        custom_target = user_settings.get(
            "layout_target_custom",
            user_settings.get("manual_screen_resolution", "1920x1080"),
        )
        if custom_target and custom_target != "auto":
            self.custom_target_combo.setEditText(str(custom_target))
        self.auto_fit_cb.setChecked(user_settings.get("auto_fit_layout_target", True))
        self.ui_scale_slider.setValue(user_settings.get_visual_font_scale())
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
        metrics = screen_metrics(self.parent_window or self)
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
            from ui.responsive_layout import profile_key_for_geometry
            from PyQt5.QtCore import QRect
            return PROFILES[profile_key_for_geometry(QRect(0, 0, target.width(), target.height()))]
        return current_profile(self)

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
        metrics = screen_metrics(self.parent_window or self)
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
        logical = screen_metrics(self.parent_window or self).logical_geometry
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
        self.parent_window.setFont(font)
        for widget in self.parent_window.findChildren(QWidget):
            widget.setFont(QFont(font))
        apply_main_window_styles(self.parent_window)

    def _save_settings(self) -> None:
        adaptive = self.adaptive_layout_cb.isChecked()
        mode = "auto" if adaptive else self.override_combo.currentData()
        user_settings.set("auto_detect_monitor_dpi", self.auto_dpi_cb.isChecked())
        user_settings.set("adaptive_layout_enabled", adaptive)
        user_settings.set("responsive_layout_mode", mode)
        target_mode = str(self.layout_target_combo.currentData() or "auto")
        user_settings.set("layout_target_mode", target_mode)
        user_settings.set("layout_target_custom", self.custom_target_combo.currentText().strip())
        user_settings.set("auto_fit_layout_target", self.auto_fit_cb.isChecked())
        user_settings.set("manual_screen_resolution", self._selected_target_text())
        user_settings.enable_adaptive_scaling(True)
        user_settings.set_visual_font_scale(self.ui_scale_slider.value())
        user_settings.save_settings()

    def apply_settings(self) -> None:
        self._save_settings()
        profile = current_profile(self.parent_window or self)
        self._apply_font()
        if self.parent_window is not None:
            apply_window_profile(self.parent_window, profile, resize_window=False)
            if hasattr(self.parent_window, "components"):
                self.parent_window.components.apply_responsive_profile(profile)
            apply_main_window_styles(self.parent_window)
            apply_density_profile(self.parent_window, profile)
        apply_main_window_styles(self)
        apply_density_profile(self, current_profile(self))
        self._refresh_summary()
        QMessageBox.information(self, "Settings Applied", "Display settings have been applied.")

    def accept_settings(self) -> None:
        self.apply_settings()
        self.accept()

    def _on_dialog_screen_changed(self, profile, screen) -> None:
        self._refresh_summary()
