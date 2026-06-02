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
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.user_settings import user_settings
from ui.responsive_layout import (
    PROFILES,
    apply_density_profile,
    apply_window_profile,
    available_screen_geometry,
    current_profile,
    install_adaptive_window_profile,
    profile_key_for_geometry,
    screen_summary,
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
        self.setMinimumWidth(500)
        self._build_ui()
        self.load_settings()
        install_adaptive_window_profile(self, self._on_dialog_screen_changed, apply_window_minimum=False)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        self.summary_card = QFrame(self)
        self.summary_card.setProperty("card", True)
        summary_layout = QFormLayout(self.summary_card)
        summary_layout.setContentsMargins(14, 12, 14, 12)
        summary_layout.setHorizontalSpacing(14)
        summary_layout.setVerticalSpacing(8)

        title = QLabel("Display Settings", self.summary_card)
        title.setProperty("cardTitle", True)
        summary_layout.addRow(title)

        self.screen_label = QLabel(self.summary_card)
        self.screen_label.setProperty("cardBody", True)
        self.profile_label = QLabel(self.summary_card)
        self.profile_label.setProperty("cardBody", True)
        summary_layout.addRow("Current screen:", self.screen_label)
        summary_layout.addRow("Current profile:", self.profile_label)
        root.addWidget(self.summary_card)

        scale_group = QGroupBox("UI Scale", self)
        scale_form = QFormLayout(scale_group)
        scale_form.setHorizontalSpacing(14)
        scale_form.setVerticalSpacing(10)

        scale_row = QHBoxLayout()
        self.ui_scale_slider = QSlider(Qt.Horizontal, scale_group)
        self.ui_scale_slider.setRange(80, 140)
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
        layout_form = QFormLayout(layout_group)
        layout_form.setHorizontalSpacing(14)
        layout_form.setVerticalSpacing(10)

        self.auto_dpi_cb = QCheckBox("Auto detect monitor DPI", layout_group)
        self.adaptive_layout_cb = QCheckBox("Adaptive layout", layout_group)
        self.adaptive_layout_cb.toggled.connect(self._on_adaptive_toggled)
        layout_form.addRow("", self.auto_dpi_cb)
        layout_form.addRow("", self.adaptive_layout_cb)

        self.override_combo = QComboBox(layout_group)
        self.override_combo.addItem("Auto", "auto")
        for key, label in self.PROFILE_LABELS.items():
            self.override_combo.addItem(label, key)
        self.override_combo.currentIndexChanged.connect(self._refresh_summary)
        layout_form.addRow("Advanced override:", self.override_combo)
        root.addWidget(layout_group)

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
        root.addLayout(button_row)

        for button in (self.apply_button, self.cancel_button, self.ok_button):
            button.setMinimumHeight(32)
            button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

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
        self.ui_scale_slider.setValue(user_settings.get_visual_font_scale())
        self._on_ui_scale_changed(self.ui_scale_slider.value())
        self._on_adaptive_toggled(adaptive)
        self._refresh_summary()

    def _on_adaptive_toggled(self, enabled: bool) -> None:
        self.override_combo.setEnabled(not enabled)
        if enabled:
            auto_index = self.override_combo.findData("auto")
            self.override_combo.setCurrentIndex(auto_index)
        self._refresh_summary()

    def _on_ui_scale_changed(self, value: int) -> None:
        self.ui_scale_label.setText(f"{value}%")

    def _refresh_summary(self) -> None:
        profile = self._selected_profile()
        self.screen_label.setText(screen_summary(self))
        self.profile_label.setText(profile.label)

    def _selected_profile(self):
        mode = self.override_combo.currentData()
        if not self.adaptive_layout_cb.isChecked() and mode in PROFILES:
            return PROFILES[mode]
        return PROFILES[profile_key_for_geometry(available_screen_geometry(self))]

    def _apply_font(self) -> None:
        if self.parent_window is None:
            return
        scale = self.ui_scale_slider.value()
        base_size = 9.0
        font = QFont(self.parent_window.font())
        font.setPointSizeF(max(7.0, base_size * scale / 100.0))
        app = QApplication.instance()
        if app is not None:
            app.setFont(font)
        self.parent_window.setFont(font)
        for widget in self.parent_window.findChildren(QWidget):
            widget.setFont(QFont(font))
        user_settings.set_visual_font_scale(scale)
        apply_main_window_styles(self.parent_window)

    def _save_settings(self) -> None:
        adaptive = self.adaptive_layout_cb.isChecked()
        mode = "auto" if adaptive else self.override_combo.currentData()
        user_settings.set("auto_detect_monitor_dpi", self.auto_dpi_cb.isChecked())
        user_settings.set("adaptive_layout_enabled", adaptive)
        user_settings.set("responsive_layout_mode", mode)
        user_settings.enable_adaptive_scaling(True)
        user_settings.set_visual_font_scale(self.ui_scale_slider.value())
        user_settings.save_settings()

    def apply_settings(self) -> None:
        self._save_settings()
        profile = self._selected_profile()
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
