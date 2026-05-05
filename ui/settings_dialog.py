"""Display settings dialog for screen-aware responsive layout."""

from __future__ import annotations

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
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
    QSpinBox,
    QVBoxLayout,
)

from core.user_settings import user_settings
from ui.responsive_layout import (
    PROFILES,
    apply_window_profile,
    available_screen_geometry,
    current_profile,
    profile_summary,
)


class SettingsDialog(QDialog):
    """Modern Display settings dialog."""

    MODE_LABELS = {
        "auto": "Auto - choose from current screen",
        "compact": "Compact - smaller laptop displays",
        "standard": "Standard - 1366/1440 class displays",
        "spacious": "Spacious - large desktop displays",
        "wide": "Wide - high-resolution workstations",
        "manual": "Manual window size",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Display Settings")
        self.setModal(True)
        self.setMinimumWidth(520)
        self._build_ui()
        self.load_settings()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        self.summary_card = QFrame(self)
        self.summary_card.setProperty("card", True)
        summary_layout = QVBoxLayout(self.summary_card)
        summary_layout.setContentsMargins(14, 12, 14, 12)
        summary_layout.setSpacing(6)
        title = QLabel("Responsive Layout", self.summary_card)
        title.setProperty("cardTitle", True)
        self.screen_label = QLabel(self.summary_card)
        self.screen_label.setProperty("cardBody", True)
        self.profile_label = QLabel(self.summary_card)
        self.profile_label.setProperty("cardMeta", True)
        summary_layout.addWidget(title)
        summary_layout.addWidget(self.screen_label)
        summary_layout.addWidget(self.profile_label)
        root.addWidget(self.summary_card)

        layout_group = QGroupBox("Layout Breakpoint", self)
        layout_form = QFormLayout(layout_group)
        layout_form.setHorizontalSpacing(14)
        layout_form.setVerticalSpacing(10)

        self.mode_combo = QComboBox(layout_group)
        for key, label in self.MODE_LABELS.items():
            self.mode_combo.addItem(label, key)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout_form.addRow("Mode:", self.mode_combo)

        self.resize_on_start_cb = QCheckBox("Resize window to the selected profile on startup", layout_group)
        layout_form.addRow("", self.resize_on_start_cb)

        custom_row = QHBoxLayout()
        self.width_spin = QSpinBox(layout_group)
        self.width_spin.setRange(900, 3200)
        self.width_spin.setSuffix(" px")
        self.height_spin = QSpinBox(layout_group)
        self.height_spin.setRange(640, 2200)
        self.height_spin.setSuffix(" px")
        custom_row.addWidget(self.width_spin)
        custom_row.addWidget(QLabel("x", layout_group))
        custom_row.addWidget(self.height_spin)
        layout_form.addRow("Manual size:", custom_row)
        root.addWidget(layout_group)

        display_group = QGroupBox("Visual Scale", self)
        display_form = QFormLayout(display_group)
        display_form.setHorizontalSpacing(14)
        display_form.setVerticalSpacing(10)

        self.adaptive_font_cb = QCheckBox("Use breakpoint font recommendation", display_group)
        display_form.addRow("", self.adaptive_font_cb)

        self.font_spin = QSpinBox(display_group)
        self.font_spin.setRange(-3, 3)
        self.font_spin.setSuffix(" levels")
        display_form.addRow("Manual font adjustment:", self.font_spin)
        root.addWidget(display_group)

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
        mode = user_settings.get("responsive_layout_mode", "auto")
        index = self.mode_combo.findData(mode)
        self.mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.resize_on_start_cb.setChecked(user_settings.get("responsive_resize_on_start", True))
        self.adaptive_font_cb.setChecked(user_settings.get("responsive_font_enabled", True))
        self.font_spin.setValue(user_settings.get_font_adjustment())

        if self.parent_window is not None:
            size = self.parent_window.size()
            self.width_spin.setValue(size.width())
            self.height_spin.setValue(size.height())
        else:
            width, height = user_settings.get_window_size()
            self.width_spin.setValue(width)
            self.height_spin.setValue(height)
        self._on_mode_changed()

    def _on_mode_changed(self) -> None:
        manual = self.mode_combo.currentData() == "manual"
        self.width_spin.setEnabled(manual)
        self.height_spin.setEnabled(manual)
        geometry = available_screen_geometry(self.parent_window)
        profile = self._selected_profile()
        self.screen_label.setText(f"Available screen: {geometry.width()} x {geometry.height()} px")
        self.profile_label.setText(profile_summary(profile, geometry))

    def _selected_profile(self):
        mode = self.mode_combo.currentData()
        if mode in PROFILES:
            return PROFILES[mode]
        return current_profile(self.parent_window)

    def _apply_font(self, profile) -> None:
        if self.parent_window is None:
            return
        base_size = 9
        adjustment = profile.font_adjustment if self.adaptive_font_cb.isChecked() else self.font_spin.value()
        font = QFont(self.parent_window.font())
        font.setPointSize(max(8, base_size + adjustment))
        self.parent_window.setFont(font)
        user_settings.set_font_adjustment(adjustment)

    def _save_settings(self) -> None:
        mode = self.mode_combo.currentData()
        user_settings.set("responsive_layout_mode", mode)
        user_settings.set("responsive_resize_on_start", self.resize_on_start_cb.isChecked())
        user_settings.set("responsive_font_enabled", self.adaptive_font_cb.isChecked())
        user_settings.enable_adaptive_scaling(mode != "manual")
        user_settings.set_window_size(self.width_spin.value(), self.height_spin.value())
        user_settings.set_font_adjustment(self.font_spin.value())
        user_settings.save_settings()

    def apply_settings(self) -> None:
        self._save_settings()
        profile = self._selected_profile()
        if self.parent_window is not None:
            if self.mode_combo.currentData() == "manual":
                self.parent_window.setMinimumSize(900, 640)
                self.parent_window.resize(self.width_spin.value(), self.height_spin.value())
            else:
                apply_window_profile(self.parent_window, profile)
            self._apply_font(profile)
            if hasattr(self.parent_window, "components"):
                self.parent_window.components.apply_responsive_profile(profile)
        self._on_mode_changed()
        QMessageBox.information(self, "Settings Applied", "Responsive display settings have been applied.")

    def accept_settings(self) -> None:
        self.apply_settings()
        self.accept()
