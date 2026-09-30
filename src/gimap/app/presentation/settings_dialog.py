"""Settings dialog: appearance, Analyze defaults and the user data folder.

Every change is applied and saved at once; there is no Apply/Cancel step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QDialog, QMessageBox, QWidget

from src.gimap.app.ports import SettingsRepository, UserPreferencesRepository

from .theme import FONT_PT_RANGE
from .theme.appearance import Appearance
from .views import SettingsDialogView

ANALYZE_SECTION = "analyze"
HEADER_CENTER_KEY = "use_header_beam_center"
FIT_SIDE_KEY = "fit_side"
"""Analyze options live in the ``analyze`` settings section, next to the
choices the Analyze page remembers itself."""
FIT_SIDES = (
    ("both_abs", "Both halves as |qy| (two colours)"),
    ("mean", "Symmetric average of both halves"),
    ("negative", "qy < 0 only"),
    ("positive", "qy > 0 only"),
)


class SettingsDialog(QDialog, SettingsDialogView):
    def __init__(
        self,
        parent=None,
        *,
        preferences: UserPreferencesRepository,
        settings: Optional[SettingsRepository] = None,
        data_dir: str | Path | None = None,
        migrated_from: Optional[dict] = None,
        extra_pages: Sequence[tuple[str, str, Callable[[QWidget], QWidget]]] = (),
    ):
        super().__init__(parent)
        self.preferences = preferences
        self.settings = settings
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.appearance = Appearance(preferences)
        self.setupUi(self)
        for title, description, create in extra_pages:
            page, layout = self.add_category(title, description)
            layout.addWidget(create(page))
        self._load(migrated_from)
        self._connect()

    def _load(self, migrated_from: Optional[dict]) -> None:
        dark = self.appearance.mode == "dark"
        self.dark_radio.setChecked(dark)
        self.light_radio.setChecked(not dark)
        low, high = FONT_PT_RANGE
        self.font_size_spin.setRange(low, high)
        self.font_size_spin.setValue(self.appearance.font_pt)
        from .i18n import LANGUAGES

        for key, title in LANGUAGES.items():
            self.language_combo.addItem(title, key)
        self.language_combo.setCurrentIndex(max(0, self.language_combo.findData(self.appearance.language)))
        for key, title in FIT_SIDES:
            self.fit_side_combo.addItem(title, key)
        if self.settings is not None:
            self.header_center_check.setChecked(
                bool(self.settings.get(ANALYZE_SECTION, HEADER_CENTER_KEY, False))
            )
            index = self.fit_side_combo.findData(
                self.settings.get(ANALYZE_SECTION, FIT_SIDE_KEY, "both_abs")
            )
            self.fit_side_combo.setCurrentIndex(max(0, index))
        for widget in (self.header_center_check, self.fit_side_combo):
            widget.setEnabled(self.settings is not None)
        self.data_folder_edit.setText(str(self.data_dir) if self.data_dir else "(not saved: in-memory session)")
        self.open_folder_button.setEnabled(self.data_dir is not None)
        files = (migrated_from or {}).get("files") or []
        self.migration_label.setText(
            "Imported once from the previous version: " + ", ".join(Path(f).name for f in files)
            if files else ""
        )
        self.migration_label.setVisible(bool(files))
        self.reset_button.setEnabled(self.settings is not None)

    def _connect(self) -> None:
        self.light_radio.toggled.connect(lambda on: on and self.appearance.set_theme("light"))
        self.dark_radio.toggled.connect(lambda on: on and self.appearance.set_theme("dark"))
        self.font_size_spin.valueChanged.connect(self.appearance.set_font_pt)
        self.font_reset_button.clicked.connect(self._reset_font)
        self.language_combo.currentIndexChanged.connect(self._language_chosen)
        self.header_center_check.toggled.connect(
            lambda on: self._remember_analyze(HEADER_CENTER_KEY, bool(on))
        )
        self.fit_side_combo.currentIndexChanged.connect(
            lambda _i: self._remember_analyze(FIT_SIDE_KEY, self.fit_side_combo.currentData())
        )
        self.open_folder_button.clicked.connect(self._open_data_folder)
        self.reset_button.clicked.connect(self._reset_all)
        self.close_button.clicked.connect(self.accept)

    def _remember_analyze(self, key: str, value) -> None:
        if self.settings is not None:
            self.settings.set(ANALYZE_SECTION, key, value)
            self.settings.save()

    def _language_chosen(self, _index: int) -> None:
        """Every open window now; windows opened later when they are shown."""
        from PyQt5.QtWidgets import QApplication

        self.appearance.set_language(self.language_combo.currentData(), QApplication.topLevelWidgets())

    def _reset_font(self) -> None:
        self.appearance.reset_font()
        self.font_size_spin.blockSignals(True)
        self.font_size_spin.setValue(self.appearance.font_pt)
        self.font_size_spin.blockSignals(False)

    def _open_data_folder(self) -> None:
        if self.data_dir is not None:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.data_dir)))

    def _reset_all(self) -> None:
        answer = QMessageBox.question(
            self,
            "Reset All Settings",
            "Reset every setting and preference to its default?\n\n"
            "Instrument profiles are kept. Restart GIMaP afterwards so every "
            "workspace reloads the defaults.",
            QMessageBox.Reset | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer != QMessageBox.Reset:
            return
        self.settings.reset()
        self.preferences.reset()
        self.settings.save()
        self.appearance.apply_saved()
        self.close()


__all__ = ["ANALYZE_SECTION", "FIT_SIDES", "FIT_SIDE_KEY", "HEADER_CENTER_KEY", "SettingsDialog"]
