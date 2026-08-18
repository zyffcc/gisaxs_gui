"""Display Settings Python View ownership tests."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation.settings_dialog import SettingsDialog
from src.gimap.app.presentation.views import SettingsDialogView
from src.gimap.integrations.state import InMemoryUserPreferencesRepository
from ui.settings_dialog import SettingsDialog as LegacySettingsDialog


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TEST_APP = None


def _app() -> QApplication:
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


def test_settings_dialog_uses_python_view_and_legacy_reexport() -> None:
    app = _app()
    dialog = SettingsDialog(preferences=InMemoryUserPreferencesRepository())

    assert LegacySettingsDialog is SettingsDialog
    assert isinstance(dialog, SettingsDialogView)
    assert dialog.objectName() == "SettingsDialog"
    assert dialog.displaySettingsContent.objectName() == "displaySettingsContent"
    assert dialog.ui_scale_slider.minimum() == 40
    assert dialog.ui_scale_slider.maximum() == 140
    assert dialog.layout_target_combo.count() == 5

    source = (
        PROJECT_ROOT / "src/gimap/app/presentation/settings_dialog.py"
    ).read_text(encoding="utf-8")
    assert "def _build_ui(" not in source

    dialog.close()
    app.processEvents()
