"""Settings opens wide enough for its widest page (no sideways scrolling where the screen allows) and shows the
whole user data folder on hover (wave 2b, the shell's visual pass)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from src.gimap.app.presentation.settings_dialog import FIRST_SIZE, SettingsDialog
from src.gimap.integrations.state import InMemorySettingsRepository, InMemoryUserPreferencesRepository


def _app():
    return QApplication.instance() or QApplication([])


def _wide_page(parent):
    page = QWidget(parent)
    layout = QVBoxLayout(page)
    label = QLabel("a page that needs room", page)
    label.setMinimumWidth(900)
    layout.addWidget(label)
    return page


def test_settings_opens_wide_enough_for_its_widest_page(tmp_path) -> None:
    _app()
    plain = SettingsDialog(preferences=InMemoryUserPreferencesRepository(), settings=InMemorySettingsRepository(),
                           data_dir=tmp_path)
    wide = SettingsDialog(preferences=InMemoryUserPreferencesRepository(), settings=InMemorySettingsRepository(),
                          extra_pages=(("Wide", "", _wide_page),))
    try:
        assert plain._first_size() == FIRST_SIZE  # the built-in pages fit
        size = wide._first_size()
        assert size.height() == FIRST_SIZE.height() and size.width() > FIRST_SIZE.width()
        wide.resize(size)  # as on a screen with the room (this offscreen screen is 800 px wide)
        wide.category_list.setCurrentRow(wide.category_list.count() - 1)
        wide.show()
        for _ in range(5):
            QApplication.processEvents()
        scroll = wide.pages.currentWidget()
        assert not scroll.horizontalScrollBar().isVisible()
        assert wide.first_size.width() <= QApplication.primaryScreen().availableGeometry().width()  # on screen
        assert plain.data_folder_edit.toolTip() == str(tmp_path)
    finally:
        plain.close()
        wide.close()
