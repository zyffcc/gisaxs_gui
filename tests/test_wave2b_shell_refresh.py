"""After a real switch of the interface language the shell's own run-time texts follow (wave 2b, shell check):
the Start page's recent rows, the Labs status bar message, the texts Settings composes, and the menus' sync
once their window is gone."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QStatusBar

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language


def _app():
    return QApplication.instance() or QApplication([])


def _rows(page) -> list[str]:
    return [button.text() for button in page.recent_box.findChildren(QPushButton, "homeRecentItem")
            if not button.isHidden()]


def test_the_start_pages_recent_rows_follow_a_language_switch(tmp_path) -> None:
    from src.gimap.app.presentation.home_page import HomePage

    _app()
    project = tmp_path / "run.gimap"
    project.write_text("{}", encoding="utf-8")
    folder = tmp_path / "frames"
    folder.mkdir()
    page = HomePage()
    page.set_recent_provider(lambda: [str(project), str(folder)])
    try:
        assert _rows(page) == [f"run.gimap — {tmp_path.name} (project)", f"frames — {tmp_path.name} (folder)"]
        apply_language("zh", [page])  # the walker alone leaves the composed rows as they were
        page.refresh_language()  # what the shell calls after the switch, on the page shown
        assert _rows(page) == [f"run.gimap — {tmp_path.name}" + i18n.ZH[" (project)"],
                               f"frames — {tmp_path.name}" + i18n.ZH[" (folder)"]]
        apply_language(DEFAULT_LANGUAGE, [page])
        page.refresh_language()
        assert _rows(page) == [f"run.gimap — {tmp_path.name} (project)", f"frames — {tmp_path.name} (folder)"]
    finally:
        apply_language(DEFAULT_LANGUAGE, [page])
        page.close()


def test_the_labs_status_message_shown_follows_a_language_switch() -> None:
    from src.gimap.app.main_window import MainWindowComponents

    _app()
    bar = QStatusBar()
    components = SimpleNamespace(ui=SimpleNamespace(statusbar=bar))
    try:
        bar.showMessage("Trainset generation started...")
        apply_language("zh", [])
        MainWindowComponents._status_bar_language(components)
        assert bar.currentMessage() == i18n.ZH["Trainset generation started..."]
        apply_language(DEFAULT_LANGUAGE, [])
        MainWindowComponents._status_bar_language(components)
        assert bar.currentMessage() == "Trainset generation started..."
        bar.showMessage("frame_00012.cbf")  # not in the table: left as it is
        apply_language("zh", [])
        MainWindowComponents._status_bar_language(components)
        assert bar.currentMessage() == "frame_00012.cbf"
        bar.deleteLater()
        QApplication.sendPostedEvents(None, 52)  # QEvent.DeferredDelete: a window that is gone is no error
        MainWindowComponents._status_bar_language(components)
    finally:
        apply_language(DEFAULT_LANGUAGE, [])


def test_settings_composes_its_own_texts_again_when_the_language_is_chosen_there() -> None:
    from src.gimap.app.presentation.settings_dialog import SettingsDialog
    from src.gimap.integrations.state import InMemorySettingsRepository, InMemoryUserPreferencesRepository

    _app()
    dialog = SettingsDialog(preferences=InMemoryUserPreferencesRepository(), settings=InMemorySettingsRepository(),
                            migrated_from={"files": ["C:/old/settings.json"]})
    try:
        assert dialog.data_folder_edit.text() == "(not saved: in-memory session)"
        assert dialog.migration_label.text() == "Imported once from the previous version: settings.json"
        dialog.language_combo.setCurrentIndex(dialog.language_combo.findData("zh"))
        assert dialog.data_folder_edit.text() == i18n.ZH["(not saved: in-memory session)"]
        assert dialog.migration_label.text() == i18n.ZH["Imported once from the previous version: {files}"].format(
            files="settings.json")
        dialog.language_combo.setCurrentIndex(dialog.language_combo.findData("en"))
        assert dialog.data_folder_edit.text() == "(not saved: in-memory session)"
        assert dialog.migration_label.text() == "Imported once from the previous version: settings.json"
    finally:
        apply_language(DEFAULT_LANGUAGE, [])
        dialog.close()


def test_the_menus_ignore_a_theme_or_language_change_after_their_window_is_gone() -> None:
    from src.gimap.app.menus import ApplicationMenus
    from src.gimap.app.presentation.menu_bar import MainMenuBar, MenuCommands

    _app()
    host = QMainWindow()
    bar = MainMenuBar(host, MenuCommands(set_theme=lambda _mode: None, set_language=lambda _key: None))
    menus = SimpleNamespace(menu_bar=bar)
    ApplicationMenus._sync_menus(menus, "dark")  # a window that is there: synced
    host.deleteLater()
    QApplication.sendPostedEvents(None, 52)
    ApplicationMenus._sync_menus(menus, "zh")  # theme_manager().changed and language_changed() outlive it
