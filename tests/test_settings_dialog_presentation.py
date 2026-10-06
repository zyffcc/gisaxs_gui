"""Settings dialog: appearance applies live; Analyze defaults are remembered."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation.settings_dialog import (
    ANALYZE_SECTION,
    FIT_SIDE_KEY,
    HEADER_CENTER_KEY,
    SettingsDialog,
)
from src.gimap.app.presentation.theme import theme_manager
from src.gimap.app.presentation.views import SettingsDialogView
from src.gimap.integrations.state import (
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)

_TEST_APP = None


def _app() -> QApplication:
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


def test_settings_dialog_uses_python_view_with_three_categories() -> None:
    _app()
    dialog = SettingsDialog(preferences=InMemoryUserPreferencesRepository())

    assert isinstance(dialog, SettingsDialogView)
    assert dialog.objectName() == "SettingsDialog"
    assert [dialog.category_list.item(i).text() for i in range(dialog.category_list.count())] == [
        "Appearance",
        "Analyze",
        "Data",
    ]
    assert dialog.pages.count() == 3
    # The header beam centre is ignored unless the user opts in.
    assert dialog.header_center_check.isChecked() is False
    assert dialog.fit_side_combo.currentData() == "both_abs"
    # Without a settings repository there is nothing to reset or to change.
    assert dialog.reset_button.isEnabled() is False
    assert dialog.header_center_check.isEnabled() is False
    dialog.close()


def test_appearance_changes_apply_immediately_and_are_saved() -> None:
    _app()
    preferences = InMemoryUserPreferencesRepository()
    settings = InMemorySettingsRepository()
    dialog = SettingsDialog(preferences=preferences, settings=settings)
    try:
        dialog.dark_radio.setChecked(True)
        assert theme_manager().mode == "dark"
        assert preferences.get("appearance.theme") == "dark"
        dialog.font_size_spin.setValue(11)
        assert theme_manager().font_pt == 11
        assert preferences.get("appearance.font_pt") == 11
        dialog.header_center_check.setChecked(True)
        assert settings.get(ANALYZE_SECTION, HEADER_CENTER_KEY) is True
        dialog.fit_side_combo.setCurrentIndex(dialog.fit_side_combo.findData("mean"))
        assert settings.get(ANALYZE_SECTION, FIT_SIDE_KEY) == "mean"
    finally:
        dialog.light_radio.setChecked(True)
        dialog._reset_font()
        dialog.close()
    assert theme_manager().mode == "light"
    assert theme_manager().font_pt == 9


def test_the_halves_read_as_in_analyze() -> None:
    from src.gimap.app.presentation.settings_dialog import FIT_SIDES
    from src.gimap.features.analyze.presentation.views.analyze_steps_view import FIT_SIDE_ITEMS

    assert tuple(FIT_SIDES) == tuple(FIT_SIDE_ITEMS)


def test_every_page_scrolls_and_the_dialog_opens_at_a_usable_size(tmp_path) -> None:
    from PyQt5.QtCore import QSize
    from PyQt5.QtWidgets import QGroupBox, QScrollArea

    from src.gimap.app.presentation.layout_metrics import available_geometry
    from src.gimap.app.presentation.task_runner import TaskRunner
    from src.gimap.features.assistant.presentation.settings_page import AssistantSettingsPage
    from tests.test_assistant_gui import _services

    _app()
    settings = InMemorySettingsRepository()
    tasks = TaskRunner()
    dialog = SettingsDialog(
        preferences=InMemoryUserPreferencesRepository(), settings=settings,
        extra_pages=(("Assistant", "The AI runs the Analyze tools on the open GIWAXS frame and reports what it finds.",
                      lambda parent: AssistantSettingsPage(settings, _services(tmp_path, None), tasks, parent)),),
    )
    try:
        assert dialog.pages.count() == dialog.category_list.count() == 4
        for index in range(dialog.pages.count()):
            assert isinstance(dialog.pages.widget(index), QScrollArea), index
        assert dialog.minimumSizeHint().height() <= 560
        area = available_geometry(dialog)
        expected = QSize(max(dialog.minimumWidth(), min(920, area.width() - 48)),
                         max(dialog.minimumHeight(), min(700, area.height() - 48)))
        assert dialog.size() == expected == dialog.first_size
        assert dialog.category_list.objectName() == "settingsCategoryList"
        dialog.show()
        dialog.category_list.setCurrentRow(3)
        for _ in range(5):
            QApplication.processEvents()
        boxes = [box for box in dialog.pages.widget(3).findChildren(QGroupBox) if box.isVisible()]
        assert boxes
        for box in boxes:
            assert box.height() >= box.minimumSizeHint().height(), box.title()
    finally:
        dialog.close()
        tasks.shutdown()
