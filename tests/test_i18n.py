"""The interface language: English by default, Chinese on request, and back to English exactly."""

from __future__ import annotations

import time

from PyQt5.QtWidgets import QApplication, QPushButton, QToolButton

from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language, current_language, translate
from src.gimap.app.presentation.i18n.zh import ZH
from tests.test_assistant_gui import _app


def _window(preferences=None):
    from main import MainWindow
    from src.gimap.app import AppContext
    from src.gimap.integrations.jobs import LocalProcessJobRunner
    from src.gimap.integrations.state import (
        InMemoryInstrumentProfileRepository,
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )

    context = AppContext(
        settings=InMemorySettingsRepository(), session=InMemorySessionRepository(),
        preferences=preferences or InMemoryUserPreferencesRepository(), jobs=LocalProcessJobRunner(),
        instrument_profiles=InMemoryInstrumentProfileRepository([]),
    )
    window = MainWindow(context)
    window.resize(1700, 1000)  # wide enough for the full command-bar texts
    window.show()
    deadline = time.monotonic() + 30
    while not window._initialization_completed and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.02)
    return window


def _texts(window) -> set[str]:
    return {button.text().strip() for button in window.findChildren(QPushButton)} | {
        button.text().strip() for button in window.findChildren(QToolButton)
    }


def test_the_table_translates_words_not_units() -> None:
    assert all(isinstance(key, str) and key and value and key != value for key, value in ZH.items())
    assert translate("Run Automatic Analysis", "zh") == "运行自动分析"
    assert translate("运行自动分析", "en") == "Run Automatic Analysis"
    assert translate("q (Å⁻¹)", "zh") is None  # units and values are never in the table


def test_chinese_is_applied_at_start_and_english_comes_back_exactly() -> None:
    from src.gimap.app.presentation.theme.appearance import Appearance
    from src.gimap.integrations.state import InMemoryUserPreferencesRepository

    _app()
    preferences = InMemoryUserPreferencesRepository()
    Appearance(preferences).set_language("zh")
    window = _window(preferences)
    try:
        page = window.components.analyze_page
        window.components.show_page("analyze")  # a hidden page is laid out when it is shown
        QApplication.processEvents()
        assert current_language() == "zh"
        assert page.run_pipeline_button.text() == "运行自动分析" and page.fit_button.text() == "发送到拟合"
        assert "分析" in _texts(window) and "拟合" in _texts(window)
        Appearance(preferences).set_language("en", [window])
        assert current_language() == DEFAULT_LANGUAGE
        assert page.run_pipeline_button.text() == "Run Automatic Analysis" and page.fit_button.text() == "Send to Fitting"
        texts = _texts(window)
        assert "Fitting" in texts and "Fit" not in {text for text in texts if text == "拟合"}  # shared Chinese, exact English
        assert not any("一" <= character <= "鿿" for text in texts for character in text)
    finally:
        apply_language(DEFAULT_LANGUAGE)
        window.components.analyze_page.dispose()
        window.close()
