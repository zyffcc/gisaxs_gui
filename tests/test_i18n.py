"""The interface language: English by default, Chinese on request, and back to English exactly."""

from __future__ import annotations

import time

from PyQt5.QtWidgets import QApplication, QPushButton, QToolButton

from src.gimap.app.presentation import i18n
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


def _keys(monkeypatch, entries: dict) -> None:
    """Entries the integrator adds to the tables (zh_shell.py); patched in only while they are missing."""
    for english, chinese in entries.items():
        if english not in i18n.ZH:
            monkeypatch.setitem(i18n.ZH, english, chinese)
            monkeypatch.setitem(i18n._TO_ENGLISH, chinese, english)


def test_table_and_tree_headers_are_translated_and_come_back() -> None:
    from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QTreeWidget, QWidget

    from src.gimap.app.presentation.i18n import ZH as TABLE
    from src.gimap.app.presentation.i18n import apply_to

    _app()
    root = QWidget()
    table = QTableWidget(2, 3, root)
    table.setHorizontalHeaderLabels(["Source", "q (Å⁻¹)", "Status"])
    table.setVerticalHeaderItem(0, QTableWidgetItem("Source"))
    tree = QTreeWidget(root)
    tree.setHeaderLabels(["Source", "Type"])
    apply_to(root, "zh")
    assert table.horizontalHeaderItem(0).text() == "来源" == TABLE["Source"]
    assert table.horizontalHeaderItem(1).text() == "q (Å⁻¹)"  # units are never translated
    assert table.verticalHeaderItem(0).text() == "来源" and table.verticalHeaderItem(1) is None
    assert tree.headerItem().text(0) == "来源"
    apply_to(root, DEFAULT_LANGUAGE)
    assert [table.horizontalHeaderItem(column).text() for column in range(3)] == ["Source", "q (Å⁻¹)", "Status"]
    assert table.verticalHeaderItem(0).text() == "Source" and tree.headerItem().text(0) == "Source"


def test_headers_that_are_data_are_never_translated() -> None:
    from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QTreeWidget, QWidget

    from src.gimap.app.presentation.i18n import DATA_HEADERS, apply_to

    _app()
    root = QWidget()
    table = QTableWidget(2, 3, root)  # e.g. Compare's distances: one column per series, named by the user
    table.setProperty(DATA_HEADERS, True)
    table.setHorizontalHeaderLabels(["", "Source", "Status"])
    table.setVerticalHeaderItem(0, QTableWidgetItem("Source"))
    tree = QTreeWidget(root)
    tree.setProperty(DATA_HEADERS, True)
    tree.setHeaderLabels(["Source", "Type"])
    for language in ("zh", DEFAULT_LANGUAGE):
        apply_to(root, language)
        assert [table.horizontalHeaderItem(column).text() for column in range(3)] == ["", "Source", "Status"]
        assert table.verticalHeaderItem(0).text() == "Source" and tree.headerItem().text(0) == "Source"


def test_spin_box_prefix_suffix_and_special_text_keep_their_padding(monkeypatch) -> None:
    from PyQt5.QtWidgets import QDoubleSpinBox, QSpinBox, QWidget

    from src.gimap.app.presentation.i18n import apply_to

    _app()
    _keys(monkeypatch, {"last": "最后", "frames": "帧", "from profile": "来自配置", "every": "每"})
    root = QWidget()
    spin = QSpinBox(root)
    spin.setRange(0, 100)
    spin.setPrefix("last ")
    spin.setSuffix(" frames")
    spin.setSpecialValueText("from profile")
    angle = QDoubleSpinBox(root)
    angle.setSuffix(" °")
    apply_to(root, "zh")
    assert spin.prefix() == i18n.ZH["last"] + " " and spin.suffix() == " " + i18n.ZH["frames"]
    assert spin.specialValueText() == i18n.ZH["from profile"] and spin.text() == i18n.ZH["from profile"]
    spin.setValue(10)
    assert spin.text() == f"{i18n.ZH['last']} 10 {i18n.ZH['frames']}" and spin.value() == 10
    assert angle.suffix() == " °"
    apply_to(root, DEFAULT_LANGUAGE)
    assert (spin.prefix(), spin.suffix(), spin.specialValueText()) == ("last ", " frames", "from profile")
    assert spin.text() == "last 10 frames"


def test_trf_fills_in_the_translated_template() -> None:
    from src.gimap.app.presentation.i18n import trf

    assert trf("{count} curves listed.", count=3) == "3 curves listed."
    try:
        apply_language("zh")
        assert trf("{count} curves listed.", count=3) == ZH["{count} curves listed."].format(count=3) == "列出了 3 条曲线。"
        assert trf("not in the table: {value}", value=1.5) == "not in the table: 1.5"
    finally:
        apply_language(DEFAULT_LANGUAGE)


def test_a_drop_down_triangle_is_one_the_chinese_fonts_have() -> None:
    assert ZH["Add Particle ▾"].endswith("▾")  # the table keeps its text …
    assert translate("Add Particle ▾", "zh") == "添加颗粒 ▼"  # … shown with a triangle the fonts draw
    assert translate("添加颗粒 ▼", "en") == "Add Particle ▾"
    assert all("▾" not in value and "▸" not in value for value in i18n.ZH.values())


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
