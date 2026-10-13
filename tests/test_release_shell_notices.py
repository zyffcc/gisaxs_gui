"""Release 0.1.0, the shell's notices and texts: a plain confirmation of the page left is closed when the page
changes (R68 / R22); the tooltips of a folded section and of the Levels button follow the language (R69, R40);
Qt's own “OK” and the menu of every text field are Chinese in Chinese (R35); a .csv table dropped on Fitting
is not sent to a reader that does not take it, and the notice says why (R7 / R56); Convert Current File… gives
the frame to an open converter through the converter's own method (R12, the shell's half)."""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, QTimer
from PyQt5.QtGui import QContextMenuEvent
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QSpinBox,
    QWidget,
)

import main as main_module
from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.components import dismiss_toasts, show_toast, visible_toasts
from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language, apply_to, translate
from tests.test_shell_polish import _drop, _navigate, _settle, _window

QT_TEXTS = {"OK": "确定", "&OK": "确定(&O)", "&Undo": "撤销(&U)", "&Redo": "重做(&R)", "Cu&t": "剪切(&T)",
            "&Copy": "复制(&C)", "&Paste": "粘贴(&P)", "Select All": "全选(&A)",
            # a number field has its own three entries (QAbstractSpinBox)
            "&Select All": "全选(&S)", "&Step up": "增大一步", "Step &down": "减小一步"}
SECTION_TIP = {"Show or hide this section": "显示或隐藏这一部分"}
"""The shell's entries of this release (zh_entries: the integrator merges them; here until they are in)."""


def _keys(monkeypatch, entries: dict) -> None:
    for english, chinese in entries.items():
        if english not in i18n.ZH:
            monkeypatch.setitem(i18n.ZH, english, chinese)
            monkeypatch.setitem(i18n._TO_ENGLISH, chinese, english)


@pytest.fixture
def window():
    shown = _window()
    yield shown
    apply_language(DEFAULT_LANGUAGE, [shown])
    shown.close()
    _settle(0.05)


# -- toasts and the page shown (R68 / R22) ------------------------------------------------------------


def test_dismiss_closes_the_plain_confirmations_only() -> None:
    QApplication.instance() or QApplication([])
    host = QWidget()
    host.resize(900, 600)
    host.show()
    try:
        assert dismiss_toasts(host) == 0  # nothing shown yet
        plain = show_toast(host, "3 series compared: 2 groups.", level="ok")
        undo = show_toast(host, "Removed 2 series.", level="info", action=("Undo", lambda: None))
        warning = show_toast(host, "2 frames were skipped.", level="warning")
        assert dismiss_toasts(host) == 1
        assert visible_toasts(host) == [undo, warning] and plain not in visible_toasts(host)
        error = show_toast(host, "Could not save.", level="error")
        assert dismiss_toasts(host) == 0
        assert visible_toasts(host) == [undo, warning, error]  # read them, or use the button: they stay
        assert dismiss_toasts(host, levels=("warning",)) == 1 and visible_toasts(host) == [undo, error]
        note = show_toast(host, "Opened cut.dat.", level="info")  # a plain note goes as a confirmation does
        assert dismiss_toasts(host) == 1 and note not in visible_toasts(host) and visible_toasts(host) == [undo, error]
    finally:
        host.close()


def test_a_confirmation_of_the_page_left_goes_when_the_page_changes(window) -> None:
    components = window.components
    components.show_page("compare")
    plain = show_toast(window, "3 series compared: 2 groups.", level="ok")
    components.show_page("compare")  # the page already shown: nothing changed, the notice stays
    assert visible_toasts(window) == [plain]
    components.show_page("trainset")
    assert visible_toasts(window) == []  # not over Trainset's buttons

    error = show_toast(window, "Could not read frame 12.", level="error")
    warning = show_toast(window, "2 frames were skipped.", level="warning")
    saved = show_toast(window, "Saved the project sample.gimap", level="ok", action=("Open Folder", lambda: None))
    _navigate(window, "predict")
    assert visible_toasts(window) == [error, warning, saved]  # still to be read, or with a function
    # Who says something about the page it opens navigates first: that notice is shown on the new page.
    _navigate(window, "analyze")
    about_new_page = show_toast(window, "Analyze mode: GIWAXS", level="info")
    assert about_new_page in visible_toasts(window)


# -- tooltips composed at run time (R69, R40) ---------------------------------------------------------


def test_a_folded_sections_own_tooltip_follows_the_language(monkeypatch) -> None:
    from src.gimap.app.presentation.components.sections import AdvancedSection

    QApplication.instance() or QApplication([])
    _keys(monkeypatch, SECTION_TIP)
    host = QWidget()
    try:
        section = AdvancedSection("Run Log", "", host)
        described = AdvancedSection("Advanced", "Options used less often", host)
        assert section.toggle_button.toolTip() == "Show or hide this section"
        assert described.toggle_button.toolTip() == "Options used less often"  # a description is the tooltip
        apply_language("zh", [host])
        assert section.toggle_button.toolTip() == "显示或隐藏这一部分"
        assert section.toggle_button.text() == i18n.ZH["Run Log"]
        built_in_chinese = AdvancedSection("Run Log", "", host)
        assert built_in_chinese.toggle_button.toolTip() == "显示或隐藏这一部分"
        apply_language(DEFAULT_LANGUAGE, [host])
        assert section.toggle_button.toolTip() == built_in_chinese.toggle_button.toolTip() == "Show or hide this section"
        assert described.toggle_button.toolTip() == "Options used less often"
    finally:
        apply_language(DEFAULT_LANGUAGE)
        host.close()


def test_the_levels_tooltip_follows_the_language() -> None:
    from src.gimap.app.presentation.components.levels import LevelControl

    QApplication.instance() or QApplication([])
    host = QWidget()
    try:
        control = LevelControl(host)
        button = control.make_button(host, lambda: None)
        english = "Auto for every frame (1–99.7 %)\nClick: auto once; ▾ for the options and exact limits"
        assert button.toolTip() == english
        apply_language("zh", [host])
        chinese = button.toolTip()
        assert chinese == (i18n.ZH["Auto for every frame ({rule})"].format(rule="1–99.7 %") + "\n"
                           + i18n.ZH["Click: auto once; ▾ for the options and exact limits"])
        assert button.text() == i18n.ZH["Levels"]
        control.fix(2.0, 50.0)  # fixed limits: composed with the values, in the language of now
        assert button.toolTip().startswith(i18n.ZH["Fixed for every frame: {low} … {high}"].format(low="2", high="50"))
        apply_language(DEFAULT_LANGUAGE, [host])
        assert button.toolTip() == ("Fixed for every frame: 2 … 50\nClick: auto once; ▾ for the options and exact limits")
        assert button.text() == "Levels (fixed)"
        button.deleteLater()  # a view that is gone before its control is told nothing, quietly
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        apply_language("zh")
    finally:
        apply_language(DEFAULT_LANGUAGE)
        host.close()


# -- Qt's own texts (R35) -----------------------------------------------------------------------------


def test_qts_ok_button_is_chinese_and_comes_back(monkeypatch) -> None:
    QApplication.instance() or QApplication([])
    _keys(monkeypatch, QT_TEXTS)
    host = QWidget()
    try:
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, host)
        ok = buttons.button(QDialogButtonBox.Ok)
        english = ok.text()
        assert english in ("OK", "&OK")
        box = QMessageBox(QMessageBox.Information, "Saved", "The parameters were saved.", QMessageBox.Ok, host)
        apply_to(host, "zh")
        assert ok.text() == i18n.ZH[english] and buttons.button(QDialogButtonBox.Cancel).text() == i18n.ZH["Cancel"]
        assert box.button(QMessageBox.Ok).text() in (i18n.ZH["OK"], i18n.ZH["&OK"])
        apply_to(host, DEFAULT_LANGUAGE)
        assert ok.text() == english and box.button(QMessageBox.Ok).text() in ("OK", "&OK")
    finally:
        host.close()


def test_the_menu_of_a_text_field_is_chinese_and_keeps_its_shortcuts(monkeypatch) -> None:
    QApplication.instance() or QApplication([])
    _keys(monkeypatch, QT_TEXTS)
    chinese = {key: i18n.ZH[key] for key in QT_TEXTS}  # the table's wording, whoever merged it
    assert not [text for text in chinese.values() if text.isascii()]
    assert len(set(chinese.values())) == len(chinese)  # each comes back to its own English text
    assert translate("&Copy\tCtrl+C", "zh") == chinese["&Copy"] + "\tCtrl+C"  # the shortcut stays as the platform has it
    assert translate(chinese["&Copy"] + "\tCtrl+C", "en") == "&Copy\tCtrl+C"
    assert translate("Select All\tCtrl+A", "zh") == chinese["Select All"] + "\tCtrl+A"
    assert translate("not in the table\tCtrl+K", "zh") is None and translate("\tCtrl+K", "zh") is None
    host = QWidget()
    try:
        for field in (QLineEdit("0.4", host), QPlainTextEdit("notes", host), QSpinBox(host)):
            menu = (field.lineEdit() if isinstance(field, QSpinBox) else field).createStandardContextMenu()
            english = [action.text() for action in menu.actions() if not action.isSeparator()]
            assert any(text.partition("\t")[0] == "&Copy" for text in english)
            apply_to(menu, "zh")  # as the translator does when a menu is shown
            labels = [action.text().partition("\t")[0] for action in menu.actions() if not action.isSeparator()]
            assert {i18n.ZH[key] for key in ("&Undo", "&Redo", "Cu&t", "&Copy", "&Paste", "Select All", "Delete")} <= set(labels)
            assert not [label for label in labels if label.isascii()], labels
            shortcuts = [action.text().partition("\t")[2] for action in menu.actions() if not action.isSeparator()]
            assert shortcuts == [text.partition("\t")[2] for text in english]
            apply_to(menu, DEFAULT_LANGUAGE)
            assert [action.text() for action in menu.actions() if not action.isSeparator()] == english
            menu.deleteLater()
    finally:
        host.close()


def _right_click(widget) -> list[str]:
    """The entries of the menu a right-click on ``widget`` opens: shown (the translator sees it), read, closed."""
    seen: list[str] = []
    done: list[bool] = []

    def read() -> None:
        menu = QApplication.activePopupWidget()
        if isinstance(menu, QMenu):
            seen.extend(action.text() for action in menu.actions() if not action.isSeparator())
        for shown in QApplication.topLevelWidgets():  # never left open: a number field's menu runs its own loop
            if isinstance(shown, QMenu) and shown.isVisible():
                shown.close()
        done.append(True)

    QTimer.singleShot(150, read)
    pos = widget.rect().center()
    # A number field shows its menu and waits here until it is closed; a text field shows it and returns.
    QApplication.sendEvent(widget, QContextMenuEvent(QContextMenuEvent.Mouse, pos, widget.mapToGlobal(pos)))
    end = time.monotonic() + 5.0
    while not done and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    return seen


def test_a_right_click_on_a_number_field_opens_a_chinese_menu(monkeypatch) -> None:
    """The menu as the user gets it. A number field replaces Select All by its own entry and adds two steps:
    with those in the table no English line is left among the Chinese ones."""
    QApplication.instance() or QApplication([])
    _keys(monkeypatch, QT_TEXTS)
    host = QWidget()
    spin, edit = QSpinBox(host), QLineEdit("0.4", host)
    edit.move(0, 40)
    host.resize(200, 100)
    host.show()
    try:
        english = _right_click(spin)
        assert [text.partition("\t")[0] for text in english][-3:] == ["&Select All", "&Step up", "Step &down"]
        apply_language("zh", [host])
        for field in (spin, edit):
            labels = [text.partition("\t")[0] for text in _right_click(field)]
            assert len(labels) >= 7 and not [label for label in labels if label.isascii()], labels
        assert [text.partition("\t")[0] for text in _right_click(spin)][-3:] == [
            i18n.ZH[key] for key in ("&Select All", "&Step up", "Step &down")]
        apply_language(DEFAULT_LANGUAGE, [host])
        assert _right_click(spin) == english  # a new menu each time: English again
    finally:
        apply_language(DEFAULT_LANGUAGE)
        host.close()


# -- a table dropped on Fitting (R7 / R56) ------------------------------------------------------------


def test_only_what_fitting_reads_is_a_curve_for_the_window() -> None:
    from src.gimap.features.fitting.application import CURVE_SUFFIXES

    assert main_module.CURVE_SUFFIXES == tuple(CURVE_SUFFIXES) == (".dat", ".txt")
    assert main_module.TABLE_SUFFIXES == (".csv",)
    assert ".dat" in main_module.NOT_A_CURVE and "_fit_input.dat" in main_module.NOT_A_CURVE


def test_a_csv_table_dropped_on_fitting_is_not_opened_and_the_notice_says_why(window, monkeypatch, tmp_path) -> None:
    table = tmp_path / "film_horizontal.csv"
    table.write_text("# GIMaP Analyze cut\n# q in 1/A\nq,I,sigma,pixels\n0.01,100,1,12\n0.02,80,1,12\n", encoding="utf-8")
    curve = tmp_path / "film_fit_input.dat"
    curve.write_text("0.1 10 1\n0.2 8 1\n0.3 6 1\n", encoding="utf-8")
    components = window.components
    fit_page = components.fitting_workspace.fit_page
    opened, sent_on = [], []
    monkeypatch.setattr(fit_page, "open_curve", lambda path, side=None: opened.append(path))
    monkeypatch.setattr(components, "open_dropped", lambda paths: sent_on.append(list(paths)))
    _navigate(window, "fitting")
    _settle()
    assert _drop(fit_page, [table])
    assert opened == [] and sent_on == []  # neither to Fitting's reader nor to Analyze as a frame
    assert components.current_page_key() == "fitting"
    notice = visible_toasts(window)[-1]
    assert notice.text() == f"Could not open film_horizontal.csv: {main_module.NOT_A_CURVE}"
    assert notice.property("level") == "warning"

    assert _drop(fit_page, [table, curve])  # with a curve beside it, the curve opens
    assert opened == [str(curve)] and sent_on == []
    window.open_dropped([str(table), str(tmp_path / "frame.tif")])  # the rest goes on to Analyze
    assert sent_on == [[str(tmp_path / "frame.tif")]]

    components.show_page("analyze")  # elsewhere a .csv is no curve of Fitting: left to the page shown, as before
    window.open_dropped([str(table)])
    assert sent_on[-1] == [str(table)] and opened == [str(curve)]


# -- Tools ▸ Convert Current File… with the converter already open (R12, the shell's half) ----------------


def test_convert_current_file_hands_the_frame_to_the_open_converter_its_own_way(window, monkeypatch) -> None:
    import src.gimap.app.menus as menus_module

    handed = []

    class Converter(QDialog):
        def use_current_file(self, path: str) -> str:  # the converter's contract: it says why it cannot read one
            handed.append(path)
            return ""

    monkeypatch.setattr(menus_module, "_format_converter_dialog", lambda parent, *, current_file="": Converter(parent))
    menus = window.menus
    monkeypatch.setattr(menus, "current_detector_file", lambda: "C:/data/frame_0001.edf")
    opened = menus.open_format_converter(include_current=False)  # Tools ▸ Format Converter
    try:
        assert handed == []
        assert menus.open_format_converter(include_current=True) is opened  # … Convert Current File…, window open
        assert handed == ["C:/data/frame_0001.edf"]
        assert menus.open_format_converter(include_current=False) is opened and len(handed) == 1  # no frame asked for
    finally:
        opened.close()
