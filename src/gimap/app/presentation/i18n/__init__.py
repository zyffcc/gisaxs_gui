"""Interface language: English (the default) or Chinese, applied to the widgets that exist.

The English text in the code stays the source of truth; ``zh.py`` maps exact English strings
to Chinese. ``apply_language`` walks a widget tree (labels, buttons, group boxes, tabs, combo
items, table and tree headers, spin-box prefixes, suffixes and special texts, placeholders,
tooltips, menus and actions) and replaces every text it knows, in either direction, so switching
back to English restores the originals. Windows, dialogs and menus that appear later are
translated when they are shown. Text that is not in the table — values, file names, messages
composed at run time — stays as it is (``tr`` / ``trf`` translate those where they are made, and
``language_changed()`` tells their owners to compose them again after a switch), and so do the headers
of a table that holds names in them (``DATA_HEADERS``).
Scientific values and units are never translated.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QEvent, QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QAction,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QPlainTextEdit,
    QSpinBox,
    QTableWidget,
    QTabBar,
    QTabWidget,
    QTreeWidget,
    QWidget,
)

from .zh import ZH as _ZH

# The Chinese interface fonts have no "▸" (U+25B8) or "▾" (U+25BE), drawn as empty boxes: use "›" in the
# menu paths and "▼" for a drop-down.
ZH = {english: chinese.replace("▸", "›").replace("▾", "▼") for english, chinese in _ZH.items()}

LANGUAGES = {"en": "English", "zh": "中文"}
LANGUAGE_KEY = "appearance.language"
DEFAULT_LANGUAGE = "en"
_TO_ENGLISH = {chinese: english for english, chinese in ZH.items()}


def normalized_language(value) -> str:
    value = str(value or "").strip().lower()
    return value if value in LANGUAGES else DEFAULT_LANGUAGE


def translate(text: str, language: str) -> Optional[str]:
    """The text in ``language``, or ``None`` when the table does not know it (padding kept)."""
    if not text:
        return None
    table = ZH if language == "zh" else _TO_ENGLISH
    found = table.get(text)
    if found is None:
        core = text.strip()
        found = table.get(core) if core and core != text else None
        if found is not None:
            found = text[: len(text) - len(text.lstrip())] + found + text[len(text.rstrip()):]
    return found


class _Translator(QObject):
    """Keeps the current language and translates windows and menus as they are shown.

    ``changed(language)`` comes after ``apply_language`` switched the language and walked its widgets: a
    widget whose text is made from its own state (a status chip, a sentence composed with ``tr``) redraws
    it there, since the walker only knows the texts in the table."""

    changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.language = DEFAULT_LANGUAGE
        self._installed = False

    def install(self) -> None:
        application = QApplication.instance()
        if application is not None and not self._installed:
            application.installEventFilter(self)
            self._installed = True

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API
        if event.type() != QEvent.Show or not isinstance(watched, QWidget):
            return False
        try:
            if (watched.isWindow() or isinstance(watched, QMenu)) and (
                self.language != DEFAULT_LANGUAGE or watched.property("gimapTranslated")
            ):
                apply_to(watched, self.language)
        except RuntimeError:  # a window being torn down
            pass
        return False


_TRANSLATOR: Optional[_Translator] = None


def translator() -> _Translator:
    global _TRANSLATOR
    if _TRANSLATOR is None:
        _TRANSLATOR = _Translator()
    return _TRANSLATOR


def current_language() -> str:
    return translator().language


def _swap(owner: QObject, key: str, getter, setter, language: str) -> None:
    """Translate one text; the English original is kept on ``owner`` so switching back is exact."""
    try:
        text = getter()
        if not text:
            return
        stored_key = f"gimapEn_{key}"
        if language == DEFAULT_LANGUAGE:
            original = owner.property(stored_key)
            if original and translate(original, "zh") == text:
                setter(original)
            else:
                english = translate(text, DEFAULT_LANGUAGE)
                if english is not None:
                    setter(english)
            return
        new = translate(text, language)
        if new is not None:
            owner.setProperty(stored_key, text)
            setter(new)
    except RuntimeError:  # the C++ object is gone
        return


def _actions(widget: QWidget, language: str) -> None:
    for action in widget.actions():
        _action(action, language)


def _action(action: QAction, language: str) -> None:
    _swap(action, "text", action.text, action.setText, language)
    _swap(action, "tip", action.toolTip, action.setToolTip, language)
    menu = action.menu()
    if menu is not None:
        _swap(menu, "title", menu.title, menu.setTitle, language)


DATA_HEADERS = "gimapDataHeaders"
"""A table or tree with this property set has headers that are data (series, sample or parameter names):
they are never translated, even when a name is also an interface word (“Background”, “Data”)."""


def _headers(widget: QWidget, language: str) -> None:
    """The column (and row) titles of a table or a tree (not when they are data: ``DATA_HEADERS``)."""
    if widget.property(DATA_HEADERS):
        return
    if isinstance(widget, QTableWidget):
        for prefix, count, item_at in (("hh", widget.columnCount(), widget.horizontalHeaderItem),
                                       ("vh", widget.rowCount(), widget.verticalHeaderItem)):
            for index in range(count):
                item = item_at(index)
                if item is not None:
                    _swap(widget, f"{prefix}{index}", item.text, item.setText, language)
    elif isinstance(widget, QTreeWidget):
        header = widget.headerItem()
        for column in range(header.columnCount() if header is not None else 0):
            _swap(widget, f"th{column}", lambda column=column: header.text(column),
                  lambda text, column=column: header.setText(column, text), language)


def _spin_texts(widget: QAbstractSpinBox, language: str) -> None:
    """“last ” 10 “ frames”, or the special text shown at the minimum (“auto”, “from profile”); padding kept."""
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        _swap(widget, "prefix", widget.prefix, widget.setPrefix, language)
        _swap(widget, "suffix", widget.suffix, widget.setSuffix, language)
    _swap(widget, "special", widget.specialValueText, widget.setSpecialValueText, language)


def apply_to(root: QWidget, language: str) -> None:
    """Translate ``root`` and every widget and action under it."""
    for widget in [root, *root.findChildren(QWidget)]:
        if isinstance(widget, (QLabel, QAbstractButton)):
            _swap(widget, "text", widget.text, widget.setText, language)
        elif isinstance(widget, QGroupBox):
            _swap(widget, "title", widget.title, widget.setTitle, language)
        if isinstance(widget, (QTableWidget, QTreeWidget)):
            _headers(widget, language)
        if isinstance(widget, QAbstractSpinBox):
            _spin_texts(widget, language)
        if isinstance(widget, (QLineEdit, QPlainTextEdit)):
            _swap(widget, "placeholder", widget.placeholderText, widget.setPlaceholderText, language)
        if isinstance(widget, QComboBox):
            for index in range(widget.count()):
                _swap(widget, f"item{index}", lambda index=index: widget.itemText(index),
                      lambda text, index=index: widget.setItemText(index, text), language)
        if isinstance(widget, (QTabWidget, QTabBar)):
            for index in range(widget.count()):
                _swap(widget, f"tab{index}", lambda index=index: widget.tabText(index),
                      lambda text, index=index: widget.setTabText(index, text), language)
        if isinstance(widget, QListWidget):
            for index in range(widget.count()):
                item = widget.item(index)
                _swap(widget, f"row{index}", item.text, item.setText, language)
        if isinstance(widget, QMenu):
            _swap(widget, "title", widget.title, widget.setTitle, language)
        if widget.isWindow():
            _swap(widget, "window", widget.windowTitle, widget.setWindowTitle, language)
        _swap(widget, "tip", widget.toolTip, widget.setToolTip, language)
        _actions(widget, language)
    for action in root.findChildren(QAction):
        _action(action, language)
    root.setProperty("gimapTranslated", language != DEFAULT_LANGUAGE)


def apply_language(language: str, roots=()) -> str:
    """Switch the interface language: the given widgets now, windows and menus when shown; then
    ``language_changed`` (when the language is not the one before)."""
    language = normalized_language(language)
    state = translator()
    previous = state.language
    state.language = language
    state.install()
    for root in roots:
        if root is not None:
            apply_to(root, language)
    if language != previous:
        state.changed.emit(language)
    return language


def language_changed():
    """The signal ``(language)`` that comes after each switch of the interface language (``apply_language``)."""
    return translator().changed


def tr(text: str) -> str:
    """``text`` in the current language (for text composed at run time)."""
    return translate(text, current_language()) or text if current_language() != DEFAULT_LANGUAGE else text


def trf(template: str, **values) -> str:
    """``tr(template).format(**values)``: the table holds the template, the values are filled in after."""
    return tr(template).format(**values)


__all__ = [
    "DATA_HEADERS", "DEFAULT_LANGUAGE", "LANGUAGES", "LANGUAGE_KEY", "apply_language", "apply_to", "current_language",
    "language_changed", "normalized_language", "tr", "translate", "trf",
]
