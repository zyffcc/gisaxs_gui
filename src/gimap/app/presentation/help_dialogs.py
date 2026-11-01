"""The windows of the Help menu: Keyboard Shortcuts, How to Cite GIMaP, About GIMaP, and the folder an example
is copied to. They show what they are given; the composition root (``src.gimap.app.help_commands``) reads the
files. Keyboard Shortcuts, How to Cite and About do not block work: each is one window, closed with Close.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR, Qt
from PyQt5.QtGui import QFontDatabase, QGuiApplication, QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMenuBar,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .assets import app_colored_logo_pixmap, app_icon
from .components.table_copy import enable_table_copy
from .i18n import current_language, language_changed, tr, trf
from .theme import theme_manager
from src.gimap.shared.version import APP_NAME, PRE_RELEASE, __version__

# -- what the program runs on ---------------------------------------------------------------------


def runtime_versions() -> dict[str, str]:
    """Python, Qt, PyQt and numpy: the versions an error report or the About box names."""
    import numpy

    return {"Python": platform.python_version(), "Qt": QT_VERSION_STR, "PyQt": PYQT_VERSION_STR,
            "numpy": numpy.__version__}


def _system() -> str:
    """The operating system and its version, e.g. ``Windows 10.0.26300 (AMD64)``: no names, no paths."""
    system = platform.system() or "unknown"
    if system == "Windows":
        detail = platform.version()
    elif system == "Darwin":
        system, detail = "macOS", platform.mac_ver()[0]
    else:
        detail = platform.release()
    machine = platform.machine()
    return " ".join(part for part in (system, detail, f"({machine})" if machine else "") if part)


def system_summary() -> str:
    """What Help ▸ Report a Problem copies: the version of GIMaP and of what it runs on, in English (it goes into
    a report on GitHub). No path, no user name and no data are in it."""
    kind = " (pre-release)" if PRE_RELEASE else ""
    lines = [f"{APP_NAME} {__version__}{kind}", f"OS: {_system()}"]
    lines += [f"{name}: {version}" for name, version in runtime_versions().items()]
    application = QGuiApplication.instance()
    if application is not None:
        lines.append(f"Qt platform: {QGuiApplication.platformName()}")
    lines.append(f"Interface: {current_language()}, {theme_manager().mode} theme")
    return "\n".join(lines)


# -- Help ▸ Keyboard Shortcuts… ------------------------------------------------------------------------


@dataclass(frozen=True)
class PageKeys:
    """Keys that belong to a page or a window: they act while it has the keyboard focus."""

    where: str
    keys: tuple[str, ...]
    """``QKeySequence`` portable text (``"Ctrl+Z"``, ``"PgUp"``); several keys do the same."""
    what: str
    owner: str = ""
    """The object name of the page whose ``QShortcut``\\ s these are (a test finds every one of the window
    here); empty for keys a widget handles itself (the polygon of a mask, Ctrl+C in tables, tool windows)."""


PAGE_KEYS = (
    PageKeys("Analyze", ("PgUp",), "Previous imported frame", "analyzePage"),
    PageKeys("Analyze", ("PgDown",), "Next imported frame (through the frames of a source, then on to the next source)",
             "analyzePage"),
    PageKeys("Analyze", ("Ctrl+Z",), "Undo a change of the set-up (menu: Edit ▸ Undo)", "analyzePage"),
    PageKeys("Analyze", ("Ctrl+Y", "Ctrl+Shift+Z"), "Redo it (menu: Edit ▸ Redo)", "analyzePage"),
    PageKeys("Analyze", ("Ctrl+E",), "Export the curves of this frame (menu: Analyze ▸ Export Curves…)", "analyzePage"),
    PageKeys("Analyze", ("Ctrl+Shift+E",), "Batch Export… (menu: Analyze ▸ Batch Export…)", "analyzePage"),
    PageKeys("Analyze", ("F5",), "Analyse the frame again (menu: Analyze ▸ Repeat Analysis)", "analyzePage"),
    PageKeys("Analyze", ("Del", "Backspace"),
             "Remove the selected source (all its frames), mask or cut region from its list (the files themselves "
             "are not touched)", "analyzePage"),
    PageKeys("Analyze", ("Esc",), "Stop picking the beam centre on the image; cancel drawing a mask", "analyzePage"),
    PageKeys("Analyze, drawing a polygon mask", ("Return", "Enter"), "Close the polygon (from three points on)"),
    PageKeys("Analyze, drawing a polygon mask", ("Backspace",), "Take back the last point"),
    PageKeys("Fitting", ("Ctrl+Return",), "Fit (menu: Fitting ▸ Fit)", "fittingSinglePage"),
    PageKeys("Fitting", ("Ctrl+Z",), "Undo a change of the model (menu: Edit ▸ Undo)", "fittingSinglePage"),
    PageKeys("Fitting", ("Ctrl+Y", "Ctrl+Shift+Z"), "Redo it (menu: Edit ▸ Redo)", "fittingSinglePage"),
    PageKeys("Compare", ("Del",), "Remove the selected series from the table", "comparePage"),
    PageKeys("Tables", ("Ctrl+C",), "Copy the selected rows with their column names"),
    PageKeys("Geometry Calibration", ("Esc",), "Close the window (it asks first while a calibration runs)"),
    PageKeys("XRR Series Extractor", ("Esc",),
             "Stop picking the direct-beam centre on the preview; when not picking, close the window"),
    PageKeys("Trainset Build (the image)", ("Esc",),
             "Leave the drawing mode: picking the direct beam, the ROI or a mask shape"),
)
"""The keys of the pages and windows, in the order of the manual's table (section 3.7)."""


def plain_text(text: str) -> str:
    """A menu text as it reads: without its mnemonic (``&File``, ``文件(&F)``) or the key after a tab."""
    text = text.split("\t")[0]
    out, index = [], 0
    while index < len(text):
        if text[index:index + 2] == "(&" and text[index + 3:index + 4] == ")":
            index += 4  # 文件(&F) → 文件
        elif text[index:index + 2] == "&&":  # a literal ampersand
            out.append("&")
            index += 2
        elif text[index] == "&":
            index += 1
        else:
            out.append(text[index])
            index += 1
    return "".join(out).strip()


def _keys_text(keys) -> str:
    return ", ".join(QKeySequence(key).toString(QKeySequence.NativeText) for key in keys)


def menu_key_rows(menubar: QMenuBar) -> list[tuple[str, str, str]]:
    """``(where, keys, what)`` of every menu entry with a key of its own, read from the menus as they are now
    (in the interface language). A page's key shown in a menu (as text after a tab: the entry holds no shortcut)
    is left to the page rows."""
    rows = []

    def walk(menu: QMenu, where: str, path: tuple[str, ...]) -> None:
        for action in menu.actions():
            if action.isSeparator():
                continue
            if action.menu() is not None:
                walk(action.menu(), where, path + (plain_text(action.text()),))
            elif action.shortcuts() and action.shortcutContext() != Qt.WidgetShortcut:
                what = " ▸ ".join(path + (plain_text(action.text()),))
                rows.append((where, _keys_text(action.shortcuts()), what))

    for top in menubar.actions():
        if top.menu() is not None:
            walk(top.menu(), trf("{menu} menu", menu=plain_text(top.text())), ())
    return rows


def page_key_rows() -> list[tuple[str, str, str]]:
    return [(tr(row.where), _keys_text(row.keys), tr(row.what)) for row in PAGE_KEYS]


class KeyboardShortcutsDialog(QDialog):
    """Every key: those of the menus (live from the menu bar of ``window``), then those of the pages."""

    HEADERS = ("Where", "Keys", "What")

    def __init__(self, window: QWidget):
        super().__init__(window)
        self.setObjectName("keyboardShortcutsDialog")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self._window = window
        layout = QVBoxLayout(self)
        self.note = QLabel(self)
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.table = QTableWidget(0, len(self.HEADERS), self)
        self.table.setObjectName("keyboardShortcutsTable")
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        enable_table_copy(self.table)
        layout.addWidget(self.table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        language_changed().connect(self.refresh_language)
        self.refresh_language()
        self.resize(820, 560)

    def rows(self) -> list[tuple[str, str, str]]:
        menubar = self._window.menuBar() if hasattr(self._window, "menuBar") else None
        return (menu_key_rows(menubar) if menubar is not None else []) + page_key_rows()

    def refresh_language(self, *_args) -> None:
        self.setWindowTitle(tr("Keyboard Shortcuts"))
        self.note.setText(tr("The keys of the menus work on every page. The keys of a page or a window act while it "
                             "has the keyboard focus: after switching pages, click into the page first. Ctrl+C "
                             "copies the selected rows."))
        self.table.setHorizontalHeaderLabels([tr(text) for text in self.HEADERS])
        rows = self.rows()
        self.table.setRowCount(len(rows))
        for row, cells in enumerate(rows):
            for column, text in enumerate(cells):
                self.table.setItem(row, column, QTableWidgetItem(text))
        self.table.resizeRowsToContents()


def _one_window(parent: Optional[QWidget], name: str) -> Optional[QDialog]:
    """The window ``name`` of ``parent`` that is open already, brought to the front."""
    shown = parent.findChild(QDialog, name, Qt.FindDirectChildrenOnly) if parent is not None else None
    if shown is not None:
        shown.show()
        shown.raise_()
        shown.activateWindow()
    return shown


def show_keyboard_shortcuts(window: QWidget) -> QDialog:
    dialog = _one_window(window, "keyboardShortcutsDialog") or KeyboardShortcutsDialog(window)
    dialog.show()
    return dialog


# -- Help ▸ How to Cite GIMaP… -------------------------------------------------------------------------


def _copy(text: str, status: QLabel, said: str) -> None:
    QApplication.clipboard().setText(text)
    status.setText(said)


def _read_only_text(parent: QWidget, text: str, name: str, *, fixed: bool = False) -> QPlainTextEdit:
    box = QPlainTextEdit(text, parent)
    box.setObjectName(name)
    box.setReadOnly(True)
    if fixed:
        box.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
    return box


def _follow_language(dialog: QDialog, retranslate: Callable[[], None]) -> None:
    """``retranslate()`` after every switch of the interface language while ``dialog`` lives (its composed texts;
    the walker translates the fixed ones)."""
    signal = language_changed()

    def changed(*_args) -> None:
        try:
            retranslate()
        except RuntimeError:  # the dialog is being deleted
            pass

    def forget(*_args) -> None:
        try:
            signal.disconnect(changed)
        except (TypeError, RuntimeError):
            pass

    signal.connect(changed)
    dialog.destroyed.connect(forget)


def show_citation(parent: Optional[QWidget], text: str, bibtex: str, *, doi: str = "") -> QDialog:
    """The reference to give, with Copy and Copy BibTeX. ``doi``: the DOI of ``CITATION.cff`` (none is made up)."""
    shown = _one_window(parent, "citationDialog")
    if shown is not None:
        shown.close()  # read again: a new window
    dialog = QDialog(parent)
    dialog.setObjectName("citationDialog")
    dialog.setAttribute(Qt.WA_DeleteOnClose, True)
    layout = QVBoxLayout(dialog)
    label = QLabel(dialog)
    label.setWordWrap(True)
    layout.addWidget(label)
    layout.addWidget(_read_only_text(dialog, text, "citationText"), 1)
    layout.addWidget(QLabel("BibTeX", dialog))
    layout.addWidget(_read_only_text(dialog, bibtex, "citationBibtex", fixed=True), 2)
    status = QLabel("", dialog)
    status.setObjectName("citationStatus")
    buttons = QDialogButtonBox(dialog)
    copy = buttons.addButton(tr("Copy"), QDialogButtonBox.ActionRole)
    copy.setObjectName("citationCopy")
    copy.clicked.connect(lambda: _copy(text, status, tr("The reference is on the clipboard.")))
    copy_bibtex = buttons.addButton(tr("Copy BibTeX"), QDialogButtonBox.ActionRole)
    copy_bibtex.setObjectName("citationCopyBibtex")
    copy_bibtex.clicked.connect(lambda: _copy(bibtex, status, tr("The BibTeX entry is on the clipboard.")))
    close = buttons.addButton(tr("Close"), QDialogButtonBox.RejectRole)
    close.setDefault(True)
    buttons.rejected.connect(dialog.close)
    row = QHBoxLayout()
    row.addWidget(status, 1)
    row.addWidget(buttons)
    layout.addLayout(row)

    def retranslate() -> None:
        dialog.setWindowTitle(tr("How to Cite GIMaP"))
        intro = tr("If you publish results obtained with GIMaP, please cite it as below and name the version you "
                   "used (the export records of GIMaP write it as software_version).")
        label.setText(intro if doi else intro + " " + tr("GIMaP has no DOI yet: cite the repository and the version."))
        status.clear()

    retranslate()
    _follow_language(dialog, retranslate)
    dialog.resize(720, 460)
    dialog.show()
    return dialog


# -- Help ▸ About GIMaP ----------------------------------------------------------------------------------

LICENCE_LINE = ("MIT licence; the Windows package also contains components under other licences — "
                "see Help ▸ Third-party Notices")


def about_text(release: str, url: str) -> str:
    """What About shows, as plain text (Copy puts it on the clipboard)."""
    versions = " · ".join(f"{name} {version}" for name, version in runtime_versions().items())
    return "\n".join((release, tr(LICENCE_LINE), versions, url))


def show_about_dialog(parent: Optional[QWidget], release: Callable[[], str], url: str) -> QDialog:
    """About: the release (``release()``, in the interface language), the summary, the licence, the versions of
    what GIMaP runs on and the repository, with Copy. One window at a time."""
    shown = _one_window(parent, "aboutDialog")
    if shown is not None:
        return shown
    dialog = QDialog(parent)
    dialog.setObjectName("aboutDialog")
    dialog.setAttribute(Qt.WA_DeleteOnClose, True)
    dialog.setWindowIcon(app_icon())
    layout = QVBoxLayout(dialog)
    row = QHBoxLayout()
    # The coloured logo is a dark blue G: on the dark theme the application icon (a white G) stands out instead.
    logo = app_icon().pixmap(88, 88) if theme_manager().mode == "dark" else app_colored_logo_pixmap(88, 88)
    if not logo.isNull():
        picture = QLabel(dialog)
        picture.setPixmap(logo)
        picture.setAlignment(Qt.AlignTop)
        row.addWidget(picture)
    text = QLabel(dialog)
    text.setObjectName("aboutText")
    text.setTextFormat(Qt.RichText)
    text.setWordWrap(True)
    text.setOpenExternalLinks(True)
    text.setTextInteractionFlags(Qt.TextBrowserInteraction)
    row.addWidget(text, 1)
    layout.addLayout(row)
    status = QLabel("", dialog)
    buttons = QDialogButtonBox(dialog)
    copy = buttons.addButton(tr("Copy"), QDialogButtonBox.ActionRole)
    copy.setObjectName("aboutCopy")
    copy.clicked.connect(lambda: _copy(about_text(release(), url), status, tr("Copied to the clipboard.")))
    close = buttons.addButton(tr("Close"), QDialogButtonBox.RejectRole)
    close.setDefault(True)
    buttons.rejected.connect(dialog.close)
    bottom = QHBoxLayout()
    bottom.addWidget(status, 1)
    bottom.addWidget(buttons)
    layout.addLayout(bottom)

    def retranslate() -> None:
        dialog.setWindowTitle(trf("About {name}", name=APP_NAME))
        summary = tr("Desktop analysis of GISAXS / GIWAXS data: detector geometry, cuts, "
                     "fitting and machine-learning assisted workflows.")
        versions = " · ".join(f"{name} {version}" for name, version in runtime_versions().items())
        text.setText(f"<b style='font-size:16pt'>{APP_NAME}</b><br>{release()}<br><br>{summary}<br><br>"
                     f"{tr(LICENCE_LINE)}<br>{versions}<br><br><a href='{url}'>{url}</a>")
        status.clear()

    retranslate()
    _follow_language(dialog, retranslate)
    dialog.resize(560, 300)
    dialog.show()
    return dialog


# -- Help ▸ Example Data ---------------------------------------------------------------------------------


def _existing(path: Path) -> str:
    """``path`` or the nearest folder above it that exists (where a folder dialog can start)."""
    for folder in (path, *path.parents):
        if folder.is_dir():
            return str(folder)
    return str(Path.home())


def ask_example_folder(parent: Optional[QWidget], name: str, default: str,
                       choose: Optional[Callable[[QWidget, str], str]] = None) -> str:
    """The folder to copy the example ``name`` to (``default`` proposed and editable, Browse… to pick one);
    ``""`` when cancelled. ``choose(dialog, start)`` replaces the folder dialog of Browse… (tests)."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(tr("Example Data"))
    layout = QVBoxLayout(dialog)
    label = QLabel(trf("GIMaP copies the example {name} to this folder and opens it in Analyze. What you export "
                       "is written next to the copy, never into the program folder.", name=name), dialog)
    label.setWordWrap(True)
    layout.addWidget(label)
    row = QHBoxLayout()
    edit = QLineEdit(default, dialog)
    edit.setObjectName("exampleFolderEdit")
    row.addWidget(edit, 1)
    browse = QPushButton(tr("Browse…"), dialog)

    def pick() -> None:
        start = _existing(Path(edit.text().strip() or default))
        chosen = (choose or (lambda owner, folder: QFileDialog.getExistingDirectory(
            owner, tr("Example Data"), folder)))(dialog, start)
        if chosen:
            edit.setText(str(Path(chosen)))

    browse.clicked.connect(pick)
    row.addWidget(browse)
    layout.addLayout(row)
    buttons = QDialogButtonBox(dialog)
    accept = buttons.addButton(tr("Copy and Open"), QDialogButtonBox.AcceptRole)
    accept.setDefault(True)
    buttons.addButton(QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.resize(620, dialog.sizeHint().height())
    try:
        return edit.text().strip() if dialog.exec_() == QDialog.Accepted else ""
    finally:
        dialog.deleteLater()


__all__ = [
    "KeyboardShortcutsDialog",
    "LICENCE_LINE",
    "PAGE_KEYS",
    "PageKeys",
    "about_text",
    "ask_example_folder",
    "menu_key_rows",
    "page_key_rows",
    "plain_text",
    "runtime_versions",
    "show_about_dialog",
    "show_citation",
    "show_keyboard_shortcuts",
    "system_summary",
]
