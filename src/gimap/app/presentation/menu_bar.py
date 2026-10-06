"""Main menu bar: File, View, Tools and Help.

The menus only know commands (plain callables); the composition root in
``src.gimap.app.menus`` decides what each command does.  A command that is
not provided hides its action, so the builder has no feature imports.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from PyQt5.QtCore import QEvent, QObject, QRect, Qt, QUrl
from PyQt5.QtGui import QDesktopServices, QKeySequence, QTextDocument
from PyQt5.QtWidgets import QAction, QActionGroup, QApplication, QMainWindow, QMenu, QMessageBox, QWidget

from .assets import app_colored_logo_pixmap, app_icon
from .i18n import LANGUAGES, tr
from .layout_metrics import available_geometry
from .navigation import NavigationItem
from .recent_items import recent_label
from .theme import theme_manager

APP_NAME = "GIMaP"
APP_VERSION = "0.1.0"
APP_RELEASE = "GIMaP v0.1.0 (Pre-release)"
GITHUB_URL = "https://github.com/zyffcc/gisaxs_gui"
USER_MANUAL = Path(__file__).resolve().parents[4] / "docs" / "User_Manual.md"

Command = Optional[Callable[[], object]]


@dataclass
class MenuCommands:
    """What the menu entries do; ``None`` hides the entry."""

    open_files: Command = None
    open_files_label: Optional[Callable[[], tuple]] = None
    """``(text, tip)`` of the Open Data entry, asked each time the File menu opens (what Ctrl+O does there)."""
    open_folder: Command = None
    recent_paths: Callable[[], list] = lambda: []
    open_recent: Optional[Callable[[str], object]] = None
    clear_recent: Command = None
    save_parameters: Command = None
    load_parameters: Command = None
    open_project: Command = None
    save_project: Command = None
    save_project_as: Command = None
    quit: Command = None
    show_workspace: Optional[Callable[[str], object]] = None
    set_theme: Optional[Callable[[str], object]] = None
    current_theme: Callable[[], str] = lambda: "light"
    set_language: Optional[Callable[[str], object]] = None
    """View ▸ Language: the same switch as Settings ▸ Appearance ▸ Language (``en`` or ``zh``)."""
    current_language: Callable[[], str] = lambda: "en"
    change_font_size: Optional[Callable[[float], object]] = None
    reset_font_size: Command = None
    set_sidebar_collapsed: Optional[Callable[[bool], object]] = None
    sidebar_collapsed: Callable[[], bool] = lambda: False
    toggle_full_screen: Command = None
    geometry_calibration: Command = None
    format_converter: Command = None
    convert_current_file: Command = None
    xrr_extractor: Command = None
    ai_fitting_workspace: Command = None
    claude_assistant: Command = None
    settings: Command = None
    open_data_folder: Command = None
    extra: dict = field(default_factory=dict)


class MainMenuBar(QObject):
    """Builds the menus into ``window.menuBar()`` and keeps their actions."""

    def __init__(
        self,
        window: QMainWindow,
        commands: MenuCommands,
        workspaces: Sequence[NavigationItem] = (),
    ):
        super().__init__(window)
        self.window = window
        self.commands = commands
        self.workspaces = tuple(workspaces)
        self.actions: dict[str, QAction] = {}
        self._submenus: list[QMenu] = []
        menubar = window.menuBar()
        menubar.clear()
        self.file_menu = menubar.addMenu("&File")
        self.view_menu = menubar.addMenu("&View")
        self.tools_menu = menubar.addMenu("&Tools")
        self.help_menu = menubar.addMenu("&Help")
        self._build_file()
        self._build_view()
        self._build_tools()
        self._build_help()
        # The status bar is hidden on most pages: the descriptions show as tooltips in the menus.
        for menu in (self.file_menu, self.view_menu, self.tools_menu, self.help_menu, *self._submenus):
            menu.setToolTipsVisible(True)

    # -- helpers ------------------------------------------------------------

    def _add(
        self,
        menu: QMenu,
        name: str,
        text: str,
        command: Command,
        *,
        shortcut: str | QKeySequence.StandardKey | None = None,
        tip: str = "",
    ) -> QAction | None:
        if command is None:
            return None
        action = QAction(text, self.window)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        if tip:
            action.setStatusTip(tip)
            action.setToolTip(tip)
        action.triggered.connect(lambda _checked=False: command())
        menu.addAction(action)
        self.actions[name] = action
        return action

    # -- menus --------------------------------------------------------------

    def _build_file(self) -> None:
        c, menu = self.commands, self.file_menu
        self._add(menu, "open_files", "&Open Data…", c.open_files,
                  shortcut=QKeySequence.Open, tip="Open detector frames (CBF, NXS, TIFF, EDF) in Analyze")
        if c.open_files is not None and c.open_files_label is not None:
            menu.aboutToShow.connect(self._sync_open_files)
        self._add(menu, "open_folder", "Open &Folder…", c.open_folder,
                  shortcut="Ctrl+Shift+O", tip="Open every frame of a folder in Analyze")
        if c.open_recent is not None:
            self.recent_menu = menu.addMenu("Open &Recent")
            self.recent_menu.aboutToShow.connect(self._fill_recent)
            self._submenus.append(self.recent_menu)
        menu.addSeparator()
        self._add(menu, "open_project", "Open &Project…", c.open_project, shortcut="Ctrl+Shift+P",
                  tip="Reopen a sample as it was left: the frames and set-up of Analyze, the curve and model of Fitting")
        self._add(menu, "save_project", "&Save Project", c.save_project, shortcut=QKeySequence.Save,
                  tip="Save what is open in Analyze and Fitting as a project (.gimap)")
        self._add(menu, "save_project_as", "Save Project &As…", c.save_project_as, shortcut="Ctrl+Shift+S",
                  tip="Save the project under another name")
        menu.addSeparator()
        if c.load_parameters is not None or c.save_parameters is not None:
            labs = self.labs_menu = menu.addMenu("&Labs Parameters")
            labs.setToolTip("The settings of 2D Prediction and Trainset Build")
            self._submenus.append(labs)
            self._add(labs, "load_parameters", "&Load…", c.load_parameters,
                      tip="Load the settings of the Labs pages (2D Prediction, Trainset Build) from a JSON file")
            self._add(labs, "save_parameters", "&Save As…", c.save_parameters,
                      tip="Save the settings of the Labs pages to a JSON file")
            menu.addSeparator()
        self._add(menu, "quit", "E&xit", c.quit, shortcut=QKeySequence.Quit)

    def _sync_open_files(self) -> None:
        """Open Data says what it opens on the page shown (Open Curve… on Fitting's Single analysis)."""
        action = self.actions.get("open_files")
        if action is None or self.commands.open_files_label is None:
            return
        text, tip = self.commands.open_files_label()
        action.setText(text)
        action.setToolTip(tip)
        action.setStatusTip(tip)

    def _fill_recent(self) -> None:
        """The projects, files and folders opened last, newest first (built each time the menu opens).

        Rows read ``&N  name — parent folder name`` plus `` (folder)`` / `` (project)``; the full path
        is the tooltip.
        """
        menu, c = self.recent_menu, self.commands
        menu.clear()
        paths = list(c.recent_paths() or [])
        if not paths:
            empty = menu.addAction("No recent data")
            empty.setEnabled(False)
        for number, path in enumerate(paths, start=1):
            path = Path(path)
            action = menu.addAction(f"&{number}  {recent_label(path)}")
            action.setToolTip(str(path))
            action.setStatusTip(str(path))
            action.triggered.connect(lambda _checked=False, target=str(path): c.open_recent(target))
        if paths and c.clear_recent is not None:
            menu.addSeparator()
            menu.addAction("Clear the List", lambda: c.clear_recent())

    def _build_view(self) -> None:
        c, menu = self.commands, self.view_menu
        if c.show_workspace is not None:
            for number, item in enumerate(self.workspaces, start=1):
                self._add(
                    menu,
                    f"workspace_{item.key}",
                    item.title.replace("&", "&&"),
                    lambda key=item.key: c.show_workspace(key),
                    shortcut=f"Ctrl+{number}" if number <= 9 else None,
                    tip=item.description,
                )
            menu.addSeparator()
        if c.set_sidebar_collapsed is not None:
            # Not checkable (a check column would indent this one item): the text says what it does.
            action = QAction(self._sidebar_text(), self.window)
            action.setShortcut(QKeySequence("Ctrl+B"))
            action.triggered.connect(lambda _checked=False: self._toggle_sidebar())
            menu.addAction(action)
            self.actions["collapse_sidebar"] = action
        self._add(menu, "full_screen", "&Full Screen", c.toggle_full_screen,
                  shortcut=QKeySequence.FullScreen)
        if any(command is not None for command in (c.set_theme, c.change_font_size, c.set_language)):
            menu.addSeparator()
        if c.set_theme is not None:
            self.theme_menu = self._choice_menu(
                menu, "&Theme", "theme", (("light", "&Light"), ("dark", "&Dark")), c.current_theme(), c.set_theme)
        if c.change_font_size is not None:
            font_menu = self.font_menu = menu.addMenu("Font Si&ze")  # S is Collapse Sidebar's key
            self._submenus.append(font_menu)
            self._add(font_menu, "font_larger", "&Larger", lambda: c.change_font_size(+0.5),
                      shortcut=QKeySequence.ZoomIn)
            self._add(font_menu, "font_smaller", "&Smaller", lambda: c.change_font_size(-0.5),
                      shortcut=QKeySequence.ZoomOut)
            self._add(font_menu, "font_reset", "&Reset", c.reset_font_size, shortcut="Ctrl+0")
        if c.set_language is not None:
            # Each language in its own name (English, 中文), in either interface language: never translated.
            self.language_menu = self._choice_menu(
                menu, "&Language", "language", tuple(LANGUAGES.items()), c.current_language(), self._set_language)

    def _choice_menu(self, menu: QMenu, title: str, name: str, choices, current: str, choose) -> QMenu:
        """A submenu of exclusive checkable entries (Theme, Language); ``choose(key)`` on a click."""
        submenu = menu.addMenu(title)
        self._submenus.append(submenu)
        group = QActionGroup(self.window)
        group.setExclusive(True)
        for key, text in choices:
            action = QAction(text, self.window)
            action.setCheckable(True)
            action.setChecked(current == key)
            action.triggered.connect(lambda _checked=False, k=key: choose(k))
            group.addAction(action)
            submenu.addAction(action)
            self.actions[f"{name}_{key}"] = action
        return submenu

    def _set_language(self, key: str) -> None:
        self.commands.set_language(key)
        self.sync()

    def _build_tools(self) -> None:
        c, menu = self.commands, self.tools_menu
        self._add(menu, "geometry_calibration", "&Geometry Calibration…", c.geometry_calibration,
                  shortcut="Ctrl+Shift+G",
                  tip="Calibrate beam centre and detector distance from a standard (AgBh, LaB6 …)")
        menu.addSeparator()
        self._add(menu, "format_converter", "&Format Converter…", c.format_converter,
                  shortcut="Ctrl+Shift+C", tip="Convert NXS, CBF, TIFF and HDF5 detector images")
        self._add(menu, "convert_current_file", "Convert &Current File…", c.convert_current_file,
                  tip="Open the converter with the file shown in the active workspace")
        self._add(menu, "xrr_extractor", "&XRR Series Extractor…", c.xrr_extractor,
                  shortcut="Ctrl+Shift+R", tip="Extract XRR intensity from NXS or CBF series")
        menu.addSeparator()
        self._add(menu, "ai_fitting_workspace", "1D &Predict — Fit Many Curves…", c.ai_fitting_workspace,
                  tip="1D Predict: a list of curve files fitted one after another, with their results")
        self._add(menu, "claude_assistant", "Process with &AI…", c.claude_assistant,
                  shortcut="Ctrl+Shift+L",
                  tip="Let the AI (Claude, DeepSeek, Qwen, OpenAI, a local model …) analyse the frame shown in Analyze, "
                      "with the same tools; its changes come back as cards you can preview, apply or undo")
        menu.addSeparator()
        self._add(menu, "settings", "&Settings…", c.settings, shortcut="Ctrl+,",
                  tip="Appearance, analysis defaults and the user data folder")

    def _build_help(self) -> None:
        c, menu = self.commands, self.help_menu
        self._add(menu, "user_manual", "&User Manual", lambda: open_user_manual(self.window),
                  shortcut=QKeySequence.HelpContents)
        self._add(menu, "github", "&GitHub Repository",
                  lambda: QDesktopServices.openUrl(QUrl(GITHUB_URL)))
        self._add(menu, "open_data_folder", "Open User &Data Folder", c.open_data_folder,
                  tip="Settings, instrument profiles and the session are kept here")
        menu.addSeparator()
        self._add(menu, "about", f"&About {APP_NAME}", lambda: show_about(self.window))

    # -- state sync ---------------------------------------------------------

    def _sidebar_text(self) -> str:
        return tr("Expand &Sidebar") if self.commands.sidebar_collapsed() else tr("Collapse &Sidebar")

    def _toggle_sidebar(self) -> None:
        self.commands.set_sidebar_collapsed(not bool(self.commands.sidebar_collapsed()))
        self.sync()

    def sync(self) -> None:
        """Reflect state changed elsewhere (settings dialog, sidebar button)."""
        c = self.commands
        for name, chosen, keys in (("theme", c.current_theme(), ("light", "dark")),
                                   ("language", c.current_language(), tuple(LANGUAGES))):
            for key in keys:
                action = self.actions.get(f"{name}_{key}")
                if action is not None:
                    action.setChecked(key == chosen)
        action = self.actions.get("collapse_sidebar")
        if action is not None:
            action.setText(self._sidebar_text())


TOOL_GEOMETRY_KEY = "tool_windows.{name}.geometry"
"""Preference: ``[x, y, width, height]`` of a tool window as it was last closed (``name``: calibration, converter …)."""
SCREEN_SHARE = 0.9
"""A tool window takes at most this share of the screen's available width and height."""
TITLE_ROOM = 32
"""Room above a window for its title bar, so it can always be grabbed and moved."""
_TOOL_NAME = "gimapToolWindow"


def clamp_to_screen(rect: QRect, area: QRect, *, share: float = SCREEN_SHARE) -> QRect:
    """``rect`` at most ``share`` of ``area`` wide and high, moved (not shrunk further) to lie inside ``area`` with
    room for the title bar above it."""
    width = max(1, min(rect.width(), int(area.width() * share)))
    height = max(1, min(rect.height(), int(area.height() * share)))
    left = min(max(rect.x(), area.left()), area.right() + 1 - width)
    top = area.top() + min(TITLE_ROOM, max(0, area.height() - height))
    return QRect(left, min(max(rect.y(), top), area.bottom() + 1 - height), width, height)


class ToolWindows(QObject):
    """Single-instance modeless tool windows, created on demand.

    A new window opens where it was last closed (``preferences``), else at its own size; either way at most
    about 90 % of the screen and on it (a 1280 × 820 window on a 1366 × 768 laptop would put its buttons
    under the task bar)."""

    def __init__(self, parent: QWidget, preferences=None):
        super().__init__(parent)
        self.owner = parent
        self.preferences = preferences
        self._open: dict[str, QWidget] = {}

    def get(self, name: str) -> QWidget | None:
        return self._open.get(name)

    def windows(self) -> list[QWidget]:
        """The tool windows open now (shown or waiting hidden for a job to end)."""
        return list(self._open.values())

    def show(self, name: str, factory: Callable[[QWidget], QWidget], *, reuse=None) -> QWidget:
        window = self._open.get(name)
        if window is None:
            window = factory(self.owner)
            # The dict, not self: as the main window goes, this object may be deleted before its tool windows.
            window.destroyed.connect(lambda *_args, key=name, opened=self._open: opened.pop(key, None))
            self._open[name] = window
            self.place(name, window)
        elif reuse is not None:
            reuse(window)
        if window.isMinimized():
            window.showNormal()
        else:
            window.show()
        self.keep_on_screen(window)
        window.raise_()
        window.activateWindow()
        return window

    # -- geometry -----------------------------------------------------------

    def place(self, name: str, window: QWidget) -> None:
        """Before ``window`` is first shown: its last geometry, or its own size, clamped to the screen; and
        remembered when it is hidden or closed."""
        window.setProperty(_TOOL_NAME, name)
        window.installEventFilter(self)
        saved = self._saved(name)
        if saved is not None:
            screen = QApplication.screenAt(saved.center())
            area = screen.availableGeometry() if screen is not None else available_geometry(self.owner)
            window.setGeometry(clamp_to_screen(saved, area))
            return
        area = available_geometry(self.owner)
        size = clamp_to_screen(QRect(area.topLeft(), window.size()), area).size()
        window.resize(size)  # a dialog is centred on the main window as it is shown, inside its screen

    def keep_on_screen(self, window: QWidget) -> None:
        """Move a shown window back inside its screen (title bar included) when it lies partly outside."""
        frame = window.frameGeometry()
        screen = QApplication.screenAt(frame.center()) or window.screen()
        if screen is None:
            return
        area = screen.availableGeometry()
        if area.contains(frame):
            return
        # move() places the frame; one larger than the screen (a large minimum size) keeps its top left corner.
        x = min(max(frame.x(), area.left()), max(area.left(), area.right() + 1 - frame.width()))
        y = min(max(frame.y(), area.top()), max(area.top(), area.bottom() + 1 - frame.height()))
        window.move(x, y)

    def remember(self, name: str, window: QWidget) -> None:
        if self.preferences is None:
            return
        rect = window.normalGeometry() if window.isMaximized() or window.isFullScreen() else window.geometry()
        if rect.isValid() and not rect.isEmpty():
            self.preferences.set(TOOL_GEOMETRY_KEY.format(name=name),
                                 [rect.x(), rect.y(), rect.width(), rect.height()])

    def remember_all(self) -> None:
        """The geometry of every tool window still shown (as the main window closes, before preferences are saved)."""
        for name, window in list(self._open.items()):
            try:
                if window.isVisible():
                    self.remember(name, window)
            except RuntimeError:  # already gone
                continue

    def _saved(self, name: str) -> QRect | None:
        if self.preferences is None:
            return None
        value = self.preferences.get(TOOL_GEOMETRY_KEY.format(name=name))
        if not isinstance(value, (list, tuple)):
            return None
        try:
            x, y, width, height = (int(part) for part in value)
        except (TypeError, ValueError):
            return None
        return QRect(x, y, width, height) if width > 0 and height > 0 else None

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API
        # Not when the window system hides it (minimised): its geometry then is not the one to come back to.
        if event.type() == QEvent.Hide and not event.spontaneous() and isinstance(watched, QWidget):
            name = watched.property(_TOOL_NAME)
            if name:
                self.remember(str(name), watched)
        return False


def open_user_manual(parent: QWidget | None = None) -> None:
    """Render the Markdown manual to HTML and open it in the default browser."""
    if not USER_MANUAL.is_file():
        QMessageBox.warning(parent, tr("User Manual"),
                            tr("The user manual was not found:\n{path}").format(path=USER_MANUAL))
        return
    document = QTextDocument()
    document.setMarkdown(USER_MANUAL.read_text(encoding="utf-8"))
    html_path = Path(tempfile.gettempdir()) / "GIMaP_User_Manual.html"
    html_path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>GIMaP User Manual</title>"
        "<style>body{max-width:980px;margin:32px auto;padding:0 24px;line-height:1.58;"
        "font-family:'Segoe UI',sans-serif;color:#1f2933}code,pre{background:#f4f6f8;"
        "border-radius:4px}pre{padding:12px;overflow-x:auto}a{color:#0b63ce}</style>"
        f"</head><body>{document.toHtml()}</body></html>",
        encoding="utf-8",
    )
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(html_path)))


def show_about(parent: QWidget | None = None) -> None:
    dialog = QMessageBox(parent)
    dialog.setWindowTitle(tr("About {name}").format(name=APP_NAME))
    dialog.setWindowIcon(app_icon())
    dialog.setTextFormat(Qt.RichText)
    # The coloured logo is a dark blue G: on the dark theme the application icon (a white G) stands out instead.
    logo = app_icon().pixmap(88, 88) if theme_manager().mode == "dark" else app_colored_logo_pixmap(88, 88)
    if not logo.isNull():
        dialog.setIconPixmap(logo)
    release = tr("{name} v{version} (Pre-release)").format(name=APP_NAME, version=APP_VERSION)
    summary = tr("Desktop analysis of GISAXS / GIWAXS data: detector geometry, cuts, "
                 "fitting and machine-learning assisted workflows.")
    dialog.setText(
        f"<b style='font-size:16pt'>{APP_NAME}</b><br>{release}<br><br>{summary}<br><br>"
        f"<a href='{GITHUB_URL}'>{GITHUB_URL}</a>"
    )
    dialog.setStandardButtons(QMessageBox.Close)  # "Close" is in the zh table; Qt's own "OK" would stay English
    dialog.exec_()


__all__ = [
    "APP_NAME",
    "APP_RELEASE",
    "APP_VERSION",
    "GITHUB_URL",
    "MainMenuBar",
    "MenuCommands",
    "SCREEN_SHARE",
    "TOOL_GEOMETRY_KEY",
    "ToolWindows",
    "clamp_to_screen",
    "open_user_manual",
    "show_about",
]
