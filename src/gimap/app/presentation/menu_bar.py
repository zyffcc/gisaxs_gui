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

from PyQt5.QtCore import QObject, Qt, QUrl
from PyQt5.QtGui import QDesktopServices, QKeySequence, QTextDocument
from PyQt5.QtWidgets import QAction, QActionGroup, QMainWindow, QMenu, QMessageBox, QWidget

from .assets import app_colored_logo_pixmap, app_icon
from .navigation import NavigationItem

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
        self._add(menu, "open_folder", "Open &Folder…", c.open_folder,
                  shortcut="Ctrl+Shift+O", tip="Open every frame of a folder in Analyze")
        if c.open_recent is not None:
            self.recent_menu = menu.addMenu("Open &Recent")
            self.recent_menu.aboutToShow.connect(self._fill_recent)
        menu.addSeparator()
        self._add(menu, "open_project", "Open &Project…", c.open_project, shortcut="Ctrl+Shift+P",
                  tip="Reopen a sample as it was left: the frames and set-up of Analyze, the curve and model of Fitting")
        self._add(menu, "save_project", "&Save Project", c.save_project, shortcut=QKeySequence.Save,
                  tip="Save what is open in Analyze and Fitting as a project (.gimap)")
        self._add(menu, "save_project_as", "Save Project &As…", c.save_project_as, shortcut="Ctrl+Shift+S",
                  tip="Save the project under another name")
        menu.addSeparator()
        if c.load_parameters is not None or c.save_parameters is not None:
            labs = menu.addMenu("&Labs Parameters")
            labs.setToolTip("The settings of 2D Prediction and Trainset Build")
            self._add(labs, "load_parameters", "&Load…", c.load_parameters,
                      tip="Load the settings of the Labs pages (2D Prediction, Trainset Build) from a JSON file")
            self._add(labs, "save_parameters", "&Save As…", c.save_parameters,
                      tip="Save the settings of the Labs pages to a JSON file")
            menu.addSeparator()
        self._add(menu, "quit", "E&xit", c.quit, shortcut=QKeySequence.Quit)

    def _fill_recent(self) -> None:
        """The files and folders opened last, newest first (built each time the menu opens)."""
        from pathlib import Path

        menu, c = self.recent_menu, self.commands
        menu.clear()
        paths = list(c.recent_paths() or [])
        if not paths:
            empty = menu.addAction("No recent data")
            empty.setEnabled(False)
        for number, path in enumerate(paths, start=1):
            path = Path(path)
            kind = "  (folder)" if path.is_dir() else "  (project)" if path.suffix.lower() == ".gimap" else ""
            label = f"&{number}  {path.name or str(path)}" + kind
            action = menu.addAction(label)
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
            action = QAction("Collapse &Sidebar", self.window)
            action.setCheckable(True)
            action.setChecked(bool(c.sidebar_collapsed()))
            action.setShortcut(QKeySequence("Ctrl+B"))
            action.toggled.connect(lambda checked: c.set_sidebar_collapsed(bool(checked)))
            menu.addAction(action)
            self.actions["collapse_sidebar"] = action
        self._add(menu, "full_screen", "&Full Screen", c.toggle_full_screen,
                  shortcut=QKeySequence.FullScreen)
        if c.set_theme is not None:
            menu.addSeparator()
            theme_menu = menu.addMenu("&Theme")
            group = QActionGroup(self.window)
            group.setExclusive(True)
            for mode, title in (("light", "&Light"), ("dark", "&Dark")):
                action = QAction(title, self.window)
                action.setCheckable(True)
                action.setChecked(c.current_theme() == mode)
                action.triggered.connect(lambda _checked=False, m=mode: c.set_theme(m))
                group.addAction(action)
                theme_menu.addAction(action)
                self.actions[f"theme_{mode}"] = action
        if c.change_font_size is not None:
            font_menu = menu.addMenu("Font &Size")
            self._add(font_menu, "font_larger", "&Larger", lambda: c.change_font_size(+0.5),
                      shortcut=QKeySequence.ZoomIn)
            self._add(font_menu, "font_smaller", "&Smaller", lambda: c.change_font_size(-0.5),
                      shortcut=QKeySequence.ZoomOut)
            self._add(font_menu, "font_reset", "&Reset", c.reset_font_size, shortcut="Ctrl+0")

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
        self._add(menu, "ai_fitting_workspace", "&Fit Settings && Batch…", c.ai_fitting_workspace,
                  tip="The fitting method, components and limits, and fitting many curve files at once")
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

    def sync(self) -> None:
        """Reflect state changed elsewhere (settings dialog, sidebar button)."""
        c = self.commands
        mode = c.current_theme()
        for key in ("light", "dark"):
            action = self.actions.get(f"theme_{key}")
            if action is not None:
                action.setChecked(key == mode)
        action = self.actions.get("collapse_sidebar")
        if action is not None and action.isChecked() != bool(c.sidebar_collapsed()):
            action.blockSignals(True)
            action.setChecked(bool(c.sidebar_collapsed()))
            action.blockSignals(False)


class ToolWindows:
    """Single-instance modeless tool windows, created on demand."""

    def __init__(self, parent: QWidget):
        self.parent = parent
        self._open: dict[str, QWidget] = {}

    def get(self, name: str) -> QWidget | None:
        return self._open.get(name)

    def show(self, name: str, factory: Callable[[QWidget], QWidget], *, reuse=None) -> QWidget:
        window = self._open.get(name)
        if window is None:
            window = factory(self.parent)
            window.destroyed.connect(lambda *_args, key=name: self._open.pop(key, None))
            self._open[name] = window
        elif reuse is not None:
            reuse(window)
        window.show()
        window.raise_()
        window.activateWindow()
        return window


def open_user_manual(parent: QWidget | None = None) -> None:
    """Render the Markdown manual to HTML and open it in the default browser."""
    if not USER_MANUAL.is_file():
        QMessageBox.warning(parent, "User Manual", f"The user manual was not found:\n{USER_MANUAL}")
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
    dialog.setWindowTitle(f"About {APP_NAME}")
    dialog.setWindowIcon(app_icon())
    dialog.setTextFormat(Qt.RichText)
    logo = app_colored_logo_pixmap(88, 88)
    if not logo.isNull():
        dialog.setIconPixmap(logo)
    dialog.setText(
        f"<b style='font-size:16pt'>{APP_NAME}</b><br>{APP_RELEASE}<br><br>"
        "Desktop analysis of GISAXS / GIWAXS data: detector geometry, cuts, "
        "fitting and machine-learning assisted workflows.<br><br>"
        f"<a href='{GITHUB_URL}'>{GITHUB_URL}</a>"
    )
    dialog.setStandardButtons(QMessageBox.Ok)
    dialog.exec_()


__all__ = [
    "APP_NAME",
    "APP_RELEASE",
    "APP_VERSION",
    "GITHUB_URL",
    "MainMenuBar",
    "MenuCommands",
    "ToolWindows",
    "open_user_manual",
    "show_about",
]
