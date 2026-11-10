"""Main menu bar: File, Edit, View, Analyze, Fitting, Tools and Help.

The menus only know commands (plain callables); the composition root in
``src.gimap.app.menus`` decides what each command does.  A command that is
not provided hides its action, and a menu left without any is not added, so the
builder has no feature imports. Edit, Analyze and Fitting are built by
``workspace_menus.py``; the tool windows live in ``tool_windows.py``; the documents of Help are rendered by
``help_documents.py`` and its windows are in ``help_dialogs.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from PyQt5.QtCore import QObject, QUrl
from PyQt5.QtGui import QDesktopServices, QKeySequence
from PyQt5.QtWidgets import QAction, QActionGroup, QMainWindow, QMenu, QWidget

from .help_dialogs import show_about_dialog, show_citation, show_keyboard_shortcuts
from .help_documents import (  # noqa: F401 - USER_MANUAL, USER_MANUAL_ZH, manual_for_language: re-exported
    USER_MANUAL,
    USER_MANUAL_ZH,
    manual_for_language,
    open_physics_manual,
    open_third_party_notices,
    open_user_manual,
)
from .i18n import LANGUAGES, tr, trf
from .navigation import NavigationItem
from .recent_items import recent_label
from .tool_windows import SCREEN_SHARE, TOOL_GEOMETRY_KEY, ToolWindows, clamp_to_screen
from .workspace_menus import ANALYZE_MENU, EDIT_MENU, FITTING_MENU, WorkspaceMenus
from src.gimap.shared.version import APP_NAME, PRE_RELEASE, __version__

APP_VERSION = __version__  # written once, in shared/version.py
GITHUB_URL = "https://github.com/zyffcc/gisaxs_gui"
ISSUES_URL = GITHUB_URL + "/issues/new"
RELEASES_URL = GITHUB_URL + "/releases"  # newest first, pre-releases too: GitHub's /releases/latest never shows one
"""Check for Updates opens this page; GIMaP itself makes no network request (beamline computers are often offline)."""

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
    sem_analysis: Command = None
    ai_fitting_workspace: Command = None
    claude_assistant: Command = None
    settings: Command = None
    open_data_folder: Command = None
    examples: Sequence = ()
    """Help ▸ Example Data: ``(key, text, tip)`` of each example GIMaP comes with (``app/help_commands.py``)."""
    open_example: Optional[Callable[[str], object]] = None
    report_problem: Command = None
    open_error_log: Command = None
    citation: Optional[Callable[[], object]] = None
    """How to Cite: an object with ``text``, ``bibtex`` and ``doi`` (read from ``CITATION.cff``)."""
    page_entries: dict = field(default_factory=dict)
    """The entries of Edit, Analyze and Fitting: ``workspace_menus.MenuEntry`` by key (``workspace_menus.py``)."""


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
        self.file_menu = QMenu("&File", menubar)
        self.edit_menu = QMenu("&Edit", menubar)
        self.view_menu = QMenu("&View", menubar)
        self.analyze_menu = QMenu("&Analyze", menubar)
        self.fitting_menu = QMenu("F&itting", menubar)  # I: F is File's key
        self.tools_menu = QMenu("&Tools", menubar)
        self.help_menu = QMenu("&Help", menubar)
        self.workspace_menus = WorkspaceMenus(window, commands.page_entries, self.actions, self._submenus)
        self._build_file()
        self.workspace_menus.fill(self.edit_menu, EDIT_MENU)
        self._build_view()
        self.workspace_menus.fill(self.analyze_menu, ANALYZE_MENU)
        self.workspace_menus.fill(self.fitting_menu, FITTING_MENU)
        self._build_tools()
        self._build_help()
        # A menu with no command provided is not shown (an empty menu would only say that nothing is there).
        self.menus = (self.file_menu, self.edit_menu, self.view_menu, self.analyze_menu, self.fitting_menu,
                      self.tools_menu, self.help_menu)
        for menu in self.menus:
            if any(not action.isSeparator() for action in menu.actions()):
                menubar.addMenu(menu)
        # The status bar is hidden on most pages: the descriptions show as tooltips in the menus.
        for menu in (*self.menus, *self._submenus):
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
                  shortcut=QKeySequence.Open, tip="Open detector frames (CBF, NeXus/HDF5, TIFF, EDF) in Analyze")
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
            # On the entry in File (a QMenu's own tooltip is never shown there).
            labs.menuAction().setToolTip("The settings of 2D Prediction, and Fitting's model")
            self._submenus.append(labs)
            self._add(labs, "load_parameters", "&Load…", c.load_parameters,
                      tip="Load the settings of the Labs pages (2D Prediction) and Fitting's model from a "
                          "JSON file; Fitting's model is replaced")
            self._add(labs, "save_parameters", "&Save As…", c.save_parameters,
                      tip="Save the settings of the Labs pages and Fitting's model to a JSON file")
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
            section = None
            for number, item in enumerate(self.workspaces, start=1):
                if item.section and item.section != section:  # "Workspaces", "Labs": as in the sidebar
                    section = item.section
                    menu.addSection(section)
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
        self._add(menu, "sem_analysis", "SE&M Particle Analysis…", c.sem_analysis,  # S is Settings' key
                  tip="Extract particle sizes from SEM images or compare SEM data with a radius prediction")
        menu.addSeparator()
        self._add(menu, "ai_fitting_workspace", "1D &Predict — Fit Many Curves…", c.ai_fitting_workspace,
                  tip="1D Predict: a list of curve files fitted one after another, with their results")
        self._add(menu, "claude_assistant", "Process with &AI…", c.claude_assistant,
                  shortcut="Ctrl+Shift+L",
                  tip="Let the AI (Claude, DeepSeek, Qwen, OpenAI, a local model …) analyse the frame shown in Analyze, "
                      "with the same tools; its changes come back as cards you can preview, apply or undo")
        menu.addSeparator()
        self._add(menu, "settings", "&Settings…", c.settings, shortcut="Ctrl+,",
                  tip="Appearance, analysis defaults, the AI assistant and the user data folder")

    def _build_help(self) -> None:
        c, menu, window = self.commands, self.help_menu, self.window
        self._add(menu, "user_manual", "&User Manual", lambda: open_user_manual(window),
                  shortcut=QKeySequence.HelpContents, tip="The manual of GIMaP, opened in your web browser")
        self._add(menu, "physics_manual", "&Physics and Methods", lambda: open_physics_manual(window),
                  tip="How GIMaP computes what it shows: geometry, q, corrections, cuts and models (in your web browser)")
        self._add(menu, "keyboard_shortcuts", "&Keyboard Shortcuts…", lambda: show_keyboard_shortcuts(window),
                  tip="Every key of the menus and of the pages, in a table you can copy")
        if c.open_example is not None and c.examples:
            examples = self.examples_menu = menu.addMenu("E&xample Data")
            examples.menuAction().setToolTip(
                "Public example frames that come with GIMaP: one is copied to a folder of your choice and opened in Analyze")
            self._submenus.append(examples)
            for key, text, tip in c.examples:
                self._add(examples, f"example_{key}", text, lambda key=key: c.open_example(key), tip=tip)
        menu.addSeparator()
        self._add(menu, "report_problem", "&Report a Problem…", c.report_problem,
                  tip="Copies the versions of GIMaP and of your system (no paths, no data) and opens a new issue "
                      "on GitHub in your browser")
        self._add(menu, "error_log", "Open Error &Log Folder", c.open_error_log,
                  tip="The folder of errors.log, where every unexpected error is written")
        self._add(menu, "open_data_folder", "Open User &Data Folder", c.open_data_folder,
                  tip="Settings, instrument profiles and the session are kept here")
        menu.addSeparator()
        self._add(menu, "github", "&GitHub Repository",
                  lambda: QDesktopServices.openUrl(QUrl(GITHUB_URL)))
        self._add(menu, "check_updates", "Check for Upda&tes…", lambda: QDesktopServices.openUrl(QUrl(RELEASES_URL)))
        if c.citation is not None:
            self._add(menu, "cite", "How to &Cite GIMaP…", self._show_citation,
                      tip="The reference to give when you publish results obtained with GIMaP")
        self._add(menu, "third_party_notices", "Third-party &Notices", lambda: open_third_party_notices(window),
                  tip="What GIMaP and its Windows package contain that is not GIMaP's own work, and the licences")
        menu.addSeparator()
        self._add(menu, "about", f"&About {APP_NAME}", lambda: show_about(window))
        menu.aboutToShow.connect(self._sync_help)
        self._sync_help()

    def _show_citation(self):
        citation = self.commands.citation()
        return show_citation(self.window, citation.text, citation.bibtex, doi=citation.doi)

    def _sync_help(self) -> None:
        """Check for Updates names the version running (composed: in the interface language)."""
        action = self.actions.get("check_updates")
        if action is not None:
            tip = trf("You are running {name} {version}. This opens the releases page on GitHub in your browser "
                      "(newest first, pre-releases included); GIMaP itself makes no network request.",
                      name=APP_NAME, version=APP_VERSION)
            action.setToolTip(tip)
            action.setStatusTip(tip)

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
        self._sync_help()


def release_text() -> str:
    """``GIMaP v<version> (Pre-release)`` while ``shared/version.PRE_RELEASE`` says so, else ``GIMaP v<version>``
    (``<version>``: ``shared/version.__version__``)."""
    if PRE_RELEASE:
        return tr("{name} v{version} (Pre-release)").format(name=APP_NAME, version=APP_VERSION)
    return tr("{name} v{version}").format(name=APP_NAME, version=APP_VERSION)


def show_about(parent: QWidget | None = None):
    """Help ▸ About GIMaP: the release, the licence, the versions it runs on, with Copy (``help_dialogs.py``)."""
    return show_about_dialog(parent, release_text, GITHUB_URL)


__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "GITHUB_URL",
    "ISSUES_URL",
    "RELEASES_URL",
    "MainMenuBar",
    "MenuCommands",
    "SCREEN_SHARE",
    "TOOL_GEOMETRY_KEY",
    "ToolWindows",
    "clamp_to_screen",
    "manual_for_language",
    "open_user_manual",
    "release_text",
    "show_about",
]
