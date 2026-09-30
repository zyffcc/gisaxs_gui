"""Composition of the main menus: which workspace, dialog or setting each entry opens."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from .presentation.app_dialogs import ask_open_json, ask_save_json, inform, warn
from .presentation.components import show_toast
from .project import PROJECT_SUFFIX

RECENT_PROJECTS_KEY = "recent_projects"
from .presentation.menu_bar import MainMenuBar, MenuCommands, ToolWindows
from .presentation.navigation import NAVIGATION_ITEMS
from .presentation.settings_dialog import SettingsDialog
from .presentation.theme import theme_manager
from .presentation.theme.appearance import Appearance
from src.gimap.shared.file_paths import normalize_path


def _geometry_calibration_dialog(parent):
    from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog

    dialog = GeometryCalibrationDialog(parent)
    dialog.setModal(False)
    return dialog


def _format_converter_dialog(parent, *, current_file=""):
    from src.gimap.features.format_converter.presentation.dialog import FormatConverterDialog

    return FormatConverterDialog(parent, current_file=current_file)


def _xrr_series_dialog(parent):
    from src.gimap.features.xrr.presentation import XrrSeriesDialog

    return XrrSeriesDialog(parent)


class ApplicationMenus:
    """Owns the menu bar of the main window and the tool windows it opens."""

    def __init__(self, window):
        self.window = window
        self.tools = ToolWindows(window)
        self.appearance = Appearance(window.app_context.preferences)
        components = window.components
        commands = MenuCommands(
            open_files=lambda: self._analyze_action("open_files"),
            open_folder=lambda: self._analyze_action("open_folder"),
            recent_paths=self.recent_paths,
            open_recent=self.open_recent,
            clear_recent=lambda: components.analyze_page.clear_recent(),
            save_parameters=self.save_parameters,
            load_parameters=self.load_parameters,
            open_project=self.open_project,
            save_project=self.save_project,
            save_project_as=self.save_project_as,
            quit=window.close,
            show_workspace=self.show_workspace,
            set_theme=self.appearance.set_theme,
            current_theme=lambda: theme_manager().mode,
            change_font_size=self.appearance.change_font,
            reset_font_size=self.appearance.reset_font,
            set_sidebar_collapsed=components.set_sidebar_collapsed,
            sidebar_collapsed=components.sidebar.is_collapsed,
            toggle_full_screen=self.toggle_full_screen,
            geometry_calibration=self.open_geometry_calibration,
            format_converter=lambda: self.open_format_converter(include_current=False),
            convert_current_file=lambda: self.open_format_converter(include_current=True),
            xrr_extractor=lambda: self.tools.show("xrr", _xrr_series_dialog),
            ai_fitting_workspace=self.open_ai_fitting_workspace,
            claude_assistant=lambda: components.assistant().start(),
            settings=self.open_settings,
            open_data_folder=self.open_data_folder,
        )
        self.menu_bar = MainMenuBar(window, commands, NAVIGATION_ITEMS)
        theme_manager().changed.connect(lambda _mode: self.menu_bar.sync())
        components.sidebar.collapsedChanged.connect(lambda _collapsed: self.menu_bar.sync())

    # -- workspaces ---------------------------------------------------------

    def show_workspace(self, key: str) -> None:
        runtime = getattr(self.window, "runtime", None)
        if runtime is not None:
            runtime.navigate(key)
        else:
            self.window.components.show_page(key)

    def _analyze_action(self, name: str) -> None:
        self.show_workspace("analyze")
        getattr(self.window.components.analyze_page, name)()

    def open_recent(self, path: str) -> None:
        if str(path).lower().endswith(PROJECT_SUFFIX):
            self.open_project(path)
            return
        self.show_workspace("analyze")
        self.window.components.analyze_page.add_paths([path])

    def recent_paths(self) -> list:
        """Recent projects first, then the recent data of Analyze."""
        projects = [path for path in self._recent_projects() if Path(path).exists()]
        return (projects + list(self.window.components.analyze_page.recent_paths()))[:12]

    # -- projects -------------------------------------------------------------

    project_path = ""

    def _recent_projects(self) -> list:
        stored = self.window.app_context.preferences.get(RECENT_PROJECTS_KEY, [])
        return [str(path) for path in stored] if isinstance(stored, list) else []

    def _remember_project(self, path) -> None:
        self.project_path = str(path)
        recent = [str(path)] + [item for item in self._recent_projects() if item != str(path)]
        self.window.app_context.preferences.set(RECENT_PROJECTS_KEY, recent[:6])
        self.window.setWindowTitle(f"GIMaP — {Path(path).stem}")

    def open_project(self, path=None) -> bool:
        from . import project

        if not path:
            folder = str(Path(self.project_path).parent) if self.project_path else ""
            path = ask_open_json(self.window, "Open Project", folder, project.PROJECT_FILTER)
            if not path:
                return False
        try:
            data = project.read(path)
        except (OSError, ValueError) as exc:
            warn(self.window, "Open Project", f"{Path(path).name} could not be opened: {exc}")
            return False
        components = self.window.components
        notes = project.apply(components, data)
        self._remember_project(path)
        page = data.get("page") if data.get("page") in components.pages else "analyze"
        self.show_workspace(page)
        text = f"Opened the project {Path(path).name}" + (": " + "; ".join(notes) if notes else ".")
        show_toast(self.window, text, level="warning" if notes else "ok")
        return True

    def save_project(self) -> bool:
        if not self.project_path:
            return self.save_project_as()
        return self._write_project(self.project_path)

    def save_project_as(self) -> bool:
        from . import project

        components = self.window.components
        files = components.analyze_page.view_model.state.files
        suggested = Path(self.project_path) if self.project_path else (
            Path(files[0]).parent / f"{Path(files[0]).stem}.gimap" if files else Path("sample.gimap"))
        path = ask_save_json(self.window, "Save Project", str(suggested), project.PROJECT_FILTER)
        return bool(path) and self._write_project(path)

    def _write_project(self, path) -> bool:
        from . import project

        try:
            written = project.save(self.window.components, path, self.window.components.current_page_key() or "")
        except OSError as exc:
            warn(self.window, "Save Project", f"The project could not be saved: {exc}")
            return False
        self._remember_project(written)
        show_toast(self.window, f"Saved the project {written.name}", level="ok",
                   action=("Open Folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(written.parent)))))
        return True

    def toggle_full_screen(self) -> None:
        if self.window.isFullScreen():
            self.window.showNormal()
        else:
            self.window.showFullScreen()

    # -- parameters ---------------------------------------------------------

    def _runtime_or_warn(self, title: str):
        runtime = getattr(self.window, "runtime", None)
        if runtime is None:
            inform(self.window, title, "GIMaP is still starting; try again in a moment.")
        return runtime

    def save_parameters(self) -> None:
        runtime = self._runtime_or_warn("Save Labs Parameters")
        if runtime is None:
            return
        path = ask_save_json(self.window, "Save Labs Parameters", "gimap_parameters.json")
        if not path:
            return
        if runtime.save_parameters_to_file(normalize_path(path)):
            folder = Path(normalize_path(path)).parent
            show_toast(self.window, f"Saved {path}", level="ok",
                       action=("Open Folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))))
        else:
            warn(self.window, "Save Labs Parameters", "The parameters could not be saved.")

    def load_parameters(self) -> None:
        runtime = self._runtime_or_warn("Load Labs Parameters")
        if runtime is None:
            return
        path = ask_open_json(self.window, "Load Labs Parameters")
        if not path:
            return
        if runtime.load_parameters_from_file(normalize_path(path)):
            show_toast(self.window, f"Loaded {path}", level="ok")
        else:
            warn(self.window, "Load Labs Parameters", "The parameters could not be loaded.")

    # -- tools --------------------------------------------------------------

    def open_geometry_calibration(self):
        return self.tools.show("calibration", _geometry_calibration_dialog)

    def current_detector_file(self) -> str:
        """File shown in the active workspace, if it shows one."""
        components = self.window.components
        key = components.current_page_key()
        if key == "analyze":
            path = components.analyze_page.view_model.current_path
            return str(path) if path else ""
        return ""

    def open_format_converter(self, *, include_current: bool):
        current = self.current_detector_file() if include_current else ""

        def reuse(dialog) -> None:
            if current:
                dialog.current_file = current
                dialog.current_button.setEnabled(True)
                dialog.add_paths([current])

        return self.tools.show(
            "converter",
            lambda parent: _format_converter_dialog(parent, current_file=current),
            reuse=reuse,
        )

    def open_ai_fitting_workspace(self) -> None:
        runtime = self._runtime_or_warn("AI Fitting")
        if runtime is not None:
            runtime.fitting.open_ai_fitting_workspace()

    def open_settings(self) -> None:
        context = self.window.app_context
        store = getattr(context.settings, "store", None)
        dialog = SettingsDialog(
            self.window,
            preferences=context.preferences,
            settings=context.settings,
            data_dir=context.data_dir,
            migrated_from=getattr(store, "migrated_from", None),
            extra_pages=(
                (
                    "Assistant",
                    "The AI runs the Analyze tools on the open GIWAXS frame and reports what it finds.",
                    lambda parent: self.window.components.assistant().settings_page(parent),
                ),
            ),
        )
        dialog.exec_()
        self.menu_bar.sync()

    def open_data_folder(self) -> None:
        folder = self.window.app_context.data_dir
        if folder is None:
            inform(self.window, "User Data", "This session keeps its data in memory.")
            return
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


__all__ = ["ApplicationMenus"]
