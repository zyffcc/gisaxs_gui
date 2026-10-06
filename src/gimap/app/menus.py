"""Composition of the main menus: which workspace, dialog or setting each entry opens."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QApplication

from .presentation.app_dialogs import JSON_FILTER, ask_open_json, ask_save_json, inform, warn
from .presentation.components import show_toast
from .presentation.i18n import current_language, language_changed, tr, trf
from .presentation.menu_bar import MainMenuBar, MenuCommands, ToolWindows
from .presentation.navigation import NAVIGATION_ITEMS
from .presentation.settings_dialog import SettingsDialog
from .presentation.theme import theme_manager
from .presentation.theme.appearance import Appearance
from .project import PROJECT_SUFFIX
from src.gimap.shared.file_paths import normalize_path

RECENT_PROJECTS_KEY = "recent_projects"


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
        self.tools = ToolWindows(window, window.app_context.preferences)  # each tool window where it was left
        self.appearance = Appearance(window.app_context.preferences)
        components = window.components
        commands = MenuCommands(
            open_files=self.open_files,
            open_files_label=self.open_files_label,
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
            set_language=self.set_language,
            current_language=current_language,
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
            claude_assistant=components.ask_ai,  # as Analyze's Ask AI: with the notes of the Results step
            settings=self.open_settings,
            open_data_folder=self.open_data_folder,
        )
        self.menu_bar = MainMenuBar(window, commands, NAVIGATION_ITEMS)
        # View ▸ Theme and View ▸ Language show what is applied, however it was chosen (Settings, or the saved
        # one at start). Both signals outlive this window: _sync_menus ignores a window that is gone.
        theme_manager().changed.connect(self._sync_menus)
        language_changed().connect(self._sync_menus)
        components.sidebar.collapsedChanged.connect(lambda _collapsed: self.menu_bar.sync())
        # The title names the project and the frame shown. After Analyze ▸ Clear the data are no
        # longer the project's, so Save Project asks for a file instead of overwriting it.
        page = components.analyze_page
        cleared = getattr(page, "filesCleared", None)  # the Analyze page's signal (contract)
        if cleared is not None:
            cleared.connect(self._files_cleared)
        page.analysisShown.connect(lambda _analysis: self._sync_title())
        failed = getattr(page, "analysisFailed", None)  # a frame that could not be read is still the current one
        if failed is not None:
            failed.connect(lambda _message: self._sync_title())
        self._sync_title()

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

    def _curve_page_shown(self) -> bool:
        """Fitting's Single analysis is shown (In-situ series hides it and keeps Open Data)."""
        components = self.window.components
        return components.current_page_key() == "fitting" and components.fitting_workspace.fit_page.isVisible()

    def open_files(self) -> None:
        """File ▸ Open Data (Ctrl+O): a curve on Fitting's Single analysis, detector frames elsewhere."""
        if self._curve_page_shown():
            self.window.components.fitting_workspace.fit_page.open_curve_dialog()
            return
        self._analyze_action("open_files")

    def open_files_label(self) -> tuple[str, str]:
        """(text, tip) of File ▸ Open Data as the File menu opens: it says what Ctrl+O does on this page."""
        if self._curve_page_shown():
            return tr("Open Curve…"), tr("Open a 1D curve (q, I, σ) — Analyze ▸ Send to Fitting opens its cut here")
        return tr("&Open Data…"), tr("Open detector frames (CBF, NXS, TIFF, EDF) in Analyze")

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
        """Stored projects, one spelling each (``str(Path)``; ``C:/a.gimap`` and ``C:\\a.gimap`` are one)."""
        stored = self.window.app_context.preferences.get(RECENT_PROJECTS_KEY, [])
        recent, seen = [], set()
        for item in stored if isinstance(stored, list) else []:
            if not item:
                continue
            path = str(Path(str(item)))
            if path.casefold() not in seen:
                seen.add(path.casefold())
                recent.append(path)
        return recent

    def _remember_project(self, path) -> None:
        key = str(Path(path))  # Save gives backslashes, the Open dialog forward slashes: one entry
        self.project_path = key
        recent = [key] + [item for item in self._recent_projects() if item.casefold() != key.casefold()]
        self.window.app_context.preferences.set(RECENT_PROJECTS_KEY, recent[:6])
        self._sync_title()

    def _files_cleared(self) -> None:
        """Analyze ▸ Clear: what is open next is not the project any more (Save Project asks where)."""
        self.project_path = ""
        self._sync_title()

    def window_title(self) -> str:
        """``GIMaP — project · frame``, ``GIMaP — frame``, ``GIMaP — project`` or ``GIMaP``."""
        path = getattr(self.window.components.analyze_page.view_model, "current_path", None)
        frame = Path(path).name if path else ""
        project = Path(self.project_path).stem if self.project_path else ""
        shown = " · ".join(part for part in (project, frame) if part)
        return f"GIMaP — {shown}" if shown else "GIMaP"

    def _sync_title(self) -> None:
        self.window.setWindowTitle(self.window_title())

    def open_project(self, path=None) -> bool:
        from . import project

        if self._jobs_keep_the_project_closed():
            return False
        if not path:
            folder = str(Path(self.project_path).parent) if self.project_path else ""
            path = ask_open_json(self.window, tr("Open Project"), folder, tr(project.PROJECT_FILTER))
            if not path:
                return False
        try:
            data = project.read(path)
        except (OSError, ValueError) as exc:  # "not a GIMaP project" and the like are in the table; others as they are
            warn(self.window, tr("Open Project"),
                 trf("{name} could not be opened: {error}", name=Path(path).name, error=tr(str(exc))))
            return False
        components = self.window.components
        notes = project.apply(components, data, path)
        self._remember_project(path)
        page = data.get("page") if data.get("page") in components.pages else "analyze"
        self.show_workspace(page)
        name = Path(path).name
        text = (trf("Opened the project {name}: {notes}", name=name, notes="; ".join(notes)) if notes
                else trf("Opened the project {name}.", name=name))
        show_toast(self.window, text, level="warning" if notes else "ok")
        return True

    def _jobs_keep_the_project_closed(self) -> bool:
        """A project replaces what Analyze, Fitting and Compare hold, so it opens only when no job runs: a
        running In-situ series would otherwise go on under the new project and Save would write its frames
        there. Refused before anything is asked or applied (the project, the title and Recent stay). A job
        cannot be stopped and waited for here: Stop lets the frame in progress finish."""
        jobs = self.window.components.running_jobs()
        if jobs:
            warn(self.window, tr("Open Project"),
                 trf("Stop {jobs} first, then open the project.", jobs=self.window.components.jobs_text(jobs)))
        return bool(jobs)

    def _project_folder(self) -> Path:
        """Where Save Project As… suggests a project with no frame listed: the folder Analyze opened last, else
        the home folder (never a bare name, which would land in the working folder)."""
        last = getattr(self.window.components.analyze_page, "last_folder", "")  # Analyze's contract
        folder = Path(last) if isinstance(last, str) and last else Path.home()
        return folder if folder.is_dir() else Path.home()

    def save_project(self) -> bool:
        if not self.project_path:
            return self.save_project_as()
        return self._write_project(self.project_path)

    def save_project_as(self) -> bool:
        from . import project

        components = self.window.components
        files = components.analyze_page.view_model.state.files
        if self.project_path:
            suggested = Path(self.project_path)
        elif files:
            suggested = Path(files[0]).parent / f"{Path(files[0]).stem}.gimap"
        else:  # nothing listed (after Clear): a real folder, not a bare name in the working folder
            suggested = self._project_folder() / "sample.gimap"
        path = ask_save_json(self.window, tr("Save Project"), str(suggested), tr(project.PROJECT_FILTER))
        return bool(path) and self._write_project(path)

    def _write_project(self, path) -> bool:
        from . import project

        try:
            written = project.save(self.window.components, path, self.window.components.current_page_key() or "")
        except OSError as exc:
            warn(self.window, tr("Save Project"), trf("The project could not be saved: {error}", error=exc))
            return False
        self._remember_project(written)
        show_toast(self.window, tr("Saved the project {name}").format(name=written.name), level="ok",
                   action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(written.parent)))))
        return True

    def set_language(self, language: str) -> None:
        """View ▸ Language: as Settings ▸ Appearance ▸ Language, saved and applied to every open window."""
        self.appearance.set_language(language, QApplication.topLevelWidgets())

    def _sync_menus(self, *_args) -> None:
        try:
            self.menu_bar.sync()
        except RuntimeError:  # the window is gone
            pass

    def toggle_full_screen(self) -> None:
        if self.window.isFullScreen():
            self.window.showNormal()
        else:
            self.window.showFullScreen()

    # -- parameters ---------------------------------------------------------

    def _runtime_or_warn(self, title: str):
        runtime = getattr(self.window, "runtime", None)
        if runtime is None:
            inform(self.window, tr(title), tr("GIMaP is still starting; try again in a moment."))
        return runtime

    def save_parameters(self) -> None:
        runtime = self._runtime_or_warn("Save Labs Parameters")
        if runtime is None:
            return
        path = ask_save_json(self.window, tr("Save Labs Parameters"), "gimap_parameters.json", tr(JSON_FILTER))
        if not path:
            return
        if runtime.save_parameters_to_file(normalize_path(path)):
            written = Path(normalize_path(path))
            show_toast(self.window, trf("Saved {name}", name=written.name), level="ok",
                       action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(written.parent)))))
        else:
            warn(self.window, tr("Save Labs Parameters"), tr("The parameters could not be saved."))

    def load_parameters(self) -> None:
        runtime = self._runtime_or_warn("Load Labs Parameters")
        if runtime is None:
            return
        path = ask_open_json(self.window, tr("Load Labs Parameters"), "", tr(JSON_FILTER))
        if not path:
            return
        if runtime.load_parameters_from_file(normalize_path(path)):
            show_toast(self.window, trf("Loaded {name}", name=Path(normalize_path(path)).name), level="ok")
        else:
            warn(self.window, tr("Load Labs Parameters"), tr("The parameters could not be loaded."))

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
        runtime = self._runtime_or_warn("1D Predict")
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
        dialog.deleteLater()  # not kept as a hidden child of the window (walked at every language switch)
        self.menu_bar.sync()

    def open_data_folder(self) -> None:
        folder = self.window.app_context.data_dir
        if folder is None:
            inform(self.window, tr("User Data"), tr("This session keeps its data in memory."))
            return
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


__all__ = ["ApplicationMenus"]
