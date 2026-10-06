"""Composition of the main window: feature pages, sidebar and page switching.

``ApplicationWindowView`` creates the shell widgets and the classic Fitting /
Prediction controls; this module builds the remaining feature pages into the
page stack and owns navigation between them.  Workspaces are addressed by the
keys of ``NAVIGATION_ITEMS`` ("analyze", "fitting", ...).
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path

from PyQt5.QtCore import QCoreApplication, QEventLoop
from PyQt5.QtWidgets import QVBoxLayout

from .presentation.collapsible_card import (
    CardContentResizeHandle,
    CollapsibleCardFrame,
)
from .presentation.components import show_toast
from .presentation.i18n import current_language, language_changed, tr, translate
from .presentation.layout_metrics import LAYOUT
from .presentation.navigation import NAVIGATION_ITEMS, NavigationSidebar
from .presentation.task_runner import TaskRunner
from src.gimap.features.prediction.presentation import (
    GisaxsPredictWorkspace,
    PredictCard,
    PredictModelLibraryCard,
)
from src.gimap.features.trainset.presentation import TrainsetBuildPage
from src.gimap.features.fitting.presentation import FittingWorkspace

SIDEBAR_COLLAPSED_KEY = "shell.sidebar_collapsed"
STATUS_BAR_PAGES = ("predict", "trainset")
"""The Labs pages report through the status bar under the page; every other page has its own status line."""
TASK_MODES = {"giwaxs": "GIWAXS", "gisaxs": "GISAXS"}


def _automatic_outcome(report: dict) -> tuple[str, str]:
    """(state, one line) of an automatic analysis, for the Results step of Analyze."""
    from src.gimap.features.assistant.presentation import automatic_outcome

    return automatic_outcome(report)


def _accepts(function, name: str) -> bool:
    """Whether ``function`` takes the keyword ``name`` (a contract another package adds in its own time)."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):  # a builtin without a signature
        return False
    return any(parameter.kind == parameter.VAR_KEYWORD or (
        parameter.name == name and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY))
        for parameter in parameters)


class MainWindowComponents:
    """Builds and owns the feature pages and the navigation sidebar."""

    def __init__(self, ui):
        self.ui = ui
        self.preferences = ui.app_context.preferences
        self.task_runner = TaskRunner(ui.centralwidget)
        self._assistant = None
        self.trainset_page = self._create_trainset_page()
        self.analyze_page = self._create_analyze_page()
        self.compare_page = self._create_compare_page()
        from src.gimap.features.fitting.bootstrap import create_fitting_view_model, create_quick_fit

        self.fitting_view_model = create_fitting_view_model(ui.app_context)
        self.fitting_workspace = FittingWorkspace(
            ui,
            LAYOUT,
            preferences=ui.app_context.preferences,
            view_model=self.fitting_view_model,
            quick_fit=create_quick_fit(),
        )
        self.predict_workspace = GisaxsPredictWorkspace(ui, LAYOUT)
        self.home_page = self._create_home_page()
        self.guided = self._create_guided_analysis()
        self.pages = {
            "home": self.home_page,
            "analyze": ui.analyzePage,
            "fitting": ui.gisaxsFittingPage,
            "predict": ui.gisaxsPredictPage,
            "trainset": ui.trainsetBuildPage,
            "compare": self.compare_page,
        }
        self.sidebar = NavigationSidebar(
            NAVIGATION_ITEMS,
            ui.centralwidget,
            collapsed=bool(self.preferences.get(SIDEBAR_COLLAPSED_KEY, False)),
        )
        ui.horizontalLayout.insertWidget(0, self.sidebar)
        self._place_status_bar()
        self.sidebar.pageRequested.connect(self._page_requested)
        self.sidebar.collapsedChanged.connect(self._remember_sidebar)
        ui.mainWindowWidget.currentChanged.connect(self._sync_sidebar)
        # After a switch of the interface language the pages compose their run-time texts again (once here;
        # a weak connection, and shutdown() ends it).
        language_changed().connect(self.refresh_language)
        # A new user starts with a question, not a toolbar; the expert pages stay one click away.
        self.show_page("home")

    # -- pages ------------------------------------------------------------

    def _create_trainset_page(self) -> TrainsetBuildPage:
        host = self.ui.trainsetBuildPage
        layout = host.layout()
        if layout is None:  # an empty QLayout is falsy (len == 0)
            layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        page = TrainsetBuildPage(host)
        layout.addWidget(page)
        self.ui.trainsetWorkspace = page
        return page

    def _create_compare_page(self):
        """Compare: Analyze's Series maps (Send to Compare) and curve files side by side."""
        from src.gimap.features.compare.bootstrap import create_compare_service
        from src.gimap.features.compare.presentation.page import ComparePage

        page = ComparePage(create_compare_service(), task_runner=self.task_runner, preferences=self.preferences,
                           analyze_map=self.analyze_page.current_series)
        self._replace_page_host("comparePageHost", page)
        self.ui.comparePage = page
        self.analyze_page.set_compare_target(self._send_series_to_compare)
        return page

    def _send_series_to_compare(self, series_map, name: str) -> None:
        """Analyze ▸ Series ▸ Send to Compare: the map joins the comparison, and Compare is shown."""
        if self.compare_page.add_map(series_map, name) is not None:
            self.show_page("compare")

    def _create_analyze_page(self):
        from src.gimap.features.analyze.bootstrap import create_analyze_view_model
        from src.gimap.features.analyze.presentation.page import AnalyzePage

        page = AnalyzePage(
            create_analyze_view_model(self.ui.app_context),
            task_runner=self.task_runner,
            calibrate=self._calibrate_for_analyze,
            send_to_fitting=self._send_curve_to_fitting,
            send_series_to_fitting=self._send_series_to_fitting,
            assistant=self.ask_ai,
        )
        page_index = self._replace_page_host("analyzePageHost", page)
        self.ui.analyzePage = page
        self.ui.analyzePageIndex = page_index
        return page

    def _create_home_page(self):
        from .presentation.home_page import HomePage

        page = HomePage()
        self.ui.mainWindowWidget.addWidget(page)
        page.openRequested.connect(lambda: self._open_and_guide(self.analyze_page.open_files))
        page.folderRequested.connect(lambda: self._open_and_guide(self.analyze_page.open_folder))
        page.batchRequested.connect(lambda: self._open_and_guide(self.analyze_page.batch_from_folder))
        page.projectRequested.connect(lambda: getattr(self.ui, "menus", None) and self.ui.menus.open_project())
        # The same list as File ▸ Open Recent (projects first), once the menus exist.
        page.set_recent_provider(
            lambda: self.ui.menus.recent_paths() if getattr(self.ui, "menus", None) else self.analyze_page.recent_paths()
        )
        page.recentRequested.connect(self._recent_requested)
        page.filesDropped.connect(self.open_dropped)
        page.taskRequested.connect(self._task_requested)
        page.askRequested.connect(lambda text: self.assistant().start(notes=text))
        return page

    def _recent_requested(self, path: str) -> None:
        menus = getattr(self.ui, "menus", None)
        if menus is not None:
            menus.open_recent(path)
            return
        self.analyze_page.add_paths([path])
        self._page_requested("analyze")

    def _create_guided_analysis(self):
        """The automatic analysis (no AI) lives in the Analyze workspace: Results step and tab."""
        from src.gimap.features.assistant.bootstrap import create_guided_analysis
        from src.gimap.features.calibration.bootstrap import create_headless_calibration
        from src.gimap.features.fitting.bootstrap import create_quick_fit

        page = self.analyze_page
        guided = create_guided_analysis(
            self.ui.app_context, automation=page.automation, calibrator=create_headless_calibration(),
            fitter=create_quick_fit(), parent=page,
        )
        page.add_step_panel("results", guided.controls)
        page.add_result_panel(guided.progress_panel)
        page.add_result_panel(guided.results)
        page.set_automatic_analysis(run=guided.run, find_geometry=guided.find_geometry, stop=guided.stop)
        page.set_model_fitter(create_quick_fit())  # batch fits of GISAXS cuts use Fitting's quick physical fit
        guided.stopping.connect(page.automatic_stopping)
        guided.started.connect(page.automatic_started)
        guided.progressed.connect(page.automatic_progress)
        guided.failed.connect(page.automatic_failed)
        guided.refineRequested.connect(page.fit_current)
        guided.solutionRequested.connect(self._show_fit_solution)
        guided.finished.connect(self._automatic_finished)
        forget = getattr(page, "forget_results", None)
        if forget is not None:  # Discard in the progress card: the Results step forgets them too
            guided.progress_panel.discardRequested.connect(lambda: forget())
        # The Results of the automatic analysis follow the frame shown (by file path) ...
        page.analysisShown.connect(lambda analysis: self._frame_shown(guided, analysis))
        # ... and Analyze ▸ Clear: Analyze forgets its results, so the automatic analysis forgets its reports.
        cleared = getattr(page, "filesCleared", None)  # the Analyze page's signal (contract)
        if cleared is not None:
            cleared.connect(lambda: self._files_cleared(guided))
        removed, forget_file = getattr(page, "fileRemoved", None), getattr(guided, "file_removed", None)
        if removed is not None and forget_file is not None:  # one file taken off the list: its reports go too
            removed.connect(lambda path: forget_file(str(path)))
        return guided

    def _automatic_finished(self, report: dict) -> None:
        """The outcome on Analyze's Results step, for the file the run started on and the frames it analysed
        (``report["frames"]``): other frames of that series are not called done."""
        self.analyze_page.automatic_finished(*_automatic_outcome(report),
                                             show_results=report.get("procedure") != "geometry",
                                             path=report.get("frame"), frames=report.get("frames"))

    @staticmethod
    def _frame_shown(guided, analysis) -> None:
        frame_shown = getattr(guided, "frame_shown", None)  # the automatic analysis's hook (contract)
        path = getattr(analysis, "path", None)
        if frame_shown is None or path is None:
            return
        index = getattr(analysis, "frame_index", None)
        if index is not None and _accepts(frame_shown, "frame"):
            # Which frames of the file (1-based first, number summed): results of other frames of a series
            # are told apart even when Analyze's status cannot say.
            frame_shown(str(path), frame=int(index) + 1, summed=int(getattr(analysis, "frame_total", 1) or 1))
        else:
            frame_shown(str(path))

    @staticmethod
    def _files_cleared(guided) -> None:
        files_cleared = getattr(guided, "files_cleared", None)  # the automatic analysis's hook (contract)
        if files_cleared is not None:
            files_cleared()

    def ask_ai(self) -> None:
        """Process with AI on the frame in Analyze (Analyze's Ask AI, Tools ▸ Process with AI…): it starts with
        the notes already typed in the Results step (this session only)."""
        self.assistant().start(notes=self._results_notes())

    def _results_notes(self) -> str:
        guided = getattr(self, "guided", None)
        notes = getattr(guided, "notes_edit", None)
        return notes.toPlainText() if notes is not None else ""

    def _open_and_guide(self, open_dialog) -> None:
        before = len(self.analyze_page.view_model.state.files)
        open_dialog()
        if len(self.analyze_page.view_model.state.files) != before:
            self._page_requested("analyze")

    def open_dropped(self, paths) -> bool:
        """Files dropped on the window: a project opens as a project, frames and folders in Analyze."""
        paths = [str(Path(path)) for path in paths]
        projects = [path for path in paths if path.lower().endswith(".gimap")]
        menus = getattr(self.ui, "menus", None)
        if projects and menus is not None:
            return bool(menus.open_project(projects[0]))
        show_paths = getattr(self.analyze_page, "show_paths", None)  # Analyze's contract: True when a frame is shown
        shown = bool(show_paths(paths)) if show_paths is not None else bool(self.analyze_page.add_paths(paths))
        if shown:
            self._page_requested("analyze")
        elif self.current_page_key() != "analyze":  # Analyze says so in its own status line
            show_toast(self.ui, tr("Nothing opened: drop detector frames (CBF, NXS, TIFF, EDF), "
                                   "a folder of them or a project (.gimap)"), level="warning")
        return shown

    def _task_requested(self, key: str) -> None:
        page = self.analyze_page
        if key in TASK_MODES:
            self._page_requested("analyze")
            choose_mode = getattr(page, "choose_mode", None)  # Analyze's contract: re-runs a shown frame
            if choose_mode is not None:
                choose_mode(key)
            else:
                page.set_mode_choice(key)
            show_toast(
                self.ui,
                tr("Analyze mode: {mode} — choose Auto in the bar above to let GIMaP decide").format(
                    mode=TASK_MODES[key]),
                level="info",
            )
            if not page.view_model.state.files:
                self._open_and_guide(page.open_files)
        elif key == "series":
            self._page_requested("analyze")
            if not page.view_model.state.files:
                self._open_and_guide(page.open_folder)
        elif key == "calibrate":
            menus = getattr(self.ui, "menus", None)
            if menus is not None:
                menus.open_geometry_calibration()

    def assistant(self):
        """“Process with AI” for the frame in Analyze (created on first use)."""
        if self._assistant is None:
            from src.gimap.features.assistant.bootstrap import create_assistant_controller
            from src.gimap.features.calibration.bootstrap import create_headless_calibration
            from src.gimap.features.fitting.bootstrap import create_quick_fit

            self._assistant = create_assistant_controller(
                self.ui,
                self.ui.app_context,
                automation=self.analyze_page.automation,
                show_analyze=lambda: self._page_requested("analyze"),
                calibrator=create_headless_calibration(),
                fitter=create_quick_fit(),
            )
        return self._assistant

    def _calibrate_for_analyze(self, path) -> None:
        """Open Geometry Calibration on the frame shown in Analyze (blocking)."""
        from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog

        dialog = GeometryCalibrationDialog(self.ui)
        tools = getattr(getattr(self.ui, "menus", None), "tools", None)
        if tools is not None:  # where Tools ▸ Geometry Calibration was left, inside the screen
            tools.place("calibration", dialog)
        try:
            if path is not None:
                dialog.load_image(str(path))
            dialog.exec_()
        finally:
            # One window per calibration (also when the frame could not be loaded): deleted now, or when a
            # thread it still waits for ends (never before exec_() returned: under its own event loop).
            dialog.dispose_when_idle()

    def _send_curve_to_fitting(self, path, side: str = "both_abs") -> None:
        """Load an Analyze curve into Fitting and show it ready to fit."""
        runtime = getattr(self.ui, "runtime", None)
        if runtime is None:
            raise RuntimeError(tr("Fitting is still starting; try again in a moment."))
        runtime.navigate("fitting")
        if not self.fitting_workspace.open_curve(path, side):
            raise RuntimeError(self.fitting_workspace.fit_page.status_label.text())

    def _show_fit_solution(self, row: dict) -> None:
        """Results ▸ Fit details ▸ Show in Fitting: the prepared curve in Fitting with that solution as its model."""
        self.analyze_page.fit_current()
        if self.current_page_key() == "fitting":
            self.fitting_workspace.show_solution(row)

    def _send_series_to_fitting(self, folder) -> None:
        """Open Fitting ▸ In-situ series on the folder of curves Analyze just wrote."""
        runtime = getattr(self.ui, "runtime", None)
        if runtime is None:
            raise RuntimeError(tr("Fitting is still starting; try again in a moment."))
        runtime.navigate("fitting")
        self.fitting_workspace.open_series(folder, "*_fit_input.dat")

    def _replace_page_host(self, host_name: str, page) -> int:
        stack = self.ui.mainWindowWidget
        host = getattr(self.ui, host_name, None)
        host_index = stack.indexOf(host) if host is not None else -1
        if host_index < 0:
            return stack.addWidget(page)
        stack.insertWidget(host_index, page)
        stack.removeWidget(host)
        host.setParent(None)
        host.deleteLater()
        delattr(self.ui, host_name)
        return host_index

    # -- navigation -------------------------------------------------------

    def _place_status_bar(self) -> None:
        """The Labs status bar goes under the page, not across the window: the sidebar keeps its full
        height on every page. Taken out of the window's own layout, it no longer shows the menus' status
        tips either, which would wipe a step instruction while a menu is open."""
        statusbar = getattr(self.ui, "statusbar", None)
        column = getattr(self.ui, "verticalLayout_2", None)  # the page column (mainContentWidget)
        if statusbar is not None and column is not None:
            column.addWidget(statusbar)

    def show_page(self, key: str) -> None:
        from .presentation.i18n import DEFAULT_LANGUAGE, apply_to, current_language

        page = self.pages[key]
        if current_language() != DEFAULT_LANGUAGE:
            apply_to(page, current_language())  # widgets built since the language was applied
        self.ui.mainWindowWidget.setCurrentWidget(page)
        self.sidebar.set_active(key)
        statusbar = getattr(self.ui, "statusbar", None)
        if statusbar is not None:
            statusbar.setVisible(key in STATUS_BAR_PAGES)

    def current_page_key(self) -> str | None:
        current = self.ui.mainWindowWidget.currentWidget()
        for key, page in self.pages.items():
            if page is current:
                return key
        return None

    def _page_requested(self, key: str) -> None:
        runtime = getattr(self.ui, "runtime", None)
        if runtime is not None:
            runtime.navigate(key)
        else:
            self.show_page(key)

    def _sync_sidebar(self, _index: int) -> None:
        key = self.current_page_key()
        if key is not None:
            self.sidebar.set_active(key)

    def set_sidebar_collapsed(self, collapsed: bool) -> None:
        self.sidebar.set_collapsed(collapsed)

    def _remember_sidebar(self, collapsed: bool) -> None:
        self.preferences.set(SIDEBAR_COLLAPSED_KEY, bool(collapsed))

    # -- lifecycle --------------------------------------------------------

    def running_jobs(self) -> list[str]:
        """The names (in the interface language) of the jobs still running, for the question on quit."""
        fitting = self.fitting_workspace
        # The Series tab's map runs as a batch that writes nothing: it is not a Batch Export.
        batch = tr("Series map") if self._batch_kind() == "series_map" else tr("Batch Export")
        checks = (
            (batch, lambda: self.analyze_page.batch_running()),
            (tr("Automatic analysis"), lambda: self.guided.running()),
            (tr("AI analysis"), lambda: self._assistant is not None and self._assistant.running()),
            (tr("Fitting"), lambda: fitting.fit_page.fit_running()),
            (tr("In-situ series"), lambda: bool(getattr(fitting.series_page, "running", False))),
            # The Labs runtime starts after the window is shown: no runtime yet is an AttributeError, skipped.
            (tr("2D Prediction"), lambda: self.ui.runtime.prediction.prediction_running()),
        )
        jobs = []
        for name, busy in checks:
            try:
                if busy():
                    jobs.append(name)
            except (AttributeError, RuntimeError):  # a page that is not built or already gone
                continue
        return jobs

    @staticmethod
    def jobs_text(jobs) -> str:
        """The names of ``jobs`` as one phrase in the interface language (``、`` between Chinese names)."""
        return ("、" if current_language() == "zh" else ", ").join(str(job) for job in jobs)

    def _batch_kind(self) -> str:
        """``series_map`` or ``export``: what Analyze's batch is (``AnalyzePage.batch_kind``, the contract)."""
        kind = getattr(self.analyze_page, "batch_kind", None)
        if kind is not None:
            return str(kind())
        # An Analyze page without the public accessor yet: the flag it sets itself.
        return "series_map" if getattr(self.analyze_page, "_batch_map_only", False) else "export"

    def refresh_language(self, _language: str = "") -> None:
        """After a switch of the interface language (``language_changed()``, which comes after the widget walker
        and only on a real switch): every page that has ``refresh_language`` composes the texts it made with
        ``tr`` / ``trf`` again, for what it shows now. One page that fails does not keep the others English."""
        self._status_bar_language()
        error = None
        for page in self._language_pages():
            refresh = getattr(page, "refresh_language", None)
            if refresh is None:
                continue
            try:
                refresh()
            except RuntimeError:  # a page whose widgets are already gone (the window is closing)
                continue
            except Exception as exc:  # the others are refreshed first, then the error is reported
                error = error or exc
        if error is not None:
            raise error

    def _language_pages(self) -> list:
        """The pages, once each, that may hold run-time texts: Fitting's pages as well as its workspace, the Labs
        pages and the tool windows open now (Tools ▸ Geometry Calibration, Format Converter, XRR).

        A Labs page is reached through its view binding once the Labs runtime has started (after the window is
        shown): the binding composes its own texts and its page's (2D Prediction, Trainset Build), and the page
        hands over to it, so only one of the two is told; before that, the page itself."""
        workspace = self.fitting_workspace
        runtime = getattr(self.ui, "runtime", None)
        tools = getattr(getattr(self.ui, "menus", None), "tools", None)
        candidates = (self.home_page, self.analyze_page, getattr(self, "guided", None), self.compare_page, workspace,
                      getattr(workspace, "fit_page", None), getattr(workspace, "series_page", None),
                      self._labs_page(getattr(runtime, "prediction", None), self.predict_workspace),
                      self._labs_page(getattr(runtime, "trainset", None), self.trainset_page),
                      self._assistant, *(tools.windows() if tools is not None else ()))
        pages, seen = [], set()
        for page in candidates:
            if page is not None and id(page) not in seen:
                seen.add(id(page))
                pages.append(page)
        return pages

    def _status_bar_language(self) -> None:
        """The Labs status bar's message shown now, in the new language when the table knows it (the next
        ones come through ``tr``; the walker does not reach a status bar message)."""
        statusbar = getattr(self.ui, "statusbar", None)
        try:
            message = statusbar.currentMessage() if statusbar is not None else ""
            switched = translate(message, current_language()) if message else None
            if switched:
                statusbar.showMessage(switched)
        except RuntimeError:  # the window is going
            pass

    @staticmethod
    def _labs_page(binding, page):
        """The Labs view binding when it composes texts after a language switch, else its page."""
        return binding if callable(getattr(binding, "refresh_language", None)) else page

    def save_state(self) -> None:
        self.fitting_workspace.save_state()
        self.preferences.save()

    def shutdown(self) -> None:
        """Stop every job (the quit question promised it), wait for background work and release
        pyqtgraph views before the window goes."""
        # 2D Prediction first: main.py shuts the job runner down after this, and a worker still preprocessing
        # must not start a model process then (a batch ends after its file).
        self._stop_predictions()
        try:
            language_changed().disconnect(self.refresh_language)
        except TypeError:  # not connected
            pass
        if self._assistant is not None:
            self._assistant.shutdown()
        self._stop_automatic_analysis()
        # A Single fit and an In-situ series: stop, wait (2 s at most) for their threads, release the plots.
        for page in (self.fitting_workspace.fit_page, self.fitting_workspace.series_page):
            try:
                page.dispose()
            except (AttributeError, RuntimeError):  # not built here, or its widgets are already gone
                continue
        self.analyze_page.dispose()
        self.task_runner.shutdown()

    def _stop_predictions(self) -> None:
        prediction = getattr(getattr(self.ui, "runtime", None), "prediction", None)  # Labs: after the window shows
        stop = getattr(prediction, "stop_predictions", None)
        if stop is None:
            return
        try:
            stop()
        except Exception as exc:  # the other jobs are still stopped
            print(f"Could not stop 2D Prediction on close: {exc}")

    def _stop_automatic_analysis(self, timeout_s: float = 2.0) -> None:
        """End the automatic analysis before its next step and give the step in progress up to
        ``timeout_s`` to finish, so it does not drive Analyze once its views are released."""
        guided = getattr(self, "guided", None)
        stop, running = getattr(guided, "stop", None), getattr(guided, "running", None)
        if stop is None:
            return
        stop()
        deadline = time.monotonic() + timeout_s
        while running is not None and running() and time.monotonic() < deadline:
            # The step may be waiting for the GUI thread; clicks and keys are not handled while the window goes.
            QCoreApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
            time.sleep(0.01)


__all__ = [
    "CardContentResizeHandle",
    "CollapsibleCardFrame",
    "FittingWorkspace",
    "GisaxsPredictWorkspace",
    "MainWindowComponents",
    "NavigationSidebar",
    "PredictCard",
    "PredictModelLibraryCard",
]
