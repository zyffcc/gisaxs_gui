"""Composition of the main window: feature pages, sidebar and page switching.

``ApplicationWindowView`` creates the shell widgets and the classic Fitting /
Prediction controls; this module builds the remaining feature pages into the
page stack and owns navigation between them.  Workspaces are addressed by the
keys of ``NAVIGATION_ITEMS`` ("analyze", "fitting", ...).
"""

from __future__ import annotations

from PyQt5.QtWidgets import QVBoxLayout

from .presentation.collapsible_card import (
    CardContentResizeHandle,
    CollapsibleCardFrame,
)
from .presentation.layout_metrics import LAYOUT
from .presentation.navigation import NAVIGATION_ITEMS, NavigationSidebar
from .presentation.task_runner import TaskRunner
from src.gimap.features.classification.presentation.page import ClassificationPage
from src.gimap.features.prediction.presentation import (
    GisaxsPredictWorkspace,
    PredictCard,
    PredictModelLibraryCard,
)
from src.gimap.features.trainset.presentation import TrainsetBuildPage
from src.gimap.features.fitting.presentation import FittingWorkspace

SIDEBAR_COLLAPSED_KEY = "shell.sidebar_collapsed"



def _automatic_outcome(report: dict) -> tuple[str, str]:
    """(state, one line) of an automatic analysis, for the Results step of Analyze."""
    from src.gimap.features.assistant.presentation import automatic_outcome

    return automatic_outcome(report)

class MainWindowComponents:
    """Builds and owns the feature pages and the navigation sidebar."""

    def __init__(self, ui):
        self.ui = ui
        self.preferences = ui.app_context.preferences
        self.task_runner = TaskRunner(ui.centralwidget)
        self._assistant = None
        self.trainset_page = self._create_trainset_page()
        self.classification_page = self._create_classification_page()
        self.analyze_page = self._create_analyze_page()
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
            "classification": ui.classificationPage,
        }
        self.sidebar = NavigationSidebar(
            NAVIGATION_ITEMS,
            ui.centralwidget,
            collapsed=bool(self.preferences.get(SIDEBAR_COLLAPSED_KEY, False)),
        )
        ui.horizontalLayout.insertWidget(0, self.sidebar)
        self.sidebar.pageRequested.connect(self._page_requested)
        self.sidebar.collapsedChanged.connect(self._remember_sidebar)
        ui.mainWindowWidget.currentChanged.connect(self._sync_sidebar)
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

    def _create_classification_page(self) -> ClassificationPage:
        host = self.ui.classificationPage
        layout = host.layout()
        if layout is None:  # an empty QLayout is falsy (len == 0)
            layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        page = ClassificationPage(host)
        layout.addWidget(page)
        self.ui.classificationWorkspace = page
        return page

    def _create_analyze_page(self):
        from src.gimap.features.analyze.bootstrap import create_analyze_view_model
        from src.gimap.features.analyze.presentation.page import AnalyzePage

        page = AnalyzePage(
            create_analyze_view_model(self.ui.app_context),
            task_runner=self.task_runner,
            calibrate=self._calibrate_for_analyze,
            send_to_fitting=self._send_curve_to_fitting,
            send_series_to_fitting=self._send_series_to_fitting,
            assistant=lambda: self.assistant().start(),
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
        page.set_recent_provider(self.analyze_page.recent_paths)
        page.recentRequested.connect(lambda path: (self.analyze_page.add_paths([path]), self._page_requested("analyze")))
        page.filesDropped.connect(self._open_dropped)
        page.taskRequested.connect(self._task_requested)
        page.askRequested.connect(lambda text: self.assistant().start(notes=text))
        return page

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
        guided.finished.connect(
            lambda report: page.automatic_finished(
                *_automatic_outcome(report), show_results=report.get("procedure") != "geometry"
            )
        )
        return guided

    def _open_and_guide(self, open_dialog) -> None:
        before = len(self.analyze_page.view_model.state.files)
        open_dialog()
        if len(self.analyze_page.view_model.state.files) != before:
            self._page_requested("analyze")

    def _open_dropped(self, paths) -> None:
        projects = [path for path in paths if str(path).lower().endswith(".gimap")]
        menus = getattr(self.ui, "menus", None)
        if projects and menus is not None:
            menus.open_project(projects[0])
            return
        if self.analyze_page.add_paths(paths):
            self._page_requested("analyze")

    def _task_requested(self, key: str) -> None:
        if key in ("giwaxs", "gisaxs", "series"):
            self._page_requested("analyze")
            mode = {"giwaxs": "giwaxs", "gisaxs": "gisaxs"}.get(key)
            if mode is not None and not self.analyze_page.view_model.state.files:
                self.analyze_page.set_mode_choice(mode)
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
        if path is not None:
            dialog.load_image(str(path))
        dialog.exec_()

    def _send_curve_to_fitting(self, path, side: str = "both_abs") -> None:
        """Load an Analyze curve into Fitting and show it ready to fit."""
        runtime = getattr(self.ui, "runtime", None)
        if runtime is None:
            raise RuntimeError("Fitting is still starting; try again in a moment.")
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
            raise RuntimeError("Fitting is still starting; try again in a moment.")
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

    def show_page(self, key: str) -> None:
        from .presentation.i18n import DEFAULT_LANGUAGE, apply_to, current_language

        page = self.pages[key]
        if current_language() != DEFAULT_LANGUAGE:
            apply_to(page, current_language())  # widgets built since the language was applied
        self.ui.mainWindowWidget.setCurrentWidget(page)
        self.sidebar.set_active(key)

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

    def save_state(self) -> None:
        self.fitting_workspace.save_state()
        self.preferences.save()

    def shutdown(self) -> None:
        """Wait for background work and release pyqtgraph views before the window goes."""
        if self._assistant is not None:
            self._assistant.shutdown()
        self.analyze_page.dispose()
        self.task_runner.shutdown()


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
