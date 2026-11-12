"""What the Edit, Analyze and Fitting menus do (``presentation/workspace_menus.py`` builds them).

Every Analyze and Fitting entry shows its workspace first (and the right part of Fitting), then runs the
page's own command, the one its button runs: the menu adds a way in, not another behaviour. Whether an
entry can run is read from the state the page keeps (its buttons, its view model) each time the menu
opens; when it cannot, the entry says why (English here, translated where it is shown). Edit ▸ Undo / Redo
act on the page shown and do not navigate.
"""

from __future__ import annotations

from .presentation.workspace_menus import MenuEntry

NO_FRAME = "Open a frame first."
NOT_ANALYSED = "The frame shown is not analysed: it could not be read, or its analysis is still running."
NO_GEOMETRY = "Set the geometry first (Geometry step)."
NO_FILES = "Open files or a folder first."
BATCH_RUNNING = "A batch is already running."
AUTOMATIC_RUNNING = "The automatic analysis is running."
AUTOMATIC_IDLE = "The automatic analysis is not running."
AUTOMATIC_STOPPING = "Stopping after the current step …"
AUTOMATIC_KEEPS_FRAME = "The automatic analysis works on the frame shown: stop it first."
FEW_FRAMES = "List two or more frames first (a folder, several files or a NeXus series)."
NO_MAP = "Build the Series map first (Series tab ▸ Build Map)."
NO_REGIONS = "Add a region first: the standard cuts are always there."
NO_MASKS = "Draw a mask first."
NO_CURVE = "Open a curve first."
FIT_RUNNING = "A fit is running."
FIT_IDLE = "No fit is running."
SERIES_RUNNING = "Stop the running series first."
SERIES_IDLE = "The In-situ series is not running."
NO_SERIES_CURVES = "Choose a folder of curves first"
NO_SERIES_MODEL = "Set up a model in Single analysis first"
STARTING = "GIMaP is still starting; try again in a moment."
NO_UNDO_PAGE = "Undo and Redo work on Analyze and on Fitting's Single analysis."
NOTHING_TO_UNDO = "Nothing to undo on this page."
NOTHING_TO_REDO = "Nothing to redo on this page."


class WorkspaceCommands:
    """Binds the menu entries to the Analyze page and the Fitting workspace of ``menus.window``."""

    def __init__(self, menus):
        self.menus = menus  # ApplicationMenus: show_workspace, _curve_page_shown, open_ai_fitting_workspace

    # -- the pages ------------------------------------------------------------

    @property
    def _components(self):
        return self.menus.window.components

    @property
    def _analyze(self):
        return self._components.analyze_page

    @property
    def _fitting(self):
        return self._components.fitting_workspace

    def _in_analyze(self, command):
        def run():
            self.menus.show_workspace("analyze")
            return command()
        return run

    def _in_fitting(self, command, context: str = "single"):
        def run():
            self.menus.show_workspace("fitting")
            self._fitting.show_context(context)
            return command()
        return run

    # -- what can run ---------------------------------------------------------

    def _frame(self) -> str:
        return "" if self._analyze.view_model.current_path is not None else NO_FRAME

    def _held(self) -> str:
        """``""`` when a frame is analysed; else why not (no file open, or the file shown has no analysis)."""
        if self._analyze.view_model.state.analysis is not None:
            return ""
        return self._frame() or NOT_ANALYSED  # _frame() is NO_FRAME when no file is open

    def _analysis(self) -> str:
        analysis = self._analyze.view_model.state.analysis
        if analysis is None:
            return self._held()
        return "" if getattr(analysis, "reduction", None) is not None else NO_GEOMETRY

    def _shown(self) -> str:
        return self._held()

    def _batch_free(self) -> str:
        return BATCH_RUNNING if self._analyze.batch_running() else ""

    def _batch(self) -> str:
        return self._batch_free() if self._analyze.view_model.state.files else NO_FILES

    def _run_automatic(self) -> str:
        page = self._analyze
        if page.automatic_running():
            return AUTOMATIC_RUNNING
        return self._frame() or ("" if page.run_pipeline_button.isEnabled() else AUTOMATIC_RUNNING)

    def _stop_automatic(self) -> str:
        page = self._analyze
        if not page.automatic_running():
            return AUTOMATIC_IDLE
        return "" if page.stop_pipeline_button.isEnabled() else AUTOMATIC_STOPPING

    def _send_fitting(self) -> str:
        return "" if self._analyze.fit_button.isEnabled() else self._analysis() or NO_GEOMETRY

    def _send_series(self) -> str:
        page = self._analyze
        if page.view_model.listed_frames() < 2:
            return FEW_FRAMES
        return self._batch_free() or ("" if page.fit_series_action.isEnabled() else STARTING)

    def _send_compare(self) -> str:
        return "" if self._analyze.current_series() is not None else NO_MAP

    def _regions(self) -> str:
        return "" if self._analyze.view_model.state.giwaxs.regions else NO_REGIONS

    def _masks(self) -> str:
        return "" if self._analyze.view_model.state.corrections.mask_shapes else NO_MASKS

    def _clear(self) -> str:
        page = self._analyze
        if not page.view_model.state.files:
            return NO_FILES
        return "" if page.clear_button.isEnabled() else AUTOMATIC_KEEPS_FRAME

    def _curve(self) -> str:
        return "" if self._fitting.fit_page.session.curve is not None else NO_CURVE

    def _fit(self) -> str:
        return FIT_RUNNING if self._fitting.fit_page.fit_running() else self._curve()

    def _stop_fit(self) -> str:
        return "" if self._fitting.fit_page.fit_running() else FIT_IDLE

    def _series_folder(self) -> str:
        return SERIES_RUNNING if self._fitting.series_page.running else ""

    def _series_start(self) -> str:
        """Read from what the page's Start button is built from (``FitSeriesPage.refresh``), not from the button:
        a model set up in Single analysis does not refresh the series page until it is shown."""
        series = self._fitting.series_page
        if series.running:
            return SERIES_RUNNING
        if not series.paths or not series._listed():
            return NO_SERIES_CURVES
        return "" if series.single.session.model.components else NO_SERIES_MODEL

    def _series_stop(self) -> str:
        return "" if self._fitting.series_page.running else SERIES_IDLE

    def _runtime(self) -> str:
        return "" if getattr(self.menus.window, "runtime", None) is not None else STARTING

    # -- Edit -----------------------------------------------------------------

    def _undo_page(self) -> str:
        """``analyze``, ``fitting`` (its Single analysis is shown) or ``""``: the page Undo and Redo act on."""
        key = self._components.current_page_key()
        if key == "analyze":
            return "analyze"
        return "fitting" if self.menus._curve_page_shown() else ""

    def _history(self, step: str):
        def run():
            page = self._undo_page()
            if page == "analyze":
                return (self._analyze.undo_setup if step == "undo" else self._analyze.redo_setup)()
            if page == "fitting":
                fit_page = self._fitting.fit_page
                return (fit_page.undo if step == "undo" else fit_page.redo)()
            return None
        return run

    def _history_state(self, step: str):
        def blocked() -> str:
            page = self._undo_page()
            if not page:
                return NO_UNDO_PAGE
            owner = self._analyze if page == "analyze" else self._fitting.fit_page
            button = owner.undo_button if step == "undo" else owner.redo_button
            return "" if button.isEnabled() else NOTHING_TO_UNDO if step == "undo" else NOTHING_TO_REDO
        return blocked

    def _history_detail(self, step: str):
        def detail() -> str:
            if self._undo_page() != "analyze":
                return ""  # Fitting's Undo does not say what it undoes
            undo, redo = self._analyze.undo_names()
            return undo if step == "undo" else redo
        return detail

    # -- the entries ----------------------------------------------------------

    def entries(self) -> dict:
        """``MenuEntry`` by key, for ``MenuCommands.page_entries``."""
        analyze, fitting = self._in_analyze, self._in_fitting
        page = lambda: self._analyze  # noqa: E731 - the page as it is when the entry runs
        fit = lambda: self._fitting.fit_page  # noqa: E731
        series = lambda: self._fitting.series_page  # noqa: E731
        return {
            "edit_undo": MenuEntry(self._history("undo"), self._history_state("undo"), self._history_detail("undo")),
            "edit_redo": MenuEntry(self._history("redo"), self._history_state("redo"), self._history_detail("redo")),
            # Analyze
            "analyze_run_automatic": MenuEntry(analyze(lambda: page().run_automatic()), self._run_automatic),
            "analyze_stop_automatic": MenuEntry(analyze(lambda: page().stop_automatic()), self._stop_automatic),
            "analyze_repeat": MenuEntry(analyze(lambda: page().run_analysis()), self._frame),
            "analyze_export": MenuEntry(analyze(lambda: page().export_current()), self._analysis),
            "analyze_batch": MenuEntry(analyze(lambda: page().batch_export_dialog()), self._batch),
            "analyze_batch_folder": MenuEntry(analyze(lambda: page().batch_from_folder()), self._batch_free),
            "analyze_send_fitting": MenuEntry(analyze(lambda: page().fit_current()), self._send_fitting),
            "analyze_send_series": MenuEntry(analyze(lambda: page().send_series_dialog()), self._send_series),
            "analyze_send_compare": MenuEntry(analyze(lambda: page().send_series_to_compare()), self._send_compare),
            "analyze_save_settings": MenuEntry(analyze(lambda: page().save_settings())),
            "analyze_load_settings": MenuEntry(analyze(lambda: page().load_settings())),
            "analyze_save_cuts": MenuEntry(analyze(lambda: page().save_cuts()), self._regions),
            "analyze_load_cuts": MenuEntry(analyze(lambda: page().load_cuts())),
            "analyze_save_masks": MenuEntry(analyze(lambda: page().save_masks()), self._masks),
            "analyze_load_masks": MenuEntry(analyze(lambda: page().load_masks())),
            "analyze_save_image": MenuEntry(analyze(lambda: page().save_image()), self._shown),
            "analyze_save_map": MenuEntry(analyze(lambda: page().save_map_data()), self._analysis),
            "analyze_save_view": MenuEntry(analyze(lambda: page().save_view_data()), self._analysis),
            "analyze_clear": MenuEntry(analyze(lambda: page().clear_files()), self._clear),
            # Fitting
            "fitting_open_curve": MenuEntry(fitting(lambda: fit().open_curve_dialog())),
            "fitting_import_curves": MenuEntry(fitting(lambda: self._fitting.import_curves()), self._series_folder),
            "fitting_fit_folder": MenuEntry(fitting(lambda: series().choose_folder(), "insitu"), self._series_folder),
            "fitting_fit": MenuEntry(fitting(lambda: fit().run_fit()), self._fit),
            "fitting_stop": MenuEntry(fitting(lambda: fit().stop_fit()), self._stop_fit),
            "fitting_load_model": MenuEntry(fitting(lambda: fit().load_model_dialog())),
            "fitting_save_model": MenuEntry(fitting(lambda: fit().save_model_dialog())),
            "fitting_save_data": MenuEntry(fitting(lambda: fit().export_data_dialog()), self._curve),
            "fitting_save_plot": MenuEntry(fitting(lambda: fit().export_plot_dialog()), self._curve),
            "fitting_series_start": MenuEntry(fitting(lambda: series().start(), "insitu"), self._series_start),
            "fitting_series_stop": MenuEntry(fitting(lambda: series().stop(), "insitu"), self._series_stop),
            "fitting_1d_predict": MenuEntry(fitting(lambda: self.menus.open_ai_fitting_workspace()), self._runtime),
        }


__all__ = ["WorkspaceCommands"]
