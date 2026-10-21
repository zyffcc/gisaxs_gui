"""Fitting workspace: Single analysis (one curve, ``single/page.py``) and In-situ series (every curve of a
folder, ``single/series_page.py``).

Curves come from Analyze (Send to Fitting, Send Series to Fitting) or from curve files. Fitting's models are
particle form factors for the horizontal cut of a GISAXS frame. A curve Analyze marked as the I(q) of a
GIWAXS frame is therefore opened with a note that says so (``handover_note``, above the page that holds such
a curve or a series of them, until it is closed) and, in Single analysis, without the particle-shape search
that a first curve otherwise starts with. A run of the In-situ series on such curves can still be started; when
it ends the note comes back, closed or not, and says what came of it (how many frames are poor fits).

The former pages are still built and never shown: the single-analysis page of before (the classic widgets
of ``FittingPageView``, arranged into the cards below) and the in-situ page of before
(``insitu_series_page.py``, the third page of the context stack). Neither runs the In-situ series — the
series page fits its frames itself. Their binding (``attach_legacy``) only follows the curve and the model
of Single analysis, and opens 1D Predict from its button; tests still pin both pages, so removing them is a
change of its own.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QEvent, QObject, QTimer, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.ports import UserPreferencesRepository
from src.gimap.app.presentation import install_safe_wheel_behavior
from src.gimap.app.presentation.components import AdvancedSection, ErrorBanner
from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.layout_metrics import LAYOUT
from src.gimap.app.presentation.layout_primitives import (
    CARD_SPACING,
    INPUT_WIDGET_TYPES,
    normalize_checkbox,
    normalize_input,
)

from .curve_card import CurveSourceCard
from .fit_steps import FitSteps
from .fitting_theme import apply_fitting_style
from .layout_primitives import detach_from_parent_layout as _detach_from_parent_layout
from .layout_primitives import take_widget as _take_widget
from .model_card import ModelParameterCard
from .preview_cards import FittingPlotControlsCard, PlotPreviewCard, StatusCard
from .run_card import FittingControlsCard
from ..application import LoadCurveRequest
from ..application.series_fit import POOR_CHI2, poor_fit, series_counts
from ..application.single_fit import to_manual
from .single.page import FitPage
from .single.series_page import FitSeriesPage
from .workspace_context import FittingContextContainer

LEGACY_Q_VIEW = {"mean": "average", "both": "fold", "positive": "positive", "negative": "negative_abs"}

GIWAXS_SOURCE = "radial_bins"
"""How Analyze marks the I(q) of a GIWAXS frame in a curve it writes for Fitting (``"source"`` of the file's
``# observation:`` line; a GISAXS cut is ``native_detector_columns``)."""
NOTE_TITLE = {"single": "A GIWAXS I(q), not a GISAXS cut", "insitu": "GIWAXS I(q) curves, not GISAXS cuts"}
NOTE_TEXT = {
    "single": "Fitting's models are particle form factors for the horizontal cut of a GISAXS frame: they do not "
              "describe diffraction peaks, so no particle-shape search is set up for this curve. For its peaks use "
              "Analyze ▸ Batch Export ▸ Peaks of the ring regions.",
    "insitu": "Fitting's models are particle form factors for the horizontal cut of a GISAXS frame: they do not "
              "describe diffraction peaks, and Start would fit one to every frame. To follow the peaks through the "
              "series use Analyze ▸ Batch Export ▸ Peaks of the ring regions.",
}
"""The note above a GIWAXS I(q) in Single analysis, and above a series of them."""
NOTE_RUN = ("Start fitted a particle form factor to every frame ({fitted} fitted). Such models do not describe "
            "diffraction peaks: read these values with care. To follow the peaks through the series use "
            "Analyze ▸ Batch Export ▸ Peaks of the ring regions.")
NOTE_RUN_POOR = ("Start fitted a particle form factor to every frame: {poor} of the {fitted} frames fitted are poor "
                 "fits (χ²ᵣ above {limit}). Such models do not describe diffraction peaks: these values are no "
                 "result. To follow the peaks through the series use Analyze ▸ Batch Export ▸ Peaks of the ring "
                 "regions.")
"""… and above such a series once a run on it has ended (``limit``: ``POOR_CHI2``)."""
GIWAXS_OPENED = "{name} is a GIWAXS I(q): no particle-shape search was set up."
LEGACY_CURVE_FAILED = "The former fitting page could not take this curve: {error}"
LEGACY_MODEL_FAILED = "The former fitting page could not take this model: {error}"

CONTROLS_MIN_WIDTH = 440
CONTROLS_TARGET_WIDTH = 540
RESULTS_MIN_WIDTH = 460


class _RunEnds(QObject):
    """Tells when a run of the In-situ series has ended: the page hides its progress bar then. Only a hide by
    the page itself counts (``HideToParent``), not one of a page around it."""

    def __init__(self, bar: QWidget, ended: QTimer):
        super().__init__(bar)
        self._ended = ended
        bar.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if event.type() == QEvent.HideToParent:
            self._ended.start()  # (once the page has said its own last word)
        return False


def _embed_in_section(card) -> None:
    """A card shown inside an AdvancedSection: the section is its title, toggle and frame."""
    card.set_expanded(True)
    card.header_widget.hide()
    card.setProperty("card", False)
    card.body_layout.setContentsMargins(0, 0, 0, 0)
    card.style().unpolish(card)
    card.style().polish(card)


class FittingWorkspace:
    """Single analysis (one curve) and In-situ series (many curves) of Fitting."""

    SETTINGS_KEY = "fitting_splitter_sizes_v3"

    def __init__(
        self,
        ui,
        profile=None,
        *,
        preferences: UserPreferencesRepository,
        view_model,
        quick_fit=None,
        frames=None,
    ):
        self.ui = ui
        self.preferences = preferences
        self.view_model = view_model
        self._legacy = None
        self.fit_page = FitPage(view_model, quick_fit=quick_fit, preferences=preferences)
        self.fit_page.setObjectName("fittingSinglePage")
        self.series_page = FitSeriesPage(view_model, self.fit_page, preferences=preferences, frames=frames)
        self.series_page.editModelRequested.connect(lambda: self.show_context("single"))
        self.series_page.openFrameRequested.connect(self._frame_in_single)
        self.fit_page.seriesRequested.connect(self._series_from_single)
        self._observed: dict[str, tuple] = {}
        self._note_closed: Optional[tuple] = None
        self._note_run: Optional[tuple] = None
        """``(the run's frames, the folder, poor fits, frames fitted)`` of the last run on a GIWAXS series."""
        self.profile = profile or LAYOUT
        legacy_scroll_area = ui.gisaxsFittingPageScrollArea
        self.page_splitter = QSplitter(Qt.Horizontal, ui.gisaxsFittingPage)
        self.page_splitter.setObjectName("fittingPageSplitter")
        self.page_splitter.setChildrenCollapsible(False)
        self.page_splitter.setHandleWidth(8)
        apply_fitting_style(self.page_splitter)  # before it is populated: one polish pass

        _take_widget(ui.gridLayout_24, ui.curvePlotControlWidget)
        _take_widget(ui.verticalLayout_19, ui.FittingTextBrowser)
        self._configure_inputs()
        self._build_controls()
        self._build_results()
        self._install_context(legacy_scroll_area)
        legacy_scroll_area.deleteLater()
        self.page_splitter.setStretchFactor(0, 0)
        self.page_splitter.setStretchFactor(1, 1)
        self.restore_sizes()
        install_safe_wheel_behavior(self.page_splitter)

    # -- left: curve, model and fitting controls ------------------------------------------

    def _build_controls(self) -> None:
        self.controls_scroll_area = QScrollArea(self.page_splitter)
        self.controls_scroll_area.setObjectName("fittingControlsScrollArea")
        self.controls_scroll_area.setWidgetResizable(True)
        self.controls_scroll_area.setFrameShape(QFrame.NoFrame)
        self.controls_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.controls_scroll_area.setMinimumWidth(CONTROLS_MIN_WIDTH)
        content = QWidget()
        content.setObjectName("fittingControlsContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(CARD_SPACING)

        self.curve_card = CurveSourceCard(self.ui, content)
        layout.addWidget(self.curve_card)

        self.model_parameters_card = ModelParameterCard(self.ui, self.profile)
        self.fitting_controls_card = FittingControlsCard(
            self.ui,
            self.profile,
            model_parameters_card=self.model_parameters_card,
            preferences=self.preferences,
        )
        fit_section = QFrame(content)
        fit_section.setObjectName("fittingRunSection")
        fit_section.setProperty("gimapSection", True)
        fit_layout = QVBoxLayout(fit_section)
        fit_layout.setContentsMargins(12, 10, 12, 12)
        fit_layout.setSpacing(6)
        title = QLabel("Model and fit", fit_section)
        title.setProperty("gimapSectionTitle", True)
        description = QLabel(
            "Choose the components and global parameters, plot the model, refine it, "
            "or let 1D Predict fit the curve.",
            fit_section,
        )
        description.setProperty("gimapSectionDescription", True)
        description.setWordWrap(True)
        fit_layout.addWidget(title)
        fit_layout.addWidget(description)
        fit_layout.addWidget(self.fitting_controls_card)
        layout.addWidget(fit_section)
        layout.addStretch(1)
        self.controls_scroll_area.setWidget(content)
        self.controls_content = content
        self.steps = FitSteps(self, lambda: getattr(getattr(self.ui, "runtime", None), "fitting", None))

    # -- right: plot, export, plot controls and log ------------------------------------------

    def _build_results(self) -> None:
        self.results_scroll_area = QScrollArea(self.page_splitter)
        self.results_scroll_area.setObjectName("fittingResultsScrollArea")
        self.results_scroll_area.setWidgetResizable(True)
        self.results_scroll_area.setFrameShape(QFrame.NoFrame)
        self.results_scroll_area.setMinimumWidth(RESULTS_MIN_WIDTH)
        panel = QWidget()
        panel.setObjectName("fittingResultsPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(CARD_SPACING)

        self.inline_feedback = QLabel("", panel)
        self.inline_feedback.setObjectName("fittingInlineFeedback")
        self.inline_feedback.setProperty("feedbackKind", "error")
        self.inline_feedback.setWordWrap(True)
        self.inline_feedback.setVisible(False)
        layout.addWidget(self.inline_feedback)

        self.fitting_plot_card = PlotPreviewCard(
            self.ui, self.ui.curvePlotControlWidget, self.ui.fitGraphicsView, self.profile
        )
        self.curve_plot_card = self.fitting_plot_card
        layout.addWidget(self.fitting_plot_card, 1)

        export_row = QHBoxLayout()
        export_row.setSpacing(8)
        export_button = self.ui.FittingExportButton
        plot_button = self.fitting_controls_card.fitExportPlotButton
        for button, text in ((export_button, "Export Data…"), (plot_button, "Export Plot…")):
            _detach_from_parent_layout(button)
            button.setParent(panel)
            button.setText(text)
            export_row.addWidget(button)
        self.ui.fitExportPlotButton = plot_button
        export_row.addStretch(1)
        layout.addLayout(export_row)

        self.plot_controls_section = AdvancedSection(
            "Fit range and plot options",
            "The q range that is fitted, sampling and which components are drawn.",
            panel,
        )
        self.fitting_controls_plot_card = FittingPlotControlsCard(
            self.ui, self.ui.curvePlotControlWidget, self.profile
        )
        _embed_in_section(self.fitting_controls_plot_card)
        self.plot_controls_section.add_widget(self.fitting_controls_plot_card)
        layout.addWidget(self.plot_controls_section)

        self.log_section = AdvancedSection("Run Log", "", panel)
        self.run_log_card = StatusCard(self.ui.FittingTextBrowser, self.profile)
        _embed_in_section(self.run_log_card)
        self.log_section.add_widget(self.run_log_card)
        layout.addWidget(self.log_section)
        self.results_scroll_area.setWidget(panel)
        self.results_panel = panel

        self.ui.fittingPlotCard = self.fitting_plot_card
        self.ui.fittingPlotControlsCard = self.fitting_controls_plot_card
        self.ui.runLogCard = self.run_log_card
        self.ui.fittingInlineFeedback = self.inline_feedback

    # -- page ------------------------------------------------------------------------------

    def _install_context(self, legacy_scroll_area) -> None:
        layout = self.ui.verticalLayout_19
        _take_widget(layout, legacy_scroll_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.context_container = FittingContextContainer(
            self.fit_page,
            self.view_model.insitu,
            self.ui.gisaxsFittingPage,
            series_page=self.series_page,
        )
        self.context_container.stack.addWidget(self.page_splitter)  # the former page: kept, never shown
        self.context_bar = self.context_container.context_bar
        self.context_stack = self.context_container.stack
        self.context_button_group = self.context_container.button_group
        self.single_context_button = self.context_container.single_button
        self.insitu_context_button = self.context_container.insitu_button
        self.insitu_series_page = self.context_container.insitu_page
        layout.addWidget(self.context_container, 1)
        self.ui.fittingWorkspace = self
        self.ui.fittingContextStack = self.context_stack
        self.ui.fittingInsituSeriesPage = self.insitu_series_page
        self.ui.fittingSingleContextButton = self.single_context_button
        self.ui.fittingInsituContextButton = self.insitu_context_button
        self.ui.fittingSinglePage = self.fit_page
        # Before In-situ series captures its set-up (the binding connects later): the model of this page.
        self.insitu_series_page.capture_recipe_requested.connect(self._sync_legacy_model)
        self._build_note()
        self.show_context("single")

    def _configure_inputs(self) -> None:
        for widget in self.ui.gisaxsFittingPage.findChildren(INPUT_WIDGET_TYPES):
            normalize_input(widget)
        for checkbox in self.ui.gisaxsFittingPage.findChildren(QCheckBox):
            normalize_checkbox(checkbox)

    def show_context(self, context: str) -> None:
        """Switch between Single analysis and In-situ series (both keep their state)."""
        self.context_container.show_context(context)
        self._sync_note()

    def show_fit_curve(self) -> None:
        """Show Single analysis."""
        self.show_context("single")

    def open_curve(self, path, side: str = "mean") -> bool:
        """A curve from Analyze (``side`` as Analyze names the halves) on the Single analysis page."""
        self._note_closed = None  # handed over anew: what it is is said again
        self.show_context("single")
        return self.fit_page.open_curve(path, side)

    def open_series(self, folder, pattern: str = "*_fit_input.dat") -> bool:
        """A folder of curves (Analyze ▸ Send Series to Fitting) on the In-situ series page."""
        self._note_closed = None
        self.show_context("insitu")
        listed = self.series_page.open_series(folder, pattern)
        self._sync_note()
        return listed

    def _series_from_single(self, folder: str, pattern: str) -> None:
        """Single analysis asked for many curves: those of ``folder`` like its curve, or (no folder) a folder to choose."""
        if folder:
            self.open_series(folder, pattern)
            return
        self.show_context("insitu")
        self.series_page.choose_folder()

    # -- what was handed over: a GIWAXS I(q) is said to be one -----------------------------------------

    def _build_note(self) -> None:
        """The note between the Single analysis / In-situ series bar and the page shown."""
        host = QWidget(self.context_container)
        host.setObjectName("fittingHandoverNoteHost")
        layout = QVBoxLayout(host)
        layout.setContentsMargins(16, 8, 16, 0)
        self.handover_note = ErrorBanner("", "", host, level="warning")
        self.handover_note.setObjectName("fittingHandoverNote")
        self.handover_note.dismiss_button.setToolTip("Close this note")
        layout.addWidget(self.handover_note)
        self.handover_note.hide()
        host.hide()
        self.note_host = host
        self.context_container.layout().insertWidget(1, host)
        self.handover_note.dismissed.connect(self._note_dismissed)
        self.fit_page.curveOpened.connect(self._single_opened)
        # The series page lists its frames anew whenever its curves change (Choose Folder, a drop, a project):
        # looked at once the listing is done (the timer goes with the note).
        self._note_timer = QTimer(host)
        self._note_timer.setSingleShot(True)
        self._note_timer.setInterval(0)
        self._note_timer.timeout.connect(self._sync_note)
        self.series_page.frame_list.model().modelReset.connect(self._note_timer.start)
        self._run_timer = QTimer(host)
        self._run_timer.setSingleShot(True)
        self._run_timer.setInterval(0)
        self._run_timer.timeout.connect(self._series_run_ended)
        self._run_ends = _RunEnds(self.series_page.status_progress, self._run_timer)

    def is_giwaxs_curve(self, path) -> bool:
        """Whether Analyze wrote the curve file as the I(q) of a GIWAXS frame. A file that does not say how
        its points were measured (a curve from elsewhere) is taken as it is."""
        path = str(path)
        try:
            stamp = os.stat(path).st_mtime_ns
        except OSError:
            return False
        known = self._observed.get(path)
        if known is None or known[0] != stamp:
            outcome = self.view_model.load_curve(LoadCurveRequest(Path(path), "angstrom"))
            observation = getattr(outcome.value, "observation", None) if outcome.error is None else None
            known = self._observed[path] = (stamp, str((observation or {}).get("source", "")))
            while len(self._observed) > 8:  # the curve of Single analysis and the first one of a few series
                self._observed.pop(next(iter(self._observed)))
        return known[1] == GIWAXS_SOURCE

    def _noted(self) -> Optional[tuple]:
        """``(kind, what it is about)`` when the page shown holds a GIWAXS I(q): the curve of Single analysis,
        or the series (judged by its first curve)."""
        if self.context_stack.currentWidget() is self.series_page:
            paths = self.series_page.paths
            return ("insitu", self.series_page.folder) if paths and self.is_giwaxs_curve(paths[0]) else None
        curve = self.fit_page.session.curve
        return ("single", curve.path) if curve is not None and curve.path and self.is_giwaxs_curve(curve.path) else None

    def _sync_note(self) -> None:
        """Show the note when the page shown holds a GIWAXS I(q) and the note was not closed for it."""
        if not hasattr(self, "note_host"):  # (the first ``show_context``, before the note is built)
            return
        noted = self._noted()
        shown = noted is not None and noted != self._note_closed
        if shown:
            self.handover_note.set_message(tr(NOTE_TITLE[noted[0]]), self._note_text(noted))
        self.handover_note.setVisible(shown)
        self.note_host.setVisible(shown)

    def _note_text(self, noted: tuple) -> str:
        """What the note says: what Fitting's models are for — and, above a series a run has just been made
        on, what came of it."""
        run = self._note_run
        if noted[0] == "insitu" and run is not None and run[0] is self.series_page.fits and run[1] == noted[1]:
            _fits, _folder, poor, fitted = run
            if poor:
                return trf(NOTE_RUN_POOR, poor=poor, fitted=fitted, limit=f"{POOR_CHI2:g}")
            return trf(NOTE_RUN, fitted=fitted)
        return tr(NOTE_TEXT[noted[0]])

    def _series_run_ended(self) -> None:
        """A run of the In-situ series ended (``_RunEnds``). On GIWAXS I(q) curves the note comes back — also
        when it was closed for this series — and says how many frames are poor fits: the page's own last
        sentence counts every converged frame as fitted."""
        page = self.series_page
        if page.running or not page.fits or not page.paths or not self.is_giwaxs_curve(page.paths[0]):
            return
        frames = list(page.fits.values())
        counts = series_counts(frames)
        poor = sum(1 for frame in frames if poor_fit(frame))
        self._note_run = (page.fits, page.folder, poor, counts["ok"] + counts["not_converged"])
        self._note_closed = None
        self._sync_note()

    def _note_dismissed(self) -> None:
        self._note_closed = self._noted()  # closed for this curve or series: not shown again for it
        self.note_host.hide()

    def _single_opened(self, path: str) -> None:
        """A curve opened in Single analysis, however it came (Analyze, Open Curve, a drop, a frame of the
        series): for a GIWAXS I(q) the particle search a first curve starts with (shape search, fit1d) is taken back — the
        Curve step is shown and the method is Refine — and the status line says why."""
        page = self.fit_page
        if self.is_giwaxs_curve(path):
            if page.method() in ("shapes", "fit1d"):  # particle searches are for a GISAXS cut
                page.set_method("local")
            page.show_step("curve")
            name = Path(path).name
            page._status(lambda: trf(GIWAXS_OPENED, name=name))
        self._sync_note()

    def refresh_language(self) -> None:
        """After a switch of the interface language: the note, said again."""
        self._sync_note()

    def _frame_in_single(self, path: str, model) -> None:
        """A frame of the series, with its fitted model, in Single analysis — on the points the series
        fitted (its halves, range and left-out points), so that the next Start reads the same settings."""
        self.show_context("single")
        settings = getattr(self.series_page, "_settings", None)
        page = self.fit_page
        if not page.open_curve(path, settings.side if settings is not None else page.session.side):
            return
        if settings is not None:
            page.session.excluded = set(settings.excluded)
            page.set_range(settings.q_range, record=False)  # with the curve: Undo then brings back the model
        if model is not None:
            page.set_model(model)
            page._status(lambda: tr("The range and left-out points of the series are kept; the series now starts from "
                                    "this frame's model (Undo brings the previous one back)."))
        else:
            page._status(lambda: tr("The range and left-out points of the series are kept."))  # again after a switch

    def show_solution(self, row: dict) -> bool:
        """A solution of Analyze's automatic analysis as the model of Single analysis."""
        self.show_context("single")
        return self.fit_page.show_solution(row)

    # -- the former pages (built, never shown) ------------------------------------------------

    def attach_legacy(self, binding) -> None:
        """The binding of the former pages follows the curve and the model of Single analysis, and opens
        1D Predict from its button (it runs no series: module docstring)."""
        self._legacy = binding
        self.fit_page.curveOpened.connect(self._legacy_curve)
        self.fit_page.batchRequested.connect(binding.open_ai_fitting_workspace)
        curve = self.fit_page.session.curve
        if curve is not None and curve.path:
            self._legacy_curve(curve.path)

    def _legacy_curve(self, path: str) -> None:
        if self._legacy is None:
            return
        try:
            self._legacy.import_1d_file(path, q_view=LEGACY_Q_VIEW.get(self.fit_page.session.side, "fold"))
        except Exception as exc:  # the former page only; Single analysis itself has the curve
            self.fit_page._log(trf(LEGACY_CURVE_FAILED, error=exc))

    def _sync_legacy_model(self) -> None:
        if self._legacy is None:
            return
        try:
            self._legacy._load_parameter_mapping(to_manual(self.fit_page.session.model))
        except Exception as exc:
            self.fit_page._log(trf(LEGACY_MODEL_FAILED, error=exc))

    # -- sizes -----------------------------------------------------------------------------

    def _set_page_sizes(self, left=None) -> None:
        total = max(self.page_splitter.width(), CONTROLS_MIN_WIDTH + RESULTS_MIN_WIDTH)
        left = int(left) if left else CONTROLS_TARGET_WIDTH
        left = max(CONTROLS_MIN_WIDTH, min(left, total - RESULTS_MIN_WIDTH))
        self.page_splitter.setSizes([left, max(RESULTS_MIN_WIDTH, total - left)])

    def restore_sizes(self) -> None:
        stored = self.preferences.get(self.SETTINGS_KEY, None)
        left = stored.get("left") if isinstance(stored, dict) else None
        QTimer.singleShot(0, lambda: self._set_page_sizes(left))

    def save_state(self) -> None:
        sizes = self.page_splitter.sizes()
        if sizes and sizes[0] > 0:
            self.preferences.set(self.SETTINGS_KEY, {"left": int(sizes[0])})


__all__ = ["FittingWorkspace"]
