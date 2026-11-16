"""Fitting workspace: Single analysis (one curve, ``single/page.py``) and In-situ series (every curve of a
folder, ``single/series_page.py``).

Curves come from Analyze (Send to Fitting, Send Series to Fitting) or from curve files. Fitting's models are
particle form factors for the horizontal cut of a GISAXS frame. A curve Analyze marked as the I(q) of a
GIWAXS frame is therefore opened with a note that says so (``handover_note``, above the page that holds such
a curve or a series of them, until it is closed) and, in Single analysis, without the particle-shape search
that a first curve otherwise starts with. A run of the In-situ series on such curves can still be started; when
it ends the note comes back, closed or not, and says what came of it (how many frames are poor fits).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QEvent, QObject, QTimer
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from src.gimap.app.ports import UserPreferencesRepository
from src.gimap.app.presentation.components import ErrorBanner
from src.gimap.app.presentation.i18n import tr, trf
from ..application import LoadCurveRequest
from ..application.series_fit import POOR_CHI2, poor_fit, series_counts
from .single.page import FitPage
from .single.series_page import FitSeriesPage
from .workspace_context import FittingContextContainer
from .workspace_imports import CurveImportMixin
from ..application.models import InSituSourceFrame

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


class FittingWorkspace(CurveImportMixin):
    """Single analysis (one curve) and In-situ series (many curves) of Fitting."""

    def __init__(
        self,
        ui,
        *,
        preferences: UserPreferencesRepository,
        view_model,
        quick_fit=None,
        frames=None,
    ):
        self.ui = ui
        self.preferences = preferences
        self.view_model = view_model
        self.fit_page = FitPage(view_model, quick_fit=quick_fit, preferences=preferences)
        self.fit_page.setObjectName("fittingSinglePage")
        self.series_page = FitSeriesPage(view_model, self.fit_page, preferences=preferences, frames=frames)
        self.series_page.editModelRequested.connect(lambda: self.show_context("single"))
        self.series_page.openFrameRequested.connect(self._frame_in_single)
        self.fit_page.seriesRequested.connect(self._series_from_single)
        self.fit_page.importCurvesRequested.connect(self.import_curves)
        self.series_page.importCurvesRequested.connect(self.import_curves)
        self._observed: dict[str, tuple] = {}
        self._note_closed: Optional[tuple] = None
        self._note_run: Optional[tuple] = None
        """``(the run's frames, the folder, poor fits, frames fitted)`` of the last run on a GIWAXS series."""
        self._install_context()

    # -- page ------------------------------------------------------------------------------

    def _install_context(self) -> None:
        layout = QVBoxLayout(self.ui.gisaxsFittingPage)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.context_container = FittingContextContainer(self.fit_page, self.series_page, self.ui.gisaxsFittingPage)
        self.context_bar = self.context_container.context_bar
        self.context_stack = self.context_container.stack
        self.context_button_group = self.context_container.button_group
        self.single_context_button = self.context_container.single_button
        self.insitu_context_button = self.context_container.insitu_button
        layout.addWidget(self.context_container, 1)
        self.ui.fittingWorkspace = self
        self.ui.fittingContextStack = self.context_stack
        self.ui.fittingSingleContextButton = self.single_context_button
        self.ui.fittingInsituContextButton = self.insitu_context_button
        self.ui.fittingSinglePage = self.fit_page
        self._build_note()
        self.show_context("single")

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
        source = InSituSourceFrame.from_token(path)
        if source.columns is not None and source.columns.sigma is None:
            # Shared-q imports cannot be Analyze exports. Do not reread their entire table for this note.
            return False
        try:
            stamp = os.stat(source.path).st_mtime_ns
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
            name = InSituSourceFrame.from_token(path).display_name
            page._status(lambda: trf(GIWAXS_OPENED, name=name))
        self._sync_note()

    def refresh_language(self) -> None:
        """After a switch of the interface language: the note, said again."""
        self._sync_note()

    def _frame_in_single(self, path: str, model) -> None:
        """A frame of the series, with its fitted model, in Single analysis — on the points the series
        fitted (its halves, range and left-out points), so that the next Start reads the same settings."""
        self.show_context("single")
        has_run = bool(self.series_page.fits or self.series_page.running)
        settings = self.series_page._settings if has_run else self.series_page._stage_settings()
        page = self.fit_page
        unit = self.series_page._unit if has_run else self.series_page.source_unit_setting()
        if not page.open_curve(path, settings.side, unit=unit):
            return
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


__all__ = ["FittingWorkspace"]
