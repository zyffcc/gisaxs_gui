"""The island height H from the q_z fringes on the page (a mixin of ``SputterPage``).

When a result is shown, fresh outputs of the fringe analysis are read at once (``FringeService.cached``); when there
are none and the result holds binned cuts, the analysis runs as a background task (``FRINGES_KEY``; a few seconds to
half a minute) and the Size Trend, the Windows table and the Reliability tab are redrawn when it is done — if the same
result is still shown. A newer result stops the older run. A failure is written to the log under Details and the page
goes on without H; that result is not measured again while the page lives (showing it again does not retry).
"""

from __future__ import annotations

from pathlib import Path

from src.gimap.app.presentation.i18n import tr

from ..application import FRINGE_STATUS_NAMES, fringe_paragraph
from .texts import message_text, text

FRINGES_KEY = "sputter_fringes"
FRINGES_LOG = "Island height from the q_z fringes: {error}"
FRINGES_RUNNING = "Measuring the island height from the q_z fringes in the background…"
FRINGES_DONE = "Island height from the q_z fringes: {status} ({folder})"
TEMPLATES = (FRINGES_LOG, FRINGES_RUNNING, FRINGES_DONE)


class FringeFlowMixin:
    def _init_fringes(self) -> None:
        self.fringes = None
        self._fringes_folder = None
        self._fringe_job = None  # {"folder": ..., "stop": bool} of the run in the background
        self._fringe_failures: set = set()  # result folders whose run failed: not run again while the page lives

    def _update_fringes(self, result) -> None:
        """Fresh outputs of ``result`` now, or a background run when it has binned cuts and none are fresh."""
        folder = Path(result.folder)
        if self._fringes_folder != folder:
            self.fringes, self._fringes_folder = None, folder
        service = getattr(self.service, "fringes", None)
        if service is None:
            return
        try:
            cached = service.cached(result)
            possible = cached is None and service.can_measure(result)
        except Exception as exc:  # noqa: BLE001 - never fail the page because of H
            self.log.appendPlainText(text(FRINGES_LOG, error=message_text(exc)))
            return
        if cached is not None:
            self.fringes = cached
            return
        job = self._fringe_job
        if not possible or folder in self._fringe_failures or (
                job is not None and job["folder"] == folder and not job["stop"]):
            return
        self._stop_fringes()
        job = self._fringe_job = {"folder": folder, "stop": False}
        self.log.appendPlainText(text(FRINGES_RUNNING))
        self.tasks.submit(FRINGES_KEY, lambda: service.measure(result, should_stop=lambda: job["stop"]),
                          on_done=lambda heights: self._fringes_done(job, heights),
                          on_error=lambda message, _details: self._fringes_failed(job, message))

    def _fringes_done(self, job: dict, heights) -> None:
        if self._fringe_job is job:
            self._fringe_job = None
        if heights is None or job["stop"] or self.result is None or Path(self.result.folder) != job["folder"]:
            return
        self.fringes, self._fringes_folder = heights, job["folder"]
        status = tr(FRINGE_STATUS_NAMES.get(heights.status, heights.status))
        self.log.appendPlainText(text(FRINGES_DONE, status=status, folder=heights.folder))
        self._draw_trend()
        self._fill_table()
        self._show_report()

    def _fringes_failed(self, job: dict, message: str) -> None:
        if self._fringe_job is job:
            self._fringe_job = None
        if not job["stop"]:
            self._fringe_failures.add(job["folder"])
            self.log.appendPlainText(text(FRINGES_LOG, error=message_text(message)))

    def _stop_fringes(self) -> None:
        if self._fringe_job is not None:
            self._fringe_job["stop"] = True
            self._fringe_job = None

    def fringe_report(self) -> str:
        return fringe_paragraph(self.fringes, text)


__all__ = ["FRINGES_KEY", "FringeFlowMixin", "TEMPLATES"]
