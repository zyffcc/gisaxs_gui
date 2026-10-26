"""Hand-overs between the workspaces, composed here because no feature imports another.

* Analyze ▸ Send to Fitting — the curve file Analyze wrote, opened in Fitting ▸ Single analysis. The notice
  on the Fitting page says where that file is: it was written on the way, next to the data or, when that
  folder cannot be written, where the person chose.
* Results ▸ Refine in Fitting and Fit details ▸ Show in Fitting — the same hand-over, the second with a
  solution of the automatic analysis as the model.
* Analyze ▸ Send Series to Fitting — the folder of curves a batch wrote, listed in Fitting ▸ In-situ series.
* Analyze ▸ Series ▸ Send to Compare — the map joins the comparison, and Compare is shown.
* The Start page's series entry — the frames open in Analyze with its Series tab in view.
* Sputter ▸ Show in Analyze — the raw frames of one time window, summed, in Analyze (``sputter_handover.py``).

What Fitting says about a curve that is no GISAXS cut is Fitting's own (``FittingWorkspace``); which curves
of a folder are one series is Compare's.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices

from .presentation.components import dismiss_toasts, show_toast, visible_toasts
from .presentation.i18n import tr, trf
from .sputter_handover import SputterHandoverMixin

SERIES_TASKS = {"series": "open_folder", "series_folder": "open_folder", "series_files": "open_files"}
"""The Start page's series entry: its card (``series``) and its two links (a folder of frames; chosen frames,
or one file that holds the whole series) — each with the Analyze page's dialog that opens it."""
SERIES_CARD = "series"
"""The card itself leads straight to the Series tab when frames are listed; a link always opens its dialog."""
SENT_TO_FITTING = "Sent to Fitting — the curve file is in {folder}"
"""The notice on the Fitting page (its status line names the file; a long name would make the notice tall)."""
SERIES_NOT_OPENED = "Series not opened in Fitting — the exported curves are in {folder}."


def short_folder(folder) -> str:
    """``folder`` as a notice has room for it: its last two names after “…” (a short path whole)."""
    parts = Path(folder).parts
    return str(folder) if len(parts) <= 3 else str(Path("…", *parts[-2:]))


class HandoverMixin(SputterHandoverMixin):
    """Needs the composed window: ``ui``, ``analyze_page``, ``compare_page``, ``fitting_workspace``,
    ``show_page``, ``current_page_key`` and ``_page_requested``."""

    # -- Analyze to Fitting ----------------------------------------------------------------------

    _sent_notice = None
    """The notice of the curve sent last (one at a time: the next send replaces it)."""

    def _fitting_runtime(self):
        runtime = getattr(self.ui, "runtime", None)
        if runtime is None:
            raise RuntimeError(tr("Fitting is still starting; try again in a moment."))
        return runtime

    def _send_curve_to_fitting(self, path, side: str = "both_abs") -> None:
        """Load an Analyze curve into Fitting and show it ready to fit; say where its file is."""
        self._fitting_runtime().navigate("fitting")
        if not self.fitting_workspace.open_curve(path, side):
            raise RuntimeError(self.fitting_workspace.fit_page.status_label.text())
        path = Path(path)
        dismiss_toasts(self.ui)  # the page's own “Opened …”: this one says the same and where the file is
        if self._sent_notice in visible_toasts(self.ui):
            self._sent_notice.close()  # (about the curve sent before)
        self._sent_notice = show_toast(
            self.ui, trf(SENT_TO_FITTING, folder=short_folder(path.parent)), level="ok",
            action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))))

    def _send_to_fitting(self) -> None:
        """Analyze's Send to Fitting for the frame shown, started from here (Refine in Fitting, Show in
        Fitting): the page's own way that asks where to write when the data folder takes no file."""
        page = self.analyze_page
        getattr(page, "send_to_fitting", page.fit_current)()

    def _show_fit_solution(self, row: dict) -> None:
        """Results ▸ Fit details ▸ Show in Fitting: the prepared curve in Fitting with that solution as its model."""
        self._send_to_fitting()
        if self.current_page_key() == "fitting":
            self.fitting_workspace.show_solution(row)

    def _send_series_to_fitting(self, folder) -> bool:
        """Open Fitting ▸ In-situ series on the folder Analyze wrote; a refused hand-over leaves the
        exported files available, also when the person chose to keep unsaved results."""
        self._fitting_runtime().navigate("fitting")
        if self.fitting_workspace.open_series(folder, "*_fit_input.dat"):
            return True
        folder = Path(folder)
        show_toast(self.ui, trf(SERIES_NOT_OPENED, folder=short_folder(folder)), level="warning",
                   action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))))
        return False

    # -- Analyze to Compare ------------------------------------------------------------------------

    def _send_series_to_compare(self, series_map, name: str) -> None:
        """Analyze ▸ Series ▸ Send to Compare: the map joins the comparison, and Compare is shown."""
        if self.compare_page.add_map(series_map, name) is not None:
            self.show_page("compare")

    # -- the Start page's series entry ---------------------------------------------------------------

    def _open_series(self, key: str) -> None:
        """Analyze with its Series tab in view: where a series is worked on. A link (``series_folder``,
        ``series_files``: one NeXus / HDF5 file may hold the whole series) always asks for its folder or files,
        also for the next series beside the frames listed; the card (``series``) asks only while none are
        listed. The tab then says how many frames are listed and what Build Map does. Nothing chosen and
        nothing listed: Analyze stays as it is."""
        page = self.analyze_page
        self._page_requested("analyze")
        if key != SERIES_CARD or not page.view_model.state.files:
            self._open_and_guide(getattr(page, SERIES_TASKS[key]))
        if page.view_model.state.files:
            page.show_right("series")


__all__ = ["HandoverMixin", "SENT_TO_FITTING", "SERIES_CARD", "SERIES_NOT_OPENED", "SERIES_TASKS", "short_folder"]
