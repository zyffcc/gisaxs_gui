"""In-situ series of detector frames: every frame cut as Analyze cuts it, then fitted as a curve.

With a pattern of detector files in the Curves step (``*.cbf``, ``*.tif``, ``*.edf``, ``*.nxs`` …) — or a folder that
holds frames and no curves — the series lists the frames of the folder (every frame of a NeXus source; Analyze's
Sum N groups them) and reads each one through the cutter of the composition root (``frames``: Analyze's set-up
applied to it, the curve Analyze writes into ``<stem>_fit_input.dat``). The set-up is taken from Analyze when the
frames are listed, and again with “Take Analyze's Set-up”; a run uses the one it started with, and the record of the
series keeps it (Analyze's settings record and where the cut is pinned). Cut frames are kept in memory only.
"""

from __future__ import annotations

from pathlib import Path

from src.gimap.app.presentation.i18n import tr, trf

from ...application.models import CurveOperationResult
from ...application.series_frames import SeriesCurveReader, is_frame_pattern

SET_UP = "Each frame is cut as Analyze cuts {frame}: instrument {profile}, αi {incidence}°, Sum {sum}{pinned}."
PINNED = ", the cut at rows {low}–{high} of that frame"
NO_SET_UP = "The frames cannot be cut yet: {why}"
TAKEN = "Analyze's set-up is taken: the frames are cut again with it."
FRAMES_LISTED = "{count} detector frames listed; each is cut as in Analyze."


class FitSeriesFramesMixin:
    """Needs the series page: ``pattern_edit``, ``frames_card``, ``frames_label``, ``take_setup_button``,
    ``view_model``, ``running``, ``_list_curves``, ``_status``, ``_preview_curves``, ``_forget_stages``."""

    def setup_series_frames(self, frames) -> None:
        self.frames_bridge = frames
        self._recipe, self._recipe_why = None, ""
        self._frame_items: dict = {}
        self._frame_cuts: dict = {}
        self._frame_cut_stamps: dict = {}
        self.take_setup_button.clicked.connect(self.take_analyze_setup)

    # -- listing ------------------------------------------------------------------------------

    def frames_mode(self) -> bool:
        return self.frames_bridge is not None and is_frame_pattern(self.pattern_edit.text())

    # -- reading ------------------------------------------------------------------------------

    def load_series_curve(self, request) -> CurveOperationResult:
        """Direct read; UI jobs use a captured ``series_curve_loader`` in the background."""
        load = self.series_curve_loader()
        outcome = load(request)
        self._publish_frame_cuts(load.loaded)
        return outcome

    def series_curve_loader(self):
        """Capture the recipe and caches on the GUI thread, before submitting any worker."""
        return self.make_curve_loader(load=self.view_model.read_series_curve, bridge=self.frames_bridge,
            recipe=self._recipe, why=self._recipe_why, items=self._frame_items, cache=self._frame_cuts,
            stamps=self._frame_cut_stamps)

    @staticmethod
    def make_curve_loader(**snapshot):
        return SeriesCurveReader(**snapshot)

    def _publish_frame_cuts(self, loaded):
        """GUI thread only, after the caller has checked the epoch and per-path version."""
        for token, (outcome, stamp) in loaded.items():
            self._frame_cuts[token], self._frame_cut_stamps[token] = outcome, stamp

    # -- Analyze's set-up ---------------------------------------------------------------------

    def _take_recipe(self) -> None:
        self._recipe, self._recipe_why = self.frames_bridge.recipe()
        self._frame_cuts = {}
        self._frame_cut_stamps = {}

    def take_analyze_setup(self) -> None:
        """Analyze's set-up as it is now, for the frames listed (listed again: Sum N may have changed)."""
        if self.running:
            self._status(tr("Stop the running series first."), "warning")
            return
        self._take_recipe()
        self._preview_curves = {}
        self._forget_stages()
        if self._recipe is None:
            self._status(lambda why=self._recipe_why: trf(NO_SET_UP, why=tr(why)), "warning")
        else:
            self._status(lambda: tr(TAKEN), "ok")
        self._list_curves(again=True)
        self.refresh_frames_card()

    def ready_to_cut(self) -> bool:
        """Whether a run of frames can start; says why not."""
        if not self.frames_mode() or self._recipe is not None:
            return True
        self._status(lambda why=self._recipe_why: trf(NO_SET_UP, why=tr(why)), "warning")
        return False

    def frames_record(self) -> dict:
        """What the series record says of the frames: how they were cut."""
        if not self.frames_mode():
            return {}
        return {"detector_frames": True, "cut_as_in_analyze": None if self._recipe is None else self._recipe.record}

    def refresh_frames_card(self) -> None:
        shown = self.frames_mode()
        self.frames_card.setVisible(shown)
        self.take_setup_button.setVisible(shown)
        if not shown:
            return
        if self._recipe is None:
            self.frames_label.setText(trf(NO_SET_UP, why=tr(self._recipe_why or "Analyze is not open.")))
            return
        record = self._recipe.record
        settings = record.get("analyze_settings") or {}
        frame = record.get("set_up_on") or {}
        rows = (record.get("series_cut") or {}).get("horizontal_rows")
        pinned = trf(PINNED, low=f"{rows[0]:.0f}", high=f"{rows[1]:.0f}") if rows else ""
        incidence = record.get("incidence_deg", settings.get("incidence_deg"))  # as the frame was analysed
        self.frames_label.setText(trf(
            SET_UP, frame=Path(str(frame.get("path") or "—")).name, profile=settings.get("profile_name") or "—",
            incidence="—" if incidence is None else f"{float(incidence):g}", sum=record.get("sum_count", 1),
            pinned=pinned))


__all__ = ["FitSeriesFramesMixin", "FRAMES_LISTED", "NO_SET_UP", "SET_UP", "TAKEN"]
