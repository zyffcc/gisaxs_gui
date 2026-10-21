"""The halves of an In-situ series: its own choice, “Halves of the cut” in the Curves step.

Every frame of a series is fitted on one curve — the mean of the halves of its cut, or one half — never on
both halves together. Which one is decided in this order (``application/series_fit.series_side``): what
the person chose here for the series that is listed (or a project brought); the halves the first listed fit
input was sent with (Analyze writes them into the file: ``"halves"`` of its ``# observation:`` line); the
halves of Single analysis unless they are “both on |q|”; the mean. A choice that was not made here says
where it comes from. Curves with q on one side only have no halves: the choice is switched off and says so.

Changing it draws the selected frame again, finds the stages again and, after a run, marks the results as
no longer those of the points shown (``_results_stale``). It is kept in a project, written into the series
record (``settings.side`` and ``side_from``) and named in the Start step's summary.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.gimap.app.presentation.i18n import tr, trf

from ...application.series_fit import series_side
from ...application.single_fit import SERIES_SIDES

HALVES_TIP = "Which half of each cut is fitted, in every frame of the series: the mean of both halves, or one half."
SOURCE_TEXT = {
    "files": "from the fit inputs written by Analyze",
    "analyze": "as Analyze sends cuts to Fitting",
    "single": "as in Single analysis",
    "project": "as saved in the project",
}
"""Where a choice that was not made on this page comes from (``halves_source``)."""
ONE_SIDED = "These curves have q on one side only: there are no halves to choose."
BOTH_IN_SINGLE = "Single analysis fits both halves on |q|; a series fits one curve per frame: their mean."


class FitSeriesHalvesMixin:
    """Needs the series page: ``single``, ``paths``, ``running``, ``side_combo``, ``side_label``,
    ``halves_note``, ``_listed()``, ``_preview_curve()``, ``refresh()``."""

    def _connect_halves(self) -> None:
        self._halves_chosen: Optional[str] = None
        """The person's own choice for the listed series (or a project's); ``None``: not chosen here."""
        self._halves_origin = ""
        """``chosen`` or ``project``: who made ``_halves_chosen``."""
        self._halves_seen: Optional[tuple] = None
        """``(frame, halves)`` of the first listed fit input read so far: what it was sent with (or None)."""
        self._curves_read = 0
        self._curves_signed = 0
        for key, text in SERIES_SIDES:
            self.side_combo.addItem(tr(text), key)
        self.side_combo.activated.connect(lambda _index: self.set_series_side(self.side_combo.currentData()))

    # -- what the listed curves say ---------------------------------------------------------

    def _forget_halves(self, *, choice: bool = True) -> None:
        """Other curves are listed: nothing is known of them yet. ``choice``: the person's choice goes too
        (a folder opened anew), else it stays (the same folder listed again)."""
        self._halves_seen, self._curves_read, self._curves_signed = None, 0, 0
        if choice:
            self._halves_chosen, self._halves_origin = None, ""

    def _note_curve(self, index: int, q, observation=None) -> None:
        """A listed curve was read: whether it has both signs of q, and the halves it was sent with."""
        q = np.asarray(q, dtype=float)
        self._curves_read += 1
        self._curves_signed += int(bool(np.any(q < 0) and np.any(q > 0)))
        halves = (observation or {}).get("halves") if isinstance(observation, dict) else None
        if self._halves_seen is None or index < self._halves_seen[0]:
            self._halves_seen = (index, halves if isinstance(halves, str) else None)

    def _probe_signed(self) -> None:
        """While no curve read so far has two halves: a look at a few more of the listed ones — the middle
        one, the last, the second. The stage search reads them all, but later (and never for fewer than three
        curves); until then a first frame that lost a half, or cannot be read, would make the whole series
        one without halves, or hide what it was sent with. Made when the curves are listed, so that the box,
        its note and the Start summary say what Start fits, and again at Start."""
        listed = self._listed()
        if not listed:
            return
        for index in (listed[len(listed) // 2], listed[-1], listed[min(1, len(listed) - 1)]):
            if self._curves_signed:
                return
            self._preview_curve(index)  # (a curve read before is not counted twice)

    def curves_signed(self) -> bool:
        """Whether a listed curve has both halves (as far as they were read; not known yet: taken as so)."""
        return self._curves_read == 0 or self._curves_signed > 0

    # -- the choice -------------------------------------------------------------------------

    def _halves_hints(self) -> tuple:
        """``(chosen, observed, single)``: the choice made here, what the fit inputs were sent with, and the
        halves of Single analysis — those only while it shows them (a cut with both halves is open there)."""
        observed = self._halves_seen[1] if self._halves_seen is not None else None
        session = self.single.session
        single = session.side if session.curve is not None and session.curve.signed else None
        return self._halves_chosen, observed, single

    def series_side(self) -> str:
        """``mean``, ``positive`` or ``negative``: what every frame of the series is fitted on."""
        return series_side(*self._halves_hints(), signed=self.curves_signed())

    def halves_source(self) -> str:
        """Where ``series_side()`` comes from: ``chosen`` (here), ``project``, ``files`` (the fit inputs),
        ``analyze`` (detector frames cut here: the halves Analyze sends to Fitting), ``single`` (Single analysis),
        ``default`` (nothing said: the mean) or ``one-sided`` (no halves)."""
        if not self.curves_signed():
            return "one-sided"
        chosen, observed, single = self._halves_hints()
        side = self.series_side()
        if chosen == side:
            return self._halves_origin or "chosen"
        if observed == side:  # frames cut here: Analyze's choice for Send to Fitting, not a file's record
            return "analyze" if self.frames_mode() else "files"
        return "single" if single == side else "default"

    def set_series_side(self, side, origin: str = "chosen") -> bool:
        """The person's choice (or a project's, ``origin="project"``); refused while a series runs."""
        if self.running or side not in dict(SERIES_SIDES):
            self._show_halves()
            return False
        self._halves_chosen, self._halves_origin = side, origin
        self.refresh()  # the selected frame, the stages and — after a run — the mark on its results
        return True

    def halves_text(self, side: Optional[str] = None, source: Optional[str] = None) -> str:
        """“Mean of both halves (as in Single analysis)”: the halves and where they come from, in the
        interface language — of now, or the given ones (those of a run)."""
        text = tr(dict(SERIES_SIDES)[self.series_side() if side is None else side])
        note = SOURCE_TEXT.get(self.halves_source() if source is None else source)
        return trf("{side} ({source})", side=text, source=tr(note)) if note else text

    def _show_halves(self) -> None:
        """The choice as it stands, where it comes from, and whether there is anything to choose."""
        combo, source = self.side_combo, self.halves_source()
        for index, (_key, text) in enumerate(SERIES_SIDES):  # again after a switch of the language
            combo.setItemText(index, tr(text))
        combo.setCurrentIndex(max(0, combo.findData(self.series_side())))
        one_sided = source == "one-sided"
        combo.setEnabled(bool(self.paths) and not one_sided and not self.running)
        if not self.paths:  # no curves yet: nothing to choose for, nothing to say
            note = ""
        elif one_sided:
            note = tr(ONE_SIDED)
        elif source == "default" and self._halves_hints()[2] == "both":
            note = tr(BOTH_IN_SINGLE)
        else:
            note = tr(SOURCE_TEXT[source]) if source in SOURCE_TEXT else ""
        self.halves_note.setText(note)
        self.halves_note.setVisible(bool(note))
        tip = tr(HALVES_TIP) + ("\n" + note if note else "")
        for widget in (combo, self.side_label):
            widget.setToolTip(tip)

    # -- a project ----------------------------------------------------------------------------

    def halves_state(self) -> dict:
        return {"side": self.series_side(), "side_from": self.halves_source()}

    def apply_halves_state(self, data: dict) -> None:
        """The halves a project kept for its series (an older project has none: the usual order decides)."""
        if data.get("side") in dict(SERIES_SIDES):
            self._halves_chosen, self._halves_origin = data["side"], "project"


__all__ = ["BOTH_IN_SINGLE", "FitSeriesHalvesMixin", "HALVES_TIP", "ONE_SIDED", "SOURCE_TEXT"]
