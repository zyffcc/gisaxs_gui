"""What the single-curve page says about its curve: the Curve step's card (the points, the q range, the unit of q
and whether σ is in the file), the fitting range in its fields, and how the points are weighted.

σ decides how much a fit can be trusted, so where it is missing it is said as a warning — in the curve card and
in the Fit step — never only in a fold or a tooltip. With σ, the weighting is the residual plot's tooltip (its axis
Δ/σ shows on every step that σ is used).
"""

from __future__ import annotations

from src.gimap.app.presentation.i18n import tr, trf

from ...application.curve_intake import SIGMA_ESTIMATED, intake_notes, sigma_level
from ...application.single_fit import NM_PER_A, prepare_points
from ..intake_texts import note_text, warning_lines
from ..views.step_widgets import show_line

CURVE_LINE = "{points} points, |q| {low}–{high} nm⁻¹; {fitted} in the fitting range"
CURVE_TIP = "|q| {low_a}–{high_a} Å⁻¹"
SIGMA_FROM_FILE = "σ from the file"
NO_SIGMA_CARD = ("⚠ No σ in the file. Predict parameters: a relative σ of {percent} %, estimated from the scatter of "
                 "ln I about its trend. Refine and Search: ln I is fitted, every point weighted alike.")
NO_SIGMA_FIT = "⚠ No σ in the file: every point weighted alike"
"""The Fit step's line for Refine and Search; for Predict parameters it is ``curve_intake.SIGMA_ESTIMATED``, the
sentence the 1D Predict window and its records use for the same estimate."""
NO_SIGMA_WHY = ("No σ in the file: Refine and Search give every point the same relative weight (ln I is fitted).",
                "No σ in the file: Predict parameters takes a relative σ from the scatter of ln I about its trend "
                "(at least 1 %), the same in the 1D Predict window.")
"""What a missing σ means for each action of the Fit step (the warnings' tooltip)."""
OPENED = "Opened {name}."
WEIGHTED_TIP = "Residuals: {formula}. Each point is weighted by 1/σ from the file."


class CurveInfoMixin:
    """Needs the page's ``session``, ``_data()``, ``_unit_line()`` and the widgets of the Curve and Fit steps."""

    def _fill_curve_card(self) -> None:
        curve, data = self.session.curve, self._data()
        full = None if curve is None else self.session.all_points()
        if curve is None or full is None or not full.q.size:  # the plot says how to open a curve
            self.curve_card.hide()
            self.curve_info.setText("")
            self.curve_card.setToolTip("")
            show_line(self.sigma_warning, "")
            show_line(self.weighting_note, "")
            show_line(self.curve_notes_label, "")
            return
        low, high = float(full.q.min()), float(full.q.max())
        for spin, value in ((self.range_min_spin, low), (self.range_max_spin, high)):
            spin.blockSignals(True)
            spin.setRange(0.0, high * 10)
        chosen = self.session.q_range or (low, high)
        self.range_min_spin.setValue(chosen[0])
        self.range_max_spin.setValue(chosen[1])
        for spin in (self.range_min_spin, self.range_max_spin):
            spin.blockSignals(False)
        fitted = 0 if data is None else data.q.size
        weighted = curve.sigma is not None
        lines = [trf(CURVE_LINE, points=full.q.size, low=f"{low:.4g}", high=f"{high:.4g}", fitted=fitted)]
        lines += [tr(SIGMA_FROM_FILE)] if weighted else []
        notes = self.curve_notes()
        kept = [note for note in notes if note.kind == "nonpositive" and not note.values.get("left_out")]
        lines += [note_text(note) for note in kept]  # with σ they are fitted like any point: said, not a warning
        lines += [line for line in [self._unit_line()] if line]  # where its unit of q came from (the last line)
        self.curve_info.setText("\n".join(lines))
        self.curve_card.setToolTip(trf(CURVE_TIP, low_a=f"{low / NM_PER_A:.4g}", high_a=f"{high / NM_PER_A:.4g}"))
        self.curve_card.show()
        why = "" if weighted else "\n".join(tr(text) for text in NO_SIGMA_WHY)
        level = next((note.values["percent"] for note in notes if note.kind == "sigma_estimated"), "–")
        other = warning_lines(n for n in notes if n.kind != "sigma_estimated" and n not in kept)  # the unit, I ≤ 0
        predict = getattr(self, "method", lambda: "")() == "predict"
        card = [] if weighted else [trf(NO_SIGMA_CARD, percent=level)]
        step = [] if weighted else ["⚠ " + trf(SIGMA_ESTIMATED, percent=level) if predict else tr(NO_SIGMA_FIT)]
        show_line(self.sigma_warning, "\n".join(card + other))
        show_line(self.weighting_note, "\n".join(step + other))  # the Fit step: what its method does
        # Results: next to the values, whatever was assumed of the curve they are of
        show_line(self.curve_notes_label, "\n".join(other + card))
        for label in (self.sigma_warning, self.weighting_note):
            label.setToolTip(why)

    def curve_notes(self) -> list:
        """What is to be said of the curve as it is fitted (``application.curve_intake.IntakeNote``; ``kind``:
        ``q_unit``, ``sigma_estimated``, ``nonpositive``; the counts and numbers in ``values``)."""
        curve, session = self.session.curve, self.session
        if curve is None:
            return []
        try:
            made = prepare_points(curve, session.side, session.q_range, session.excluded)
        except ValueError:
            return []
        data = made.data
        estimated = data.sigma is None
        return intake_notes(curve.q, unit=curve.source_unit, unit_from=curve.unit_from,
                            level=sigma_level(data.q, data.intensity) if estimated else None,
                            negative=made.negative, zero=made.zero, points=int(data.q.size), left_out=estimated)

    def _opened_status(self, name: str, note=None) -> None:
        """The status line after a curve opened: ``note`` (its unit differs from the saved one) or “Opened …”, and
        what was assumed of the curve that changes a result (the unit's range, negative and zero intensities)."""
        said = [n for n in self.curve_notes() if n.kind in ("q_unit", "nonpositive")]
        warn = note or any(n.kind == "q_unit" or n.values.get("left_out") for n in said)
        first = note or (lambda: trf(OPENED, name=name))
        self._status(lambda: " ".join([first(), *(note_text(n) for n in said)]), "warning" if warn else "ok")

    @staticmethod
    def _residuals_tip(weighted: bool) -> str:
        """The residual plot's tooltip: its formula and, with σ, how the points are weighted."""
        if weighted:
            return trf(WEIGHTED_TIP, formula="Δ/σ = (I − model)/σ")
        return tr("Residuals: {formula}").format(formula="Δ ln I = ln(I / model)")


__all__ = ["CURVE_LINE", "CurveInfoMixin", "NO_SIGMA_CARD", "NO_SIGMA_FIT", "OPENED", "SIGMA_FROM_FILE", "WEIGHTED_TIP"]
