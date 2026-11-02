"""What the single-curve page says about its curve: the Curve step's card (the points, the q range, the unit of q
and whether σ is in the file), the fitting range in its fields, and how the points are weighted.

σ decides how much a fit can be trusted, so where it is missing it is said as a warning — in the curve card and
in the Fit step — never only in a fold or a tooltip. With σ, the weighting is the residual plot's tooltip (its axis
Δ/σ shows on every step that σ is used).
"""

from __future__ import annotations

from src.gimap.app.presentation.i18n import tr, trf

from ...application.single_fit import NM_PER_A
from ..views.step_widgets import show_line

CURVE_LINE = "{points} points, |q| {low}–{high} nm⁻¹; {fitted} in the fitting range"
CURVE_TIP = "|q| {low_a}–{high_a} Å⁻¹"
SIGMA_FROM_FILE = "σ from the file"
NO_SIGMA_CARD = "⚠ No σ in the file: every point weighted alike (ln I fitted)"
NO_SIGMA_FIT = "⚠ No σ in the file: every point weighted alike"
NO_SIGMA_WHY = ("No σ in the file: every point gets the same relative weight (ln I is fitted).",
                "No σ in the file: fit1d assumes Poisson errors; the selected systematic error floor is added in "
                "quadrature.")
"""What “every point weighted alike” means for each action of the Fit step (the warnings' tooltip)."""
WEIGHTED_TIP = ("Residuals: {formula}. Each point is weighted by 1/σ from the file; fit1d adds the systematic error "
                "floor (Options).")


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
        lines += [line for line in [self._unit_line()] if line]  # where its unit of q came from (the last line)
        self.curve_info.setText("\n".join(lines))
        self.curve_card.setToolTip(trf(CURVE_TIP, low_a=f"{low / NM_PER_A:.4g}", high_a=f"{high / NM_PER_A:.4g}"))
        self.curve_card.show()
        why = "" if weighted else "\n".join(tr(text) for text in NO_SIGMA_WHY)
        show_line(self.sigma_warning, "" if weighted else tr(NO_SIGMA_CARD))
        show_line(self.weighting_note, "" if weighted else tr(NO_SIGMA_FIT))  # the Fit step: both of its actions
        for label in (self.sigma_warning, self.weighting_note):
            label.setToolTip(why)

    @staticmethod
    def _residuals_tip(weighted: bool) -> str:
        """The residual plot's tooltip: its formula and, with σ, how the points are weighted."""
        if weighted:
            return trf(WEIGHTED_TIP, formula="Δ/σ = (I − model)/σ")
        return tr("Residuals: {formula}").format(formula="Δ ln I = ln(I / model)")


__all__ = ["CURVE_LINE", "CurveInfoMixin", "NO_SIGMA_CARD", "NO_SIGMA_FIT", "SIGMA_FROM_FILE", "WEIGHTED_TIP"]
