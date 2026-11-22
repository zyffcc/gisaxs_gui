"""The beam tail's ranges follow the curve on the page (Physics and Methods, section 9.6).

When a curve is opened, the tail's ranges that nobody chose — those of the starting model, or those this page set
for the previous curve — are set for this curve (``single_fit.tail_for_curve``: A up to 10× the largest intensity, w
from 0.3 q_min to min(30 q_min, q_max), ν from 1 to 6); a starting model also takes w and ν from the curve's first
points. Ranges the person typed are left as they are. Like the scales of a starting model (``fit_scales``), this is
part of opening the curve, not a step of Undo.
"""

from __future__ import annotations

from src.gimap.app.presentation.i18n import trf

from ...application.single_fit import INFO, tail_for_curve

TAIL = ("res_amplitude", "res_width", "res_exponent")
TIED = ("The beam tail's ranges were set from this curve: A ≤ {amplitude}, w {low}–{high} nm⁻¹, ν ≤ {exponent} "
        "(Ranges shows them in Model).")


def _ranges(model) -> tuple:
    return tuple((model.globals[key].lower, model.globals[key].upper) for key in TAIL)


def _generic(model):
    for key in TAIL:
        info = INFO[key]
        value = min(max(model.globals[key].value, info.lower), info.upper)
        model = model.with_parameter(("globals", key), lower=info.lower, upper=info.upper, value=value)
    return model


class TailRangesMixin:
    """Needs ``session``."""

    _tail_auto = None
    """The tail's ranges this page set for the last curve (they follow the next curve too)."""

    def follow_tail(self, data) -> None:
        model = self.session.model
        auto = self._tail_auto is not None and _ranges(model) == self._tail_auto
        if self.session.edited and not auto:
            return  # the person's model stays as it is; generic ranges are tied when it is fitted (fit_resolving)
        if auto:
            model = _generic(model)
        followed = tail_for_curve(model, data, start=not self.session.edited)
        if followed is not model:
            self.session.model = followed
            self._tail_auto = _ranges(followed)

    def tie_tail_before_fit(self, data) -> None:
        """Before Refine or Search: a tail still with the generic ranges gets this curve's, in Model (a step of Undo,
        said in the status line), so that the fit keeps exactly the ranges Model shows."""
        model = self.session.model
        followed = tail_for_curve(model, data)
        if followed is model:
            return
        self._tail_auto = _ranges(followed)
        g = followed.globals
        self.set_model(followed, message=lambda: trf(
            TIED, amplitude=f"{g['res_amplitude'].upper:.3g}", low=f"{g['res_width'].lower:.3g}",
            high=f"{g['res_width'].upper:.3g}", exponent=f"{g['res_exponent'].upper:g}"))


__all__ = ["TailRangesMixin"]
