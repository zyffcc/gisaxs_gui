"""Which method the single-curve page recommends for a curve, and the line under the method that says why.

A GISAXS horizontal cut that has not been fitted yet gets Predict parameters (when GIMaP's network is there); every
other curve keeps the method chosen before. Also the texts of a fit's notes (``note_text``) and of its errors."""

from __future__ import annotations

import re

from src.gimap.app.presentation.i18n import tr, trf

from ...application.single_fit import TOO_FEW_FOR_FREE
from .predict import MISSING
from ..views.step_widgets import show_line

GISAXS_CUT = "native_detector_columns"
"""``observation["source"]`` of a GISAXS horizontal cut sent from Analyze."""
RECOMMENDED = "Recommended for this curve: a GISAXS cut not fitted yet."
STATE_WORDS = ("upper", "lower")


def number(value: float) -> str:
    """Three significant digits, without an exponent for lengths up to 10⁶ nm."""
    return f"{value:.0f}" if 1000 <= abs(value) < 1e6 else f"{value:.3g}"


_TOO_FEW = re.compile(re.escape(TOO_FEW_FOR_FREE).replace(r"\{points\}", r"(?P<points>\d+)")
                      .replace(r"\{free\}", r"(?P<free>\d+)").replace(r"\{minimum\}", r"(?P<minimum>\d+)"))


def fit_error_text(message: str) -> str:
    """A fit's error in the interface language (the application's sentences; one carries numbers)."""
    found = _TOO_FEW.fullmatch(str(message))
    if found:
        return trf(TOO_FEW_FOR_FREE, **found.groupdict())
    prefix = "The network record changed after loading: "
    if message.startswith(prefix):
        return trf("The network record changed after loading: {path}", path=message[len(prefix):])
    return tr(message)


def note_text(note: dict, name=None) -> str:
    """A note of the fit (``template`` and ``values``) in the interface language; ``name``: the name to show for
    ``values["path"]`` (a parameter of Fitting's model)."""
    values = dict(note.get("values") or {})
    if not values:
        return fit_error_text(note["template"])
    if values.get("side") in STATE_WORDS:
        values["side"] = tr(values["side"])
    if values.get("reason"):
        values["reason"] = tr(values["reason"])
    if name is not None and values.get("path"):
        values["name"] = name(tuple(values["path"]))
    return trf(note["template"], **values)


class MethodHintMixin:
    def recommended_method(self) -> str:
        """``predict`` for a GISAXS horizontal cut not fitted yet when the network is there; else empty."""
        session = self.session
        unfitted = session.result is None and not session.solutions
        if session.curve is None or not getattr(self, "_gisaxs_cut", False) or not unfitted:
            return ""
        return "predict" if self.predict_available() else ""

    def _curve_opened_step(self, observation) -> None:
        """After a curve is opened: the method recommended for it and the step to show."""
        curve = self.session.curve
        self._gisaxs_cut = (observation or {}).get("source") == GISAXS_CUT or (not observation and curve.signed)
        if self.recommended_method():
            self.set_method("predict")
            self.show_step("fit")
        else:
            self.show_step("model")
        self._refresh_method_hint()

    def _refresh_method_hint(self) -> None:
        """The line under the method: why it is recommended for this curve, or that Predict's network is missing."""
        if not hasattr(self, "method_combo"):
            return
        method = self.method()
        hint = ""
        if method and method == self.recommended_method():
            hint = tr(RECOMMENDED)
        elif method == "predict" and not self.predict_available():
            from ...application.single_predict import network_file
            hint = trf(MISSING, name=network_file())
        show_line(self.method_hint, hint)


__all__ = ["GISAXS_CUT", "MethodHintMixin", "RECOMMENDED", "fit_error_text", "note_text", "number"]
