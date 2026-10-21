"""fit1d on the single-curve page, around the fit: which method is recommended for the curve, the notes before a
fit, the constraints it used and the compositions it tried (``auto:``)."""

from __future__ import annotations

from PyQt5.QtWidgets import QTableWidgetItem

from src.gimap.app.presentation.i18n import tr, trf

from ...application.single_fit import FitModel, model_from_solution
from ...application.single_fit1d import (
    AUTO_IGNORES_MODEL,
    COMPOSITIONS,
    NO_PARTICLES,
    SHAPES,
    fit1d_constraints,
    is_auto,
    prefit_notes,
)
from ..views.step_widgets import fit_to_rows

GISAXS_CUT = "native_detector_columns"
"""``observation["source"]`` of a GISAXS horizontal cut sent from Analyze."""
RECOMMENDED = "Recommended for this curve: a GISAXS cut not fitted yet."
SOURCES = {"model": " (Model)", "widened": " (widened here)", "follow": " (near the previous frame)",
           "bound_upper": " (a bound: every larger value fits as well)",
           "bound_lower": " (a bound: every smaller value fits as well)"}
CONSTRAINTS_USED = "Constraints passed to fit1d: {items}."
NO_CONSTRAINTS = "No constraints: fit1d searched its default ranges for this q range."
MEASURABLE = "This q range measures R {r_low}–{r_high} nm and D {d_low}–{d_high} nm."
TRIED = "Compositions tried: {composition}; the best, {fitted}, is shown (Model's values and ranges not used)."
INDISTINGUISHABLE = "indistinguishable from {others}"
STATE_WORDS = ("upper", "lower")
FLAG_TEXT = {"at_bound": "a parameter at an edge", "weakly_constrained": "not all pinned down",
             "quality": "misses the curve", "ood": "unusual curve", "fallback": "robust search"}
"""The kit's flags of a candidate, in words."""


def _cell(text, tip: str = "") -> QTableWidgetItem:
    item = QTableWidgetItem(str(text))
    if tip:
        item.setToolTip(tip)
    return item


def _number(value: float) -> str:
    """Three significant digits, without an exponent for lengths up to 10⁶ nm."""
    return f"{value:.0f}" if 1000 <= abs(value) < 1e6 else f"{value:.3g}"


number = _number


def note_text(note: dict, name=None) -> str:
    """A note of the fit (``template`` and ``values``) in the interface language; ``name``: the name to show for
    ``values["path"]`` (a parameter of Fitting's model)."""
    values = dict(note.get("values") or {})
    if values.get("side") in STATE_WORDS:
        values["side"] = tr(values["side"])
    if values.get("reason"):
        values["reason"] = tr(values["reason"])
    if name is not None and values.get("path"):
        values["name"] = name(tuple(values["path"]))
    return trf(note["template"], **values)


class Fit1DReviewMixin:
    def setup_fit1d_review(self) -> None:
        self.fit1d_candidates_table.itemSelectionChanged.connect(self._fit1d_candidate_chosen)
        self.fit1d_candidates_table.cellDoubleClicked.connect(lambda _row, _column: self.use_fit1d_candidate())
        self.fit1d_use_candidate_button.clicked.connect(self.use_fit1d_candidate)

    # -- before a fit ------------------------------------------------------------------------

    def recommended_method(self) -> str:
        """fit1d for a GISAXS horizontal cut that has not been fitted yet; else empty (no recommendation)."""
        session = self.session
        unfitted = session.result is None and session.native_result is None and not session.solutions
        return "fit1d" if session.curve is not None and self._gisaxs_cut and unfitted else ""

    def _curve_opened_step(self, observation) -> None:
        """After a curve is opened: the method recommended for it and the step to show."""
        curve = self.session.curve
        self._gisaxs_cut = (observation or {}).get("source") == GISAXS_CUT or (not observation and curve.signed)
        self._fit1d_candidates = []
        if self.recommended_method() == "fit1d":
            if not self.session.edited:  # no particles of the user's: let fit1d try the compositions
                self.fit1d_composition.setCurrentIndex(self.fit1d_composition.findData("auto:sphere,vc:3"))
            self.set_method("fit1d")
            self.show_step("fit")
        elif not self.session.edited and self.quick_fit is not None:
            self.set_method("shapes")  # a first curve: find the family first
            self.show_step("fit")
        else:
            if self.method() == "fit1d":  # chosen for the curve before; this one is not known to be a GISAXS cut
                self.set_method("local")
            self.show_step("model")
        self._refresh_fit1d_action()

    def _refresh_fit1d_action(self) -> None:
        """The sentence under the method (what it does, whether it is the one recommended) and fit1d's notes."""
        if not hasattr(self, "fit1d_composition") or not hasattr(self, "method_combo"):
            return
        method = self.method()
        hint = tr(self.method_combo.itemData(self.method_combo.currentIndex(), 3) or "")  # Qt.ToolTipRole
        if method and method == self.recommended_method():
            hint = tr(RECOMMENDED) + " " + hint
        self.method_hint.setText(hint)
        for index, (_value, text) in enumerate(COMPOSITIONS):
            self.fit1d_composition.setItemText(index, tr(text))
        notes = []
        if method == "fit1d":
            settings = self.fit1d_settings()
            model = self.session.model
            if is_auto(settings):
                try:
                    ignored = bool(fit1d_constraints(model, {**settings, "fit1d_composition": "model"}))
                except ValueError:
                    ignored = False
                notes = [tr(AUTO_IGNORES_MODEL)] if ignored else []
            elif not model.components or any(c.family not in SHAPES for c in model.components):
                notes = [tr(NO_PARTICLES)]
            else:
                notes = [trf(note.template, **note.values) for note in prefit_notes(self._data(), model, settings)]
        self.fit1d_notes.setText("\n".join(notes))
        self.fit1d_notes.setVisible(bool(notes))

    # -- after a fit -----------------------------------------------------------------------------

    def fit1d_constraints_text(self, native) -> str:
        record = native.get("fit1d_constraints") or {}
        bounds = native.get("fit1d_bounds") or {}
        lines = []
        if str(record.get("composition") or "").startswith("auto"):
            lines.append(trf(TRIED, composition=record["composition"],
                             fitted=record.get("fitted_composition") or ""))
        used = record.get("used") or []
        if used:
            lines.append(trf(CONSTRAINTS_USED, items="; ".join(item["text"] + tr(SOURCES.get(item.get("source"), ""))
                                                                for item in used)))
        elif not lines:
            lines.append(tr(NO_CONSTRAINTS))
        window = bounds.get("measurable") or {}
        if window.get("R_nm") and window.get("D_nm"):
            (r_low, r_high), (d_low, d_high) = window["R_nm"], window["D_nm"]
            lines.append(trf(MEASURABLE, r_low=_number(r_low), r_high=_number(r_high), d_low=_number(d_low),
                             d_high=_number(d_high)))
        return "\n".join(lines)

    def fill_fit1d_candidates(self, native) -> bool:
        """The compositions fit1d tried, when it tried more than one; False when there are none to show."""
        candidates = self._fit1d_candidates if native is not None else []
        many = len(candidates) > 1
        for widget in (self.fit1d_candidates_intro, self.fit1d_candidates_table, self.fit1d_candidate_details,
                       self.fit1d_use_candidate_button):
            widget.setVisible(many)
        if not many:
            return False
        table = self.fit1d_candidates_table
        table.setHorizontalHeaderLabels(["#", tr("composition"), "χ²/point", "ΔBIC", tr("note")])
        selected = table.currentRow()
        table.blockSignals(True)
        table.setRowCount(len(candidates))
        shown = native.get("summary", {}).get("composition")
        for line, candidate in enumerate(candidates):
            notes = [tr("in Results")] if candidate.get("composition") == shown else []
            if candidate.get("indistinguishable_from"):
                notes.append(trf(INDISTINGUISHABLE, others=", ".join(candidate["indistinguishable_from"])))
            notes.extend(tr(FLAG_TEXT.get(flag, flag)) for flag in candidate.get("flags") or ())
            tip = "\n".join([tr(message) for message in candidate.get("messages") or ()] + [" · ".join(notes)])
            cells = (str(candidate.get("rank")), candidate.get("composition") or "",
                     f"{candidate.get('chi2_per_point') or 0:.3g}", f"{candidate.get('delta_bic') or 0:.1f}",
                     " · ".join(notes))
            for column, text in enumerate(cells):
                table.setItem(line, column, _cell(text, tip))
        table.blockSignals(False)
        fit_to_rows(table)
        if not 0 <= selected < len(candidates):
            selected = next((i for i, c in enumerate(candidates) if c.get("composition") == shown), 0)
        table.selectRow(selected)
        self._fit1d_candidate_chosen()
        return True

    def _fit1d_candidate_chosen(self) -> None:
        line = self.fit1d_candidates_table.currentRow()
        candidates = self._fit1d_candidates
        details = self.fit1d_candidate_details
        details.setHorizontalHeaderLabels([tr("parameter"), tr("value ± error"), tr("equivalent range"), tr("state")])
        rows = candidates[line]["table"] if 0 <= line < len(candidates) else []
        details.setRowCount(len(rows))
        for index, row in enumerate(rows):
            name, value, _state = self.fit1d_row_cells(row)
            bounds = row.get("range")
            equivalent = f"{_number(bounds[0])}–{_number(bounds[1])} {row.get('unit') or ''}".strip() if bounds else "—"
            for column, text in enumerate((name, value, equivalent, _state)):
                details.setItem(index, column, _cell(text, self.fit1d_row_tip(row)))
        fit_to_rows(details)
        self.fit1d_use_candidate_button.setEnabled(0 <= line < len(candidates) and not self.fit_running())

    def use_fit1d_candidate(self) -> None:
        """Predict again with the selected composition: its values in Model as the start, its edges resolved."""
        line = self.fit1d_candidates_table.currentRow()
        if self.fit_running() or not 0 <= line < len(self._fit1d_candidates):
            return
        candidate = self._fit1d_candidates[line]
        try:
            model, _deviation = model_from_solution(candidate["mapping_row"], FitModel())
        except (ValueError, KeyError, TypeError) as exc:
            self._status(tr("The fit failed: {reason}").format(reason=str(exc)), "error")
            return
        self.set_model(model)
        self.fit1d_composition.setCurrentIndex(self.fit1d_composition.findData("model"))
        self.run_fit("fit1d")


__all__ = ["Fit1DReviewMixin", "note_text"]
