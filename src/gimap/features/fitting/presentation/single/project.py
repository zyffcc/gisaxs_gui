"""Single analysis in a project (File ▸ Save Project / Open Project): the curve, its halves, range and left-out points,
the model, the method — and the results: the last fit (values, errors, χ²ᵣ, bound flags, correlations, the points it
was made on), every solution of Predict parameters (probability, intervals, flags), where a fit started from and the
run of Predict. Opened again, the page shows them as they were (Results, the selector, Fit details).

The results are kept with the SHA-256 of the curve's file. When the file changed since, they are shown all the same,
as of other points (stale: Fit again), and the project's notes say so.
"""

from __future__ import annotations

from src.gimap.app.presentation.i18n import tr

from ...application.fit_state import (
    STATE_SCHEMA,
    number,
    read_number,
    result_from_dict,
    result_to_dict,
    selection_from_dict,
    selection_to_dict,
)
from ...application.single_fit import model_from_dict, model_to_dict
from .curve_unit import SETTING_KEY
from .session import Solution

CHANGED_FILE = "the Fitting curve {name} changed since the project was saved: its fit results are shown as stale"
NOT_READ = "the Fitting results could not be read: {reason}"
STALE = ("changed", None, frozenset())
"""The points of results whose curve file changed: none that is chosen now (``FitSession.selection``)."""


def solution_to_dict(solution: Solution) -> dict:
    key = "bic_weight" if "bic" in solution.flags else "probability"
    return {"model": model_to_dict(solution.model), "chi2": number(solution.chi2), "source": solution.source,
            "deviation": number(solution.deviation),
            key: number(solution.probability),
            "intervals": [[name, [number(value) if value is not None else None for value in values]]
                          for name, values in solution.intervals],
            "flags": list(solution.flags), "points": list(solution.points), "tried": solution.tried,
            "composition": solution.composition, "ess": number(solution.ess) if solution.ess is not None else None,
            "diagnostics": solution.diagnostics, "posterior_summary": solution.posterior_summary,
            "incomplete": bool(solution.incomplete)}


def solution_from_dict(data: dict) -> Solution:
    weight = data.get("bic_weight", data.get("probability"))
    return Solution(model_from_dict(data["model"]), read_number(data.get("chi2")), str(data.get("source") or ""),
                    float(data.get("deviation") or 0.0),
                    probability=None if weight is None else float(weight), diagnostics=data.get("diagnostics") or {},
                    posterior_summary=data.get("posterior_summary") or {},
                    intervals=tuple((str(name), tuple(values)) for name, values in data.get("intervals") or ()),
                    flags=tuple(data.get("flags") or ()), points=tuple(int(v) for v in data.get("points") or ()),
                    tried=data.get("tried"), composition=str(data.get("composition") or ""), ess=data.get("ess"),
                    incomplete=bool(data.get("incomplete")))


def _sha256(curve) -> str | None:
    from ...application.curve_sources import source_record

    try:
        return source_record(curve.path).get("sha256") if curve is not None and curve.path else None
    except (OSError, ValueError):
        return None


class FitProjectMixin:
    """Needs the page's ``session``, ``open_curve``, ``set_range``, ``set_method``, ``method``, ``unit_setting``,
    ``set_unit_setting``, ``_saved_unit_note``, ``_model_changed``, ``_render_all``, ``status_label``."""

    def project_state(self) -> dict:
        session, curve = self.session, self.session.curve
        return {
            "curve": None if curve is None or not curve.path else {
                "path": curve.path, "unit": curve.source_unit, "side": session.side,
                "q_range": None if session.q_range is None else list(session.q_range),
                "excluded": sorted(session.excluded)},
            "model": model_to_dict(session.model), "method": self.method(), SETTING_KEY: self.unit_setting(),
            "results": self.results_state(),
        }

    def results_state(self) -> dict | None:
        """The fit and the solutions of the curve (module docstring); ``None`` when there are none."""
        session = self.session
        if session.curve is None or (session.result is None and not session.solutions):
            return None
        return {"schema": STATE_SCHEMA, "curve_sha256": _sha256(session.curve),
                "result": result_to_dict(session.result), "result_selection": selection_to_dict(session.result_selection),
                "solutions": [solution_to_dict(solution) for solution in session.solutions],
                "solutions_selection": selection_to_dict(session.solutions_selection),
                "model_from": session.model_from, "refined_from": session.refined_from,
                "predict_run": session.predict_run}

    def apply_results_state(self, data) -> list[str]:
        """The results of a project on the curve just opened (``project_state``); notes for the project's report."""
        session = self.session
        if not isinstance(data, dict) or session.curve is None:
            return []
        try:
            result = result_from_dict(data.get("result"))
            solutions = [solution_from_dict(item) for item in data.get("solutions") or ()]
            result_selection = selection_from_dict(data.get("result_selection"))
            solutions_selection = selection_from_dict(data.get("solutions_selection"))
        except (ValueError, KeyError, TypeError) as exc:
            return [tr(NOT_READ).format(reason=exc)]
        notes = []
        saved, now = data.get("curve_sha256"), _sha256(session.curve)
        if saved and now and saved != now:  # another file under the name: the results are of other points
            result_selection = STALE if result_selection is not None else None
            solutions_selection = STALE if solutions_selection is not None else None
            notes.append(tr(CHANGED_FILE).format(name=session.curve.name))
        session.result, session.result_selection = result, result_selection
        session.solutions, session.solutions_selection = solutions, solutions_selection
        model_from = data.get("model_from")
        session.model_from = model_from if isinstance(model_from, int) and 0 <= model_from < len(solutions) else None
        session.refined_from = data.get("refined_from") if isinstance(data.get("refined_from"), dict) else None
        session.predict_run = data.get("predict_run") if isinstance(data.get("predict_run"), dict) else None
        return notes

    def apply_project_state(self, data: dict) -> list[str]:
        notes = []
        self._gisaxs_cut = False
        if isinstance(data.get("model"), dict):
            try:
                self.session.model = model_from_dict(data["model"])
                self.session.edited = True
            except (ValueError, KeyError, TypeError) as exc:
                notes.append(tr("the Fitting model could not be read: {reason}").format(reason=exc))
        self.set_method(data.get("method"))
        self.set_unit_setting(data.get(SETTING_KEY))  # (an older project has none: its curve's unit is used)
        curve = data.get("curve") or {}
        saved = _saved_source(curve.get("path")) if curve.get("path") else None
        if curve.get("path") and (saved is None or saved.path.is_file()):
            if self.open_curve(curve["path"], curve.get("side"), unit=curve.get("unit")):
                notes += [note() for note in [self._saved_unit_note(curve.get("unit"))] if note]
                self.session.excluded = {float(value) for value in curve.get("excluded") or ()}
                if curve.get("q_range"):
                    self.set_range(curve["q_range"], record=False)
                notes += self.apply_results_state(data.get("results"))
            else:
                notes.append(self.status_label.text())
        elif curve.get("path"):
            notes.append(tr("the Fitting curve {name} is no longer there").format(
                name=saved.display_name))
        if data.get("method") is not None:
            self.set_method(data["method"])  # a saved choice takes precedence over a new-curve recommendation
        self.session.clear_history()  # a new start: Undo does not bring back the work before the project
        self._model_changed(editor=True)
        self._render_all()
        return notes


def _saved_source(token):
    from .page import _saved_source as saved_source

    return saved_source(token)


__all__ = ["CHANGED_FILE", "FitProjectMixin", "NOT_READ", "solution_from_dict", "solution_to_dict"]
