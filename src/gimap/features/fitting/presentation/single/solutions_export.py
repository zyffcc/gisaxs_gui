"""Saving the solutions of Predict parameters (Save ▸ Solutions…, Results ▸ Export Solutions…) and adding them to the
record of Save ▸ Data and Fit (``solutions_record``).

Save ▸ Solutions… writes three files next to each other (``application/solution_export.py`` makes their contents):

* ``<curve>_solutions.csv`` — one row per solution and parameter: the composition, probability, χ²ᵣ and its points,
  the flags, whether it is incomplete, and the parameter's value, median, 68 % and 95 % intervals and error;
* ``<curve>_solutions.json`` — the record of Save ▸ Data and Fit with every solution (its model, values and
  intervals) under ``predict``;
* ``<curve>_solutions_curves.csv`` — the points fitted and every solution's model, residual and terms.

The table and the record are written as a pair (both or neither, ``results.write_pair``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.gimap.app.presentation.i18n import tr

from ...application.single_fit import fit_record
from ...application.fit_reproduction import with_prediction
from ...application.solution_export import solutions_block, solutions_curves, solutions_table, table_text

SAVE_SOLUTIONS = "Save Solutions"
NO_SOLUTIONS = "No solutions to save: run Predict parameters first."
SAVED = "Saved {name}, its record {record} and the curves {curves}."
CURVES_SUFFIX = "_curves"


class SolutionsExportMixin:
    """Needs the page's ``session``, ``_data``, ``_save_path``, ``_stem``, ``_status``, ``_save_failed``."""

    def setup_solutions_export(self) -> None:
        for widget in (self.export_solutions_action, self.export_solutions_button):
            (widget.triggered if hasattr(widget, "triggered") else widget.clicked).connect(
                lambda *_args: self.export_solutions_dialog())

    def solutions_record(self, record=None, *, exporting_solutions=False) -> dict:
        """``{"predict": …}``: every solution with its intervals and flags; ``{}`` without solutions."""
        session = self.session
        if not session.solutions:
            return {}
        refined = session.refined_from if session.result is not None and session.result_is_current() else None
        run = session.predict_run
        block = solutions_block(session.solutions, session.model, refined_from=refined,
                                run=None if run is None else {k: v for k, v in run.items() if k != "reproduction"})
        if record is not None:
            selected = block["in_model"] if session.solutions_searched() else None
            if session.solutions_searched() and not (run or {}).get("reproduction"):
                # Old projects have the posterior values but lack the inputs to its sampler.
                block["reproduction_status"] = "The original Predict inputs were not recorded."
                if exporting_solutions or record.get("reproduction", {}).get("operation") != "manual":
                    record.pop("reproduction", None)
            else:
                with_prediction(record, run, None if selected is None else selected - 1)
        block["points_changed_since"] = session.solutions_searched() and not session.solutions_are_current()
        return {"predict": block}

    def refresh_solutions_export(self) -> None:
        there = bool(self.session.solutions)
        self.export_solutions_action.setEnabled(there)
        self.export_solutions_button.setVisible(there)

    def export_solutions_dialog(self, path=None):
        from .results import folder_action, write_pair, write_text_whole

        session, data = self.session, self._data()
        if not session.solutions or data is None or not data.q.size:
            self._status(tr(NO_SOLUTIONS), "warning")
            return None
        path = path or self._save_path(SAVE_SOLUTIONS, f"{self._stem()}_solutions.csv", "CSV (*.csv)")
        if not path:
            return None
        path = Path(path)
        curves_path = path.with_name(path.stem + CURVES_SUFFIX + ".csv")
        names, rows = solutions_table(session.solutions, session.model)
        curve_names, columns = solutions_curves(session.solutions, data)
        record = fit_record(session.model, data, session.curve, side=session.side, q_range=session.q_range,
                            result=session.result if session.result_is_current() else None, excluded=session.excluded)
        record.update(self.solutions_record(record, exporting_solutions=True))
        record["files"] = {"table": path.name, "curves": curves_path.name}
        try:
            write_text_whole(curves_path, table_text(curve_names, np.where(np.isfinite(columns), columns, np.nan).tolist()))
            record_path = write_pair(path, lambda target: Path(target).write_text(table_text(names, rows),
                                                                                  encoding="utf-8"), record)
        except OSError as exc:
            return self._save_failed(exc)
        self._status(lambda: tr(SAVED).format(name=path.name, record=record_path.name, curves=curves_path.name),
                     "ok", action=folder_action(path))
        return str(path)


__all__ = ["CURVES_SUFFIX", "NO_SOLUTIONS", "SAVED", "SAVE_SOLUTIONS", "SolutionsExportMixin"]
