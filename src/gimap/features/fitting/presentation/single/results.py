"""The Results step: how good the fit is, every value with its error, the solutions of the other
methods, and saving (the points with the model and its terms plus a JSON record, the plot, the model;
the solutions: ``solutions_export.py``; the figure: ``figure_export.py``).

± of a value is its least-squares error after a fit; with a solution of Predict parameters in Model (no fit since),
half the width of the solution's 68 % posterior interval (``solution_texts.predict_plus``, said in its tooltip).
"""

from __future__ import annotations

import errno
import json
import math
import os
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QFileDialog, QHeaderView, QTableWidgetItem

from src.gimap.app.presentation.i18n import tr, trf

from src.gimap.shared import fit_names
from ...application.solution_export import posterior_by_path
from ...application.single_fit import (
    GRAZING,
    INFO,
    export_table,
    grazing,
    qz_plus_r,
    derived_values,
    fit_record,
    model_to_dict,
    parameter_info,
    parameter_text,
)
from ...application.fit_nxcansas import fit_curves, write_curves
from ..views.step_widgets import fit_to_rows, show_line
from .method_hint import note_text
from . import predict_texts
from .session import path_name
from .solution_texts import PLUS_HEADER_TIP, numbered_labels, predict_plus, refined_text
from ...application.models import InSituSourceFrame
from ...application.curve_sources import source_stem

QZ_PLUS_R = "q₊R of the spheres (q₊ = kᵢ + k_f): {values}"
PLOT_FILTER = "PNG image (*.png);;SVG vector (*.svg);;PDF (*.pdf)"
SAVE_DATA_FILTER = "CSV (*.csv);;NXcanSAS (*.h5)"
"""Save Data and Fit: the table with every term, or NXcanSAS for SasView and other NeXus programs."""
BY_NUMBER = "Sizes by number | by intensity: {values}"


def _item(text: str, tip: str = "") -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    if tip:
        item.setToolTip(tip)
    return item


def folder_action(path) -> tuple:
    """The toast's “Open Folder” after a file was written."""
    folder = str(Path(path).resolve().parent)
    return tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(folder))


def start_folder(*folders) -> Path:
    """Where a save dialog starts: the first of ``folders`` that is there, else the person's home folder —
    never a bare file name, which is the working folder of the program (for the portable version, the
    folder it is installed in)."""
    for folder in folders:
        if folder and Path(str(folder)).is_dir():
            return Path(str(folder))
    return Path.home()


def write_plot(item, path) -> None:
    """A pyqtgraph plot as PNG (1600 px wide) or SVG, on the light plot palette in either theme (a figure for
    paper and slides); ``OSError`` when the file could not be written."""
    from src.gimap.app.presentation.components.plot_export import export_plot

    export_plot(item, path)


def _part(path: Path) -> Path:
    """Where a file is written before it takes its name (``x.csv.part``)."""
    return path.with_name(path.name + ".part")


def _quietly(action, *paths) -> None:
    try:
        action(*paths)
    except OSError:
        pass


def _refuse_read_only(*paths: Path) -> None:
    """``PermissionError`` before anything is written when an earlier file of these names is read-only
    (renaming it would succeed on Windows, so a file the person protected would be replaced)."""
    for path in paths:
        if path.exists() and not os.access(path, os.W_OK):
            raise PermissionError(errno.EACCES, tr("The file is read-only"), str(path))


def _aside(path: Path) -> Path:
    """A free name to set an earlier file aside under, ``x.csv.old`` (or ``x.csv.1.old`` …): a stray one
    of an interrupted save is never overwritten — it may hold the only copy — and never blocks a save."""
    earlier = path.with_name(path.name + ".old")
    number = 0
    while earlier.exists() or earlier.is_symlink():
        number += 1
        earlier = path.with_name(f"{path.name}.{number}.old")
    return earlier


def write_text_whole(path, text: str) -> None:
    """``text`` into ``path`` under a temporary name first: a failed save leaves an earlier file as it was."""
    path = Path(path)
    _refuse_read_only(path)
    part = _part(path)
    try:
        part.write_text(text, encoding="utf-8")
        os.replace(part, path)
    except OSError:
        _quietly(os.remove, part)
        raise


def write_pair(path, write_table, record: dict) -> Path:
    """The table (``write_table(target)``) and its JSON record next to it: both or neither. Both are
    written under temporary names first and take their names only then; an earlier table of the same
    name is set aside until the new record is in place. A failed save so leaves an earlier pair as it
    was and no new file; a read-only earlier table or record is refused before anything is written.
    Returns the record's path; ``OSError`` on failure."""
    path = Path(path)
    record_path = path.with_suffix(".json")
    if record_path == path:  # a table named x.json: its record is x.record.json
        record_path = path.with_name(path.stem + ".record.json")
    _refuse_read_only(path, record_path)  # a protected table or record stays as it is
    table_part, record_part, earlier = _part(path), _part(record_path), _aside(path)
    text = json.dumps(record, indent=2, ensure_ascii=False)
    set_aside = placed = False
    try:
        write_table(table_part)
        record_part.write_text(text, encoding="utf-8")
        if path.exists():
            os.replace(path, earlier)  # fails while the table is open elsewhere: nothing has changed yet
            set_aside = True
        os.replace(table_part, path)
        placed = True
        os.replace(record_part, record_path)
    except OSError:
        if set_aside:
            _quietly(os.replace, earlier, path)  # the earlier table again
        elif placed:
            _quietly(os.remove, path)
        _quietly(os.remove, table_part)
        _quietly(os.remove, record_part)
        raise
    if set_aside:
        _quietly(os.remove, earlier)
    return record_path


def value_cells(key: str, value: float, error, free: bool) -> tuple[str, str]:
    """The value with its unit, and “± error” (no unit) or “fixed”: the unit stays in the value column."""
    unit = INFO[key].unit
    text = parameter_text(key, value, error)
    if unit and text.endswith(unit):
        text = text[: -len(unit)].strip()
    number, _sep, err = text.partition(" ± ")
    return f"{number} {unit}".strip(), f"± {err}" if err else ("" if free else tr("fixed"))


CORRELATED_SHOWN = 2
MORE_PAIRS = "and {count} more (hover)"
ANALYZE_CHI2 = ("χ²ᵣ of Analyze's fit (Predict parameters), over the points Analyze fitted. Fit (Refine) gives it for "
                "the points here.")


class FitResultsMixin:
    """Needs the page's widgets, ``session``, ``_status``, ``_log`` and ``_data``."""

    def setup_results(self) -> None:
        self.parameters_table.setHorizontalHeaderLabels(["", tr("value"), "±", tr("posterior median")])
        self.parameters_table.horizontalHeaderItem(2).setToolTip(tr(PLUS_HEADER_TIP))
        self.solutions_table.setHorizontalHeaderLabels(["#", tr("model"), "R (nm)", "D (nm)", "χ²", "p"])
        self.use_solution_button.clicked.connect(self.use_selected_solution)
        self.solutions_table.cellDoubleClicked.connect(lambda _row, _column: self.use_selected_solution())
        self.solutions_table.itemSelectionChanged.connect(
            lambda: self.use_solution_button.setEnabled(self.solutions_table.currentRow() >= 0))
        for action, button, slot in (
            (self.export_data_action, self.export_data_step_button, self.export_data_dialog),
            (self.export_plot_action, self.export_plot_step_button, self.export_plot_dialog),
            (self.save_model_action, self.save_model_button, self.save_model_dialog),
        ):
            action.triggered.connect(slot)
            button.clicked.connect(slot)

    # -- render -----------------------------------------------------------------------------

    @staticmethod
    def _state_text(result) -> str:
        if getattr(result, "start_kept", False):  # nothing tried was better than the start: not a minimum found
            return tr("did not improve on the start values")
        return tr("converged") if result.converged else tr("stopped") if result.stopped else tr("not converged")

    @staticmethod
    def _chi_name(result) -> str:
        return "χ²ᵣ" if result.weighting == "sigma" else "var ln I"

    @classmethod
    def _quality_text(cls, result) -> str:
        """The whole of it, said once: in the Results box (the rail has the points and free values on their rows)."""
        return tr("{chi} {value} · {points} points · {free} free · {state}").format(
            chi=cls._chi_name(result), value=f"{result.chi2_reduced:.3g}", points=result.points, free=result.free,
            state=cls._state_text(result))

    @classmethod
    def _rail_quality(cls, result) -> str:
        """The Fit row of the rail: “χ²ᵣ 1.39 · converged”."""
        return trf("{chi} {value} · {state}", chi=cls._chi_name(result), value=f"{result.chi2_reduced:.3g}",
                   state=cls._state_text(result))

    def _rail_points(self, count: int) -> str:
        """The Curve row of the rail: the points in the fit; after Predict parameters, how many of them it fitted
        (it leaves out those below the network's q window)."""
        session = self.session
        best = session.solutions[0] if session.solutions and session.result is None \
            and session.solutions_are_current() else None
        if best is not None and best.points and best.points[0] != best.points[1]:
            return trf("{fitted} of {given} points fitted", fitted=best.points[0], given=best.points[1])
        return tr("{points} points fitted").format(points=count)

    def render_results(self) -> None:
        self.compare_saved_fit_button.setVisible(self.session.saved_fit is not None)
        self._render_model_results()
        self.render_result_actions()  # next steps, what the fit did by itself, the folds (result_actions.py)
        notes = self.warnings_label.text()
        self.fit_notes_button.setVisible(bool(notes or not self.actions_label.isHidden()))
        self.fit_notes_button.setText(tr("Review fit warnings…") if "⚠" in notes else tr("Fit details…"))
        self.refresh_equivalent_selector()
        self.refresh_uncertainty()  # the Uncertainty and Report window (uncertainty.py)

    def _render_model_results(self) -> None:
        header = self.parameters_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # a long name is cut short (its tooltip has it whole)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.parameters_table.setHorizontalHeaderLabels(["", tr("value"), "±", tr("posterior median")])
        self.parameters_table.horizontalHeaderItem(2).setToolTip(tr(PLUS_HEADER_TIP))
        session = self.session
        result = session.result
        old_search = session.solutions_searched() and not session.solutions_are_current()
        search_note = ["⚠ " + tr("The halves, the fitting range or the left-out points changed after the search: "
                                 "the χ² of the solutions is of the points before.")] if old_search else []
        if result is None and session.solutions:
            self.quality_label.setText(trf("{solutions} below. Refine reports fit quality and uncertainties where estimable.",
                                           solutions=predict_texts.count_solutions(len(session.solutions)))
                                       + self._incomplete_note())
            current = next((s for s in session.solutions if s.model == session.model), None)
            if current and any(flag.split(":")[0] in fit_names.GRAZING_FLAGS for flag in current.flags):
                self.quality_label.setText(tr("Needs attention") + " · " + current.label)
            self.warnings_label.setText("\n".join(self.posterior_lines() + search_note))
            self.warnings_label.setToolTip("")
            stale = tr("Changed after the fit: Fit again") + " · " if old_search else ""
            show_line(self.step_intro["results"], stale + trf(
                "{solutions} to compare.", solutions=predict_texts.count_solutions(len(session.solutions))))
        elif result is None:
            saved = session.saved_fit
            if saved is None:
                self.quality_label.setText(tr("No fit yet."))
            else:
                text = trf("Saved model from {name} · {count} points", name=saved.path.name, count=saved.count)
                if saved.quality is not None:
                    metric = "χ²ᵣ" if saved.weighting == "sigma" else "var ln I"
                    text += f" · {metric} {saved.quality:.4g}"
                self.quality_label.setText(text)
            self.warnings_label.setText("")
            self.warnings_label.setToolTip("")
            show_line(self.step_intro["results"], "")  # the box below and the rail's “After a fit” say it
        else:
            origin = refined_text(session.refined_from, session.solutions) if session.result_is_current() else ""
            self.quality_label.setText(tr(
                "{quality}\nlog RMSE {rmse} · {evaluations} evaluations · {seconds} s · {method}"
            ).format(quality=self._quality_text(result), rmse=f"{result.log_rmse:.3g}", evaluations=result.evaluations,
                     seconds=f"{result.seconds:.1f}", method=tr("Search the ranges") if result.method == "global" else tr("Refine"))
                + (f"\n{origin}" if origin else ""))
            self._correlated_tip = ""
            self.warnings_label.setText("\n".join(self._size_lines() + self._grazing_lines() + self._warnings(result)
                                                   + search_note))
            self.warnings_label.setToolTip(self._correlated_tip)
            show_line(self.step_intro["results"], "")  # the box right below has it all (and a stale fit: its warnings)
        self.quality_label.setToolTip("" if result is None and not session.solutions
                                      else predict_texts.chi2_definition())
        self._fill_parameters(result if session.result_is_current() else None)
        self._fill_solutions()

    def _warnings(self, result) -> list[str]:
        lines = []
        if not self.session.result_model_is_current():
            lines.append("⚠ " + tr("The model was changed after this fit: the errors belong to the fitted values."))
        if not self.session.result_selection_is_current():
            lines.append("⚠ " + tr("The fitting range or the left-out points changed after this fit: "
                                   "Fit again for its quality and errors."))
        notes = self._fit_notes if self._fit_notes and self._fit_notes.get("result") is result else None
        if notes is not None:  # what the fit could not resolve by itself
            lines.extend("⚠ " + note_text(note.to_dict(), self._path_name) for note in notes["remaining"])
        else:
            for path in result.at_bounds:
                lines.append("⚠ " + tr("{name} stopped at a bound of its range: widen the range or fix it.").format(
                    name=self._path_name(path)))
        lines.extend(self.flag_lines(result))  # values the curve does not determine, a beam tail at an edge
        if result.correlated:
            pairs = [f"{self._path_name(first)} ↔ {self._path_name(second)} ({rho:+.2f})"
                     for first, second, rho in result.correlated]
            shown = "; ".join(pairs[:CORRELATED_SHOWN])
            if len(pairs) > CORRELATED_SHOWN:  # the rest in the tooltip: a long list hides everything else
                shown += " " + trf(MORE_PAIRS, count=len(pairs) - CORRELATED_SHOWN)
            lines.append("⚠ " + tr("Strongly correlated, so the data do not separate them (fix one, or read their "
                                   "errors as a range): {pairs}.").format(pairs=shown))
            self._correlated_tip = "\n".join(pairs)
        if result.weighting == "sigma" and result.chi2_reduced > 3 and not result.stopped:
            lines.append("⚠ " + tr("χ²ᵣ well above 1: the model misses features of the curve, or σ is underestimated."))
        other_points = predict_texts.refine_points_line(result, self.session.solutions)
        if other_points:  # Predict left out the points below the network's q window: another χ²ᵣ for the same values
            lines.append(other_points)
        return lines

    def _unused(self, path: tuple, parameter) -> bool:
        if path[0] != "globals" or parameter.free:
            return False
        return parameter.value == 0 or (path[1] in GRAZING and grazing(self.session.model) is None)

    def _size_lines(self) -> list[str]:
        """Each particle's mean radius and relative spread by number (as a microscope counts) and by intensity (V²,
        what dominates the curve), as information (Physics and Methods, section 9.3)."""
        parts = []
        for index, component in enumerate(self.session.model.components, start=1):
            values = derived_values(component)
            if values["R_number_mean_nm"] is None:
                text = f"{index}: R " + tr("not defined (the distribution rises towards small R)")
            else:
                text = f"{index}: R {values['R_number_mean_nm']:.3g} nm, σR/R {values['sigma_R_number_rel']:.2g}"
            text += f" | R {values['R_intensity_mean_nm']:.3g} nm, σR/R {values['sigma_R_intensity_rel']:.2g}"
            parts.append(text)
        return [trf(BY_NUMBER, values="; ".join(parts))] if parts else []

    def _grazing_lines(self) -> list[str]:
        """q₊R of the spheres, as information (the range of each sphere model is in the Physics Manual)."""
        values = qz_plus_r(self.session.model)
        if not values:
            return []
        shown = ", ".join(f"{index + 1}: {value:.2f}" for index, value in values)
        return [trf(QZ_PLUS_R, values=shown)]

    def _path_name(self, path: tuple) -> str:
        return path_name(self.session.model, path)

    def _incomplete_note(self) -> str:
        if not any(getattr(solution, "incomplete", False) for solution in self.session.solutions):
            return ""
        if any("bic" in s.flags for s in self.session.solutions):
            return "\n⚠ " + tr("Incomplete. BIC weights refer to the finished solutions.")
        return "\n⚠ " + tr("Incomplete: Predict was stopped; these are the solutions it finished, their probabilities "
                           "over those alone.")

    def _fill_parameters(self, result) -> None:
        rows = []
        predicted = None if result is not None else next(  # a solution of Predict in Model: ± from its interval
            (s for s in self.session.solutions if s.model == self.session.model and s.intervals), None)
        posterior = posterior_by_path(predicted) if predicted else {}
        self.parameters_table.setColumnHidden(3, predicted is None)
        self.parameters_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.parameters_table.horizontalHeaderItem(1).setText(tr("least-squares\nvalue") if predicted else tr("value"))
        self.parameters_table.horizontalHeaderItem(3).setToolTip(tr("Posterior median. Hover over a value for its central 68 % interval."))
        for path, parameter in self.session.model.parameters():
            if self._unused(path, parameter):
                continue  # an unused term (no resolution peak, no background, spheres at q_z = 0)
            error = None if result is None else result.errors.get(path)
            full = path_name(self.session.model, path, full=True)
            component = None if path[0] == "globals" else self.session.model.components[path[0]]
            full += "\n" + tr(parameter_info(component, path[1]).tip)
            value, plus = value_cells(path[1], parameter.value, error, parameter.free)
            why = ""
            if predicted is not None and parameter.free:
                plus, why = predict_plus(predicted, path)
            if result is not None and parameter.free and not plus:  # a free value the fit gives no error for
                plus, why = "–", predict_texts.no_error_reason(path, parameter, result, self.session.model)
            plus, why = self.flag_cell(path, result, plus, why)
            rows.append(((self._path_name(path), full), value, (plus, why), posterior.get(path)))
        self.parameters_table.setRowCount(len(rows))
        self._row_paths = [path for path, parameter in self.session.model.parameters()
                           if not self._unused(path, parameter)]
        for line, ((name, full), value, (plus, why), sampled) in enumerate(rows):
            self.parameters_table.setItem(line, 0, _item(name, full))
            self.parameters_table.setItem(line, 1, _item(value))
            self.parameters_table.setItem(line, 2, _item(plus, why))
            median = f"{sampled['median']:.5g}" if sampled else "—"
            interval = (fit_names.interval_text(sampled["median"], sampled["low68"], sampled["high68"])
                        if sampled and "low68" in sampled else "")
            self.parameters_table.setItem(line, 3, _item(median, interval))
        fit_to_rows(self.parameters_table)  # every row shown: only the step's page scrolls

    def _fill_solutions(self) -> None:
        solutions = self.session.solutions
        self.solutions_title.setVisible(bool(solutions))
        self.solutions_table.setVisible(bool(solutions))
        self.use_solution_button.setVisible(bool(solutions))
        curve = self.session.curve
        searched = self.session.solutions_searched()  # its χ²ᵣ computed here (run.py); else Analyze's fit
        predicted = any(s.points or s.probability is not None for s in solutions) or not searched
        chi = "χ²ᵣ" if predicted or curve is None or curve.sigma is not None else "var ln I"
        bic = any("bic" in s.flags for s in solutions)
        weight = tr(fit_names.BIC_WEIGHT) if bic else "p"
        self.solutions_table.setHorizontalHeaderLabels(["#", tr("model"), "R (nm)", "D (nm)", chi, weight])
        for column, tip in ((2, tr(predict_texts.SIZES_TIP)), (3, tr(predict_texts.SIZES_TIP)),
                            (4, predict_texts.chi2_definition() if chi == "χ²ᵣ" else ""), (5, weight if bic else tr(predict_texts.P_TIP))):
            self.solutions_table.horizontalHeaderItem(column).setToolTip(tip)
        self.solutions_table.setColumnHidden(5, not any(s.probability is not None for s in solutions))
        shown = tuple(solutions)
        fresh = shown != getattr(self, "_solutions_shown", None)
        self._solutions_shown = shown
        self.solutions_table.setRowCount(len(solutions))
        labels = numbered_labels(solutions)
        for line, solution in enumerate(solutions):
            radii, distances = predict_texts.sizes_cells(solution.model)  # every population, by radius
            tip = tr("From {source}.").format(source=tr(solution.source))
            for extra in (predict_texts.points_text(solution.points), predict_texts.differing_tip(solution, solutions)):
                if extra:
                    tip += " " + (extra if extra.endswith(".") else trf(predict_texts.POINTS, points=extra))
            if not searched:
                tip += " " + tr(ANALYZE_CHI2)
            if solution.deviation > 0.01:
                tip += " " + tr("This model differs from the solution by up to {percent} %.").format(
                    percent=f"{100 * solution.deviation:.3g}")
            alone = solution.tried == 1 and len(solutions) == 1  # one composition tried: p is no finding
            if alone and not bic:
                tip += " " + tr(predict_texts.ONLY_ONE)
            if getattr(solution, "incomplete", False):
                tip += " " + (tr("Incomplete. BIC weights refer to the finished solutions.") if bic else
                              tr("Incomplete: Predict was stopped; its probability is over the solutions finished."))
            cells = (str(line + 1), labels[line], radii, distances,
                     f"{solution.chi2:.3g}" if math.isfinite(solution.chi2) else "—",
                     "—" if solution.probability is None or (alone and not bic) else f"{solution.probability:.2f}")
            for column, text in enumerate(cells):
                self.solutions_table.setItem(line, column, _item(text, tip))
        if fresh or self.solutions_table.currentRow() < 0:  # new solutions: the one in Model is selected
            line = next((line for line, solution in enumerate(solutions) if solution.model == self.session.model), -1)
            if line >= 0:
                self.solutions_table.selectRow(line)
        self.use_solution_button.setEnabled(0 <= self.solutions_table.currentRow() < len(solutions))

    def use_selected_solution(self) -> None:
        line = self.solutions_table.currentRow()
        if not 0 <= line < len(self.session.solutions):
            return
        solution, solutions = self.session.solutions[line], list(self.session.solutions)
        self.session.result = None
        self.session.refined_from = None
        self.set_model(solution.model, message=lambda: tr(  # named in the language of when it is said
            "{model} is in Model (Undo brings the previous one back).").format(model=numbered_labels(solutions)[line]),
            label=("solution {number} in Model", {"number": line + 1}))
        self.session.model_from = line

    # -- save -------------------------------------------------------------------------------

    def _save_path(self, title: str, name: str, filters: str):
        """Asks where to save, starting in the folder of the last save, else of the last curve opened, else
        of this curve, else the home folder (Save Plot and Save Model also work with no curve)."""
        curve = self.session.curve
        folder = start_folder(self._remembered("save_folder"), self._remembered("folder"),
                              InSituSourceFrame.from_token(curve.path).path.parent
                              if curve is not None and curve.path else None)
        path, _ = QFileDialog.getSaveFileName(self, tr(title), str(folder / name), filters)
        if path:
            self._remember(save_folder=str(Path(path).parent))
        return path

    def _stem(self) -> str:
        curve = self.session.curve
        return (source_stem(curve.path) if curve.path else Path(curve.name).stem) if curve is not None else "fit"

    def export_data_dialog(self, path=None):
        data = self._data()
        if data is None or not data.q.size:
            self._status(tr("Open a curve first."), "warning")
            return None
        path = path or self._save_path("Save Curve and Parameters…", f"{self._stem()}_fit.csv", SAVE_DATA_FILTER)
        if not path:
            return None
        names, table = export_table(self.session.model, data)
        record = fit_record(self.session.model, data, self.session.curve, side=self.session.side,
                            q_range=self.session.q_range,
                            result=self.session.result if self.session.result_is_current() else None,
                            excluded=self.session.excluded, curve_table=(names, table))
        record.update(self.solutions_record(record))
        saved = self.session.saved_fit
        if saved is not None:
            record["parent_fit_record"] = {"path": str(saved.path), "sha256": saved.sha256}
        if Path(path).suffix.lower() == ".h5":  # NXcanSAS: the points fitted and the model, q in 1/nm
            curves = fit_curves(self.session.model, data, record, self._stem())
            write = lambda target: write_curves(target, curves)  # noqa: E731
        else:
            write = lambda target: np.savetxt(  # noqa: E731
                target, table, delimiter=",", header=",".join(names), comments="", fmt="%.17g")
        try:
            record_path = write_pair(path, write, record)
        except OSError as exc:
            return self._save_failed(exc)
        self._status(lambda: tr("Saved {name} and its record {record}.").format(name=Path(path).name, record=record_path.name),
                     "ok", action=folder_action(path))
        return path

    def export_plot_dialog(self, path=None):
        """The plot with its residuals as one figure (``figure_export.py``): PNG, SVG or PDF."""
        path = path or self._save_path("Save Plot", f"{self._stem()}_fit.png", PLOT_FILTER)
        if not path:
            return None
        return self.write_figure(path)

    def save_model_dialog(self, path=None):
        path = path or self._save_path("Save Model", f"{self._stem()}_model.json", "Fit model (*.json)")
        if not path:
            return None
        try:
            write_text_whole(path, json.dumps(model_to_dict(self.session.model), indent=2))
        except OSError as exc:
            return self._save_failed(exc)
        self._status(lambda: tr("Saved {name}.").format(name=Path(path).name), "ok", action=folder_action(path))
        return path

    def _save_failed(self, exc: OSError) -> None:
        self._status(lambda: tr("Could not save: {error}").format(error=exc), "error")
        return None


__all__ = ["FitResultsMixin", "start_folder"]
