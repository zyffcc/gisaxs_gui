"""The Results step: how good the fit is, every value with its error, the solutions of the other
methods, and saving (the points with the model and its terms plus a JSON record, the plot, the model).
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
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem

from src.gimap.app.presentation.i18n import tr

from ...application.single_fit import (
    INFO,
    export_table,
    fit_record,
    model_to_dict,
    parameter_text,
)
from ..views.fit_steps_view import fit_to_rows
from .session import path_name


def _item(text: str, tip: str = "") -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    if tip:
        item.setToolTip(tip)
    return item


def folder_action(path) -> tuple:
    """The toast's “Open Folder” after a file was written."""
    folder = str(Path(path).resolve().parent)
    return tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(folder))


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


class FitResultsMixin:
    """Needs the page's widgets, ``session``, ``_status``, ``_log`` and ``_data``."""

    def setup_results(self) -> None:
        self.parameters_table.setHorizontalHeaderLabels(["", tr("value"), "±"])
        self.solutions_table.setHorizontalHeaderLabels(["#", tr("model"), "R (nm)", "D (nm)", "χ²"])
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
    def _quality_text(result) -> str:
        chi = "χ²ᵣ" if result.weighting == "sigma" else "var ln I"
        state = tr("converged") if result.converged else tr("stopped") if result.stopped else tr("not converged")
        return tr("{chi} {value} · {points} points · {free} free · {state}").format(
            chi=chi, value=f"{result.chi2_reduced:.3g}", points=result.points, free=result.free, state=state)

    def render_results(self) -> None:
        session = self.session
        result = session.result
        old_search = session.solutions_searched() and not session.solutions_are_current()
        search_note = ["⚠ " + tr("The halves, the fitting range or the left-out points changed after the search: "
                                 "the χ² of the solutions is of the points before.")] if old_search else []
        if result is None and session.solutions:
            self.quality_label.setText(tr("{count} solutions below; the best is in Model. Fit (Refine) gives its quality "
                                          "and the error of every value.").format(count=len(session.solutions)))
            self.warnings_label.setText("\n".join(search_note))
            stale = tr("Changed after the fit: Fit again") + " · " if old_search else ""
            self.step_intro["results"].setText(
                stale + tr("{count} solutions to compare.").format(count=len(session.solutions)))
        elif result is None:
            self.quality_label.setText(tr("No fit yet: Fit shows here how good it is and the error of every value."))
            self.warnings_label.setText("")
            self.step_intro["results"].setText(tr("After a fit: its quality, the errors and the solutions to compare."))
        else:
            self.quality_label.setText(tr(
                "{quality}\nlog RMSE {rmse} · {evaluations} evaluations · {seconds} s · {method}"
            ).format(quality=self._quality_text(result), rmse=f"{result.log_rmse:.3g}", evaluations=result.evaluations,
                     seconds=f"{result.seconds:.1f}", method=tr("Search the ranges") if result.method == "global" else tr("Refine")))
            self.warnings_label.setText("\n".join(self._warnings(result) + search_note))
            stale = "" if session.result_is_current() else tr("Changed after the fit: Fit again") + " · "
            self.step_intro["results"].setText(stale + self._quality_text(result) + ".")
        self._fill_parameters(result if session.result_is_current() else None)
        self._fill_solutions()

    def _warnings(self, result) -> list[str]:
        lines = []
        if not self.session.result_model_is_current():
            lines.append("⚠ " + tr("The model was changed after this fit: the errors belong to the fitted values."))
        if not self.session.result_selection_is_current():
            lines.append("⚠ " + tr("The fitting range or the left-out points changed after this fit: "
                                   "Fit again for its quality and errors."))
        for path in result.at_bounds:
            lines.append("⚠ " + tr("{name} stopped at a bound of its range: widen the range or fix it.").format(
                name=self._path_name(path)))
        if result.correlated:
            pairs = "; ".join(f"{self._path_name(first)} ↔ {self._path_name(second)} ({rho:+.2f})"
                              for first, second, rho in result.correlated)
            lines.append("⚠ " + tr("Strongly correlated, so the data do not separate them (fix one, or read their "
                                   "errors as a range): {pairs}.").format(pairs=pairs))
        if result.weighting == "sigma" and result.chi2_reduced > 3 and not result.stopped:
            lines.append("⚠ " + tr("χ²ᵣ well above 1: the model misses features of the curve, or σ is underestimated."))
        return lines

    def _path_name(self, path: tuple) -> str:
        return path_name(self.session.model, path)

    def _fill_parameters(self, result) -> None:
        rows = []
        for path, parameter in self.session.model.parameters():
            if path[0] == "globals" and not parameter.free and parameter.value == 0:
                continue  # an unused term (no resolution peak, no background)
            error = None if result is None else result.errors.get(path)
            full = path_name(self.session.model, path, full=True)  # “1·Sphere R”, also when shown as “R”
            rows.append(((self._path_name(path), full), *value_cells(path[1], parameter.value, error, parameter.free)))
        self.parameters_table.setRowCount(len(rows))
        for line, ((name, full), *cells) in enumerate(rows):
            self.parameters_table.setItem(line, 0, _item(name, full))
            for column, text in enumerate(cells, start=1):
                self.parameters_table.setItem(line, column, _item(text))
        fit_to_rows(self.parameters_table)  # every row shown: only the step's page scrolls

    def _fill_solutions(self) -> None:
        solutions = self.session.solutions
        self.solutions_title.setVisible(bool(solutions))
        self.solutions_table.setVisible(bool(solutions))
        self.use_solution_button.setVisible(bool(solutions))
        curve = self.session.curve
        chi = "var ln I" if curve is not None and curve.sigma is None else "χ²ᵣ"
        self.solutions_table.setHorizontalHeaderLabels(["#", tr("model"), "R (nm)", "D (nm)", chi])
        shown = tuple(solutions)
        fresh = shown != getattr(self, "_solutions_shown", None)
        self._solutions_shown = shown
        self.solutions_table.setRowCount(len(solutions))
        for line, solution in enumerate(solutions):
            component = solution.model.components[0] if solution.model.components else None
            tip = tr("From {source}.").format(source=tr(solution.source))
            if solution.deviation > 0.01:
                tip += " " + tr("This model differs from the solution by up to {percent} %.").format(
                    percent=f"{100 * solution.deviation:.3g}")
            cells = (str(line + 1), solution.label,
                     "—" if component is None else f"{component.value('R'):.3g}",
                     "—" if component is None or not component.structure else f"{component.value('D'):.3g}",
                     f"{solution.chi2:.3g}" if math.isfinite(solution.chi2) else "—")
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
        solution = self.session.solutions[line]
        self.session.result = None
        self.set_model(solution.model, message=tr("{model} is in Model (Undo brings the previous one back).").format(
            model=solution.label))

    # -- save -------------------------------------------------------------------------------

    def _save_path(self, title: str, name: str, filters: str):
        folder = Path(str(self._remembered("save_folder") or self._remembered("folder") or ""))
        path, _ = QFileDialog.getSaveFileName(self, tr(title), str(folder / name), filters)
        if path:
            self._remember(save_folder=str(Path(path).parent))
        return path

    def _stem(self) -> str:
        curve = self.session.curve
        return Path(curve.name).stem if curve is not None else "fit"

    def export_data_dialog(self, path=None):
        data = self._data()
        if data is None or not data.q.size:
            self._status(tr("Open a curve first."), "warning")
            return None
        path = path or self._save_path("Save Data and Fit", f"{self._stem()}_fit.csv", "CSV (*.csv)")
        if not path:
            return None
        names, table = export_table(self.session.model, data)
        record = fit_record(self.session.model, data, self.session.curve, side=self.session.side,
                            q_range=self.session.q_range,
                            result=self.session.result if self.session.result_is_current() else None,
                            excluded=self.session.excluded)
        try:
            record_path = write_pair(path, lambda target: np.savetxt(
                target, table, delimiter=",", header=",".join(names), comments="", fmt="%.8g"), record)
        except OSError as exc:
            return self._save_failed(exc)
        self._status(lambda: tr("Saved {name} and its record {record}.").format(name=Path(path).name, record=record_path.name),
                     "ok", action=folder_action(path))
        return path

    def export_plot_dialog(self, path=None):
        path = path or self._save_path("Save Plot", f"{self._stem()}_fit.png", "PNG image (*.png);;SVG vector (*.svg)")
        if not path:
            return None
        try:
            write_plot(self.plot.plot, path)
        except OSError as exc:
            return self._save_failed(exc)
        self._status(lambda: tr("Saved {name}.").format(name=Path(path).name), "ok", action=folder_action(path))
        return path

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


__all__ = ["FitResultsMixin"]
