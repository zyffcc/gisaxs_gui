"""The Results step: how good the fit is, every value with its error, the solutions of the other
methods, and saving (the points with the model and its terms plus a JSON record, the plot, the model).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem

from src.gimap.app.presentation.i18n import tr

from ...application.single_fit import (
    export_table,
    fit_record,
    model_to_dict,
    parameter_text,
)
from .session import path_name


def _item(text: str, tip: str = "") -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    if tip:
        item.setToolTip(tip)
    return item


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
        if result is None and session.solutions:
            self.quality_label.setText(tr("{count} solutions below; the best is in Model. Fit (Refine) gives its quality "
                                          "and the error of every value.").format(count=len(session.solutions)))
            self.warnings_label.setText("")
            self.step_intro["results"].setText(tr("{count} solutions to compare.").format(count=len(session.solutions)))
        elif result is None:
            self.quality_label.setText(tr("No fit yet: Fit shows here how good it is and the error of every value."))
            self.warnings_label.setText("")
            self.step_intro["results"].setText(tr("After a fit: its quality, the errors and the solutions to compare."))
        else:
            self.quality_label.setText(tr(
                "{quality}\nlog RMSE {rmse} · {evaluations} evaluations · {seconds} s · {method}"
            ).format(quality=self._quality_text(result), rmse=f"{result.log_rmse:.3g}", evaluations=result.evaluations,
                     seconds=f"{result.seconds:.1f}", method=tr("Search the ranges") if result.method == "global" else tr("Refine")))
            self.warnings_label.setText("\n".join(self._warnings(result)))
            self.step_intro["results"].setText(self._quality_text(result) + ".")
        self._fill_parameters(result if session.result_is_current() else None)
        self._fill_solutions()

    def _warnings(self, result) -> list[str]:
        lines = []
        if not self.session.result_is_current():
            lines.append("⚠ " + tr("The model was changed after this fit: the errors belong to the fitted values."))
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
            text = parameter_text(path[1], parameter.value, error)
            value, _sep, err = text.partition(" ± ")
            rows.append((self._path_name(path), value if err else text,
                         err or (tr("fixed") if not parameter.free else "")))
        self.parameters_table.setRowCount(len(rows))
        for line, cells in enumerate(rows):
            for column, text in enumerate(cells):
                self.parameters_table.setItem(line, column, _item(text))

    def _fill_solutions(self) -> None:
        solutions = self.session.solutions
        self.solutions_title.setVisible(bool(solutions))
        self.solutions_table.setVisible(bool(solutions))
        self.use_solution_button.setVisible(bool(solutions))
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
        self.use_solution_button.setEnabled(self.solutions_table.currentRow() >= 0)

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
        np.savetxt(path, table, delimiter=",", header=",".join(names), comments="", fmt="%.8g")
        record = fit_record(self.session.model, data, self.session.curve, side=self.session.side,
                            q_range=self.session.q_range,
                            result=self.session.result if self.session.result_is_current() else None,
                            excluded=self.session.excluded)
        Path(path).with_suffix(".json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        self._status(tr("Saved {name} and its record {record}.").format(
            name=Path(path).name, record=Path(path).with_suffix(".json").name), "ok")
        return path

    def export_plot_dialog(self, path=None):
        path = path or self._save_path("Save Plot", f"{self._stem()}_fit.png", "PNG image (*.png);;SVG vector (*.svg)")
        if not path:
            return None
        from pyqtgraph import exporters

        item = self.plot.plot
        exporter = exporters.SVGExporter(item) if str(path).lower().endswith(".svg") else exporters.ImageExporter(item)
        if isinstance(exporter, exporters.ImageExporter):
            exporter.parameters()["width"] = 1600
        exporter.export(str(path))
        self._status(tr("Saved {name}.").format(name=Path(path).name), "ok")
        return path

    def save_model_dialog(self, path=None):
        path = path or self._save_path("Save Model", f"{self._stem()}_model.json", "Fit model (*.json)")
        if not path:
            return None
        Path(path).write_text(json.dumps(model_to_dict(self.session.model), indent=2), encoding="utf-8")
        self._status(tr("Saved {name}.").format(name=Path(path).name), "ok")
        return path


__all__ = ["FitResultsMixin"]
