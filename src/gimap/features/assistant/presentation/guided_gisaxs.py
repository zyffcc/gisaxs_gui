"""The GISAXS results of the automatic analysis: the cut and its halves, the spacing, the fit.

The fit is shown as a plot (the curve that was fitted and the solution selected in the table)
and a table of the solutions; **Fit details** (closed until opened) give every number of the
selected solution and how it was fitted, and open it in Fitting. The curve, the fitted curve and
the table can be saved, and the prepared curve opened in Fitting to refine it with a chosen model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QPushButton, QToolButton, QVBoxLayout, QWidget

from src.gimap.app.presentation.components import AdvancedSection, CurvePlot, FlowLayout, show_toast

from ..application import fit_curve_table, fit_on, fit_solutions_csv
from .guided_details import SolutionDetails
from .guided_text import GOOD, NOTE, WARN, badge, label, table

NM_PER_A = 10.0
FIT_PLOT_HEIGHT = 300
COLUMNS = (("R", "R (nm)"), ("h", "h (nm)"), ("D", "D (nm)"))
DETAILS = (("R", "R", " nm"), ("sigma_R", "σR/R", ""), ("h", "h", " nm"), ("sigma_h", "σh/h", ""), ("D", "D", " nm"),
           ("sigma_D", "σD/D", ""), ("weight", "weight", ""))


def model_name(model: str) -> str:
    return str(model or "").replace("_", " ")


def _details(row: dict) -> str:
    """Every parameter and warning of a solution (the row's tooltip)."""
    component = (row.get("components") or [{}])[0]
    values = ", ".join(f"{name} {_number(component[key])}{unit}" for key, name, unit in DETAILS if component.get(key) is not None)
    notes = list(row.get("warnings") or ()) + ([] if row.get("converged") else ["not converged"])
    return f"{model_name(row['model'])}: {values}; χ² {_number(row['chi2'])}" + "".join(f"\n⚠ {note}" for note in notes)


def _number(value, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}g}"


def outcome_text(report: dict) -> str:
    """One line: where the cut is, which halves, the spacing and the best fit."""
    gisaxs = report.get("gisaxs") or {}
    parts = []
    halves = gisaxs.get("halves")
    if halves:
        parts.append(f"halves: {halves['side']}")
    spacing = gisaxs.get("spacing")
    if spacing:
        parts.append(f"D ≈ {spacing['distance_nm']:.3g} nm" + ("" if spacing.get("kind") == "maximum" else " (shoulder)"))
    solutions = (gisaxs.get("fit") or {}).get("solutions") or []
    if solutions:
        best = solutions[0]
        component = (best.get("components") or [{}])[0]
        parts.append(f"best fit {model_name(best['model'])} R {_number(component.get('R'))} nm (χ² {_number(best['chi2'])})")
    return "Done: " + (", ".join(parts) if parts else "the horizontal cut is ready") + "."


class GisaxsResults(QWidget):
    """Sections of the Results tab for a GISAXS report."""

    refineRequested = pyqtSignal()
    """Open the prepared curve in Fitting."""
    solutionRequested = pyqtSignal(dict)
    """… with the selected solution drawn (Fit details ▸ Show in Fitting)."""

    def __init__(self, report: dict, parent: Optional[QWidget] = None, *,
                 details: Optional[Callable[[], AdvancedSection]] = None):
        """``details`` makes the (empty) Fit details section to put under the table."""
        super().__init__(parent)
        self.setObjectName("guidedGisaxs")
        self.report = report
        self.fit = (report.get("gisaxs") or {}).get("fit") or {}
        self.fit_plot: Optional[CurvePlot] = None
        self.fit_table = None
        self._make_details = details
        self.fit_details: Optional[AdvancedSection] = None
        self.solution_details: Optional[SolutionDetails] = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._cut(layout)
        self._spacing(layout)
        self._fit(layout)

    # -- sections ----------------------------------------------------------------------

    def _heading(self, text: str, layout: QVBoxLayout) -> None:
        heading = label(text, self)
        heading.setProperty("gimapInspectorTitle", True)
        layout.addWidget(heading)

    def _cut(self, layout: QVBoxLayout) -> None:
        gisaxs = self.report.get("gisaxs") or {}
        cuts = gisaxs.get("cuts") or {}
        self._heading("Horizontal cut", layout)
        rows = cuts.get("horizontal_rows") or [None, None]
        if cuts.get("horizontal_source") == "yoneda":
            layout.addWidget(badge(
                f"At the Yoneda band (αf = {_number(cuts.get('yoneda_alpha_f_deg'))}°), rows {_number(rows[0], 4)}–{_number(rows[1], 4)}.",
                GOOD, self,
            ))
        elif cuts.get("horizontal_source") == "horizon":
            layout.addWidget(badge("No Yoneda band found: the cut sits just above the horizon. Check αi.", WARN, self))
        else:
            layout.addWidget(label(f"Set by hand: rows {_number(rows[0], 4)}–{_number(rows[1], 4)}.", self))
        symmetry = gisaxs.get("symmetry")
        if symmetry:
            shift = float(symmetry.get("shift_px") or 0.0)
            layout.addWidget(label(
                f"Beam centre on the symmetry axis: x = {_number(symmetry.get('x_px'), 5)} px ({shift:+.2f} px).", self,
            ))
        halves = gisaxs.get("halves")
        if halves:
            layout.addWidget(label(f"Halves: {halves['side']} — {halves['reason']}", self, role="muted"))
        layout.addWidget(label(
            "The cut is drawn on the image in Analyze: Sources ▸ on the canvas shows where each curve comes from.",
            self, role="muted",
        ))

    def _spacing(self, layout: QVBoxLayout) -> None:
        spacing = (self.report.get("gisaxs") or {}).get("spacing")
        self._heading("In-plane spacing", layout)
        if spacing is None:
            layout.addWidget(label("No side maximum or shoulder away from qy = 0 in the measured range.", self))
        elif spacing.get("kind") == "maximum":
            layout.addWidget(badge(
                f"D ≈ 2π/q* = {spacing['distance_nm']:.3g} nm (maximum at |qy| = {spacing['q']:.4g} Å⁻¹).", GOOD, self,
            ))
        else:
            layout.addWidget(badge(
                f"A shoulder at |qy| = {spacing['q']:.4g} Å⁻¹ (2π/q ≈ {spacing['distance_nm']:.3g} nm): a hint, not a resolved peak.",
                NOTE, self,
            ))

    def _fit(self, layout: QVBoxLayout) -> None:
        self._heading("Fit of I(qy)", layout)
        refine = QPushButton("Refine in Fitting", self)
        refine.setObjectName("guidedRefineFit")
        refine.setToolTip("Open the prepared curve (the halves chosen) in Fitting and fit it with a model you choose.")
        refine.clicked.connect(self.refineRequested)
        solutions = self.fit.get("solutions") or []
        if not solutions:
            layout.addWidget(label("Not fitted here. Open the prepared curve in Fitting to fit it with a model.", self))
            layout.addWidget(refine)
            return
        self.fit_plot = CurvePlot("", self, log_y=True, log_x=True)
        self.fit_plot.setObjectName("guidedFitPlot")
        self.fit_plot.set_labels("|qy| (Å⁻¹)", "Intensity")
        self.fit_plot.setMinimumHeight(FIT_PLOT_HEIGHT)
        save_plot = QToolButton(self.fit_plot)
        save_plot.setObjectName("guidedSaveFitPlot")
        save_plot.setText("Save Plot…")
        save_plot.setToolTip("The plot as shown: PNG (image) or SVG (vector).")
        save_plot.clicked.connect(lambda: self.save_plot())
        self.fit_plot.header_layout.insertWidget(1, save_plot)
        layout.addWidget(self.fit_plot)
        headers = [("model", "Particle family; ⚠: a warning of the fit (Fit details below: every parameter).")]
        headers += [(name, "") for _key, name in COLUMNS]
        headers += [("χ²", "Weighted mean squared residual: close values do not decide between models.")]
        rows = []
        for row in solutions:
            component = (row.get("components") or [{}])[0]
            details = _details(row)
            flagged = bool(row.get("warnings")) or not row.get("converged")
            rows.append([(model_name(row["model"]) + (" ⚠" if flagged else ""), details),
                         *((_number(component.get(key)), details) for key, _name in COLUMNS), (_number(row["chi2"]), details)])
        self.fit_table = table(headers, rows, self, "guidedFitTable")
        self.fit_table.setMinimumHeight(min(260, 60 + 30 * len(rows)))
        self.fit_table.itemSelectionChanged.connect(self._show_solution)
        layout.addWidget(self.fit_table)
        layout.addWidget(label(
            f"Fitted: {self.fit.get('curve')} ({self.fit.get('points')} points). Numerical fits of single particle "
            "families with size dispersity and a paracrystal distance D; choose the model from what you know of the sample.",
            self, role="muted",
        ))
        buttons = FlowLayout()
        save_curve = QPushButton("Save Fitted Curve…", self)
        save_curve.setObjectName("guidedSaveFitCurve")
        save_curve.setToolTip("CSV: q (Å⁻¹), I, σ of the fitted curve and the best fit at the same q.")
        save_curve.clicked.connect(self.save_curve)
        save_table = QPushButton("Save Fit Table…", self)
        save_table.setObjectName("guidedSaveFitTable")
        save_table.setToolTip("CSV: every solution with its parameters (nm) and χ².")
        save_table.clicked.connect(self.save_table)
        for button in (save_curve, save_table, refine):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        if self._make_details is not None:
            self.fit_details = self._make_details()
            self.fit_details.setParent(self)
            self.solution_details = SolutionDetails(self.fit_details)
            self.solution_details.showRequested.connect(self.solutionRequested)
            self.fit_details.add_widget(self.solution_details)
            layout.addWidget(self.fit_details)
        self.fit_table.selectRow(0)

    # -- interaction -------------------------------------------------------------------

    def _show_solution(self) -> None:
        if self.fit_plot is None:
            return
        index = max(0, self.fit_table.currentRow())
        data = self.fit.get("data") or {}
        q = np.abs(np.asarray(data.get("q_inv_angstrom") or (), dtype=float))
        order = np.argsort(q)
        curves = [("data", q[order], np.asarray(data.get("intensity") or (), dtype=float)[order])]
        fitted = (self.fit.get("curves") or [])
        if index < len(fitted):
            model = fit_on(q[order], fitted[index])
            curves.append((f"fit {index + 1}: {model_name(self.fit['solutions'][index]['model'])}", q[order], model))
        self.fit_plot.set_curves(curves)
        if self.solution_details is not None:
            self.solution_details.show_solution(self.fit, index)

    def save_curve(self, path: Optional[str] = None) -> Optional[str]:
        header, columns = fit_curve_table(self.fit)
        path = path or QFileDialog.getSaveFileName(self, "Save Fitted Curve", "gisaxs_fit_curve.csv", "CSV (*.csv)")[0]
        if not path:
            return None
        np.savetxt(path, columns, delimiter=",", header=header, comments="# ", encoding="utf-8")
        show_toast(self.window(), f"Saved {Path(path).name}", level="ok")
        return path

    def save_table(self, path: Optional[str] = None) -> Optional[str]:
        path = path or QFileDialog.getSaveFileName(self, "Save Fit Table", "gisaxs_fit_solutions.csv", "CSV (*.csv)")[0]
        if not path:
            return None
        with open(path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(fit_solutions_csv(self.fit))
        show_toast(self.window(), f"Saved {Path(path).name}", level="ok")
        return path

    def save_plot(self, path: Optional[str] = None) -> Optional[str]:
        """The fit plot as shown, PNG or SVG by the file suffix."""
        if self.fit_plot is None:
            return None
        path = path or QFileDialog.getSaveFileName(
            self, "Save Fit Plot", "gisaxs_fit.png", "PNG image (*.png);;SVG vector (*.svg)"
        )[0]
        if not path:
            return None
        from pyqtgraph import exporters

        item = self.fit_plot.plot
        exporter = exporters.SVGExporter(item) if str(path).lower().endswith(".svg") else exporters.ImageExporter(item)
        if isinstance(exporter, exporters.ImageExporter):
            exporter.parameters()["width"] = 1600
        exporter.export(str(path))
        show_toast(self.window(), f"Saved {Path(path).name}", level="ok")
        return path

    def dispose(self) -> None:
        if self.fit_plot is not None:
            self.fit_plot.dispose()


__all__ = ["GisaxsResults", "outcome_text"]
