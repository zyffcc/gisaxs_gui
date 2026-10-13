"""The GISAXS results of the automatic analysis: the cut and its halves, the spacing, the fit.

The fit is shown as a plot (the curve that was fitted and the solution selected in the table)
and a table of the solutions; **Fit details** (closed until opened) give every number of the
selected solution and how it was fitted, and open it in Fitting. The curve, the fitted curve and
the table can be saved, and the prepared curve opened in Fitting to refine it with a chosen model.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QPushButton, QToolButton, QVBoxLayout, QWidget

from src.gimap.app.presentation.components import AdvancedSection, CurvePlot, FlowLayout
from src.gimap.app.presentation.i18n import current_language, tr, trf

from ..application import cut_spans, fit_curve_table, fit_on, fit_solutions_csv, pixel_span
from ..application.operations import halves_label
from .guided_details import SolutionDetails
from .guided_text import GOOD, NOTE, WARN, ask_save_path, badge, label, listed, save_failed_toast, saved_toast, table
from .guided_words import parts, words

NM_PER_A = 10.0
FIT_PLOT_HEIGHT = 300
COLUMNS = (("R", "R (nm)"), ("h", "h (nm)"), ("D", "D (nm)"))
FIT_HEADERS = (("model", "Particle family; ⚠: a warning of the fit (Fit details below: every parameter)."),
               ("R", "R (nm): the particle radius."), ("h", "h (nm): the cylinder height (— for a sphere)."),
               ("D", "D (nm): the paracrystal distance between particles."),
               ("χ²", "Weighted mean squared residual: close values do not decide between models."))
"""The fit table's columns: short, so it fits the Results tab (the unit in the tooltip and the line under it; the
model names in it are names, never translated)."""
FIT_NUMBERS = (1, 2, 3, 4)
FIT_COPY_HEADERS = ("model", "R (nm)", "h (nm)", "D (nm)", "χ²")
"""The fit table's columns with their units, for copied rows."""
DETAILS = (("R", "R", " nm"), ("sigma_R", "σR/R", ""), ("h", "h", " nm"), ("sigma_h", "σh/h", ""), ("D", "D", " nm"),
           ("sigma_D", "σD/D", ""), ("weight", "weight", ""))


def model_name(model: str) -> str:
    return str(model or "").replace("_", " ")


def _details(row: dict) -> str:
    """Every parameter and warning of a solution (the row's tooltip), its words in the interface language."""
    component = (row.get("components") or [{}])[0]
    values = ", ".join(f"{tr(name)} {_number(component[key])}{unit}" for key, name, unit in DETAILS if component.get(key) is not None)
    notes = list(row.get("warnings") or ()) + ([] if row.get("converged") else ["not converged"])
    return f"{model_name(row['model'])}: {values}; χ² {_number(row['chi2'])}" + "".join(f"\n⚠ {tr(note)}" for note in notes)


def _number(value, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}g}"


def cut_rows(report: dict) -> str:
    """The rows of the horizontal cut in the report, both ends included (as the Cuts card says them)."""
    cuts = (report.get("gisaxs") or {}).get("cuts") or {}
    return cut_spans(cuts, (report.get("tables") or {}).get("status"))[0]


def halves_text(side: str) -> str:
    """Which halves of the cut, in the interface language and lower case ("mean of both halves")."""
    text = halves_label(side, current_language())
    return text[:1].lower() + text[1:] if text[:1].isascii() else text


def outcome_summary(report: dict) -> str:
    """Which halves, the spacing and the best fit, in one line (no technique, no full stop)."""
    gisaxs = report.get("gisaxs") or {}
    parts = []
    halves = gisaxs.get("halves")
    if halves:
        parts.append(tr("halves: {halves}").format(halves=halves_text(halves["side"])))
    spacing = gisaxs.get("spacing")
    if spacing:
        gap = "" if current_language() == "zh" else " "  # a Chinese full-width bracket needs no space before it
        parts.append(f"D ≈ {spacing['distance_nm']:.3g} nm" + ("" if spacing.get("kind") == "maximum" else gap + tr("(shoulder)")))
    solutions = (gisaxs.get("fit") or {}).get("solutions") or []
    if solutions:
        best = solutions[0]
        component = (best.get("components") or [{}])[0]
        parts.append(tr("best fit {model} R {radius} nm (χ² {chi2})").format(
            model=model_name(best["model"]), radius=_number(component.get("R")), chi2=_number(best["chi2"])))
    return listed(parts) if parts else tr("the horizontal cut is ready")  # Chinese: its own comma


def outcome_text(report: dict) -> str:
    """One line that names the technique: which halves, the spacing and the best fit."""
    return f"GISAXS — {outcome_summary(report)}" + ("。" if current_language() == "zh" else ".")


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
        heading = label(tr(text), self)
        heading.setProperty("gimapInspectorTitle", True)
        layout.addWidget(heading)

    def _cut(self, layout: QVBoxLayout) -> None:
        gisaxs = self.report.get("gisaxs") or {}
        cuts = gisaxs.get("cuts") or {}
        self._heading("Horizontal cut", layout)
        rows = cut_rows(self.report)  # the pixels the cut uses, both ends included, as in Analyze's Cuts card
        if cuts.get("horizontal_source") == "yoneda":
            layout.addWidget(badge(
                tr("At the Yoneda band (αf = {alpha}°), rows {rows}.").format(
                    alpha=_number(cuts.get("yoneda_alpha_f_deg")), rows=rows),
                GOOD, self,
            ))
        elif cuts.get("horizontal_source") == "horizon":
            layout.addWidget(badge(tr("No Yoneda band found: the cut sits just above the horizon. Check αi."), WARN, self))
        else:
            layout.addWidget(label(tr("Set by hand: rows {rows}.").format(rows=rows), self))
        symmetry = gisaxs.get("symmetry")
        if symmetry:
            shift = float(symmetry.get("shift_px") or 0.0)
            layout.addWidget(label(trf("Beam centre on the symmetry axis: x = {x} px ({shift} px).",
                                       x=_number(symmetry.get("x_px"), 5), shift=f"{shift:+.2f}"), self))
        halves = gisaxs.get("halves")
        if halves:
            layout.addWidget(label(tr("Halves: {halves} — {reason}").format(
                halves=halves_label(halves["side"], current_language()), reason=words(halves["reason"])), self, role="muted"))
        layout.addWidget(label(tr(
            "The cut is drawn on the image in Analyze: Sources ▸ on the canvas shows where each curve comes from."),
            self, role="muted",
        ))

    def _spacing(self, layout: QVBoxLayout) -> None:
        spacing = (self.report.get("gisaxs") or {}).get("spacing")
        self._heading("In-plane spacing", layout)
        if spacing is None:
            layout.addWidget(label(tr("No side maximum or shoulder away from qy = 0 in the measured range."), self))
            return
        values = {"d": f"{spacing['distance_nm']:.3g}", "q": f"{spacing['q']:.4g}"}
        if spacing.get("kind") == "maximum":
            layout.addWidget(badge(trf("D ≈ 2π/q* = {d} nm (maximum at |qy| = {q} Å⁻¹).", **values), GOOD, self))
        else:
            layout.addWidget(badge(
                trf("A shoulder at |qy| = {q} Å⁻¹ (2π/q ≈ {d} nm): a hint, not a resolved peak.", **values), NOTE, self))

    def _fit(self, layout: QVBoxLayout) -> None:
        self._heading("Fit of I(qy)", layout)
        refine = QPushButton(tr("Refine in Fitting"), self)
        refine.setObjectName("guidedRefineFit")
        refine.setToolTip(tr("Open the prepared curve (the halves chosen) in Fitting and fit it with a model you choose."))
        refine.clicked.connect(self.refineRequested)
        solutions = self.fit.get("solutions") or []
        if not solutions:
            layout.addWidget(label(tr("Not fitted here. Open the prepared curve in Fitting to fit it with a model."), self))
            layout.addWidget(refine)
            return
        self.fit_plot = CurvePlot("", self, log_y=True, log_x=True)
        self.fit_plot.setObjectName("guidedFitPlot")
        self.fit_plot.set_labels("|qy| (Å⁻¹)", "Intensity")
        self.fit_plot.setMinimumHeight(FIT_PLOT_HEIGHT)
        save_plot = QToolButton(self.fit_plot)
        save_plot.setObjectName("guidedSaveFitPlot")
        save_plot.setText(tr("Save Plot…"))
        save_plot.setToolTip(tr("The plot as shown: PNG (image) or SVG (vector)."))
        save_plot.clicked.connect(lambda: self.save_plot())
        self.fit_plot.header_layout.insertWidget(1, save_plot)
        layout.addWidget(self.fit_plot)
        rows = []
        for row in solutions:
            component = (row.get("components") or [{}])[0]
            details = _details(row)
            flagged = bool(row.get("warnings")) or not row.get("converged")
            rows.append([(model_name(row["model"]) + (" ⚠" if flagged else ""), details),
                         *((_number(component.get(key)), details) for key, _name in COLUMNS), (_number(row["chi2"]), details)])
        self.fit_table = table(FIT_HEADERS, rows, self, "guidedFitTable", numbers=FIT_NUMBERS,  # Fit details follow the row
                               copy_headers=FIT_COPY_HEADERS)
        self.fit_table.setMinimumHeight(min(260, 60 + 30 * len(rows)))
        self.fit_table.itemSelectionChanged.connect(self._show_solution)
        layout.addWidget(self.fit_table)
        layout.addWidget(label(trf(
            "Fitted: {curve} ({points} points); R, h and D in nm. Numerical fits of single particle families with size "
            "dispersity and a paracrystal distance D; choose the model from what you know of the sample.",
            curve=parts(self.fit.get("curve")), points=self.fit.get("points")), self, role="muted",
        ))
        buttons = FlowLayout()
        save_curve = QPushButton(tr("Save Fitted Curve…"), self)
        save_curve.setObjectName("guidedSaveFitCurve")
        save_curve.setToolTip(tr("CSV: q (Å⁻¹), I, σ of the fitted curve and the best fit at the same q."))
        save_curve.clicked.connect(self.save_curve)
        save_table = QPushButton(tr("Save Fit Table…"), self)
        save_table.setObjectName("guidedSaveFitTable")
        save_table.setToolTip(tr("CSV: every solution with its parameters (nm) and χ²."))
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

    def _path(self, path: Optional[str], title: str, suffix: str, filters: str) -> str:
        """``path``, or one chosen in a dialog that starts next to the data (``<stem>_<suffix>``)."""
        return path or ask_save_path(self, title, self.report.get("frame"), suffix, filters)[0]

    def _written(self, path: str, write: Callable[[], object]) -> Optional[str]:
        """Write, then say so with Open Folder; a failure (an error, or an exporter that returns False) is a toast too."""
        try:
            if write() is False:
                raise OSError(tr("the file was not written"))
        except OSError as exc:
            save_failed_toast(self, path, exc.strerror or str(exc))
            return None
        saved_toast(self, path)
        return path

    def save_curve(self, path: Optional[str] = None) -> Optional[str]:
        header, columns = fit_curve_table(self.fit)
        path = self._path(path, "Save Fitted Curve", "gisaxs_fit_curve.csv", "CSV (*.csv)")
        if not path:
            return None
        return self._written(path, lambda: np.savetxt(
            path, columns, delimiter=",", header=header, comments="# ", encoding="utf-8"))

    def save_table(self, path: Optional[str] = None) -> Optional[str]:
        path = self._path(path, "Save Fit Table", "gisaxs_fit_solutions.csv", "CSV (*.csv)")
        if not path:
            return None

        def write() -> None:
            with open(path, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(fit_solutions_csv(self.fit))

        return self._written(path, write)

    def save_plot(self, path: Optional[str] = None) -> Optional[str]:
        """The fit plot as shown, PNG or SVG by the file suffix."""
        if self.fit_plot is None:
            return None
        path = self._path(path, "Save Fit Plot", "gisaxs_fit.png", "PNG image (*.png);;SVG vector (*.svg)")
        if not path:
            return None
        from src.gimap.app.presentation.components.plot_export import export_plot

        plot = self.fit_plot  # on the light plot palette in either theme, as Compare and Fitting save theirs
        return self._written(path, lambda: export_plot(plot, path))

    def dispose(self) -> None:
        if self.fit_plot is not None:
            self.fit_plot.dispose()


__all__ = ["GisaxsResults", "cut_rows", "outcome_text", "pixel_span"]
