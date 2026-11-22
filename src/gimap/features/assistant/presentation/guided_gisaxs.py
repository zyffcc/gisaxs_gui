"""The GISAXS results of the automatic analysis: the cut and its halves, the length scale 2π/q*, the fit.

The fit (Predict parameters) is shown as a plot (the curve that was fitted and the solution selected in
the table) and a table of the solutions, most probable first, with their probability; the one the run
reports (the most probable) is selected first and the values at a bound of their range are named under
the table; **Fit details** (closed until opened) give every number of the selected solution with its
intervals and how it was fitted, and open it in Fitting. The curve, the fitted curve and the table can be
saved, and the prepared curve opened in Fitting to refine it with a chosen model.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QPushButton, QToolButton, QVBoxLayout, QWidget

from src.gimap.app.presentation.components import AdvancedSection, CurvePlot, FlowLayout
from src.gimap.app.presentation.i18n import current_language, tr, trf
from src.gimap.shared import fit_names
from src.gimap.shared.numbers import format_number
from src.gimap.shared.export_text import export_text

from ..application import (
    ASK_PROBABILITY,
    bound_phrases,
    chosen_solution,
    cut_spans,
    fit_curve_table,
    fit_on,
    fit_solutions_csv,
    pixel_span,
    probability,
    rivals,
)
from ..application.gisaxs_choice import component_number, population_values
from ..application.operations import halves_label
from .guided_details import SolutionDetails
from .guided_text import GOOD, NOTE, WARN, ask_save_path, badge, label, listed, save_failed_toast, saved_toast, table
from .guided_words import parts, words

NM_PER_A = 10.0
FIT_PLOT_HEIGHT = 300
COLUMNS = (("R", "R (nm)"), ("h", "h (nm)"), ("D", "D (nm)"))
FIT_HEADERS = (("model", "The particles of the solution: one or two populations; (S): with a paracrystal distance D. "
                          "⚠: a warning of the fit. A value marked ⚠ ended on a bound of its range, which the curve does "
                          "not determine (listed under the table; Fit details below: every parameter)."),
               ("p", "p: the posterior probability of the solution, relative to the compositions that were tried; the "
                     "solutions are sorted by it, most probable first. Another solution with p of 0.2 or more: the "
                     "curve alone does not decide the model."),
               ("R", "R (nm): the particle radius of every population, in the order of the radii."),
               ("h", "h (nm): the cylinder height of every population, in the order of the radii (– for a sphere)."),
               ("D", "D (nm): the paracrystal distance of every population, in the order of the radii (– without a "
                     "paracrystal)."),
               ("χ²ᵣ", fit_names.CHI2_DEFINITION))
"""The fit table's columns: short, so it fits the Results tab (the unit in the tooltip and the line under it; the
model names in it are names, never translated). R, h and D are of every population, in the order of the radii
(“3.06 / 6.02”; Fit details: every value)."""
FIT_NUMBERS = (1, 2, 3, 4, 5)
FIT_COPY_HEADERS = ("model", "probability", "R (nm)", "h (nm)", "D (nm)", "χ²ᵣ")
"""The fit table's columns with their units, for copied rows."""
DETAILS = (("R", "R", " nm"), ("sigma_R", "σR/R", ""), ("h", "h", " nm"), ("sigma_h", "σh/h", ""), ("D", "D", " nm"),
           ("sigma_D", "σD/D", ""), ("weight", "weight", ""))
BOUNDS_WARNING = "Not determined by the curve (at a search bound) — {rows}"
BOUNDS_WHOLE = "Values at a search bound, not determined by the curve — {rows}."
"""The values of the solutions that ended on a bound of their range: a warning with their names by rank, which bound
in its tooltip."""
FIT_TIP = ("Predict parameters: GIMaP's network proposes the particles (one or two populations of spheres and upright "
           "cylinders) and their values; each solution is refined by least squares, p is its posterior probability. "
           "Choose the model from what you know of the sample and refine it in Fitting.")
"""The tooltip of “Fit of I(qy)”: what the fits are and are not."""
PROBABLE_NOTE = ("{n} solutions have p ≥ {limit}: the curve alone does not decide the model. Compare their fits (select "
                 "a row) and choose from what you know of the sample.")
BIC_TIP = "Evidence sampling failed. The solutions are ranked by normalised BIC weights."
BIC_NOTE = "{n} solutions have BIC weights ≥ {limit}. Select a row to compare their fits."
BIC_OUTCOME = "{model} R {radius} nm (BIC weight {p}, χ²ᵣ {chi2})"
"""Under the table when another solution is probable too (``gisaxs_choice.rivals``)."""


def model_name(model: str) -> str:
    return str(model or "").replace("_", " ")


def _details(row: dict) -> str:
    """Every parameter and warning of a solution (the row's tooltip), its words in the interface language."""
    components = row.get("components") or [{}]
    values = "; ".join(
        (f"{number}: " if len(components) > 1 else "")
        + ", ".join(f"{tr(name)} {component_number(row, key, number - 1)}{unit}" for key, name, unit in DETAILS if component.get(key) is not None)
        for number, component in enumerate(components, start=1))
    notes = list(row.get("warnings") or ()) + ([] if row.get("converged", True) else ["not converged"])
    label = tr(fit_names.BIC_WEIGHT) if fit_names.weight_key(row) == "bic_weight" else "p"
    chance = "" if _probability(row) == "—" else f"; {label} {_probability(row)}"
    more = [words(note) for note in row.get("more_warnings") or ()]
    return (f"{model_name(row['model'])}: {values}; χ²ᵣ {_number(row['chi2'])}{chance}"
            + "".join(f"\n⚠ {words(note)}" for note in notes) + "".join(f"\n⚠ {phrase}" for phrase in bound_words(row))
            + ("\n" + trf(fit_names.MORE, notes="; ".join(more)) if more else ""))


def _probability(row: dict) -> str:
    if row.get("alone"):  # the one solution of a composition that was given: its 1.00 is no finding
        return "—"
    chance = probability(row)
    return "—" if chance != chance else f"{chance:.2f}"  # NaN: the fit gave none


def bound_words(row: dict) -> list[str]:
    """The values of a solution at a search bound (``D at its upper bound``), in the interface language."""
    return [words(phrase) for phrase in bound_phrases(row)]


def bound_names(row: dict) -> list[str]:
    """The names of a solution's values at a search bound (``D``, ``σh/h``: names, never translated)."""
    return [phrase.rsplit(" at its ", 1)[0] for phrase in bound_phrases(row)]


def _joined(items) -> str:
    return ("、" if current_language() == "zh" else ", ").join(items)


def _number(value, digits: int = 3) -> str:
    return format_number(value, digits=digits)


def cut_rows(report: dict) -> str:
    """The rows of the horizontal cut in the report, both ends included (as the Cuts card says them)."""
    cuts = (report.get("gisaxs") or {}).get("cuts") or {}
    return cut_spans(cuts, (report.get("tables") or {}).get("status"))[0]


def halves_text(side: str) -> str:
    """Which halves of the cut, in the interface language and lower case ("mean of both halves")."""
    text = halves_label(side, current_language())
    return text[:1].lower() + text[1:] if text[:1].isascii() else text


def outcome_summary(report: dict) -> str:
    """Which halves, the length scale 2π/q* and the fit the run reports, in one line (no technique, no full stop)."""
    gisaxs = report.get("gisaxs") or {}
    parts = []
    halves = gisaxs.get("halves")
    if halves:
        parts.append(tr("halves: {halves}").format(halves=halves_text(halves["side"])))
    spacing = gisaxs.get("spacing")
    if spacing:
        gap = "" if current_language() == "zh" else " "  # a Chinese full-width bracket needs no space before it
        parts.append(f"2π/q* = {spacing['distance_nm']:.3g} nm" + ("" if spacing.get("kind") == "maximum" else gap + tr("(shoulder)")))
    _index, best = chosen_solution(gisaxs.get("fit"))
    if best is not None:
        values = {"model": model_name(best["model"]), "radius": population_values(best, "R"),
                  "chi2": _number(best["chi2"])}
        chance = _probability(best)  # a report written before Predict parameters gives none
        parts.append(trf("best fit {model} R {radius} nm (χ²ᵣ {chi2})", **values) if chance == "—" else
                     trf(BIC_OUTCOME, p=chance, **values) if fit_names.weight_key(best) == "bic_weight" else
                     trf("most probable fit {model} R {radius} nm (p {p}, χ²ᵣ {chi2})", p=chance, **values))
        if bound_words(best):  # values the curve does not determine
            parts[-1] += " ⚠ " + _joined(bound_words(best))
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

    def _heading(self, text: str, layout: QVBoxLayout):
        heading = label(tr(text), self)
        heading.setProperty("gimapInspectorTitle", True)
        layout.addWidget(heading)
        return heading

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
                                       x="—" if symmetry.get("x_px") is None else f"{float(symmetry['x_px']):.2f}",
                                       shift=f"{shift:+.2f}"), self))
        halves = gisaxs.get("halves")
        if halves:  # the decision on the line, its reason in the tooltip
            chosen = label(trf("Halves: {halves}", halves=halves_label(halves["side"], current_language())), self,
                           role="muted")
            chosen.setObjectName("guidedHalves")
            chosen.setToolTip(words(halves["reason"]))
            layout.addWidget(chosen)

    def _spacing(self, layout: QVBoxLayout) -> None:
        spacing = (self.report.get("gisaxs") or {}).get("spacing")
        self._heading("Length scale 2π/q*", layout)
        if spacing is None:
            layout.addWidget(label(tr("No side maximum or shoulder away from qy = 0 in the measured range."), self))
            return
        values = {"d": f"{spacing['distance_nm']:.3g}", "q": f"{spacing['q']:.4g}"}
        # A note either way: a length the curve shows, which a form factor or the beam tail makes as well.
        if spacing.get("kind") == "maximum":
            layout.addWidget(badge(trf(
                "2π/q* = {d} nm at the peak of the cut (|qy| = {q} Å⁻¹): not necessarily the particle spacing.",
                **values), NOTE, self))
        else:
            layout.addWidget(badge(trf(
                "2π/q* = {d} nm at a shoulder of the cut (|qy| = {q} Å⁻¹, no resolved peak): not necessarily the "
                "particle spacing.", **values), NOTE, self))

    def _fit(self, layout: QVBoxLayout) -> None:
        solutions = self.fit.get("solutions") or []
        by_bic = bool(solutions) and fit_names.weight_key(solutions[0]) == "bic_weight"
        self._heading("Fit of I(qy)", layout).setToolTip(tr(BIC_TIP if by_bic else FIT_TIP))
        refine = QPushButton(tr("Refine in Fitting"), self)
        refine.setObjectName("guidedRefineFit")
        refine.setToolTip(tr("Open the prepared curve (the halves chosen) in Fitting and fit it with a model you choose."))
        refine.clicked.connect(self.refineRequested)
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
        rows, bounded, named = [], [], []
        for row in solutions:
            details = _details(row)
            bounds = bound_words(row)  # a value at a bound is not determined by the curve: ⚠ on its cell too
            ended = {item.get("parameter") for item in row.get("at_bounds") or ()}
            flagged = bool(row.get("warnings")) or not row.get("converged", True) or bool(bounds)
            rows.append([(model_name(row["model"]) + (" ⚠" if flagged else ""), details), (_probability(row), details),
                         *((population_values(row, key) + (" ⚠" if key in ended else ""), details)
                           for key, _name in COLUMNS),
                         (_number(row["chi2"]), details)])
            if bounds:
                bounded.append(trf("{rank}: {bounds}", rank=row.get("rank"), bounds=_joined(bounds)))
                named.append(trf("{rank}: {bounds}", rank=row.get("rank"), bounds=_joined(bound_names(row))))
        headers, copy_headers = list(FIT_HEADERS), list(FIT_COPY_HEADERS)
        if by_bic:
            headers[1], copy_headers[1] = (fit_names.BIC_WEIGHT, BIC_TIP), fit_names.BIC_WEIGHT
        self.fit_table = table(headers, rows, self, "guidedFitTable", numbers=FIT_NUMBERS, copy_headers=copy_headers)
        self.fit_table.setMinimumHeight(min(260, 60 + 30 * len(rows)))
        self.fit_table.itemSelectionChanged.connect(self._show_solution)
        layout.addWidget(self.fit_table)
        if bounded:  # the values at a bound by rank (the table is too narrow for them): a warning; which bound: its tooltip
            separator = "；" if current_language() == "zh" else "; "
            bounds_label = label("⚠ " + trf(BOUNDS_WARNING, rows=separator.join(named)), self, role="warning")
            bounds_label.setObjectName("guidedFitBounds")
            bounds_label.setToolTip(trf(BOUNDS_WHOLE, rows=separator.join(bounded)))
            layout.addWidget(bounds_label)
        index, _best = chosen_solution(self.fit)
        others = rivals(solutions)
        if others:  # another solution about as probable: the person chooses the model
            layout.addWidget(badge(trf(BIC_NOTE if by_bic else PROBABLE_NOTE, n=len(others) + 1,
                                       limit=f"{ASK_PROBABILITY:g}"), WARN, self))
        layout.addWidget(label(trf(
            "Fitted: {curve} ({points} points) · R, h and D in nm.",
            curve=parts(self.fit.get("curve")), points=self.fit.get("points")), self, role="muted",
        ))
        buttons = FlowLayout()
        save_curve = QPushButton(tr("Save Fitted Curve…"), self)
        save_curve.setObjectName("guidedSaveFitCurve")
        save_curve.setToolTip(tr("CSV: q (Å⁻¹), I, σ of the fitted curve and the fit the run reports at the same q."))
        save_curve.clicked.connect(self.save_curve)
        save_table = QPushButton(tr("Save Fit Table…"), self)
        save_table.setObjectName("guidedSaveFitTable")
        save_table.setToolTip(tr("CSV: every solution with its BIC weight, parameters (nm), intervals and χ²ᵣ."
                                 if by_bic else "CSV: every solution with its probability, parameters (nm), intervals and χ²ᵣ."))
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
        self.fit_table.selectRow(index)  # the solution the run reports

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
            path, columns, delimiter=",", header=export_text(header), comments="# ", encoding="utf-8"))

    def save_table(self, path: Optional[str] = None) -> Optional[str]:
        path = self._path(path, "Save Fit Table", "gisaxs_fit_solutions.csv", "CSV (*.csv)")
        if not path:
            return None

        def write() -> None:
            with open(path, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(export_text(fit_solutions_csv(self.fit)))

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


__all__ = ["BOUNDS_WARNING", "BOUNDS_WHOLE", "FIT_TIP", "GisaxsResults", "bound_names", "cut_rows", "outcome_text",
           "pixel_span"]
