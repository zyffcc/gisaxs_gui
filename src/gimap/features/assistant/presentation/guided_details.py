"""The details of one fit in the Results tab: shown for the row selected in a table, only when asked.

A table gives one line per peak (GIWAXS) or per model solution (GISAXS); **Fit details** below it —
closed until opened, then following the selected row — gives everything behind that line:

* **a peak**: the points the Gaussian was fitted to, the fit and its local background drawn together;
  every parameter with its error; how it was fitted (weights, window, error scaling) and found (the
  search thresholds); its warnings; where on the detector it was measured;
* **a model solution** (Predict parameters): every parameter of every component with the 68 % posterior
  interval of R, h and D, the global ones (background, resolution), its probability and the quality
  numbers, the fit settings (q range, points, who chose the composition, algorithm, time), its warnings,
  and **Show in Fitting** to continue from it.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from src.gimap.app.presentation.components import CurvePlot
from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.theme import theme_manager
from src.gimap.shared import fit_names
from src.gimap.shared.numbers import format_number

from ..application import bound_phrases, probability
from ..application.gisaxs_choice import composition_modes, displayed_estimate, model_probability, weight_text
from ..application.pipeline_report import FLAG_WORDS
from ..application.report_numbers import peak_value
from .guided_text import ScaledPixmapLabel, label
from .guided_words import words

FIT_PLOT_HEIGHT = 190
ERROR_WORDS = {"poisson": "Poisson (counting) error", "scaled_poisson": "Poisson (counting) error",
               "sigma_frame": "Given σ frame",
               "pixel_scatter": "pixel-scatter error (statistical, not counting noise)"}
"""The errors the peaks were fitted with, by the frame's error model (the report's ``error_model``)."""
NAME_WIDTH = 64
"""The narrowest a name of Fit details wraps to (px): a few words or Chinese characters per line."""
NM_PER_A = 10.0
COMPONENT_VALUES = (("R", "R", "nm"), ("sigma_R", "σR/R", ""), ("h", "h", "nm"), ("sigma_h", "σh/h", ""),
                    ("D", "D", "nm"), ("sigma_D", "σD/D", ""), ("weight", "weight", ""), ("amplitude", "Scale", ""))
GLOBAL_VALUES = (("background", "Background", ""), ("resolution_amplitude", "Tail A", ""),
                 ("sigma_Res", "Tail w", "nm⁻¹"), ("nu_Res", "Tail ν", ""), ("q_resolution", "σ_q", "nm⁻¹"))
"""The names of Fitting's Model table (``shared/fit_names``): one name per quantity everywhere."""
MORE_NOTES = "Background, beam tail and resolution: {notes}"


def _parameter_text(solution, key, unit, index=None):
    value, bounds = displayed_estimate(solution, key, index)
    if bounds is None:
        return f"{format_number(value)} {unit}".strip()
    best, low, high = fit_names.interval_values(value, *bounds)
    return f"{best} {unit}".strip() + f" (68 %: {low}–{high})"


def method_text(peak: dict, report: dict) -> str:
    """How a GIWAXS peak was fitted and found, with the thresholds of the run (the tooltip of Fit details)."""
    fit, search = peak.get("fit") or {}, report.get("peak_search") or {}
    return tr(
        "{model}, least squares weighted by the {errors} of each bin; errors × √χ²ᵣ when χ²ᵣ > 1. Found in "
        "the radial I(q) of the whole detector: a SNIP background (window {window} Å⁻¹), at least {snr}σ and "
        "{height} % above it. Size: Scherrer, L = 2π·0.9 / FWHM (instrument broadening not removed)."
    ).format(model=tr(fit.get("model") or "Gaussian on a local linear background"),
             errors=tr(ERROR_WORDS.get(str(report.get("error_model") or ""), "statistical error")),
             window=format_number(search.get("background_window")), snr=format_number(search.get("min_snr")),
             height=format_number(100 * float(search.get("min_relative_height") or 0.02)))


class _Values(QWidget):
    """Name–value pairs in two columns, or in one when the panel is too narrow for two (the widgets are
    replaced for every row shown). The width is the panel's: the pairs never push it wider (a value and its
    error stay on one line, so two columns of them were wider than the Results tab of a 1280-px window)."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        policy = self.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Ignored)  # the width the panel gives, never more
        self.setSizePolicy(policy)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(2)
        self._cells: list[tuple[QLabel, QLabel]] = []
        self._columns = 0

    def set_values(self, pairs) -> None:
        while self.grid.count():
            widget = self.grid.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)  # off the screen now, not when the deletion comes round
                widget.deleteLater()
        self._cells = []
        for name, value in pairs:
            key = label(tr(name), self, role="muted")
            key.setMinimumWidth(min(key.sizeHint().width(), NAME_WIDTH))  # wrapped, never a character per line
            shown = label(str(value), self)
            shown.setWordWrap(False)  # a number and its error on one line; the names wrap
            shown.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._cells.append((key, shown))
        self._arrange(self._fitting_columns())

    def _fitting_columns(self) -> int:
        """2 when two columns of pairs fit the width given (names wrapped to their longest word), else 1."""
        half = (len(self._cells) + 1) // 2
        if not half:
            return 2
        columns = (self._cells[:half], self._cells[half:])
        needed = sum(max((max(cell[part].minimumSizeHint().width(), cell[part].minimumWidth()) for cell in column),
                         default=0) for column in columns for part in (0, 1))
        return 2 if needed + 3 * self.grid.horizontalSpacing() <= self.width() else 1

    def _arrange(self, columns: int) -> None:
        self._columns = columns
        for key, shown in self._cells:
            self.grid.removeWidget(key)
            self.grid.removeWidget(shown)
        rows = (len(self._cells) + 1) // 2 if columns == 2 else len(self._cells)
        for index, (key, shown) in enumerate(self._cells):
            row, column = index % max(1, rows), 2 * (index // max(1, rows))
            self.grid.addWidget(key, row, column)
            self.grid.addWidget(shown, row, column + 1)
        self.grid.setColumnStretch(1, 1)
        self.grid.setColumnStretch(3, 1 if columns == 2 else 0)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        columns = self._fitting_columns()
        if self._cells and columns != self._columns:
            self._arrange(columns)


class PeakDetails(QWidget):
    """How one GIWAXS peak was fitted. ``picture(target, rings, width)`` draws it on the q map."""

    def __init__(self, picture: Callable[[QLabel, list, int], bool], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("guidedPeakDetails")
        self._picture = picture
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.plot = CurvePlot("", self, log_y=False)
        self.plot.setObjectName("guidedPeakFitPlot")
        self.plot.set_labels("q (Å⁻¹)", "I (counts/pixel)")
        self.plot.setMinimumHeight(FIT_PLOT_HEIGHT)
        layout.addWidget(self.plot)
        self.values = _Values(self)
        layout.addWidget(self.values)
        self.warnings = label("", self)  # (how it was fitted and found: the section's tooltip, ``method_text``)
        layout.addWidget(self.warnings)
        self.where = ScaledPixmapLabel(self)  # shrinks with a narrow panel instead of pushing it wider
        self.where.setObjectName("guidedPeakMap")
        self.where.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.where.setToolTip(tr("Where the peak lies on the detector: white = measured, orange = in a shadow, "
                                 "red dashed = not measured."))
        layout.addWidget(self.where)

    def show_peak(self, peak: dict, report: dict, *, map_width: int) -> None:
        fit = peak.get("fit") or {}
        self.plot.setVisible(bool(fit.get("x")))
        if fit.get("x"):
            x = np.asarray(fit["x"], dtype=float)
            dense = np.linspace(float(x.min()), float(x.max()), 200)
            width = float(peak["fwhm"]) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
            background = float(fit.get("background") or 0.0) + float(fit.get("slope") or 0.0) * (dense - float(peak["q"]))
            model = float(fit.get("height") or 0.0) * np.exp(-0.5 * ((dense - float(peak["q"])) / width) ** 2) + background
            self.plot.set_curves(
                [(tr("Gaussian + background"), dense, model), (tr("local background"), dense, background),
                 (tr("measured"), x, np.asarray(fit["y"], dtype=float))],  # the points on top of the fit
                ["#2563eb", "#f97316", theme_manager().color("plot_fg")],
            )
        size = peak.get("size_nm")
        size_text = "—"
        if size is not None:
            lower = "≥ " if peak.get("size_is_lower_bound") else ""
            size_text = lower + format_number(size, peak.get("size_err_nm")) + " nm"
        pairs = [
            ("q", peak_value(peak, "q") + " Å⁻¹"),
            ("d = 2π/q", peak_value(peak, "d_A") + " Å"),
            ("FWHM", peak_value(peak, "fwhm") + " Å⁻¹"),
            ("height above background", format_number(fit.get("height"))),
            ("area", format_number(fit.get("area"))),
            ("background at the peak", format_number(fit.get("background"))),
            ("background slope", format_number(fit.get("slope"))),
            ("signal / noise", f"{format_number(peak.get('snr'))} σ"),
            ("χ²ᵣ of the fit", format_number(fit.get("reduced_chi2"))),
            ("points fitted", str(fit.get("points", "—"))),
            ("fit window", "—" if not fit.get("window") else
             f"{format_number(fit['window'][0])}–{format_number(fit['window'][1])} Å⁻¹"),
            ("crystallite size", size_text),
            ("out-of-plane / in-plane", format_number(peak.get("ratio_out_in"), peak.get("ratio_out_in_err"))),
        ]
        self.values.set_values(pairs)
        caveat = str(peak.get("caveat") or "")
        warnings = [tr(caveat)] if caveat else []
        warnings.extend(tr(FLAG_WORDS[flag]) for flag in peak.get("flags") or () if flag in FLAG_WORDS)
        self.warnings.setText("\n".join("⚠ " + note for note in warnings) or tr("No warnings for this peak."))
        ring = min(report.get("rings") or (), key=lambda item: abs(item["q"] - peak["q"]), default=None)
        shadowed = ring.get("shadowed") if ring and abs(ring["q"] - peak["q"]) <= peak.get("fwhm", 0.0) else ()
        overlay = {"q": peak["q"], "shadowed": shadowed or (), "label": f"q = {peak_value(peak, 'q', value_only=True)} Å⁻¹"}
        self._picture(self.where, [overlay], map_width)

    def dispose(self) -> None:
        self.plot.dispose()


def candidate_row(fit: dict, index: int) -> Optional[dict]:
    """A solution as Fitting takes it (a ``native_v5`` candidate: components, globals, its fitted points)."""
    solutions = fit.get("solutions") or []
    native = fit.get("native") or []
    if not 0 <= index < len(solutions) or index >= len(native):
        return None
    solution = solutions[index]
    components = []
    for component in solution.get("components") or ():
        params = {key: value for key, value in component.items() if key not in ("type", "weight", "amplitude")}
        components.append({"type": component.get("type"), "weight": component.get("weight", 0.0), "params": params})
    points = native[index]
    if not points.get("q_inv_nm") or not points.get("observed") or not points.get("display_q"):
        return None
    return {
        "workflow": "native_v5", "rank": solution.get("rank", index + 1),
        **({"physics_backend": "cut_model", "model": points["model"]} if "model" in points else {}),
        "combination": solution.get("combination") or solution.get("model"),
        "side": fit.get("side") or "", "components": components, "global_params": dict(solution.get("globals") or {}),
        "native_q": list(points["q_inv_nm"]), "native_fit": list(points.get("fit") or ()),
        "observed": list(points["observed"]), "sigma": list(points.get("sigma") or ()),
        "display_q": list(points["display_q"]), "display_fit": list(points.get("display_fit") or ()),
        "best_chi2_weighted": solution.get("chi2"), "warnings": list(solution.get("warnings") or ()),
        "best_source": points.get("source") or "Analyze automatic analysis", "unit_contract": dict(points.get("units") or {}),
    }


class SolutionDetails(QWidget):
    """Every number of one GISAXS model solution and how it was fitted; Show in Fitting."""

    showRequested = pyqtSignal(dict)
    """The solution as a Fitting candidate (``candidate_row``)."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("guidedSolutionDetails")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.values = _Values(self)
        layout.addWidget(self.values)
        self.settings = label("", self, role="muted")
        layout.addWidget(self.settings)
        self.warnings = label("", self)
        layout.addWidget(self.warnings)
        self.show_button = QPushButton(tr("Show in Fitting"), self)
        self.show_button.setObjectName("guidedShowSolution")
        self.show_button.setToolTip(tr("Open the fitted curve in Fitting with this solution drawn; Export Data… "
                                       "there saves the fit with its parameters"))
        layout.addWidget(self.show_button, 0, Qt.AlignLeft)
        self.show_button.clicked.connect(self._show_in_fitting)
        self._fit: dict = {}
        self._index = 0
        self.allowed = True
        """False while another frame is shown in Analyze: Show in Fitting would draw this solution on its curve."""

    def set_allowed(self, allowed: bool) -> None:
        self.allowed = bool(allowed)
        self.show_button.setEnabled(self.allowed and candidate_row(self._fit, self._index) is not None)

    def show_solution(self, fit: dict, index: int) -> None:
        self._fit, self._index = fit, int(index)
        solution = (fit.get("solutions") or [{}])[self._index]
        pairs = []
        for number, component in enumerate(solution.get("components") or (), start=1):
            prefix = str(component.get("type", "")).replace("_", " ") + (f" {number}" if len(solution["components"]) > 1 else "")
            for key, name, unit in COMPONENT_VALUES:
                if component.get(key) is not None:
                    shown = _parameter_text(solution, key, unit, number - 1)
                    pairs.append((f"{prefix}: {name}" if key == "R" else name, shown))
        for key, name, unit in GLOBAL_VALUES:
            if (solution.get("globals") or {}).get(key) is not None:
                pairs.append((name, _parameter_text(solution, key, unit)))
        fitted = solution.get("points") or fit.get("fitted_points") or {}
        over = "" if fitted.get("fitted") is None else " (" + trf(
            "{fitted} of {given} points", fitted=fitted["fitted"], given=fitted.get("given", fitted["fitted"])) + ")"
        by_bic = fit_names.weight_key(solution) == "bic_weight"
        weight = model_probability(solution)
        weight_name = fit_names.BIC_WEIGHT if by_bic else "probability"
        if solution.get("alone"):
            pairs.append((weight_name, tr(fit_names.ONLY_ONE)))
        else:
            multiple = composition_modes(solution, fit.get("solutions") or []) > 1
            if multiple:
                weight_name = "model BIC weight" if by_bic else "model probability"
            pairs.append((weight_name, weight_text(weight)))
            if multiple:
                pairs.append(("this solution", weight_text(probability(solution))))
        pairs += [("χ²ᵣ", format_number(solution.get("chi2")) + over), ("log RMSE", format_number(solution.get("log_rmse"))),
                  ("converged", tr("yes") if solution.get("converged", True) else tr("no")),
                  ("time", "—" if solution.get("seconds") is None else f"{solution['seconds']:.1f} s")]
        for name, stat in (solution.get("posterior_summary") or {}).items():
            median, low, high = fit_names.interval_values(stat["median"], stat["low68"], stat["high68"])
            pairs.append((name, trf("Posterior median {value} (68 % {low}–{high})", value=median, low=low, high=high)))
        self.values.set_values(pairs)
        low, high = (fit.get("q_range_inv_angstrom") or [math.nan, math.nan])[:2]
        screen = tr("the composition you gave") if solution.get("screen") == "user_composition" else tr(
            "the composition the network chose")
        setting = (
            "{points} points, |qy| {low}–{high} Å⁻¹ ({low_nm}–{high_nm} nm⁻¹); {screen}. {algorithm}. Sizes in nm; "
            "σ/value are relative size spreads; the weight is the share of the fitted amplitude, not a volume "
            "fraction; the intervals hold 68 % of the posterior."
        ) if not by_bic else (
            "{points} points, |qy| {low}–{high} Å⁻¹ ({low_nm}–{high_nm} nm⁻¹). {screen}. {algorithm}. "
            "BIC fallback intervals use least-squares errors."
        )
        self.settings.setText(tr(setting).format(points=fit.get("points", "—"), low=format_number(low), high=format_number(high),
                 low_nm=format_number(NM_PER_A * low), high_nm=format_number(NM_PER_A * high),
                 screen=screen, algorithm=words(solution.get("algorithm")) or tr("Bounded least squares")))
        notes = [words(note) for note in list(solution.get("warnings") or ())
                 + ([] if solution.get("converged", True) else ["not converged"])]
        notes += [words(phrase) for phrase in bound_phrases(solution)]  # which values ended on a bound of their range
        lines = ["⚠ " + note for note in notes]
        lines += [trf(template, **values) for template, values in fit_names.diagnostic_lines(
            solution.get("diagnostics") or {}, solution.get("ess"))]
        more = [words(note) for note in solution.get("more_warnings") or ()]
        if more:  # not the particles' values: one paragraph after the others
            lines.append(trf(MORE_NOTES, notes="; ".join(more)))
        self.warnings.setText("\n".join(lines) if lines else tr("No warnings for this solution."))
        self.show_button.setEnabled(self.allowed and candidate_row(fit, self._index) is not None)

    def _show_in_fitting(self) -> None:
        row = candidate_row(self._fit, self._index)
        if row is not None and self.allowed:
            self.showRequested.emit(row)


__all__ = ["PeakDetails", "SolutionDetails", "candidate_row", "method_text"]
