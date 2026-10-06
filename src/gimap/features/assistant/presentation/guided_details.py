"""The details of one fit in the Results tab: shown for the row selected in a table, only when asked.

A table gives one line per peak (GIWAXS) or per model solution (GISAXS); **Fit details** below it —
closed until opened, then following the selected row — gives everything behind that line:

* **a peak**: the points the Gaussian was fitted to, the fit and its local background drawn together;
  every parameter with its error; how it was fitted (weights, window, error scaling) and found (the
  search thresholds); its warnings; where on the detector it was measured;
* **a model solution**: every parameter of every component and the global ones (background,
  resolution), the quality numbers, the fit settings (q range, points, halves, start of D, screen,
  algorithm, evaluations, time), its warnings, and **Show in Fitting** to continue from it.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from src.gimap.app.presentation.components import CurvePlot
from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.theme import theme_manager

from .guided_text import ScaledPixmapLabel, label
from .guided_words import words

FIT_PLOT_HEIGHT = 190
NAME_WIDTH = 64
"""The narrowest a name of Fit details wraps to (px): a few words or Chinese characters per line."""
NM_PER_A = 10.0
COMPONENT_VALUES = (("R", "R", "nm"), ("sigma_R", "σR / R", ""), ("h", "h", "nm"), ("sigma_h", "σh / h", ""),
                    ("D", "D", "nm"), ("sigma_D", "σD / D", ""), ("weight", "weight", ""), ("amplitude", "amplitude", ""))
GLOBAL_VALUES = (("background", "constant background", ""), ("resolution_amplitude", "resolution amplitude", ""),
                 ("sigma_Res", "σ_Res", "nm⁻¹"), ("nu_Res", "ν_Res", ""))


def _number(value, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(value) else f"{value:.{digits}g}"


def _plus_minus(value, error, unit: str = "", digits: int = 5) -> str:
    text = _number(value, digits)
    if error is not None and _number(error) != "—":
        text += f" ± {_number(error, 2)}"
    return f"{text} {unit}".strip()


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
        self.method = label("", self, role="muted")
        layout.addWidget(self.method)
        self.warnings = label("", self)
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
        pairs = [
            ("q", _plus_minus(peak.get("q"), fit.get("q_err"), "Å⁻¹")),
            ("d = 2π/q", _plus_minus(peak.get("d_A"), fit.get("d_err_A"), "Å")),
            ("FWHM", _plus_minus(peak.get("fwhm"), fit.get("fwhm_err"), "Å⁻¹", 4)),
            ("height above background", _number(fit.get("height"))),
            ("area", _number(fit.get("area"))),
            ("background at the peak", _number(fit.get("background"))),
            ("background slope", _number(fit.get("slope"))),
            ("signal / noise", f"{_number(peak.get('snr'), 3)} σ"),
            ("χ²ᵣ of the fit", _number(fit.get("reduced_chi2"), 3)),
            ("points fitted", str(fit.get("points", "—"))),
            ("fit window", "—" if not fit.get("window") else
             f"{_number(fit['window'][0])}–{_number(fit['window'][1])} Å⁻¹"),
            ("crystallite size", "—" if size is None else ("≥ " if peak.get("size_is_lower_bound") else "") + f"{size:.3g} nm"),
            ("out-of-plane / in-plane", _number(peak.get("ratio_out_in"), 3)),
        ]
        self.values.set_values(pairs)
        search = report.get("peak_search") or {}
        self.method.setText(tr(
            "{model}, least squares weighted by the Poisson error of each bin; errors × √χ²ᵣ when χ²ᵣ > 1. Found in "
            "the radial I(q) of the whole detector: a SNIP background (window {window} Å⁻¹), at least {snr}σ and "
            "{height} % above it. Size: Scherrer, L = 2π·0.9 / FWHM (instrument broadening not removed)."
        ).format(model=tr(fit.get("model") or "Gaussian on a local linear background"),
                 window=_number(search.get("background_window"), 3), snr=_number(search.get("min_snr"), 3),
                 height=_number(100 * float(search.get("min_relative_height") or 0.02), 3)))
        caveat = str(peak.get("caveat") or "")
        self.warnings.setText("⚠ " + tr(caveat) if caveat else tr("No warnings for this peak."))
        ring = min(report.get("rings") or (), key=lambda item: abs(item["q"] - peak["q"]), default=None)
        shadowed = ring.get("shadowed") if ring and abs(ring["q"] - peak["q"]) <= peak.get("fwhm", 0.0) else ()
        self._picture(self.where, [{"q": peak["q"], "shadowed": shadowed or (), "label": f"q = {peak['q']:.4g} Å⁻¹"}],
                      map_width)

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
        "workflow": "native_v5", "rank": solution.get("rank", index + 1), "combination": solution.get("model"),
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
                    pairs.append((f"{prefix}: {name}" if key == "R" else name, f"{_number(component[key])} {unit}".strip()))
        for key, name, unit in GLOBAL_VALUES:
            value = (solution.get("globals") or {}).get(key)
            if value is not None:
                pairs.append((name, f"{_number(value)} {unit}".strip()))
        pairs += [("χ² (weighted)", _number(solution.get("chi2"))), ("log RMSE", _number(solution.get("log_rmse"))),
                  ("converged", tr("yes") if solution.get("converged") else tr("no")),
                  ("evaluations", _number(solution.get("evaluations"), 6)),
                  ("time", "—" if solution.get("seconds") is None else f"{solution['seconds']:.1f} s")]
        self.values.set_values(pairs)
        low, high = (fit.get("q_range_inv_angstrom") or [math.nan, math.nan])[:2]
        screen = tr("the single particle families screened") if solution.get("screen") != "user_composition" else tr(
            "the composition you gave")
        start = fit.get("distance_start_nm")
        self.settings.setText(tr(
            "{points} points, |qy| {low}–{high} Å⁻¹ ({low_nm}–{high_nm} nm⁻¹); {screen}; D started at {start}. "
            "{algorithm}. Sizes in nm; σ/value are relative size spreads; the weight is the share of the fitted "
            "amplitude, not a volume fraction."
        ).format(points=fit.get("points", "—"), low=_number(low), high=_number(high),
                 low_nm=_number(NM_PER_A * low), high_nm=_number(NM_PER_A * high),
                 screen=screen, start=tr("the in-plane spacing, {d} nm").format(d=_number(start)) if start else tr("free"),
                 algorithm=words(solution.get("algorithm")) or tr("Bounded least squares")))
        notes = list(solution.get("warnings") or ()) + ([] if solution.get("converged") else ["not converged"])
        self.warnings.setText("\n".join("⚠ " + tr(note) for note in notes) if notes else tr("No warnings for this solution."))
        self.show_button.setEnabled(self.allowed and candidate_row(fit, self._index) is not None)

    def _show_in_fitting(self) -> None:
        row = candidate_row(self._fit, self._index)
        if row is not None and self.allowed:
            self.showRequested.emit(row)


__all__ = ["PeakDetails", "SolutionDetails", "candidate_row"]
