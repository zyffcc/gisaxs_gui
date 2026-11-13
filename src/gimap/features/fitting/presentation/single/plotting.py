"""Fitting plot layers: measured points, total model, optional additive contributions and residuals.

Component visibility affects only presentation. The default limits follow the data and total;
component tails can extend beyond them. Explicitly including components in auto range opts in
on Reset. Switching individual curves leaves a manual zoom intact.
"""
from __future__ import annotations

import re

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QAction

from src.gimap.app.presentation.components.curve_plot import LineStyle
from src.gimap.app.presentation.i18n import tr

from ...application.single_fit import evaluate, left_out_points, residuals
from .points import DATA_COLOR, measured_series

MODEL_POINTS = 600
OUTSIDE_COLOR, MODEL_COLOR, LEFT_OUT_COLOR = "#94a3b8", "#f97316", "#dc2626"
# Distinct from data blue and total orange; also readable on the dark plot and in light exports.
COMPONENT_COLORS = ("#0d9488", "#8b5cf6", "#b08728", "#b85e9d", "#609f42", "#628ba8")


def component_label(name: str) -> str:
    """Human labels only: native keys and exported scientific records stay untouched."""
    if name == "background_B":
        return tr("constant background")
    if name == "low_q_term":
        return tr("low-q term")
    if name == "resolution peak":
        return tr("beam-tail term")
    match = re.fullmatch(r"pop(\d+)_(sphere|random_cylinder|vertical_cylinder)", name)
    if match:
        family = {"sphere": "Sphere", "random_cylinder": "Random cylinder",
                  "vertical_cylinder": "Vertical cylinder"}[match[2]]
        return f"{match[1]}: {tr(family)}"
    number, separator, family = name.partition(": ")
    return f"{number}: {tr(family)}" if separator and number.isdigit() else tr(name)


def component_style(name: str, index: int) -> tuple[str, LineStyle]:
    if name in ("constant background", "background_B"):
        return "#7c8b9b", LineStyle(width=1.3, style=Qt.DotLine, z=0)
    if name in ("resolution peak", "low_q_term"):
        return "#b08728", LineStyle(width=1.4, style=Qt.DashDotLine, z=0)
    return COMPONENT_COLORS[index % len(COMPONENT_COLORS)], LineStyle(width=1.4, style=Qt.DashLine, z=0)


class FitPlotMixin:
    def setup_component_plot(self) -> None:
        self._component_actions = {}
        self._component_choices = {}
        self._component_menu_labels = ()
        self.terms_check.toggled.connect(lambda _on: self._redraw(preserve_view=True))
        self.component_range_action.toggled.connect(self._component_range_changed)

    def _sync_component_menu(self, parts) -> None:
        labels = tuple((name, component_label(name)) for name in parts)
        if labels != self._component_menu_labels:
            for action in self._component_actions.values():
                self.component_menu.removeAction(action)
                action.deleteLater()
            self._component_actions = {}
            self._component_choices = {name: self._component_choices.get(name, True) for name in parts}
            for name, label in labels:
                action = QAction(label, self.component_menu)
                action.setCheckable(True)
                action.setChecked(self._component_choices[name])
                action.toggled.connect(lambda on, name=name: self._component_chosen(name, on))
                self.component_menu.insertAction(self.component_separator, action)
                self._component_actions[name] = action
            self._component_menu_labels = labels
        self.terms_check.setEnabled(bool(parts))
        self.component_options_button.setEnabled(bool(parts) and self.terms_check.isChecked())

    def _component_chosen(self, name: str, on: bool) -> None:
        self._component_choices[name] = on
        self._redraw(preserve_view=True)

    def _component_range_changed(self, _on: bool) -> None:
        self._redraw(preserve_view=True)
        self.plot.reset_view()

    def _refresh_residual_tooltip(self) -> None:
        data = self._data()
        if data is None:
            return
        tip = (tr("fit1d residual: (I − model) / effective σ") if self.session.native_is_current()
               else self._residuals_tip(data.sigma is not None))
        self.residual_plot.setToolTip(tip)
        self.residual_plot.plot.getAxis("left").setToolTip(tip)

    def _redraw(self, *, preserve_view: bool = False) -> None:
        session = self.session
        full, data = (None, None) if session.curve is None else (session.all_points(), self._data())
        curves, colors, markers, styles, bounds = [], [], [], [], []
        parts = {}
        if full is not None and full.q.size:
            inside = np.zeros(full.q.size, bool) if data is None else np.isin(full.q, data.q)
            left_out = left_out_points(full.q, session.excluded) if session.excluded \
                else np.zeros(full.q.size, bool)  # as the fit takes the names (the nearest point, half a step)
            outside = ~inside & ~left_out
            if outside.any():
                curves.append((tr("outside the range"), full.q[outside], full.intensity[outside]))
                colors.append(OUTSIDE_COLOR)
                markers.append(True)
            if left_out.any():
                curves.append((tr("left out"), full.q[left_out], full.intensity[left_out]))
                colors.append(LEFT_OUT_COLOR)
                markers.append("x")
            for name, q, intensity, color in measured_series(session, data):  # both halves on |q|: two colours
                curves.append((tr(name), q, intensity))
                colors.append(color)
                markers.append(True)
            q_model = np.geomspace(max(full.q.min(), 1e-6), full.q.max(), MODEL_POINTS)
            try:
                if session.native_is_current():
                    native = session.native_result["curves"]
                    q_model = np.asarray(native["q_nm_inv"], float)
                    total, parts = native["model"], native.get("components", {})
                else:
                    total, parts = evaluate(session.model, q_model, parts=True)
            except (ValueError, FloatingPointError):
                total, parts = None, {}
            if total is not None:
                label = ("fit1d v2 model" if session.native_is_current() else "manual starting model"
                         if session.native_result is not None and session.result is None else "model")
                curves.append((tr(label), q_model, total))
                colors.append(MODEL_COLOR)
                markers.append(False)
                styles = [LineStyle(z=2) for _ in curves[:-1]] + [LineStyle(width=2.3, z=3)]
                bounds = [True] * len(curves)
                self._sync_component_menu(parts)
                if self.terms_check.isChecked():
                    for index, (name, values) in enumerate(parts.items()):
                        if not self._component_choices.get(name, True):
                            continue
                        curves.append((component_label(name), q_model, values))
                        color, style = component_style(name, index)
                        colors.append(color)
                        markers.append(False)
                        styles.append(style)
                        bounds.append(self.component_range_action.isChecked())
        self._sync_component_menu(parts)
        self.plot.set_curves(curves, colors, markers=markers, line_styles=styles, auto_range=bounds,
                             preserve_view=preserve_view)
        self._draw_band(full)
        if not preserve_view:  # appearance-only changes do not redraw/re-range the linked residual axes
            self._draw_residuals(data)

    def _draw_band(self, full) -> None:
        if full is None or not full.q.size:
            self.plot.hide_window()
            return
        low, high = self.session.q_range or (float(full.q.min()), float(full.q.max()))
        self.plot.show_window(low, high)

    def _draw_residuals(self, data) -> None:
        if self._draw_fit1d_residuals():
            return
        if data is None or not data.q.size:
            self.residual_plot.set_curves([])
            return
        try:
            values = residuals(self.session.model, data)
        except (ValueError, FloatingPointError):
            values = np.full(data.q.size, np.nan)
        weighted = data.sigma is not None
        self.residual_plot.set_labels("|q| (nm⁻¹)", "Δ/σ" if weighted else "Δ ln I")  # short: the plot is low
        tip = self._residuals_tip(weighted)  # with σ, how the points are weighted (``curve_info``)
        self.residual_plot.setToolTip(tip)
        self.residual_plot.plot.getAxis("left").setToolTip(tip)
        self.residual_plot.set_curves([("0", data.q[[0, -1]], np.zeros(2)), (tr("residual"), data.q, values)],
                                      [OUTSIDE_COLOR, DATA_COLOR], markers=[False, True])

