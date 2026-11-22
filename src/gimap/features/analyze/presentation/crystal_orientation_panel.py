"""The phase-aware joint fit in Analyze's Pole Figure tab."""

from __future__ import annotations

from threading import Event

import numpy as np
from PyQt5.QtCore import QSignalBlocker, Qt

from src.gimap.app.presentation.components.curve_plot import LineStyle
from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.shared.numbers import format_number

from .indexing_texts import percent_text, ring_text

CRYSTAL_MODE = "crystal_orientation"
CRYSTAL_LABEL = "Crystal orientation (from phase)"
FAMILY_COLORS = ("#148776", "#9764bd", "#bb7524", "#cc4772", "#578b31", "#5b80b9")


def _axes(directions):
    if directions is None:
        return "—"
    return " + ".join("(" + " ".join(str(i) for i in axis) + ")" for axis in directions) or tr("Isotropic")


def _film_summary(optics):
    if not optics.get("available"):
        return tr("Film optical constants unavailable")
    constants = dict(delta=f"{optics['delta']:.3g}", beta=f"{optics['beta']:.3g}")
    if optics.get("model") == "frame corrected with refracted waves":
        return trf("Frame corrected for refracted film waves · δ {delta} · β {beta}", **constants)
    if optics.get("model") == "refracted waves":
        return trf("Refracted film waves · δ {delta} · β {beta}", **constants)
    # Older fit records contain only the reciprocal exit transmission.
    return trf("Film exit transmission |T|² from CIF composition · δ {delta} · β {beta}", **constants)


def _summary(phase, ring):
    status = tr("Orientation determined") if phase["determined"] else tr("Orientation not determined")
    reasons = [tr(reason) for reason in phase.get("reasons", [])]
    populations = [
        trf("{plane} ∥ substrate · mosaic FWHM {width}° · {fraction}% of phase",
            plane=_axes([p["axis"]]), width=format_number(p["fwhm_deg"], p["fwhm_err_deg"]),
            fraction=percent_text(p["fraction"], p["fraction_err"]))
        for p in phase["populations"]
    ]
    weights = tr("Free reflection weights") if phase.get("family_weights") == "free" else tr("Kinematic reflection weights")
    near_random = phase.get("near_random", [])
    fraction_label = ("Resolved oriented {oriented}% · isotropic including near-random {isotropic}%"
                      if near_random else "Total oriented {oriented}% · isotropic {isotropic}%")
    stats = dict(delta=format_number(phase.get("delta_bic")), joint=format_number(phase["reduced_chi2"]),
                 ring=format_number(ring["reduced_chi2"]))
    if near_random:
        comparison = trf("Profile-model ΔBIC {delta} · joint χ²red {joint} · ring χ²red {ring}", **stats)
    else:
        comparison = trf("Runner-up {axes} · ΔBIC {delta} · joint χ²red {joint} · ring χ²red {ring}",
                         axes=_axes(phase.get("runner_up_axes")), **stats)
    lines = [
        " · ".join((f"{phase['phase']} · {status}", *reasons)),
        *populations,
        *[trf("Near-random · {reason}", reason=tr(p["reason"])) for p in near_random],
        trf(fraction_label,
            oriented=percent_text(phase["oriented_fraction"], phase["oriented_fraction_err"]),
            isotropic=percent_text(phase["fraction_iso"], phase["fraction_iso_err"])),
        comparison,
        trf("Crystal Hermans f {crystal} · measured ring f {ring}",
            crystal=format_number(phase["herman_crystal"]),
            ring=format_number(ring["herman_measured"], ring.get("herman_measured_err")))
        + f" · {weights}",
        _film_summary(phase.get("film_optics", {})),
    ]
    if phase.get("search_settings", {}).get("noncollinear_poles_available") is False:
        lines.append(tr("Only one pole direction constrains this phase. Two distinct fibre axes cannot be established."))
    return "\n".join(lines)


def _plot(plot, phase, ring):
    chi = np.asarray(ring["chi_deg"], dtype=float)
    measured = np.asarray(ring["measured"], dtype=bool)
    used = np.asarray(ring.get("fit_mask", measured), dtype=bool)
    observed = np.asarray(ring["observed"], dtype=float)
    curves = [(tr("Measured N(χ)"), chi[used], observed[used])]
    colors = ["#4385d2"]
    styles = [LineStyle(z=3)]
    excluded = measured & ~used
    if excluded.any():
        curves.append((tr("not fitted"), chi[excluded], observed[excluded]))
        colors.append("#9ca3af")
        styles.append(LineStyle(z=3))
    markers = [True] * len(curves)
    curves.append((tr("Joint fit"), chi, np.asarray(ring["fitted"], dtype=float)))
    colors.append("#ea7526")
    styles.append(LineStyle(width=2.2, z=2))
    for i, family in enumerate(ring["components"]):
        label = f"{family['hkl']} ({100 * family['weight']:.1f}%)"
        curves.append((label, chi, np.asarray(family["values"], dtype=float)))
        colors.append(FAMILY_COLORS[i % len(FAMILY_COLORS)])
        styles.append(LineStyle(width=1.5, style=Qt.DashLine, z=2.5))
    plot.set_labels("χ (°)", tr("N(χ) = net intensity × sin χ"))
    low, high = ring["q_window"]
    plot.set_title(f"{phase['phase']} · q = {low:.3f}–{high:.3f} Å⁻¹")
    plot.legend.setColumnCount(1 if len(curves) <= 6 else 2)
    plot.set_curves(curves, colors, markers=markers + [False] * (len(curves) - len(markers)), line_styles=styles)


class CrystalOrientationMixin:
    def _setup_crystal_orientation(self):
        self._crystal_result = []
        self._crystal_key = None
        self._crystal_cancel = None
        self.crystal_ring_combo.currentIndexChanged.connect(self._show_crystal_ring)
        self.model_combo.currentIndexChanged.connect(self._model_chosen)
        self.chi_max_spin.valueChanged.connect(self._refresh_crystal_options)
        self.cancel_crystal_button.clicked.connect(self._cancel_crystal)
        self.finished.connect(self._cancel_crystal)

    def _crystal_mode(self):
        return self.model_combo.currentData() == CRYSTAL_MODE

    def _crystal_source(self):
        return (id(self._page.view_model.state.analysis), tuple(self.model.phases), self.chi_max_spin.value())

    def _refresh_crystal_options(self, *_args):
        index = self.model_combo.findData(CRYSTAL_MODE)
        if self.model.active and index < 0:
            self.model_combo.addItem(tr(CRYSTAL_LABEL), CRYSTAL_MODE)
        elif not self.model.active and index >= 0:
            if self._crystal_mode():
                self.model_combo.setCurrentIndex(0)
            self.model_combo.removeItem(index)
        if self._crystal_key is not None and self._crystal_key != self._crystal_source():
            self._cancel_crystal()
            self._crystal_key = None
            self._crystal_result = []
            self.crystal_ring_combo.clear()
            if self._crystal_mode():
                self._show_crystal_ring()
        if self._crystal_mode():
            self._crystal_controls()

    def _crystal_controls(self):
        crystal = self._crystal_mode()
        for widget in (self.direction_label, self.direction_combo, self.direction_spin,
                       self.background_label, self.background_check, self.background_spin):
            widget.setVisible(not crystal)
        for widget in (self.crystal_ring_label, self.crystal_ring_combo):
            widget.setVisible(crystal and self.crystal_ring_combo.count() > 0)
        self.cancel_crystal_button.setVisible(crystal and self._crystal_cancel is not None)
        self.fit_button.setEnabled(not crystal or self._crystal_cancel is None)
        self.fit_button.setText(tr("Fit Crystal Orientation") if crystal else tr("Fit Pole Figure"))
        self.export_button.setToolTip(tr("Export the displayed ring as CSV and the phase fits as JSON")
                                      if crystal else tr("The pole figure as CSV and the indexing record (.json) beside it"))
        if crystal:
            self.ring_label.setText(tr("Fit the assigned rings of each phase together, using each ring’s local background."))
            self.export_button.setEnabled(self._selected_crystal_ring() is not None)
        else:
            self.ring_label.setText(ring_text(self._page.view_model.state.analysis))
            self.export_button.setEnabled(True)

    def _model_chosen(self, *_args):
        self._crystal_controls()
        if self._crystal_mode():
            self._show_crystal_ring()
        else:
            self._cancel_crystal()
            self.pole_plot.legend.setColumnCount(1)
            if self._displayed_pole is not None:
                self._show_pole(self._displayed_pole)
            else:
                self._clear_crystal_plot()
                self.pole_plot.set_labels("χ (°)", tr("N(χ) ∝ crystallites per degree"))

    def _clear_crystal_plot(self):
        self.pole_plot.set_title("")
        self.pole_plot.set_curves([])
        self.result_label.clear()

    def _fit_crystal_orientation(self):
        from ..application.crystal_orientation import frame_crystal_orientation

        analysis = self._page.view_model.state.analysis
        if analysis is None or analysis.reduction is None:
            self.result_label.setText(tr("Open a GIWAXS frame with a geometry first."))
            return
        cancelled = Event()
        self._crystal_cancel = cancelled
        self._crystal_key = self._crystal_source()
        self._crystal_result = []
        self.crystal_ring_combo.clear()
        self._clear_crystal_plot()
        self._crystal_controls()
        self.result_label.setText(tr("Fitting crystal orientations …"))
        phases = tuple(self.model.phases)
        cap = self.chi_max_spin.value()
        frames = self._page.view_model.frames

        def check_cancel():
            if cancelled.is_set():
                raise InterruptedError("Cancelled")

        self._page.tasks.submit(
            CRYSTAL_MODE,
            lambda: frame_crystal_orientation(analysis, phases, chi_max=cap, frame_source=frames, check_cancel=check_cancel),
            on_done=self._crystal_ready, on_error=self._crystal_failed,
        )

    def _cancel_crystal(self, *_args):
        if self._crystal_cancel is not None:
            self._crystal_cancel.set()
            self._page.tasks.cancel(CRYSTAL_MODE)
            self._crystal_cancel = None
            self._crystal_controls()
            if self._crystal_mode():
                self.result_label.setText(tr("Cancelled"))

    def _crystal_failed(self, message, _details):
        self._crystal_cancel = None
        self._crystal_controls()
        self.result_label.setText(tr(message))

    def _crystal_ready(self, records):
        self._crystal_cancel = None
        if self._crystal_key != self._crystal_source():
            self._refresh_crystal_options()
            return
        self._crystal_result = records
        self._refresh_cell()
        analysis = self._page.view_model.state.analysis
        window = analysis.reduction.markers.get("chi_q_window")
        center = np.mean(window) if window is not None else None
        distances = []
        combo = self.crystal_ring_combo
        with QSignalBlocker(combo):
            combo.clear()
            for i, phase in enumerate(records):
                for j, ring in enumerate(phase.get("rings", [])):
                    combo.addItem(f"{phase['phase']} · {ring['q']:.3f} Å⁻¹", (i, j))
                    distances.append(abs(ring["q"] - center) if center is not None else 0)
                if not phase.get("rings"):
                    combo.addItem(phase["phase"], (i, -1))
                    distances.append(float("inf"))
            if distances:
                combo.setCurrentIndex(int(np.argmin(distances)))
        self._crystal_controls()
        self._show_crystal_ring()
        if not records:
            self.result_label.setText(tr("No assigned rings could be fitted."))

    def _selected_crystal_ring(self):
        selection = self.crystal_ring_combo.currentData()
        if selection is None or not self._crystal_result:
            return None
        phase, ring = selection
        return self._crystal_result[phase]["rings"][ring] if ring >= 0 else None

    def _show_crystal_ring(self, *_args):
        if not self._crystal_mode():
            return
        ring = self._selected_crystal_ring()
        self.export_button.setEnabled(ring is not None)
        if ring is None:
            if self._crystal_cancel is None:
                self._clear_crystal_plot()
                selected = self.crystal_ring_combo.currentData()
                if selected is not None:
                    reasons = self._crystal_result[selected[0]].get("reasons", [])
                    self.result_label.setText("\n".join([tr("No assigned rings could be fitted."), *map(tr, reasons)]))
            return
        phase = self._crystal_result[self.crystal_ring_combo.currentData()[0]]
        _plot(self.pole_plot, phase, ring)
        self.result_label.setText(_summary(phase, ring))
