"""The GIWAXS procedure's phase steps: the phases given, the assignment of the rings, the pole figure of each ring.

All three go through Analyze's Phases window (``AnalysisWorkbench.set_phases``, ``assign_peaks``, ``pole_figure``),
so the run reports the numbers the window shows. Without those methods (another workbench) nothing changes.
"""

from __future__ import annotations

import math
from typing import Optional

from src.gimap.shared.orientation_notes import coverage_warning, measured_f_text
from src.gimap.shared.radial_peaks import orientation_peak_rows

from .pipeline_assignment import assignment_lines, assignment_text, axis_text
from .pipeline_progress import PipelineStopped

POLE_MODEL = "gaussian"
POLE_RULE = (
    "A Gaussian in polar angle plus isotropic density is fitted to the measured bins after the exit-angle cut. "
    "The strongest maximum sets χ0 to 0° below 20° or 90° above 70°. Otherwise χ0 is fitted. "
    "The fitted f includes the model in the missing wedge. 1 − φ is conditional on the background b."
)


def pole_direction(ring) -> Optional[float]:
    """χ0 of the automatic pole-figure fit from the highest maximum of the ring's I(χ) (``None``: fitted)."""
    maxima = getattr(ring, "maxima", None) or ()
    if not maxima:
        return 0.0
    top = float(maxima[0].get("chi", 0.0))
    return 0.0 if top < 20.0 else 90.0 if top > 70.0 else None


ASSIGN_RULE = (
    "Nearest powder-ring position of the phases given ({phases}) within max(3 σ(q0), FWHM/2). "
    "Multiple reflection families in the window make the assignment ambiguous."
)


def _short(value: Optional[float], digits: int) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


class PhaseStepsMixin:
    """Needs ``catalog``, ``options``, ``_run``, ``_decide`` and ``_attention`` (``StandardPipeline``)."""

    _assignment: Optional[dict] = None
    _crystal_orientation: Optional[list] = None
    _reported_pole: bool = False

    def _workbench_with(self, name: str):
        workbench = getattr(self.catalog, "workbench", None)
        return workbench if workbench is not None and hasattr(workbench, name) else None

    def _phases_given(self) -> None:
        """The phases of the options replace those of the Phases window (``--phase``)."""
        self._reported_pole = False
        self._assignment = None
        self._crystal_orientation = None
        phases = tuple(getattr(self.options, "phases", ()) or ())
        workbench = self._workbench_with("set_phases")
        if not phases or workbench is None:
            return
        try:
            result = workbench.set_phases(list(phases))
        except (OSError, ValueError) as error:
            self._attention("phases", f"A phase could not be read: {error}.", "phase",
                            "Give a CIF with _cell_length_a, _b, _c (and its symmetry operations).")
            return
        names = ", ".join(phase.get("name", "?") for phase in (result or {}).get("phases") or ())
        self._decide("phases", names or "none", "given (--phase, phases=)")

    def _assign(self, search) -> None:
        workbench = self._workbench_with("assign_peaks")
        if workbench is None:
            return
        found = workbench.assign_peaks(orientation_peak_rows(search)) or {}
        if not found.get("phases"):
            return
        self._assignment = found
        items = found.get("assignments") or []
        assigned = [item for item in items if item.get("assigned")]
        missing = [f"{item['q']:.4f}" for item in items if not item.get("assigned")]
        decision = f"{len(assigned)} of {len(items)} peaks: " + "; ".join(assignment_text(item) for item in assigned)
        if missing:
            decision += f"; not assigned: {', '.join(missing)}"
        names = ", ".join(phase.get("name", "?") for phase in found["phases"])
        self._decide("phase assignment", decision, ASSIGN_RULE.format(phases=names) + " " + axis_text(found.get("q_axis") or {}))

    def _pole(self) -> None:
        ring = self.catalog.results.rings[-1]
        pole = ring.pole_figure
        if not pole:
            return
        if not self._reported_pole:
            self._decide("pole figure", "fitted for every analysed ring", POLE_RULE)
            self._reported_pole = True
        reasons = pole.get("quality_reasons") or []
        q = sum(ring.q_window) / 2
        warning = coverage_warning(pole.get("systematics") or {})
        if ring.herman is not None and warning:
            self._attention("orientation factor", f"q = {q:.4g} Å⁻¹. {warning}", None,
                            "Check the angular coverage and the range endpoints in Phases ▸ Pole Figure.")
        if reasons:
            self._attention("pole figure", f"At q = {q:.4g} Å⁻¹ the pole fit is not determined ({', '.join(reasons)}).",
                            None, "Inspect the profile and background in Phases ▸ Pole Figure.")

    def _fit_crystal_orientation(self) -> None:
        workbench = self._workbench_with("crystal_orientation")
        if workbench is None or not self._assignment:
            return

        def check_cancel():
            if self.stop is not None and self.stop.is_set():
                raise PipelineStopped("crystal orientation")

        check_cancel()
        self.progress("Fitting crystal orientation")
        try:
            self._crystal_orientation = workbench.crystal_orientation(self._assignment, check_cancel)
        except ValueError as error:
            self._attention("crystal orientation", str(error), None, "Inspect the phases and ring assignments.")
            return
        for phase in self._crystal_orientation:
            if not phase["determined"]:
                self._attention("crystal orientation", f"{phase['phase']}: " + ", ".join(phase["reasons"]),
                                None, "Inspect the assigned rings in Phases ▸ Pole Figure.")

    def _phase_rows(self, report: dict) -> dict:
        """The assignment in the peak rows and the pole figures in the ring rows of ``report``."""
        if self._assignment:
            report["phase_assignment"] = dict(self._assignment)
            if self._crystal_orientation is not None:
                report["crystal_orientation"] = self._crystal_orientation
            items = self._assignment.get("assignments") or []
            for row in report.get("peaks") or []:
                match = min(items, key=lambda item: abs(item["q"] - row.get("q", math.inf)), default=None)
                if match is not None and abs(match["q"] - row.get("q", math.inf)) < 1e-9 + 1e-6 * match["q"]:
                    row.update({key: match.get(key) for key in ("phase", "hkl", "q_hkl", "delta_q", "delta_sigma",
                                                                 "relative_d", "ambiguous", "tolerance", "tentative")})
                    row["alternatives"] = match.get("alternatives") or []
        return report


def fit_ring_pole(workbench, ring, background=None) -> dict:
    """Use the Phases fit so the tool, automatic report and dialog share one set of numbers."""
    fit = getattr(workbench, "pole_figure", None)
    if fit is None:
        return {}
    try:
        record = fit(background=background, model=POLE_MODEL, chi0_deg=pole_direction(ring))
    except ValueError as error:
        record = {"ok": False, "message": str(error)}
    return pole_summary(record)


def ring_step_summary(q, ring):
    """Orientation result used in the tool response and the report's steps."""
    pole = ring.pole_figure
    verdict = "pole fit not determined" if pole.get("quality_reasons") else ring.reason or ring.texture
    if ring.herman is None:
        return f"q = {q:.3g}: {verdict}"
    systematics = pole.get("systematics") or {}
    warning = coverage_warning(systematics)
    measured = measured_f_text(ring.herman, ring.herman_err, systematics.get("range"), isotropic=ring.herman_isotropic)
    return f"q = {q:.3g}: {warning or verdict.rstrip('.') + '.'} Measured f = {measured}"


def pole_summary(record: dict) -> dict:
    """The values of a pole-figure record a report keeps (full precision)."""
    values, errors = record.get("values") or {}, record.get("errors") or {}
    keys = ("oriented_fraction", "fwhm_deg", "chi0_deg", "herman", "herman_oriented", "order_2d")
    summary = {"ok": bool(record.get("ok")), "model": record.get("model"), "message": record.get("message") or ""}
    for key in keys:
        if key in values:
            summary[key] = values[key]
            summary[f"{key}_err"] = errors.get(key)
    for key in ("herman_measured", "herman_measured_err", "herman_isotropic", "reduced_chi2", "points", "weighted_coverage", "fitted_up_to_chi_deg",
                "lorentz_corrected_input"):
        summary[key] = record.get(key)
    summary["at_bound"] = list(record.get("at_bound") or ())
    summary["quality_reasons"] = list(record.get("quality_reasons") or ())
    if not summary["ok"] and not summary["quality_reasons"]:
        summary["quality_reasons"].append(summary["message"] or "fit did not converge")
    for key in ("systematics", "converged", "min_exit_deg", "exit_angle_source", "polarization"):
        if key in record:
            summary[key] = record[key]
    background = record.get("background")
    summary["background"] = background.get("value") if isinstance(background, dict) else background
    return summary


def pole_estimate(pole: dict, key: str) -> str:
    value, error = pole.get(key), pole.get(f"{key}_err")
    if value is None or not math.isfinite(value):
        return "—"
    estimate = f"{value:#.3g}".removesuffix(".")
    if error is not None and math.isfinite(error) and error > 0:
        estimate += " ± " + f"{error:#.3g}".removesuffix(".")
    return estimate


def pole_line(pole: dict) -> str:
    """One line beside the measured f, with a verdict on the fit before its estimates."""
    reasons = pole.get("quality_reasons") or []
    lead = ("not determined (" + ", ".join(reasons) + "). " if reasons else "")
    if not pole.get("ok"):
        return "pole figure: " + (lead.rstrip() or "not determined")

    return (f"pole figure (Gaussian + isotropic): {lead}f = {pole_estimate(pole, 'herman')}, "
            f"oriented fraction 1 − φ = {pole_estimate(pole, 'oriented_fraction')}, FWHM {pole_estimate(pole, 'fwhm_deg')}°, "
            f"χ0 = {_short(pole.get('chi0_deg'), 1)}°, χ²_red = {_short(pole.get('reduced_chi2'), 1)}. "
            "1 − φ is conditional on the background b.")


__all__ = ["PhaseStepsMixin", "assignment_lines", "assignment_text", "pole_direction", "pole_line", "pole_summary"]
