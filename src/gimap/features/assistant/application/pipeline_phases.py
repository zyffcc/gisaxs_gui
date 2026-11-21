"""The GIWAXS procedure's phase steps: the phases given, the assignment of the rings, the pole figure of each ring.

All three go through Analyze's Phases window (``AnalysisWorkbench.set_phases``, ``assign_peaks``, ``pole_figure``),
so the run reports the numbers the window shows. Without those methods (another workbench) nothing changes.
"""

from __future__ import annotations

import math
from typing import Optional

POLE_MODEL = "gaussian"
POLE_RULE = (
    "a Gaussian about the direction χ0 plus an isotropic fraction, fitted to the folded, Lorentz-corrected I(χ) of "
    "each analysed ring minus the radial background, on the measured bins up to χ = 85°; χ0 is 0° when the highest "
    "maximum of I(χ) lies below 20°, 90° when above 70°, fitted otherwise; the oriented fraction and the f of the "
    "fit include the missing wedge as the model fills it (Physics and Methods 7.11)"
)
POOR_FIT = 10.0
"""A pole-figure fit above this χ²_red is flagged: one Gaussian and an isotropic part do not describe the ring."""


def pole_direction(ring) -> Optional[float]:
    """χ0 of the automatic pole-figure fit from the highest maximum of the ring's I(χ) (``None``: fitted)."""
    maxima = getattr(ring, "maxima", None) or ()
    if not maxima:
        return 0.0
    top = float(maxima[0].get("chi", 0.0))
    return 0.0 if top < 20.0 else 90.0 if top > 70.0 else None
ASSIGN_RULE = (
    "the nearest powder-ring position |q| of the phases given ({phases}; Analyze ▸ Phases or --phase) within "
    "max(3 σ(q0), FWHM/2); q is kinematic unless αc is given, so rings near the horizon can be shifted by up to "
    "k sin αi; other reflections within the window are listed as alternatives (report.json); the nearest match is "
    "an assignment, not a proof of the phase"
)


def _short(value: Optional[float], digits: int) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def assignment_text(item: dict) -> str:
    """``q = 0.9973 → MAPbI3 110 (Δq = −0.0040 Å⁻¹, −41 σ)`` of one assignment record."""
    if not item.get("assigned"):
        return f"q = {item['q']:.4f}: none within ±{item['tolerance']:.4f} Å⁻¹"
    sigma = item.get("delta_sigma")
    in_sigma = f", {sigma:+.1f} σ" if sigma is not None and math.isfinite(sigma) else ""
    return (f"q = {item['q']:.4f} → {item['phase']} {item['hkl']} (q_hkl = {item['q_hkl']:.4f}, "
            f"Δq = {item['delta_q']:+.4f} Å⁻¹{in_sigma})")


class PhaseStepsMixin:
    """Needs ``catalog``, ``options``, ``_run``, ``_decide`` and ``_attention`` (``StandardPipeline``)."""

    _assignment: Optional[dict] = None
    _poles: list

    def _workbench_with(self, name: str):
        workbench = getattr(self.catalog, "workbench", None)
        return workbench if workbench is not None and hasattr(workbench, name) else None

    def _phases_given(self) -> None:
        """The phases of the options replace those of the Phases window (``--phase``)."""
        self._poles = []
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

    def _assign(self, peaks) -> None:
        workbench = self._workbench_with("assign_peaks")
        if workbench is None or not peaks:
            return
        found = workbench.assign_peaks([
            {"q": peak.q, "q_err": peak.q_err if math.isfinite(peak.q_err) else None, "fwhm": peak.fwhm}
            for peak in peaks
        ]) or {}
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
        self._decide("phase assignment", decision, ASSIGN_RULE.format(phases=names))

    def _pole(self) -> None:
        """The pole figure of the ring just analysed (its I(χ) window is still set)."""
        workbench = self._workbench_with("pole_figure")
        rings = self.catalog.results.rings
        search = self.catalog.results.peak_searches.get("radial")
        if workbench is None or not rings:
            return
        low, high = rings[-1].q_window
        centre = 0.5 * (low + high)
        background = search.background_at(centre) if search is not None else None
        try:
            record = workbench.pole_figure(background=background, model=POLE_MODEL, chi0_deg=pole_direction(rings[-1]))
        except ValueError as error:
            record = {"ok": False, "message": str(error)}
        if not self._poles:
            self._decide("pole figure", "fitted for every analysed ring", POLE_RULE)
        self._poles.append((centre, record))

    def _phase_rows(self, report: dict) -> dict:
        """The assignment in the peak rows and the pole figures in the ring rows of ``report``."""
        if self._assignment:
            report["phase_assignment"] = {key: self._assignment[key] for key in ("phases", "rule", "assignments")}
            items = self._assignment.get("assignments") or []
            for row in report.get("peaks") or []:
                match = min(items, key=lambda item: abs(item["q"] - row.get("q", math.inf)), default=None)
                if match is not None and abs(match["q"] - row.get("q", math.inf)) < 1e-9 + 1e-6 * match["q"]:
                    row.update({key: match.get(key) for key in ("phase", "hkl", "q_hkl", "delta_q", "delta_sigma")})
                    row["alternatives"] = match.get("alternatives") or []
        for row in report.get("rings") or []:
            pole = min(getattr(self, "_poles", None) or [], key=lambda entry: abs(entry[0] - row.get("q", math.inf)),
                       default=None)
            if pole is not None and abs(pole[0] - row.get("q", math.inf)) <= 2e-3 * pole[0]:  # row q: 4 digits
                row["pole_figure"] = pole_summary(pole[1])
        return report


def pole_summary(record: dict) -> dict:
    """The values of a pole-figure record a report keeps (full precision)."""
    values, errors = record.get("values") or {}, record.get("errors") or {}
    keys = ("oriented_fraction", "fwhm_deg", "chi0_deg", "herman", "herman_oriented", "order_2d")
    summary = {"ok": bool(record.get("ok")), "model": record.get("model"), "message": record.get("message") or ""}
    for key in keys:
        if key in values:
            summary[key] = values[key]
            summary[f"{key}_err"] = errors.get(key)
    for key in ("herman_measured", "reduced_chi2", "points", "weighted_coverage", "fitted_up_to_chi_deg",
                "lorentz_corrected_input"):
        summary[key] = record.get(key)
    background = record.get("background")
    summary["background"] = background.get("value") if isinstance(background, dict) else background
    return summary


def pole_line(pole: dict) -> str:
    """One line of report.md for a ring's pole figure."""
    if not pole.get("ok"):
        return f"pole figure: no fit ({pole.get('message') or 'no result'})"

    def pm(key: str, digits: int) -> str:
        error = pole.get(f"{key}_err")
        text = _short(pole.get(key), digits)
        return text + (f" ± {error:.{digits}f}" if error is not None and math.isfinite(error) and error > 0 else "")

    return (f"pole figure (Gaussian + isotropic): oriented fraction {pm('oriented_fraction', 3)}, "
            f"FWHM {pm('fwhm_deg', 1)}°, χ0 = {pm('chi0_deg', 1)}°, f = {pm('herman', 3)} (missing wedge filled; "
            f"from the measured bins {_short(pole.get('herman_measured'), 3)}), χ²_red = "
            f"{_short(pole.get('reduced_chi2'), 1)}" + (f"; {pole['message']}" if pole.get("message") else "")
            + ("; poor fit (χ²_red above 10): one Gaussian and an isotropic part do not describe this ring"
               if (pole.get("reduced_chi2") or 0.0) > POOR_FIT else ""))


def assignment_lines(report: dict) -> list[str]:
    """The *Phase assignment* section of report.md (empty without phases)."""
    found = report.get("phase_assignment")
    if not found:
        return []
    names = ", ".join(phase.get("name", "?") for phase in found.get("phases") or ())
    lines = [f"Phases given: {names}. Rule: {found.get('rule')}.", ""]
    for item in found.get("assignments") or []:
        others = "; ".join(f"{other['phase']} {other['hkl']} ({other['delta_q']:+.4f})"
                           for other in item.get("alternatives") or [])
        lines.append(f"- {assignment_text(item)}" + (f"; also within the window: {others}" if others else ""))
    return lines


__all__ = ["PhaseStepsMixin", "assignment_lines", "assignment_text", "pole_direction", "pole_line", "pole_summary"]
