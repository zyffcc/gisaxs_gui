"""GISAXS tools of the catalog: symmetry axis, cuts, halves, and a physical fit of I(qy).

The settings (beam centre, cuts, halves) go through ``_tracked`` like every other settings
change, so they can be previewed, applied and undone. ``fit_horizontal_cut`` needs a
``CurveFitter`` (the Fitting feature's numerical fit, injected by the composition root).
"""

from __future__ import annotations

import math
from typing import Optional

from ..domain import Profile, choose_halves, find_peaks, fit_curve, spacing_from_peaks
from .models import ToolInputError, ToolOutcome

MIN_SHIFT_PX = 0.05
TOP_SOLUTIONS = 3
SPACING_MIN_POINTS = 10
"""A side maximum closer to qy = 0 than this many bins belongs to the central (Yoneda / beam-stop) peak."""
SPACING_WINDOW = 0.02
"""Background window of the GISAXS peak search (Å⁻¹): a small-angle curve spans only a few 0.1 Å⁻¹."""


def _solution(row: dict) -> dict:
    """What a person reads of one fit solution (units: nm; q in nm⁻¹ in the fit itself)."""
    return {
        "rank": row.get("rank"),
        "model": row.get("combination"),
        "components": [
            {"type": component["type"], "weight": round(float(component.get("weight", 0.0)), 3),
             "amplitude": float(component.get("amplitude", math.nan)),
             **{key: round(float(value), 4) for key, value in component["params"].items()}}
            for component in row.get("components") or ()
        ],
        "globals": {key: float(value) for key, value in (row.get("global_params") or {}).items() if value is not None},
        "chi2": round(float(row.get("best_chi2_weighted", math.nan)), 3),
        "log_rmse": round(float(row.get("best_log_rmse", math.nan)), 4),
        "converged": bool(row.get("converged")),
        "warnings": list(row.get("warnings") or ()),
        "evaluations": row.get("nfev"),
        "seconds": None if row.get("seconds") is None else round(float(row["seconds"]), 2),
        "algorithm": row.get("algorithm"),
        "screen": row.get("conditions_source"),
    }


def _curve(row: dict) -> dict:
    """The fitted intensity of one solution on its own q grid (nm⁻¹)."""
    return {"q_inv_nm": [abs(float(value)) for value in row.get("display_q", ())],
            "intensity": [float(value) for value in row.get("display_fit", ())]}


def _native(row: dict) -> dict:
    """What Fitting needs to draw a solution (signed q in nm⁻¹): the fitted points, their σ and both fits."""
    record = {key: [float(value) for value in row.get(source, ())] for key, source in (
        ("q_inv_nm", "native_q"), ("fit", "native_fit"), ("observed", "observed"), ("sigma", "sigma"),
        ("display_q", "display_q"), ("display_fit", "display_fit"))}
    return {**record, "units": dict(row.get("unit_contract") or {}), "source": row.get("best_source")}


class GisaxsToolsMixin:
    """Needs ``workbench``, ``results``, ``fitter``, ``_tracked`` and ``_ok``."""

    fitter = None

    def _gisaxs_status(self) -> dict:
        status = self.workbench.status()
        self.results.status = status
        if status.get("measurement") != "gisaxs" or not status.get("gisaxs"):
            raise ToolInputError("This needs a GISAXS reduction (set_measurement_mode gisaxs, and a geometry).")
        return status

    def _tool_refine_beam_center_symmetry(self) -> ToolOutcome:
        status = self._gisaxs_status()
        found = self.workbench.symmetry_center()
        shift = float(found["x_px"]) - float(found["initial_x_px"])
        record = {**found, "shift_px": shift}
        self.results.gisaxs["symmetry"] = record
        if abs(shift) < MIN_SHIFT_PX:
            return self._ok({**record, "changed": False}, f"already symmetric (shift {shift:+.2f} px)")
        centre_y = float((status.get("geometry") or {}).get("beam_center_px", [0, 0])[1])
        outcome = self._tracked("set_beam_center", {"x_px": float(found["x_px"]), "y_px": centre_y})
        if outcome.is_error:
            return outcome
        return self._ok(
            {**record, "changed": True},
            f"centre x {found['initial_x_px']:.2f} → {found['x_px']:.2f} px (asymmetry {found['loss_before']:.3g} → {found['loss_after']:.3g})",
        )

    def _tool_set_beam_center(self, x_px: float, y_px: float) -> ToolOutcome:
        status = self.workbench.set_beam_center(float(x_px), float(y_px))
        self.results.status = status
        return self._ok(status, f"beam centre ({x_px:.2f}, {y_px:.2f}) px")

    def _tool_set_gisaxs_cuts(
        self,
        automatic: bool = False,
        horizontal_row: Optional[float] = None,
        horizontal_half_height: Optional[float] = None,
        vertical_column: Optional[float] = None,
        vertical_half_width: Optional[float] = None,
    ) -> ToolOutcome:
        values = (horizontal_row, horizontal_half_height, vertical_column, vertical_half_width)
        if automatic:
            values = (None, None, None, None)
        elif all(value is None for value in values):
            raise ToolInputError("Give automatic=true or at least one of the cut positions and widths.")
        status = self.workbench.set_gisaxs_cuts(*values)
        self.results.status = status
        cuts = status.get("gisaxs") or {}
        return self._ok(status, f"horizontal cut rows {cuts.get('horizontal_rows')}, vertical columns {cuts.get('vertical_columns')}")

    def _horizontal(self):
        curve = self.workbench.curve("horizontal")
        if curve is None:
            raise ToolInputError("There is no horizontal cut (a GISAXS reduction with a geometry is needed).")
        return curve

    def _tool_choose_halves(self) -> ToolOutcome:
        curve = self._horizontal()
        choice = choose_halves(curve.x, curve.y, curve.pixels)
        self.results.gisaxs["halves"] = choice
        data = {
            "recommended": choice.side, "fit_side": choice.fit_side, "reason": choice.reason,
            "coverage": choice.coverage, "reach_inv_angstrom": choice.reach, "median_mismatch": choice.mismatch,
        }
        return self._ok(data, f"{choice.side}: {choice.reason}")

    def _tool_set_halves(self, side: str) -> ToolOutcome:
        status = self.workbench.set_halves(side)
        self.results.status = status
        return self._ok(status, f"halves: {side}")

    def _fit_input(self, side: Optional[str]):
        curve = self._horizontal()
        if side is None:
            choice = self.results.gisaxs.get("halves") or choose_halves(curve.x, curve.y, curve.pixels)
            side = choice.fit_side
        return (side, *fit_curve(curve.x, curve.y, curve.sigma, curve.pixels, side))

    def _tool_in_plane_spacing(self, side: Optional[str] = None, min_points: int = SPACING_MIN_POINTS,
                               background_window: Optional[float] = None) -> ToolOutcome:
        side, x, y, sigma, note = self._fit_input(side)
        search = find_peaks(Profile.of(x, y, sigma), background_window=background_window or SPACING_WINDOW)
        spacing = spacing_from_peaks(search.peaks, q_min=max(1, int(min_points)) * float(search.step or 0.0))
        record = None if spacing is None else {
            "q": spacing.q, "distance_nm": spacing.distance_nm, "snr": spacing.snr, "kind": spacing.kind,
        }
        self.results.gisaxs["spacing"] = record
        data = {
            "side": side, "curve": note, "spacing": record,
            "peaks": [{"q": peak.q, "d_nm": peak.d / 10.0, "fwhm": peak.fwhm, "snr": peak.snr, "flags": peak.flags}
                      for peak in search.peaks],
        }
        if spacing is None:
            return self._ok(data, "no side maximum or shoulder away from qy = 0")
        return self._ok(data, f"{spacing.kind} at |qy| = {spacing.q:.4g} Å⁻¹ → D ≈ {spacing.distance_nm:.3g} nm")

    def _tool_fit_horizontal_cut(self, side: Optional[str] = None, components=None, distance_nm=None) -> ToolOutcome:
        if self.fitter is None:
            raise ToolInputError("No curve fitter is available in this session: use Send to Fitting in the GUI.")
        side, x, y, sigma, note = self._fit_input(side)
        if distance_nm is None:
            distance_nm = (self.results.gisaxs.get("spacing") or {}).get("distance_nm")
        rows = self.fitter(
            x, y, sigma, components=tuple(components or ()), distance_nm=distance_nm, cancelled=self.cancelled,
        )
        if not rows:
            raise ToolInputError("The fit found no solution.")
        solutions = [_solution(row) for row in rows]
        best = rows[0]
        self.results.gisaxs["fit"] = {
            "side": side, "curve": note, "points": int(x.size), "q_range_inv_angstrom": [float(x.min()), float(x.max())],
            "distance_start_nm": distance_nm,
            "solutions": solutions,
            "best_curve": _curve(best),
            "curves": [_curve(row) for row in rows],
            "native": [_native(row) for row in rows],
            "data": {"q_inv_angstrom": x.tolist(), "intensity": y.tolist(), "sigma": sigma.tolist()},
        }
        head = solutions[0]
        params = head["components"][0] if head["components"] else {}
        summary = (
            f"best: {head['model']}, R = {params.get('R', math.nan):.3g} nm, D = {params.get('D', math.nan):.3g} nm, "
            f"χ² = {head['chi2']:.3g}"
        )
        return self._ok({"side": side, "curve": note, "points": int(x.size), "solutions": solutions[:TOP_SOLUTIONS]}, summary)


__all__ = ["GisaxsToolsMixin"]
