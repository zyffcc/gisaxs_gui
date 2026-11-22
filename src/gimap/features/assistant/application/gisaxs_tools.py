"""GISAXS tools of the catalog: symmetry axis, cuts, halves, and Predict parameters on I(qy).

The settings (beam centre, cuts, halves) go through ``_tracked`` like every other settings
change, so they can be previewed, applied and undone. ``fit_horizontal_cut`` needs a
``CurveFitter`` (the Fitting feature's Predict parameters, injected by the composition root).
"""

from __future__ import annotations

import math
from typing import Optional

from src.gimap.shared import fit_names

from ..domain import Profile, choose_halves, find_peaks, fit_curve, spacing_from_peaks
from .gisaxs_choice import bounds_text, chance, probability, rivals, sizes
from .models import ToolInputError, ToolOutcome

MIN_SHIFT_PX = 0.0001
MAX_SYMMETRY_SHIFT_PX = 100.0
"""The bound the automatic analysis gives ``refine_beam_center_symmetry`` (``max_shift_px``): a symmetry axis
further than this from the calibrated beam-centre column is not applied. A calibration of the measurement is off by
a few pixels, so such a distance is the mark of a calibration that is not this frame's (or of a cut that is not
symmetric about the beam). The run says so and keeps the calibrated centre. The tool called without a bound moves
the centre whatever the distance, as before."""
TOP_SOLUTIONS = 3
SPACING_MIN_POINTS = 10
"""A side maximum closer to qy = 0 than this many bins belongs to the central (Yoneda / beam-stop) peak."""
SPACING_WINDOW = 0.02
"""Background window of the GISAXS peak search (Å⁻¹): a small-angle curve spans only a few 0.1 Å⁻¹."""
AT_BOUND_FLAG = " stopped at a bound of its range"
"""The end of the fit's warning for a value at a bound: ``at_bounds`` names a non-linear one more exactly (which
bound), the report names a linear term at 0 itself (``gisaxs_choice.zero_terms``)."""
LINEAR_NAMES = ("background", "res_amplitude", "Int")
"""How the fit names the linear terms in its warnings (``Int_1``: the amplitude of the first population)."""
INTERVAL_NAMES = ("median", "low68", "high68", "low95", "high95")
"""The posterior of a value as a row gives it: best, then these, then the least-squares error."""


SPHERE_MODEL = "effective interface (q_z = 0)"
"""The sphere model recorded when the angles of the cut are unavailable."""


def cut_qz(status: Optional[dict]) -> Optional[dict]:
    """The vacuum q_z of the horizontal cut, k(sin αi + sin αf) in nm⁻¹ (``q_z_inv_nm``), with the angles and λ it is
    from: αi and λ of the geometry, αf of the Yoneda band the cut is at. ``None`` when one of them is not known (a
    cut elsewhere, no αi): nothing is guessed."""
    status = status or {}
    geometry, cuts = status.get("geometry") or {}, status.get("gisaxs") or {}
    wavelength, alpha_i, alpha_f = (geometry.get("wavelength_A"), geometry.get("incidence_deg"),
                                    cuts.get("yoneda_alpha_f_deg"))
    if cuts.get("horizontal_source") != "yoneda" or None in (wavelength, alpha_i, alpha_f):
        return None
    wavelength, alpha_i, alpha_f = float(wavelength), float(alpha_i), float(alpha_f)
    if not (wavelength > 0 and 0 < alpha_i < 90 and 0 < alpha_f < 90):
        return None
    k = 2.0 * math.pi / (wavelength / 10.0)  # nm⁻¹
    return {"q_z_inv_nm": k * (math.sin(math.radians(alpha_i)) + math.sin(math.radians(alpha_f))),
            "alpha_i_deg": alpha_i, "alpha_f_deg": alpha_f, "wavelength_A": wavelength}


def _intervals(row: dict) -> dict:
    """The posterior intervals of a solution's values (``R_1``, ``D_2`` …; nm, nm⁻¹, spreads relative)."""
    found = {}
    for name, values in (row.get("posterior") or {}).items():
        try:
            numbers = [None if value is None else float(value) for value in list(values)[1:6]]
        except (TypeError, ValueError):
            continue
        if len(numbers) == len(INTERVAL_NAMES):
            keys = ("least_squares_value", *INTERVAL_NAMES[1:]) if fit_names.weight_key(row) == "bic_weight" else INTERVAL_NAMES
            found[str(name)] = dict(zip(keys, numbers))
    return found


def _warnings(row: dict) -> list[str]:
    """The fit's warnings; one for a value at a bound only when nothing else names it: not for a linear term, not
    when ``at_bounds`` lists the values at a bound."""
    kept = []
    for item in map(str, row.get("warnings") or ()):
        if item.endswith(AT_BOUND_FLAG):
            name = item[: -len(AT_BOUND_FLAG)]
            if row.get("at_bounds") or name.split("_")[0] in LINEAR_NAMES or name in LINEAR_NAMES:
                continue
        kept.append(item)
    return kept


def _notes(row: dict) -> tuple[list[str], list[str]]:
    """(the notes of the populations and of the solution, those of background, beam tail and resolution), as
    sentences with the Model table's names (``shared/fit_names``), from the row's ``flags``; a value at a bound is
    left to ``at_bounds`` and ``zero_terms`` as in ``_warnings``. A row without flags: its sentences as they come."""
    if row.get("flags") is None:
        return _warnings(row), [str(item) for item in row.get("more_warnings") or ()]
    kept = []
    for flag in map(str, row["flags"]):
        kind, names = fit_names.flag_names(flag)
        if kind == "at_bound" and (row.get("at_bounds") or any(
                fit_names.split_name(name)[0] in LINEAR_NAMES for name in names)):
            continue
        kept.append(flag)
    first, second = fit_names.split_flags(kept)
    many = len(row.get("components") or ()) > 1
    return fit_names.sentences(first, many), fit_names.sentences(second, many)


def _chi2(row: dict) -> float:
    """χ²ᵣ of the solution over the fitted points (``chi2_reduced``; a row without it: its ``best_chi2_weighted``)."""
    value = row.get("chi2_reduced", row.get("best_chi2_weighted", math.nan))
    return math.nan if value is None else float(value)


def _solution(row: dict) -> dict:
    """What a person reads of one solution (units: nm; q in nm⁻¹ in the fit itself). ``model``: the composition
    as Predict names it ("sphere (S) + upright cylinder"; "(S)": with a paracrystal), ``combination``: its particle
    types as Fitting takes them."""
    notes, more = _notes(row)
    return {
        "rank": row.get("rank"),
        "model": row.get("composition") or row.get("combination"),
        "combination": row.get("combination"),
        fit_names.weight_key(row): None if math.isnan(probability(row)) else round(probability(row), 4),
        "components": [
            {"type": component["type"], "weight": round(float(component.get("weight", 0.0)), 3),
             "amplitude": float(component.get("amplitude", math.nan)),
             **{key: round(float(value), 4) for key, value in component["params"].items()}}
            for component in row.get("components") or ()
        ],
        "globals": {key: float(value) for key, value in (row.get("global_params") or {}).items() if value is not None},
        # χ²ᵣ over the fitted points, the number Fitting shows for the same solution; ``points``: how many those
        # are (fitted, given, below_window: below the network's q window, unusable).
        "chi2": round(_chi2(row), 3),
        "points": None if not row.get("points") else dict(row["points"]),
        "compared_by": row.get("compared_by"),
        "diagnostics": row.get("diagnostics") or {},
        "ess": row.get("ess"),
        "posterior_summary": row.get("posterior_summary") or {},
        "least_squares_values": row.get("least_squares_values") or {},
        "log_rmse": round(float(row.get("best_log_rmse", math.nan)), 4),
        "converged": bool(row.get("converged", True)),
        "warnings": notes,
        "more_warnings": more,  # of background, beam tail and resolution: said after the others, in one paragraph
        "flags": [str(flag) for flag in row.get("flags") or ()],
        # The values that ended on a bound of their range ({"parameter": "D", "bound": "upper", "component": 0}).
        "at_bounds": [dict(item) for item in row.get("at_bounds") or ()],
        "intervals": _intervals(row),
        "seconds": None if row.get("seconds") is None else round(float(row["seconds"]), 2),
        "algorithm": row.get("algorithm"),
        "screen": row.get("conditions_source"),
    }


def _brief(solution: dict) -> str:
    """Every population's R (and D), χ²ᵣ and the probability, as the report's decision says them."""
    said = chance(solution)
    return f"{solution['model']}, {sizes(solution)}, χ²ᵣ = {solution['chi2']:.3g}" + (f", {said}" if said else "")


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

    def _tool_refine_beam_center_symmetry(self, max_shift_px: Optional[float] = None) -> ToolOutcome:
        status = self._gisaxs_status()
        found = self.workbench.symmetry_center()
        shift = float(found["x_px"]) - float(found["initial_x_px"])
        record = {**found, "shift_px": shift}
        self.results.gisaxs["symmetry"] = record
        if abs(shift) < MIN_SHIFT_PX:
            return self._ok({**record, "changed": False}, f"already symmetric (shift {shift:+.2f} px)")
        if max_shift_px is not None and abs(shift) > float(max_shift_px):
            record["not_applied"] = True
            self.results.gisaxs["symmetry"] = record
            return self._ok(
                {**record, "changed": False},
                f"symmetry axis at x {found['x_px']:.2f} px, {shift:+.0f} px from the calibrated centre: not applied "
                f"(more than {float(max_shift_px):g} px; the calibration does not match this frame)",
            )
        centre_y = float((status.get("geometry") or {}).get("beam_center_px", [0, 0])[1])
        outcome = self._tracked("set_beam_center", {"x_px": float(found["x_px"]), "y_px": centre_y})
        if outcome.is_error:
            return outcome
        before = "unavailable" if found["loss_before"] is None else f"{found['loss_before']:.3g}"
        return self._ok(
            {**record, "changed": True},
            f"centre x {found['initial_x_px']:.2f} → {found['x_px']:.2f} px "
            f"(asymmetry {before} → {found['loss_after']:.3g})",
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
        return self._ok(data, f"{spacing.kind} of the cut at |qy| = {spacing.q:.4g} Å⁻¹ → 2π/q* = "
                              f"{spacing.distance_nm:.3g} nm (a length scale, not necessarily the particle spacing)")

    def _tool_fit_horizontal_cut(self, side: Optional[str] = None, components=None) -> ToolOutcome:
        if self.fitter is None:
            raise ToolInputError("No curve fitter is available in this session: use Send to Fitting in the GUI.")
        side, x, y, sigma, note = self._fit_input(side)
        cut = cut_qz(self.results.status)
        rows = self.fitter(x, y, sigma, components=tuple(components or ()), cancelled=self.cancelled,
                           cut_qz=None if cut is None else cut["q_z_inv_nm"])
        if not rows:
            raise ToolInputError("Predict parameters found no solution.")
        solutions = [_solution(row) for row in rows]  # most probable first: the first is the one reported
        if len(solutions) == 1 and components:  # one composition, given: its probability of 1 is no finding
            solutions[0]["alone"] = True
        self.results.gisaxs["fit"] = {
            "side": side, "curve": note, "points": int(x.size), "q_range_inv_angstrom": [float(x.min()), float(x.max())],
            "substrate_level": 2 if cut is not None else 1,
            "sphere_model": "Born term at the cut's q_z" if cut is not None else SPHERE_MODEL,
            "cut_qz": cut,
            # the points χ²ᵣ is of: fitted, given, below_window (below the network's q window), unusable
            "fitted_points": None if not rows[0].get("points") else dict(rows[0]["points"]),
            # what the fit says of the curve as it took it: an estimated σ, negative and zero intensities
            "curve_notes": [dict(item) for item in rows[0].get("curve_notes") or ()],
            "solutions": solutions,
            "chosen": 0,
            # What made the solutions: the proposal network by name and the SHA-256 of its weights, and the seed.
            "method": {"name": "Predict parameters", "network": rows[0].get("network"), "seed": rows[0].get("seed")},
            "best_curve": _curve(rows[0]),  # the reported (most probable) solution's curve: fit_curve.csv's I_fit
            "curves": [_curve(row) for row in rows],
            "native": [_native(row) for row in rows],
            "data": {"q_inv_angstrom": x.tolist(), "intensity": y.tolist(), "sigma": sigma.tolist()},
        }
        head = solutions[0]
        by_bic = fit_names.weight_key(head) == "bic_weight"
        lead = "highest BIC weight" if by_bic else "most probable"
        summary = f"{lead}: {_brief(head)}" + (f" ({bounds_text(head)})" if bounds_text(head) else "")
        others = rivals(solutions)
        if others:
            summary += ("; alternatives: " if by_bic else "; also probable: ") + "; ".join(
                f"rank {item['rank']}, {_brief(item)}" for item in others)
        return self._ok({"side": side, "curve": note, "points": int(x.size), "solutions": solutions[:TOP_SOLUTIONS],
                         "chosen": head}, summary)


__all__ = ["GisaxsToolsMixin", "SPHERE_MODEL", "cut_qz"]
