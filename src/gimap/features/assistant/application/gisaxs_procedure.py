"""The GISAXS part of the standard procedure: cut, symmetry, halves, spacing, fit.

After the geometry and the frames (shared with GIWAXS): the horizontal cut at the Yoneda band
(or a question when there is none), the beam-centre column moved to the symmetry axis, the
halves chosen with their reason, the length scale 2π/q* of a side maximum or shoulder of I(qy)
(a length the curve shows, not necessarily the particle spacing), and — when a fitter is
available — Predict parameters on the curve: the most probable solution is reported with its
model probability and intervals. Competing compositions prompt a model choice. Solutions of the
reported composition are checked against each other's size intervals.
Each step is a tool call, so the GUI shows it and the AI could do the same. Like the GIWAXS run, the
report always says what its numbers are (q kinematic, uncertainties statistical only:
``giwaxs_decisions``), and a reported fit far from its curve (χ²ᵣ above Fitting's poor-fit limit),
with a value at a bound of its range, or with a sphere too large for the form factor at q_z = 0 (``qzR``) is a point
to check.
"""

from __future__ import annotations

from typing import Optional

from src.gimap.shared import fit_names

from ..domain.notes import substrate_from_notes
from ..domain.substrate_optics import substrate_record
from .gisaxs_choice import (
    INTERVAL_KEYS,
    LENGTH_SCALE,
    LINEAR_TERMS,
    POOR_CHI2,
    QZ_R_HINT,
    bounds_text,
    chance,
    chosen,
    compare_same_model,
    component_estimate,
    component_number,
    interface_caveat,
    interval,
    model_probability,
    poor_fit,
    rivals,
    sizes,
    weight_text,
    zero_terms,
)
from .gisaxs_report import cut_locations
from .gisaxs_tools import MAX_SYMMETRY_SHIFT_PX
from .giwaxs_decisions import flat_field_decision, refraction_decision, uncertainty_decision
from .pipeline_roll import choose_frame_angles
from .ring_frames import NO_FIT

PREDICTED = (
    "the most probable solution of Predict parameters on {curve}: GIMaP's network proposes compositions and values, "
    "each solution is refined by least squares and weighted by its posterior probability; "
    "χ²ᵣ = Σ((I − model)²/σ²)/(N − number of free values) over the fitted points{points}; "
    "probabilities are relative to the compositions that were tried"
)
"""Why the first solution is the one the run reports."""
INTERVALS = "{values} (refined value [68 % posterior interval])"
INTERVALS_BIC = ("{values} (refined value [68 % interval from the least-squares errors: the posterior's sampling did "
                 "not converge])")
INTERVALS_WHY = (
    "the posterior of Predict parameters given the σ of the points, sampled around the solution; the geometry and "
    "the choice of model are not in it"
)
"""The intervals of the reported solution's sizes (a decision of their own)."""
AT_ZERO = "{names} at 0, the non-negativity limit"
AT_ZERO_WHY = (
    "the particle amplitudes, the background and the beam-tail amplitude cannot go below 0: a term at 0 is held there "
    "by that limit, not by the curve, and is not counted among the values at a bound of their range, which are the "
    "non-linear ones"
)
BEAM_TAIL_OFF = "; with Tail A = 0 the beam tail is absent, so Tail w and Tail ν do not change the curve"
MORE_CAVEATS = "background, beam tail and resolution: {notes}"
MORE_CAVEATS_WHY = "reported by the fit; these are not values of the particles"
"""The ``fit caveat`` of a reported solution whose linear terms ended at 0 (``gisaxs_choice.zero_terms``)."""
AT_BOUND = "The reported solution ({model}) has {bounds}: those values are not determined by the curve."
AT_BOUND_HINT = (
    "Refine the model you expect in Fitting (Send to Fitting), with the value at the bound fixed or its range changed."
)
"""A reported solution with a non-linear value at a bound of its range (``at_bounds``)."""
POOR_FIT = (
    "The reported solution ({model}) has χ²ᵣ = {chi2}, above {limit}: it misses the curve by far more than the σ of "
    "its points, so none of the compositions Predict parameters tried describes this cut, and its R and D are not "
    "measurements."
)
POOR_FIT_HINT = (
    "Look at the fit over the curve (the Results tab, fit_curve.csv): Bragg rings of a powder or a calibrant, a cut "
    "off the Yoneda band or a second population give such a misfit. Refine a model you expect in Fitting."
)
"""A converged reported solution with χ²ᵣ above ``POOR_CHI2`` (the limit of Fitting's poor fit)."""
PROBABLE = "The curve alone does not decide the model. Model probabilities: {models}."
BIC_DECISION = "Predict parameters on {curve} compared the solutions by BIC weight after evidence sampling failed."
BIC_RIVALS = "The curve alone does not decide the model. Model BIC weights: {models}."
PROBABLE_HINT = "Choose the particle shape you expect and refine that model in Fitting; compare the fitted curves."
OVERLAP = "{n} solutions of this model with overlapping 68 % intervals."
DIFFERENT_VALUES = "Two solutions of the same model give {name} values of {first_value} and {second_value}."
VALUES_UNDECIDED = "The curve does not decide between these values."
VALUES_HINT = "In Fitting, refine from each start and compare the fitted curves and parameter values."
SPACING = "length scale"
"""The decision about 2π/q* (the report's key)."""
UNCORRELATED_D_NM = 250.0
"""D beyond this: no correlation peak in a GISAXS q range (a structure factor that is flat there)."""
SAME_D = 0.15
"""A fitted D within 15 % of 2π/q* is named in the model question (a coincidence of lengths, not a proof)."""


def _distance(solution: dict) -> Optional[float]:
    components = solution.get("components") or []
    return float(components[0]["D"]) if components and components[0].get("D") is not None else None


def _solution_text(solution: dict, solutions=()) -> str:
    """Formatted once, from the fit record (the tool's answer is rounded to 4 significant digits: 48.3484 → 48.35,
    which ``.3g`` then made 48.4 while the table said 48.3). D only for a solution with a paracrystal."""
    said = chance(solution, solutions)
    return f"{solution['model']}: {sizes(solution)}, χ²ᵣ = {solution['chi2']:.3g}" + (f", {said}" if said else "")


def _interval_text(solution: dict) -> str:
    """The refined R, h and D of every component with their 68 % intervals, named as everywhere else
    (``R (1) 3.06 [3.02, 3.23] nm, D (1) 49.4 [37.8, 65.0] nm``; one population: ``R 9.44 [9.12, 9.68] nm``; "" when
    the fit gave no interval)."""
    many = len(solution.get("components") or ()) > 1
    parts = []
    for number, component in enumerate(solution.get("components") or ()):
        for key in INTERVAL_KEYS:
            found = interval(solution, key, number) if component.get(key) is not None else None
            if found is not None:
                best = component_estimate(solution, key, number)  # an interval always holds its refined value (rounding apart)
                parts.append(f"{fit_names.display_name(f'{key}_{number + 1}', many)} "
                             + fit_names.interval_text(best, min(found[0], best), max(found[1], best), "nm"))
    return ", ".join(parts)


SYMMETRY_FAR = (
    "The calibration does not match this frame: the symmetry axis of the horizontal cut is at x {x} px, {shift} px "
    "from the calibrated beam centre. A calibration of this measurement is off by a few pixels at most, so the "
    "refined centre was not applied and every qy is that of the calibrated centre."
)
SYMMETRY_FAR_DECISION = ("the symmetry axis found (x {x} px) is {shift} px away, more than a calibration of this "
                         "measurement can be off: not applied")
SYMMETRY_FAR_HINT = ("Give the calibration of this measurement (--calibration); if it is the right one, check the "
                     "horizontal cut: its halves should mirror each other about the beam.")
SYMMETRY_FAILED = ("The symmetry axis of the horizontal cut could not be found ({reason}): the beam-centre column is "
                   "the calibration's, not checked against the frame.")
SYMMETRY_FAILED_HINT = ("Check that the calibration belongs to this frame and that the cut has measured pixels on "
                        "both sides of the beam.")


class GisaxsProcedureMixin:
    """Needs ``catalog``, ``options``, ``progress``, ``_call``, ``_decide``, ``_attention`` and ``_incidence`` (the αi
    given or noted, ``None`` when none was: ``StandardPipeline``)."""

    def _analyse_gisaxs(self) -> None:
        results = self.catalog.results
        symmetry = self._call("refine_beam_center_symmetry", {"max_shift_px": MAX_SYMMETRY_SHIFT_PX})
        if symmetry is not None:
            # The tool's answer is rounded to 4 significant digits (597.263 → 597.3): the decision states the
            # centre applied, from the record the tool kept at full precision.
            symmetry = {**symmetry, **(results.gisaxs.get("symmetry") or {})}
            shift = float(symmetry.get("shift_px", 0.0))
            if symmetry.get("not_applied"):
                self._decide("beam centre", f"kept at x {symmetry['initial_x_px']:.2f} px", SYMMETRY_FAR_DECISION.format(
                    x=f"{symmetry['x_px']:.2f}", shift=f"{shift:+.0f}"))
                self._attention("calibration", SYMMETRY_FAR.format(shift=f"{shift:+.0f}", x=f"{symmetry['x_px']:.2f}"),
                                "calibration", SYMMETRY_FAR_HINT)
            else:
                self._decide(
                    "beam centre", f"x {symmetry['initial_x_px']:.2f} → {symmetry['x_px']:.2f} px ({shift:+.2f} px)"
                    if symmetry.get("changed") else "already on the symmetry axis",
                    "the horizontal cut must be symmetric about qy = 0; the direct beam itself is hidden by the beam stop",
                )
            if abs(shift) > 20 and not symmetry.get("not_applied"):
                self._attention(
                    "beam centre",
                    f"The symmetry axis is {shift:+.0f} px from the calibrated centre: check the calibration or "
                    "the sample alignment.", None, "Look at the horizontal cut: its halves should mirror each other.",
                )
        else:  # never plain OK on a centre nobody could check
            self._attention("beam centre", SYMMETRY_FAILED.format(reason=self._run.steps[-1]["summary"]), None,
                            SYMMETRY_FAILED_HINT)
        # The angle record belongs to the final calibrated beam centre, including any x refinement.
        if not getattr(self, "_transmission_frame", None):
            choose_frame_angles(self)
        status = results.status or {}
        self._decide(**refraction_decision(status.get("geometry"), "gisaxs", given=self._incidence is not None,
                                          incidence_measurement=status.get("incidence_measurement")))
        self._decide(**uncertainty_decision(status.get("error_model"), "gisaxs", corrections=status.get("corrections")))
        flat = flat_field_decision(status)
        if flat is not None:
            self._decide(**flat)
        self._horizontal_cut(status.get("gisaxs") or {})
        halves = self._call("choose_halves")
        if halves is not None:
            self._call("set_halves", {"side": halves["recommended"]})
            self._decide("halves", halves["recommended"], halves["reason"])
        self._spacing()
        if getattr(self, "_transmission_frame", None):
            self._decide("fit", "not run", NO_FIT)
        elif self.options.fit and self.catalog.fitter is not None:
            self._fit()
        elif self.catalog.fitter is None:
            self._decide("fit", "not run here", "no fitter in this session: Send to Fitting fits the prepared curve")

    def _horizontal_cut(self, cuts: dict) -> None:
        source = cuts.get("horizontal_source")
        # The rows the cut uses, both ends included (as Analyze's Cuts card says them).
        where = cut_locations(cuts, self.catalog.results.status)[0]
        if source == "yoneda":
            self._decide(
                "horizontal cut", f"at the Yoneda band, αf = {cuts.get('yoneda_alpha_f_deg'):.3f}° ({where})",
                "the diffuse maximum just above the sample horizon, where the in-plane structure is strongest",
            )
        elif source == "horizon":
            self._attention(
                "Yoneda band",
                "No Yoneda band was found above the horizon; the horizontal cut sits just above it.",
                "incidence_deg",
                "Check αi (the horizon moves with it), or drag the yellow band on the image to the Yoneda row.",
            )
        elif source == "series":
            self._decide("horizontal cut", f"cut of the series ({where})", "the cut of one frame, held for every frame")
        else:
            self._decide("horizontal cut", f"set by hand ({where})", "kept as chosen")

    def _spacing(self) -> None:
        found = self._call("in_plane_spacing")
        if found is None:
            return
        spacing = found.get("spacing")
        if spacing is None:
            self._decide(
                SPACING, "no side maximum",
                "I(qy) has no maximum or shoulder away from qy = 0 above the noise: the cut gives no length scale "
                "2π/q* in the q range",
            )
        elif spacing["kind"] == "maximum":  # the decision says where; the reason says what it is, once
            self._decide(
                SPACING, f"2π/q* = {spacing['distance_nm']:.3g} nm at the peak of the cut (|qy| = {spacing['q']:.4g} Å⁻¹)",
                LENGTH_SCALE,
            )
        else:
            self._decide(
                SPACING, f"2π/q* = {spacing['distance_nm']:.3g} nm at a shoulder of the cut (|qy| = {spacing['q']:.4g} "
                "Å⁻¹), no resolved peak",
                f"{LENGTH_SCALE}; a form factor or the beam tail makes a shoulder too",
            )

    def _fit(self) -> None:
        # The time grows with the points and the compositions tried (``pipeline_progress.FIT_TAKES``): no one figure.
        self.progress("fit_horizontal_cut: predicting the particles of I(qy) (some seconds to a few minutes)…")
        options = getattr(self, "options", None)
        substrate = getattr(options, "substrate", None)
        try:
            if substrate is None:
                substrate = substrate_from_notes(getattr(options, "notes", ""))
            if substrate is not None:
                substrate = substrate_record(substrate)
        except ValueError as exc:
            self._attention("substrate", str(exc), "--substrate", "Give one material or a complete optical description.")
            return
        outcome = (self._call("fit_horizontal_cut", {"substrate": substrate}) if substrate is not None else
                   self._call("fit_horizontal_cut"))
        if outcome is None:
            return
        # The fit as the tool kept it, at full precision: every number below is formatted once from it.
        fit = self.catalog.results.gisaxs.get("fit") or {}
        solutions = fit.get("solutions") or []
        if not solutions:
            return
        if fit.get("substrate_metadata_warning"):
            self._decide("substrate", "file optics unused", fit["substrate_metadata_warning"])
        if fit.get("substrate_level") == 4:
            self._decide("model fit", "at the fourth level", fit["sphere_model"])
        elif fit.get("substrate_level") == 2:
            self._decide("model fit", "at the second level",
                         f"Born term at the cut's q_z = {fit['cut_qz']['q_z_inv_nm']:.4g} nm⁻¹")
        elif fit.get("substrate_level") == 1:
            self._decide("model fit", "at the first level (q_z = 0)", "the angles of the cut are not known")
        _index, best = chosen(fit)
        over = fit_names.points_text(best.get("points") or fit.get("fitted_points"))
        by_errors = fit_names.weight_key(best) == "bic_weight"
        reason = BIC_DECISION if by_errors else PREDICTED
        overlap, differences = compare_same_model(solutions, best)
        decision = _solution_text(best, solutions)
        if overlap > 1:
            decision += ". " + OVERLAP.format(n=overlap)
        self._decide("fit", decision, reason.format(curve=fit.get("curve"), points=f" ({over})" if over else ""))
        ranges = _interval_text(best)
        if ranges:  # the refined values with their intervals; from the errors when the sampling did not converge
            self._decide("fit intervals", (INTERVALS_BIC if by_errors else INTERVALS).format(values=ranges),
                         "reported by the fit" if by_errors else INTERVALS_WHY)
        for template, values in fit_names.diagnostic_lines(best.get("diagnostics") or {}, best.get("ess")):
            self._decide("fit", template.format(**values), "reported by the fit")
        for warning in best.get("warnings") or ():
            self._decide("fit caveat", warning, "reported by the fit")
        if best.get("more_warnings"):  # one line for the values that are not the particles'
            self._decide("fit caveat", MORE_CAVEATS.format(notes="; ".join(best["more_warnings"])), MORE_CAVEATS_WHY)
        if bounds_text(best):
            self._decide("fit caveat", bounds_text(best), "a value on the edge of its range is not determined by the curve")
        zeros = zero_terms(best)  # linear terms at their non-negativity limit: said, not counted as a bound
        if zeros:
            self._decide("fit caveat", AT_ZERO.format(names=" and ".join(zeros)),
                         AT_ZERO_WHY + (BEAM_TAIL_OFF if LINEAR_TERMS[-1][1] in zeros else ""))
        distance = _distance(best)
        if distance is not None and distance >= UNCORRELATED_D_NM:
            self._decide(
                "fit caveat", f"D = {component_number(best, 'D', 0)} nm: this solution has no interparticle correlation in the q range",
                "a structure factor that is flat here; D itself is not determined by the curve",
            )
        others = rivals(solutions, best)
        if others:
            self._model_question([best, *others], solutions)
        if differences:
            details = [DIFFERENT_VALUES.format(name=name, first_value=first, second_value=second)
                       for name, first, second in differences]
            self._attention("values", " ".join([*details, VALUES_UNDECIDED]), None, VALUES_HINT)
        if bounds_text(best):
            self._attention("fit", AT_BOUND.format(model=best["model"], bounds=bounds_text(best)), None, AT_BOUND_HINT)
        if not best.get("converged", True):
            self._attention(
                "fit", "The best fit did not converge.", None,
                "Open the curve in Fitting (Send to Fitting) and refine it with a chosen model.",
            )
        caveat = interface_caveat(best, fit.get("cut_qz"))
        if caveat:  # fitted at q_z = 0 while q_z·R of the cut is beyond what that treatment holds for
            self._attention("fit", caveat, None, QZ_R_HINT)
        grazing = [flag for flag in best.get("flags") or () if flag.split(":", 1)[0] in fit_names.GRAZING_FLAGS
                   and not flag.startswith("qzR:")]
        for note in fit_names.sentences(grazing, len(best.get("components") or ()) > 1):
            self._attention("fit", note, None, "")
        if poor_fit(best):  # converged, but far from the curve: Fitting's poor-fit limit
            self._attention("fit", POOR_FIT.format(model=best["model"], chi2=f"{best['chi2']:.3g}", limit=f"{POOR_CHI2:g}"),
                            None, POOR_FIT_HINT)

    def _model_question(self, probable: list[dict], solutions: list[dict]) -> None:
        """Name the competing compositions and keep the spacing comparison tied to their representative fits."""
        spacing = self.catalog.results.gisaxs.get("spacing")
        agree = [
            item for item in probable
            if spacing and _distance(item) and abs(_distance(item) - spacing["distance_nm"]) <= SAME_D * spacing["distance_nm"]
        ]
        hint = ""
        if agree:
            hint += (f"Solutions whose D is within {SAME_D:.0%} of 2π/q* of the cut: "
                     + "; ".join(_solution_text(item, solutions) for item in agree)
                     + " (2π/q* is a length scale, not necessarily the particle spacing). ")
        listed = ", ".join(f"{item['model']} ({weight_text(model_probability(item))})" for item in probable)
        template = BIC_RIVALS if fit_names.weight_key(probable[0]) == "bic_weight" else PROBABLE
        self._attention("model", template.format(models=listed), None, hint + PROBABLE_HINT)

    def _gisaxs_report(self) -> Optional[dict]:
        results = self.catalog.results
        status = results.status or {}
        if not results.gisaxs and status.get("measurement") != "gisaxs":
            return None
        halves = results.gisaxs.get("halves")
        return {
            "cuts": status.get("gisaxs"),
            "symmetry": results.gisaxs.get("symmetry"),
            "halves": None if halves is None else {
                "side": halves.side, "fit_side": halves.fit_side, "reason": halves.reason,
                "coverage": halves.coverage, "reach": halves.reach, "mismatch": halves.mismatch,
            },
            "spacing": results.gisaxs.get("spacing"),
            "fit": results.gisaxs.get("fit"),
            "curves": [item for item in status.get("curves") or () if item.get("key") in ("horizontal", "vertical")],
        }


__all__ = ["AT_BOUND", "AT_ZERO", "GisaxsProcedureMixin", "POOR_FIT", "POOR_FIT_HINT", "PROBABLE"]
