"""The GISAXS part of the standard procedure: cut, symmetry, halves, spacing, fit.

After the geometry and the frames (shared with GIWAXS): the horizontal cut at the Yoneda band
(or a question when there is none), the beam-centre column moved to the symmetry axis, the
halves chosen with their reason, the length scale 2π/q* of a side maximum or shoulder of I(qy)
(a length the curve shows, not necessarily the particle spacing), and — when a fitter is
available — Predict parameters on the curve: the most probable solution is reported with its
probability and intervals, and when another solution is probable too the person chooses the model.
Each step is a tool call, so the GUI shows it and the AI could do the same. Like the GIWAXS run, the
report always says what its numbers are (q kinematic, uncertainties statistical only:
``giwaxs_decisions``), and a reported fit far from its curve (χ²/N above Fitting's poor-fit limit) or
with a value at a bound of its range is a point to check.
"""

from __future__ import annotations

import math
from typing import Optional

from .gisaxs_choice import (
    ASK_PROBABILITY,
    INTERVAL_KEYS,
    LENGTH_SCALE,
    LINEAR_TERMS,
    POOR_CHI2,
    bounds_text,
    chosen,
    interval,
    poor_fit,
    probability,
    rivals,
    zero_terms,
)
from .gisaxs_report import cut_spans
from .giwaxs_decisions import refraction_decision, uncertainty_decision

PREDICTED = (
    "the most probable solution of Predict parameters on {curve}: GIMaP's network proposes compositions and values, "
    "each solution is refined by least squares and weighted by its posterior probability; χ²/N is the mean of r² "
    "over the points, with no degrees-of-freedom correction"
)
"""Why the first solution is the one the run reports."""
INTERVALS = "{values} (68 % of the posterior)"
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
BEAM_TAIL_OFF = "; with A = 0 the beam tail is absent, so σ_Res and ν_Res do not change the curve"
"""The ``fit caveat`` of a reported solution whose linear terms ended at 0 (``gisaxs_choice.zero_terms``)."""
AT_BOUND = "The reported solution ({model}) has {bounds}: those values are not determined by the curve."
AT_BOUND_HINT = (
    "Refine the model you expect in Fitting (Send to Fitting), with the value at the bound fixed or its range changed."
)
"""A reported solution with a non-linear value at a bound of its range (``at_bounds``)."""
POOR_FIT = (
    "The reported solution ({model}) has χ²/N = {chi2}, above {limit}: it misses the curve by far more than the σ of "
    "its points, so none of the compositions Predict parameters tried describes this cut, and its R and D are not "
    "measurements."
)
POOR_FIT_HINT = (
    "Look at the fit over the curve (the Results tab, fit_curve.csv): Bragg rings of a powder or a calibrant, a cut "
    "off the Yoneda band or a second population give such a misfit. Refine a model you expect in Fitting."
)
"""A converged reported solution with χ²/N above ``POOR_CHI2`` (the limit of Fitting's poor fit)."""
PROBABLE = "{n} solutions have a probability of at least {limit} ({solutions}): the curve alone does not decide the model."
PROBABLE_HINT = "Choose the particle shape you expect and refine that model in Fitting; compare the fitted curves."
"""Another solution is probable too (``gisaxs_choice.rivals``): the person chooses the model."""
SPACING = "length scale"
"""The decision about 2π/q* (the report's key)."""
UNCORRELATED_D_NM = 250.0
"""D beyond this: no correlation peak in a GISAXS q range (a structure factor that is flat there)."""
SAME_D = 0.15
"""A fitted D within 15 % of 2π/q* is named in the model question (a coincidence of lengths, not a proof)."""


def _distance(solution: dict) -> Optional[float]:
    components = solution.get("components") or []
    return float(components[0]["D"]) if components and components[0].get("D") is not None else None


def _solution_text(solution: dict) -> str:
    """Formatted once, from the fit record (the tool's answer is rounded to 4 significant digits: 48.3484 → 48.35,
    which ``.3g`` then made 48.4 while the table said 48.3). D only for a solution with a paracrystal."""
    component = (solution.get("components") or [{}])[0]
    distance = f", D = {component['D']:.3g} nm" if component.get("D") is not None else ""
    chance = probability(solution)
    return (
        f"{solution['model']}: R = {component.get('R', math.nan):.3g} nm{distance}, χ²/N = {solution['chi2']:.3g}"
        + ("" if math.isnan(chance) else f", probability {chance:.2f}")
    )


def _interval_text(solution: dict) -> str:
    """The 68 % intervals of R, h and D of every component (``R 9.1–9.8 nm, D 45–52 nm``; "" when none)."""
    many = len(solution.get("components") or ()) > 1
    parts = []
    for number, component in enumerate(solution.get("components") or ()):
        for key in INTERVAL_KEYS:
            found = interval(solution, key, number) if component.get(key) is not None else None
            if found is not None:
                parts.append(f"{key}{f' {number + 1}' if many else ''} {found[0]:.3g}–{found[1]:.3g} nm")
    return ", ".join(parts)


class GisaxsProcedureMixin:
    """Needs ``catalog``, ``options``, ``progress``, ``_call``, ``_decide``, ``_attention`` and ``_incidence`` (the αi
    given or noted, ``None`` when none was: ``StandardPipeline``)."""

    def _analyse_gisaxs(self) -> None:
        results = self.catalog.results
        status = results.status or {}
        cuts = status.get("gisaxs") or {}
        # What the numbers are, in every report (the GIWAXS run says the same in its own words).
        self._decide(**refraction_decision(status.get("geometry"), "gisaxs", given=self._incidence is not None))
        self._decide(**uncertainty_decision(status.get("error_model"), "gisaxs"))
        self._horizontal_cut(cuts)
        symmetry = self._call("refine_beam_center_symmetry")
        if symmetry is not None:
            # The tool's answer is rounded to 4 significant digits (597.263 → 597.3): the decision states the
            # centre applied, from the record the tool kept at full precision.
            symmetry = {**symmetry, **(results.gisaxs.get("symmetry") or {})}
            shift = float(symmetry.get("shift_px", 0.0))
            self._decide(
                "beam centre", f"x {symmetry['initial_x_px']:.2f} → {symmetry['x_px']:.2f} px ({shift:+.2f} px)"
                if symmetry.get("changed") else "already on the symmetry axis",
                "the horizontal cut must be symmetric about qy = 0; the direct beam itself is hidden by the beam stop",
            )
            if abs(shift) > 20:
                self._attention(
                    "beam centre",
                    f"The symmetry axis is {shift:+.0f} px from the calibrated centre: check the calibration or "
                    "the sample alignment.", None, "Look at the horizontal cut: its halves should mirror each other.",
                )
        halves = self._call("choose_halves")
        if halves is not None:
            self._call("set_halves", {"side": halves["recommended"]})
            self._decide("halves", halves["recommended"], halves["reason"])
        self._spacing()
        if self.options.fit and self.catalog.fitter is not None:
            self._fit()
        elif self.catalog.fitter is None:
            self._decide("fit", "not run here", "no fitter in this session: Send to Fitting fits the prepared curve")

    def _horizontal_cut(self, cuts: dict) -> None:
        source = cuts.get("horizontal_source")
        # The rows the cut uses, both ends included (as Analyze's Cuts card says them).
        where = f"rows {cut_spans(cuts, self.catalog.results.status)[0]}"
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
        self.progress("fit_horizontal_cut: predicting the particles of I(qy) (about 15 s)…")
        if self._call("fit_horizontal_cut") is None:
            return
        # The fit as the tool kept it, at full precision: every number below is formatted once from it.
        fit = self.catalog.results.gisaxs.get("fit") or {}
        solutions = fit.get("solutions") or []
        if not solutions:
            return
        _index, best = chosen(fit)
        self._decide("fit", _solution_text(best), PREDICTED.format(curve=fit.get("curve")))
        ranges = _interval_text(best)
        if ranges:
            self._decide("fit intervals", INTERVALS.format(values=ranges), INTERVALS_WHY)
        for warning in best.get("warnings") or ():
            self._decide("fit caveat", warning, "reported by the fit")
        if bounds_text(best):
            self._decide("fit caveat", bounds_text(best), "a value on the edge of its range is not determined by the curve")
        zeros = zero_terms(best)  # linear terms at their non-negativity limit: said, not counted as a bound
        if zeros:
            self._decide("fit caveat", AT_ZERO.format(names=" and ".join(zeros)),
                         AT_ZERO_WHY + (BEAM_TAIL_OFF if LINEAR_TERMS[-1][1] in zeros else ""))
        distance = _distance(best)
        if distance is not None and distance >= UNCORRELATED_D_NM:
            self._decide(
                "fit caveat", f"D = {distance:.3g} nm: this solution has no interparticle correlation in the q range",
                "a structure factor that is flat here; D itself is not determined by the curve",
            )
        others = rivals(solutions)
        if others:
            self._model_question([best, *others])
        if bounds_text(best):
            self._attention("fit", AT_BOUND.format(model=best["model"], bounds=bounds_text(best)), None, AT_BOUND_HINT)
        if not best.get("converged", True):
            self._attention(
                "fit", "The best fit did not converge.", None,
                "Open the curve in Fitting (Send to Fitting) and refine it with a chosen model.",
            )
        if poor_fit(best):  # converged, but far from the curve: Fitting's poor-fit limit
            self._attention("fit", POOR_FIT.format(model=best["model"], chi2=f"{best['chi2']:.3g}", limit=f"{POOR_CHI2:g}"),
                            None, POOR_FIT_HINT)

    def _model_question(self, probable: list[dict]) -> None:
        """Another solution is about as probable: the person chooses (the probabilities said, and the solutions
        whose D is close to 2π/q* named — a length scale, not necessarily a spacing)."""
        spacing = self.catalog.results.gisaxs.get("spacing")
        agree = [
            item for item in probable
            if spacing and _distance(item) and abs(_distance(item) - spacing["distance_nm"]) <= SAME_D * spacing["distance_nm"]
        ]
        hint = ""
        if agree:
            hint += (f"Solutions whose D is within {SAME_D:.0%} of 2π/q* of the cut: "
                     + "; ".join(_solution_text(item) for item in agree)
                     + " (2π/q* is a length scale, not necessarily the particle spacing). ")
        listed = "; ".join(f"rank {item['rank']}, {_solution_text(item)}" for item in probable)
        self._attention("model", PROBABLE.format(n=len(probable), limit=f"{ASK_PROBABILITY:g}", solutions=listed),
                        None, hint + PROBABLE_HINT)

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
