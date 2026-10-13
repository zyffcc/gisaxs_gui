"""The GISAXS part of the standard procedure: cut, symmetry, halves, spacing, fit.

After the geometry and the frames (shared with GIWAXS): the horizontal cut at the Yoneda band
(or a question when there is none), the beam-centre column moved to the symmetry axis, the
halves chosen with their reason, the dominant in-plane spacing 2π/q* when I(qy) has a side
maximum, and — when a fitter is available — a physical fit of the curve. Each step is a tool
call, so the GUI shows it and the AI could do the same.
"""

from __future__ import annotations

import math
from typing import Optional

from .gisaxs_report import cut_spans

CLOSE_CHI2 = 1.1
"""Solutions within 10 % of the best χ² are not distinguished by the curve."""
UNCORRELATED_D_NM = 250.0
"""D beyond this (half the search bound of 500 nm): no correlation peak in a GISAXS q range."""
SAME_D = 0.15
"""A fitted D within 15 % of 2π/q* agrees with the observed spacing."""


def _distance(solution: dict) -> Optional[float]:
    components = solution.get("components") or []
    return float(components[0]["D"]) if components and components[0].get("D") is not None else None


def _solution_text(solution: dict) -> str:
    component = (solution.get("components") or [{}])[0]
    return (
        f"{solution['model']}: R = {component.get('R', math.nan):.3g} nm, D = {component.get('D', math.nan):.3g} nm, "
        f"χ² = {solution['chi2']:.3g}"
    )


class GisaxsProcedureMixin:
    """Needs ``catalog``, ``options``, ``progress``, ``_call``, ``_decide`` and ``_attention`` (``StandardPipeline``)."""

    def _analyse_gisaxs(self) -> None:
        results = self.catalog.results
        status = results.status or {}
        cuts = status.get("gisaxs") or {}
        self._horizontal_cut(cuts)
        symmetry = self._call("refine_beam_center_symmetry")
        if symmetry is not None:
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
        else:
            self._decide("horizontal cut", f"set by hand ({where})", "kept as chosen")

    def _spacing(self) -> None:
        found = self._call("in_plane_spacing")
        if found is None:
            return
        spacing = found.get("spacing")
        if spacing is None:
            self._decide(
                "spacing", "no side maximum",
                "I(qy) has no maximum or shoulder away from qy = 0 above the noise: no dominant in-plane distance in the q range",
            )
        elif spacing["kind"] == "maximum":
            self._decide(
                "spacing", f"D ≈ 2π/q* = {spacing['distance_nm']:.3g} nm (side maximum at |qy| = {spacing['q']:.4g} Å⁻¹)",
                "a correlation peak of I(qy): the mean in-plane distance between scatterers (a first estimate; the fit refines it)",
            )
        else:
            self._decide(
                "spacing", f"a shoulder at |qy| = {spacing['q']:.4g} Å⁻¹ (2π/q ≈ {spacing['distance_nm']:.3g} nm), no resolved maximum",
                "a change of slope, not a peak: a hint for the interparticle distance (the fit also starts from it), not a measurement",
            )

    def _fit(self) -> None:
        self.progress("fit_horizontal_cut: fitting spheres and cylinders to I(qy) (about half a minute)…")
        fit = self._call("fit_horizontal_cut")
        if fit is None:
            return
        solutions = fit.get("solutions") or []
        if not solutions:
            return
        best = solutions[0]
        self._decide("fit", _solution_text(best), f"lowest χ² of the numerical physical fits of {fit.get('curve')}")
        for warning in best.get("warnings") or ():
            self._decide("fit caveat", warning, "reported by the fit")
        distance = _distance(best)
        if distance is not None and distance >= UNCORRELATED_D_NM:
            self._decide(
                "fit caveat", f"D = {distance:.3g} nm: this solution has no interparticle correlation in the q range",
                "a structure factor that is flat here; D itself is not determined by the curve",
            )
        close = [item for item in solutions if item["chi2"] <= CLOSE_CHI2 * best["chi2"]]
        if len({item["model"] for item in close}) > 1:
            spacing = self.catalog.results.gisaxs.get("spacing")
            agree = [
                item for item in close
                if spacing and _distance(item) and abs(_distance(item) - spacing["distance_nm"]) <= SAME_D * spacing["distance_nm"]
            ]
            hint = (
                "Solutions whose D agrees with the observed spacing: " + "; ".join(_solution_text(item) for item in agree) + ". "
                if agree else ""
            )
            self._attention(
                "model",
                f"{len(close)} solutions of {len({item['model'] for item in close})} particle families fit within "
                f"{CLOSE_CHI2 - 1:.0%} of the best χ²: the curve alone does not decide the model.",
                None,
                hint + "Choose the particle shape you expect and refine that model in Fitting; compare the fitted curves.",
            )
        if not best.get("converged"):
            self._attention(
                "fit", "The best fit did not converge.", None,
                "Open the curve in Fitting (Send to Fitting) and refine it with a chosen model.",
            )

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


__all__ = ["GisaxsProcedureMixin"]
