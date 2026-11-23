"""Joint fibre texture of the rings assigned to each visible phase on a GIWAXS frame."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.gimap.shared.profiles import Profile
from src.gimap.shared.radial_peaks import find_peaks, orientation_peak_rows

from ..domain.crystal.film_optics import film_optics, ring_optical_factors, ring_transmission
from ..domain.crystal.pole_figure import MIN_BINS, fold_profile
from ..domain.crystal.texture import TextureRing, fit_crystal_orientation
from ..domain.giwaxs import giwaxs_maps
from ..domain.intensity import film_absorption_on
from ..domain.mask_bitmap import bitmap_record
from ..domain.orientation_reduction import measured_ring_profiles
from ..domain.peak_fit import GAUSSIAN, fit_peak
from ..domain.refracted_film import REFRACTED_FILM_FORMULA
from .indexing import assign_peaks
from .pole_figure import _radial_in_q
from .reduction_errors import reduction_errors
from .use_cases import AnalyzeFrame


def _pixels(analysis, frame_source):
    settings = analysis.corrections
    if not (settings.lorentz or settings.mirror_fill):
        return analysis.data, analysis.valid, reduction_errors(settings, vars(analysis))
    # Use an independent preprocessing cache on the worker. Keep the mask of the displayed frame.
    reducer = AnalyzeFrame(frame_source, None)
    return reducer.measured_orientation_pixels(
        vars(analysis), settings, analysis.geometry, analysis.raw_data, analysis.raw_valid,
        analysis.path.name, bitmap_record(analysis.drawn_mask, analysis.shape))


def _assign(analysis, specs):
    radial = _radial_in_q(analysis)
    if radial is None:
        raise ValueError("The radial curve is unavailable.")
    profile = Profile.of(*radial, analysis.reduction.curve("radial").pixels)
    rows = orientation_peak_rows(find_peaks(profile))
    request = analysis.request
    window = request.giwaxs.chi_q_window if request else None
    if window:
        # An explicitly selected band may cover a weak peak outside the default search cap.
        low, high = sorted(window)
        inside = [row for row in rows if low <= row["q"] <= high]
        if len(inside) == 1:
            inside[0]["q_window"] = [low, high]
        elif not inside:
            selected = fit_peak(*radial, (low, high), profile=GAUSSIAN)
            if selected.ok:
                rows.append({"q": selected.center, "q_err": selected.center_err,
                             "fwhm": selected.fwhm, "q_window": [low, high]})
    return assign_peaks(sorted(rows, key=lambda row: row["q"]), specs, geometry=analysis.geometry)


def _profile(maps, pixels, analysis, window, q, optics, *, fit_absorption=False):
    corrections = analysis.corrections
    auto_exit = optics["available"] and corrections.orientation_min_exit_deg is None
    if auto_exit:
        corrections = replace(corrections, orientation_min_exit_deg=0.)
    film_index = (optics["delta"], optics["beta"]) if optics["available"] else None
    profiles, indices, sines, chi = measured_ring_profiles(
        maps, *pixels, analysis.geometry, window, maps.q_step, corrections, film_index=film_index)
    curve = profiles[1.0]
    region = curve.region
    if optics["available"]:
        # Weak net bins next to the horizon carry information on transmission. Background, shadow
        # detection, explicit exit-angle limit and the active q-map geometry still apply.
        if auto_exit:
            region["exit_angle_source"] = "refracted film waves"
        region.update(horizon_fraction=0., horizon_excluded_deg=None)
        profile = fold_profile(curve.x, curve.intensity, curve.sigma, region["background"])
        if optics.get("model") == "frame corrected with refracted waves":
            return profile, region, None, None
        rays = (optics, analysis.geometry, q, chi, maps.qz.flat[indices], curve)
        # Preserve the surface field as a separate diagnostic. The model uses the full factor.
        region["exit_transmission"] = ring_transmission(*rays, sin_exit=sines, internal_bins=True).tolist()
        factor, absorption = ring_optical_factors(
            *rays, sin_exit=sines, internal_bins=True, fit_absorption=fit_absorption,
            thickness_nm=optics.get("thickness_nm", np.inf), geometric_depth=optics.get("geometric_optical_depth"))
        return profile, region, factor, absorption
    profile = fold_profile(curve.x, curve.intensity, curve.sigma, region["background"],
                           horizon_chi_deg=region.get("horizon_chi_deg"))
    return profile, region, None, None


def _frame_optics(phase, geometry, corrections, fit_absorption):
    wave_corrected = film_absorption_on(corrections) and getattr(corrections, "film_delta", None) is not None
    if wave_corrected:
        return {"model": "frame corrected with refracted waves", "available": True,
                "formula": REFRACTED_FILM_FORMULA,
                "source": "frame CIF correction" if corrections.film_cif_path else "typed frame optical constants",
                "cif_path": corrections.film_cif_path,
                "wavelength_angstrom": geometry.wavelength_angstrom,
                "delta": corrections.film_delta, "beta": corrections.film_beta,
                "depth_source": "frame thickness" if corrections.film_thickness_nm is not None else "thick limit",
                "thickness_nm": corrections.film_thickness_nm}
    optics = film_optics(phase, geometry.wavelength_angstrom)
    if geometry.incidence_deg <= 0.:
        optics.update(available=False, reason="Film optics requires a positive incidence angle.")
    if not optics["available"]:
        return {**optics, "model": "unavailable"}
    optics.update(model="refracted waves", formula=REFRACTED_FILM_FORMULA,
                  depth_source="fitted" if fit_absorption else "thick limit")
    if film_absorption_on(corrections):
        d = corrections.film_thickness_nm
        optics.update(depth_source="frame thickness", thickness_nm=d,
                      geometric_optical_depth=d / (1000 * corrections.attenuation_length_um))
    return optics


def _skip(item):
    if item.get("tentative"):
        return "Phase symmetry is unknown."
    if item.get("overlapping_phases"):
        return "Reflections from several phases fall in the integration window."
    return None


def frame_crystal_orientation(analysis, specs, *, chi_max=90.0, frame_source=None, check_cancel=None,
                              assignment=None, fit_absorption=False):
    """Fit all assigned, phase-resolved rings, independently of the three-ring summaries.

    An automatic run supplies its own peak list and integration windows via its assignment record.
    The dialog applies the same default radial search. The optional cancellation callback raises to stop.
    """
    specs = [spec for spec in specs if spec.visible]
    if not specs:
        return []
    if analysis is None or analysis.kind != "giwaxs" or analysis.geometry is None or analysis.reduction is None:
        raise ValueError("Open a GIWAXS frame with a geometry first.")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Use a distinct name for each visible phase.")
    if fit_absorption and film_absorption_on(analysis.corrections):
        raise ValueError("The frame already includes a film absorption correction.")
    if assignment is None:
        assignment = _assign(analysis, specs)
    maps = pixels = None
    result = []
    for spec in specs:
        phase = optics = None
        rings, metadata, excluded = [], {}, []
        for index, item in enumerate(assignment["assignments"]):
            if item["phase"] != spec.name:
                continue
            if check_cancel:
                check_cancel()
            window = item["q_window"]
            reason = _skip(item)
            if reason is None:
                if phase is None:
                    phase = spec.phase()
                    optics = _frame_optics(phase, analysis.geometry, analysis.corrections, fit_absorption)
                if maps is None:
                    maps = giwaxs_maps(analysis.shape, analysis.geometry)
                    pixels = _pixels(analysis, frame_source)
                profile, settings, factor, absorption = _profile(
                    maps, pixels, analysis, window, item["q"], optics, fit_absorption=fit_absorption)
                use = profile.measured & (profile.chi <= chi_max)
                use &= np.isfinite(profile.value) & np.isfinite(profile.sigma) & (profile.sigma > 0)
                if use.sum() < MIN_BINS:
                    reason = "Fewer than eight measured bins."
                elif np.max(profile.value[use] / profile.sigma[use]) < 3:
                    reason = "No measured bin reaches three standard errors above background."
            if reason:
                excluded.append({"q": item["q"], "q_window": window, "reason": reason,
                                 "overlapping_phases": item.get("overlapping_phases", [])})
                continue
            rings.append(TextureRing(index, item["q"], *window, replace(profile, measured=use),
                                     transmission=factor, absorption=absorption))
            transmission = settings.pop("exit_transmission", None)
            metadata[index] = {"q_window": window, "profile_settings": settings,
                               "exit_transmission": transmission,
                               "measured": profile.measured.tolist(), "fit_mask": use.tolist(),
                               "observed": [float(v) if m else None for v, m in zip(profile.value, profile.measured)],
                               "sigma": [float(s) if m else None for s, m in zip(profile.sigma, profile.measured)]}
        if rings:
            record = fit_crystal_orientation(phase, rings, check_cancel=check_cancel)
            for ring in record.get("rings", []):
                ring.update(metadata[ring["ring_id"]])
        else:
            record = {"phase": spec.name, "ok": False, "determined": False, "parameter_status": "unavailable", "rings": [],
                      "reasons": ["No usable ring assigned to this phase."]}
        record.setdefault("excluded_rings", []).extend(excluded)
        record.update(phase_spec=spec.record(), fitted_up_to_chi_deg=chi_max,
                      q_axis=assignment.get("q_axis"))
        if optics is not None:
            record["film_optics"] = optics
        result.append(record)
    return result
