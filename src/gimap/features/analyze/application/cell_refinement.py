"""Refine the reference phases against the measured radial or sector peak centres of one frame."""

from __future__ import annotations

from dataclasses import replace

from src.gimap.shared.profiles import Profile
from src.gimap.shared.radial_peaks import MAX_PEAKS, find_peaks, orientation_peak_rows

from ..domain.crystal.cell_observations import CellPeak, Refraction, corrected
from ..domain.crystal.film_optics import film_optics
from ..domain.crystal.refinement import refine_fibre_strain, select_peaks
from ..domain.giwaxs import giwaxs_maps, q_from_two_theta, radial_sector_curves
from ..domain.models import GiwaxsSettings
from .cell_assignments import fit_assignments
from .indexing import IndexingSettings, assign_peaks, film_critical_angle
from .reduction_errors import reduction_errors


def _peak_rows(curve, geometry, cap):
    if curve is None or curve.is_empty:
        return []
    q = curve.x
    if str(curve.x_label).startswith("2θ"):
        q = q_from_two_theta(q, geometry.wavelength_angstrom)
    profile = Profile.of(q, curve.intensity, curve.sigma, curve.pixels)
    return orientation_peak_rows(find_peaks(profile, max_peaks=cap))


def _curves(analysis, geometry):
    if geometry == analysis.geometry:
        return [analysis.reduction.curve(key) for key in ("radial", "in_plane", "out_of_plane")]
    # A radial centre alone has no exit angle. Rebin measured pixels before fitting the peak on the transmitted map.
    maps = giwaxs_maps(analysis.shape, geometry)
    usable = analysis.valid & maps.above_horizon
    if not usable.any():
        return [None] * 3
    options = analysis.request.giwaxs if analysis.request else GiwaxsSettings()
    curves = radial_sector_curves(analysis.data, usable, maps, options,
                                 reduction_errors(analysis.corrections, vars(analysis)), analysis.corrections)
    return list(curves[:3])


def _correct_spots(spots, source, geometry, specs):
    optics = Refraction(source.wavevector_inv_angstrom, source.incidence_deg, geometry.critical_angle_deg,
                        source.critical_angle_deg)
    rows, exclusions, originals = [], [], []
    for item in spots:
        try:
            peak = CellPeak((0, 0, 0), item["q"], item["q_err"], item["q_parallel"], item["qz"])
            q, sigma, (parallel, vertical) = corrected(peak, optics)
        except ValueError as error:
            exclusions.append({"phase": item["phase"], "q": item["q"], "reason": str(error)})
            continue
        rows.append({"q": q, "q_err": sigma, "fwhm": (item.get("fwhm") or 0.) * sigma/peak.sigma,
                     "q_parallel": q*parallel, "qz": q*vertical})
        originals.append({k: item[k] for k in ("q", "q_err", "q_parallel", "qz")})
    assignment = assign_peaks(rows, specs, geometry=geometry)
    for row, original in zip(assignment["assignments"], originals):
        row["before_refraction"] = original
    return assignment, exclusions


def _sector_assignments(curves, geometry, specs, cap, offsets):
    """Independent fitted radii on non-overlapping |χ| sectors. Preserve each sector's angular aperture."""
    rows = []
    limits = []
    for curve in curves:
        if curve is None or curve.is_empty:
            continue
        low, high = curve.region["chi_deg"]
        if any(min(high, b) > max(low, a) for a, b in limits):
            continue
        limits.append((low, high))
        fitted = _peak_rows(curve, geometry, cap)
        assigned = assign_peaks(fitted, specs, geometry=geometry, q_offsets=offsets)
        for item in assigned["assignments"]:
            rows.append({**item, "sector": curve.key, "chi_range": (low, high)})
    return rows


def _fibre_fit(spec, axis, sectors, system):
    selected, excluded = select_peaks(spec.name, sectors)
    # The angular aperture identifies an equivalent pole. Only the radius was measured.
    peaks = [replace(peak, chi_range_deg=sectors[peak.source_index].get("chi_range")) for peak in selected]
    record = refine_fibre_strain(spec.reference_cell, axis, peaks, cell_errors=spec.cell_errors, system=system)
    record["excluded"].extend(excluded)
    for entry in [*record["excluded"], *record["residuals"]]:
        item = sectors[entry["source_index"]]
        entry["sector"] = item.get("sector")
        if "before_refraction" in item:
            entry["before_refraction"] = item["before_refraction"]
    return record


def frame_cell_refinement(analysis, specs, settings=IndexingSettings(), *, assignment=None, orientations=(),
                          max_peaks=MAX_PEAKS, check_cancel=None):
    """One record per visible phase. A supplied assignment preserves the automatic run's actual fitted centres."""
    if analysis is None or analysis.kind != "giwaxs" or analysis.geometry is None or analysis.reduction is None:
        raise ValueError("Open a GIWAXS frame with a geometry first.")
    specs = tuple(spec for spec in specs if spec.visible)
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Use a distinct name for each visible phase.")
    if assignment is not None and [spec.record() for spec in specs] != assignment.get("phases", []):
        raise ValueError("The visible phases changed after peak assignment.")
    source = analysis.geometry
    critical = film_critical_angle(settings, source) if settings.refraction else source.critical_angle_deg
    results = _fit_map(analysis, specs, critical, assignment, orientations, max_peaks, check_cancel)
    compared = {critical: results}
    for spec, record in zip(specs, results):
        angle = float(settings.critical_angle_deg or source.critical_angle_deg or 0.)
        origin = "indexing or map setting"
        if not angle:
            optical = film_optics(spec.phase(), source.wavelength_angstrom)
            angle = optical.get("critical_angle_deg", 0.)
            origin = optical["source"]
        if not angle:
            record["refraction_sensitivity"] = {"available": False, "reason": "No film critical angle available."}
            continue
        for trial in (0., angle):
            if trial not in compared:
                compared[trial] = _fit_map(analysis, specs, trial, assignment, (), max_peaks, check_cancel)
        on_off = {}
        for key, trial in (("off", 0.), ("on", angle)):
            fit = next(f for f in compared[trial] if f["phase"] == spec.name)
            on_off[key] = {k: fit.get(k) for k in ("ok", "reason", "values", "errors", "chi2_red", "metric_rank",
                                                   "seeds", "zero_shift", "zero_shift_error")}
        record["refraction_sensitivity"] = {"available": True, "critical_angle_deg": angle, "source": origin,
                                            **on_off, "on_minus_off_A": {
            k: on_off["on"]["values"][k] - v for k, v in on_off["off"]["values"].items()
            if k in ("a", "b", "c") and k in on_off["on"]["values"]}}
    return results


def _fit_map(analysis, specs, critical, assignment, orientations, max_peaks, check_cancel):
    source = analysis.geometry
    geometry = source.with_critical_angle(critical)
    original = (assignment or {}).get("assignments", ())
    spots = [r for r in original if r.get("q_parallel") is not None and r.get("qz") is not None]
    only_spots = bool(spots) and len(spots) == len(original)
    skipped = []
    rebinned = source != geometry and not only_spots
    if spots and source != geometry:
        spot_assignment, skipped = _correct_spots(spots, source, geometry, specs)
        spots = spot_assignment["assignments"]
        if only_spots:
            assignment = spot_assignment
    curves = _curves(analysis, geometry) if not only_spots else []
    if assignment is None or rebinned:
        assignment = assign_peaks(_peak_rows(curves[0], geometry, max_peaks), specs, geometry=geometry)
    axes = {}
    for item in orientations or ():
        resolved = {tuple(axis) for axis in item.get("axes", ())}
        if item.get("determined") and len(resolved) == 1:
            axes[item["phase"]] = resolved.pop()
    sectors = None
    results, models, offsets = fit_assignments(specs, assignment, check_cancel=check_cancel)
    for spec, record in zip(specs, results):
        if check_cancel:
            check_cancel()
        record["excluded"].extend(row for row in skipped if row["phase"] == spec.name)
        record["q_axis"] = {**(assignment.get("q_axis") or {}), "critical_angle_deg": geometry.critical_angle_deg,
                            "rebinned_for_refraction": rebinned}
        if spec.name in axes:
            if not spots and not only_spots and sectors is None:
                sectors = _sector_assignments(curves[1:], geometry, models, max_peaks, offsets)
            measured = assign_peaks(spots, models, q_offsets=offsets)["assignments"] if spots else sectors or []
            record["fibre_strain"] = _fibre_fit(spec, axes[spec.name], measured, record["system"])
    return results
