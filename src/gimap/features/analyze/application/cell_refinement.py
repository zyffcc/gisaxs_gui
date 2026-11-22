"""Refine the reference phases against the measured radial or sector peak centres of one frame."""

from __future__ import annotations

from dataclasses import replace

from src.gimap.shared.profiles import Profile
from src.gimap.shared.radial_peaks import MAX_PEAKS, find_peaks, orientation_peak_rows

from ..domain.crystal.cell_observations import CellPeak, Refraction, corrected, finite_number
from ..domain.crystal.refinement import refine_cell, refine_fibre_strain, select_peaks, system_for_cell
from ..domain.giwaxs import giwaxs_maps, q_from_two_theta, radial_sector_curves
from ..domain.models import GiwaxsSettings
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


def _sector_assignments(curves, geometry, specs, cap):
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
        assigned = assign_peaks(fitted, specs, geometry=geometry)
        for item in assigned["assignments"]:
            rows.append({**item, "sector": curve.key, "chi_range": (low, high)})
    return rows


def _fibre_fit(phase, axis, sectors):
    selected, excluded = select_peaks(phase.name, sectors, with_atoms=bool(phase.atoms))
    # The angular aperture identifies an equivalent pole. Only the radius was measured.
    peaks = [replace(peak, chi_range_deg=sectors[peak.source_index].get("chi_range")) for peak in selected]
    record = refine_fibre_strain(phase.cell, axis, peaks)
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
    rows = assignment["assignments"]
    axes = {}
    for item in orientations or ():
        resolved = {tuple(axis) for axis in item.get("axes", ())}
        if item.get("determined") and len(resolved) == 1:
            axes[item["phase"]] = resolved.pop()
    sectors = None
    results = []
    for spec in specs:
        if check_cancel:
            check_cancel()
        phase = spec.phase()
        reference = spec.reference_cell
        system = system_for_cell(reference, number=spec.space_group_number, symbol=spec.space_group,
                                 operations=spec.operations)
        peaks, excluded = select_peaks(spec.name, rows, with_atoms=bool(phase.atoms))
        record = refine_cell(reference, system, peaks)
        widths = {row["q"]: row.get("fwhm") for row in rows}
        invalid = {row["source_index"] for row in record["excluded"]}
        peaks = [p for p in peaks if p.source_index not in invalid]
        record["excluded"].extend(excluded)
        for entry in record["excluded"]:
            src = rows[entry["source_index"]]
            candidates = [src, *src.get("alternatives", ())]
            entry.update(sigma=finite_number(src.get("q_err")), candidate_hkl=", ".join(
                c["hkl"] for c in candidates if c["phase"] == spec.name))
        record["excluded"].extend(row for row in skipped if row["phase"] == spec.name)
        for row in record["residuals"]:
            src = rows[row["source_index"]]
            if "before_refraction" in src:
                row["before_refraction"] = src["before_refraction"]
        record.update(phase=spec.name, phase_spec=spec.record(), unassigned_q=assignment.get("unassigned_q", []),
                      selection_rule="Within max(3 sigma_q, FWHM/2), exclude cross-phase or tentative "
                                     "assignments. For competing families require one CIF sum(|F|²) at least "
                                     "three times the combined intensity of the others.",
                      seeds=[{"hkl": list(p.hkl), "q": p.q, "sigma": p.sigma, "fwhm": widths[p.q]} for p in peaks],
                      q_axis={**(assignment.get("q_axis") or {}), "critical_angle_deg": geometry.critical_angle_deg,
                              "rebinned_for_refraction": rebinned})
        if spec.name in axes:
            if not spots and not only_spots and sectors is None:
                sectors = _sector_assignments(curves[1:], geometry, specs, max_peaks)
            measured = spots or sectors or []
            record["fibre_strain"] = _fibre_fit(replace(phase, cell=reference), axes[spec.name], measured)
        results.append(record)
    return results
