"""Re-index all measured centres against all visible phases after each cell fit."""

from __future__ import annotations

from dataclasses import replace

from ..domain.crystal.cell_observations import finite_number, select_peaks
from ..domain.crystal.refinement import refine_cell, system_for_cell
from .indexing import assign_peaks


UNSTABLE = "The reflection assignments did not stabilize at the fitted metrics."
LOST_TRACK = "The fitted centre no longer matches the tracked reflection."
SELECTION = ("Use one family within max(3 sigma_q, FWHM/2). Omit unresolved same-phase and cross-phase "
             "overlaps because the family intensity fractions on a textured frame are unknown.")


def _fit(spec, system, rows):
    peaks, excluded = select_peaks(spec.name, rows)
    retained = []
    for peak in peaks:
        row = rows[peak.source_index]
        if "tracked_hkl" in row and (row["tracked_phase"] != spec.name or
                tuple(row["tracked_hkl"]) not in (peak.equivalents or (peak.hkl,))):
            excluded.append({"source_index": peak.source_index, "q": peak.q,
                             "reason": LOST_TRACK})
        else:
            retained.append(peak)
    peaks = retained
    for i, row in enumerate(rows):
        if row.get("tracked_phase") == spec.name and all(
            r["phase"] != spec.name for r in (row, *row.get("alternatives", ()))
        ):
            excluded.append({"source_index": i, "q": row["q"], "reason": LOST_TRACK})
    fit = refine_cell(spec.reference_cell, system, peaks, cell_errors=spec.cell_errors)
    invalid = {item["source_index"] for item in fit["excluded"]}
    peaks = [p for p in peaks if p.source_index not in invalid]
    fit["excluded"].extend(excluded)
    for entry in fit["excluded"]:
        src = rows[entry["source_index"]]
        candidates = [r for r in (src, *src.get("alternatives", ())) if r["phase"] == spec.name]
        entry.update(sigma=finite_number(src.get("q_err")), candidate_hkl=", ".join(r["hkl"] for r in candidates),
                     candidates=[{"hkl": r["hkl"], "q": r["q_hkl"]} for r in candidates])
    for row in fit["residuals"]:
        src = rows[row["source_index"]]
        if "before_refraction" in src:
            row["before_refraction"] = src["before_refraction"]
    fit.update(phase=spec.name, selection_rule=SELECTION, phase_spec=spec.record(),
               seeds=[{"hkl": list(p.hkl), "q": p.q, "sigma": p.sigma,
                       "fwhm": rows[p.source_index].get("fwhm")} for p in peaks])
    return fit, peaks


def _omissions(spec, fit, peaks):
    """Influence of each retained singleton, conditional on the other final index assignments."""
    draws = []
    for i, peak in enumerate(peaks):
        trial = refine_cell(spec.reference_cell, fit["system"], peaks[:i] + peaks[i+1:])
        draws.append({"omitted_q": peak.q, "omitted_hkl": list(peak.hkl), "reason": trial["reason"],
                      "values": trial["values"], "chi2_red": trial["chi2_red"],
                      "zero_shift_fitted": trial.get("zero_shift_fitted")})
    return {"method": "Omit one retained centre in turn with the other indices fixed. Refit the metric and offset.",
            "fits": draws, "range_A": {k: [min(vals), max(vals)] for k in ("a", "b", "c")
                if (vals := [r["values"][k] for r in draws if k in r["values"]])}}


def fit_assignments(specs, assignment, *, check_cancel=None, initial_models=None, initial_offsets=None):
    """Iterate to a fixed set of (measurement, family) pairs, testing the complete catalogue each time."""
    systems = [system_for_cell(s.reference_cell, number=s.space_group_number, symbol=s.space_group,
                               operations=s.operations) for s in specs]
    measured = assignment["assignments"]
    models, offsets = list(initial_models or specs), dict(initial_offsets or {})
    seen = set()
    previous = None
    history = [[] for _ in specs]
    for iteration in range(13):
        if check_cancel:
            check_cancel()
        rows = assignment["assignments"]
        pairs = [_fit(s, system, rows) for s, system in zip(specs, systems)]
        signature = tuple(tuple((p.source_index, p.hkl) for p in peaks) for _, peaks in pairs)
        for steps, (fit, peaks) in zip(history, pairs):
            steps.append({"indices": [{"source_index": p.source_index, "hkl": list(p.hkl)} for p in peaks],
                          "values": fit["values"], "reason": fit["reason"], "chi2_red": fit["chi2_red"],
                          "zero_shift": fit.get("zero_shift"), "metric_rank": fit["metric_rank"]})
        if signature == previous or not any(f["ok"] for f, _ in pairs):
            break
        if signature in seen or iteration == 12:
            for fit, _ in pairs:
                fit.update(ok=False, reason=UNSTABLE, values={}, errors={}, strain={}, strain_errors={},
                           residuals=[], seeds=[], zero_shift_fitted=False, zero_shift=None, zero_shift_error=None)
            break
        seen.add(signature)
        previous = signature
        for i, (fit, _) in enumerate(pairs):
            if fit["ok"]:
                models[i] = replace(specs[i], cell=tuple(fit["native_cell"]))
                offsets[specs[i].name] = fit["zero_shift"]
        assignment = assign_peaks(measured, models, q_offsets=offsets)
    for spec, steps, (fit, peaks) in zip(specs, history, pairs):
        fit.update(assignment_iterations=steps, unassigned_q=assignment["unassigned_q"])
        if fit["ok"]:
            fit["assignment_sensitivity"] = _omissions(spec, fit, peaks)
    return [fit for fit, _ in pairs], models, offsets
