"""Capture complete equivalent fits before the kit reduces them to parameter ranges.

This compatibility hook lives only in an isolated fit job, is restored even on failure, and
does not modify the bundled kit. It serialises at most six distinct, already computed fits;
there is no additional optimisation. Older external kits still return their normal result.
"""
from __future__ import annotations

import copy
import importlib
import inspect
from contextlib import contextmanager

import numpy as np

from ...domain.fit1d_bounds import Constraint
from ...domain.fit1d_edges import mark_bounds


@contextmanager
def capture_alternatives():
    try:
        api = importlib.import_module("research.fit1d.api")
        ensemble = importlib.import_module("research.fit1d.ensemble")
    except ImportError:  # external/older kits and protocol fakes do not expose this hook
        yield
        return
    original = getattr(api, "_composition_result", None)
    if (original is None or tuple(inspect.signature(original).parameters) != ("run", "settings", "ood")
            or not callable(getattr(ensemble, "equivalent", None))):
        yield
        return

    def capture(run, settings, ood):
        result = original(run, settings, ood)
        if run.born is not None:  # pre-DWBA alternatives do not share the final curve's objective
            return result
        seen = [run.best]
        alternatives = []
        for candidate in sorted(ensemble.equivalent(run), key=lambda c: c["chi2"]):
            if any(np.allclose(candidate["u"], other["u"], rtol=1e-3, atol=1e-4)
                   and np.allclose(candidate["coef"], other["coef"], rtol=1e-3, atol=1e-8) for other in seen):
                continue
            view = copy.copy(run)
            view.candidates = [candidate]
            # Keep the ensemble for uncertainty/ambiguity diagnostics, even when another fit is displayed.
            view.tracks = list(run.tracks) + list(run.candidates)
            alternative = original(view, settings, ood)
            # Profile alternatives may have no covariance. The ensemble still establishes
            # their ambiguity/range; do not make it disappear when such a fit is selected.
            if not alternative["flags"].get("ambiguity"):
                labels = {pop["slot"]: pop["pop"] for pop in alternative["populations"]}
                ambiguity = ensemble.assess(view, labels)
                alternative["flags"]["ambiguity"] = ambiguity
                alternative["flags"]["ambiguous"] = [name for name, item in ambiguity.items() if item["ambiguous"]]
                for pop in alternative["populations"]:
                    for key in ("R_nm", "D_nm"):
                        item = ambiguity.get(f"{pop['pop']}.{key}")
                        if item:
                            pop[f"{key}_range"] = [item["min"], item["max"]]
            alternative["delta_chi2"] = alternative["chi2"] - result["chi2"]
            alternatives.append(alternative)
            seen.append(candidate)
            if len(alternatives) >= 6:
                break
        result["gimap_equivalent_solutions"] = alternatives
        return result

    api._composition_result = capture
    try:
        yield
    finally:
        api._composition_result = original


def native_alternatives(raw, native, options, make_row):
    """Complete selectable snapshots, all from the final constrained composition fit."""
    best = (raw.get("results") or [{}])[0]
    if not best.get("gimap_equivalent_solutions"):
        return []
    kit_results = importlib.import_module("gisaxs1d_kit.results")
    out = []
    for entry in best.get("gimap_equivalent_solutions") or ():
        own = kit_results.gui_result(dict(raw, results=[entry]), model_name=native["summary"]["model"],
                                      kit_version=native.get("kit_version") or "")
        for key in ("settings", "measurement_sigma", "fit1d_constraints"):
            own[key] = copy.deepcopy(native.get(key))
        mark_bounds(own, [Constraint.from_dict(item) for item in
                         (own.get("fit1d_constraints") or {}).get("used", [])])
        # Alternatives have their own errors/flags; do not attach the winner's edge resolution notes.
        own["selection"] = {"kind": "equivalent", "delta_chi2": entry["delta_chi2"],
                             "winner": entry.get("winner", {})}
        row = make_row(own)
        own["warnings"] = row.get("warnings", [])
        out.append(dict(native=own, mapping_row=row, delta_chi2=entry["delta_chi2"]))
    return out
