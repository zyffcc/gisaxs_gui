"""Compare complete runs on identical draws, grouped by the largest true q₊R on each cut."""

import gzip
import json
from pathlib import Path

import numpy as np


def _median(values):
    return float(np.median(values)) if values else None


def _mean(values):
    return float(np.mean(values)) if values else None


def _parameter_stats(pairs):
    out = {}
    for key in ("R", "D"):
        usable = [(t, p) for t, p in pairs if p.get(key) is not None]
        record = dict(n=len(usable), median_abs_relative=_median([abs(p[key] / t[key] - 1) for t, p in usable]))
        for level in (68, 95):
            field = f"{key}_{level}"
            record[f"coverage{level}"] = _mean([p[field][0] <= t[key] <= p[field][1] for t, p in usable])
        out[key] = record
    return out


def _shape_confusion(rows, stage, q_plus):
    """Both axes refer to one population. Count errors in population number separately."""
    count_errors = {"single_to_multiple": 0, "multiple_to_single": 0, "other": 0}
    bins = {name: {"sphere_to_sphere": 0, "sphere_to_upright": 0, "upright_to_sphere": 0, "upright_to_upright": 0}
            for name in ("fitted_q+R<=2", "fitted_q+R>2")}
    families = {"sphere": "sphere", "vertical_cylinder": "upright"}
    for row in rows:
        result = row[stage]
        actual, fitted = row["truth_families"], result["families"]
        if len(actual) != len(fitted):
            key = "single_to_multiple" if len(actual) == 1 else "multiple_to_single" if len(fitted) == 1 else "other"
            count_errors[key] += 1
        elif len(actual) == 1 and actual[0] in families and fitted[0] in families:
            radius = result["populations"][0].get("R")
            if radius is None:
                continue
            group = bins["fitted_q+R>2" if q_plus * radius > 2.0 else "fitted_q+R<=2"]
            group[f"{families[actual[0]]}_to_{families[fitted[0]]}"] += 1
    for b in bins.values():
        fitted_upright = b["sphere_to_upright"] + b["upright_to_upright"]
        b["sphere_fraction_of_fitted_uprights"] = b["sphere_to_upright"] / fitted_upright if fitted_upright else None
    return dict(population_count_errors=count_errors, single_population=bins)


def _group(rows, q_plus):
    out = {"cuts": len(rows)}
    for stage in ("fit", "choose", "auto"):
        measured = [r for r in rows if stage in r]
        if not measured:
            continue
        ok = [r for r in measured if r[stage]["ok"]]
        correct, first, listed = [], [], []
        pairs = []
        for row in ok:
            truth = sorted(row["case"]["populations"], key=lambda p: p["R"])
            result = row[stage]
            same = result["families"] == row["truth_families"]
            correct.append(same)

            def in_mode(solution):
                pops = solution["populations"]
                return (solution["families"] == row["truth_families"] and len(pops) == len(truth) and
                        all(p.get(k) is not None and abs(p[k] / t[k] - 1) <= .2
                            for p, t in zip(pops, truth) for k in ("R", "D")))

            first.append(in_mode(result))
            listed.append(first[-1] or any(in_mode(a) for a in result.get("alternatives", [])))
            if len(truth) == len(result["populations"]):
                pairs.extend(zip(truth, result["populations"]))
        out[stage] = dict(n_ok=len(ok), n_failed=len(measured) - len(ok), composition_right=_mean(correct),
                          truth_mode_first=_mean(first), truth_mode_listed=_mean(listed),
                          seconds=_median([r[stage]["seconds"] for r in measured]),
                          parameters=_parameter_stats(pairs))
        out[stage]["shape_confusion"] = _shape_confusion(ok, stage, q_plus)
    return out


def _population_classes(rows, q_plus):
    """Parameter errors of known-composition fits, binned at each particle population's own q₊R."""
    pairs = []
    for row in rows:
        solution = row.get("fit", {})
        if solution.get("ok"):
            truth = sorted(row["case"]["populations"], key=lambda p: p["R"])
            if len(truth) == len(solution["populations"]):
                pairs.extend(zip(truth, solution["populations"]))
    if not pairs:
        return {}
    out = {}
    for family in ("sphere", "vc"):
        for lo, hi in ((0., np.inf), (0., 1.2), (1.2, 1.7), (1.7, 2.3), (2.3, 3.5), (3.5, np.inf)):
            group = [(t, p) for t, p in pairs if t["shape"] == family and lo <= q_plus * t["R"] < hi]
            label = "overall" if lo == 0 and np.isinf(hi) else f"q+R {lo:g}-{hi:g}"
            out.setdefault(family, {})[label] = _parameter_stats(group)
    return out


def compare_runs(paths, q_plus):
    """Reject mismatched seeds, case draws and missing cuts before computing any comparisons."""
    reference, out, particles = None, {}, {}
    for label, path in paths:
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            records = json.load(stream)["results"]
        cases = {r["case"]["index"]: r["case"] for r in records}
        if len(cases) != len(records) or (reference is not None and cases != reference):
            raise ValueError(f"The case draws in {path.name} do not match the paired cuts")
        reference = cases
        groups = {"overall": records}
        groups["spheres only"] = [r for r in records if all(p["shape"] == "sphere" for p in r["case"]["populations"])]
        for composition in ("sphere", "vc", "sphere,sphere", "sphere,vc"):
            groups[composition] = [r for r in records if r["case"]["composition"] == composition]
        for lo, hi in ((0., 1.2), (1.2, 1.7), (1.7, 2.3), (2.3, np.inf)):
            groups[f"q+R {lo:g}-{hi:g}"] = [r for r in records if lo <=
                                             q_plus * max(p["R"] for p in r["case"]["populations"]) < hi]
        out[label] = {key: _group(rows, q_plus) for key, rows in groups.items()}
        populations = _population_classes(records, q_plus)
        if populations:
            particles[label] = populations
    return dict(n_curves=len(reference), q_plus_inv_nm=q_plus, runs=out, known_composition_populations=particles)
