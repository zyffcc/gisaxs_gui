"""The single-page fit1d handler; imported only by the isolated JobRunner process.

One fit of the kit with the Model's constraints, then its parameters at an edge resolved like an analyst would:
widen a range, fit again and keep what does not make the fit worse (``domain.fit1d_edges.resolve``); a value that
stays at an edge is given as a bound, at the least extreme value the curve allows (``settle``). A frame of a series
(``payload["previous"]``: the previous frame's populations) is also fitted as a continuation of the previous frame,
which is kept when it fits as well (``domain.fit1d_follow``). With an ``auto:`` composition the kit first fits and ranks every composition;
the best is then resolved the same way, and all of them come back as candidates (``MAX_CANDIDATES`` best by BIC),
each with its parameters, errors, equivalent ranges and flags, and its values for Model.
"""

from __future__ import annotations

import importlib
import time

import numpy as np

from .fit1d_method import _clean, load_kit, solution_row
from ...application.workflow_v5 import FIT1D_SHAPES, validate_options
from ...domain.fit1d_bounds import Constraint, as_kit, measurable
from ...domain.fit1d_edges import mark_bounds, remaining_note, resolve, settle
from ...domain.fit1d_follow import follow_note, same_basin, tolerance, window

MAX_CANDIDATES = 6
SHAPE_IDS = {"sphere": 1, "vertical_cylinder": 3}


def _fit(core, arguments, constraints):
    """The kit's result and its raw fit1d result (the compositions it tried, best first)."""
    result = core.fit_curve({**arguments, "constraints": as_kit(constraints) or None, "include_raw": True})
    return result, result.pop("raw")


def _item(result):
    # Its own effective errors are also the errors of the displayed curve and of the saved score.
    return dict(q=result["curves"]["q_nm_inv"], observed=result["curves"]["intensity"],
                sigma=result["curves"]["sigma"], sigma_estimated=False, side="positive", sign=1)


def _mapping(row) -> dict:
    """What ``model_from_solution`` needs of a row (the curves of the result are kept once, in the result)."""
    return {key: row[key] for key in ("components", "global_params", "native_q", "native_fit", "warnings")}


def _candidates(raw, result, options, seconds):
    """Every composition tried, best first: its numbers and its row for Model."""
    # By name, as ``fit1d_method.load_kit``: the kit is an optional folder beside the program (on sys.path once loaded).
    kit_results = importlib.import_module("gisaxs1d_kit.results")

    tried = list(raw.get("results") or ())
    in_set = [r.get("composition_text") for r in tried if r.get("in_candidate_set")]
    candidates = []
    for rank, entry in enumerate(tried[:MAX_CANDIDATES], start=1):
        alone = kit_results.gui_result(dict(raw, results=[entry]), model_name=result["summary"]["model"],
                                       kit_version=result.get("kit_version") or "")
        own = dict(options, components=[SHAPE_IDS[p["shape"]] for p in alone["populations"] if p["shape"] in SHAPE_IDS])
        row = solution_row(alone, _item(alone), own, seconds)
        text = entry.get("composition_text")
        candidates.append(dict(
            rank=rank, composition=text, chi2_per_point=entry.get("chi2_per_point"), delta_bic=entry.get("delta_bic"),
            indistinguishable=bool(entry.get("indistinguishable")),
            indistinguishable_from=[other for other in in_set if other != text] if entry.get("indistinguishable") else [],
            flags=alone["summary"]["flags"], messages=alone["summary"]["messages"], table=alone["table"],
            mapping_row=_mapping(row)))
    return candidates


def _chi2(result) -> tuple:
    summary = result.get("summary") or {}
    chi2 = summary.get("chi2")
    if not isinstance(chi2, (int, float)):
        chi2 = float(summary.get("chi2_per_point") or 0.0) * float(summary.get("n_points") or 1)
    return float(chi2), summary.get("dof")


def _settled(fit, constraints, result, raw, q_min, q_max, rounds, cancelled, say):
    """Widen what stops at an edge (``resolve``), then give what stays there as a bound (``settle``)."""
    resolved = resolve(fit, constraints, result, raw, q_min, q_max, rounds=rounds, cancelled=cancelled, report=say)
    return settle(fit, resolved, cancelled=cancelled, report=say) if rounds else resolved


def run_single_fit1d(payload, report, is_cancelled):
    options = validate_options(payload["options"])
    composition = str(payload.get("composition") or "model")
    auto = composition.startswith("auto")
    constraints = [Constraint.from_dict(item) for item in payload.get("constraints") or ()]
    rounds = int(payload.get("rounds", 2) or 0)
    if is_cancelled():
        raise RuntimeError("Fitting cancelled")
    report(0, 1, "Loading fit1d", {})
    core = load_kit()
    arguments = dict(q=payload["q"], intensity=payload["intensity"], sigma=payload.get("sigma"), q_unit="nm^-1",
                     composition=composition if auto else ",".join(FIT1D_SHAPES[t] for t in options["components"]),
                     mode=options["fit1d_mode"], policy=options["fit1d_policy"], model=options["fit1d_model"],
                     qz=float(options["fit1d_qz"] or 0.0) if options["fit1d_mode"] == "gisaxs" else 0.0,
                     sigma_floor=float(options["fit1d_sigma_floor"]))
    q = np.abs(np.asarray(payload["q"], float))
    q_min, q_max = float(q[q > 0].min()), float(q.max())
    started = time.perf_counter()
    report(0, 1, "Trying compositions with fit1d" if auto else "Fitting the particles in Model with fit1d", {})
    result, raw = _fit(core, arguments, constraints)
    if is_cancelled():
        raise RuntimeError("Fitting cancelled")
    tried = raw
    if auto:  # the best composition from here on, by its name (its populations are numbered the same)
        arguments = dict(arguments, composition=result["summary"]["composition"])
    fit = lambda trial: _fit(core, arguments, trial)  # noqa: E731
    say = lambda line: report(0, 1, line, {})  # noqa: E731
    resolved = _settled(fit, constraints, result, raw, q_min, q_max, rounds, is_cancelled, say)
    follow = None
    previous = payload.get("previous") or []
    if previous and not auto:  # a frame of a series: also its continuation of the previous frame
        say("Fitting the continuation of the previous frame")
        trial = window(previous, constraints)
        near = _settled(fit, trial, *fit(trial), q_min, q_max, rounds, is_cancelled, say)
        chi2_free, _dof = _chi2(resolved.result)
        chi2_follow, _ = _chi2(near.result)
        points = int(resolved.result["summary"].get("n_points") or 1)
        note = follow_note(chi2_free, chi2_follow, points)
        follow = dict(followed=same_basin(chi2_free, chi2_follow, points), chi2_free=chi2_free,
                      chi2_follow=chi2_follow, tolerance=tolerance(chi2_free, points), previous=previous)
        if follow["followed"]:
            resolved = near
        resolved.actions.append(note)
    if is_cancelled():
        raise RuntimeError("Fitting cancelled")
    result, raw = resolved.result, resolved.raw
    mark_bounds(result, resolved.constraints)
    seconds = time.perf_counter() - started
    row = solution_row(result, _item(result), options, seconds)
    actions = [note.to_dict() for note in resolved.actions]
    remaining = [dict(edge.to_dict(), note=remaining_note(edge).to_dict()) for edge in resolved.edges]
    result["warnings"] = row["warnings"] + [item["note"]["text"] for item in remaining]
    result["measurement_sigma"] = payload.get("sigma")
    result["settings"] = options
    result["fit1d_constraints"] = dict(
        composition=composition, fitted_composition=result["summary"]["composition"],
        given=[item.to_dict() for item in constraints], used=[item.to_dict() for item in resolved.constraints],
        population_order=payload.get("population_order") or [],
        mapping="R = R_nm, sigma_R = sigma_R_rel (number-weighted), D = D_nm, sigma_D = w_rel / sqrt(2 pi)")
    measured = measurable(q_min, q_max)
    result["fit1d_bounds"] = dict(measurable={key: list(value) for key, value in measured.items()},
                                  q_range_nm_inv=[q_min, q_max], actions=actions, edges=remaining,
                                  refits=resolved.refits)
    candidates = _candidates(tried, result, options, seconds) if auto else []
    if candidates:  # the best, as resolved
        best = _candidates(dict(tried, results=[raw["results"][0]]), result, options, seconds)[0]
        candidates[0].update({key: best[key] for key in ("chi2_per_point", "flags", "messages", "table", "mapping_row")})
    result["candidates"] = [{key: value for key, value in item.items() if key not in ("mapping_row", "table")}
                            for item in candidates] or result.get("candidates")
    if follow is not None:
        result["fit1d_follow"] = follow
    report(1, 1, "fit1d finished", {})
    return _clean(dict(native=result, mapping_row=row, candidates=candidates))


__all__ = ["MAX_CANDIDATES", "run_single_fit1d"]
