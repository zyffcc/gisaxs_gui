"""Repeat the recorded optimizer or posterior with the exact points originally passed to it."""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import replace
from itertools import count
import json
import math

import numpy as np

from src.gimap.shared.reproduction import file_reference, verified_path

from ..application.fit_reproduction import joint_tables, model_tables, points_from_record, predict_tables
from ..application.single_fit import converged_again
from ..domain.fit_engine import fit
from ..domain.fit_model import model_from_dict
from ..domain.posterior import parallel, refine
from ..domain.posterior.flow import Network


def predict_options(payload):
    """Materialize every numerical option of the Predict call, including its default sampling budgets."""
    defaults = dict(qz=0., qz_minus=None, compositions=None, allowed=None, seed=0, law="number_gauss",
                    coupling="lma", grazing=None, shortlist=refine.SHORTLIST, samples=refine.SAMPLES,
                    local_samples=refine.LOCAL_SAMPLES, refine_modes=refine.REFINED, explored=refine.EXPLORED,
                    model_error_floor=refine.MODEL_ERROR_FLOOR, min_probability=0.01,
                    reflected_beam=None, known_heights=None, measured_qz=None)
    options = {key: payload.get(key, value) for key, value in defaults.items()}
    if payload.get("substrate") is not None:
        options["substrate"] = payload["substrate"]
        if payload.get("upright_heights"):
            options["upright_heights"] = payload["upright_heights"]
    options["qz"] = float(options["qz"] or 0.)
    options["qz_minus"] = None if options["qz_minus"] is None else float(options["qz_minus"])
    options["seed"] = int(options["seed"])
    return options


def prediction_plan(points, options, network_path, network_meta, *, display_q=None, sign=1, limit=None, source=None):
    manifest = network_path.with_suffix(".json")
    if json.loads(manifest.read_text(encoding="utf-8")) != network_meta:
        raise ValueError(f"The network record changed after loading: {manifest}")
    plan = {"operation": "predict", "report_version": 2, "points": points, "options": options,
            "weights": {**file_reference(network_path), "sha256": network_meta["sha256"]},
            "network_record": file_reference(manifest)}
    if display_q is not None:
        plan.update(display_q=display_q.tolist(), sign=sign)
    if limit is not None:
        plan["limit"] = limit
    if source is not None:
        plan["source"] = source
    return plan


def _predict(plan, folder, should_cancel):
    from .adapters.posterior_job import encode

    if plan.get("source") is not None:
        verified_path(plan["source"], folder)
    weights = verified_path(plan["weights"], folder)
    manifest = verified_path(plan["network_record"], folder)
    if manifest != weights.with_suffix(".json"):
        raise ValueError(f"Expected the network record beside its weights: {weights.with_suffix('.json')}")
    network = Network.load(weights)
    data = points_from_record(plan["points"])
    partial_solutions = None
    completed = 0

    def partial(solutions):
        nonlocal partial_solutions, completed
        completed += 1
        if completed == plan.get("partial_after"):
            partial_solutions = solutions

    solutions = refine.predict(network, data.q, data.intensity, data.sigma, **plan["options"], partial=partial,
                               cancelled=lambda: _check_cancel(should_cancel), threads=parallel.default_threads())
    if plan.get("partial_after") is not None:
        if partial_solutions is None:
            raise ValueError("Predict did not reach the partial solution recorded when the fit stopped.")
        solutions = partial_solutions
    solutions = solutions[:plan.get("limit")]
    grid = plan.get("display_q")
    encoded = [encode(s) for s in solutions]
    tables = predict_tables(encoded, data, None if grid is None else np.asarray(grid), plan.get("sign", 1),
                            report_version=plan.get("report_version", 1))
    return solutions, tables


def _check_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        raise CancelledError()
    return False


def _stop_callback(limit, should_cancel):
    calls = count()
    def stop():
        _check_cancel(should_cancel)
        return limit is not None and next(calls) >= limit
    return stop


def _joint(recipe, record_dir, should_cancel):
    from ..domain.joint_fit import JointCurve, fit_curves

    curves = []
    for spec in recipe["curves"]:
        if spec["source"]:
            verified_path(spec["source"], record_dir)
        curves.append(JointCurve(spec["name"], points_from_record(spec["points"]), model_from_dict(spec["start_model"])))
    result = fit_curves(curves, recipe["shared"], method=recipe["method"], seed=recipe["seed"],
                        max_evaluations=recipe["max_evaluations"],
                        stop=_stop_callback(recipe["stop_after_calls"], should_cancel))
    return joint_tables(result, curves)


def replay_fit(record, record_dir, *, should_cancel=None):
    recipe = record["reproduction"]
    source = recipe.get("source")
    method = recipe["operation"]
    if method == "joint":
        return _joint(recipe, record_dir, should_cancel)
    if method == "predict":
        return _predict(recipe, record_dir, should_cancel)[1]
    if source is not None:
        verified_path(source, record_dir)
    tables, predicted = {}, []
    if "prediction" in recipe:
        predicted, tables = _predict(recipe["prediction"], record_dir, should_cancel)
    data = points_from_record(recipe["points"])
    result = None
    if method == "manual":
        start = (predicted[recipe["start_solution_index"]].model if "start_solution_index" in recipe else
                 model_from_dict(recipe["start_model"]))
        result = fit(start, data, method=recipe["method"], max_evaluations=recipe["max_evaluations"], seed=recipe["seed"],
                     stop=_stop_callback(recipe["stopped_after"], should_cancel),
                     seeds=tuple(model_from_dict(m) for m in recipe["population_seeds"]))
        result = converged_again(result, data, restarts=recipe.get("restarts", 0),
                                 max_evaluations=recipe.get("restart_budget"), stop=lambda: _check_cancel(should_cancel))
        if recipe.get("clear_start_errors") and result.start_kept:
            result = replace(result, errors={p: math.nan for p in result.errors}, correlated=())
        model = result.model
    elif method == "selected_solution":
        model = predicted[recipe["solution_index"]].model
    elif method == "evaluate":
        model = model_from_dict(recipe["model"])
    else:
        raise ValueError(f"Unknown fit operation {method!r}")
    tables.update(model_tables(model, data, result))
    return tables
