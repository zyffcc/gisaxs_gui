"""Inputs and numerical references saved with fitted curves and posterior solutions."""

from __future__ import annotations

import numpy as np

from src.gimap.shared.reproduction import replay_header

from ..domain.fit_model import model_from_dict, model_to_dict


def input_points(data):
    return {"q": data.q.tolist(), "intensity": data.intensity.tolist(),
            "sigma": None if data.sigma is None else data.sigma.tolist(),
            "dq": None if data.dq is None else data.dq.tolist(), "weight_mode": data.weight_mode}


def model_tables(model, data, result=None, curve_table=None):
    from .single_fit import export_table

    names, columns = export_table(model, data) if curve_table is None else curve_table
    values = {f"{owner}_{key}": [float(parameter.value)] for (owner, key), parameter in model.parameters()}
    tables = {"fit_curve": dict(zip(names, columns.T)), "parameters": values}
    if result is not None:
        values.update({f"{owner}_{key}_error": [value] for (owner, key), value in result.errors.items()})
        values.update(chi2_reduced=[result.chi2_reduced], log_rmse=[result.log_rmse])
        if result.covariance:
            tables["covariance"] = {f"{owner}_{key}": column for (owner, key), column in
                                    zip(result.covariance_paths, np.asarray(result.covariance).T)}
    return tables


def solver_inputs(result):
    return {"start_model": model_to_dict(result.start_model), "method": result.method,
            "seed": result.seed, "max_evaluations": result.evaluation_budget,
            "population_seeds": [model_to_dict(model) for model in result.population_seeds],
            "stopped_after": result.stopped_after, "restarts": result.restarts, "restart_budget": result.restart_budget}


def fit_recipe(model, data, result, source=None, curve_table=None):
    # A fit restored from an older project only has its final values.
    if result is not None and result.start_model is None:
        return None
    recipe = replay_header("fit", model_tables(model, data, result, curve_table))
    recipe.update(operation="manual" if result is not None else "evaluate", points=input_points(data))
    if source is not None:
        recipe["source"] = source
    if result is None:
        recipe["model"] = model_to_dict(model)
    else:
        recipe.update(solver_inputs(result))
        recipe["clear_start_errors"] = bool(result.start_kept and
                                            all(np.isnan(v) for v in result.errors.values()))
    return recipe


def predict_tables(encoded, data, display_q=None, sign=1, *, report_version=2):
    from ..domain.fit_model import evaluate

    tables = {}
    for i, row in enumerate(encoded, 1):
        model = model_from_dict(row["model"])
        for name, columns in model_tables(model, data).items():
            tables[f"solution_{i}_{name}"] = columns
        summary = tables[f"solution_{i}_parameters"]
        weight = "bic_weight" if "bic_weight" in row else "probability"
        summary[weight if report_version >= 2 else "probability"] = [np.nan if row[weight] is None else row[weight]]
        for name in ("chi2", "log_rmse", "log_evidence", "ess", "samples", "composition"):
            summary[name] = [np.nan if row[name] is None else row[name]]
        if report_version >= 2:
            for name, quantiles in row["posterior_summary"].items():
                for metric, value in quantiles.items():
                    summary[f"{name}_posterior_{metric}"] = [value]
            for name, value in row["diagnostics"].items():
                if isinstance(value, (int, float)):
                    summary[name] = [value]
            limits = row["diagnostics"].get("composition_probability_mc68")
            if limits is not None:
                summary["composition_mc_low68"], summary["composition_mc_high68"] = [limits[0]], [limits[1]]
        for name, interval in row["values"].items():
            for metric, value in zip(("best", "median", "low68", "high68", "low95", "high95", "error"), interval):
                summary[f"{name}_{metric}"] = [np.nan if value is None else value]
        if display_q is not None:
            table = tables.setdefault("display_curves", {"q_nm^-1": sign * display_q})
            table[f"solution_{i}"] = evaluate(model, display_q)
    return tables


def points_from_record(points):
    from ..domain.fit_engine import FitData

    # Already prepared in exactly this order, including tied |q| from opposite halves.
    sigma, dq = points["sigma"], points.get("dq")
    return FitData(np.asarray(points["q"], float), np.asarray(points["intensity"], float),
                   None if sigma is None else np.asarray(sigma, float), None if dq is None else np.asarray(dq, float),
                   points.get("weight_mode", "file"))


def prediction_record(plan, encoded):
    data = points_from_record(plan["points"])
    display = plan.get("display_q")
    tables = predict_tables(encoded, data, None if display is None else np.asarray(display), plan.get("sign", 1),
                            report_version=plan.get("report_version", 1))
    return {**replay_header("fit", tables), **plan}


def with_prediction(record, predict_run, selected):
    """Keep the prediction as well as the active fit, even after changing the points or refining."""
    prediction = (predict_run or {}).get("reproduction")
    recipe = record.get("reproduction")
    if prediction is None or recipe is None:
        return
    recipe["prediction"] = {k: v for k, v in prediction.items() if k != "reference"}
    recipe["reference"].update(prediction["reference"])
    if selected is not None and recipe["operation"] == "evaluate":
        recipe.update(operation="selected_solution", solution_index=selected)
        del recipe["model"]


def joint_tables(result, curves):
    from .joint_fit import curve_errors, key_name

    tables = {"joint_fit": {"chi2_reduced": [result.chi2_reduced]}}
    for i, (curve, model) in enumerate(zip(curves, result.models)):
        each = model_tables(model, curve.data)
        each["parameters"].update({f"{owner}_{key}_error": [value] for (owner, key), value in
                                   curve_errors(result, i).items()})
        each["parameters"]["chi2_per_point"] = [result.curve_chi2[i]]
        tables.update({f"curve_{i + 1}_{name}": columns for name, columns in each.items()})
    if result.covariance:
        tables["covariance"] = {key_name(key): column for key, column in zip(result.covariance_keys,
                                                                             np.asarray(result.covariance).T)}
    return tables


def joint_recipe(result, specs):
    from .curve_sources import source_record

    curves = result.start_curves
    inputs = [{"name": c.name, "points": input_points(c.data), "start_model": model_to_dict(c.model),
               "source": spec.source or (source_record(spec.path) if spec.path else None)}
              for c, spec in zip(curves, specs)]
    return {**replay_header("fit", joint_tables(result, curves)), "operation": "joint", "curves": inputs,
            "shared": [list(path) for path in sorted(result.shared, key=str)], "method": result.method,
            "max_evaluations": result.evaluation_budget, "seed": result.seed,
            "stop_after_calls": (result.evaluations - 1) * len(curves) if result.stopped else None}
