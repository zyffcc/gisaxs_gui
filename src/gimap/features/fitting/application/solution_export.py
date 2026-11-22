"""The solutions of Predict parameters (or of Analyze) as records and tables: what Save ▸ Data and Fit adds to its JSON
record (``solutions_block``) and what Save ▸ Solutions… writes (``solutions_table``, ``solutions_curves``).

Every solution with its composition, evidence probability or BIC weight, χ²ᵣ and fitted points, flags, whether it is
incomplete (Predict was stopped before every solution was sampled: its probability is over the solutions finished),
and every parameter of its model: the value (the refined value of least squares), whether it was free, and where the
posterior gives one (the shapes, spacings and beam tail; not the scales and background, solved for each sample) the
posterior median, the 68 % and 95 % intervals and the least-squares error. ``±`` in the Results table of a solution is
half the width of its 68 % interval (``half_width``).

A solution is any object with ``model``, ``chi2``, ``source`` and, for Predict, ``composition``, ``probability``,
``intervals`` ((name, (best, median, low68, high68, low95, high95, error)) …), ``flags``, ``points`` (fitted, given,
below the network's q window, unusable), ``ess``, ``incomplete`` (the page's ``Solution``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from src.gimap.shared import fit_names

from ..domain.fit_engine import FitData, residuals
from ..domain.fit_model import FitModel, INFO, evaluate, model_to_dict
from .fit_state import number, path_key

DIAGNOSTIC_COLUMNS = ("compared_by", "interval_kind", "posterior_kind", "ess", "ess_source", "log_evidence", "log_evidence_source",
                      "log_evidence_mc_error", "composition_log_evidence", "composition_log_evidence_mc_error",
                      "point_count", "points_per_decade", "peak_q_nm_inv", "peak_spacing_nm_inv",
                      "validation_spacing_nm_inv")
POSTERIOR_COLUMNS = ("least_squares_value", "posterior_median", "posterior_low68", "posterior_high68",
                     "posterior_low95", "posterior_high95", "reported_interval_kind")
INTERVAL_FIELDS = ("median", "low68", "high68", "low95", "high95", "error")
GLOBAL_NAMES = {"tail_width": "res_width", "tail_exponent": "res_exponent", "q_resolution": "q_resolution",
                "sigma_Res": "res_width", "nu_Res": "res_exponent"}
"""The posterior's names of values of the whole curve → the keys of the model's globals."""
INTERVALS_NOTE = ("least_squares_value is the value in Model. posterior_median and posterior_low68/high68 are "
                  "the sampled central quantiles, before extension to the optimum. The reported low68/high68 and "
                  "low95/high95 include the optimum. On BIC fallback these use least-squares errors where available. "
                  "The legacy median field can then be the least-squares value. A missing sampled estimate is null.")
PLUS_MINUS_NOTE = "± in the Results table of a Predict solution: half the width of its 68 % interval"
TABLE_COLUMNS = ("solution", "label", "composition", "in_model", "incomplete", "probability", "chi2_reduced",
                 "points_fitted", "points_given", "flags", "parameter", "name", "unit", "free", "value",
                 *INTERVAL_FIELDS, *POSTERIOR_COLUMNS, *DIAGNOSTIC_COLUMNS)


def interval_path(name: str, model: FitModel) -> Optional[tuple]:
    """The model's path of a posterior value (``R_1`` → (0, "R"), ``tail_width`` → ("globals", "res_width"));
    ``None`` when the model has no such value. Populations are numbered as Predict's models hold them."""
    key, slot = fit_names.split_name(name)
    if slot is not None:
        owner = slot - 1
        if 0 <= owner < len(model.components) and key in model.components[owner].keys():
            return owner, key
        return None
    key = GLOBAL_NAMES.get(key, key)
    return ("globals", key) if key in model.globals else None


def intervals_by_path(solution) -> dict:
    """{path: (best, median, low68, high68, low95, high95, error)} of a solution's posterior values."""
    found = {}
    for name, values in getattr(solution, "intervals", ()) or ():
        path = interval_path(name, solution.model)
        if path is not None:
            found[path] = tuple(values)
    return found


def posterior_by_path(solution) -> dict:
    """Unextended sampled quantiles, including those distinct from the BIC fallback intervals."""
    summary = getattr(solution, "posterior_summary", {})
    if summary or getattr(solution, "diagnostics", {}) or "bic" in getattr(solution, "flags", ()):
        return {interval_path(name, solution.model): values for name, values in summary.items()}
    # Old records retained the sampled median, but extended their interval to the optimum.
    return {path: {"median": values[1]} for path, values in intervals_by_path(solution).items()
            if len(values) > 1 and values[1] is not None}


def half_width(values) -> Optional[float]:
    """Half the width of a 68 % interval (``None`` when it has no ends)."""
    low, high = values[2], values[3]
    if low is None or high is None or not np.isfinite(low) or not np.isfinite(high):
        return None
    return 0.5 * (float(high) - float(low))


def composition_of(solution) -> str:
    """The composition's name (``sphere (S) + vertical cylinder``) or, for another solution, where it came from."""
    name = getattr(solution, "composition", "") or ""
    return name or str(solution.source).split(" · ", 1)[-1]


def numbered_names(solutions: Sequence) -> list[str]:
    """The compositions, ``#1``, ``#2`` … added where two solutions have the same one."""
    names = [composition_of(solution) for solution in solutions]
    out, seen = [], {}
    for name in names:
        if names.count(name) > 1:
            seen[name] = seen.get(name, 0) + 1
            out.append(f"{name} #{seen[name]}")
        else:
            out.append(name)
    return out


def parameter_rows(solution) -> list[dict]:
    """Every parameter of the solution's model: value, free, unit and, where the posterior gives them, its median,
    68 % and 95 % intervals and the least-squares error."""
    model, posterior = solution.model, intervals_by_path(solution)
    summary = posterior_by_path(solution)
    many = len(model.components) > 1
    rows = []
    for path, parameter in model.parameters():
        owner, key = path
        name = key if owner == "globals" else f"{key}_{owner + 1}"
        row = {"parameter": path_key(path), "name": fit_names.display_name(name, many), "unit": INFO[key].unit,
               "free": bool(parameter.free), "value": number(parameter.value)}
        values = posterior.get(path)
        for index, field in enumerate(INTERVAL_FIELDS, start=1):
            row[field] = None if values is None or values[index] is None else number(values[index])
        sampled = summary.get(path, {})
        row.update(least_squares_value=row["value"], posterior_median=sampled.get("median"),
                   posterior_low68=sampled.get("low68"), posterior_high68=sampled.get("high68"),
                   posterior_low95=sampled.get("low95"), posterior_high95=sampled.get("high95"))
        row["reported_interval_kind"] = (fit_names.reported_interval_kind(
            solution.flags, values[6], getattr(solution, "diagnostics", {}).get("posterior_kind")) if values else None)
        rows.append(row)
    return rows


def points_record(points) -> Optional[dict]:
    if not points:
        return None
    fitted, given, below, unusable = (tuple(points) + (0, 0, 0))[:4]
    return {"fitted": int(fitted), "given": int(given), "below_window": int(below), "unusable": int(unusable)}


def solution_record(solution, number_: int, label: str, in_model: bool) -> dict:
    probability = getattr(solution, "probability", None)
    diagnostics = dict(getattr(solution, "diagnostics", {}) or {})
    flags = list(getattr(solution, "flags", ()) or ())
    key = fit_names.weight_key({**diagnostics, "flags": flags})
    if probability is not None:
        diagnostics.setdefault("compared_by", "bic" if key == "bic_weight" else "evidence")
        diagnostics.setdefault("interval_kind", "bic" if key == "bic_weight" else
                               "tempered" if "low_ess" in flags else "evidence")
    return {**diagnostics, "solution": number_, "label": label, "composition": composition_of(solution), "source": solution.source,
            "in_model": bool(in_model), "incomplete": bool(getattr(solution, "incomplete", False)),
            key: None if probability is None else number(probability),
            "chi2_reduced": number(solution.chi2), "points": points_record(getattr(solution, "points", ())),
            "flags": list(getattr(solution, "flags", ()) or ()), "ess": number(getattr(solution, "ess", None)),
            "model": model_to_dict(solution.model), "parameters": parameter_rows(solution)}


def solutions_block(solutions: Sequence, model: FitModel, *, refined_from: Optional[dict] = None,
                    run: Optional[dict] = None) -> dict:
    """The solutions in a record (Save ▸ Data and Fit's JSON, Save ▸ Solutions…'s JSON): every one, which is in
    Model, where the values of a fit came from (``refined_from``: a refinement of a solution) and the run
    (``run``: the network, how long it took, whether it was stopped)."""
    labels = numbered_names(solutions)
    in_model = next((index for index, solution in enumerate(solutions) if solution.model == model), None)
    incomplete = any(getattr(solution, "incomplete", False) for solution in solutions)
    weight_label = "BIC weights" if any("bic" in s.flags for s in solutions) else "probabilities"
    return {
        "solutions": [solution_record(solution, index + 1, labels[index], index == in_model)
                      for index, solution in enumerate(solutions)],
        "in_model": None if in_model is None else in_model + 1, "refined_from": refined_from, "run": run,
        weight_label: ("over the solutions finished before Predict was stopped" if incomplete
                       else "relative to the compositions that were tried"),
        "intervals": INTERVALS_NOTE, "plus_minus": PLUS_MINUS_NOTE,
        "chi2_reduced": fit_names.CHI2_DEFINITION,
    }


def solutions_table(solutions: Sequence, model: FitModel) -> tuple[list[str], list[list]]:
    """One row per solution and parameter (``TABLE_COLUMNS``): the solution's own columns repeat on its rows."""
    labels = numbered_names(solutions)
    rows = []
    keys = {fit_names.weight_key({"flags": s.flags}) for s in solutions}
    weight_columns = [key for key in ("probability", "bic_weight") if key in keys] or ["probability"]
    columns = list(TABLE_COLUMNS)
    columns[5:6] = weight_columns
    composition_columns = [key for key in ("composition_probability", "composition_bic_weight")
                           if any(key in getattr(s, "diagnostics", {}) for s in solutions)]
    columns += composition_columns + ["composition_mc_low68", "composition_mc_high68"]
    for index, solution in enumerate(solutions):
        record = solution_record(solution, index + 1, labels[index], solution.model == model)
        points = record["points"] or {}
        head = [index + 1, labels[index], record["composition"], int(record["in_model"]), int(record["incomplete"]),
                *(record.get(key) for key in weight_columns), record["chi2_reduced"], points.get("fitted"), points.get("given"),
                ";".join(record["flags"])]
        for row in record["parameters"]:
            rows.append(head + [row["parameter"], row["name"], row["unit"], int(row["free"]), row["value"],
                                *(row[field] for field in (*INTERVAL_FIELDS, *POSTERIOR_COLUMNS)),
                                *(record.get(field) for field in DIAGNOSTIC_COLUMNS + tuple(composition_columns)),
                                *(record.get("composition_probability_mc68") or [None, None])])
    return columns, rows


def _column_name(name: str) -> str:
    return name.replace(" ", "_").replace(":", "")


def solutions_curves(solutions: Sequence, data: FitData) -> tuple[list[str], np.ndarray]:
    """The points fitted (q in nm⁻¹ and Å⁻¹, I, σ) and of every solution k its model, residual and terms
    (``s<k>_model``, ``s<k>_residual``, ``s<k>_<term>``)."""
    names = ["q_nm^-1", "q_A^-1", "I", "sigma"]
    columns = [data.q, data.q / 10.0, data.intensity,
               data.sigma if data.sigma is not None else np.full_like(data.q, np.nan)]
    for index, solution in enumerate(solutions, start=1):
        total, parts = evaluate(solution.model, data.q, parts=True)
        names += [f"s{index}_model", f"s{index}_residual"]
        columns += [total, residuals(solution.model, data)]
        for name, curve in parts.items():
            names.append(f"s{index}_{_column_name(name)}")
            columns.append(curve)
    return names, np.column_stack(columns)


def table_text(names: Sequence[str], rows: Sequence[Sequence]) -> str:
    """CSV text of a table whose cells are numbers, text or empty (``None``)."""
    import csv
    import io

    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(names)
    for row in rows:
        writer.writerow(["" if value is None else value for value in row])
    return stream.getvalue()


__all__ = ["GLOBAL_NAMES", "INTERVALS_NOTE", "INTERVAL_FIELDS", "PLUS_MINUS_NOTE", "TABLE_COLUMNS", "composition_of",
           "half_width", "interval_path", "intervals_by_path", "numbered_names", "parameter_rows", "points_record",
           "solution_record", "solutions_block", "solutions_curves", "solutions_table", "table_text"]
