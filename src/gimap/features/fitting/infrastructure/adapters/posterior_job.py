"""Predict parameters in a job process: GIMaP's proposal network, the importance-weighted posterior of the
compositions asked for, each mode refined by least squares (``domain.posterior.refine.predict``).

Payload: q (nm⁻¹), intensity, sigma (or None), qz (k_i + k_f, nm⁻¹; 0: the effective interface), qz_minus (k_i − k_f
when both angles are known: spheres with the exit-reflected path; None otherwise), compositions (indices of
``encoding.COMPOSITIONS``) or allowed (the head chooses among them), method, samples, seed, law, coupling, grazing
(the person's angles and λ to keep; the reflection too when qz_minus is None). The result is JSON: every solution's
model (``model_to_dict``), χ²ᵣ (over the fitted points, with the curve's σ: ``refine``'s, the one number every text
shows), probability, values with their posterior intervals, flags (``shared/fit_names.clean_flags``: none for a term
that is absent), ``points`` (``fitted_points``: how many were fitted, how many lay below the network's q window);
``tried`` (the compositions given; None: the network chose) and the network's name and SHA-256.

While it runs, the job reports its stages (``StageReport``): the details of a report say the stage (``loading``,
``proposals``, ``refinement``, ``starts``, ``search``, ``sampling``), how many of its items are done of how many, the
seconds since the start and an estimate of the seconds left in the stage; after each solution's posterior is sampled
(but the last) they also carry ``partial``: the solutions sampled so far, encoded as the result's, their probabilities
over those alone — what the page keeps when Predict is stopped. The numbers computed are those of a run without
reports (``refine.predict``).
"""

from __future__ import annotations

import math
import time

import numpy as np

from src.gimap.shared import fit_names

from ...domain import fit_model as fm
from ...domain.posterior import network as posterior_network
from ...domain.posterior import refine


def _number(value):
    return None if value is None or not math.isfinite(float(value)) else float(value)


def amplitudes(model) -> dict:
    """The linear terms of a model by their keys (``res_amplitude``, ``background``, ``Int_1`` …)."""
    found = {key: float(model.globals[key].value) for key in ("res_amplitude", "background")}
    found.update({f"Int_{n}": float(c.value("Int")) for n, c in enumerate(model.components, start=1)})
    return found


def solution_flags(solution) -> tuple:
    """The solution's flags without those of an absent term (Tail A = 0, a population with Scale 0)."""
    return fit_names.clean_flags(solution.flags, amplitudes(solution.model))


def fitted_points(q, intensity, sigma=None, qz: float = 0.0, qz_minus=None) -> dict:
    """Which of the points handed to Predict parameters it fits: ``fitted``, ``given``, ``below_window`` (q below
    q_max / 300, outside the network's window), ``unusable`` (not finite, q = 0, σ ≤ 0) and ``q_min`` of the fitted
    ones. χ²ᵣ of a solution is over the fitted points."""
    from ...domain.posterior.infer import Cut

    cut = Cut.prepare(q, intensity, sigma, float(qz or 0.0), qz_minus)
    given = int(np.size(q))
    return {"fitted": int(cut.q.size), "given": given, "below_window": int(cut.dropped),
            "unusable": max(0, given - int(cut.q.size) - int(cut.dropped)),
            "q_min": float(cut.q.min()) if cut.q.size else None}


def encode(solution, points=None) -> dict:
    extra = solution.extras or {}
    diagnostics = dict(extra.get("diagnostics", {}))
    diagnostics["reported_interval_kinds"] = {
        name: fit_names.reported_interval_kind(solution.flags, v.error, diagnostics.get("posterior_kind"))
        for name, v in solution.values.items()}
    key = fit_names.weight_key({"compared_by": extra.get("compared_by"), "flags": solution.flags})
    return {
        "diagnostics": diagnostics, "posterior_summary": extra.get("posterior_summary", {}),
        "importance_starts": extra.get("importance_starts", []),
        "least_squares_values": {**{name: _number(v.best) for name, v in solution.values.items()},
                                 **{key: _number(solution.model.globals[key].value) for key in refine.REFLECTION_KEYS
                                    if key in extra.get("posterior_summary", {})}},
        "points": None if points is None else dict(points),
        "compared_by": (solution.extras or {}).get("compared_by"),
        "composition": int(solution.composition), "name": solution.name, key: _number(solution.probability),
        "model": fm.model_to_dict(solution.model), "chi2": _number(solution.chi2_reduced),
        "log_rmse": _number(solution.log_rmse), "method": solution.method, "flags": list(solution_flags(solution)),
        "log_evidence": _number(diagnostics.get("log_evidence", solution.log_evidence)), "ess": _number(solution.ess), "samples": solution.samples,
        "values": {name: [_number(v.best), _number(v.median), _number(v.low68), _number(v.high68), _number(v.low95),
                          _number(v.high95), _number(v.error)] for name, v in solution.values.items()},
    }


STAGES = {"loading": (0.0, 0.02), "proposals": (0.02, 0.15), "refinement": (0.15, 0.5), "starts": (0.5, 0.55),
          "search": (0.5, 0.55), "sampling": (0.55, 1.0)}
"""Each stage's share of the bar (from, to): a guide only, the stages take other times on other curves."""
STAGE_LINES = {"loading": "Loading the proposal network", "proposals": "Network proposals: composition {number} of {total}",
               "refinement": "Refining the modes: composition {number} of {total}",
               "starts": "Refining from other starts: composition {number} of {total}",
               "search": "Searching the ranges for a composition the proposals miss",
               "sampling": "Sampling the posterior: solution {number} of {total}"}
"""The English line of each stage (the log; the page builds its own from the details)."""


class StageReport:
    """The job's reports of ``refine.predict``'s stages and of the solutions finished so far (module docstring)."""

    def __init__(self, report, points: dict, tick: float):
        self.report, self.points, self.tick = report, points, tick
        self.stage, self.started, self.last = "", tick, (0, 0)

    def progress(self, stage: str, done: int, total: int, **extra) -> None:
        """``done`` of ``total`` items of ``stage`` finished (the item after them is under way)."""
        now = time.perf_counter()
        if stage != self.stage:
            self.stage, self.started = stage, now
        self.last = (int(done), int(total))
        low, high = STAGES.get(stage, (0.0, 1.0))
        share = min(1.0, done / total) if total > 0 else 0.0
        left = (now - self.started) / done * (total - done) if 0 < done < total else None
        details = {"stage": stage, "done": int(done), "total": int(total), "elapsed": round(now - self.tick, 2),
                   "left": None if left is None else round(left, 1), **extra}
        line = STAGE_LINES.get(stage, stage).format(number=min(int(done) + 1, max(int(total), 1)), total=total)
        try:  # a report never stops the run
            self.report(low + (high - low) * share, 1, line, details)
        except Exception:  # noqa: BLE001 - e.g. a value JSON cannot take: the run goes on without this report
            pass

    def partial(self, solutions) -> None:
        """The solutions of those sampled so far (``refine.predict``'s ``partial``, right after a sampling)."""
        try:
            encoded = [encode(s, self.points) for s in solutions]
        except Exception:  # noqa: BLE001 - nothing to keep from this report; the run goes on
            return
        done, total = self.last
        self.progress("sampling", done + 1, total, partial=encoded)


def run_predict(payload: dict, report, cancelled) -> dict:
    tick = time.perf_counter()
    points = fitted_points(payload["q"], payload["intensity"], payload.get("sigma"), payload.get("qz") or 0.0,
                           None if payload.get("qz_minus") is None else float(payload["qz_minus"]))
    stages = StageReport(report, points, tick)
    stages.progress("loading", 0, 1)
    network = posterior_network.load()
    if cancelled():
        raise RuntimeError("Fitting cancelled")
    solutions = refine.predict(
        network, payload["q"], payload["intensity"], payload.get("sigma"), qz=float(payload.get("qz") or 0.0),
        qz_minus=None if payload.get("qz_minus") is None else float(payload["qz_minus"]),
        compositions=payload.get("compositions"), allowed=payload.get("allowed"),
        seed=int(payload.get("seed", 0)),
        law=payload.get("law", "number_gauss"), coupling=payload.get("coupling", "lma"),
        grazing=payload.get("grazing"), reflected_beam=payload.get("reflected_beam"),
        known_heights=payload.get("known_heights"), measured_qz=payload.get("measured_qz"), progress=stages.progress, partial=stages.partial)
    if cancelled():
        raise RuntimeError("Fitting cancelled")
    report(1, 1, "Done")
    tried = payload.get("compositions")
    return {"solutions": [encode(s, points) for s in solutions], "points": points,
            "tried": None if tried is None else len(tried),
            "network": {"name": posterior_network.path().stem, "sha256": network.meta.get("sha256")},
            "seconds": time.perf_counter() - tick}


TYPE_IDS = {"sphere": 1, "cylinder": 2, "vertical_cylinder": 3}
TYPE_NAMES = {"sphere": "sphere", "cylinder": "random_cylinder", "vertical_cylinder": "vertical_cylinder"}


def batch_compositions(types) -> dict:
    """The 1D Predict window's Complete composition (1 sphere, 2 random cylinder, 3 vertical cylinder; each with a
    paracrystal) as compositions; empty: the network chooses among all."""
    from ...domain.posterior import encoding

    if not types:
        return {"allowed": encoding.select()}
    if len(types) > encoding.SLOTS:
        raise ValueError(f"Predict parameters takes 1–{encoding.SLOTS} components")
    families = {1: "sphere", 2: "cylinder", 3: "vertical_cylinder"}
    return {"compositions": [encoding.composition_index([(families[int(t)], True) for t in types])]}


ROW_NAMES = {"res_width": "sigma_Res", "res_exponent": "nu_Res"}
"""Fitting's global names as the keys of the rows (the other values keep their keys; what a person reads of them:
``shared/fit_names.display_name``)."""


def flag_sentences(flags, many: bool = True) -> list[str]:
    """The flags as English sentences with the Model table's names (``shared/fit_names``)."""
    return fit_names.sentences(flags, many)


def at_bounds(model, tolerance: float = 1e-4) -> list[dict]:
    """The free shape and tail values of ``model`` on a bound of their range (within ``tolerance`` of it), as the
    rows list them: ``parameter``, ``bound`` (lower / upper), ``component`` (index; None for the tail). With
    Tail A = 0 the tail is absent and its width and exponent are not listed; nor are the values of a population
    whose Scale is 0."""
    out = []
    no_tail = float(model.globals["res_amplitude"].value) <= 0.0
    for path, parameter in model.parameters():
        owner, key = path
        if not parameter.free or key in ("Int", "background", "res_amplitude", "k", "q_resolution"):
            continue
        if owner == "globals" and (key not in ROW_NAMES or no_tail):
            continue
        if owner != "globals" and float(model.components[int(owner)].value("Int")) <= 0.0:
            continue
        lower, upper = parameter.lower, parameter.upper
        width = (upper - lower) if math.isfinite(upper) and math.isfinite(lower) else max(abs(parameter.value), 1.0)
        for bound, edge in (("lower", lower), ("upper", upper)):
            if math.isfinite(edge) and abs(parameter.value - edge) <= tolerance * width:
                out.append(dict(parameter=ROW_NAMES.get(key, key), bound=bound,
                                component=None if owner == "globals" else int(owner)))
    return out


def curve_rows(item, solutions, render_q, *, chosen_by: str) -> list[dict]:
    """Rows of a curve's solutions with what the analyses that screened a cut read of a row: ``rank``, ``converged``,
    ``at_bounds``, ``flags`` (cleaned of absent terms), sentences in ``warnings`` (the flags of the populations and
    of the solution) and ``more_warnings`` (those of background, tail and resolution), ``seconds``, ``algorithm``,
    ``conditions_source`` (``chosen_by``), ``unit_contract``."""
    rows = []
    points = fitted_points(item["q"], item["observed"], item["sigma"])
    for rank, solution in enumerate(solutions, start=1):
        row = solution_row(solution, item, render_q, points)
        row.update(
            rank=rank, converged=True, at_bounds=at_bounds(solution.model), seconds=float(solution.seconds), nfev=None, conditions_source=chosen_by,
            algorithm="Predict parameters: the proposal network's importance-weighted samples, each mode refined by "
                      "bounded least squares, the posterior sampled around the best solutions",
            unit_contract=dict(q="nm^-1", R_h_D="nm", sigma_R_h_D="relative standard deviation",
                               sigma_nn="sigma_nn/D, Hosemann; sigma_D = sigma_nn / sqrt(2 pi)", sigma_Res="nm^-1",
                               nu_Res="dimensionless", amplitudes="input intensity units"))
        rows.append(row)
    return rows


def solution_row(solution, item, render_q, points=None) -> dict:
    """A solution as a row of the 1D Predict window (``native_v5`` layout, Fitting's conventions: cut_model).
    ``best_chi2_weighted`` and ``chi2_reduced`` are the solution's χ²ᵣ (``refine``: over the fitted points, with the
    degrees of freedom), the same number wherever the solution is shown; ``points`` (``fitted_points``) says how
    many points that is and how many lay below the network's q window. The weighted RMS and the Huber loss are of
    the fitted points too. ``flags``: the solution's, cleaned of absent terms; ``warnings``: the sentences of those
    of the populations and of the solution, ``more_warnings``: of background, beam tail and resolution."""
    from ...domain.fit_model import evaluate

    model = solution.model
    q, y, sigma = (np.asarray(item[k], float) for k in ("q", "observed", "sigma"))
    points = points or fitted_points(q, y, sigma)
    fitted = evaluate(model, q)
    used = np.isfinite(q) & np.isfinite(y) & (sigma > 0)
    if points.get("q_min") is not None:
        used &= np.abs(q) >= points["q_min"] * (1 - 1e-12)
    residual = ((fitted - y) / sigma)[used]
    flags = solution_flags(solution)
    first, second = fit_names.split_flags(flags)
    many = len(model.components) > 1
    positive = (y > 0) & (fitted > 0)
    components = []
    for component in model.components:
        p = {key: float(component.value(key)) for key in component.keys() if key != "Int"}
        if "sigma_nn" in p:
            p["sigma_D"] = p["sigma_nn"] / math.sqrt(2 * math.pi)
        components.append(dict(type=TYPE_NAMES[component.family], type_id=TYPE_IDS[component.family], params=p,
                               amplitude=float(component.value("Int")), structure_factor=bool(component.structure)))
    total = sum(c["amplitude"] for c in components)
    for c in components:
        c["weight"] = c["amplitude"] / total if total > 0 else 0.0
    g = model.globals
    record = encode(solution)
    key = fit_names.weight_key(record)
    return dict(
        workflow="native_v5", physics_backend="cut_model", law=model.components[0].law if components else None,
        best_source="posterior", side=item["side"], combination=" + ".join(c["type"] for c in components),
        composition=solution.name, components=components,
        global_params=dict(background=float(g["background"].value), resolution_amplitude=float(g["res_amplitude"].value),
                           sigma_Res=float(g["res_width"].value), nu_Res=float(g["res_exponent"].value),
                           q_resolution=float(g["q_resolution"].value),
                           **{key: float(g[key].value) for key in ("alpha_i", "alpha_f", "wavelength")}),
        **{key: record[key]}, posterior=record["values"], posterior_summary=record["posterior_summary"],
        least_squares_values=record["least_squares_values"], diagnostics=record["diagnostics"],
        best_log_rmse=float(np.sqrt(np.mean(np.log(fitted[positive] / y[positive]) ** 2))) if positive.any() else None,
        best_chi2_weighted=float(solution.chi2_reduced), chi2_reduced=float(solution.chi2_reduced),
        points={key: value for key, value in points.items() if key != "q_min"}, signed_weighted_rms=float(np.sqrt(np.mean(residual**2))),
        signed_huber_delta2=float(np.mean(np.where(abs(residual) <= 2, 0.5 * residual**2, 2 * (abs(residual) - 1)))),
        comparison_label=("BIC weight" if key == "bic_weight" else "Posterior probability of the composition and mode"),
        flags=list(flags), warnings=flag_sentences(first, many), more_warnings=flag_sentences(second, many),
        compared_by=(solution.extras or {}).get("compared_by"), ess=_number(solution.ess),
        native_q=(q * item["sign"]).tolist(), observed=y.tolist(), sigma=sigma.tolist(), native_fit=fitted.tolist(),
        display_q=(render_q * item["sign"]).tolist(), display_fit=evaluate(model, render_q).tolist())


def fit_and_write(output, item, options, report, cancelled) -> list:
    """Predict parameters for one half of a curve of the 1D Predict window; its files as the other methods write."""
    import json

    network = posterior_network.load()
    report(0, 1, f"{item['side']}: weighting the network's samples", {})
    solutions = refine.predict(network, item["q"], item["observed"], item["sigma"],
                               qz=float(options.get("cut_qz") or 0.0), reflected_beam=options.get("reflected_beam"),
                               measured_qz=options.get("measured_qz"), **batch_compositions(options["components"]))
    if cancelled():
        raise RuntimeError("Prediction cancelled")
    q = np.asarray(item["q"], float)
    render_q = np.linspace(q.min(), q.max(), options["render_points"])
    rows = [solution_row(s, item, render_q) for s in solutions[:options["max_solutions"]]]
    if not rows:
        raise RuntimeError("Predict parameters found no solution for this curve")
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    output.mkdir(parents=True, exist_ok=True)
    (output / "solutions.json").write_text(json.dumps(dict(method="posterior", options=options, network={
        "name": posterior_network.path().stem, "sha256": network.meta.get("sha256")}, solutions=rows),
        indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "display.npz", reference_q=item["q"], observed=item["observed"], sigma=item["sigma"],
                        render_q=np.abs(rows[0]["display_q"]), rendered_curves=[r["display_fit"] for r in rows])
    np.savetxt(output / "fitting_curves.csv", np.column_stack([rows[0]["display_q"], *[r["display_fit"] for r in rows]]),
               delimiter=",", header="q_nm^-1," + ",".join(f"candidate_{i + 1}" for i in range(len(rows))), comments="")
    return rows


__all__ = ["STAGES", "STAGE_LINES", "StageReport", "amplitudes", "batch_compositions", "encode", "fit_and_write",
           "fitted_points", "flag_sentences", "run_predict", "solution_flags", "solution_row"]
