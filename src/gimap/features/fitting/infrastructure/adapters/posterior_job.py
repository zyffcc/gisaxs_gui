"""Predict parameters in a job process: GIMaP's proposal network, the importance-weighted posterior of the
compositions asked for, each mode refined by least squares (``domain.posterior.refine.predict``).

Payload: q (nm⁻¹), intensity, sigma (or None), qz (nm⁻¹; 0: the effective interface), compositions (indices of
``encoding.COMPOSITIONS``) or allowed (the head chooses among them), method, samples, seed, law, coupling, grazing
(the person's angles, λ and reflection to keep). The result is JSON: every solution's model (``model_to_dict``),
χ²/N, probability, values with their posterior intervals, flags; and the network's name and SHA-256.
"""

from __future__ import annotations

import math
import time

from ...domain import fit_model as fm
from ...domain.posterior import network as posterior_network
from ...domain.posterior import refine


def _number(value):
    return None if value is None or not math.isfinite(float(value)) else float(value)


def encode(solution) -> dict:
    return {
        "composition": int(solution.composition), "name": solution.name, "probability": _number(solution.probability),
        "model": fm.model_to_dict(solution.model), "chi2": _number(solution.chi2_reduced),
        "log_rmse": _number(solution.log_rmse), "method": solution.method, "flags": list(solution.flags),
        "log_evidence": _number(solution.log_evidence), "ess": _number(solution.ess), "samples": solution.samples,
        "values": {name: [_number(v.best), _number(v.median), _number(v.low68), _number(v.high68), _number(v.low95),
                          _number(v.high95), _number(v.error)] for name, v in solution.values.items()},
    }


def run_predict(payload: dict, report, cancelled) -> dict:
    tick = time.perf_counter()
    report(0, 1, "Loading the proposal network")
    network = posterior_network.load()
    if cancelled():
        raise RuntimeError("Fitting cancelled")
    report(0, 1, "Weighting the network's samples and refining each solution")
    solutions = refine.predict(
        network, payload["q"], payload["intensity"], payload.get("sigma"), qz=float(payload.get("qz") or 0.0),
        compositions=payload.get("compositions"), allowed=payload.get("allowed"),
        seed=int(payload.get("seed", 0)),
        law=payload.get("law", "number_gauss"), coupling=payload.get("coupling", "lma"),
        grazing=payload.get("grazing"))
    if cancelled():
        raise RuntimeError("Fitting cancelled")
    report(1, 1, "Done")
    return {"solutions": [encode(s) for s in solutions],
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


FLAG_SENTENCES = {
    "unconstrained": "the curve says little about {name}",
    "at_bound": "{name} stopped at a bound of its range",
    "correlated": "{name} are strongly correlated",
    "low_ess": "few effective samples: the intervals are rough",
    "bic": "compared by χ² and the number of values (BIC): the posterior's sampling did not converge",
}
"""The flags of a solution as the sentences of a row's ``warnings``."""
ROW_NAMES = {"res_width": "sigma_Res", "res_exponent": "nu_Res"}
"""Fitting's global names as the rows name them (the other values keep their keys)."""


def flag_sentences(flags) -> list[str]:
    out = []
    for flag in flags:
        kind, _, name = flag.partition(":")
        if kind in FLAG_SENTENCES:
            out.append(FLAG_SENTENCES[kind].format(name=name.replace("~", " and ")))
    return out


def at_bounds(model, tolerance: float = 1e-4) -> list[dict]:
    """The free shape and tail values of ``model`` on a bound of their range (within ``tolerance`` of it), as the
    rows list them: ``parameter``, ``bound`` (lower / upper), ``component`` (index; None for the tail)."""
    import math

    out = []
    for path, parameter in model.parameters():
        owner, key = path
        if not parameter.free or key in ("Int", "background", "res_amplitude", "k", "q_resolution"):
            continue
        if owner == "globals" and key not in ROW_NAMES:
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
    ``at_bounds``, sentences in ``warnings``, ``seconds``, ``algorithm``, ``conditions_source`` (``chosen_by``),
    ``unit_contract``."""
    rows = []
    for rank, solution in enumerate(solutions, start=1):
        row = solution_row(solution, item, render_q)
        row.update(
            rank=rank, converged=True, at_bounds=at_bounds(solution.model), warnings=flag_sentences(solution.flags),
            flags=list(solution.flags), seconds=float(solution.seconds), nfev=None, conditions_source=chosen_by,
            algorithm="Predict parameters: the proposal network's importance-weighted samples, each mode refined by "
                      "bounded least squares, the posterior sampled around the best solutions",
            unit_contract=dict(q="nm^-1", R_h_D="nm", sigma_R_h_D="relative standard deviation",
                               sigma_nn="sigma_nn/D, Hosemann; sigma_D = sigma_nn / sqrt(2 pi)", sigma_Res="nm^-1",
                               nu_Res="dimensionless", amplitudes="input intensity units"))
        rows.append(row)
    return rows


def solution_row(solution, item, render_q) -> dict:
    """A solution as a row of the 1D Predict window (``native_v5`` layout, Fitting's conventions: cut_model)."""
    import numpy as np

    from ...domain.fit_model import evaluate

    model = solution.model
    q, y, sigma = (np.asarray(item[k], float) for k in ("q", "observed", "sigma"))
    fitted = evaluate(model, q)
    residual = (fitted - y) / sigma
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
    return dict(
        workflow="native_v5", physics_backend="cut_model", law=model.components[0].law if components else None,
        best_source="posterior", side=item["side"], combination=" + ".join(c["type"] for c in components),
        composition=solution.name, components=components,
        global_params=dict(background=float(g["background"].value), resolution_amplitude=float(g["res_amplitude"].value),
                           sigma_Res=float(g["res_width"].value), nu_Res=float(g["res_exponent"].value),
                           q_resolution=float(g["q_resolution"].value),
                           **{key: float(g[key].value) for key in ("alpha_i", "alpha_f", "wavelength")}),
        probability=_number(solution.probability), posterior={k: v for k, v in encode(solution)["values"].items()},
        best_log_rmse=float(np.sqrt(np.mean(np.log(fitted[positive] / y[positive]) ** 2))) if positive.any() else None,
        best_chi2_weighted=float(np.mean(residual**2)), signed_weighted_rms=float(np.sqrt(np.mean(residual**2))),
        signed_huber_delta2=float(np.mean(np.where(abs(residual) <= 2, 0.5 * residual**2, 2 * (abs(residual) - 1)))),
        probability_status="Posterior probability of the composition and mode (importance sampling)",
        warnings=list(solution.flags), ess=_number(solution.ess),
        native_q=(q * item["sign"]).tolist(), observed=y.tolist(), sigma=sigma.tolist(), native_fit=fitted.tolist(),
        display_q=(render_q * item["sign"]).tolist(), display_fit=evaluate(model, render_q).tolist())


def fit_and_write(output, item, options, report, cancelled) -> list:
    """Predict parameters for one half of a curve of the 1D Predict window; its files as the other methods write."""
    import json

    import numpy as np

    network = posterior_network.load()
    report(0, 1, f"{item['side']}: weighting the network's samples", {})
    solutions = refine.predict(network, item["q"], item["observed"], item["sigma"],
                               qz=float(options.get("cut_qz") or 0.0), **batch_compositions(options["components"]))
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


__all__ = ["batch_compositions", "encode", "fit_and_write", "run_predict", "solution_row"]
