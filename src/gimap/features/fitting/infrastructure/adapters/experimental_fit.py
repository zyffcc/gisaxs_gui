"""Fast experimental physical fits with explicit, independently fitted amplitudes.

This is numerical variable projection, not a neural-network prediction. Auto
compares single-component families; a user-specified composition is fitted whole.
Candidate scores are not posterior probabilities or proof of unique parameters.

The model is GIMaP's one forward model of a cut, ``cut_model``, in Fitting's conventions: Gaussian sizes by number,
the paracrystal's σnn/D (Hosemann), every family normalised to T(0) = 1, so an amplitude is Fitting's Int. A solution
says so (``physics_backend`` "cut_model", ``law``) and also gives σD/D = (σnn/D)/√(2π) under ``sigma_D`` for the
reports that show it.
"""

import json
import math
import time

import numpy as np
from scipy.optimize import least_squares, nnls

from ...domain.cut_model.model import population_terms

NAMES = {1: "sphere", 2: "random_cylinder", 3: "vertical_cylinder"}
MAX_NFEV = 150
"""Function evaluations one start may use (``least_squares``). With those of its numerical Jacobian, a start
evaluates the model at most ``MAX_NFEV * (free values + 1)`` times: the ``budget`` given to ``evaluated``."""


SHAPE_VALUES = ("R", "sigma_R", "D", "sigma_nn")
"""The fitted values of every component, in the order of the fit vector (D through η: η = 0 is D_min, 1 is 500 nm)."""
CYLINDER_VALUES = ("h", "sigma_h")
"""… followed by these for a random cylinder."""
BOUND_TOLERANCE = 1e-4
"""A fitted value closer to a bound than this fraction of its range ended on the bound."""


def ended_on_bounds(z, lower, upper, types, options):
    """The fitted values that ended on a bound of their search range, as ``(j, name, "lower" | "upper")``: ``j`` the
    component in the order of ``types`` (``None`` for the beam-tail values ``sigma_Res``, ``nu_Res``), ``name`` as
    the solution names it (``R``, ``sigma_R``, ``D``, ``sigma_nn``, ``h``, ``sigma_h``). Such a value is not
    determined by the curve: the condition of the "reached a search bound" warning."""
    names = []
    for j, typ in enumerate(types):
        names += [(j, name) for name in SHAPE_VALUES + (CYLINDER_VALUES if typ == 2 else ())]
    names += [(None, name) for key, name in (("sigma_res", "sigma_Res"), ("nu_res", "nu_Res")) if options[key] is None]
    z, lower, upper = (np.asarray(values, dtype=float) for values in (z, lower, upper))
    tolerance = BOUND_TOLERANCE * (upper - lower)
    ended = []
    for (j, name), value, low, high, tol in zip(names, z, lower, upper, tolerance):
        if abs(value - low) < tol:
            ended.append((j, name, "lower"))
        elif abs(value - high) < tol:
            ended.append((j, name, "upper"))
    return ended


LAW = "number_gauss"
FAMILY_OF = {1: "sphere", 2: "cylinder", 3: "vertical_cylinder"}
SQRT_2PI = math.sqrt(2.0 * math.pi)


def component(q, type_id, radius, spread, height, height_spread, distance, sigma_nn, structured=True):
    """T(q) of one component (normalised to 1 at q = 0) from cut_model: sizes ``LAW``, relative spreads, the
    paracrystal of mean distance D and σnn/D."""
    family = FAMILY_OF[int(type_id)]
    cylinder = family == "cylinder"
    return population_terms(family, np.abs(np.asarray(q, float)), radius=radius, radius_spread=spread, law=LAW,
                            height=height if cylinder else 10.0, height_spread=height_spread if cylinder else 0.0,
                            distance=distance if structured else 0.0,
                            distance_spread=sigma_nn if structured else 0.0)[0]


def component_function():
    return component


def forward(q, components, globals_):
    """Reconstruct in input intensity units; no data-dependent normalization. A component with ``sigma_nn`` is one of
    this fit (sizes by number); one with ``sigma_D`` only is of 1D Predict's earlier solutions (sizes by intensity,
    σnn/D = √(2π) σD/D), drawn in those conventions."""
    q = np.abs(np.asarray(q, float))
    result = np.full_like(q, globals_["background"])
    for c in components:
        p = c["params"]
        legacy = "sigma_nn" not in p
        family = FAMILY_OF[int(c["type_id"])]
        cylinder = family == "cylinder"
        result += c["amplitude"] * population_terms(
            family, q, radius=p["R"], radius_spread=p["sigma_R"], law="intensity_gauss" if legacy else LAW,
            height=(p.get("h") or 10) if cylinder else 10.0, height_spread=(p.get("sigma_h") or 0.2) if cylinder else 0.0,
            distance=p["D"], distance_spread=SQRT_2PI * p["sigma_D"] if legacy else p["sigma_nn"])[0]
    result += globals_["resolution_amplitude"] / (
        1 + (q / globals_["sigma_Res"]) ** globals_["nu_Res"]
    )
    return result


def fit_candidates(item, options, report, cancelled, evaluated=None):
    """``report(0, 1, line, details)`` before every start (``details``: which start of how many, its families).
    ``evaluated(details)``, when given, at every evaluation of the model: the same details plus how many
    ``evaluations`` this start has made of its ``budget`` — how far a start is, for a page that shows one curve.
    A job of many files does not pass it: its ``report`` writes a log line per call."""
    component = component_function()
    q, y, sigma = (item[k] for k in ("q", "observed", "sigma"))
    combinations = [options["components"]] if options["components"] else [[1], [3], [2]]
    results = []
    hint = options.get("distance_hint_nm")
    starts = [(1.5, None), (4.0, None)] + ([(4.0, float(hint))] if hint and np.isfinite(hint) and hint > 0 else [])
    number, start_count = 0, len(combinations) * len(starts)
    for types in combinations:
        for radius, distance in starts:
            if cancelled():
                raise RuntimeError("Fitting cancelled")
            number += 1
            start = f"initial R={radius:g} nm" + (f", D={distance:.3g} nm" if distance else "")
            # (done, total) stay (0, 1): in the 1D Predict job of many files the same callback counts the
            # files, so which start this is goes in the free details (the single-curve page reads them).
            details = {"start": number, "starts": start_count, "types": [int(typ) for typ in types]}
            report(0, 1, f"{item['side']}: numerical fit {types}, {start}", dict(details))
            used = [0]
            tick = time.perf_counter()
            initial, lower, upper = [], [], []
            for j, typ in enumerate(types):
                r = radius * (1 + 0.6 * j)
                dmin = max(3, 2 * r * 1.001)
                eta = np.log(1.5) / np.log(500 / dmin)
                if distance:
                    eta = float(np.clip(np.log(max(distance, 1.01 * dmin) / dmin) / np.log(500 / dmin), 0.01, 0.99))
                initial.extend([np.log(r), 0.2, eta, 0.3])
                lower.extend([0, 0.02, 0, 0.05])
                upper.extend([np.log(100), 0.9, 1, 1.0])
                if typ == 2:
                    initial.extend([np.log(10), 0.2])
                    lower.extend([np.log(2), 0.02])
                    upper.extend([np.log(500), 0.9])
            for key, value, lo, hi in (("sigma_res", 0.02, 0.001, 0.1), ("nu_res", 2.5, 1, 20)):
                if options[key] is None:
                    initial.append(value)
                    lower.append(lo)
                    upper.append(hi)

            def decode(z):
                components, offset = [], 0
                for typ in types:
                    lr, sr, eta, sd = z[offset : offset + 4]
                    offset += 4
                    r = float(np.exp(lr))
                    dmin = max(3, 2 * r * 1.001)
                    p = dict(
                        R=r,
                        sigma_R=float(sr),
                        D=float(dmin * (500 / dmin) ** eta),
                        sigma_nn=float(sd),
                        sigma_D=float(sd) / SQRT_2PI,
                    )
                    if typ == 2:
                        p.update(h=float(np.exp(z[offset])), sigma_h=float(z[offset + 1]))
                        offset += 2
                    components.append(dict(type=NAMES[typ], type_id=typ, params=p))
                resolution = {}
                for key in ("sigma_res", "nu_res"):
                    resolution[key] = options[key] if options[key] is not None else float(z[offset])
                    if options[key] is None:
                        offset += 1
                return components, resolution

            budget = MAX_NFEV * (len(initial) + 1)

            def evaluate(z):
                if cancelled():
                    raise RuntimeError("Fitting cancelled")
                if evaluated is not None:
                    used[0] += 1
                    evaluated({**details, "evaluations": used[0], "budget": budget})
                cs, res = decode(z)
                columns = []
                for c in cs:
                    p = c["params"]
                    columns.append(
                        component(
                            q,
                            c["type_id"],
                            p["R"],
                            p["sigma_R"],
                            p.get("h", 10),
                            p.get("sigma_h", 0.2),
                            p["D"],
                            p["sigma_nn"],
                            True,
                        )
                    )
                columns.extend([np.ones_like(q), 1 / (1 + (q / res["sigma_res"]) ** res["nu_res"])])
                basis = np.stack(columns, axis=1)
                design = basis / sigma[:, None]
                scales = np.maximum(np.linalg.norm(design, axis=0), 1e-30)
                coefficients = nnls(design / scales, y / sigma)[0] / scales
                return basis @ coefficients, coefficients, cs, res

            optimum = least_squares(
                lambda z: (evaluate(z)[0] - y) / sigma,
                initial,
                bounds=(lower, upper),
                loss="soft_l1",
                max_nfev=MAX_NFEV,
                diff_step=1e-4,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
            )
            prediction, coefficients, cs, res = evaluate(optimum.x)
            weights = coefficients[: len(cs)]
            total = weights.sum()
            for c, amplitude in zip(cs, weights):
                c.update(
                    amplitude=float(amplitude),
                    weight=float(amplitude / total) if total > 0 else 0.0,
                    structure_factor=True,
                )
            ended = ended_on_bounds(optimum.x, lower, upper, types, options)
            owners = list(cs)  # the components in the order of ``types``, before they are sorted
            cs.sort(key=lambda c: (c["type_id"], c["params"]["R"], c["params"].get("h", 0)))
            at_bounds = [
                dict(parameter=name, bound=side,
                     component=None if j is None else next(k for k, c in enumerate(cs) if c is owners[j]))
                for j, name, side in ended
            ]
            globals_ = dict(
                background=float(coefficients[-2]),
                resolution_amplitude=float(coefficients[-1]),
                sigma_Res=res["sigma_res"],
                nu_Res=res["nu_res"],
            )
            residual = (prediction - y) / sigma
            positive = y > 0
            logrmse = float(
                np.sqrt(np.mean(np.log(np.maximum(prediction[positive], 1e-30) / y[positive]) ** 2))
            )
            score = float(
                np.mean(np.where(abs(residual) <= 2, 0.5 * residual**2, 2 * (abs(residual) - 1)))
            )
            render_q = np.linspace(q[0], q[-1], options["render_points"])
            warnings = []
            if ended:
                warnings.append(
                    "One or more shape/resolution parameters reached a search bound; values may be unidentifiable."
                )
            if not optimum.success:
                warnings.append("Evaluation budget reached; convergence is not established.")
            if total <= 0 or np.any(weights <= max(total, 1e-30) * 1e-6):
                warnings.append(
                    "A requested component has negligible fitted amplitude; its parameters are unconstrained."
                )
            results.append(
                dict(
                    workflow="native_v5",
                    physics_backend="cut_model",
                    law=LAW,
                    side=item["side"],
                    combination=" + ".join(c["type"] for c in cs),
                    components=cs,
                    global_params=globals_,
                    best_source="experimental_physical",
                    best_log_rmse=logrmse,
                    best_chi2_weighted=float(np.mean(residual**2)),
                    signed_huber_delta2=score,
                    signed_weighted_rms=float(np.sqrt(np.mean(residual**2))),
                    probability_status="Not calibrated",
                    conditions_source="user_composition"
                    if options["components"]
                    else "single_component_screen",
                    validation_scope="Numerical experimental fit. Auto tests single-component families only; no exhaustive parameter or composition coverage.",
                    algorithm="Bounded soft-L1 least_squares with weighted nonnegative linear-amplitude least squares at each shape evaluation",
                    weight_definition="Normalized particle amplitude; not posterior probability or volume fraction",
                    unit_contract=dict(
                        q="nm^-1",
                        R_h_D="nm",
                        sigma_R_h_D="relative standard deviation",
                        sigma_nn="sigma_nn/D, Hosemann; sigma_D = sigma_nn / sqrt(2 pi)",
                        sigma_Res="nm^-1",
                        nu_Res="dimensionless",
                        amplitudes="input intensity units",
                    ),
                    warnings=warnings,
                    at_bounds=at_bounds,
                    nfev=int(optimum.nfev),
                    converged=bool(optimum.success),
                    seconds=time.perf_counter() - tick,
                    native_q=(q * item["sign"]).tolist(),
                    observed=y.tolist(),
                    sigma=sigma.tolist(),
                    native_fit=prediction.tolist(),
                    display_q=(render_q * item["sign"]).tolist(),
                    display_fit=forward(render_q, cs, globals_).tolist(),
                )
            )
    results.sort(key=lambda r: r["signed_huber_delta2"])
    selected = []
    for row in results:
        # Preserve distinct parameter modes even when their forward curves overlap.
        vector = np.array(
            [v for c in row["components"] for v in [*c["params"].values(), c["amplitude"]]]
            + list(row["global_params"].values())
        )
        duplicate = False
        for other in selected:
            if row["combination"] != other["combination"]:
                continue
            previous = np.array(
                [v for c in other["components"] for v in [*c["params"].values(), c["amplitude"]]]
                + list(other["global_params"].values())
            )
            if (
                np.max(
                    abs(vector - previous)
                    / np.maximum(np.maximum(abs(vector), abs(previous)), 1e-12)
                )
                < 0.01
            ):
                duplicate = True
                break
        if not duplicate:
            row["rank"] = len(selected) + 1
            selected.append(row)
        if len(selected) >= options["max_solutions"]:
            break
    return selected


def fit_and_write(output, item, options, report, cancelled):
    rows = fit_candidates(item, options, report, cancelled)
    output.mkdir(parents=True, exist_ok=True)
    (output / "solutions.json").write_text(
        json.dumps(
            dict(method="experimental_physical", options=options, solutions=rows),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        output / "display.npz",
        reference_q=item["q"],
        observed=item["observed"],
        sigma=item["sigma"],
        render_q=np.abs(rows[0]["display_q"]),
        rendered_curves=[r["display_fit"] for r in rows],
    )
    np.savetxt(
        output / "fitting_curves.csv",
        np.column_stack([rows[0]["display_q"], *[r["display_fit"] for r in rows]]),
        delimiter=",",
        header="q_nm^-1," + ",".join(f"candidate_{i + 1}" for i in range(len(rows))),
        comments="",
    )
    return rows
