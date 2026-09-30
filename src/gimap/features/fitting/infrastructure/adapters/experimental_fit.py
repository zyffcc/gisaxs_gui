"""Fast experimental physical fits with explicit, independently fitted amplitudes.

This is numerical variable projection, not a neural-network prediction. Auto
compares single-component families; a user-specified composition is fitted whole.
Candidate scores are not posterior probabilities or proof of unique parameters.
"""

import json
import sys
import time

import numpy as np
from scipy.optimize import least_squares, nnls

from ...application.workflow_v5 import bundled_workflow

NAMES = {1: "sphere", 2: "random_cylinder", 3: "vertical_cylinder"}


def component_function():
    root = str(bundled_workflow())
    if root not in sys.path:
        sys.path.insert(0, root)
    from numpy_forward64 import component

    return component


def forward(q, components, globals_):
    """Reconstruct in input intensity units; no data-dependent normalization."""
    component = component_function()
    q = np.abs(np.asarray(q, float))
    result = np.full_like(q, globals_["background"])
    for c in components:
        p = c["params"]
        result += c["amplitude"] * component(
            q,
            c["type_id"],
            p["R"],
            p["sigma_R"],
            p.get("h") or 10,
            p.get("sigma_h") or 0.2,
            p["D"],
            p["sigma_D"],
            True,
        )
    result += globals_["resolution_amplitude"] / (
        1 + (q / globals_["sigma_Res"]) ** globals_["nu_Res"]
    )
    return result


def fit_candidates(item, options, report, cancelled):
    component = component_function()
    q, y, sigma = (item[k] for k in ("q", "observed", "sigma"))
    combinations = [options["components"]] if options["components"] else [[1], [3], [2]]
    results = []
    hint = options.get("distance_hint_nm")
    starts = [(1.5, None), (4.0, None)] + ([(4.0, float(hint))] if hint and np.isfinite(hint) and hint > 0 else [])
    for types in combinations:
        for radius, distance in starts:
            if cancelled():
                raise RuntimeError("Fitting cancelled")
            start = f"initial R={radius:g} nm" + (f", D={distance:.3g} nm" if distance else "")
            report(0, 1, f"{item['side']}: numerical fit {types}, {start}", {})
            tick = time.perf_counter()
            initial, lower, upper = [], [], []
            for j, typ in enumerate(types):
                r = radius * (1 + 0.6 * j)
                dmin = max(3, 2 * r * 1.001)
                eta = np.log(1.5) / np.log(500 / dmin)
                if distance:
                    eta = float(np.clip(np.log(max(distance, 1.01 * dmin) / dmin) / np.log(500 / dmin), 0.01, 0.99))
                initial.extend([np.log(r), 0.2, eta, 0.12])
                lower.extend([0, 0.02, 0, 0.05])
                upper.extend([np.log(100), 0.9, 1, 0.9])
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
                        sigma_D=float(sd),
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

            def evaluate(z):
                if cancelled():
                    raise RuntimeError("Fitting cancelled")
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
                            p["sigma_D"],
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
                max_nfev=150,
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
            cs.sort(key=lambda c: (c["type_id"], c["params"]["R"], c["params"].get("h", 0)))
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
            if np.any(np.abs(optimum.x - lower) < 1e-4 * (np.array(upper) - lower)) or np.any(
                np.abs(optimum.x - upper) < 1e-4 * (np.array(upper) - lower)
            ):
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
                    physics_backend="experimental_calibrated",
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
                        sigma_Res="nm^-1",
                        nu_Res="dimensionless",
                        amplitudes="input intensity units",
                    ),
                    warnings=warnings,
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
