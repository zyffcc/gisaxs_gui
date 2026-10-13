"""The fit method ``fit1d`` of 1D Predict: the external gisaxs1d kit (network proposals + physical fit).

The kit is research code outside the program (``application.workflow_v5.fit1d_kit_home``): it is put on
``sys.path`` and imported here only, inside the job process that runs ``run_workflow_job``. It needs numpy and
scipy, no TensorFlow.

The composition is the user's (1 sphere, 3 vertical cylinder; this GUI route does not offer random cylinders): one curve does
not identify it, and the fit does not test it against others. A row is the kit's fit in the ``native_v5``
format of the other methods, with these differences, repeated in every row (``unit_contract``,
``validation_scope``):

* the physics is the kit's "v2": a number-weighted size distribution (σR/R of the number distribution), a
  Hosemann paracrystal with w = w_rel·D, spheres at |q| = sqrt(q_par² + qz²), vertical cylinders at q_par.
  ``native_fit`` and ``display_fit`` are the kit's own curve at the measured q; Fitting's manual model
  (intensity-weighted sizes, damping exp(−π q² (σD·D)²), no qz) draws another curve from the same numbers;
* ``params["sigma_D"]`` is the kit's ``legacy_sigma_D`` = w_rel / sqrt(2π): the σD/D that gives the same
  paracrystal damping in Fitting's model. The kit's own values, their 1σ errors, the ranges over
  chi²-equivalent solutions and the status of every parameter are in the component's ``fit1d``;
* what the kit flags (another solution fits equally well, a parameter at a bound or not pinned down, the
  model not describing the curve) is in ``warnings``. The kit gives the ranges of equivalent solutions, not
  the solutions themselves, so there is one row per side.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
import time

import numpy as np

from ...application.workflow_v5 import FIT1D_SHAPES, fit1d_kit_home

TYPES = {"sphere": (1, "sphere"), "vertical_cylinder": (3, "vertical_cylinder")}
"""The kit's shape → (component id, the name of the type in a row)."""
SCOPE = (
    "fit1d (network proposals + physical fit, physics v2) for the composition given by the user; the "
    "composition is not tested against others. The curve is fit1d's own: Fitting's manual model draws a "
    "different curve from the same numbers. Parameter statuses and equivalent solutions: see warnings."
)
KIT_MISSING = (
    "The fit1d kit was not found in {home}. Copy the folder gisaxs1d_gui_kit (it contains gisaxs1d_kit/, "
    "app/, models/ and tables_packed/) to that place, or set the environment variable GISAXS1D_KIT_HOME to "
    "the folder, then start GIMaP again."
)
KIT_BROKEN = "The fit1d kit in {home} could not be loaded ({reason}). Run its selftest, or copy the folder again."
QZ_ZERO = (
    "No qz of the cut was given: spheres are evaluated at qz = 0 (|q| = q_par). For a Yoneda cut set "
    "“Cut qz” in Parameters…; the sphere radius depends on it."
)
SIGMA_ESTIMATED = (
    "The curve has no error bars: sigma was estimated by 1D Predict (Relative σ), so chi² and the errors "
    "of the parameters are relative to that estimate."
)
STATUS_TEXT = {
    "ambiguous": "another solution fits equally well",
    "weak": "not pinned down by this curve",
    "at_bound": "at a bound of its range",
    "inactive": "population without contribution",
}


def load_kit():
    """``gisaxs1d_kit.core`` with its code and tables ready; a message that says what to do when it is not there."""
    home = fit1d_kit_home()
    if not (home / "gisaxs1d_kit" / "core.py").is_file():
        raise FileNotFoundError(KIT_MISSING.format(home=home))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(name, "1")  # for processes the kit starts: many BLAS threads each made them far slower
    if str(home) not in sys.path:
        sys.path.insert(0, str(home))
    try:
        # By name: the kit is an optional folder beside the program, not a package of requirements.txt.
        core = importlib.import_module("gisaxs1d_kit.core")
        core.ensure_ready()
    except (ImportError, OSError) as exc:
        raise RuntimeError(KIT_BROKEN.format(home=home, reason=f"{type(exc).__name__}: {exc}")) from exc
    return core


def kit_arguments(item, options) -> dict:
    """The arguments of ``gisaxs1d_kit.core.fit_curve`` for one half of a curve (q in nm⁻¹, sorted by |q|)."""
    unknown = [t for t in options["components"] if t not in FIT1D_SHAPES]
    if unknown or not options["components"]:
        raise ValueError("fit1d needs a composition of Sphere (1) and Vertical cylinder (3) only")
    gisaxs = options["fit1d_mode"] == "gisaxs"
    arguments = dict(
        q=np.asarray(item["q"], float),
        intensity=np.asarray(item["observed"], float),
        sigma=np.asarray(item["sigma"], float),
        q_unit="nm^-1",
        fold="abs",
        composition=",".join(FIT1D_SHAPES[t] for t in options["components"]),
        mode=options["fit1d_mode"],
        qz=float(options["fit1d_qz"] or 0.0) if gisaxs else 0.0,
        policy=options["fit1d_policy"],
        model=options["fit1d_model"],
        sigma_floor=float(options["fit1d_sigma_floor"]),
    )
    if gisaxs and options["nu_res"] is not None:
        arguments["lowq_nu"] = float(options["nu_res"])
    return arguments


def _clean(value):
    """JSON without NaN (rows are written with ``allow_nan=False`` and cross the process boundary)."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _number(value, default=0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _components(result, structure_factor: bool) -> list[dict]:
    components = []
    for population in result["populations"]:
        if population["shape"] not in TYPES:
            raise ValueError(f"fit1d returned a shape 1D Predict has no row for: {population['shape']}")
        type_id, name = TYPES[population["shape"]]
        distance, spread = _number(population.get("D_nm")), _number(population.get("w_rel"))
        params = dict(
            R=_number(population.get("R_nm")),
            sigma_R=_number(population.get("sigma_R_rel")),
            D=distance if structure_factor else 0.0,
            sigma_D=spread / math.sqrt(2 * math.pi) if structure_factor and distance > 0 else 0.0,
        )
        own = {
            key: population.get(key)
            for key in ("R_nm", "R_nm_err", "R_nm_range", "sigma_R_rel", "sigma_R_rel_err", "D_nm", "D_nm_err",
                        "D_nm_range", "w_rel", "w_rel_err", "intensity_fraction", "status")
        }
        components.append(
            dict(type=name, type_id=type_id, params=params, amplitude=_number(population.get("amplitude")),
                 structure_factor=structure_factor, name=population.get("name"),
                 inactive=bool(population.get("inactive")), fit1d=own)
        )
    total = sum(c["amplitude"] for c in components)
    for component in components:
        component["weight"] = component["amplitude"] / total if total > 0 else 0.0
    return components


def _warnings(result, item, options, used: int) -> list[str]:
    summary = result["summary"]
    warnings = [str(message) for message in summary.get("messages") or ()]
    for row in result.get("table") or ():
        text = STATUS_TEXT.get(row.get("status"))
        if text is None or row.get("parameter") == "intensity share" or row.get("group") == "global":
            continue
        line = f"{row['key']} = {row.get('text') or row.get('value')}{' ' + row['unit'] if row.get('unit') else ''}: {text}"
        bounds = row.get("range")
        if row["status"] == "ambiguous" and bounds:
            line += f" (equivalent solutions: {bounds[0]:.4g}–{bounds[1]:.4g} {row.get('unit') or ''})".replace(" )", ")")
        warnings.append(line)
    if options["fit1d_mode"] == "gisaxs" and options["fit1d_qz"] is None and 1 in options["components"]:
        warnings.append(QZ_ZERO)
    if item.get("sigma_estimated"):
        warnings.append(SIGMA_ESTIMATED)
    if used != len(item["q"]):
        warnings.append(f"fit1d used {used} of the {len(item['q'])} points of this half.")
    return warnings


def solution_row(result, item, options, seconds: float) -> dict:
    """The kit's result (``gisaxs1d_gui_result_v1``) as one ``native_v5`` row."""
    curves, summary = result["curves"], result["summary"]
    q = np.asarray(curves["q_nm_inv"], float)
    model = np.asarray(curves["model"], float)
    fitted_sigma = np.asarray(curves["sigma"], float)  # with the systematic floor: what chi² refers to
    y = np.asarray(curves["intensity"], float)
    same = q.size == len(item["q"]) and np.allclose(q, item["q"], rtol=1e-9, atol=0.0)
    sigma = fitted_sigma
    if not (q.size and model.size == q.size and np.isfinite(model).all()):
        raise RuntimeError("fit1d returned no finite fitted curve")
    residual = (model - y) / fitted_sigma
    positive = (y > 0) & (model > 0)
    logrmse = float(np.sqrt(np.mean(np.log(model[positive] / y[positive]) ** 2))) if positive.any() else None
    gisaxs = options["fit1d_mode"] == "gisaxs"
    components = _components(result, structure_factor=gisaxs)
    low = result.get("low_q") or {}
    resolution = gisaxs and _number(low.get("sigma_res_nm_inv")) > 0 and _number(low.get("nu")) > 0
    globals_ = dict(
        background=_number(result.get("background")),
        resolution_amplitude=_number(low.get("C")) if resolution else 0.0,
        sigma_Res=_number(low.get("sigma_res_nm_inv")) if resolution else 0.0,
        nu_Res=_number(low.get("nu")) if resolution else 0.0,
    )
    qz = summary.get("qz_nm_inv")
    return dict(
        rank=1,
        workflow="native_v5",
        physics_backend="fit1d_physics_v2",
        side=item["side"],
        combination=" + ".join(c["type"] for c in components),
        components=components,
        global_params=globals_,
        best_source="fit1d",
        best_log_rmse=logrmse,
        best_chi2_weighted=float(np.mean(residual**2)),
        signed_huber_delta2=float(np.mean(np.where(abs(residual) <= 2, 0.5 * residual**2, 2 * (abs(residual) - 1)))),
        signed_weighted_rms=float(np.sqrt(np.mean(residual**2))),
        probability_status="Not calibrated",
        conditions_source="user_composition",
        validation_scope=SCOPE,
        algorithm="gisaxs1d kit: network proposals, (R, D) grid alignment, Levenberg-Marquardt with linear "
                  "nonnegative amplitudes, check for chi²-equivalent solutions",
        weight_definition="Amplitude of the population divided by the sum of the amplitudes; not a "
                          "probability or a volume fraction",
        unit_contract=dict(
            q="nm^-1",
            R_D="nm",
            sigma_R="relative width of the number-weighted size distribution (fit1d physics v2)",
            sigma_D="legacy_sigma_D = w_rel / sqrt(2 pi): the relative spread that gives fit1d's Hosemann "
                    "damping in Fitting's model exp(-pi q^2 (sigma_D D)^2); fit1d's own w_rel = w / D is in "
                    "components[i].fit1d",
            sphere_q="|q| = sqrt(q_par^2 + qz^2), qz in fit1d.qz_nm_inv; vertical cylinders at q_par",
            sigma_Res="nm^-1",
            nu_Res="dimensionless",
            amplitudes="input intensity units times fit1d's form-factor normalisation; not Fitting's Int",
            curves="native_fit and display_fit: fit1d's model at the measured q",
            chi2="residuals over fit1d's effective sigma (sigma with fit1d_sigma_floor)",
        ),
        warnings=_warnings(result, item, options, int(q.size)),
        converged=True,
        seconds=seconds,
        fit1d=dict(
            status=summary.get("status"), flags=summary.get("flags"), signals=summary.get("signals"),
            stop_stage=summary.get("stop_stage"), policy=summary.get("policy"), mode=summary.get("mode"),
            model=summary.get("model"), composition=summary.get("composition"), qz_nm_inv=qz,
            qz_given=options["fit1d_qz"] is not None, sigma_floor=summary.get("sigma_floor"),
            chi2_per_point=summary.get("chi2_per_point"), n_points=summary.get("n_points"),
            notes=summary.get("notes"), table=result.get("table"), kit_version=result.get("kit_version"),
            fit1d_version=result.get("fit1d_version"), conventions=result.get("conventions"),
        ),
        native_q=(q * item["sign"]).tolist(),
        observed=(np.asarray(item["observed"], float) if same else y).tolist(),
        sigma=sigma.tolist(),
        measurement_sigma=(np.asarray(item["sigma"], float) if same else fitted_sigma).tolist(),
        native_fit=model.tolist(),
        display_q=(q * item["sign"]).tolist(),
        display_fit=model.tolist(),
        display_components={name: np.asarray(values, float).tolist()
                            for name, values in curves.get("components", {}).items()},
    )


def fit_and_write(output, item, options, report, cancelled, core=None):
    """Fit one half with the kit and write ``solutions.json``, ``display.npz`` and ``fitting_curves.csv``."""
    if cancelled():
        raise RuntimeError("Fitting cancelled")
    core = core or load_kit()
    arguments = kit_arguments(item, options)
    report(0, 1, f"{item['side']}: fit1d {arguments['composition']} ({arguments['policy']}, "
                 f"network {arguments['model']})", {})
    tick = time.perf_counter()
    result = core.fit_curve(arguments)
    row = _clean(solution_row(result, item, options, time.perf_counter() - tick))
    rows = [row]
    output.mkdir(parents=True, exist_ok=True)
    (output / "solutions.json").write_text(
        json.dumps(dict(method="fit1d", options=options, solutions=rows), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    parts = {f"component_{name}": np.asarray(values, float)
             for name, values in (result["curves"].get("components") or {}).items()}
    np.savez_compressed(
        output / "display.npz",
        reference_q=np.abs(row["native_q"]),
        observed=row["observed"],
        sigma=row["sigma"],
        render_q=np.abs(row["display_q"]),
        rendered_curves=[row["display_fit"]],
        **parts,
    )
    np.savetxt(
        output / "fitting_curves.csv",
        np.column_stack([row["display_q"], row["display_fit"]]),
        delimiter=",",
        header="q_nm^-1,candidate_1",
        comments="",
    )
    return rows


__all__ = ["fit_and_write", "kit_arguments", "load_kit", "solution_row"]
