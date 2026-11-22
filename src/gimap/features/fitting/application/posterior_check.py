"""The posterior check of Results ▸ Uncertainty and Report (Physics and Methods 9.19): a short MCMC run on the posterior
of the solution in Model (Predict parameters) or of the last Refine or Search, beside the intervals it checks.

A check only: nothing it finds goes into the model, the Results or Predict's values. ``target_for`` chooses what is
checked — a Predict solution whose model is the one in Model (its posterior: ``domain.posterior_check.PredictTarget``,
the cut as Predict was given it: the points, the cut's q_z and, with both angles, k_i − k_f), else the current
Refine or Search (``FitTarget``). ``rows`` sets the run beside what it checks: Predict's median and 68 % / 95 %
intervals, or the fit's value ± 1σ (and ± 1.96σ).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.gimap.shared.fit_names import interval_values
from src.gimap.shared.version import SOFTWARE

from ..domain import mcmc
from ..domain.fit_engine import FitData
from ..domain.fit_model import FitModel
from ..domain.posterior.refine import MODEL_ERROR_FLOOR, model_values
from ..domain.posterior_check import ENSEMBLES, CheckResult, FitTarget, PredictTarget, run_check
from .single_predict import compositions, cut_geometry, cut_qz_minus, upright_distributions

NOTHING = ("Nothing to check: put a solution of Predict parameters in Model, or fit with Refine or Search first.")
CHANGED = "The model changed since the run: the check is of the model it was started for."
CONVERGED = ("Converged: {kept} steps kept (the first half discarded) ≥ {factor} × the longest autocorrelation time "
             "({tau} steps), R̂ ≤ {rhat}; about {samples} independent samples.")
NOT_CONVERGED = ("Not converged: {kept} steps kept, the longest autocorrelation time {tau} steps (needs ≥ {factor} "
                 "× of it), R̂ up to {rhat} (needs < {limit}). Run longer before reading the intervals.")
CHECK_ONLY = "A check only: the values and intervals in Results are not changed."
PREDICT_CHECK = ("a second sampler of the same posterior (same likelihood, model error and prior); "
                 "checks the Monte Carlo accuracy of this mode")
BIC_REFERENCE = "The reference intervals are least-squares intervals from the BIC fallback."


@dataclass(frozen=True)
class Checked:
    """What a run checks: the target and the intervals beside it (name → (median or value, low68, high68, low95,
    high95)), and what they are (``predict`` or ``fit``)."""
    target: object
    kind: str
    reference: dict
    reference_kind: str = "least_squares"


def target_for(model: FitModel, data: Optional[FitData], solutions=(), result=None) -> Checked:
    """The posterior to check for the model in Model (module docstring); ``ValueError(NOTHING)`` when there is none."""
    if data is None or data.q.size < 3:
        raise ValueError(NOTHING)
    solution = next((s for s in solutions if s.model == model and getattr(s, "intervals", ())), None)
    if solution is not None:
        composition = compositions(model, "model")["compositions"][0]
        centre = model_values(model, composition)
        if centre is None:
            raise ValueError(NOTHING)
        qz, grazing = cut_geometry(model)
        diagnostics = getattr(solution, "diagnostics", None) or {}
        substrate = model.substrate if diagnostics.get("substrate_level") == 4 else None
        heights = upright_distributions(model) if substrate else {}
        # Height-specified uprights carry their own law and coupling. Every other population uses
        # Predict's shared defaults, which can differ from those of the first upright.
        default = next((c for c in model.components if not c.heights), model.components[0]) if heights else model.components[0]
        law, coupling = default.law, default.coupling
        target = PredictTarget(data.q, data.intensity, data.sigma, composition=composition, centre=centre,
                               substrate=substrate, grazing=grazing, upright_heights=heights,
                               qz=diagnostics.get("cut_qz", qz),
                               qz_minus=diagnostics.get("cut_qz_minus", cut_qz_minus(model)), law=law, coupling=coupling,
                               model_error_floor=diagnostics.get("model_error_floor", MODEL_ERROR_FLOOR),
                               importance_starts=getattr(solution, "importance_starts", ()))
        summary = getattr(solution, "posterior_summary", {})
        if summary:
            keys = ("median", "low68", "high68", "low95", "high95")
            return Checked(target, "predict", {name: tuple(row[k] for k in keys) for name, row in summary.items()},
                           diagnostics.get("posterior_kind", "posterior"))
        reference = {name: tuple(v[1:6]) for name, v in dict(solution.intervals).items()}
        kind = "least_squares" if "bic" in getattr(solution, "flags", ()) else "reported_intervals"
        return Checked(target, "predict", reference, kind)
    if result is None:
        raise ValueError(NOTHING)
    target = FitTarget(result.model, data)
    reference = {}
    for path, name in zip(target.paths, target.names):
        value, error = result.model.get(path).value, result.errors.get(path, math.nan)
        spread = error if error is not None and math.isfinite(error) else math.nan
        reference[name] = (value, value - spread, value + spread, value - 1.96 * spread, value + 1.96 * spread)
    return Checked(target, "fit", reference)


def run(checked: Checked, *, steps: int = 3000, walkers: Optional[int] = None, seed: int = 0, progress=None,
        stop=None) -> CheckResult:
    return run_check(checked.target, steps=steps, walkers=walkers, seed=seed, progress=progress, stop=stop)


def rows(result: CheckResult, reference: dict) -> list[dict]:
    """One row per value reported: the value checked, the reference intervals, the run's median and intervals, τ and
    R̂, and whether the run's 68 % interval overlaps the reference's."""
    out = []
    for index, name in enumerate(result.names):
        if name not in result.reported:
            continue
        q = result.quantiles[:, index]
        ref = reference.get(name)
        overlap = None
        if ref is not None and all(v is not None and math.isfinite(v) for v in ref[1:3]):
            overlap = bool(q[1] <= ref[2] and ref[1] <= q[2])
        out.append({"name": name, "checked": float(result.best[index]), "reference": ref,
                    "median": float(q[0]), "low68": float(q[1]), "high68": float(q[2]), "low95": float(q[3]),
                    "high95": float(q[4]), "tau": float(result.diagnostics.tau[index]),
                    "rhat": float(result.diagnostics.rhat[index]), "overlap": overlap})
    return out


def verdict(result: CheckResult) -> tuple[str, dict]:
    """(template, values as text) of whether the run is long enough: ``CONVERGED`` or ``NOT_CONVERGED``."""
    d = result.diagnostics
    taus = [t for t in d.tau if math.isfinite(t)]
    rhats = [r for r in d.rhat if not math.isnan(r)]
    values = {"kept": str(d.kept_steps), "factor": f"{mcmc.TAU_FACTOR:g}",
              "tau": f"{max(taus):.0f}" if taus else "—", "rhat": f"{max(rhats):.3f}" if rhats else "—"}
    if d.converged:
        return CONVERGED, {**values, "samples": f"{d.effective_samples:.0f}"}
    return NOT_CONVERGED, {**values, "limit": f"{mcmc.RHAT_LIMIT:g}"}


def record(result: CheckResult, checked: Checked) -> dict:
    """The run as JSON (Save Check…)."""
    def number(value):
        return None if value is None or not math.isfinite(float(value)) else float(value)

    return {
        "software": SOFTWARE, "kind": "gimap_posterior_check_v1", "checked": checked.kind,
        "sampler": "affine-invariant ensemble: differential-evolution (80 %) and stretch (20 %) moves",
        "likelihood": ("Predict parameters' posterior (model error and scales integrated out, the network's prior)"
                       if checked.kind == "predict" else
                       "the fit's: -sum r^2 / (2 s^2), s^2 = chi2_red of the fit, scales solved; uniform in the fit's "
                       "coordinates"),
        "steps": result.steps, "walkers": result.walkers, "ensembles": result.diagnostics.ensembles,
        "walkers_per_ensemble": result.walkers, "burn_in": mcmc.BURN, "seed": result.seed,
        "starts": result.starts_from,
        "rhat_method": "split R-hat across independent ensembles' mean traces",
        "reference_kind": checked.reference_kind,
        "stopped": result.stopped, "seconds": result.seconds, "converged": result.diagnostics.converged,
        "acceptance": number(result.diagnostics.acceptance),
        "values": [{key: (number(value) if isinstance(value, float) else value) for key, value in row.items()
                    if key != "reference"} | {"reference": None if row["reference"] is None else
                                              [number(v) for v in row["reference"]]}
                   for row in rows(result, checked.reference)],
        "note": PREDICT_CHECK if checked.kind == "predict" else CHECK_ONLY,
    }


def report_blocks(result: CheckResult, checked: Checked) -> list:
    """The completed check on the current data and model, in the fit report's block format."""
    template, values = verdict(result)
    parts = [("h2", "Posterior check (MCMC)"),
             ("p", PREDICT_CHECK if checked.kind == "predict" else CHECK_ONLY),
             ("p", f"{result.diagnostics.ensembles} independent ensembles, {result.walkers} walkers per ensemble, "
                   f"{result.steps} steps each. " + template.format(**values))]
    if checked.kind == "predict" and checked.reference_kind == "least_squares":
        parts.append(("p", BIC_REFERENCE))
    if result.stopped:
        parts.append(("p", "Stopped before the requested number of steps."))
    table = []
    for row in rows(result, checked.reference):
        median, low, high = interval_values(row["median"], row["low68"], row["high68"])
        table.append([row["name"], median, f"{low} – {high}", f"{row['tau']:.1f}", f"{row['rhat']:.3f}"])
    parts.append(("table", ["Parameter", "MCMC median", "MCMC 68 %", "τ (steps)", "split R̂"], table))
    return parts


def trace(result: CheckResult, name: str) -> tuple[np.ndarray, np.ndarray]:
    """(steps, values (steps, W)) of one value, for the trace plot."""
    index = result.names.index(name)
    return result.trace_steps, result.trace[:, :, index]


__all__ = ["BIC_REFERENCE", "CHECK_ONLY", "CHANGED", "CONVERGED", "ENSEMBLES", "NOT_CONVERGED", "PREDICT_CHECK",
           "Checked", "NOTHING", "record", "report_blocks", "rows", "run", "target_for", "trace", "verdict"]
