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

from src.gimap.shared.version import SOFTWARE

from ..domain import mcmc
from ..domain.fit_engine import FitData
from ..domain.fit_model import FitModel
from ..domain.posterior.refine import model_values
from ..domain.posterior_check import CheckResult, FitTarget, PredictTarget, run_check
from .single_predict import compositions, cut_geometry, cut_qz_minus

NOTHING = ("Nothing to check: put a solution of Predict parameters in Model, or fit with Refine or Search first.")
CHANGED = "The model changed since the run: the check is of the model it was started for."
CONVERGED = ("Converged: {kept} steps kept (the first half discarded) ≥ {factor} × the longest autocorrelation time "
             "({tau} steps), R̂ ≤ {rhat}; about {samples} independent samples.")
NOT_CONVERGED = ("Not converged: {kept} steps kept, the longest autocorrelation time {tau} steps (needs ≥ {factor} "
                 "× of it), R̂ up to {rhat} (needs < {limit}). Run longer before reading the intervals.")
CHECK_ONLY = "A check only: the values and intervals in Results are not changed."


@dataclass(frozen=True)
class Checked:
    """What a run checks: the target and the intervals beside it (name → (median or value, low68, high68, low95,
    high95)), and what they are (``predict`` or ``fit``)."""
    target: object
    kind: str
    reference: dict


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
        qz, _grazing = cut_geometry(model)
        law = model.components[0].law
        coupling = model.components[0].coupling
        target = PredictTarget(data.q, data.intensity, data.sigma, composition=composition, centre=centre, qz=qz,
                               qz_minus=cut_qz_minus(model), law=law, coupling=coupling)
        reference = {name: (v[1], v[2], v[3], v[4], v[5]) for name, v in dict(solution.intervals).items()}
        return Checked(target, "predict", reference)
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
        if ref is not None and all(math.isfinite(v) for v in ref[1:3]):
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
    rhats = [r for r in d.rhat if math.isfinite(r)]
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
        "steps": result.steps, "walkers": result.walkers, "burn_in": mcmc.BURN, "seed": result.seed,
        "stopped": result.stopped, "seconds": result.seconds, "converged": result.diagnostics.converged,
        "acceptance": number(result.diagnostics.acceptance),
        "values": [{key: (number(value) if isinstance(value, float) else value) for key, value in row.items()
                    if key != "reference"} | {"reference": None if row["reference"] is None else
                                              [number(v) for v in row["reference"]]}
                   for row in rows(result, checked.reference)],
        "note": CHECK_ONLY,
    }


def trace(result: CheckResult, name: str) -> tuple[np.ndarray, np.ndarray]:
    """(steps, values (steps, W)) of one value, for the trace plot."""
    index = result.names.index(name)
    return result.trace_steps, result.trace[:, :, index]


__all__ = ["CHECK_ONLY", "CHANGED", "CONVERGED", "NOT_CONVERGED", "Checked", "NOTHING", "record", "rows", "run", "target_for", "trace", "verdict"]
