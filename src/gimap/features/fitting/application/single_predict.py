"""Predict parameters on the single-curve page: what the proposal network is asked (the particles, the cut's q_z and
geometry) and what comes back (solutions as models, with their probability and posterior intervals)."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.gimap.app.jobs import JobRequest

from ..domain import fit_model as fm
from ..domain.posterior import encoding as E

HANDLER = "src.gimap.features.fitting.infrastructure.adapters.posterior_job:run_predict"
PARTICLES = (("The particles in Model", "model"),
             ("Spheres and upright cylinders (1–2)", "spheres_cylinders"),
             ("Any particles (1–3)", "any"))
"""What Predict tries: the particles of Model (their families, and a paracrystal or not), or a choice of the network
among compositions of these families."""
METHOD_TITLES = {"lm": "Refine each solution (least squares)", "global": "Search from the samples",
                 "posterior": "Posterior samples only"}


@dataclass(frozen=True)
class Predicted:
    """One solution of Predict: a model of Fitting, refined, with what the posterior says of it."""
    model: fm.FitModel
    chi2: float
    name: str
    probability: float
    intervals: dict = field(default_factory=dict)
    """Name (``R_1``, ``D_1``, ``tail_width`` …) → (best, median, low68, high68, low95, high95, error)."""
    flags: tuple = ()
    ess: float = 0.0


def compositions(model: fm.FitModel, particles: str) -> dict:
    """``{"compositions": [...]}`` for the particles of Model, or ``{"allowed": [...]}`` for a choice."""
    if particles == "model":
        if not 1 <= len(model.components) <= E.SLOTS:
            raise ValueError(f"Predict needs 1–{E.SLOTS} particles in Model, or Particles set to a choice")
        ordered = sorted(model.components, key=lambda c: c.value("R"))
        return {"compositions": [E.composition_index([(c.family, bool(c.structure)) for c in ordered])]}
    if particles == "spheres_cylinders":
        return {"allowed": E.select(("sphere", "vertical_cylinder"), 2)}
    if particles == "any":
        return {"allowed": E.select()}
    raise ValueError(f"Unknown choice of particles {particles!r}")


def cut_geometry(model: fm.FitModel) -> tuple[float, dict]:
    """(vacuum q_z of the cut in nm⁻¹, Model's grazing values to keep): q_z = k_i + k_f with the angles of Model;
    0 and nothing without them (the effective interface)."""
    geometry = fm.grazing(model)
    if geometry is None:
        return 0.0, {}
    return float(geometry.qz_born), {key: float(model.globals[key].value) for key in fm.GRAZING}


def predict_request(data, model: fm.FitModel, particles: str, *, method: str = "lm", samples: int = 2000,
                    seed: int = 0) -> JobRequest:
    qz, grazing = cut_geometry(model)
    law = model.components[0].law if model.components else fm.DEFAULT_LAW
    coupling = model.components[0].coupling if model.components else "lma"
    return JobRequest(handler=HANDLER, payload=dict(
        q=data.q.tolist(), intensity=data.intensity.tolist(), sigma=None if data.sigma is None else data.sigma.tolist(),
        qz=qz, grazing=grazing or None, method=method, samples=int(samples), seed=int(seed), law=law,
        coupling=coupling, **compositions(model, particles)), timeout_seconds=None)


def predicted(value: dict) -> list[Predicted]:
    """The solutions of a finished job, most probable first."""
    out = []
    for row in value.get("solutions") or ():
        chi2 = row.get("chi2")
        out.append(Predicted(fm.model_from_dict(row["model"]), float("nan") if chi2 is None else float(chi2),
                             str(row.get("name", "")), float(row.get("probability") or 0.0),
                             {k: tuple(v) for k, v in (row.get("values") or {}).items()}, tuple(row.get("flags") or ()),
                             float(row.get("ess") or 0.0)))
    return out


__all__ = ["HANDLER", "METHOD_TITLES", "PARTICLES", "Predicted", "compositions", "cut_geometry", "predict_request",
           "predicted"]
