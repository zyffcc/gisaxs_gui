"""Predict parameters on the single-curve page: what the proposal network is asked (the particles, the cut's q_z and
geometry) and what comes back (solutions as models, with their probability and posterior intervals)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.gimap.app.jobs import JobRequest

from ..domain import fit_model as fm
from ..domain.posterior import encoding as E
from ..domain.posterior import network as _network

HANDLER = "src.gimap.features.fitting.infrastructure.adapters.posterior_job:run_predict"
MAX_PARTICLES = E.SLOTS
"""Predict takes one to this many populations."""


def network_available() -> bool:
    """Whether the proposal network (its weights and record) is where GIMaP looks for it."""
    return _network.available()


def network_file() -> str:
    """The file name of the proposal network's weights (for the message when it is missing)."""
    return _network.path().name

PARTICLES = (("The particles in Model", "model"),
             ("Spheres and upright cylinders (1–2)", "spheres_cylinders"),
             ("Any particles (1–3)", "any"))
"""What Predict tries: the particles of Model (their families, and a paracrystal or not), or a choice of the network
among compositions of these families."""

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
    points: Optional[dict] = None
    """The points χ² is of: ``fitted``, ``given``, ``below_window`` (below the network's q window), ``unusable``."""
    tried: Optional[int] = None
    """How many compositions were given to try (1: the particles in Model); None: the network chose among several."""
    diagnostics: dict = field(default_factory=dict)
    posterior_summary: dict = field(default_factory=dict)
    importance_starts: tuple = ()


MANUAL_ONLY = ("Predict parameters knows spheres, random and vertical cylinders with the 1D paracrystal: Model has a "
               "core–shell sphere or another interference function. Use Refine or Search, or set Particles to a choice.")


def compositions(model: fm.FitModel, particles: str) -> dict:
    """``{"compositions": [...]}`` for the particles of Model, or ``{"allowed": [...]}`` for a choice."""
    if particles == "model":
        if not 1 <= len(model.components) <= E.SLOTS:
            raise ValueError(f"Predict needs 1–{E.SLOTS} particles in Model, or Particles set to a choice")
        ordered = sorted(model.components, key=lambda c: c.value("R"))
        if any(c.family not in fm.NETWORK_FAMILIES or (c.structure and c.interference != fm.DEFAULT_INTERFERENCE)
               for c in ordered):
            raise ValueError(MANUAL_ONLY)
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


def cut_qz_minus(model: fm.FitModel) -> Optional[float]:
    """k_i − k_f of the cut (nm⁻¹) when Model has both angles (spheres then get the exit-reflected path with a free
    effective reflection coefficient), else None (the Born term at q_z, or the effective interface)."""
    geometry = fm.grazing(model)
    if geometry is None or geometry.qz_born <= 0:
        return None
    minus = geometry.k_i - geometry.k_f
    return float(minus) if abs(minus) < geometry.qz_born else None


def predict_request(data, model: fm.FitModel, particles: str, *, seed: int = 0) -> JobRequest:
    qz, grazing = cut_geometry(model)
    law = model.components[0].law if model.components else fm.DEFAULT_LAW
    coupling = model.components[0].coupling if model.components else "lma"
    heights = {str(slot): c.value("h") for slot, c in enumerate(sorted(model.components, key=lambda c: c.value("R")), 1)
               if particles == "model" and c.family == "vertical_cylinder" and "h" in c.keys()}
    return JobRequest(handler=HANDLER, payload=dict(known_heights=heights,
        q=data.q.tolist(), intensity=data.intensity.tolist(), sigma=None if data.sigma is None else data.sigma.tolist(),
        qz=qz, qz_minus=cut_qz_minus(model), grazing=grazing or None, seed=int(seed), law=law,
        coupling=coupling, **compositions(model, particles)), timeout_seconds=None)


def predicted(value: dict) -> list[Predicted]:
    """The solutions of a finished job, most probable first."""
    out = []
    for row in value.get("solutions") or ():
        chi2 = row.get("chi2")
        out.append(Predicted(fm.model_from_dict(row["model"]), float("nan") if chi2 is None else float(chi2),
                             str(row.get("name", "")), float(row.get("bic_weight", row.get("probability")) or 0.0),
                             {k: tuple(v) for k, v in (row.get("values") or {}).items()}, tuple(row.get("flags") or ()),
                             float(row.get("ess") or 0.0), row.get("points") or value.get("points"),
                             value.get("tried"), row.get("diagnostics") or {}, row.get("posterior_summary") or {},
                             tuple(row.get("importance_starts") or ())))
    return out


__all__ = ["HANDLER", "MAX_PARTICLES", "PARTICLES", "Predicted", "compositions", "cut_geometry", "cut_qz_minus",
           "network_available", "network_file", "predict_request", "predicted"]
