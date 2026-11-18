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
        shortlist=int(payload.get("shortlist", 4)), samples=int(payload.get("samples", 2000)),
        method=payload.get("method", "lm"), seed=int(payload.get("seed", 0)),
        law=payload.get("law", "number_gauss"), coupling=payload.get("coupling", "lma"),
        grazing=payload.get("grazing"))
    if cancelled():
        raise RuntimeError("Fitting cancelled")
    report(1, 1, "Done")
    return {"solutions": [encode(s) for s in solutions],
            "network": {"name": posterior_network.path().stem, "sha256": network.meta.get("sha256")},
            "seconds": time.perf_counter() - tick}


__all__ = ["encode", "run_predict"]
