"""A fit1d job for the single-curve page, independent of V5/TensorFlow assets."""

from __future__ import annotations

import numpy as np

from src.gimap.app.jobs import JobRequest

from .workflow_v5 import validate_options

SHAPES = {"sphere": 1, "vertical_cylinder": 3}


def fit1d_request(data, model, settings: dict) -> JobRequest:
    """The families in Model are conditions; fit1d searches its own physical ranges."""
    if not 1 <= len(model.components) <= 4 or any(c.family not in SHAPES for c in model.components):
        raise ValueError("fit1d needs 1–4 particles in Model: Sphere or Vertical cylinder")
    options = validate_options(dict(method="fit1d", components=[SHAPES[c.family] for c in model.components],
                                    **settings))
    return JobRequest(
        handler="src.gimap.features.fitting.infrastructure.adapters.single_fit1d:run_single_fit1d",
        payload=dict(q=data.q.tolist(), intensity=data.intensity.tolist(),
                     sigma=None if data.sigma is None else data.sigma.tolist(), options=options),
        timeout_seconds=None,
    )


def native_export(result: dict) -> tuple[list[str], np.ndarray]:
    """Measured points, effective errors and the original v2 curve/terms, without re-evaluation."""
    curves = result["curves"]
    q = np.asarray(curves["q_nm_inv"], float)
    names = ["q_nm^-1", "q_A^-1", "I", "sigma", "model", "residual"]
    columns = [q, q / 10.0, curves["intensity"], curves["sigma"], curves["model"], curves["residual"]]
    for name, values in curves.get("components", {}).items():
        names.append(name.replace(" ", "_").replace(":", ""))
        columns.append(values)
    return names, np.column_stack(columns)


__all__ = ["fit1d_request", "native_export"]
