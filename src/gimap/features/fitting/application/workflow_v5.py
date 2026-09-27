"""V5 options and portable artifact location; no runtime imports."""

from pathlib import Path
import numpy as np


def bundled_workflow() -> Path:
    return (
        Path(__file__).resolve().parents[5] / "modules/Fitting_1D_Model/Workflow_v5/conditional_v2"
    )


def default_options() -> dict:
    return dict(
        method="model",
        components=[],
        sigma_res=None,
        nu_res=None,
        numerical=True,
        amplitude_calibration=True,
        search_combinations=12,
        condition_combinations=3,
        max_solutions=8,
        relative_noise=0.10,
        absolute_noise=0.0,
        normalizer=None,
        q_unit="nm^-1",
        render_points=500,
        side="both",
    )


def validate_options(options: dict) -> dict:
    out = {**default_options(), **options}
    # Legacy recipes migrate this value into detector preprocessing at capture/runtime.
    out.pop("cbf_gap_margin", None)
    if out["method"] not in ("stable", "model", "experimental"):
        raise ValueError("Unknown fitting method")
    if not isinstance(out["amplitude_calibration"], bool):
        raise ValueError("amplitude_calibration must be true or false")
    if out["q_unit"] not in ("nm^-1", "A^-1"):
        raise ValueError("q unit must be nm^-1 or A^-1")
    if out["side"] not in ("both", "positive", "negative"):
        raise ValueError("Unknown q side")
    types = out["components"]
    # Immutable in-situ Recipe snapshots freeze JSON lists as tuples.
    if (
        not isinstance(types, (list, tuple))
        or len(types) > 4
        or any(t not in (1, 2, 3) for t in types)
    ):
        raise ValueError(
            "Use 1–4 components: sphere=1, random cylinder=2, vertical cylinder=3; or Auto"
        )
    out["components"] = list(types)
    resolution_bounds = (
        (("sigma_res", 0.001, 0.1), ("nu_res", 1.0, 20.0))
        if out["method"] in ("stable", "experimental")
        else (("sigma_res", 0.007, 0.013), ("nu_res", 5.0, 10.0))
    )
    for key, lo, hi in resolution_bounds:
        value = out[key]
        if value is not None and (not np.isfinite(value) or not lo <= value <= hi):
            raise ValueError(f"{key} must be within [{lo}, {hi}]")
    for key, lo, hi in (
        ("search_combinations", 1, 34),
        ("condition_combinations", 1, 34),
        ("max_solutions", 1, 100),
        ("render_points", 2, 2000),
    ):
        if not isinstance(out[key], int) or not lo <= out[key] <= hi:
            raise ValueError(f"{key} must be an integer in [{lo}, {hi}]")
    for key in ("relative_noise", "absolute_noise"):
        if not np.isfinite(out[key]) or out[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if out["normalizer"] is not None and (
        not np.isfinite(out["normalizer"]) or out["normalizer"] <= 0
    ):
        raise ValueError("Intensity normalizer must be positive")
    return out
