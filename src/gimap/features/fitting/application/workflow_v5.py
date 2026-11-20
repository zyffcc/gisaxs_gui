"""The options of the 1D Predict window (its one method: Predict parameters); no runtime imports. Options saved with
the methods retired in 0.2.0 — V5's network, the Single RC specialist, the physical fit and fit1d — open as Predict
parameters."""

import numpy as np

from .curve_units import FROM_FILE, FROM_REQUEST, FROM_SETTING, q_unit_read  # noqa: F401 - the origins, also from here

METHODS = ("posterior",)
RETIRED_METHODS = ("model", "stable", "experimental", "fit1d")
"""V5's network, the Single RC specialist, the physical fit and fit1d (retired in 0.2.0): Predict parameters instead."""
RETIRED = ("numerical", "amplitude_calibration", "search_combinations", "condition_combinations", "normalizer",
           "sigma_res", "nu_res", "fit1d_policy", "fit1d_mode", "fit1d_model", "fit1d_sigma_floor")
"""Options only those had."""

STATED_UNITS = {"angstrom": "A^-1", "nm": "nm^-1"}
"""A unit of q as Fitting names it (``CurveData.q_source_unit``) → the value of the option ``q_unit``."""
_SETTING_UNITS = {option: unit for unit, option in STATED_UNITS.items()}


def q_unit_of_file(stated_unit, path, options: dict) -> dict:
    """How 1D Predict reads the q of the curve file ``path``: Fitting's rule (``curve_units.q_unit_read``: the unit its
    header states, else the option ``q_unit``, the setting for files without one). ``stated_unit`` reads the header
    (``CurveRepository.stated_q_unit``). ``{"q_unit_read": "A^-1" | "nm^-1", "q_unit_from": FROM_FILE |
    FROM_SETTING}``, or ``{"q_unit_read": None, "q_unit_from": None, "error": why}`` when the header cannot be read or
    states both units: such a file is not fitted. The window shows it before a run, the job follows it and records
    it."""
    try:
        unit, origin = q_unit_read(stated_unit, path, _SETTING_UNITS[options["q_unit"]])
    except (OSError, ValueError) as exc:
        return dict(q_unit_read=None, q_unit_from=None, error=str(exc) or type(exc).__name__)
    return dict(q_unit_read=STATED_UNITS[unit], q_unit_from=origin)


def default_options() -> dict:
    return dict(
        method="posterior",
        components=[],
        max_solutions=8,
        relative_noise=0.10,
        absolute_noise=0.0,
        q_unit="nm^-1",
        render_points=500,
        side="both",
        cut_qz=None,
    )


def validate_options(options: dict) -> dict:
    out = {**default_options(), **options}
    # Legacy recipes migrate this value into detector preprocessing at capture/runtime.
    out.pop("cbf_gap_margin", None)
    if "fit1d_qz" in out:  # the cut's q_z, saved under fit1d's name before 0.2.0
        qz = out.pop("fit1d_qz")
        if out.get("cut_qz") is None:
            out["cut_qz"] = qz
    for retired in RETIRED:  # options saved with the retired methods
        out.pop(retired, None)
    if out["method"] in RETIRED_METHODS:
        out["method"] = "posterior"
    if out["method"] not in METHODS:
        raise ValueError("Unknown fitting method")
    if out["q_unit"] not in ("nm^-1", "A^-1"):
        raise ValueError("q unit must be nm^-1 or A^-1")
    if out["side"] not in ("both", "positive", "negative"):
        raise ValueError("Unknown q side")
    types = out["components"]
    # Immutable in-situ Recipe snapshots freeze JSON lists as tuples.
    if not isinstance(types, (list, tuple)) or len(types) > 3 or any(t not in (1, 2, 3) for t in types):
        raise ValueError("Use 1–3 components: sphere=1, random cylinder=2, vertical cylinder=3; or Auto")
    out["components"] = list(types)
    qz = out["cut_qz"]
    if qz is not None and (isinstance(qz, bool) or not np.isfinite(qz) or not 0 <= qz <= 10):
        raise ValueError("cut_qz must be within [0, 10] nm^-1, or empty")
    for key, lo, hi in (
        ("max_solutions", 1, 100),
        ("render_points", 2, 2000),
    ):
        if not isinstance(out[key], int) or not lo <= out[key] <= hi:
            raise ValueError(f"{key} must be an integer in [{lo}, {hi}]")
    for key in ("relative_noise", "absolute_noise"):
        if not np.isfinite(out[key]) or out[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    return out
