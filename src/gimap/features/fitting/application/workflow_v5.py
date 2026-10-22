"""V5 options and portable artifact location; no runtime imports."""

import os
from pathlib import Path
import numpy as np

from .curve_units import FROM_FILE, FROM_REQUEST, FROM_SETTING, q_unit_read  # noqa: F401 - the origins, also from here

FIT1D_SHAPES = {1: "sphere", 3: "vc"}
"""Component id → the shape name of the fit1d kit. It has no random cylinder (2)."""
FIT1D_POLICIES = ("fast", "auto", "robust")
FIT1D_MODES = ("gisaxs", "saxs")

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


def bundled_workflow() -> Path:
    return (
        Path(__file__).resolve().parents[5] / "modules/Fitting_1D_Model/Workflow_v5/conditional_v2"
    )


def fit1d_kit_home() -> Path:
    """The folder of the fit1d kit (it contains ``gisaxs1d_kit/``): the environment variable
    ``GISAXS1D_KIT_HOME``, else ``utils/gisaxs1d_gui_kit`` in the program folder. Not imported here."""
    env = os.environ.get("GISAXS1D_KIT_HOME")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[5] / "utils/gisaxs1d_gui_kit"


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
        fit1d_policy="fast",
        fit1d_mode="gisaxs",
        fit1d_model="general",
        fit1d_qz=None,
        fit1d_sigma_floor=0.0,
    )


def _validate_fit1d(out: dict) -> None:
    """The fit1d values are checked for every method (they are saved with it); the composition and the
    resolution only when fit1d is the method."""
    if out["fit1d_policy"] not in FIT1D_POLICIES:
        raise ValueError("fit1d_policy must be fast, auto or robust")
    if out["fit1d_mode"] not in FIT1D_MODES:
        raise ValueError("fit1d_mode must be gisaxs or saxs")
    if not isinstance(out["fit1d_model"], str) or not out["fit1d_model"].strip():
        raise ValueError("fit1d_model must name a network of the fit1d kit (general, au_np_si)")
    qz = out["fit1d_qz"]
    if qz is not None and (isinstance(qz, bool) or not np.isfinite(qz) or not 0 <= qz <= 10):
        raise ValueError("fit1d_qz must be within [0, 10] nm^-1, or empty")
    floor = out["fit1d_sigma_floor"]
    if isinstance(floor, bool) or not np.isfinite(floor) or not 0 <= floor <= 0.5:
        raise ValueError("fit1d_sigma_floor must be a relative error within [0, 0.5]")
    if out["method"] != "fit1d":
        return
    if not out["components"]:
        raise ValueError(
            "fit1d needs the complete composition (Sphere / Vertical cylinder): "
            "one curve does not identify it"
        )
    if 2 in out["components"]:
        raise ValueError("fit1d has no random cylinder: use Sphere and Vertical cylinder")
    if out["sigma_res"] is not None:
        raise ValueError("fit1d cannot fix sigma_res; leave it free (nu_res can be fixed)")
    nu = out["nu_res"]
    if nu is not None and (not np.isfinite(nu) or not 0.5 <= nu <= 12.0):
        raise ValueError("nu_res must be within [0.5, 12.0] for fit1d")


def validate_options(options: dict) -> dict:
    out = {**default_options(), **options}
    # Legacy recipes migrate this value into detector preprocessing at capture/runtime.
    out.pop("cbf_gap_margin", None)
    if out["method"] not in ("stable", "model", "experimental", "fit1d"):
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
    _validate_fit1d(out)
    if out["method"] == "fit1d":
        resolution_bounds = ()
    elif out["method"] in ("stable", "experimental"):
        resolution_bounds = (("sigma_res", 0.001, 0.1), ("nu_res", 1.0, 20.0))
    else:
        resolution_bounds = (("sigma_res", 0.007, 0.013), ("nu_res", 5.0, 10.0))
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
