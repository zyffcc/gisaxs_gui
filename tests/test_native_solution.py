"""A solution of 1D Predict's earlier engines (``native_v5``: sizes by intensity, σD/D) put into Fitting's manual
model: the same curve, stated when not. Its curves come from those engines' own forward model (V5's
numpy_forward64), as their solutions do; the quick fit gives its solutions in Fitting's conventions now
(test_experimental_fitting.py)."""

from __future__ import annotations


import numpy as np
import pytest

from scipy.special import j1

from src.gimap.features.fitting.domain.native_solution import native_solution_mapping


# The forward model of 1D Predict's earlier solutions (V5's numpy_forward64.component, retired in 0.2.0, as it was):
# sizes Gaussian by intensity on 25 (sphere), 26 (vertical cylinder) or 13 × 13 nodes and 24 orientations (random
# cylinder), the paracrystal with σD/D; fit1d's rows are converted the same way.
def _nodes(mu, sigma, n, nsig):
    xx = np.linspace(max(mu - nsig * sigma, 0.0), mu + nsig * sigma, n)
    w = np.exp(-0.5 * ((xx - mu) / max(sigma, 1e-12)) ** 2)
    return np.maximum(xx, 1e-8), w / w.sum()


def _sphere(x):
    a = np.abs(x) < 0.1
    safe = np.where(a, 1.0, x)
    x2 = x * x
    return np.where(a, 1 - x2 / 10 + x2 * x2 / 280 - x2**3 / 15120, 3 * (np.sin(safe) - safe * np.cos(safe)) / safe**3)


def _radial(x):
    a = np.abs(x) < 1e-4
    safe = np.where(a, 1.0, x)
    return np.where(a, 1 - x * x / 8 + x**4 / 192, 2 * j1(safe) / safe)


def _component(q, typ, r, sr, h, sh, dist, sd):
    if typ == 1:
        rr, w = _nodes(r, r * sr, 25, 4)
        form = w @ (_sphere(rr[:, None] * q[None, :]) ** 2)
    elif typ == 3:
        rr, w = _nodes(r, r * sr, 26, 3)
        form = w @ (_radial(rr[:, None] * q[None, :]) ** 2)
    else:
        rr, wr = _nodes(r, r * sr, 13, 4)
        hh, wh = _nodes(h, h * sh, 13, 4)
        alpha = np.linspace(0, np.pi / 2, 24)
        wa = np.sin(alpha)
        wa /= wa.sum()
        fr = _radial(rr[:, None, None] * np.sin(alpha)[None, :, None] * q[None, None, :])
        fh = np.sinc(hh[:, None, None] * np.cos(alpha)[None, :, None] * q[None, None, :] / (2 * np.pi))
        form = np.sum(wa[:, None] * np.sum(wr[:, None, None] * fr**2, 0) * np.sum(wh[:, None, None] * fh**2, 0), 0)
    lp = -np.pi * q * q * (dist * sd) ** 2
    phi = np.exp(lp)
    return form * (-np.expm1(2 * lp)) / np.maximum((-np.expm1(lp)) ** 2 + 4 * phi * np.sin(0.5 * q * dist) ** 2, 1e-15)


def forward(q, components, globals_):
    q = np.abs(np.asarray(q, float))
    result = np.full_like(q, globals_["background"])
    for c in components:
        p = c["params"]
        result += c["amplitude"] * _component(q, c["type_id"], p["R"], p["sigma_R"], p.get("h") or 10,
                                              p.get("sigma_h") or 0.2, p["D"], p["sigma_D"])
    return result + globals_["resolution_amplitude"] / (1 + (q / globals_["sigma_Res"]) ** globals_["nu_Res"])


Q_NM = np.linspace(0.05, 2.5, 240)
GLOBALS = {"background": 1.3, "resolution_amplitude": 800.0, "sigma_Res": 0.02, "nu_Res": 4.0}
SPHERE = {"R": 9.4, "sigma_R": 0.3, "D": 53.4, "sigma_D": 0.38}
CYLINDER = {"R": 6.1, "sigma_R": 0.45, "h": 20.7, "sigma_h": 0.18, "D": 398.0, "sigma_D": 0.19}
VERTICAL = {"R": 7.8, "sigma_R": 0.32, "D": 31.2, "sigma_D": 0.38}
TYPE_IDS = {"sphere": 1, "random_cylinder": 2, "vertical_cylinder": 3}


def _row(kind: str, params: dict, amplitude: float = 1200.0, global_params=None) -> dict:
    components = [{"type": kind, "type_id": TYPE_IDS[kind], "weight": 1.0, "amplitude": amplitude, "params": params}]
    curve = forward(Q_NM, components, GLOBALS)
    return {"workflow": "native_v5", "combination": kind, "components": components,
            "global_params": dict(GLOBALS if global_params is None else global_params),
            "native_q": (-Q_NM).tolist(), "native_fit": curve.tolist()}


@pytest.mark.parametrize("kind, params", [("sphere", SPHERE), ("random_cylinder", CYLINDER)])
def test_the_manual_model_draws_the_same_curve(kind, params) -> None:
    result = native_solution_mapping(_row(kind, params))
    assert result.max_deviation < 1e-6  # the same formulas once the spreads are in nm
    component = result.mapping.components[0]
    assert component.shape == {"sphere": "Sphere", "random_cylinder": "Cylinder"}[kind]
    values = component.parameters
    assert values["intensity"] == pytest.approx(1200.0, rel=1e-6)
    assert values["radius"] == params["R"] and values["sigma_radius"] == pytest.approx(params["R"] * params["sigma_R"])
    assert values["sigma_diameter"] == pytest.approx(params["D"] * params["sigma_D"])  # σD in nm in Fitting
    if kind == "random_cylinder":
        assert values["sigma_height"] == pytest.approx(params["h"] * params["sigma_h"])
    globals_ = result.mapping.global_parameters
    assert globals_["background"] == pytest.approx(1.3) and globals_["int_res"] == pytest.approx(800.0)
    assert (globals_["sigma_res"], globals_["nu_res"], globals_["k_value"]) == (0.02, 4.0, 1.0)


def test_a_vertical_cylinder_is_loaded_as_a_start_and_the_difference_is_measured() -> None:
    result = native_solution_mapping(_row("vertical_cylinder", VERTICAL))
    values = result.mapping.components[0].parameters
    assert result.mapping.components[0].shape == "Vertical Cylinder"
    assert values["sigma_radius"] == pytest.approx(0.32)  # Fitting's Vertical Cylinder takes σR/R
    assert values["intensity"] > 0 and result.max_deviation > 0.05  # radii weighted by R⁴ there: not the same curve


def test_background_terms_in_other_units_are_found_from_the_curve() -> None:
    normalised = {**GLOBALS, "background": 1.3e-3, "resolution_amplitude": 0.8}
    result = native_solution_mapping(_row("sphere", SPHERE, global_params=normalised))
    assert result.max_deviation < 1e-6
    assert result.mapping.global_parameters["background"] == pytest.approx(1.3, rel=1e-4)


def test_a_solution_without_components_or_with_an_unknown_family_is_refused() -> None:
    with pytest.raises(ValueError):
        native_solution_mapping({"components": []})
    with pytest.raises(ValueError, match="no manual model"):
        native_solution_mapping({"components": [{"type": "core_shell", "params": SPHERE}]})


def test_a_1d_predict_row_without_distance_or_resolution_peak_converts() -> None:
    """1D Predict gives D = None (no interference) and normalised background terms (sigma_Res None)."""
    components = [{"type": "sphere", "type_id": 1, "weight": 0.7, "amplitude": 900.0,
                   "params": {"R": 8.0, "sigma_R": 0.3, "h": None, "sigma_h": None, "D": None, "sigma_D": None}}]
    flat = {**GLOBALS, "resolution_amplitude": 0.0}
    no_structure = [{**components[0], "params": {**components[0]["params"], "D": 1e9, "sigma_D": 0.01}}]
    curve = forward(Q_NM, no_structure, flat)  # S(q) → 1 for a huge, ordered distance
    row = {"combination": "sphere", "components": components,
           "global_params": {"rho_BG": 1e-4, "sigma_Res": None, "nu_Res": None, "rho_Res": None},
           "native_q": Q_NM.tolist(), "native_fit": curve.tolist()}
    result = native_solution_mapping(row)
    assert result.mapping.components[0].parameters["diameter"] == 0.0  # no S(q) in the manual model
    assert result.mapping.global_parameters["int_res"] == 0.0
    assert result.mapping.global_parameters["background"] == pytest.approx(1.3, rel=1e-3)  # found from the curve
    assert result.max_deviation < 1e-3
