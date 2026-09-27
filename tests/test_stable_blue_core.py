"""Numerical and routing contracts for the shipped local prediction branch."""

import json

import numpy as np
import pytest

from src.gimap.features.fitting.application.workflow_v5 import validate_options
from src.gimap.features.fitting.domain.blue_rc_forward import forward
from src.gimap.features.fitting.infrastructure.adapters.stable_blue import (
    BluePredictor,
    _row,
    calibrate_amplitudes,
    eligibility,
    encode_features,
    route_diagnostics,
)
from tools.blue_curve_distillation import EDGES, features as research_features
from tools.blue_curve_distillation import decode, forward as research_forward


def case():
    q = np.linspace(0.003, 4.2, 571)
    p = decode(np.full(11, 0.5))
    cs = [
        dict(
            type_id=2,
            amplitude=p["A"],
            params={k: p[k] for k in ("R", "sigma_R", "D", "sigma_D", "h", "sigma_h")},
        )
    ]
    gs = dict(
        background=p["B"], resolution_amplitude=p["C"], sigma_Res=p["sigma_res"], nu_Res=p["nu_res"]
    )
    y = forward(q, cs, gs)
    item = dict(
        q=q,
        observed=y,
        count=np.full(len(q), 6.0),
        observation_metadata=dict(
            source="native_cbf_columns",
            intensity_unit="counts_per_pixel",
            threshold_enabled=False,
            mirror_replaced_pixels=0,
            stack_count=1,
        ),
    )
    return item, cs, gs


def test_runtime_and_training_forward_features_have_exact_semantics():
    item, cs, gs = case()
    np.testing.assert_allclose(
        forward(item["q"], cs, gs), research_forward(item["q"], np.full(11, 0.5)), rtol=1e-13
    )
    item["count"][200:220] = 0
    a = encode_features(item["q"], item["observed"], item["count"], EDGES)
    b = research_features(item["q"], item["observed"], item["count"])
    np.testing.assert_array_equal(a, b)


def test_one_linear_solve_recovers_amplitudes_without_moving_shapes():
    item, cs, gs = case()
    seed_cs = [{**cs[0], "amplitude": cs[0]["amplitude"] * 0.8}]
    seed_gs = {
        **gs,
        "background": gs["background"] * 2,
        "resolution_amplitude": gs["resolution_amplitude"] * 0.9,
    }
    out_cs, out_gs, pred = calibrate_amplitudes(
        item["q"], item["observed"], item["count"], seed_cs, seed_gs, 0.10, return_prediction=True
    )
    assert out_cs[0]["params"] == cs[0]["params"]
    assert out_gs["sigma_Res"] == gs["sigma_Res"] and out_gs["nu_Res"] == gs["nu_Res"]
    np.testing.assert_allclose(pred, item["observed"], rtol=1e-10)
    np.testing.assert_allclose(forward(item["q"], out_cs, out_gs), pred, rtol=1e-13)


def test_routing_metric_rejects_bad_shape_without_altering_observations():
    item, _, _ = case()
    before = item["observed"].copy()
    assert route_diagnostics(item, before)["accepted_for_fast_route"]
    assert not route_diagnostics(item, before * 3)["accepted_for_fast_route"]
    np.testing.assert_array_equal(before, item["observed"])


@pytest.mark.parametrize(
    "flag,value", [("threshold_enabled", True), ("mirror_replaced_pixels", 2), ("stack_count", 3)]
)
def test_altered_counting_preprocessing_routes_to_numerical(flag, value):
    item, _, _ = case()
    item["observation_metadata"][flag] = value
    assert eligibility(item, validate_options(dict(method="stable", components=[2]))) is not None


def test_calibration_default_does_not_inherit_legacy_numerical_flag():
    options = validate_options(dict(method="stable", numerical=False))
    assert options["method"] == "stable" and options["amplitude_calibration"]
    assert not validate_options(dict(amplitude_calibration=False))["amplitude_calibration"]


def test_unknown_composition_does_not_implicitly_use_the_single_rc_specialist():
    item, _, _ = case()
    assert validate_options({})["method"] == "model"
    reason = eligibility(item, validate_options(dict(method="stable")))
    assert "explicitly selected" in reason
    assert eligibility(item, validate_options(dict(method="stable", components=[2]))) is None


def test_checksum_failure_prevents_loading_network(tmp_path):
    (tmp_path / "protocol.json").write_text("{}")
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps(dict(forward_version="blue_rc_gauss48_96_v1", files={"protocol.json": "wrong"}))
    )
    with pytest.raises(ValueError, match="checksum"):
        BluePredictor(tmp_path)


def test_flat_background_fit_warns_that_particle_parameters_are_unconstrained():
    item, cs, gs = case()
    item.update(observed=np.full(len(item["q"]), 10.0), sigma=np.ones(len(item["q"])), sign=1, side="positive")
    out_cs, out_gs, curve = calibrate_amplitudes(
        item["q"], item["observed"], item["count"], cs, gs, 0.1, return_prediction=True
    )
    checks = route_diagnostics(item, curve)
    assert checks["accepted_for_fast_route"]
    result = _row(item, validate_options({}), out_cs, out_gs, "stable_amplitude_calibrated", checks, 0, "test", curve)
    assert result["max_particle_intensity_fraction"] < 1e-6
    assert "unconstrained" in result["warnings"][0]
    np.testing.assert_allclose(result["native_fit"], item["observed"], atol=1e-10)
