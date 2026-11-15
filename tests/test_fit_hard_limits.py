"""Synthetic contracts: every displayed solution respects the requested parameters."""
import copy
import math

import numpy as np
import pytest

from src.gimap.features.fitting.application.single_fit import (
    FitData, FitModel, evaluate, fit_resolving, new_component, residuals,
)
from src.gimap.features.fitting.application.single_fit1d import fit1d_request
from src.gimap.features.fitting.domain.fit_limits import constrain_proposal, explicit_limits
from src.gimap.features.fitting.domain.fit1d_bounds import model_constraints


@pytest.mark.parametrize("method", ["local", "global"])
def test_all_limits_and_fixed_values_survive_even_with_resolve_on(method):
    truth = FitModel((new_component("sphere", radius=7),)).with_values({(0, "Int"): 700, ("globals", "background"): 20})
    q = np.linspace(.08, 1.7, 80)
    data = FitData.prepare(q, evaluate(truth, q), np.ones_like(q))
    start = truth.with_parameter((0, "R"), value=4, lower=3, upper=5)
    start = start.with_parameter((0, "Int"), value=100, lower=80, upper=110)
    start = start.with_parameter(("globals", "background"), value=2, lower=1, upper=3)
    start = start.with_parameter((0, "D"), free=False)
    result, actions, _notes = fit_resolving(start, data, method=method, max_evaluations=180, resolve=True)
    assert not actions
    for path, requested in start.parameters():
        actual = result.model.get(path)
        assert (actual.free, actual.lower, actual.upper) == (requested.free, requested.lower, requested.upper)
        if requested.free:
            assert requested.lower <= actual.value <= requested.upper
        else:
            assert actual.value == requested.value
    assert np.sum(residuals(result.model, data) ** 2) <= np.sum(residuals(start, data) ** 2) + 1e-8


def test_proposals_cannot_overwrite_a_fixed_value_or_widen_a_range():
    base = FitModel((new_component("sphere"),)).with_parameter((0, "R"), value=4, free=False)
    base = base.with_parameter((0, "D"), lower=10, upper=20)
    proposal = base.with_parameter((0, "R"), value=80).with_parameter((0, "D"), value=90)
    result = constrain_proposal(proposal, base)
    assert result.get((0, "R")).value == 4
    assert result.get((0, "D")).value == 20
    with pytest.raises(ValueError, match="composition"):
        constrain_proposal(FitModel((new_component("vertical_cylinder"),)), base)


def test_refinement_without_sigma_cannot_discard_a_better_start():
    start = FitModel((new_component("sphere", radius=6),)).with_parameter((0, "Int"), value=2000)
    q = np.geomspace(.08, 1.7, 80)
    data = FitData.prepare(q, evaluate(start, q))
    result, _, _ = fit_resolving(start, data, max_evaluations=40)
    assert result.log_rmse < 1e-10


def test_shape_and_ai_proposals_are_refined_under_the_same_constraints(tmp_path):
    from tests.test_fit_page import _page, _curve_file, _shape_solution

    page = _page()
    try:
        page.open_curve(_curve_file(tmp_path))
        base = page.session.model.with_parameter((0, "R"), value=4, free=False)
        base = base.with_parameter((0, "Int"), lower=80, upper=110)
        for source in ("shape search", "1D Predict"):
            choices = page._as_solutions([_shape_solution(8)], page.session.data(), base, source)
            assert len(choices) == 1
            assert choices[0].model.get((0, "R")).value == 4
            assert 80 <= choices[0].model.get((0, "Int")).value <= 110
    finally:
        page.dispose()


def test_predict_refuses_inapplicable_constraints_instead_of_ignoring_them():
    model = FitModel((new_component("sphere"),))
    data = FitData.prepare([.1, .2, .3], [10, 8, 5])
    with pytest.raises(ValueError, match="Choose The particles in Model"):
        fit1d_request(data, model.with_parameter((0, "R"), free=False), {"fit1d_composition": "auto:vc:3"})
    for path in ((0, "Int"), ("globals", "res_width"), ("globals", "res_amplitude")):
        with pytest.raises(ValueError, match="cannot preserve"):
            model_constraints(model.with_parameter(path, free=False))
    with pytest.raises(ValueError, match="background range"):
        model_constraints(model.with_parameter(("globals", "background"), upper=10))


def test_linear_range_editor_distinguishes_infinity_from_a_finite_limit():
    from tests.test_fit_page import _page

    page = _page()
    try:
        page.set_model(FitModel((new_component("sphere"),)))
        page.show_ranges_check.setChecked(True)
        row = page.model_editor.row((0, "Int"))
        assert row.unbounded.isChecked() and not row.upper.isEnabled()
        row.unbounded.setChecked(False)
        row.upper.setValue(150)
        assert page.session.model.get((0, "Int")).upper == 150
        assert row.upper.isEnabled()
        row.unbounded.setChecked(True)
        assert math.isinf(page.session.model.get((0, "Int")).upper)
        assert not row.upper.isEnabled()
    finally:
        page.dispose()


def test_equivalent_dropdown_switches_whole_native_result_and_export(tmp_path):
    from tests.test_fit_page import _page, _curve_file, _shape_solution

    page = _page()
    try:
        page.open_curve(_curve_file(tmp_path))
        data = page.session.data()
        native = dict(settings=page.fit1d_settings(),
                      summary=dict(chi2_per_point=1., n_points=data.q.size, model="synthetic", status="ok", composition="sphere"),
                      populations=[], table=[], warnings=[],
                      curves=dict(q_nm_inv=data.q.tolist(), intensity=data.intensity.tolist(), sigma=data.sigma.tolist(),
                                  model=data.intensity.tolist(), residual=np.zeros_like(data.q).tolist(), components={}))
        value = dict(native=native, mapping_row=_shape_solution(5), candidates=[])
        alternative = copy.deepcopy(value)
        alternative["native"]["curves"]["model"] = [v * 1.01 for v in value["native"]["curves"]["model"]]
        alternative["delta_chi2"] = 1.0
        value["equivalents"] = [alternative]
        page._fit1d_finished(value, page.session.selection(), page.session.curve)
        assert page.solution_selector.count() == 2
        page.choose_equivalent_solution(1)
        assert page.session.native_result is alternative["native"]
        assert page.session.native_is_current()
        _names, table, record = page._fit1d_export()
        assert table[:, 4] == pytest.approx(alternative["native"]["curves"]["model"])
        assert record["native_result"]["selected_solution"] == 1
        page.set_range((.2, 1.0))
        assert not page.solution_selector.isEnabled()
    finally:
        page.dispose()


def test_repeat_auto_prediction_ignores_only_its_untouched_generated_start(tmp_path):
    from tests.test_fit_page import _page, _curve_file, _shape_solution

    page = _page()
    try:
        page.open_curve(_curve_file(tmp_path))
        page._apply_fit1d_settings({"fit1d_composition": "auto:sphere,vc:3"})
        data = page.session.data()
        native = dict(settings=page.fit1d_settings(),
                      summary=dict(chi2_per_point=1., n_points=data.q.size, model="synthetic", status="ok", composition="sphere"),
                      populations=[], table=[], warnings=[],
                      curves=dict(q_nm_inv=data.q.tolist(), intensity=data.intensity.tolist(), sigma=data.sigma.tolist(),
                                  model=data.intensity.tolist(), residual=np.zeros_like(data.q).tolist(), components={}))
        # A predicted radius can lie outside the manual model's default range.
        page._fit1d_finished(dict(native=native, mapping_row=_shape_solution(.2)),
                            page.session.selection(), page.session.curve)
        assert explicit_limits(page.session.model)
        request = fit1d_request(data, page.fit1d_request_model(), page.fit1d_settings())
        assert not request.payload["constraints"]
        page.set_model(page.session.model.with_parameter((0, "R"), free=False))
        with pytest.raises(ValueError, match="Choose The particles in Model"):
            fit1d_request(data, page.fit1d_request_model(), page.fit1d_settings())
    finally:
        page.dispose()
