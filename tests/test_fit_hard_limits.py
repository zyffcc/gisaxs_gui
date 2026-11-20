"""Synthetic contracts: every displayed solution respects the requested parameters."""
import math

import numpy as np
import pytest

from src.gimap.features.fitting.application.single_fit import (
    FitData, FitModel, evaluate, fit_resolving, new_component, residuals,
)
from src.gimap.features.fitting.domain.fit_limits import constrain_proposal


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


def test_a_solution_of_analyze_is_put_in_model_under_its_constraints(tmp_path):
    from tests.test_fit_page import _analyze_solution, _curve_file, _page

    page = _page()
    try:
        page.open_curve(_curve_file(tmp_path))
        base = page.session.model.with_parameter((0, "R"), value=4, free=False)
        base = base.with_parameter((0, "Int"), lower=80, upper=110)
        page.set_model(base)
        assert page.show_solution(dict(_analyze_solution(8), best_chi2_weighted=1.0))
        model = page.session.model
        assert model.get((0, "R")).value == 4 and not model.get((0, "R")).free
        assert 80 <= model.get((0, "Int")).value <= 110
        assert (model.get((0, "Int")).lower, model.get((0, "Int")).upper) == (80, 110)
    finally:
        page.dispose()


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


def test_equivalent_dropdown_switches_whole_predicted_solutions_and_export(tmp_path):
    import json

    from tests.test_fit_page import _curve_file, _page, _predicting, _wait

    page = _page()
    try:
        _predicting(page, (6.0, 5.0))
        page.open_curve(_curve_file(tmp_path))
        page.set_method("predict")
        page.run_fit()
        _wait(page)
        solutions = page.session.solutions
        assert page.solution_selector.count() == 2 and page.solution_selector.isEnabled()
        assert page.session.model == solutions[0].model
        page.choose_equivalent_solution(1)
        assert page.session.model == solutions[1].model and page.solution_selector.currentIndex() == 1
        page.export_data_dialog(str(tmp_path / "second.csv"))
        record = json.loads((tmp_path / "second.json").read_text(encoding="utf-8"))
        assert record["model"]["components"][0]["params"]["R"]["value"] == pytest.approx(5.0)
        page.set_range((.2, 1.0))
        assert not page.solution_selector.isEnabled()  # solutions of other points
        page.choose_equivalent_solution(0)
        assert page.session.model == solutions[1].model
    finally:
        page.dispose()
