"""Links between parameters and inequality constraints in Refine and Search (``domain.fit_links``, ``linked_fit``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.gimap.features.fitting.application.single_fit import FitData, FitModel, evaluate, fit_record, new_component
from src.gimap.features.fitting.domain import fit_engine, fit_links
from src.gimap.features.fitting.domain.fit_model import model_from_dict, model_to_dict

Q = np.geomspace(0.05, 2.0, 250)


def _two_populations():
    truth = FitModel((new_component("sphere", radius=3.0, distance=12.0),
                      new_component("sphere", radius=6.0, structure=False)))
    return truth.with_values({(0, "Int"): 3000.0, (1, "Int"): 20000.0, (0, "sigma_R"): 0.12, (1, "sigma_R"): 0.12,
                              ("globals", "background"): 2.0})


def _data(truth, seed=1):
    counts = np.random.default_rng(seed).poisson(evaluate(truth, Q)).astype(float)
    return FitData.prepare(Q, counts, np.sqrt(np.maximum(counts, 1.0)))


@pytest.mark.parametrize("text", ["__import__('os')", "R_1.real", "a[0]", "lambda: 1", "R_1 if 1 else 2",
                                  "open('x')", "(lambda x: x)(1)", "sqrt", "R_1 == 2", "[R_1]", "f'{R_1}'"])
def test_an_expression_is_numbers_names_and_a_few_functions_only(text):
    with pytest.raises(ValueError):
        fit_links.Expression(text)({"R_1": 1.0})


def test_expressions_evaluate_without_eval():
    expression = fit_links.Expression("2*R_1 + sqrt(D_1) - max(1, sigma_R_1) ** 2 / pi")
    assert expression.names == {"R_1", "D_1", "sigma_R_1"}
    assert expression({"R_1": 3.0, "D_1": 16.0, "sigma_R_1": 0.2}) == pytest.approx(6 + 4 - 1 / math.pi)
    with pytest.raises(ValueError, match="exponent"):
        fit_links.Expression("9**9**9")({})
    with pytest.raises(ValueError):
        fit_links.Expression("log(R_1)")({"R_1": -1.0})


def test_shared_spread_and_a_radius_ratio_are_fitted_with_propagated_errors():
    truth = _two_populations()
    data = _data(truth)
    start = truth.with_values({(0, "R"): 2.6, (1, "R"): 5.0, (0, "sigma_R"): 0.2, (1, "sigma_R"): 0.2, (0, "D"): 14.0})
    start = fit_links.linked(start, (1, "sigma_R"), "sigma_R_1")
    start = fit_links.linked(start, (1, "R"), "2*R_1")
    assert start.get((1, "R")).value == pytest.approx(5.2) and not start.get((1, "R")).free
    result = fit_engine.fit(start, data)
    model = result.model
    assert model.get((0, "R")).value == pytest.approx(3.0, rel=0.02)
    assert model.get((1, "R")).value == pytest.approx(2 * model.get((0, "R")).value, rel=1e-12)
    assert model.get((1, "sigma_R")).value == model.get((0, "sigma_R")).value
    assert result.errors[(1, "R")] == pytest.approx(2 * result.errors[(0, "R")], rel=1e-3)  # σ of 2·R₁ is 2σ
    assert result.errors[(1, "sigma_R")] == pytest.approx(result.errors[(0, "sigma_R")], rel=1e-3)
    assert (1, "R") not in result.covariance_paths and result.free == len(result.covariance_paths)


def test_a_linked_scale_joins_the_scales_solved_exactly():
    truth = _two_populations().with_values({(1, "Int"): 24000.0})
    data = _data(truth, seed=2)
    start = fit_links.linked(truth.with_values({(0, "Int"): 1.0}), (1, "Int"), "8*Int_1")
    solved = fit_engine.fit_scales(start, data)
    assert solved.get((1, "Int")).value == pytest.approx(8 * solved.get((0, "Int")).value)
    assert solved.get((0, "Int")).value == pytest.approx(3000.0, rel=0.02)
    with pytest.raises(ValueError, match="linear"):
        fit_engine.fit(fit_links.linked(truth, (1, "Int"), "Int_1**2/1000"), data)
    with pytest.raises(ValueError, match="scale"):
        fit_links.linked(truth, (1, "R"), "Int_1/1000")


def test_links_refuse_circles_and_unknown_names_and_follow_a_removed_component():
    model = fit_links.linked(_two_populations(), (1, "R"), "2*R_1")
    with pytest.raises(ValueError, match="circle|itself"):
        fit_links.linked(model, (0, "R"), "R_2 / 2")
    with pytest.raises(ValueError, match="does not have"):
        fit_links.linked(model, (0, "R"), "R_3")
    three = FitModel((new_component("sphere"), *model.components), model.globals)
    three = three.with_parameter((2, "R"), tie="2*R_2")
    moved = fit_links.renumbered(three, 0)
    assert moved.get((2, "R")).tie == "2*R_1"
    gone = fit_links.renumbered(three, 1)
    assert gone.get((2, "R")).tie is None


def test_an_inequality_holds_in_the_fit_and_contact_can_be_switched_off():
    truth = _two_populations()
    data = _data(truth)
    start = fit_links.with_inequalities(truth.with_values({(0, "R"): 2.7, (0, "D"): 13.0}),
                                        [fit_links.Inequality.parse("D_1 >= 3*R_1 + 4")])
    result = fit_engine.fit(start, data)
    assert result.model.get((0, "D")).value >= 3 * result.model.get((0, "R")).value + 4 - 1e-9
    assert fit_links.Inequality.parse("D_1 ≥ 2*R_1").relation == ">="
    with pytest.raises(ValueError):
        fit_links.Inequality.parse("2*R_1 >= D_1 + 1 >= 3")
    off = FitModel(truth.components, truth.globals, contact=False)
    assert fit_engine._Problem(off, data, None, None, 1).pairs == []
    assert fit_engine._Problem(truth, data, None, None, 1).pairs


def test_links_constraints_and_contact_are_in_the_model_file_and_the_record():
    model = fit_links.linked(_two_populations(), (1, "R"), "2*R_1")
    model = fit_links.with_inequalities(model, [fit_links.Inequality.parse("D_1 >= 2*R_1 + 1")])
    model = FitModel(model.components, model.globals, contact=False, inequalities=model.inequalities)
    saved = model_to_dict(model)
    assert saved["components"][1]["params"]["R"]["tie"] == "2*R_1" and saved["contact"] is False
    assert saved["constraints"] == [{"parameter": "D_1", "relation": ">=", "expression": "2*R_1 + 1"}]
    assert model_to_dict(model_from_dict(saved)) == saved
    plain = model_to_dict(_two_populations())
    assert "contact" not in plain and "constraints" not in plain and "tie" not in plain["globals"]["background"]
    data = _data(_two_populations())
    result = fit_engine.fit(model, data)
    record = fit_record(result.model, data, None, side="mean", q_range=None, result=result)
    assert record["links"] == ["R_2 = 2*R_1", "D_1 >= 2*R_1 + 1"]
    names = record["fit"]["correlation"]["parameters"]
    matrix = np.array(record["fit"]["correlation"]["matrix"], dtype=float)
    assert matrix.shape == (len(names), len(names)) and "R_2" not in names
    assert np.allclose(np.diag(matrix)[np.isfinite(np.diag(matrix))], 1.0)
