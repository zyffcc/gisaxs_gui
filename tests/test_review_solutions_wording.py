"""How a solution of Predict parameters is worded: one χ²ᵣ for one solution with the points it is of, every
population in the headline and the table, the refined value with its interval (never a median), one name per
quantity (``shared/fit_names``), no flag for a term that is absent, the notes of background, tail and resolution
after those of the particles (first-user review, items C–G)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.assistant.application import gisaxs_tools
from src.gimap.features.assistant.application.gisaxs_choice import bound_phrases, chance, sizes
from src.gimap.features.assistant.application.gisaxs_export import fit_solutions_csv
from src.gimap.features.assistant.application.gisaxs_procedure import _interval_text, _solution_text
from src.gimap.features.assistant.application.gisaxs_report import fit_rows, gisaxs_batch_row, gisaxs_sections
from src.gimap.features.fitting.application.single_fit import FitModel, new_component
from src.gimap.features.fitting.domain.cut_model import model as cut
from src.gimap.features.fitting.infrastructure.adapters import posterior_job
from src.gimap.features.fitting.presentation.single import predict_texts
from src.gimap.features.fitting.presentation.single.session import Solution
from src.gimap.shared import fit_names
from tests.test_fit_page import _page, _wait

NETWORK = Path(__file__).resolve().parent / "data" / "ml1d" / "flow_fixture"


# -- one name per quantity, one sentence per flag -------------------------------------------------------------


def test_every_key_of_a_quantity_has_the_model_tables_name() -> None:
    for keys, name in ((("res_exponent", "tail_exponent", "nu_Res"), "Tail ν"),
                       (("res_width", "tail_width", "sigma_Res"), "Tail w"),
                       (("res_amplitude", "resolution_amplitude"), "Tail A"), (("q_resolution",), "σ_q")):
        assert {fit_names.display_name(key) for key in keys} == {name}
    assert fit_names.display_name("sigma_R_1") == "σR/R (1)" and fit_names.display_name("Int_2") == "Scale (2)"
    assert fit_names.display_name("sigma_R_1", many=False) == "σR/R"  # one population: no number
    legend = fit_names.legend(2)
    assert "sigma_R_n: σR/R (n)" in legend and "res_exponent / tail_exponent / nu_Res: Tail ν" in legend


def test_flags_read_as_sentences_with_those_names_one_per_kind() -> None:
    flags = ["unconstrained:tail_exponent", "at_bound:res_exponent", "at_bound:sigma_R_1",
             "correlated:Int_2~res_amplitude", "correlated:R_2~sigma_R_2", "low_ess", "bic", "model_error"]
    said = fit_names.sentences(flags)
    text = " | ".join(said)
    for internal in ("tail_exponent", "res_exponent", "sigma_R_1", "Int_2", "res_amplitude", "the prior"):
        assert internal not in text
    assert "Tail ν, σR/R (1) stopped at a bound of its range" in said
    assert "Scale (2) and Tail A; R (2) and σR/R (2) are strongly correlated" in said
    for kind in ("low_ess", "bic", "model_error"):  # each has a human sentence
        assert fit_names.SENTENCES[kind] in said
    assert "median" not in fit_names.SENTENCES["bic"]
    first, second = fit_names.split_flags(flags)
    assert second == ["unconstrained:tail_exponent", "at_bound:res_exponent"]  # not the particles' values
    assert "at_bound:sigma_R_1" in first and "bic" in first


def test_an_absent_term_is_not_flagged() -> None:
    flags = ("at_bound:res_amplitude", "at_bound:res_exponent", "unconstrained:tail_width", "at_bound:R_1",
             "at_bound:Int_2", "unconstrained:R_2", "correlated:res_width~res_exponent")
    kept = fit_names.clean_flags(flags, {"res_amplitude": 0.0, "background": 1.0, "Int_1": 5.0, "Int_2": 0.0})
    assert kept == ("at_bound:R_1", "absent:2")  # no tail: nothing of A, w, ν; population 2 unused: said once
    with_tail = fit_names.clean_flags(flags[:3], {"res_amplitude": 3.0})
    assert with_tail == flags[:3]


def test_an_interval_is_printed_with_digits_that_tell_it_from_its_value() -> None:
    assert fit_names.interval_text(6.0, 5.99, 6.01, "nm") == "6.000 [5.990, 6.010] nm"
    assert fit_names.interval_text(4.556, 4.556, 7.52, "nm") == "4.6 [4.6, 7.5] nm"
    assert fit_names.interval_text(0.2, 0.1996, 0.2004) == "0.20000 [0.19960, 0.20040]"
    assert fit_names.at_edge(4.556, 4.556, 7.52) and not fit_names.at_edge(6.0, 5.99, 6.01)


# -- every population, the same χ²ᵣ, the points it is of -------------------------------------------------------


def _row(**changes) -> dict:
    row = {
        "rank": 1, "composition": "upright cylinder (S) + upright cylinder", "combination": "vertical_cylinder + vertical_cylinder",
        "probability": 0.42, "best_chi2_weighted": 1.05, "chi2_reduced": 1.05, "best_log_rmse": 0.14,
        "points": {"fitted": 596, "given": 598, "below_window": 2, "unusable": 0},
        "components": [
            {"type": "vertical_cylinder", "weight": 0.9, "amplitude": 800.0, "params": {"R": 6.02, "sigma_R": 0.35}},
            {"type": "vertical_cylinder", "weight": 0.1, "amplitude": 70.0,
             "params": {"R": 3.06, "sigma_R": 0.22, "D": 49.4, "sigma_nn": 0.38, "sigma_D": 0.15}},
        ],
        "global_params": {"background": 1.9, "resolution_amplitude": 440.0, "sigma_Res": 0.1, "nu_Res": 1.8},
        "flags": ["unconstrained:tail_exponent", "unconstrained:q_resolution", "at_bound:res_exponent",
                  "correlated:R_1~sigma_R_1", "correlated:Int_1~res_amplitude"],
        "at_bounds": [{"parameter": "nu_Res", "bound": "lower", "component": None}],
        "posterior": {"R_1": [6.02, 6.1, 5.57, 6.31, 5.2, 6.6, 0.3], "R_2": [3.06, 3.1, 3.02, 3.23, 2.9, 3.4, 0.1],
                      "D_2": [49.4, 48.0, 37.8, 65.0, 30.0, 80.0, 9.0]},
        "conditions_source": "network_choice", "compared_by": "evidence",
    }
    row.update(changes)
    return row


def test_the_headline_and_the_table_show_every_population_in_the_order_of_the_radii() -> None:
    solution = gisaxs_tools._solution(_row())
    assert sizes(solution) == "R = 3.06 / 6.02 nm, D = 49 / – nm"
    assert _solution_text(solution) == ("upright cylinder (S) + upright cylinder: R = 3.06 / 6.02 nm, "
                                        "D = 49 / – nm, χ²ᵣ = 1.05, probability 0.42")
    assert gisaxs_tools._brief(solution).startswith("upright cylinder (S) + upright cylinder, R = 3.06 / 6.02 nm, D = ")
    report = {"gisaxs": {"fit": {"solutions": [solution], "chosen": 0}}}
    assert "R 3.06 / 6.02 nm, D 49 / – nm, χ²ᵣ 1.05, p 0.42" in gisaxs_batch_row(report)[2]
    one = gisaxs_tools._solution(_row(components=_row()["components"][1:]))
    assert sizes(one) == "R = 3.06 nm, D = 49.4 nm"  # one population: as before


def test_the_interval_line_gives_the_refined_value_under_the_same_names() -> None:
    solution = gisaxs_tools._solution(_row())
    text = _interval_text(solution)
    assert text == "R (1) 6.02 [5.57, 6.31] nm, R (2) 3.06 [3.02, 3.23] nm, D (2) 49 [38, 65] nm"
    assert "R 1 " not in text
    assert bound_phrases(solution) == ["Tail ν at its lower bound"]
    # the flags' names are the same; a value listed in at_bounds is not repeated as a sentence
    assert solution["warnings"] == ["R (1) and σR/R (1); Scale (1) and Tail A are strongly correlated"]
    assert solution["more_warnings"] == ["the curve says little about Tail ν, σ_q (its 68 % interval spans most of its "
                                         "allowed range)"]


def test_the_report_says_which_points_chi2_is_of_and_keeps_the_nuisance_notes_apart() -> None:
    solution = gisaxs_tools._solution(_row())
    assert solution["chi2"] == 1.05 and solution["points"]["fitted"] == 596
    fit = {"solutions": [solution], "chosen": 0, "curve": "the qy > 0 half", "points": 598,
           "q_range_inv_angstrom": [1e-4, 0.28], "fitted_points": solution["points"]}
    lines = gisaxs_sections({"gisaxs": {"fit": fit}})
    text = "\n".join(lines)
    assert "- fitted: 596 of 598 points; 2 below the network's q window (χ²ᵣ is over the fitted points)" in lines
    assert "| χ²ᵣ |" in text and "χ²/N" not in text and "no degrees-of-freedom" not in text
    assert fit_names.CHI2_DEFINITION in text and fit_names.RELATIVE in text
    assert "nu_Res: Tail ν" in text  # the legend: each key of the records with its name
    table = "\n".join(fit_rows(fit))
    assert "1 more on background, beam tail and resolution (below)" in table and "says little" not in table
    assert "Background, beam tail and resolution — rank 1: the curve says little about Tail ν, σ_q" in text
    csv = fit_solutions_csv(fit)
    assert "# fitted points: 596 of 598 points; 2 below the network's q window" in csv
    assert ",1.05," in csv and "the reduced chi^2" in csv


def test_one_composition_tried_is_not_a_probability_of_one() -> None:
    solution = gisaxs_tools._solution(_row(probability=1.0, conditions_source="user_composition"))
    solution["alone"] = True
    assert chance(solution) == "only this composition was tried"
    assert _solution_text(solution).endswith("χ²ᵣ = 1.05, only this composition was tried")
    assert "| only this composition was tried |" in "\n".join(fit_rows({"solutions": [solution], "chosen": 0}))


# -- Fitting's rows and job -------------------------------------------------------------------------------------


def _curve_with_points_below_the_window():
    q = np.concatenate([[0.003, 0.004, 0.005], np.linspace(0.02, 2.0, 200)])
    population = cut.Population("sphere", 4.0, 0.15, distance=14.0, distance_spread=0.2)
    y = np.random.default_rng(0).poisson(1e4 * cut.unit_terms([population], cut.Settings(), q)[0] + 2.0).astype(float)
    return q, y, np.sqrt(np.maximum(y, 1))


def test_a_row_carries_the_solutions_own_chi2_and_the_points_it_is_of() -> None:
    q, y, sigma = _curve_with_points_below_the_window()
    points = posterior_job.fitted_points(q, y, sigma)
    assert (points["fitted"], points["given"], points["below_window"], points["unusable"]) == (200, 203, 3, 0)
    model = FitModel((new_component("sphere", radius=4.0),))  # Tail A = 0: no beam tail in this model
    assert float(model.globals["res_amplitude"].value) == 0.0
    solution = SimpleNamespace(
        composition=0, name="sphere (S)", probability=1.0, model=model, chi2_reduced=1.234, log_rmse=0.1, method="lm",
        flags=("at_bound:res_amplitude", "at_bound:res_exponent", "unconstrained:tail_exponent", "at_bound:sigma_R_1"),
        log_evidence=0.0, ess=100.0, samples=400, seconds=1.0, values={}, extras={"compared_by": "evidence"})
    item = {"q": q, "observed": y, "sigma": sigma, "side": "positive", "sign": 1.0}
    (row,) = posterior_job.curve_rows(item, [solution], np.linspace(0.02, 2.0, 50), chosen_by="user_composition")
    assert row["best_chi2_weighted"] == row["chi2_reduced"] == 1.234  # the one number, not a mean over other points
    assert row["points"] == {"fitted": 200, "given": 203, "below_window": 3, "unusable": 0}
    assert row["flags"] == ["at_bound:sigma_R_1"]  # nothing about the absent tail
    assert row["warnings"] == ["σR/R stopped at a bound of its range"] and row["more_warnings"] == []
    encoded = posterior_job.encode(solution, points)
    assert encoded["chi2"] == 1.234 and encoded["points"]["below_window"] == 3
    assert encoded["flags"] == ["at_bound:sigma_R_1"]
    tail_at_bound = model.with_parameter(("globals", "res_exponent"), value=model.globals["res_exponent"].lower,
                                         free=True)
    assert posterior_job.at_bounds(tail_at_bound) == []  # Tail ν at its bound means nothing without a tail
    with_tail = tail_at_bound.with_parameter(("globals", "res_amplitude"), value=5.0)
    assert [item["parameter"] for item in posterior_job.at_bounds(with_tail)] == ["nu_Res"]


# -- the Fit page -----------------------------------------------------------------------------------------------


def _solution(model, **changes) -> Solution:
    values = dict(chi2=1.04, source="Predict parameters · sphere (S)", probability=1.0, points=(596, 598, 2, 0), tried=1,
                  intervals=(("R_1", (4.556, 6.8, 4.556, 7.52, 4.2, 8.0, 0.4)),
                             ("sigma_R_1", (0.2, 0.21, 0.15, 0.25, 0.1, 0.3, 0.05)),
                             ("tail_exponent", (1.8, 1.9, 1.8, 2.1, 1.8, 2.2, 0.3))),
                  flags=("unconstrained:tail_exponent", "at_bound:res_exponent", "low_ess"))
    values.update(changes)
    return Solution(model, **values)


def test_fit_details_give_the_value_in_model_with_its_interval_and_few_warning_lines() -> None:
    model = FitModel((new_component("sphere", radius=4.556),))
    solution = _solution(model)
    lines = predict_texts.solution_lines(solution, [solution], many=False)
    text = "\n".join(lines)
    assert "R 4.6 [4.6, 7.5] nm" in lines[0] and "6.8" not in text and "median" not in text
    assert "Only this composition was tried: no probability." in lines and "probability 1.00" not in text
    assert "χ²ᵣ is over 596 of 598 points; 2 below the network's q window." in lines
    assert ("⚠ the refined value lies at the edge of the posterior: the model describes this curve only within the "
            "model error (R)") in lines
    assert sum(line.startswith("⚠") for line in lines) == 2  # the edge and the effective samples
    assert lines[-1].startswith("Background, beam tail and resolution: the curve says little about Tail ν")
    assert "tail_exponent" not in text and "res_exponent" not in text and "the prior" not in text
    by_errors = _solution(model, flags=("bic",), tried=None, probability=0.6)
    lines = predict_texts.solution_lines(by_errors, [by_errors], many=False)
    assert lines[0].startswith("Values in Model with 68 % intervals from the least-squares errors")
    assert "median" not in "\n".join(lines) and "posterior intervals" not in lines[0]
    assert "BIC weight 0.6 among the solutions compared." in lines
    assert not any("edge of the posterior" in line for line in lines)  # an interval from errors has no such edge


def test_the_solutions_table_shows_every_population_the_points_and_what_differs() -> None:
    two = FitModel((new_component("vertical_cylinder", radius=6.02, structure=False),
                    new_component("vertical_cylinder", radius=3.06, distance=49.4)))
    first = _solution(two, source="Predict parameters · a", tried=None, probability=0.42, intervals=(
        ("R_1", (3.06, 3.1, 3.0, 3.2, 2.9, 3.3, 0.1)), ("sigma_nn_1", (0.435, 0.4, 0.2, 0.5, 0.1, 0.5, 0.2)),
        ("q_resolution", (0.0103, 0.01, 0.001, 0.1, 0.0001, 0.1, 0.03))))
    second = _solution(two.with_parameter((1, "sigma_nn"), value=0.201), source="Predict parameters · b", tried=None,
                       probability=0.25, intervals=(
        ("R_1", (3.06, 3.1, 3.0, 3.2, 2.9, 3.3, 0.1)), ("sigma_nn_1", (0.201, 0.2, 0.1, 0.3, 0.1, 0.5, 0.2)),
        ("q_resolution", (0.031, 0.03, 0.001, 0.1, 0.0001, 0.1, 0.03))))
    assert predict_texts.sizes_cells(two) == ("3.06 / 6.02", "49.4 / –")
    tip = predict_texts.differing_tip(second, [first, second])
    assert tip == ("Differs from solution 1 in: σnn/D (1) 0.20 (there 0.44); σ_q 0.031 nm⁻¹ (there 0.010 nm⁻¹).")
    page = _page()
    try:
        page.session.solutions = [first, second]
        page.session.solutions_selection = page.session.selection()
        page.set_model(two)
        page.render_results()
        table = page.solutions_table
        assert [table.item(0, column).text() for column in (2, 3, 4, 5)] == ["3.06 / 6.02", "49.4 / –", "1.04", "0.42"]
        assert table.horizontalHeaderItem(4).text() == "χ²ᵣ"
        assert table.horizontalHeaderItem(4).toolTip() == fit_names.CHI2_DEFINITION
        assert "relative to the compositions that were tried" in table.horizontalHeaderItem(5).toolTip()
        assert "χ²ᵣ is over 596 of 598 points; 2 below the network's q window." in table.item(0, 4).toolTip()
        assert "Differs from solution 1 in: σnn/D (1) 0.20 " in table.item(1, 1).toolTip()
        assert page.quality_label.text().startswith("2 solutions below")
        assert page.quality_label.toolTip() == fit_names.CHI2_DEFINITION
        page.session.solutions = [first]
        page.render_results()
        assert page.quality_label.text().startswith("1 solution below.")
        assert "uncertainties where estimable" in page.quality_label.text()
    finally:
        page.dispose()


def test_predict_then_refine_on_the_page_says_the_points_of_each_chi2(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GIMAP_ML1D_NETWORK", str(NETWORK))
    q, y, sigma = _curve_with_points_below_the_window()
    path = tmp_path / "spheres.dat"
    np.savetxt(path, np.column_stack([q / 10, y, sigma]))
    page = _page()
    try:
        assert page.open_curve(path), page.status_label.text()
        page.set_model(FitModel((new_component("sphere", radius=3.0),)))
        page.set_method("predict")
        page.predict_particles.setCurrentIndex(page.predict_particles.findData("model"))
        page.run_fit()
        _wait(page)
        solutions = page.session.solutions
        assert solutions and solutions[0].points == (200, 203, 3, 0) and solutions[0].tried == 1
        status = page.status_label.text()
        assert "solution; the best" in status or "solutions; the best" in status
        assert "1 solutions" not in status
        chi2 = f"{solutions[0].chi2:.3g}"
        assert f"(χ²ᵣ {chi2} over 200 of 203 points; 3 below the network's q window)" in status
        assert page.solutions_table.item(0, 4).text() == chi2  # the same number in the table
        details = page.warnings_label.text()
        assert "Values in Model with" in details and "Posterior median [central 68 % interval]" in details
        assert "χ²ᵣ is over 200 of 203 points; 3 below the network's q window." in details
        radius = page.session.model.components[0].value("R")
        _, _, low, high, *_ = dict(solutions[0].intervals)["R_1"]
        assert radius == pytest.approx(4.0, rel=0.05) and "R " + fit_names.interval_text(radius, low, high, "nm") in details
        page.run_fit("local")
        _wait(page)
        result = page.session.result
        assert result is not None and result.points == 203
        assert ("Refine fitted 203 points; Predict parameters fitted 200 of 203 (3 below the network's q window): the "
                "two χ²ᵣ are not of the same points.") in page.warnings_label.text()
        table = page.parameters_table
        for line in range(table.rowCount()):  # a free value without an error says why
            plus = table.item(line, 2)
            assert plus.text() != ""
            if plus.text() == "–":
                assert plus.toolTip().startswith("No error: ")
    finally:
        page.dispose()


def test_a_free_value_without_an_error_says_why() -> None:
    model = FitModel((new_component("sphere", radius=4.0),))
    at_bound = SimpleNamespace(at_bounds=[(0, "sigma_R")])
    parameter = model.get((0, "sigma_R"))
    assert predict_texts.no_error_reason((0, "sigma_R"), parameter, at_bound, model) == (
        "No error: the value is at a bound of its range.")
    nothing = SimpleNamespace(at_bounds=[])
    assert predict_texts.no_error_reason((0, "R"), model.get((0, "R")), nothing, model) == (
        "No error: the curve does not constrain this value.")
    unused = model.with_parameter((0, "Int"), value=0.0)
    assert predict_texts.no_error_reason((0, "R"), unused.get((0, "R")), nothing, unused) == (
        "No error: the Scale of this population is 0, so its values change nothing.")
    assert predict_texts.no_error_reason(("globals", "res_width"), model.get(("globals", "res_width")), nothing,
                                         model) == "No error: Tail A is 0, so this value changes nothing."
    assert predict_texts.count_solutions(1) == "1 solution" and predict_texts.count_solutions(3) == "3 solutions"
