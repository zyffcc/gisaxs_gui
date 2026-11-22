"""The flag ``qzR``: a sphere or random cylinder fitted at q_z = 0 (the effective interface, Predict's default) whose
radius times the cut's q_z = k(sin α_i + sin α_f) is above 1.7, where that treatment biases the radius (Physics and
Methods 9.10, 9.13). A flag and its sentence only: no value, interval or probability of a solution changes."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.gimap.features.assistant.application import gisaxs_choice
from src.gimap.features.assistant.application.gisaxs_tools import cut_qz
from src.gimap.features.fitting.domain import fit_model as fm
from src.gimap.features.fitting.domain.posterior import refine
from src.gimap.shared import fit_names

QZ = 0.5
"""The cut's q_z (nm⁻¹) of these tests: a radius of 5 nm gives q_z·R = 2.5."""


def _model(*populations, qz_born: float = 0.0, reflection: float = 0.0) -> fm.FitModel:
    """Populations ``(family, R)`` without a paracrystal; ``qz_born`` > 0: the cut's angles at λ = 0.1 nm that give
    that q_z (α_i = α_f, as Predict sets them for a cut with q_z alone), with |r| = ``reflection``."""
    model = fm.FitModel(tuple(fm.new_component(family, radius=radius, distance=None, structure=False)
                              for family, radius in populations))
    if qz_born > 0:
        angle = math.degrees(math.asin(0.5 * qz_born / (2 * math.pi / 0.1)))
        for key, value in (("alpha_i", angle), ("alpha_f", angle), ("wavelength", 0.1), ("refl_amp", reflection)):
            model = model.with_parameter(("globals", key), value=value)
    return model


# -- the flag of a solution ---------------------------------------------------------------------------------------


def test_a_sphere_beyond_the_limit_is_flagged_at_q_z_0_and_not_with_the_angles() -> None:
    sphere = _model(("sphere", 5.0))
    assert fm.grazing(sphere) is None  # fitted without the angles: q_z = 0
    assert refine.interface_flags(sphere, QZ) == ["qzR:R_1"]  # q_z·R = 2.5
    born = _model(("sphere", 5.0), qz_born=QZ)  # level 2: the Born term at the cut's q_z
    assert fm.grazing(born).qz_born == pytest.approx(QZ)
    assert refine.interface_flags(born, QZ) == []
    assert refine.interface_flags(_model(("sphere", 5.0), qz_born=QZ, reflection=0.8), QZ) == []  # two paths
    assert refine.interface_flags(_model(("sphere", 3.2)), QZ) == []  # q_z·R = 1.6: within the limit
    assert refine.QZR_LIMIT == 1.7


def test_upright_cylinders_are_not_flagged_and_the_populations_keep_their_numbers() -> None:
    assert refine.interface_flags(_model(("vertical_cylinder", 5.0)), QZ) == []  # its cut is exact at any q_z
    assert refine.interface_flags(_model(("cylinder", 5.0)), QZ) == ["qzR:R_1"]  # random cylinders: also at q_z = 0
    mixed = _model(("sphere", 8.0), ("vertical_cylinder", 5.0), ("sphere", 2.0))
    assert refine.interface_flags(mixed, QZ) == ["qzR:R_3"]  # in the order of the radii, as large / small


def test_without_the_angles_of_the_cut_there_is_no_flag() -> None:
    sphere = _model(("sphere", 50.0))
    for unknown in (None, 0.0, float("nan")):
        assert refine.interface_flags(sphere, unknown) == []


def test_the_flag_reads_as_one_sentence_with_the_model_tables_names() -> None:
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language, tr

    template = fit_names.SENTENCES["qzR"]
    assert f"above {refine.QZR_LIMIT:g}" in template and gisaxs_choice.QZ_R_LIMIT == refine.QZR_LIMIT
    one = fit_names.sentences(["qzR:R_1"], many=False)
    assert one == [template.format(name="R")] and "both angles of the cut" in one[0] and "too small" in one[0]
    assert fit_names.sentences(["qzR:R_1", "qzR:R_3"])[0].startswith("R (1), R (3): q_z·R is above 1.7")
    assert fit_names.split_flags(["qzR:R_2", "at_bound:res_exponent"]) == (["qzR:R_2"], ["at_bound:res_exponent"])
    assert fit_names.clean_flags(("qzR:R_1", "qzR:R_2"), {"Int_1": 0.0, "Int_2": 3.0}) == ("qzR:R_2", "absent:1")
    apply_language("zh")
    try:
        said = tr(template)
        assert said != template and said.startswith("{name}：在切线的 q_z 处 q_z·R 大于 1.7") and "{name}" in said
    finally:
        apply_language(DEFAULT_LANGUAGE)


def test_the_fit_page_says_the_flag_with_the_other_notes() -> None:
    from src.gimap.features.fitting.presentation.single import predict_texts
    from src.gimap.features.fitting.presentation.single.session import Solution

    solution = Solution(_model(("sphere", 5.0)), 1.02, "Predict parameters · sphere", probability=1.0, tried=1,
                        intervals=(("R_1", (5.0, 5.0, 4.9, 5.1, 4.8, 5.2, 0.05)),), flags=("qzR:R_1",))
    lines = predict_texts.solution_lines(solution, [solution], many=False)
    assert "⚠ " + fit_names.SENTENCES["qzR"].format(name="R") in lines


def test_predict_gives_the_same_values_with_and_without_the_flag() -> None:
    """The real network on a synthetic sphere cut (R = 5 nm, the composition given, the fixed seed): the cut's q_z
    adds the flag and its sentence and changes nothing else of any solution."""
    from src.gimap.features.fitting.bootstrap import create_curve_fit
    from src.gimap.features.fitting.domain.cut_model import model as cut

    q = np.linspace(0.1, 2.5, 240)
    population = cut.Population("sphere", 5.0, 0.15, distance=18.0, distance_spread=0.2)
    truth = 1e4 * cut.unit_terms([population], cut.Settings(), q)[0] + 2.0
    counts = np.random.default_rng(4).poisson(truth).astype(float)
    sigma = np.sqrt(np.maximum(counts, 1.0))
    fit = create_curve_fit()
    plain = fit(q / 10.0, counts, sigma, components=(1,))  # q in Å⁻¹
    flagged = fit(q / 10.0, counts, sigma, components=(1,), cut_qz=QZ)

    def rest(row: dict) -> str:
        diagnostic = {k: v for k, v in row.get("diagnostics", {}).items() if k not in ("measured_qz", "reflected_beam")}
        return json.dumps({k: v for k, v in row.items() if k not in ("flags", "warnings", "seconds", "diagnostics", "reproduction")}
                          | {"diagnostics": diagnostic}, sort_keys=True)

    assert [rest(row) for row in plain] == [rest(row) for row in flagged]
    plans = [rows[0]["reproduction"] for rows in (plain, flagged)]
    assert [plan["options"]["measured_qz"] for plan in plans] == [None, QZ]
    assert [plan["options"]["reflected_beam"] for plan in plans] == [None, True]
    def numerical_tables(plan):
        return {name: {key: column for key, column in columns.items() if key not in ("measured_qz", "reflected_beam")}
                for name, columns in plan["reference"].items()}
    assert numerical_tables(plans[0]) == numerical_tables(plans[1])
    assert not any(flag.startswith("qzR") for row in plain for flag in row["flags"])
    best = flagged[0]
    assert best["components"][0]["params"]["R"] == pytest.approx(5.0, rel=0.1) and "qzR:R_1" in best["flags"]
    assert "qzR_cov:R_1" in best["flags"]
    assert [flag for flag in best["flags"] if flag not in ("qzR:R_1", "qzR_cov:R_1")] == plain[0]["flags"]
    assert fit_names.SENTENCES["qzR"].format(name="R") in best["warnings"]


# -- the automatic analysis ------------------------------------------------------------------------------------


def test_the_cut_q_z_is_taken_from_the_geometry_and_the_yoneda_band_or_not_at_all() -> None:
    status = {"geometry": {"wavelength_A": 1.0, "incidence_deg": 0.4},
              "gisaxs": {"horizontal_source": "yoneda", "yoneda_alpha_f_deg": 0.15}}
    found = cut_qz(status)
    k = 2 * math.pi / 0.1
    assert found["q_z_inv_nm"] == pytest.approx(k * (math.sin(math.radians(0.4)) + math.sin(math.radians(0.15))))
    assert (found["alpha_i_deg"], found["alpha_f_deg"], found["wavelength_A"]) == (0.4, 0.15, 1.0)
    for changed in ({"gisaxs": {**status["gisaxs"], "horizontal_source": "manual"}},  # a cut set by hand
                    {"gisaxs": {**status["gisaxs"], "yoneda_alpha_f_deg": None}},
                    {"geometry": {**status["geometry"], "incidence_deg": 0.0}},  # no αi
                    {"geometry": {"incidence_deg": 0.4}}):  # no λ
        assert cut_qz({**status, **changed}) is None, changed
    assert cut_qz(None) is None


def test_bic_gisaxs_report_preserves_weights_and_flags_and_names_their_kind():
    from types import SimpleNamespace

    from src.gimap.features.assistant.application.gisaxs_export import fit_solutions_csv
    from src.gimap.features.assistant.application.gisaxs_procedure import GisaxsProcedureMixin
    from src.gimap.features.assistant.application.gisaxs_report import gisaxs_sections
    from src.gimap.features.assistant.application.gisaxs_tools import _solution
    from src.gimap.features.assistant.presentation.guided_gisaxs import outcome_summary

    flags = ["bic", "qzR_cov:R_1", "qzR_fail:R_1", "cyl_sphere:R_1"]
    base = dict(rank=1, composition="cylinder", flags=flags, compared_by="bic", bic_weight=0.7, best_chi2_weighted=1.,
                components=[{"type": "cylinder", "params": {"R": 6.}}],
                posterior_summary={"R_1": {"median": 5.8, "low68": 5.5, "high68": 5.9}},
                least_squares_values={"R_1": 6.}, ess=24., diagnostics={"compared_by": "bic", "posterior_kind": "tempered"})
    solution = _solution(base)
    legacy = {**base, "rank": 2, "probability": 0.3}
    del legacy["bic_weight"]
    fit = {"solutions": [solution, _solution(legacy)], "chosen": 0}
    report = {"gisaxs": {"fit": fit}}
    assert fit["solutions"][1]["bic_weight"] == 0.3 and "probability" not in solution
    assert "BIC weight 0.70" in outcome_summary(report)
    table = fit_solutions_csv(fit)
    assert "rank,model,chi2,bic_weight," in table and "R_posterior_median" in table and "ess," in table
    text = "\n".join(gisaxs_sections(report))
    assert "| BIC weight |" in text and "Effective sample size 24" in text and "| R_1 | 6 | 5.8 |" in text

    procedure = GisaxsProcedureMixin()
    attention = []
    procedure._attention = lambda item, why, option, hint: attention.append(why)
    procedure._decide = lambda *args: None
    procedure.progress = lambda message: None
    procedure._call = lambda name: fit
    procedure.catalog = SimpleNamespace(results=SimpleNamespace(gisaxs=report["gisaxs"]))
    procedure._fit()
    assert any("BIC weight of at least 0.2" in item for item in attention)
    assert all(fit_names.SENTENCES[flag.split(":")[0]].format(name="R") in attention for flag in flags[1:])


def test_the_caveat_names_each_flagged_population_with_its_q_z_r() -> None:
    solution = {"model": "sphere + sphere (S)", "flags": ["qzR:R_2", "low_ess"],
                "components": [{"type": "sphere", "R": 6.0}, {"type": "sphere", "R": 2.0}]}
    text = gisaxs_choice.interface_caveat(solution, {"q_z_inv_nm": QZ})
    assert text == gisaxs_choice.QZ_R.format(model="sphere + sphere (S)", qz="0.5", values="3.00", names="R (2)",
                                             limit="1.7")
    assert gisaxs_choice.interface_caveat(solution, None) == ""  # the cut's angles not known
    assert gisaxs_choice.interface_caveat({**solution, "flags": ["low_ess"]}, {"q_z_inv_nm": QZ}) == ""


def test_a_sphere_too_large_for_q_z_0_needs_attention_in_the_report() -> None:
    import tools.gimap_agent as agent
    from src.gimap.app.headless_assistant import open_headless_session
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
    from src.gimap.features.assistant.application.pipeline import PipelineOptions
    from src.gimap.features.assistant.application.pipeline_report import pipeline_markdown
    from src.gimap.features.assistant.presentation.guided_words import words
    from tests.test_guided_gisaxs import DATA, GALAXI, FakeFitter, _geometry, _row

    class LargeSphere(FakeFitter):
        def __call__(self, q, intensity, sigma, *, components=(), report=None, cancelled=None, cut_qz=None):
            self.calls.append({"cut_qz": cut_qz})
            return [_row(q, intensity, sigma, 1, "sphere", 9.4, 53.4, 1.1, flags=["qzR:R_1"])]

    session = open_headless_session(DATA / GALAXI["frame"], saved_profiles=False)
    try:
        session.catalog.fitter = fitter = LargeSphere()
        report = session.run_pipeline(PipelineOptions(technique="gisaxs", geometry=_geometry(),
                                                      incidence_deg=GALAXI["incidence_deg"]))
    finally:
        session.close()
    fit = report["gisaxs"]["fit"]
    cut = fit["cut_qz"]
    assert fit["sphere_model"] == "effective interface (q_z = 0)"
    assert report["gisaxs"]["cuts"]["horizontal_source"] == "yoneda"
    assert cut["alpha_i_deg"] == pytest.approx(GALAXI["incidence_deg"]) and cut["wavelength_A"] == pytest.approx(1.34)
    assert cut["alpha_f_deg"] == pytest.approx(report["gisaxs"]["cuts"]["yoneda_alpha_f_deg"])
    assert fitter.calls[-1]["cut_qz"] == pytest.approx(cut["q_z_inv_nm"]) and 0.3 < cut["q_z_inv_nm"] < 1.0
    point = next(item for item in report["needs_attention"] if item["item"] == "fit")
    assert point["why"].startswith("The reported solution (sphere) was fitted with the form factor at q_z = 0")
    assert f"q_z·R is {9.4 * cut['q_z_inv_nm']:.2f} for R, above 1.7" in point["why"]
    assert point["hint"] == gisaxs_choice.QZ_R_HINT and agent.exit_code([report]) == 2
    caveats = [item["decision"] for item in report["decisions"] if item["what"] == "fit caveat"]
    assert fit_names.SENTENCES["qzR"].format(name="R") in caveats
    assert point["why"] in pipeline_markdown(report).split("## Geometry")[0]  # under Needs attention
    apply_language("zh")
    try:
        said = words(point["why"])
        assert said.startswith("报告的解（sphere）是用 q_z = 0 处的形状因子") and "大于 1.7" in said
        assert words(point["hint"]).startswith("请发送到拟合")
    finally:
        apply_language(DEFAULT_LANGUAGE)
