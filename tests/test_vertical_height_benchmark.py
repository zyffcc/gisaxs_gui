"""The height benchmark records the joint-fit dialog's diagnostics on the same counts in both ranges."""

import json

import numpy as np
import pytest

from src.gimap.features.fitting.application import joint_fit as joint
from src.gimap.features.fitting.domain.fit_model import evaluate
from tools.benchmarks import vertical_height_bias as bench


def model_cuts(case):
    model = bench.page_model(case).with_values({
        (0, "R"): bench.RADIUS, (0, "h"): bench.HEIGHT, (0, "D"): case.distance, (0, "sigma_R"): bench.SPREAD,
        (0, "sigma_h"): bench.SPREAD, (0, "sigma_nn"): bench.SIGMA_NN, ("globals", "background"): 0.0})
    q = np.linspace(0.06, 2.4, 100)
    angles = np.linspace(0.04, 1.2, 96)
    qz = bench.K * (np.sin(np.radians(bench.ALPHA_I)) + np.sin(np.radians(angles)))
    horizontal = joint.CurveSpec("row", q, np.ones_like(q), alpha_i=bench.ALPHA_I, alpha_f=bench.ALPHA_C,
                                 wavelength=bench.WAVELENGTH)
    vertical = joint.CurveSpec("column", qz, np.ones_like(qz), kind="vertical", alpha_i=bench.ALPHA_I,
                               wavelength=bench.WAVELENGTH, q_parallel=bench.Q_Y)
    row, column = (joint.curve_model(spec, model, bench.SUBSTRATE, True) for spec in (horizontal, vertical))
    # Two heights on the column give the range check a known discrepancy even on noise-free data.
    signal = np.where(angles < column.cut.min_exit_angle_deg, evaluate(column.with_values({(0, "h"): 4.8}), qz),
                      evaluate(column, qz))
    return dict(q=q, qz=qz, horizontal=evaluate(row, q), vertical=signal, seconds=0.0)


def test_case_json_matches_the_page_fit_and_its_warnings(tmp_path, monkeypatch):
    from src.gimap.features.fitting.presentation.single.joint_fit_dialog import JointFitDialog
    from tests.test_fit_page import _page

    page = _page()
    dialog = JointFitDialog(page, page.view_model, page)
    try:
        page_limit, page_method = dialog.joint_min_exit.value(), dialog.joint_method.currentData()
    finally:
        dialog.dispose()
        page.dispose()
        page.deleteLater()

    results = []
    run_joint_fit = joint.run_joint_fit

    def run(specs, *args, **kwargs):
        result = run_joint_fit(specs, *args, **kwargs)
        results.append((result, specs))
        return result

    case = bench.CASES[0]
    monkeypatch.setattr(joint, "run_joint_fit", run)
    monkeypatch.setattr(bench, "bornagain_cuts", model_cuts)
    monkeypatch.setattr(bench, "uniform_layer_check", lambda: {"average layer on": 0.0, "average layer off": 0.0})
    path = tmp_path / "height.json"
    bench.main(["--cases", case.name, "--seeds", "1", "--workers", "1", "--out", str(path)])
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["code"]["commit"] and record["code"]["fitting_tree"]
    assert isinstance(record["code"]["fitting_code_modified"], bool)
    assert record["windows_min_exit_angle_ac"] == {"default": page_limit, "full": 0.0}
    saved, = record["cases"]
    fits = saved["fits"]
    assert len(fits) == len(results) == 4
    by_draw = {(row["window"], row["seed"]): row for row in fits}
    assert by_draw.keys() == {(w, s) for w in ("default", "full") for s in (None, 1)}
    for row, (result, specs) in zip(fits, results):
        default = row["window"] == "default"
        assert result.method == page_method
        assert result.models[1].cut.min_exit_angle_ac == (page_limit if default else 0.0)
        assert result.height_range_check.min_exit_angle_ac == (0.0 if default else page_limit)
        comparison, = row["height_range_rows"]
        assert row["height_range_rows"] == joint.height_range_rows(result)
        other = by_draw["full" if default else "default", row["seed"]]
        assert comparison["selected_nm"] == row["h"]
        assert comparison["alternate_nm"] == pytest.approx(other["h"])
        assert row["relative_bias"] == pytest.approx({k: row[k] / v - 1 for k, v in bench.truth(case).items()})
        assert row["vertical_chi2_reduced"] == pytest.approx(result.reduced_chi2_for(1))
        messages = joint.needs_attention(result, specs)
        assert row["needs_attention"] == [template.format(**args) for template, args in messages]
        fired = {template for template, _ in messages}
        layer, height = joint.LAYER_WARNING in fired, joint.HEIGHT_RANGE_WARNING in fired
        assert row["warnings"] == dict(layer=layer, height_range=height, either=layer or height)
        if row["seed"] == 1:
            assert saved["warning_fractions"][row["window"]] == {k: float(v) for k, v in row["warnings"].items()}
    assert by_draw["default", None]["warnings"] == dict(layer=False, height_range=True, either=True)
    assert by_draw["full", None]["warnings"]["layer"]


def test_warning_fractions_count_only_poisson_draws_and_count_either_once():
    rows = [dict(seed=seed, warnings=dict(layer=layer, height_range=height, either=layer or height))
            for seed, layer, height in ((None, True, True), (1, False, True), (2, True, True), (3, False, False))]
    assert bench.warning_fractions(rows) == dict(layer=1/3, height_range=2/3, either=2/3)
    assert bench.warning_fractions(rows[:1]) == dict(layer=None, height_range=None, either=None)
