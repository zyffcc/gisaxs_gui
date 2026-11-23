"""Inputs and provenance of the search baseline, including replay of the archived first set."""
import json

import numpy as np
import pytest

from tools.benchmarks import check_case_overlap, search_baseline as search


def test_search_and_predict_fit_identical_curves(monkeypatch):
    predict = search.pr
    seed = search.SETS["fit1d"][1]
    monkeypatch.setattr(predict, "OPTIONS", dict(set="fit1d", seed0=seed, stages=["fit"],
                                                samples=1, local_samples=1, refined=1))
    search_inputs = []
    predict_inputs = []

    def capture_search(model, data, **kwargs):
        search_inputs.append(data)
        raise RuntimeError("captured input")

    def capture_predict(network, q, intensity, sigma, *, qz, qz_minus, **kwargs):
        predict_inputs.append(predict.infer.Cut.prepare(q, intensity, sigma, qz, qz_minus))
        raise RuntimeError("captured input")

    monkeypatch.setattr(search.fit_engine, "fit", capture_search)
    monkeypatch.setattr(predict.refine, "predict", capture_predict)
    for i in (0, 2, 7):
        result = search.run(i, budget=200, set_name="fit1d", seed0=seed)
        assert result["case"] == predict.run(i)["case"]
        actual, expected = search_inputs.pop(), predict_inputs.pop()
        for field in ("q", "intensity", "sigma"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


def test_legacy_search_replays_archived_first_draws(monkeypatch):
    # Search was archived as a summary. The Predict per-cut file preserves the same input draws.
    archive = check_case_overlap.load_cases(search.HERE / "per_cut/posterior_recovery.json.gz")
    cases = {case["index"]: case for case in archive}
    summary = json.loads((search.HERE / "search_baseline_summary.json").read_text())
    assert len(cases) == summary["n"]

    def skip_fit(*args, **kwargs):
        raise RuntimeError("draw only")

    monkeypatch.setattr(search.fit_engine, "fit", skip_fit)
    for i in (0, 7, len(cases) - 1):
        case = search.run(i, budget=summary["budget"], set_name="legacy-first",
                          seed0=search.SETS["legacy-first"][1])["case"]
        expected = cases[i]
        assert case.keys() == expected.keys()
        for field in ("index", "composition", "level", "seed"):
            assert case[field] == expected[field]
        for field in ("peak_counts", "background"):
            assert case[field] == pytest.approx(expected[field], rel=1e-14)
        for pop, archived in zip(case["populations"], expected["populations"], strict=True):
            assert pop["shape"] == archived["shape"]
            for field in archived.keys() - {"shape"}:
                assert pop[field] == pytest.approx(archived[field], rel=1e-14)


def test_two_curve_search_writes_predict_stamp(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "HERE", tmp_path)
    search.main(["--n", "2", "--budget", "200", "--workers", "1", "--out", "smoke"])
    archive = json.loads((tmp_path / "results/smoke.json").read_text())
    summary = json.loads((tmp_path / "smoke_summary.json").read_text())
    assert archive["summary"] == summary
    assert summary["failed"] == 0
    assert (summary["set"], summary["seed0"], summary["stream_scheme"]) == (
        "fit1d", search.SETS["fit1d"][1], "seedsequence-v1")
    assert summary["code"] == dict(search.pr._code_state(), source_sha256=search.pr._source_digest())
    assert [r["case"] for r in archive["results"]] == [search.fr.draw(i, set_name="fit1d") for i in (0, 1)]


@pytest.mark.parametrize("arguments,seed", [([], search.SETS["legacy-third"][1]), (["--seed0", "0"], 0)])
def test_search_cli_uses_selected_seed(tmp_path, monkeypatch, arguments, seed):
    monkeypatch.setattr(search, "HERE", tmp_path)

    def skip_fit(*args, **kwargs):
        raise RuntimeError("draw only")

    monkeypatch.setattr(search.fit_engine, "fit", skip_fit)
    search.main(["--set", "legacy-third", "--n", "1", "--workers", "1", *arguments])
    archive = json.loads((tmp_path / "results/search_baseline_legacy-third.json").read_text())
    assert archive["summary"]["seed0"] == seed
    assert archive["summary"]["stream_scheme"] == "legacy-offset"
    assert archive["results"][0]["case"] == search.fr.draw(0, seed, set_name="legacy-third")
