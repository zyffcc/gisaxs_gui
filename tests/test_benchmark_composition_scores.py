"""Benchmarks score the program's ordered, structure-aware compositions before display filtering."""
import json
from types import SimpleNamespace

import pytest

from tools.benchmarks.composition_scores import archive_weights, composition_score, summarize_compositions
from tools.benchmarks import posterior_recovery as recovery
from tools.benchmarks.posterior_cut_classes import _group
from tools.benchmarks.posterior_pairs import group_stats


def _row(branch):
    case = dict(level="model", populations=[dict(R=3., D=9., shape="vc"), dict(R=2., D=6., shape="sphere")])
    truth = recovery._composition(case)
    reverse = recovery.E.composition_index([("vertical_cylinder", True), ("sphere", True)])
    unstructured = recovery.E.composition_index([("sphere", False), ("vertical_cylinder", True)])
    # The 0.004 mode is below both Predict's default display cutoff and the old archive's 0.05 cutoff.
    modes = [(reverse, .499, .499), (truth, .497, .501), (truth, .004, .501)]
    field = "composition_bic_weight" if branch == "bic" else "composition_probability"
    solutions = [SimpleNamespace(composition=index, probability=p,
                                 extras={"compared_by": branch, "diagnostics": {"compared_by": branch, field: total}})
                 for index, p, total in modes]
    candidates = [reverse, truth, unstructured]
    record = archive_weights(solutions, candidates)
    with pytest.raises(ValueError, match="Incomplete retained modes"):
        archive_weights(solutions[:2], candidates)
    families = recovery._families(truth)
    stage = dict(ok=True, **record, composition_index=reverse, probability=.499, families=families,
                 populations=[], alternatives=[dict(populations=[])], candidates=candidates, seconds=.1)
    return json.loads(json.dumps(dict(case=case, truth_composition_index=truth, truth_families=families, choose=stage)))


@pytest.mark.parametrize("branch", ["evidence", "bic"])
def test_cut_and_summary_use_all_modes_of_ordered_composition(branch):
    row = _row(branch)
    score = composition_score(row, "choose")
    assert score.correct and score.winner == row["truth_composition_index"] and score.weight == .501
    assert score.branch == branch and score.winner != row["choose"]["composition_index"]
    assert len(row["choose"]["composition_weights"]) == 3
    no_modes = row["choose"]["candidates"][2]
    assert row["choose"]["composition_weights"][str(no_modes)] == 0
    assert no_modes not in [m["composition_index"] for m in row["choose"]["retained_modes"]]
    results = recovery.summary([row])["composition"]["all"]
    assert results["n_correct"] == _group([row], .5)["choose"]["composition"]["n_correct"] == 1
    assert group_stats([dict(record=row, first=None, listed=None, chi2=None)])["exact"] == 1
    assert results["reliability_by_branch"][branch]["bins"][1]["n_correct"] == 1
    # Same unordered families with opposite radius order, or different paracrystal status.
    for index in row["choose"]["candidates"]:
        row["truth_composition_index"] = index
        assert composition_score(row, "choose").correct == (index == score.winner)


def test_reliability_preserves_failed_and_unscorable_cut_counts():
    evidence, bic = _row("evidence"), _row("bic")
    bic["truth_composition_index"] = bic["choose"]["composition_index"]
    legacy = dict(choose=dict(ok=True, probability=1., alternatives=[], flags=[]))
    counts = summarize_compositions([evidence, legacy, bic, dict(choose=dict(ok=False))], "choose")
    assert (counts["n"], counts["n_scored"], counts["n_correct"], counts["n_failed"], counts["n_missing_weights"]) == (
        4, 2, 1, 1, 1)
    for branch, hits in (("bic", 0), ("evidence", 1)):
        r = counts["reliability_by_branch"][branch]
        assert r["n"] == 1 and r["n_correct"] == hits
        assert sum(b["n"] for b in r["bins"]) == 1
        assert r["brier"] == pytest.approx((.501 - hits) ** 2)
    evidence["choose"]["composition_weights"].clear()
    with pytest.raises(ValueError, match="weights"):
        composition_score(evidence, "choose")
