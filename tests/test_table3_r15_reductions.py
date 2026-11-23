"""Revision-15 Table 3 cells from small synthetic archives with independently counted answers."""
import json

import pytest

from tools.benchmarks import posterior_recovery as recovery
from tools.benchmarks import table3_metrics_r15 as metrics
from tools.benchmarks.composition_scores import summarize_compositions


def archive(tmp_path, name, payload):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def fitted(radius, error_pct, hit68, hit95):
    # The lower and upper interval endpoints both occur as exact hits.
    inner = [radius, radius + 1] if hit68 else [radius - 2, radius - 1]
    high95 = radius + 2 if hit68 else (radius if hit95 else radius - .5)
    return {"R": radius * (1 + error_pct / 100),
            "R_68": inner, "R_95": [radius - 3, high95]}


def choice(correct, weight, branch):
    # Both candidates have the same displayed family. Their radius-ordered indices differ.
    winner = 1 if correct else 0
    weights = {str(winner): weight, str(1 - winner): 1 - weight}
    return dict(ok=True, compared_by=branch, composition_weights=weights,
                retained_modes=[{"composition_index": int(k), "probability": v} for k, v in weights.items()],
                families=["sphere"], probability=1, alternatives=[])


@pytest.fixture
def posterior(tmp_path):
    failed = {"ok": False}
    inputs = [
        # level, true radii in draw order, fit in radius order, composition choice
        ("model", [20, 10], [fitted(10, 1.234, True, True), fitted(20, 3.456, False, True)],
         choice(True, .99, "evidence")),
        ("model", [8], [fitted(8, 30, False, False)], choice(False, .999, "evidence")),
        ("measured", [10], [fitted(10, 0, True, True)], choice(True, .99, "bic")),
        ("measured", [25], [fitted(25, 7.2, False, True)], choice(True, .989, "bic")),
        ("model", [8], None, failed),
        ("measured", [9, 7], [fitted(7, 90, False, False)], choice(False, .5, "bic")),
    ]
    auto = [
        dict(choice(False, .8, "bic"), populations=[fitted(10, 90, False, False)]),
        dict(choice(True, .9, "evidence"), populations=[fitted(8, 7.51, True, True)]),
        dict(choice(True, .8, "bic"), populations=[fitted(10, 12.73, False, True)]),
        failed, failed,
        dict(choice(False, .8, "bic"), populations=[fitted(7, 90, False, False)]),
    ]
    rows = []
    for i, (level, radii, pops, choose) in enumerate(inputs):
        fit = network = failed
        if pops is not None:
            fit = dict(ok=True, populations=pops)
            predicted = [{"R": r * (1 + (10 * i + j + 3) / 100)} for j, r in enumerate(sorted(radii)[:len(pops)])]
            network = dict(ok=True, populations=predicted)
        rows.append(dict(case=dict(index=i, level=level, populations=[{"R": r} for r in radii]),
                         fit=fit, network=network, choose=choose, auto=auto[i],
                         truth_families=["sphere"], truth_composition_index=1))
    levels = ("model", "measured", "all")
    summary = dict(n=len(rows), parameters={
        f"{stage}/{level}": recovery._errors(rows, stage, level)
        for stage in ("fit", "network", "auto") for level in levels})
    for stage, name in (("choose", "composition"), ("auto", "composition_auto")):
        summary[name] = {level: summarize_compositions(
            [r for r in rows if level == "all" or r["case"]["level"] == level], stage) for level in levels}
    return archive(tmp_path, "posterior", dict(summary=summary, results=rows))


def test_recovery_cells_and_denominators(posterior):
    expected = {
        "first.error": ["3.5", "25", "3.6", "6.5"],
        "first.coverage": ["33", "67", "50", "100"],
        "first.composition": ["50", "67"],
        "first.high": ["1/2", "1/1"],
        "third.error": ["3.5", "21"],
        "third.coverage": ["40"],
        "third.composition": ["60"],
        "third.high": ["1/2", "1/1"],
        "third.auto": ["50", "10.1", "50"],
        "extended.error": ["3.5--3.6", "25--6"],
        "extended.composition": ["50", "67"],
        "network.error": ["4", "28"],
    }
    rows = {key: metrics.reduce_row(key, [posterior]) for key in expected}
    assert {key: row["printed"] for key, row in rows.items()} == expected
    assert rows["first.coverage"]["counts"]["model"] == {"n": 3, "total": 4, "hits68": 1, "hits95": 2}
    assert rows["third.auto"]["counts"]["radii"] == 2
    assert rows["third.auto"]["counts"]["true_radii"] == 8


def test_unrounded_summary_disagreements(posterior):
    summary = posterior["summary"]
    summary["parameters"]["fit/model"]["R"]["median_abs"] += .000001
    with pytest.raises(ValueError, match="median_abs"):
        metrics.reduce_row("first.error", [posterior])
    summary["composition"]["all"]["exact"] = .61
    with pytest.raises(ValueError, match="exact"):
        metrics.reduce_row("third.composition", [posterior])
    summary["composition"]["all"]["reliability_by_branch"]["bic"]["bins"][-1]["n"] += 1
    with pytest.raises(ValueError, match="bic"):
        metrics.reduce_row("first.high", [posterior])
    del posterior["results"][0]["choose"]["composition_weights"]
    with pytest.raises(ValueError):
        metrics.reduce_row("third.high", [posterior])


def test_search_and_four_path_cells(tmp_path):
    search = archive(tmp_path, "search", {"n": 4, "parameters": {
        "model": {"R": {"median_abs": .0351, "p90_abs": .416}},
        "measured": {"R": {"median_abs": .083, "p90_abs": 1.21}}}})
    assert metrics.reduce_row("search", [search])["printed"] == ["4", "42", "8", "121"]
    four_path = {"runs": {label: {"overall": {"choose": {
        "composition_right": rate, "parameters": {"R": {"median_abs_relative": error}}}}}
        for label, rate, error in [("si_l2", .29, .2423), ("si_l4", .665, .0316)]}}
    pairs = archive(tmp_path, "dwba4", four_path)
    assert metrics.reduce_row("dwba4", [pairs])["printed"] == ["66.5", "3.2", "29", "24.2"]


def test_paired_sphere_bin(tmp_path):
    qz = recovery.fr.GEOM.qz_vacuum
    pairs = dict(n_curves=3, runs={})
    runs = []
    for label, error in (("si_tp", .152), ("si_born", .057), ("si_eia", .359)):
        records = []
        for i, (qz_r, shape) in enumerate(((3., "sphere"), (3., "vc"), (4., "sphere"))):
            radius = qz_r / qz
            truth = dict(R=radius, shape=shape)
            found = dict(R=radius * (1 + (error if i == 0 else .91)))
            records.append(dict(case=dict(index=i, populations=[truth]), fit=dict(ok=True, populations=[found])))
        runs.append(archive(tmp_path, label, dict(summary={"n": 3}, results=records)))
        pairs["runs"][label] = {"sphere q+R 2.3-3.5": {"R": {"n": 1, "median_abs": error}}}
    pairs = archive(tmp_path, "pairs", pairs)
    result = metrics.reduce_row("dwba", [pairs, *runs])
    assert result["printed"] == ["15", "6", "36"]
    assert result["counts"]["eligible_spheres"] == [1, 1, 1]
    pairs["runs"]["si_eia"]["sphere q+R 2.3-3.5"]["R"]["median_abs"] += .00001
    with pytest.raises(ValueError, match="median_abs"):
        metrics.reduce_row("dwba", [pairs, *runs])
