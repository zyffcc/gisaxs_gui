import pytest

from tools.benchmarks.coverage_bootstrap import coverage_by_cut


def _row(index, radii, intervals):
    fit = [dict(R=radius, R_68=bounds, R_95=(0, 20))
           for radius, bounds in zip(sorted(radii), intervals)]
    return dict(case=dict(level="model", index=index, populations=[dict(R=r) for r in radii]),
                fit=dict(ok=True, populations=fit))


def test_population_coverage_bootstraps_the_whole_cut():
    # A single miss on one cut and two correlated hits on the other.
    rows = [_row(0, [1], [(2, 3)]), _row(1, [8, 4], [(3, 5), (7, 9)])]
    result = coverage_by_cut(rows, "model", "R", resamples=2048)
    assert (result["hits68"], result["populations"], result["cuts"]) == (2, 3, 2)
    assert result["coverage68"] == pytest.approx(2 / 3)
    # Both draws pick the same cut in half the replicates. Bootstrap proportions
    # are 0, 2/3, 1 with probabilities 1/4, 1/2, 1/4.
    assert result["bootstrap68"] == [0, 1]
    assert result["coverage95"] == 1 and result["bootstrap95"] == [1, 1]
    central = coverage_by_cut(rows, "model", "R", confidence=0.25, resamples=2048)
    assert central["bootstrap68"] == pytest.approx([2 / 3, 2 / 3])
    assert coverage_by_cut(rows[::-1], "model", "R", resamples=2048) == result


def test_coverage_conditions_on_eligible_populations_and_cuts():
    rows = [_row(0, [1, 2], [(0, 2), (1, 3)])]
    del rows[0]["fit"]["populations"][0]["R_68"]
    mismatched = _row(1, [1, 2], [(0, 2)])
    failed = _row(2, [1], [(0, 2)])
    failed["fit"]["ok"] = False
    measured = _row(3, [1], [(0, 2)])
    measured["case"]["level"] = "measured"
    rows.extend([mismatched, failed, measured])
    result = coverage_by_cut(rows, "model", "R", resamples=3)
    assert (result["hits68"], result["populations"], result["cuts"]) == (1, 1, 1)
    empty = coverage_by_cut(rows, "all", "D", resamples=3)
    assert empty["cuts"] == empty["populations"] == 0
    assert empty["coverage68"] is empty["bootstrap68"] is None
