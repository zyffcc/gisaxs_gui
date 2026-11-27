"""Frozen input replay and independence of the simulated cuts used for new runs."""
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from tools.benchmarks import check_case_overlap as audit
from tools.benchmarks import heldout_cases, recovery_cases
from tools.benchmarks.case_streams import SETS, case_streams

BENCH = Path(__file__).resolve().parents[1] / "tools/benchmarks"


@pytest.mark.parametrize("name,archive,generator", [
    ("legacy-first", "posterior_recovery", recovery_cases),
    ("legacy-extended", "posterior_extended", heldout_cases),
    ("legacy-third", "posterior_newseed", recovery_cases),
])
def test_legacy_draws_match_shipped_cases(name, archive, generator):
    cases = audit.load_cases(BENCH / "per_cut" / f"{archive}.json.gz")
    n, _ = SETS[name]
    assert len(cases) == n
    for case in cases:
        actual = generator.draw(case["index"], set_name=name)
        assert actual.keys() == case.keys()
        for field in case.keys() - {"populations", "peak_counts", "background"}:
            assert actual[field] == case[field]
        # Different libm implementations may round the final exp or power differently.
        for field in ("peak_counts", "background"):
            assert actual[field] == pytest.approx(case[field], rel=1e-14)
        for p, expected in zip(actual["populations"], case["populations"], strict=True):
            assert p["shape"] == expected["shape"]
            for field in expected.keys() - {"shape"}:
                assert p[field] == pytest.approx(expected[field], rel=1e-14)


def test_fresh_streams_and_levels():
    n, seed = SETS["fresh"]
    fresh = [recovery_cases.draw(i, seed, set_name="fresh") for i in range(n)]
    # Anchor the named set for subsequent reruns, independently of worker order.
    assert fresh[0]["seed"] == 132446915519091051651014313264147083700
    assert fresh[0]["populations"][0]["R"] == pytest.approx(3.779788692130237, rel=1e-14)
    assert Counter(c["level"] for c in fresh) == {"model": 150, "measured": 150}
    assert set(Counter(c["composition"] for c in fresh).values()) == {75}
    assert len({c["seed"] for c in fresh}) == n
    for i in (34, 2, 0):
        assert recovery_cases.draw(i, seed, set_name="fresh") == fresh[i]
        params, noise, _ = case_streams("fresh", seed, i)
        assert params.random() != np.random.default_rng(noise).random()
        for generate in (recovery_cases.draw, heldout_cases.draw):
            other = generate(i, seed)
            assert other["seed"] != fresh[i]["seed"]
            assert other["populations"][0]["R"] != fresh[i]["populations"][0]["R"]


def test_overlap_with_all_archived_runs():
    archives = audit.archived_sets(BENCH / "per_cut")
    assert any("reconstructed" in name and len(cases) == 1200 for name, cases in archives.items())
    n, seed = SETS["fresh"]
    fresh = [recovery_cases.draw(i, seed, set_name="fresh") for i in range(n)]
    _, pairs = audit.compare(archives, fresh)
    historic = {}
    for (left, right), fields in pairs.items():
        if right == "new" and "posterior_r15_fresh" in left:  # the fresh set itself, archived after its run
            assert all(len(matches) == n for matches in fields.values())
        elif right == "new":
            assert not any(fields.values())
        else:
            for name in ("recovery", "extended"):
                first = f"posterior_{name}.json.gz"
                third = "posterior_newseed.json.gz"
                if (first in left and third in right) or (third in left and first in right):
                    historic[name] = fields
    assert {k: len(v["noise_seed"]) for k, v in historic.items()} == {"recovery": 194, "extended": 221}
    for fields in historic.values():
        assert fields["first_population_draw"] == fields["noise_seed"]


def test_audit_fails_for_reused_cut_even_with_new_composition(tmp_path, capsys):
    original = recovery_cases.draw(0)
    reused = dict(original, composition="vc", populations=[dict(original["populations"][0], shape="vc")])
    archive = tmp_path / "posterior_toy.json.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as out:
        json.dump({"results": [{"case": original}]}, out)
    incoming = tmp_path / "new.json"
    incoming.write_text(json.dumps({"results": [{"case": reused}]}))
    assert audit.main(["--archive-dir", str(tmp_path), "--new-archive", str(incoming), "--indices"]) == 1
    report = capsys.readouterr().out
    assert "first_population_draw: [(0, 0)]" in report
    assert "noise_seed: [(0, 0)]" in report


def test_a_draw_repeated_on_another_computer_is_the_same_draw():
    """exp of another libm moves R and D by a few units in the last place (Linux CI); a draw differing in the eighth
    digit is another draw."""
    original = recovery_cases.draw(0, set_name="fresh")
    first = original["populations"][0]
    repeated = dict(original, seed=original["seed"] + 1, populations=[dict(
        first, R=float(np.nextafter(np.nextafter(first["R"], 99.0), 99.0)), D=float(np.nextafter(first["D"], 0.0)))])
    other = dict(original, seed=original["seed"] + 2, populations=[dict(first, R=first["R"] * (1 + 1e-7))])
    for incoming, matched in ((repeated, True), (other, False)):
        _, pairs = audit.compare({"archive": [original]}, [incoming])
        assert bool(pairs["archive", "new"]["first_population_draw"]) is matched
        assert not pairs["archive", "new"]["noise_seed"]
