"""Publication sources and provenance for revision 18: the r18 archives, run at one freeze with root seeds of their
own."""
import gzip
import json

import pytest

from tools.benchmarks import table3_sources as table3


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)
    return path


def first_archive(commit=table3.FROZEN18[0], seed=table3.SEEDS18["fit1d"]):
    n = table3.SETS["fit1d"][0]
    parameters = {f"network/{level}": {"R": {"n": n // 2, "median_abs": .125}} for level in ("model", "measured")}
    summary = dict(code={"commit": commit, "fitting_code_modified": False, "fitting_tree": table3.FITTING_TREE17},
                   n=n, set="fit1d", seed0=seed, parameters=parameters,
                   settings=dict(stream_scheme="seedsequence-v1", truth="independent", auto=True, cut="born",
                                 substrate="si", scattering="born_qz"))
    return dict(summary=summary, results=[
        dict(case=dict(index=i, level="model" if i % 2 == 0 else "measured", populations=[{"R": 8}]),
             network=dict(ok=True, populations=[{"R": 9}])) for i in range(n)])


def test_revision_18_has_the_rows_of_revision_15_on_r18_sources_with_their_own_seeds():
    assert list(table3.ROWS18) == list(table3.ROWS15)
    items = [item for items, _, _ in table3.ROWS18.values() for item in items]
    assert not any("r15" in name or "r17" in name for name, _, _ in items)
    assert table3.REVISIONS[18] is table3.ROWS18
    runs = [(name, command) for name, _, command in items if name.startswith("per_cut/posterior_r18_")]
    assert runs and all(f"--seed0 {table3.SEEDS18[command.split('--set ')[1].split()[0]]}" in command
                        for _, command in runs)
    assert not set(table3.SEEDS18.values()) & {seed for _, seed in table3.SETS.values()}


def test_r18_archive_at_the_freeze_with_the_new_seed_passes(tmp_path):
    write_json(tmp_path / "per_cut/posterior_r18_first.json.gz", first_archive())
    row = table3.verify(tmp_path, {}, {"network.error": [12, 12]}, ["network.error"], revision=18)["network.error"]
    assert not row["issues"]
    assert row["sources"][0]["producing_commit"] == table3.FROZEN18[0]


@pytest.mark.parametrize("change", [dict(commit=table3.FROZEN17[1]), dict(seed=table3.SETS["fit1d"][1])])
def test_r18_archive_rejects_another_commit_or_the_development_seed(tmp_path, change):
    write_json(tmp_path / "per_cut/posterior_r18_first.json.gz", first_archive(**change))
    row = table3.verify(tmp_path, {}, only=["network.error"], revision=18)["network.error"]
    assert row["issues"] and "printed" not in row
