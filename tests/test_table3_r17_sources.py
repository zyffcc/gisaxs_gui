"""Publication sources and provenance for revision 17: the r17 archives of one frozen fitting tree."""
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


def first_archive(commit):
    n, seed = table3.SETS["fit1d"]
    parameters = {f"network/{level}": {"R": {"n": n // 2, "median_abs": .125}} for level in ("model", "measured")}
    summary = dict(code={"commit": commit, "fitting_code_modified": False, "fitting_tree": table3.FITTING_TREE17},
                   n=n, set="fit1d", seed0=seed, parameters=parameters,
                   settings=dict(stream_scheme="seedsequence-v1", truth="independent", auto=True, cut="born",
                                 substrate="si", scattering="born_qz"))
    return dict(summary=summary, results=[
        dict(case=dict(index=i, level="model" if i % 2 == 0 else "measured", populations=[{"R": 8}]),
             network=dict(ok=True, populations=[{"R": 9}])) for i in range(n)])


def test_revision_17_has_the_rows_of_revision_15_on_r17_sources():
    assert list(table3.ROWS17) == list(table3.ROWS15)
    names = [item[0] for items, _, _ in table3.ROWS17.values() for item in items]
    assert not any("r15" in name for name in names)
    assert "per_cut/posterior_r17_first.json.gz" in names and "search_baseline_r17_summary.json" in names
    assert table3.REVISIONS[17] is table3.ROWS17


@pytest.mark.parametrize("commit", table3.FROZEN17)
def test_r17_archive_accepts_either_commit_of_the_frozen_tree(tmp_path, commit):
    write_json(tmp_path / "per_cut/posterior_r17_first.json.gz", first_archive(commit))
    row = table3.verify(tmp_path, {}, {"network.error": [12, 12]}, ["network.error"], revision=17)["network.error"]
    assert not row["issues"]
    assert row["sources"][0]["producing_commit"] == commit


@pytest.mark.parametrize("change", [{"commit": table3.FROZEN}, {"fitting_tree": table3.FITTING_TREE},
                                    {"fitting_code_modified": True}])
def test_r17_archive_rejects_another_tree_or_commit(tmp_path, change):
    raw = first_archive(table3.FROZEN17[0])
    raw["summary"]["code"].update(change)
    write_json(tmp_path / "per_cut/posterior_r17_first.json.gz", raw)
    row = table3.verify(tmp_path, {}, only=["network.error"], revision=17)["network.error"]
    assert row["issues"] and "printed" not in row
