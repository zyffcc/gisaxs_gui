"""Historical Table 3 evidence and failure modes, without running any fits."""
import json
from tools.benchmarks import table3_sources as table3


def test_shipped_archive_cells_and_missing_external_sources():
    rows = table3.verify(table3.HERE, {}, revision=14)
    assert len(rows) == 23
    # galaxi: the shipped galaxi_posterior.json is the revision-17 rerun, whose median R (4.9 nm) is the archived cell
    assert {key for key, row in rows.items() if row["issues"]} == {"all_pixel_q", "calibration_heldout", "au"}
    assert rows["extended.composition"]["counts"]["measured"] == {"correct": 179, "eligible": 300}
    assert rows["third.choices"]["counts"]["all_high"] == {"correct": 135, "eligible": 153}
    assert rows["third.auto"]["counts"]["radii"] == 424
    assert rows["third.auto"]["counts"]["true_radii"] == 450
    assert all(s["producing_commit"] == "7179b85" for s in rows["dwba"]["sources"])
    assert all(s["file"].endswith(".json.gz") for s in rows["dwba"]["sources"])


def test_missing_corrupt_and_changed_row_fail(tmp_path, capsys):
    def check(key):
        status = table3.main(["--revision", "14", "--bench", str(tmp_path), "--only", key])
        row = json.loads(capsys.readouterr().out)[key]
        assert status == 1 and row["issues"]
        return row

    check("third.error")
    path = tmp_path / "per_cut/posterior_newseed.json.gz"
    path.parent.mkdir()
    path.write_bytes(b"not a gzip file")
    check("third.error")
    f = tmp_path / "pyfai_comparison.json"
    f.write_text(json.dumps({"frames": [{"frame": f"frame-{i}", "intensity_same_estimator": {"p95_relative": 0.003}}
                                        for i in range(4)]}))
    assert "pinned" in check("radial")["issues"][0]


def test_explicit_reference_strict_provenance_and_rounding(tmp_path):
    f = tmp_path / "run.json"
    f.write_text(json.dumps({"coverage": {"as_shipped": {"identified": "7/9"},
                                         "cases": 9, "workers": 1, "seed0": 15},
                             "heldout": {"commit": "45a32e74"}}))
    external = {"heldout-calibration": f}
    key = "calibration_heldout"
    verified = table3.verify(tmp_path, external, {key: [9, 78]}, [key], strict=True, revision=14)[key]
    assert not verified["issues"]
    assert verified["counts"] == {"identified": 7, "frames": 9}
    assert "--seed 15" in verified["sources"][0]["command"]
    assert table3.verify(tmp_path, external, {key: [9, 100]}, [key], revision=14)[key]["issues"]
    assert table3.verify(tmp_path, external, only=[key], revision=14)[key]["issues"]
    assert table3.verify(table3.HERE, {}, only=["search"], strict=True, revision=14)["search"]["issues"]
    assert table3.tex_values(r"$3.1 \times 10^{-5}$ / $10^{-7}$") == [3.1e-5, 1e-7]
    assert table3.tex_values(r"$-$5\% / $-$(4--9)\%") == [-5, -4, -9]
