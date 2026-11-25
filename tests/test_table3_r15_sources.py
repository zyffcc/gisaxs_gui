"""Publication sources, provenance, archive selection and manuscript input for revision 15."""
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


def first_archive():
    n, seed = table3.SETS["fit1d"]
    parameters = {f"network/{level}": {"R": {"n": n // 2, "median_abs": .125}} for level in ("model", "measured")}
    summary = dict(code={"commit": table3.FROZEN, "fitting_code_modified": False, "fitting_tree": table3.FITTING_TREE},
                   n=n, set="fit1d", seed0=seed, parameters=parameters,
                   settings=dict(stream_scheme="seedsequence-v1", truth="independent", auto=True, cut="born",
                                 substrate="si", scattering="born_qz"))
    return dict(summary=summary, results=[
        dict(case=dict(index=i, level="model" if i % 2 == 0 else "measured", populations=[{"R": 8}]),
             network=dict(ok=True, populations=[{"R": 9}])) for i in range(n)])


def test_disk_archive_fallback_and_precedence(tmp_path):
    group = "network.error"
    raw = first_archive()
    write_json(tmp_path / "results/posterior_r15_first.json", raw)

    def verify():
        return table3.verify(tmp_path, {}, {group: [12, 12]}, [group])[group]

    row = verify()
    assert not row["issues"]
    source, = row["sources"]
    assert source["file"].endswith("/results/posterior_r15_first.json")
    assert source["producing_commit"] == table3.FROZEN
    assert source["command"].endswith("--workers 5 --set fit1d --n 200 --auto")
    packed = tmp_path / "per_cut/posterior_r15_first.json.gz"
    write_json(packed, raw)
    assert verify()["sources"][0]["file"].endswith("/per_cut/posterior_r15_first.json.gz")
    packed.write_bytes(b"an interrupted archive")
    assert verify()["issues"]


@pytest.mark.parametrize("change", [{"commit": "71abc123"}, {"fitting_code_modified": True},
                                    {"fitting_code_modified": None}, {"fitting_tree": "abcd"}])
def test_archive_stamp_must_match(tmp_path, change):
    raw = first_archive()
    raw["summary"]["code"].update(change)
    write_json(tmp_path / "results/posterior_r15_first.json", raw)
    row = table3.verify(tmp_path, {}, only=["first.error"])["first.error"]
    assert row["issues"] and "printed" not in row


def test_missing_r15_inputs_report_each_file(tmp_path):
    rows = table3.verify(tmp_path, {})
    assert list(rows) == (
        "all_pixel_q radial agbh calibration calibration_heldout independent_model bornagain "
        "first.error first.coverage first.composition first.high third.error third.coverage "
        "third.composition third.high third.auto extended.error extended.composition network.error "
        "search dwba dwba4 galaxi height au"
    ).split()
    assert all(row["issues"] for row in rows.values())
    assert "results/posterior_r15_fresh.json" in rows["third.auto"]["issues"][0]
    assert len(rows["dwba"]["issues"]) == 4


def test_four_path_row_requires_branch_scored_pair_report(tmp_path):
    pairs = {"runs": {label: {"overall": {"choose": {"composition_right": correct,
              "parameters": {"R": {"median_abs_relative": error}}}}}
              for label, correct, error in (("si_l2", .5, .103), ("si_l4", .605, .041))}}
    path = write_json(tmp_path / "posterior_r15_dwba4_pairs.json", pairs)
    expected = {"dwba4": [60.5, 4.1, 50, 10.3]}
    assert table3.verify(tmp_path, {}, expected, only=expected)["dwba4"]["issues"]
    write_json(path, dict(pairs, composition_method=table3.METHOD))
    row = table3.verify(tmp_path, {}, expected, only=expected)["dwba4"]
    assert not row["issues"]
    assert "--by-cut" in row["sources"][0]["command"]


def test_default_cli_compares_unpinned_cells_with_manuscript(tmp_path, capsys):
    numbers = tmp_path / "numbers.tex"
    numbers.write_text(r"\newcommand{\radii}{37\% / 65\% / 52\% / 78\%}", encoding="utf-8")
    table = tmp_path / "table.tex"
    cells = {"search": r"\radii", "first.high": "11/20, 5/7", "extended.error": "1.0--2.0, 13--14"}
    table.write_text("\n".join(f"{key} & {cells.get(key, '0')} & source " + r"\\"
                               for key in table3.ROWS15), encoding="utf-8")
    values = table3.manuscript_rows(table, numbers)
    assert values["first.high"] == [11, 20, 5, 7]
    assert values["extended.error"] == [1, 2, 13, 14]
    assert all(pin is None for _, _, pin in table3.ROWS15.values())
    summary = dict(code=dict(commit="ba123456", fitting_code_modified=False), n=200, set="fit1d",
                   seed0=table3.SETS["fit1d"][1], stream_scheme="seedsequence-v1", parameters={
                       "model": {"R": {"median_abs": .37, "p90_abs": .65}},
                       "measured": {"R": {"median_abs": .52, "p90_abs": .78}}})
    write_json(tmp_path / "search_baseline_r15_summary.json", summary)
    args = ["--revision", "15", "--bench", str(tmp_path), "--table", str(table), "--numbers", str(numbers),
            "--only", "search"]
    assert table3.main(args) == 0
    row = json.loads(capsys.readouterr().out)["search"]
    assert row["sources"][0]["producing_commit"] == "ba123456"
    numbers.write_text(r"\newcommand{\radii}{99/65/52/78}", encoding="utf-8")
    assert table3.main(args) == 1
    assert "supplied Table 3" in json.loads(capsys.readouterr().out)["search"]["issues"][0]


def test_merged_commit_is_read_from_each_rerun(tmp_path):
    code = dict(commit="fab43210", modified=False)
    galaxi = dict(code=code, fits=[{"variant": "born1", "source": "bornagain_1",
                                   "best": {"values": {"R_1": {"best": 5.25}}}}],
                  truth={"R_number_mean": 6})
    write_json(tmp_path / "galaxi_posterior.json", galaxi)
    biases = {"a_off_D14": .001, "a_off_D24": 0, "a_off_D12": 0, "b_avg_10_Si": -.1,
              "b_avg_10_Au": -.2, "b_avg_30_Au": -.3, "b_avg_40_Au": -.4}
    write_json(tmp_path / "vertical_height_bias.json", dict(code=code, cases=[
        {"name": name, "summary": {"full": {"h": {"noise_free_bias": bias}}}} for name, bias in biases.items()]))
    au = write_json(tmp_path / "fit.json", dict(code=code, cycle=7, values={"R_1": {
        "median": 25, "low95": 24, "high95": 27}}))
    sem = write_json(tmp_path / "sem.json", dict(citation="Synthetic example", diameter_mean_nm=42, diameter_sd_nm=4))
    expected = dict(galaxi=[5.2, 6], height=[.1, -10, -20, -40], au=[21, 2, 25, 95, 24, 27])
    rows = table3.verify(tmp_path, {"au-fit": au, "sem-reference": sem}, expected, only=expected, strict=True)
    for row in rows.values():
        assert not row["issues"]
        assert row["sources"][0]["producing_commit"] == code["commit"]
