"""Verify the revision-14 Table 3 row by row. See --help and the benchmark README."""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

if __package__:
    from .table3_metrics import reduce_row
else:
    from table3_metrics import reduce_row

HERE = Path(__file__).resolve().parent
BENCH = "tools/benchmarks/"
NETWORK = "src/gimap/features/fitting/domain/posterior/data/flow_v1"


def run(name, commit, options):
    output = f"posterior_{name}" if name.startswith("newseed") else f"posterior_{name}_new"
    return (f"per_cut/posterior_{name}.json.gz", commit,
            f"python {BENCH}posterior_recovery.py --network {NETWORK} --out {output} {options}")


FIRST = run("recovery", "2eb0243", "--n 200 --workers 10")
THIRD = run("newseed", "6bb1a18", "--n 300 --seed0 20261122 --auto --workers 8")
AUTO = run("newseed_auto2", "f73d9c8", "--n 300 --seed0 20261122 --stages auto --auto-level 2 --workers 8")
EXT = run("extended", "2eb0243", "--set extended --n 600 --seed0 20261201 --workers 10")
DWBA = [run("dwba_" + label, "7179b85", f"--n 200 --workers 10 --scattering dwba --substrate si --cut {cut}")
        for label, cut in (("tp", "two_path"), ("born", "born"), ("eia", "eia"))]


def source(name, script, options="", commit=None):
    return name, commit, f"python {BENCH}{script}.py {options}".rstrip()


RADIUS = "results[].case.populations sorted by R; matched-count fit.populations[].R (values.best)"
INTERVALS = RADIUS + ", R_68, R_95; inclusive endpoints"
HISTORIC = ("results[].truth_families; choose.families/probability and choose.alternatives[].families/probability; "
            "revision-14 sum by unordered families over displayed modes only")

# Each entry is one printed row. Pinned cells are reproducible from the distributed files;
# rows requiring third-party inputs must be checked against the separately supplied table.
# In ranges and paired statistics the order is the printed reading order.
ROWS = {
    "all_pixel_q": ([("@all-pixel-q", None, "python figures/scripts/q_difference_pyfai.py")],
                    "max(frames[].max_abs_relative_all_pixels); upper bound to one significant digit", None),
    "radial": ([source("pyfai_comparison.json", "pyfai_comparison", "--json <output.json>")],
               "max(frames[].intensity_same_estimator.p95_relative); upper power of ten", [1e-7, 95]),
    "agbh": ([source("calibration_check.json", "calibration_check", "--part all --cases 300 --workers 3")],
             "max abs(real_frame.compare_gimap_convention.*.difference_over_combined_sigma_jackknife)", [1.8]),
    "calibration": ([source("per_cut/calibration_check_frames.json.gz", "calibration_check",
                            "--part all --cases 300 --workers 3")],
                    "main.frames[].as_shipped.identified; frame count and percent identified", [300, 100]),
    "calibration_heldout": ([("@heldout-calibration", "45a32e74",
                             f"python {BENCH}calibration_check.py --part coverage --cases <cases> --seed <seed0>")],
                            "coverage.as_shipped.identified (found/total); heldout.commit", None),
    "independent_model": ([source("cut_model_checks.json", "cut_model_checks", "--out <output.json>")],
                          "checks[reference=truth, sigma=0.2].max_rel, maximum", [2.7e-5]),
    "bornagain": ([source("cut_model_bornagain.json", "cut_model_bornagain")],
                  "checks[distributed sizes, sigma=0.2].max_rel, sphere then upright cylinder", [3.9e-4, 2.8e-4]),
    "first.error": ([FIRST], RADIUS + "; median and P90 percent absolute errors by level", [0.54, 10, 0.69, 14]),
    "first.coverage": ([FIRST], INTERVALS + "; percentages by level", [70, 87, 65, 83]),
    "first.composition": ([FIRST], HISTORIC + "; correct percent by level", [87, 63]),
    "first.high": ([FIRST], HISTORIC + "; percent correct and n at weight >= 0.99", [87, 102]),
    "third.error": ([THIRD], RADIUS + "; median and P90 percent absolute errors", [0.7, 15]),
    "third.coverage": ([THIRD], INTERVALS + "; percent covered at 68%", [72]),
    "third.choices": ([THIRD], HISTORIC + "; overall percent, percent correct and n at weight >= 0.99", [77, 88, 153]),
    "third.auto": ([AUTO], "results[].auto.families, truth_families (leading unordered families); "
                   "auto.populations[].R (values.best), R_68 against radius-ordered case.populations[].R", [75, 0.8, 67]),
    "extended.error": ([EXT], RADIUS + "; medians by level then P90s by level", [1.5, 2.1, 28, 45]),
    "extended.composition": ([EXT], HISTORIC + "; correct percent by level", [69, 60]),
    "network.error": ([FIRST], "results[].network.populations[].R (posterior median of network samples), "
                      "case.populations[].R; median absolute relative errors by level", [7, 9]),
    "search": ([source("search_baseline_summary.json", "search_baseline", "--n 200 --workers 10")],
               "parameters.{model,measured}.R.{median_abs,p90_abs}", [3, 81, 7, 136]),
    "dwba": (DWBA, RADIUS + "; median error over spheres with 2.3 <= q_z R < 3.5, levels 3/2/1", [7, 11, 20]),
    "galaxi": ([source("galaxi_posterior.json", "galaxi_posterior", commit="4ea7fb7")],
               "median(fits[BornAgain, born1].best.values.R_1.best); truth.R_number_mean", [4.9, 6.2]),
    "height": ([source("vertical_height_bias.json", "vertical_height_bias")],
               "cases[name].summary.full.h.noise_free_bias; max absolute off, Si 10%, "
               "least/most negative of Au 10/30/40%. Full angular window, historical height model.", [0.1, -3, -13, -60]),
    "au": ([("@au-fit", "e14282d", "python series.py fit <cycle>"),
            ("@sem-reference", None, "divide published diameter mean and SD by two")],
           "SEM diameter_mean_nm, diameter_sd_nm; series fit values.R_1.{median,low95,high95}, code.commit", None),
}


@lru_cache(maxsize=12)
def read_json(path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def evidence(item, bench, external):
    name, commit, command = item
    path = external.get(name[1:]) if name.startswith("@") else bench / name
    if path is None:
        raise ValueError("missing --" + name[1:])
    d = json.load(sys.stdin) if name == "@sem-reference" and path == Path("-") else read_json(path)
    record = dict(file=path.name if name.startswith("@") else BENCH + name,
                  producing_commit=commit, command=command)
    if "summary" in d:
        code = d["summary"].get("code", {})
        if code and (code["commit"] != commit or code.get("fitting_code_modified")):
            raise ValueError(f"{name}: archive code stamp disagrees with recorded producer")
        if d["summary"]["n"] != int(command.split("--n ", 1)[1].split()[0]):
            raise ValueError(f"{name}: archive cut count disagrees with the producing command")
        record["commit_source"] = "summary.code.commit" if code else "per_cut/README.md"
    elif name == "galaxi_posterior.json":
        record["commit_source"] = "git log -1 420b0e29"
    elif name == "@heldout-calibration":
        coverage = d["coverage"]
        if not d["heldout"]["commit"].startswith(commit):
            raise ValueError("held-out calibration has a different producing commit")
        record.update(producing_commit=d["heldout"]["commit"], commit_source="heldout.commit",
                      command=f"python {BENCH}calibration_check.py --part coverage --cases {coverage['cases']} "
                              f"--seed {coverage['seed0']} --workers {coverage['workers']} --json <output.json>")
    elif name == "@au-fit":
        if d["code"].get("modified") or not d["code"]["commit"].startswith(commit):
            raise ValueError("measured fit has local changes or a different producing commit")
        record.update(producing_commit=d["code"]["commit"], commit_source="code.commit",
                      command=f"python series.py fit {d['cycle']}")
    elif name == "@sem-reference":
        record.update(reference=d["citation"], commit_source="published measurement")
    if not record["producing_commit"] and name != "@sem-reference":
        record["commit_source"] = "not recorded"
    return d, record


NUMBER = r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def tex_values(cell):
    text = cell.replace("$", "").replace(r"\,", "").replace("~", " ")
    text = re.sub(r"(?:(" + NUMBER + r")\s*\\times\s*)?10\^\{(-?\d+)\}",
                  lambda m: (m[1] or "1") + "e" + m[2], text)
    text = re.sub(r"-\s*\(([\d.]+)--([\d.]+)\)", r"-\1 -\2", text).replace("--", " ")
    return [float(x) for x in re.findall("-?" + NUMBER, text)]


def manuscript_rows(table, numbers):
    macros = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$",
                             numbers.read_text(encoding="utf-8"), re.M))
    rows = []
    for line in table.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("%") or line.count("&") != 2 or not line.rstrip().endswith(r"\\"):
            continue
        label, cell, _ = line.split("&")
        if label.strip() == "Test":
            continue
        expand = lambda m: macros.get(m[1], m[0])
        rows.append((re.sub(r"\\([A-Za-z]+)", expand, label), re.sub(r"\\([A-Za-z]+)", expand, cell)))
    if len(rows) != len(ROWS):
        raise ValueError(f"expected {len(ROWS)} data rows in the revision-14 Table 3, found {len(rows)}")
    expected = {}
    for key, (label, cell) in zip(ROWS, rows):
        printed = tex_values(cell)
        if key in ("calibration", "calibration_heldout", "au"):
            printed = tex_values(label) + printed
        expected[key] = printed
    return expected


def verify(bench, external, expected=None, only=None, strict=False):
    read_json.cache_clear()
    report = {}
    for key, (items, fields, pinned) in ROWS.items():
        if only and key not in only:
            continue
        row = report[key] = dict(fields=fields, sources=[], issues=[])
        inputs = []
        for item in items:
            try:
                d, record = evidence(item, bench, external)
                inputs.append(d)
                if strict and record["producing_commit"] is None and item[0] != "@sem-reference":
                    row["issues"].append(f"unknown producing commit for {item[0]}")
            except (KeyError, ValueError, TypeError, OSError, EOFError) as exc:
                record = dict(file=item[0], producing_commit=item[1], command=item[2])
                row["issues"].append(str(exc))
            row["sources"].append(record)
        if len(inputs) != len(items):
            continue
        try:
            row.update(reduce_row(key, inputs))
            got = [float(v) for v in row["printed"]]
            if pinned is not None and got != pinned:
                row["issues"].append(f"disagrees with pinned historical cells {pinned}")
            if expected is not None and got != expected[key]:
                row["issues"].append(f"disagrees with supplied Table 3 cells {expected[key]}")
            if pinned is None and expected is None:
                row["issues"].append("supply --table and --numbers to verify the external result")
        except (KeyError, IndexError, ValueError, ArithmeticError, TypeError) as exc:
            row["issues"].append(f"cannot regenerate: {exc}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=HERE, help="directory containing the shipped benchmark files")
    parser.add_argument("--table", type=Path, help="revision-14 LaTeX Table 3, with --numbers")
    parser.add_argument("--numbers", type=Path, help="LaTeX macro definitions used by the table")
    parser.add_argument("--strict-provenance", action="store_true", help="also fail on unrecorded producing commits")
    parser.add_argument("--only", choices=list(ROWS), nargs="+", help="verify selected row IDs")
    for name, desc in (("all-pixel-q", "figure's all-pixel geometry JSON"),
                       ("heldout-calibration", "separate held-out calibration run JSON"),
                       ("au-fit", "stamped series fit JSON for the measured row"),
                       ("sem-reference", "JSON with citation, diameter_mean_nm and diameter_sd_nm from the SEM "
                        "reference (use - to read JSON from stdin)")):
        parser.add_argument("--" + name, type=Path, help=desc)
    args = parser.parse_args(argv)
    if bool(args.table) != bool(args.numbers):
        parser.error("supply --table and --numbers together")
    try:
        expected = manuscript_rows(args.table, args.numbers) if args.table else None
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    external = {k.replace("_", "-"): v for k, v in vars(args).items()}
    rows = verify(args.bench, external, expected, args.only, args.strict_provenance)
    print(json.dumps(rows, indent=2, ensure_ascii=True, allow_nan=False))
    return int(any(r["issues"] for r in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
