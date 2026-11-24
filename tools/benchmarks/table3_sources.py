"""Verify Table 3 row by row from the benchmark records and supplied manuscript."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

if __package__:
    from .case_streams import SETS
    from .composition_scores import METHOD
    from .table3_metrics import reduce_row
    from .table3_metrics_r15 import reduce_row as reduce_r15
else:
    from case_streams import SETS
    from composition_scores import METHOD
    from table3_metrics import reduce_row
    from table3_metrics_r15 import reduce_row as reduce_r15

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
ROWS14 = {
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
    "height": ([source("history/vertical_height_bias.json", "vertical_height_bias")],
               "cases[name].summary.full.h.noise_free_bias; max absolute off, Si 10%, "
               "least/most negative of Au 10/30/40%. Full angular window, historical height model.", [0.1, -3, -13, -60]),
    "au": ([("@au-fit", "e14282d", "python series.py fit <cycle>"),
            ("@sem-reference", None, "divide published diameter mean and SD by two")],
           "SEM diameter_mean_nm, diameter_sd_nm; series fit values.R_1.{median,low95,high95}, code.commit", None),
}


FROZEN = "65a640b0"
FITTING_TREE = "ea44c0cbf8875fc84dbf6303e847c2b7e904d1e0"


def predict_run(name, options):
    stem = f"posterior_r15_{name}"
    return source(f"per_cut/{stem}.json.gz", "posterior_recovery",
                  f"--network {NETWORK} --out {stem} --checkpoint <checkpoint-dir>/{stem} {options}", FROZEN)


FIRST15 = predict_run("first", "--workers 5 --set fit1d --n 200 --auto")
FRESH15 = predict_run("fresh", "--workers 5 --set fresh --auto")
EXT15 = predict_run("extended", "--workers 4 --set extended --n 600")
SI_CUT = "--workers 4 --set fit1d --n 200 --scattering dwba --substrate si --cut "
DWBA15 = [predict_run("dwba_tp", SI_CUT + "two_path"), predict_run("dwba_born", SI_CUT + "born --auto"),
          predict_run("dwba_eia", SI_CUT + "eia")]


def paired_run(stem, labels, by_cut=False):
    pairs = " ".join(f"{label}={BENCH}results/posterior_r15_{run}.json" for label, run in labels)
    return source(stem, "posterior_pairs", ("--by-cut " if by_cut else "") + BENCH + stem + " " + pairs)


PAIRS15 = paired_run("posterior_r15_dwba_pairs.json",
                     [("born", "first"), ("si_tp", "dwba_tp"), ("si_born", "dwba_born"), ("si_eia", "dwba_eia")])
BY_CUT15 = paired_run("posterior_r15_dwba4_pairs.json",
                      [("born", "first"), ("si_l1", "dwba_eia"), ("si_l2", "dwba_born"), ("si_l3", "dwba_tp"),
                       ("si_auto2", "dwba_born"), ("si_l4", "dwba4_si"), ("ps100_l4", "dwba4_ps100"),
                       ("si_auto4", "dwba4_si"), ("ps100_auto4", "dwba4_ps100")], by_cut=True)
CHOICE = ("composition_score on results[].choose.composition_weights and truth_composition_index "
          "(radius-ordered families and structure)")
HIGH = CHOICE + ", winning weight >= 0.99, correct/eligible for evidence then bic"


def unpinned(key):
    items, fields, _ = ROWS14[key]
    return items, fields, None


ROWS15 = {
    **{key: unpinned(key) for key in ("all_pixel_q", "radial", "agbh", "calibration")},
    # rerun at the frozen commit: the producer is the file's own stamp, as for the measured Au fit
    "calibration_heldout": ([(name, None, cmd) for name, _, cmd in ROWS14["calibration_heldout"][0]],
                            ROWS14["calibration_heldout"][1], None),
    **{key: unpinned(key) for key in ("independent_model", "bornagain")},
    "first.error": ([FIRST15], ROWS14["first.error"][1], None),
    "first.coverage": ([FIRST15], ROWS14["first.coverage"][1], None),
    "first.composition": ([FIRST15], CHOICE + ", correct percent by level", None),
    "first.high": ([FIRST15], HIGH, None),
    "third.error": ([FRESH15], ROWS14["third.error"][1], None),
    "third.coverage": ([FRESH15], ROWS14["third.coverage"][1], None),
    "third.composition": ([FRESH15], CHOICE + ", overall correct percent", None),
    "third.high": ([FRESH15], HIGH, None),
    "third.auto": ([FRESH15], CHOICE.replace(".choose.", ".auto.") +
                   ", overall correct percent, matched-count auto radius median error and 68% coverage", None),
    "extended.error": ([EXT15], ROWS14["extended.error"][1], None),
    "extended.composition": ([EXT15], CHOICE + ", correct percent by level", None),
    "network.error": ([FIRST15], ROWS14["network.error"][1], None),
    "search": ([source("search_baseline_r15_summary.json", "search_baseline",
                       "--set fit1d --n 200 --workers 8 --out search_baseline_r15")], ROWS14["search"][1], None),
    "dwba": ([PAIRS15, *DWBA15], 'runs.{si_tp,si_born,si_eia}["sphere q+R 2.3-3.5"].R.median_abs', None),
    "dwba4": ([BY_CUT15], "runs.{si_l4,si_l2}.overall.choose.{composition_right,parameters.R.median_abs_relative}", None),
    "galaxi": ([source("galaxi_posterior.json", "galaxi_posterior")], ROWS14["galaxi"][1], None),
    "height": ([source("vertical_height_bias.json", "vertical_height_bias")],
               "cases[name].summary.full.h.noise_free_bias, max absolute off, Si 10%, least/most negative Au", None),
    "au": ([(name, None, cmd) for name, _, cmd in ROWS14["au"][0]], ROWS14["au"][1], None),
}
REVISIONS = {14: ROWS14, 15: ROWS15}


@lru_cache(maxsize=12)
def read_json(path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def recovery_stamp(summary, command):
    code = summary["code"]
    if not code["commit"].startswith(FROZEN) or code.get("fitting_code_modified") is not False:
        raise ValueError(f"expected Predict commit {FROZEN} with fitting_code_modified: false")
    if "fitting_tree" in code and code["fitting_tree"] != FITTING_TREE:
        raise ValueError("archive fitting tree disagrees with the frozen commit")
    args = command.split()
    set_name = args[args.index("--set") + 1]
    n, seed = SETS[set_name]
    if (summary["n"], summary["set"], summary["seed0"]) != (n, set_name, seed):
        raise ValueError("archive count, set or root seed disagrees with the producing command")
    expected = dict(stream_scheme="seedsequence-v1", truth="independent", auto="--auto" in args)
    for option, default in (("cut", "born"), ("substrate", "si"), ("scattering", "born_qz")):
        flag = "--" + option
        expected[option] = args[args.index(flag) + 1] if flag in args else default
    for key, value in expected.items():
        if summary["settings"][key] != value:
            raise ValueError(f"archive setting {key} disagrees with the producing command")
    return code["commit"]


def evidence(item, bench, external, revision):
    name, commit, command = item
    path = external.get(name[1:]) if name.startswith("@") else bench / name
    if path is None:
        raise ValueError("missing --" + name[1:])
    if name.startswith("per_cut/posterior_r15_") and not path.exists():
        unpacked = bench / "results" / path.stem
        if not unpacked.exists():
            raise ValueError(f"missing archive: {name} or results/{path.stem}")
        path = unpacked
    d = json.load(sys.stdin) if name == "@sem-reference" and path == Path("-") else read_json(path)
    record = dict(file=path.name if name.startswith("@") else BENCH + path.relative_to(bench).as_posix(),
                  producing_commit=commit, command=command)
    if name.startswith("per_cut/posterior_r15_"):
        record.update(producing_commit=recovery_stamp(d["summary"], command), commit_source="summary.code.commit")
    elif name == BY_CUT15[0] and d.get("composition_method") != METHOD:
        raise ValueError(f"{name}: paired report must use {METHOD}")
    elif name == "search_baseline_r15_summary.json":
        n, seed = SETS["fit1d"]
        if (d["n"], d["set"], d["seed0"], d["stream_scheme"]) != (n, "fit1d", seed, "seedsequence-v1"):
            raise ValueError("search baseline did not use the r15 first set")
        code = d["code"]
        if not code["commit"] or code.get("fitting_code_modified") is not False:
            raise ValueError("search baseline needs a producing commit and unmodified fitting code")
        record.update(producing_commit=code["commit"], commit_source="code.commit")
    elif revision == 15 and name in ("galaxi_posterior.json", "vertical_height_bias.json"):
        code = d.get("code", {})
        if not code.get("commit") or code.get("fitting_code_modified") or code.get("modified"):
            raise ValueError(f"{name}: supply the stamped rerun from the merged commit")
        record.update(producing_commit=code["commit"], commit_source="code.commit")
    elif "summary" in d:
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
        if not d["heldout"]["commit"] or (commit and not d["heldout"]["commit"].startswith(commit)):
            raise ValueError("held-out calibration has a different producing commit")
        record.update(producing_commit=d["heldout"]["commit"], commit_source="heldout.commit",
                      command=f"python {BENCH}calibration_check.py --part coverage --cases {coverage['cases']} "
                              f"--seed {coverage['seed0']} --workers {coverage['workers']} --json <output.json>")
    elif name == "@au-fit":
        producer = d["code"]["commit"]
        if not producer or d["code"].get("modified") or (commit and not producer.startswith(commit)):
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


def manuscript_rows(table, numbers, revision=15):
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
    manifest = REVISIONS[revision]
    if len(rows) != len(manifest):
        raise ValueError(f"expected {len(manifest)} data rows in the revision-{revision} Table 3, found {len(rows)}")
    expected = {}
    for key, (label, cell) in zip(manifest, rows):
        printed = tex_values(cell)
        if key in ("calibration", "calibration_heldout", "au"):
            printed = tex_values(label) + printed
        expected[key] = printed
    return expected


def verify(bench, external, expected=None, only=None, strict=False, revision=15):
    read_json.cache_clear()
    report = {}
    reducer = reduce_r15 if revision == 15 else reduce_row
    for key, (items, fields, pinned) in REVISIONS[revision].items():
        if only and key not in only:
            continue
        row = report[key] = dict(fields=fields, sources=[], issues=[])
        inputs = []
        for item in items:
            try:
                d, record = evidence(item, bench, external, revision)
                inputs.append(d)
                if strict and record["producing_commit"] is None and item[0] != "@sem-reference":
                    row["issues"].append(f"unknown producing commit for {item[0]}")
            except (KeyError, ValueError, TypeError, AttributeError, OSError, EOFError) as exc:
                record = dict(file=item[0], producing_commit=item[1], command=item[2])
                row["issues"].append(str(exc))
            row["sources"].append(record)
        if len(inputs) != len(items):
            continue
        try:
            row.update(reducer(key, inputs))
            got = [float(value) for cell in row["printed"] for value in re.split("--|/", cell)]
            if not all(map(math.isfinite, got)):
                raise ValueError("non-finite value in printed cells")
            if pinned is not None and got != pinned:
                row["issues"].append(f"disagrees with pinned historical cells {pinned}")
            if expected is not None and got != expected[key]:
                row["issues"].append(f"disagrees with supplied Table 3 cells {expected[key]}")
            if pinned is None and expected is None:
                row["issues"].append("supply --table and --numbers to verify the printed result")
        except (KeyError, IndexError, ValueError, ArithmeticError, TypeError) as exc:
            row["issues"].append(f"cannot regenerate: {exc}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", choices=REVISIONS, type=int, default=15, help="paper revision (default: 15)")
    parser.add_argument("--bench", type=Path, default=HERE, help="directory containing the shipped benchmark files")
    parser.add_argument("--table", type=Path, help="LaTeX Table 3 for the selected revision, with --numbers")
    parser.add_argument("--numbers", type=Path, help="LaTeX macro definitions used by the table")
    parser.add_argument("--strict-provenance", action="store_true", help="also fail on unrecorded producing commits")
    parser.add_argument("--only", choices=sorted(ROWS14.keys() | ROWS15.keys()), nargs="+", help="verify selected row IDs")
    for name, desc in (("all-pixel-q", "figure's all-pixel geometry JSON"),
                       ("heldout-calibration", "separate held-out calibration run JSON"),
                       ("au-fit", "stamped series fit JSON for the measured row"),
                       ("sem-reference", "JSON with citation, diameter_mean_nm and diameter_sd_nm from the SEM "
                        "reference (use - to read JSON from stdin)")):
        parser.add_argument("--" + name, type=Path, help=desc)
    args = parser.parse_args(argv)
    if args.only and (unknown := set(args.only).difference(REVISIONS[args.revision])):
        parser.error(f"unknown row(s) for revision {args.revision}: {', '.join(sorted(unknown))}")
    if bool(args.table) != bool(args.numbers):
        parser.error("supply --table and --numbers together")
    try:
        expected = manuscript_rows(args.table, args.numbers, args.revision) if args.table else None
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    external = {k.replace("_", "-"): v for k, v in vars(args).items()}
    rows = verify(args.bench, external, expected, args.only, args.strict_provenance, args.revision)
    print(json.dumps(rows, indent=2, ensure_ascii=True, allow_nan=False))
    return int(any(r["issues"] for r in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
