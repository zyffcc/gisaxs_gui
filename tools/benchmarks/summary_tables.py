"""Print the tables of the Predict parameters benchmarks from the stored summaries, so that no number in Physics and
Methods 9.13, the model card or the README of this folder is typed by hand.

usage: python tools/benchmarks/summary_tables.py [--lang en|zh] [--only NAME ...]
Reads posterior_*_summary.json and posterior_dwba_pairs.json beside this script (the paper's runs) and, for the
comparison with the removed fit1d method, history/fit1d_recovery_summary.json. Writes Markdown to stdout; nothing is
computed again (``large`` alone counts cuts of per_cut/). NAME is one of: recovery, extended, sensitivity, own, dwba,
pairs, timing, v5, large, card.

"true composition most probable" is judged on the most probable SOLUTION (``composition.exact`` of a summary: the
composition of the first solution Predict lists is the true one); "true composition listed" is
``true_in_equivalent_set``; "truth's mode first / listed" is ``truth_mode`` (every R and D of a solution within 20 %
of the truth).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VALUES = (("R", "R", True), ("sR", "σR/R", False), ("D", "D", True), ("s_nn", "σnn/D", False))
LEVELS = ("model", "measured")
ZH = {
    "level": "层级", "value": "数值", "run": "运行",
    "Predict: median / P90": "预测参数：中位数 / P90", "fit1d: median / P90": "fit1d：中位数 / P90",
    "fit1d: 68 %": "fit1d：68 %", "Predict: 68 %": "预测参数：68 %", "Predict: 95 %": "预测参数：95 %",
    "true composition most probable (model / measured)": "真实组成是最可能的解的组成（model / measured）",
    "true composition listed (model / measured)": "真实组成在列出的解中（model / measured）",
    "truth's mode first (model / measured)": "真值所在的模态排第一（model / measured）",
    "truth's mode listed (model / measured)": "真值所在的模态在列出的解中（model / measured）",
    "curves": "曲线数", "68 % intervals (four values)": "68 % 区间（四个数值）",
    "95 % intervals (four values)": "95 % 区间（四个数值）", "R: median": "R：中位数", "R: P90": "R：P90",
    "R: 68 %": "R：68 %", "composition most probable": "真实组成最可能", "composition listed": "真实组成列出",
    "mode first": "模态排第一", "mode listed": "模态列出", "q₊R of the sphere": "球的 q₊R", "spheres": "球数",
    "largest sphere's q₊R": "最大球的 q₊R", "up to 1.7 or no sphere": "不超过 1.7 或没有球",
    "above 2.3": "大于 2.3", "above 3.5": "大于 3.5", "up to 1.2": "不超过 1.2",
    "spheres off by more than 10 %: curves": "有球的 R 偏离超过 10 % 的曲线数", "of them χ²/N > 2": "其中 χ²/N > 2",
    "their median χ²/N": "它们的 χ²/N 中位数", "stage": "阶段", "median seconds": "秒（中位数）",
    "network": "网络", "composition given": "给定组成", "composition chosen": "选择组成",
    "model error floor 3 %": "模型误差下限 3 %", "model error floor 5 % (default)": "模型误差下限 5 %（默认）",
    "model error floor 8 %": "模型误差下限 8 %", "amplitude range 100": "幅度范围 100",
    "Born-term truth, given q_z": "Born 项真值，给定 q_z", "DWBA on Si, given both angles": "Si 上 DWBA，给定两个角",
    "DWBA on Si, given q_z": "Si 上 DWBA，只给 q_z", "DWBA on Si, given neither": "Si 上 DWBA，两者都不给",
    "Fitting's own model": "拟合工作区自己的模型", "independent truth model": "独立的真值模型",
    "V5: median / P90": "V5：中位数 / P90", "seconds per curve": "每条曲线的秒数", "set": "曲线组",
    "measured curves": "measured 曲线数", "one population more than the truth, flagged large": "比真值多一群且带 large 标记",
    "R of that population (nm)": "这一群的 R（nm）", "fit1d's set": "fit1d 的一组", "extended set": "扩展的一组",
}
_lang = "en"


def t(text: str) -> str:
    return ZH.get(text, text) if _lang == "zh" else text


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def pct(x: float) -> str:
    """A fraction as a percentage: one decimal below 10 %, none above."""
    v = 100 * x
    return f"{v:.1f} %" if abs(v) < 9.95 else f"{v:.0f} %"


def percent(x: float) -> int:
    """A share as a whole percentage, rounded as its two-decimal form is (0.805 reads 0.81 and 81 %)."""
    return int(round(100 * float(f"{x:.2f}")))


def whole(x: float | None) -> str:
    return "–" if x is None else f"{percent(x)} %"


def signed(x: float) -> str:
    v = int(round(100 * x))
    return f"{v:+d} %" if v else "0 %"


def error(row: dict, relative: bool) -> str:
    if relative:
        return f"{pct(row['median_abs'])} / {pct(row['p90_abs'])}"
    return f"{row['median_abs']:.3f} / {row['p90_abs']:.3f}"


def span(values) -> str:
    values = [percent(v) for v in values]
    return f"{min(values)}–{max(values)} %"


def pair(summary: dict, block: str, key: str) -> str:
    return " / ".join(whole(summary[block][level][key]) for level in LEVELS)


def table(header, rows) -> None:
    print("| " + " | ".join(t(h) for h in header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(t(str(c)) for c in row) + " |")
    print()


def source(summary: dict, name: str) -> None:
    code = summary.get("code") or {}
    made = f"commit {code['commit']}, fitting tree {code['fitting_tree'][:8]}" if code else "no code field"
    s = summary.get("settings", {})
    print(f"`{name}`: n = {summary['n']}, seed0 {summary.get('seed0')}, set {summary.get('set')}, truth "
          f"{summary.get('truth')}, scattering {s.get('scattering')}, cut {s.get('cut')}, floor {s.get('floor')}, "
          f"constants {s.get('constants')}, failed {summary.get('failed')}; {made}\n")


def recovery(name: str, fit1d: str | None) -> None:
    """Errors and coverage with the true composition given, per level and value; the choice of composition."""
    s = load(name)
    old = load(fit1d) if fit1d else None
    source(s, name)
    header = ["level", "value", "Predict: median / P90", "Predict: 68 %", "Predict: 95 %"]
    header += ["fit1d: median / P90", "fit1d: 68 %"] if old else []
    rows = []
    for level in LEVELS:
        for key, label, relative in VALUES:
            row = s["parameters"][f"fit/{level}"][key]
            cells = [level, label, error(row, relative), whole(row["posterior_coverage68"]),
                     whole(row["posterior_coverage95"])]
            if old:
                o = old["parameters"][f"fit/{level}"][key]
                cells += [error(o, relative), whole(o["coverage_1sigma"])]
            rows.append(cells)
    for label, block, key in (("true composition most probable (model / measured)", "composition", "exact"),
                              ("true composition listed (model / measured)", "composition", "true_in_equivalent_set"),
                              ("truth's mode first (model / measured)", "truth_mode", "first"),
                              ("truth's mode listed (model / measured)", "truth_mode", "listed")):
        cells = ["", label, pair(s, block, key), "", ""]
        if old:
            cells += [pair(old, block, key) if block in old else "–", ""]
        rows.append(cells)
    table(header, rows)
    for level in LEVELS:
        f = s["parameters"][f"fit/{level}"]
        line = (f"- {level}: 68 % intervals {span(f[k]['posterior_coverage68'] for k, *_ in VALUES)}, 95 % intervals "
                f"{span(f[k]['posterior_coverage95'] for k, *_ in VALUES)}")
        if old:
            o = old["parameters"][f"fit/{level}"]
            line += f"; fit1d ±1 error {span(o[k]['coverage_1sigma'] for k, *_ in VALUES)}"
        print(line)
    net = s["parameters"]["network/all"]
    print("- network alone (all): " + ", ".join(f"{label} {error(net[k], rel)}" for k, label, rel in VALUES))
    print(f"- composition, all curves: most probable {whole(s['composition']['all']['exact'])}, listed "
          f"{whole(s['composition']['all']['true_in_equivalent_set'])}; truth's mode first "
          f"{whole(s['truth_mode']['all']['first'])}, listed {whole(s['truth_mode']['all']['listed'])}")
    print(f"- effective samples: median {s['fit_ess']['median']:.0f}, below 50 in {whole(s['fit_ess']['below_50'])}")
    print(f"- median seconds of this run (its workers shared the processor): composition given "
          f"{s['median_seconds']['fit']:.1f}, chosen {s['median_seconds']['choose']:.1f}")
    if old:
        print(f"- fit1d, median seconds: composition given {old['median_seconds']['fit']:.1f}, chosen "
              f"{old['median_seconds']['choose']:.1f}")
    print()


def overview(runs) -> None:
    """One row per run: coverage of the four values, R, the composition and the truth's mode, model / measured."""
    rows = []
    for label, name in runs:
        s = load(name)
        f = {level: s["parameters"][f"fit/{level}"] for level in LEVELS}
        rows.append([label, s["n"],
                     " / ".join(span(f[lv][k]["posterior_coverage68"] for k, *_ in VALUES) for lv in LEVELS),
                     " / ".join(span(f[lv][k]["posterior_coverage95"] for k, *_ in VALUES) for lv in LEVELS),
                     " / ".join(pct(f[lv]["R"]["median_abs"]) for lv in LEVELS),
                     " / ".join(pct(f[lv]["R"]["p90_abs"]) for lv in LEVELS),
                     " / ".join(whole(f[lv]["R"]["posterior_coverage68"]) for lv in LEVELS),
                     pair(s, "composition", "exact"), pair(s, "composition", "true_in_equivalent_set"),
                     pair(s, "truth_mode", "first"), pair(s, "truth_mode", "listed")])
    table(["run", "curves", "68 % intervals (four values)", "95 % intervals (four values)", "R: median", "R: P90",
           "R: 68 %", "composition most probable", "composition listed", "mode first", "mode listed"], rows)


PAIR_RUNS = (("born", "Born-term truth, given q_z"), ("si_born_new", "DWBA on Si, given q_z"),
             ("si_tp_new", "DWBA on Si, given both angles"), ("si_eia_new", "DWBA on Si, given neither"))
SPHERE_BINS = (("sphere q+R 0-1.2", "up to 1.2"), ("sphere q+R 1.2-1.7", "1.2–1.7"), ("sphere q+R 1.7-2.3", "1.7–2.3"),
               ("sphere q+R 2.3-3.5", "2.3–3.5"), ("sphere q+R 3.5-99", "above 3.5"))
CURVE_BINS = (("curves largest sphere q+R 0-1.7 or none", "up to 1.7 or no sphere"),
              ("curves largest sphere q+R 1.7-2.3", "1.7–2.3"), ("curves largest sphere q+R 2.3-99", "above 2.3"))


def pairs() -> None:
    """The DWBA pairs by q+R (posterior_dwba_pairs.json: the same cases, curve by curve)."""
    p = load("posterior_dwba_pairs.json")["runs"]
    print("R of the spheres, the true composition given: median error (bias) · share off by more than 20 % · 68 % "
          "interval holds the truth\n")
    rows = []
    for key, label in SPHERE_BINS:
        cells = [label, p["born"][key]["R"]["n"]]
        for run, _ in PAIR_RUNS:
            e = p[run][key]
            cells.append(f"{pct(e['R']['median_abs'])} ({signed(e['R']['bias'])}) · {whole(e['R']['off20'])} · "
                         f"{whole(e['in68'])}")
        rows.append(cells)
    table(["q₊R of the sphere", "spheres", *(label for _, label in PAIR_RUNS)], rows)
    print("Curves by their largest sphere: true composition most probable · truth's mode listed · median χ²/N\n")
    rows = []
    for key, label in CURVE_BINS:
        cells = [label, p["born"][key]["n"]]
        for run, _ in PAIR_RUNS:
            e = p[run][key]
            cells.append(f"{whole(e['exact'])} · {whole(e['listed'])} · {e['chi2_median']:.2f}")
        rows.append(cells)
    table(["largest sphere's q₊R", "curves", *(label for _, label in PAIR_RUNS)], rows)
    rows = []
    for run, label in PAIR_RUNS:
        e = p[run]["sphere R off >10 %"]
        rows.append([label, e["curves"], whole(e["chi2_above_2"]), f"{e['chi2_median']:.2f}"])
    table(["run", "spheres off by more than 10 %: curves", "of them χ²/N > 2", "their median χ²/N"], rows)


def timing() -> None:
    s = load("posterior_timing_summary.json")
    source(s, "posterior_timing_summary.json")
    m = s["median_seconds"]
    table(["stage", "median seconds"], [["network", f"{m['network']:.2f}"], ["composition given", f"{m['fit']:.1f}"],
                                        ["composition chosen", f"{m['choose']:.1f}"]])


def v5() -> None:
    """Predict parameters against the removed 1D Predict V5 on the 200 curves (history/v5_recovery_summary.json)."""
    s, old = load("posterior_recovery_summary.json"), load("history/v5_recovery_summary.json")
    rows = []
    for level in LEVELS:
        for key, label, relative in VALUES[:3]:
            rows.append([level, label, error(s["parameters"][f"fit/{level}"][key], relative),
                         error(old["parameters"][f"fit/{level}"][key], relative)])
    rows.append(["", "seconds per curve", f"{load('posterior_timing_summary.json')['median_seconds']['fit']:.1f}",
                 f"{old['median_seconds']:.1f}"])
    table(["level", "value", "Predict: median / P90", "V5: median / P90"], rows)
    a = old["parameters"]["fit/all"]
    print(f"- V5, all curves: median error of R {pct(a['R']['median_abs'])}, of D {pct(a['D']['median_abs'])}")
    print()


def large() -> None:
    """Measured curves whose chosen solution has one population more than the truth and the flag *large*."""
    import gzip
    rows = []
    for label, name in (("fit1d's set", "posterior_recovery"), ("extended set", "posterior_extended")):
        data = json.loads(gzip.decompress((HERE / "per_cut" / f"{name}.json.gz").read_bytes()))
        measured = [r for r in data["results"] if r["case"]["level"] == "measured" and r["choose"].get("ok")]
        hits = [max(p["R"] for p in r["choose"]["populations"]) for r in measured
                if len(r["choose"]["populations"]) == len(r["case"]["populations"]) + 1
                and any(f.startswith("large") for f in r["choose"]["flags"])]
        rows.append([label, len(measured), len(hits), f"{min(hits):.0f}–{max(hits):.0f}" if hits else "–"])
    table(["set", "measured curves", "one population more than the truth, flagged large", "R of that population (nm)"],
          rows)


CARD = (("posterior_recovery", "200 held-out curves, one or two populations"),
        ("posterior_extended", "600 held-out curves of the extended set, one to three populations"),
        ("posterior_dwba_tp", "the 200 curves drawn with the four DWBA terms on Si, both angles given"))


def card() -> None:
    """The benchmark sections of the model card (tools/ml1d/model_card.py's layout, with the 95 % coverage)."""
    for name, title in CARD:
        s = load(f"{name}_summary.json")
        print(f"## Benchmark: {name} ({title})\n")
        rows = []
        for stage in ("network/all", "fit/model", "fit/measured", "fit/all"):
            cells = [stage]
            for key, _, _ in VALUES:
                e = s["parameters"][stage][key]
                cover = (f" · {e['posterior_coverage68']:.2f} / {e['posterior_coverage95']:.2f}"
                         if e.get("posterior_coverage68") is not None else "")
                cells.append(f"{e['median_abs']:.4f} / {e['p90_abs']:.3f}{cover}")
            rows.append(cells)
        table(["stage", "R (median / P90 · 68 % / 95 % coverage)", "σR/R", "D", "σnn/D"], rows)
        c, m = s["composition"], s["truth_mode"]
        print("True composition most probable (the composition of the most probable solution is the true one), "
              f"model / measured / all: {c['model']['exact']:.2f} / {c['measured']['exact']:.2f} / "
              f"{c['all']['exact']:.2f}; true composition listed: {c['model']['true_in_equivalent_set']:.2f} / "
              f"{c['measured']['true_in_equivalent_set']:.2f} / {c['all']['true_in_equivalent_set']:.2f}; truth's mode "
              f"first: {m['model']['first']:.2f} / {m['measured']['first']:.2f} / {m['all']['first']:.2f}; listed: "
              f"{m['model']['listed']:.2f} / {m['measured']['listed']:.2f} / {m['all']['listed']:.2f}.\n")


def main() -> int:
    global _lang
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", choices=("en", "zh"), default="en")
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    _lang = a.lang
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sections = {
        "recovery": ("fit1d's 200 held-out curves (posterior_recovery)",
                     lambda: recovery("posterior_recovery_summary.json", "history/fit1d_recovery_summary.json")),
        "extended": ("the extended held-out set (posterior_extended)",
                     lambda: recovery("posterior_extended_summary.json", None)),
        "sensitivity": ("the model error floor and the amplitude range, on the 200 curves", lambda: overview((
            ("model error floor 3 %", "posterior_floor03_summary.json"),
            ("model error floor 5 % (default)", "posterior_recovery_summary.json"),
            ("model error floor 8 %", "posterior_floor08_summary.json"),
            ("amplitude range 100", "posterior_amplitude100_summary.json")))),
        "own": ("curves of Fitting's own model (posterior_own, development seeds)", lambda: overview((
            ("Fitting's own model", "posterior_own_summary.json"),
            ("independent truth model", "posterior_recovery_summary.json")))),
        "dwba": ("the same 200 cases drawn with the four DWBA terms on Si", lambda: overview((
            ("Born-term truth, given q_z", "posterior_recovery_summary.json"),
            ("DWBA on Si, given q_z", "posterior_dwba_born_summary.json"),
            ("DWBA on Si, given both angles", "posterior_dwba_tp_summary.json"),
            ("DWBA on Si, given neither", "posterior_dwba_eia_summary.json")))),
        "pairs": ("the DWBA pairs by q₊R (posterior_dwba_pairs.json)", pairs),
        "timing": ("time per curve, one worker (posterior_timing)", timing),
        "v5": ("against 1D Predict V5 (history/v5_recovery_summary.json)", v5),
        "large": ("an extra large population on the measured curves (per_cut/)", large),
        "card": ("the benchmark sections of MODEL_CARD.md", card),
    }
    for key, (title, write) in sections.items():
        if a.only and key not in a.only:
            continue
        if key != "card":
            print(f"### {title}\n")
        write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
