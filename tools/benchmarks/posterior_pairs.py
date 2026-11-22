"""Pairs of posterior_recovery.py runs on the same held-out cases that differ only in the truth's physics (the Born
term or the four DWBA terms, Si or polystyrene on Si: --scattering, --substrate), in what Predict is given of the
cut (--cut born, two_path, dwba or eia) or in the network (--network).
Per population, the error of R and D (the true composition given, the most probable solution) binned by q+R, the
vacuum q_z of the cut times the true R, for spheres and upright cylinders; per curve, by its largest sphere's q+R,
how often the composition is chosen right (``comp``), the true composition listed (``set``), the truth's mode first
or listed, and the median chi2/N; and whether chi2/N > 2 marks the curves where a sphere's R is off by more than 10 %.
usage: python tools/benchmarks/posterior_pairs.py OUT.json LABEL=results/RUN.json [LABEL=results/RUN.json ...]
(--by-cut before OUT.json: fit, choose and auto by the largest q+R of any population and by composition,
including timing and 68/95% coverage for R and D; requires identical case draws.)
(the per-curve files posterior_recovery.py writes to results/, or their gzip-compressed copies in per_cut/; prints a
table and writes it to OUT.json)"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

import recovery_cases as fr

QPLUS = float(fr.GEOM.qz_vacuum)    # k_i + k_f: alpha_i 0.40 deg, the Si Yoneda angle, 0.1045 nm (0.577 nm^-1)
BINS = (0.0, 1.2, 1.7, 2.3, 3.5, 99.0)
MODE = 0.2


def rows(path):
    raw = Path(path).read_bytes()  # results/NAME.json, or its gzip-compressed copy in per_cut/
    data = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    out = {}
    for r in data["results"]:
        c = r["case"]
        truth = sorted(c["populations"], key=lambda p: p["R"])
        fit = r.get("fit") or {}
        pops = []
        if fit.get("ok") and len(fit["populations"]) == len(truth):
            for t, g in zip(truth, fit["populations"]):
                pops.append(dict(shape=t["shape"], qR=QPLUS * t["R"], R=(g["R"] - t["R"]) / t["R"],
                                 D=(g["D"] - t["D"]) / t["D"], z=(g["R"] - t["R"]) / g["R_err"]
                                 if g.get("R_err") else np.nan,
                                 in68=g["R_68"][0] <= t["R"] <= g["R_68"][1] if "R_68" in g else None))
        mode_first = bool(pops) and all(abs(p["R"]) <= MODE and abs(p["D"]) <= MODE for p in pops)
        listed = mode_first
        for a in fit.get("alternatives", []) if fit.get("ok") else []:
            if len(a["populations"]) == len(truth) and all(
                    abs(g["R"] - t["R"]) <= MODE * t["R"] and abs(g["D"] - t["D"]) <= MODE * t["D"]
                    for t, g in zip(truth, a["populations"])):
                listed = True
        ch = r.get("choose") or {}
        out[c["index"]] = dict(level=c["level"], comp=c["composition"], pops=pops, chi2=fit.get("chi2"),
                               nsph=sum(p["shape"] == "sphere" for p in truth),
                               maxqR=max(QPLUS * p["R"] for p in truth if p["shape"] == "sphere")
                               if any(p["shape"] == "sphere" for p in truth) else 0.0,
                               first=mode_first, listed=listed,
                               exact=ch.get("ok") and ch["families"] == r["truth_families"],
                               in_set=ch.get("ok") and r["truth_families"] in ch["in_set"],
                               flags=fit.get("flags", []))
    return out


def stats(values):
    a = np.asarray(values, float)
    if not a.size:
        return dict(n=0)
    return dict(n=int(a.size), median_abs=float(np.median(np.abs(a))), p90_abs=float(np.percentile(np.abs(a), 90)),
                bias=float(np.median(a)), off20=float(np.mean(np.abs(a) > 0.2)))


def main():
    if sys.argv[1] == "--by-cut":
        from posterior_cut_classes import compare_runs

        report = compare_runs([arg.split("=", 1) for arg in sys.argv[3:]], QPLUS)
        Path(sys.argv[2]).write_text(json.dumps(report, indent=1, allow_nan=False), encoding="utf-8")
        print(f"Wrote {len(report['runs'])} runs of {report['n_curves']} cuts to {sys.argv[2]}")
        return
    out_path = sys.argv[1]
    runs = [arg.split("=", 1) for arg in sys.argv[2:]]
    loaded = {label: rows(path) for label, path in runs}
    common = sorted(set.intersection(*(set(v) for v in loaded.values())))
    report = dict(n_curves=len(common), bins=BINS, runs={})
    for label, data in loaded.items():
        res = {}
        for shape in ("sphere", "vc"):
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                pick = [p for i in common for p in data[i]["pops"] if p["shape"] == shape and lo <= p["qR"] < hi]
                if shape == "vc" and not pick:
                    continue
                key = f"{shape} q+R {lo:g}-{hi:g}"
                res[key] = dict(R=stats([p["R"] for p in pick]), D=stats([p["D"] for p in pick]),
                                in68=float(np.mean([p["in68"] for p in pick if p["in68"] is not None]))
                                if pick else None)
        for lo, hi in ((0, 1.7), (1.7, 2.3), (2.3, 99)):
            group = [i for i in common if data[i]["nsph"] and lo <= data[i]["maxqR"] < hi] if lo else \
                [i for i in common if not data[i]["nsph"] or data[i]["maxqR"] < hi]
            res[f"curves largest sphere q+R {lo:g}-{hi:g}" + (" or none" if not lo else "")] = dict(
                n=len(group), exact=float(np.mean([data[i]["exact"] for i in group])),
                in_set=float(np.mean([data[i]["in_set"] for i in group])),
                first=float(np.mean([data[i]["first"] for i in group])),
                listed=float(np.mean([data[i]["listed"] for i in group])),
                chi2_median=float(np.median([data[i]["chi2"] for i in group if data[i]["chi2"] is not None])))
        for level in ("model", "measured"):
            group = [i for i in common if data[i]["level"] == level]
            res[f"all {level}"] = dict(n=len(group), exact=float(np.mean([data[i]["exact"] for i in group])),
                                       in_set=float(np.mean([data[i]["in_set"] for i in group])),
                                       first=float(np.mean([data[i]["first"] for i in group])),
                                       listed=float(np.mean([data[i]["listed"] for i in group])),
                                       R=stats([p["R"] for i in group for p in data[i]["pops"]]),
                                       D=stats([p["D"] for i in group for p in data[i]["pops"]]))
        # does chi2/N warn when a sphere's R is off by more than 10 %?
        bad = [i for i in common if any(p["shape"] == "sphere" and abs(p["R"]) > 0.1 for p in data[i]["pops"])]
        res["sphere R off >10 %"] = dict(curves=len(bad), chi2_above_2=float(np.mean(
            [(data[i]["chi2"] or 0) > 2 for i in bad])) if bad else None, chi2_median=float(np.median(
                [data[i]["chi2"] for i in bad if data[i]["chi2"] is not None])) if bad else None)
        report["runs"][label] = res
    Path(out_path).write_text(json.dumps(report, indent=1), encoding="utf-8")
    labels = list(loaded)
    keys = list(report["runs"][labels[0]])
    for key in keys:
        line = f"{key:38s}"
        for label in labels:
            v = report["runs"][label].get(key, {})
            if "R" in v and isinstance(v["R"], dict) and v["R"].get("n"):
                line += f" | {label}: n{v['R']['n']} R med {100 * v['R']['median_abs']:.1f} P90 {100 * v['R']['p90_abs']:.1f}" \
                        f" bias {100 * v['R']['bias']:+.1f} >20% {100 * v['R']['off20']:.0f} D P90 {100 * v['D']['p90_abs']:.1f}"
                if "exact" in v:
                    line += f" comp {100 * v['exact']:.0f} set {100 * v['in_set']:.0f} first {100 * v['first']:.0f} listed {100 * v['listed']:.0f}"
            elif "exact" in v:
                line += f" | {label}: n{v['n']} comp {100 * v['exact']:.0f} set {100 * v['in_set']:.0f} first " \
                        f"{100 * v['first']:.0f} listed {100 * v['listed']:.0f} chi2 {v['chi2_median']:.2f}"
            elif "curves" in v:
                line += f" | {label}: {v['curves']} curves, chi2>2 {v['chi2_above_2']}, chi2 med {v['chi2_median']}"
        print(line)


if __name__ == "__main__":
    main()
