'One command from a folder of frames to a report (the entry point for a GUI): intake -> predict -> report -> present.'

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

POLY = ("PS", "PNO", "PSBMA", "PET")
FAMILIES = {("Au", "Si"): ("si_p3h", None), ("Ag", "Si"): ("si_p3h", None),
            ("Au", "TiO2"): ("tio2_v2", None), ("Ag", "TiO2"): ("tio2_v2", None),
            ("Cu", "PEO"): ("cu_p3", None),
            **{("Ag", s): ("polyag_p3", "ref04" if s == "PNO" else "ref05") for s in POLY}}


ALTERNATIVES = {"polyag_p3": ["polyag_v5"], "si_p3h": ["si_v6"], "cu_p3": ["cu_v4"]}
MODES = dict(quick=dict(refine=0, sequence=0, max_windows=60), refine=dict(refine=8, sequence=0, max_windows=24),
             full=dict(refine=30, sequence=30, max_windows=96))
FILES = dict(report_md="REPORT.md", report_png="report.png", report_csv="report.csv", windows_csv="windows.csv",
             evolution_png="evolution.png", patterns_png="patterns.png", summary_json="summary.json",
             main_json="main/main.json")


def choose(metal, substrate):
    '(family, like) for a sample, or SystemExit with the supported combinations.'
    if (metal, substrate) not in FAMILIES:
        sup = ", ".join(f"{m} on {s}" for (m, s) in FAMILIES)
        raise SystemExit(f"no network for {metal} on {substrate}. Supported: {sup}. "
                         f"The direct readings do not need a network: run with --family of the closest case to get them.")
    return FAMILIES[(metal, substrate)]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--"); argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--metal", required=True, choices=["Au", "Ag", "Cu"])
    ap.add_argument("--substrate", required=True, choices=["Si", "TiO2", "PS", "PET", "PSBMA", "PNO", "PEO"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rate-nm-s", type=float)
    g.add_argument("--qcm-nm-s", type=float)
    ap.add_argument("--alpha-i", type=float, default=0.40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=list(MODES) + ["auto"], default="quick")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--max-windows", type=int, default=0, help="0: the default of the mode")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--family")
    ap.add_argument("--like")
    ap.add_argument("--alternatives", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--data")
    a = ap.parse_args(argv)
    fam, like = choose(a.metal, a.substrate) if not a.family else (a.family, None)
    like = a.like or like
    if a.mode == "auto":
        a.mode = "quick" if fam.split("_")[0] in ("si", "tio2") else "refine"
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data).resolve() if a.data else Path(os.environ.get("IAM_DATA", "")).resolve() if os.environ.get("IAM_DATA") else None
    mode = MODES[a.mode]
    steps, status = [], "ok"

    def run(name, args):
        nonlocal status
        t0 = time.time()
        print(f"[{dict(intake=1, predict=2, report=3)[name]}/4] {name} ...", flush=True)
        with open(out / f"analyze_{name}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.run([sys.executable, "-m", f"iampred.{name}"] + [str(v) for v in args], stdout=fh, stderr=subprocess.STDOUT,
                                cwd=str(HERE.parent)).returncode
        steps.append(dict(step=name, returncode=rc, seconds=round(time.time() - t0, 1), log=str(out / f"analyze_{name}.log")))
        print(f"      {'done' if rc == 0 else 'FAILED (see ' + steps[-1]['log'] + ')'} in {steps[-1]['seconds']:.0f} s", flush=True)
        if rc != 0 and status == "ok":
            status = f"failed at {name}"
        return rc == 0

    def run_quiet(name, args, tag):
        'An alternative network, or the main-screen step: its failure does not fail the analysis.'
        t0 = time.time()
        with open(out / f"analyze_{name}_{tag}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.run([sys.executable, "-m", f"iampred.{name}"] + [str(v) for v in args], stdout=fh, stderr=subprocess.STDOUT,
                                cwd=str(HERE.parent)).returncode
        steps.append(dict(step=f"{name} ({tag})", returncode=rc, seconds=round(time.time() - t0, 1), log=str(out / f"analyze_{name}_{tag}.log")))
        print(f"      {name} ({tag}): {'done' if rc == 0 else 'failed'} in {steps[-1]['seconds']:.0f} s", flush=True)
        return rc == 0

    have = bool(data and (data / a.label / "scan.json").exists())
    ok = True
    if have:
        steps.append(dict(step="intake", returncode=0, seconds=0.0, log=None, note="scan already imported: reused"))
        print("[1/4] intake ... reused the imported scan", flush=True)
    else:
        args = ["--frames", a.frames, "--label", a.label, "--metal", a.metal, "--substrate", a.substrate, "--alpha-i", a.alpha_i, "--workers", a.workers]
        args += ["--rate-nm-s", a.rate_nm_s] if a.rate_nm_s is not None else ["--qcm-nm-s", a.qcm_nm_s]
        args += ["--data", data] if a.data else []
        args += ["--shadow"] if a.substrate == "Si" and "--shadow" not in extra else []
        ok = run("intake", args + extra)
    tried = {}
    if ok:
        alts = ALTERNATIVES.get(fam, []) if (a.alternatives == "on" or (a.alternatives == "auto" and a.mode != "full")) and not a.family else []
        for f in [fam] + alts:
            sub = out / f
            args = [f, "--scan", a.label, "--out", sub, "--times", "all", "--step", a.step, "--max-windows", a.max_windows or mode["max_windows"],
                    "--k", 3, "--workers", a.workers, "--refine", mode["refine"]]
            args += ["--sequence", mode["sequence"]] if mode["sequence"] else []
            args += ["--like", like] if like else []
            args += ["--data", data] if a.data else []
            good = run("predict", args) if f == fam else run_quiet("predict", args, f)
            if good and (sub / "summary.json").exists():
                s = json.loads((sub / "summary.json").read_text(encoding="utf-8"))
                tried[f] = next((float(s[k]) for k in ("median_dev5_sequence", "median_dev5_refined", "median_dev5_best") if s.get(k) is not None), float("inf"))
        ok = bool(tried)
        if ok:
            fam = min(tried, key=tried.get)
            for v in ("windows.csv", "summary.json", "evolution.png", "patterns.png", "sequence_record.json"):
                if (out / fam / v).exists():
                    shutil.copyfile(out / fam / v, out / v)
            status = "ok" if status.startswith("failed at predict") else status
    if ok:
        ok = run("report", [out])
    if ok:
        print("[4/4] present ...", flush=True)
        args = [out / fam, "--k", 3, "--workers", a.workers] + (["--data", data] if a.data else [])
        if run_quiet("present", args, "main") and (out / fam / "main" / "main.json").exists():
            shutil.copytree(out / fam / "main", out / "main", dirs_exist_ok=True)
    files = {k: (str(out / v) if (out / v).exists() else None) for k, v in FILES.items()}
    res = dict(status=status, label=a.label, metal=a.metal, substrate=a.substrate, family=fam, families_tried=tried, like=like, mode=a.mode, steps=steps, files=files,
               note="development reference: not validated against independent characterisation")
    (out / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"status: {status}; report: {files['report_md']}", flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
