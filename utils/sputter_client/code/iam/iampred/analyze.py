'One command from a folder of frames to a report (the entry point for a GUI): intake -> predict -> report -> present.'

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from . import live as LV

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent.parent / "assets"

POLY = ("PS", "PNO", "PSBMA", "PET")
FILMS = POLY + ("PEO",)


GENERIC_HOST = {"Ag": "PS", "Cu": "PEO"}
FAMILIES = {("Au", "Si"): ("si_p3h", None), ("Ag", "Si"): ("si_p3h", None),
            ("Au", "TiO2"): ("tio2_v2", None), ("Ag", "TiO2"): ("tio2_v2", None),
            **{("Cu", s): ("cu_p5", None) for s in FILMS},
            **{("Ag", s): ("polyag_p3", "ref04" if s == "PNO" else "ref05") for s in FILMS}}


ALTERNATIVES = {"polyag_p3": ["polyag_v5"], "si_p3h": ["si_v6"], "cu_p5": ["cu_v4"]}
MODES = dict(quick=dict(refine=0, sequence=0, max_windows=60), refine=dict(refine=8, sequence=0, max_windows=24),
             full=dict(refine=30, sequence=30, max_windows=96))
FILES = dict(report_md="REPORT.md", report_png="report.png", report_csv="report.csv", windows_csv="windows.csv",
             evolution_png="evolution.png", patterns_png="patterns.png", summary_json="summary.json",
             main_json="main/main.json")


def host_of(metal, substrate):
    'The substrate the import and the forward model use: a concrete polymer for --substrate Polymer.'
    return GENERIC_HOST.get(metal, substrate) if substrate == "Polymer" else substrate


def choose(metal, substrate):
    '(family, like) for a sample, or SystemExit with the supported combinations.'
    substrate = host_of(metal, substrate)
    if (metal, substrate) not in FAMILIES:
        sup = ", ".join(f"{m} on {s}" for (m, s) in FAMILIES) + ", Ag or Cu on Polymer"
        raise SystemExit(f"no network for {metal} on {substrate}. Supported: {sup}. "
                         f"The direct readings do not need a network: run with --family of the closest case to get them.")
    return FAMILIES[(metal, substrate)]


def family_mask(fam):
    "(file, sha256) of the detector mask the family's networks were trained with (env IAM_PIXMASK, a sha256 prefix),"
    import hashlib
    want = (json.loads((HERE / "families" / f"{fam}.json").read_text(encoding="utf-8")).get("env") or {}).get("IAM_PIXMASK")
    if not want or want == "none":
        return None
    for f in sorted(ASSETS.glob("*.npy")):
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        if digest.startswith(want):
            return f, digest
    return None


def data_for_mask(data, digest):
    'The data root for caches of this mask: DATA itself when it is empty or holds this mask, else DATA/mask-<prefix>'
    marker = data / "pixmask.json"
    if marker.exists():
        same = json.loads(marker.read_text(encoding="utf-8")).get("sha256") == digest
    else:
        same = not (data.exists() and any(p.is_dir() for p in data.iterdir()))
    return data if same else data / f"mask-{digest[:12]}"


def _mask_used(data, label):
    'The pixel mask the scan was imported with (its marker): the mask file and what the automatic mask added.'
    marker = data / label / "pixmask.json" if data is not None else None
    if marker is None or not marker.exists():
        return None
    m = json.loads(marker.read_text(encoding="utf-8"))
    auto = m.get("auto_mask") or {}
    return dict(file=m.get("file"), automatic=bool(auto), base_used=not auto.get("base"),
                **{k: auto.get(k) for k in ("dead", "hot", "shadow", "holder", "invalid", "kept", "pixels") if k in auto})


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--"); argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--metal", required=True, choices=["Au", "Ag", "Cu"])
    ap.add_argument("--substrate", required=True, choices=["Si", "TiO2", "PS", "PET", "PSBMA", "PNO", "PEO", "Polymer"],
                    help="Polymer: a polymer film of unknown or another kind (the optics of the metal network's polymer)")
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
    if "--mask" not in extra and data is not None and family_mask(fam) is not None:
        mask_file, digest = family_mask(fam)
        extra = extra + ["--mask", str(mask_file)]
        nested = data_for_mask(data, digest)
        if nested != data:
            print(f"mask {mask_file.name}: {data} holds caches of another mask, this scan goes to {nested}", flush=True)
            if (nested / a.label).is_dir() and not (nested / a.label / "scan.json").exists():
                shutil.rmtree(nested / a.label)
        data = nested

    live = out / "live"
    LV.clear(live)

    def run(name, args):
        nonlocal status
        t0 = time.time()
        print(f"[{dict(intake=1, predict=2, report=3)[name]}/4] {name} ...", flush=True)
        rc = LV.run_step([sys.executable, "-m", f"iampred.{name}"] + [str(v) for v in args], out / f"analyze_{name}.log",
                         HERE.parent, live / name, name)
        steps.append(dict(step=name, returncode=rc, seconds=round(time.time() - t0, 1), log=str(out / f"analyze_{name}.log")))
        print(f"      {'done' if rc == 0 else 'FAILED (see ' + steps[-1]['log'] + ')'} in {steps[-1]['seconds']:.0f} s", flush=True)
        if rc != 0 and status == "ok":
            status = f"failed at {name}"
        return rc == 0

    def run_quiet(name, args, tag):
        'An alternative network, or the main-screen step: its failure does not fail the analysis.'
        t0 = time.time()
        rc = LV.run_step([sys.executable, "-m", f"iampred.{name}"] + [str(v) for v in args],
                         out / f"analyze_{name}_{tag}.log", HERE.parent, live / f"{name}_{tag}", name)
        steps.append(dict(step=f"{name} ({tag})", returncode=rc, seconds=round(time.time() - t0, 1), log=str(out / f"analyze_{name}_{tag}.log")))
        print(f"      {name} ({tag}): {'done' if rc == 0 else 'failed'} in {steps[-1]['seconds']:.0f} s", flush=True)
        return rc == 0

    have = bool(data and (data / a.label / "scan.json").exists())
    ok = True
    if have:
        steps.append(dict(step="intake", returncode=0, seconds=0.0, log=None, note="scan already imported: reused"))
        print("[1/4] intake ... reused the imported scan", flush=True)
    else:
        args = ["--frames", a.frames, "--label", a.label, "--metal", a.metal, "--substrate", host_of(a.metal, a.substrate),
                "--alpha-i", a.alpha_i, "--workers", a.workers]
        args += ["--rate-nm-s", a.rate_nm_s] if a.rate_nm_s is not None else ["--qcm-nm-s", a.qcm_nm_s]
        args += ["--data", data] if data is not None else []
        args += ["--shadow"] if a.substrate == "Si" and "--shadow" not in extra else []
        ok = run("intake", args + extra)
    tried = {}
    if ok:
        alts = ALTERNATIVES.get(fam, []) if (a.alternatives == "on" or (a.alternatives == "auto" and a.mode != "full")) and not a.family else []
        LV.say("models", "predict", families=[fam] + alts, mode=a.mode)
        for f in [fam] + alts:
            sub = out / f
            args = [f, "--scan", a.label, "--out", sub, "--times", "all", "--step", a.step, "--max-windows", a.max_windows or mode["max_windows"],
                    "--k", 3, "--workers", a.workers, "--refine", mode["refine"]]
            args += ["--sequence", mode["sequence"]] if mode["sequence"] else []
            args += ["--like", like] if like else []
            args += ["--data", data] if data is not None else []
            good = run("predict", args) if f == fam else run_quiet("predict", args, f)
            if good and (sub / "summary.json").exists():
                s = json.loads((sub / "summary.json").read_text(encoding="utf-8"))
                tried[f] = next((float(s[k]) for k in ("median_dev5_sequence", "median_dev5_refined", "median_dev5_best") if s.get(k) is not None), float("inf"))
        ok = bool(tried)
        if ok:
            fam = min(tried, key=tried.get)
            LV.say("chosen", "predict", family=fam, deviations={k: LV.rounded(v, 3) for k, v in tried.items()})
            for v in ("windows.csv", "summary.json", "evolution.png", "patterns.png", "sequence_record.json"):
                if (out / fam / v).exists():
                    shutil.copyfile(out / fam / v, out / v)
            status = "ok" if status.startswith("failed at predict") else status
    if ok:
        ok = run("report", [out])
    if ok:
        print("[4/4] present ...", flush=True)
        args = [out / fam, "--k", 3, "--workers", a.workers] + (["--data", data] if data is not None else [])
        if run_quiet("present", args, "main") and (out / fam / "main" / "main.json").exists():
            shutil.copytree(out / fam / "main", out / "main", dirs_exist_ok=True)
    files = {k: (str(out / v) if (out / v).exists() else None) for k, v in FILES.items()}
    res = dict(status=status, label=a.label, metal=a.metal, substrate=a.substrate, host=host_of(a.metal, a.substrate), family=fam, families_tried=tried, like=like, mode=a.mode, steps=steps, files=files,
               mask=_mask_used(data, a.label), note="development reference: not validated against independent characterisation")
    (out / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"status: {status}; report: {files['report_md']}", flush=True)
    LV.clear(live)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
