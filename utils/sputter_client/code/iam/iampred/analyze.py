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
from . import numerics as NUM
from . import registry as REG
from . import stop_hint as SH

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent.parent / "assets"

POLY = ("PS", "PNO", "PSBMA", "PET")
FILMS = POLY + ("PEO",)


GENERIC_HOST = {"Ag": "PS", "Cu": "PEO"}
FAMILIES = {("Au", "Si"): ("si_p3h", None), ("Ag", "Si"): ("si_p3h", None),
            ("Au", "TiO2"): ("tio2_u1", None), ("Ag", "TiO2"): ("tio2_u1", None),
            **{("Cu", s): ("cu_p5", None) for s in FILMS},
            **{("Ag", s): ("polyag_p3", "ref04" if s == "PNO" else "ref05") for s in FILMS}}


ALTERNATIVES = {"polyag_p3": ["polyag_u1"], "si_p3h": ["si_v6", "si_u1"], "cu_p5": ["cu_v4", "cu_u1"],
                "tio2_u1": ["tio2_v2"]}


UNIVERSAL = {"si_p3h": "si_u1", "polyag_p3": "polyag_u1", "cu_p5": "cu_u1"}


ROUTE_OFF_NATIVE = {"si_p3h", "polyag_p3", "cu_p5"}


PRIOR_DEFAULT_ON = {("Ag", "PS")}


def growth_prior_used(setting, metal, substrate):
    'Whether the whole-scan refinement gets the growth-law prior: on, off, or auto (PRIOR_DEFAULT_ON).'
    return setting == "on" or (setting == "auto" and (metal, substrate) in PRIOR_DEFAULT_ON)


VARIANT_FAMILIES = ("cu_p5", "cu_u1", "cu_v4", "polyag_p3", "polyag_u1")
VARIANT_ARGS = ["--variant", "lnRP=1.791759,lnAP=0", "--variant", "fb=0.9,lnR=1.1,lnRP=1.791759,lnAP=0",
                "--variant-refine", 15, "--monotone", "lnh_mass,lnt_film"]


def variant_args(family, sequence):
    'The start-variant arguments of predict for this family when it runs the whole-scan refinement.'
    return list(VARIANT_ARGS) if sequence and family in VARIANT_FAMILIES else []


MODES = dict(quick=dict(refine=0, sequence=0, max_windows=60), refine=dict(refine=0, sequence=6, max_windows=24),
             full=dict(refine=30, sequence=30, max_windows=96))
FILES = dict(report_md="REPORT.md", report_png="report.png", report_csv="report.csv", windows_csv="windows.csv",
             evolution_png="evolution.png", patterns_png="patterns.png", summary_json="summary.json",
             main_json="main/main.json", process_png="main/process.png")


def optics_key(fam, metal, lam):
    "Which optics ``fam`` renders a scan at wavelength ``lam`` (nm) with: 'own' (the scan names no wavelength),"
    cfg = REG.load(fam)
    if lam is None:
        return "own"
    if REG.lam_outside(metal, float(lam), cfg):
        return "table"
    if abs(float(lam) / REG.model_lam(cfg) - 1) <= 1e-4:
        return "hand"
    return ("anchored" if cfg.get("energy_range_kev") else "const", round(float(lam), 9))


def same_optics(fam, alts, metal, lam):
    'The alternatives that render with the optics of ``fam`` at wavelength ``lam`` (optics_key): only then does the'
    key = optics_key(fam, metal, lam)
    return [f for f in alts if optics_key(f, metal, lam) == key]


def routed(fam, metal, lam):
    'The family that analyses a scan at wavelength ``lam``: the any-energy network of ``fam`` (UNIVERSAL) when ``fam``'
    uni = UNIVERSAL.get(fam)
    if uni is None or fam not in ROUTE_OFF_NATIVE or lam is None:
        return fam
    if REG.lam_outside(metal, float(lam), REG.load(fam)) and not REG.lam_outside(metal, float(lam), REG.load(uni)):
        return uni
    return fam


def scan_lam(data, label, extra):
    "The scan's wavelength (nm): its scan.json (written by the import), else the intake options; None when neither."
    sj = data / label / "scan.json" if data is not None else None
    try:
        lam = json.loads(sj.read_text(encoding="utf-8")).get("lam") if sj is not None and sj.exists() else None
    except (OSError, ValueError):
        lam = None
    return float(lam) if lam is not None else _lam_of(extra)


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


def _lam_of(extra) -> "float | None":
    'The wavelength (nm) the intake options give (--wavelength-nm or --energy-ev), None when neither.'
    for i, item in enumerate(extra):
        if item in ("--wavelength-nm", "--energy-ev") and i + 1 < len(extra):
            try:
                value = float(extra[i + 1])
            except ValueError:
                return None
            return value if item == "--wavelength-nm" else 1239.841984 / value
    return None


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
    ap.add_argument("--growth-prior", choices=["auto", "on", "off"], default="auto",
                    help="growth-law prior of the whole-scan refinement: auto = on for Ag on PS (held-out test passed), "
                         "on = for every system with a law in the package's reference, off = never")
    a = ap.parse_args(argv)
    fam, like = choose(a.metal, a.substrate) if not a.family else (a.family, None)
    if not a.family:
        fam = routed(fam, a.metal, _lam_of(extra))
    like = a.like or like
    a.prior_used = growth_prior_used(a.growth_prior, a.metal, a.substrate)
    if a.mode == "auto":
        a.mode = "quick" if fam.split("_")[0] in ("si", "tio2") else "refine"
        lam = _lam_of(extra)
        if a.mode == "quick" and lam is not None and REG.lam_outside(a.metal, lam, REG.load(fam)):
            a.mode = "refine"
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

    shutil.rmtree(out / "trace", ignore_errors=True)
    os.environ[LV.TRACE_ENV] = str(out / "trace")

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

    def finish(fam_, preliminary=False):
        return _finish(a, out, data, fam_, tried, like, steps, run, run_quiet, lambda: status, preliminary)

    def write_analysis(fam_, st):
        _analysis(a, out, data, fam_, tried, like, steps, st)

    have = bool(data and (data / a.label / "scan.json").exists())
    LV.say_step("import", "running")
    ok = True
    if have:
        steps.append(dict(step="intake", returncode=0, seconds=0.0, log=None, note="scan already imported: reused"))
        kept = data / f"{a.label}.trace" if data is not None else None
        for sub, name in (("import.centre", "detector.json"), ("import.mask", "detector.json"),
                          ("import.read", "reading.json"), ("import.onset", "onset.json")):
            LV.say_step(sub, "skipped", summary="imported before",
                        detail=name if kept is not None and (kept / name).exists() else None)
        LV.say_step("import", "done", summary="imported before: no frame read again")
        print("[1/4] intake ... reused the imported scan", flush=True)
    else:
        args = ["--frames", a.frames, "--label", a.label, "--metal", a.metal, "--substrate", host_of(a.metal, a.substrate),
                "--alpha-i", a.alpha_i, "--workers", a.workers]
        args += ["--rate-nm-s", a.rate_nm_s] if a.rate_nm_s is not None else ["--qcm-nm-s", a.qcm_nm_s]
        args += ["--data", data] if data is not None else []
        args += ["--shadow"] if a.substrate == "Si" and "--shadow" not in extra else []


        args += ["--windows-only", a.step] if a.mode != "full" and "--windows-only" not in extra else []


        if data is not None:
            shutil.rmtree(data / f"{a.label}.trace", ignore_errors=True)
            os.environ[LV.TRACE_ENV] = str(data / f"{a.label}.trace")
        try:
            ok = run("intake", args + extra)
        finally:
            os.environ[LV.TRACE_ENV] = str(out / "trace")
        reading = _reading_of(data, a.label)
        LV.say_step("import", "done" if ok else "failed",
                    summary=(f"{reading['frames_read']} of {reading['frames_total']} frames in {reading['seconds']:.0f} s"
                             if ok and reading else "see the log of the import" if not ok else None))
    if data is not None and (data / f"{a.label}.trace").is_dir():
        (out / "trace").mkdir(parents=True, exist_ok=True)
        for f in (data / f"{a.label}.trace").iterdir():
            shutil.copyfile(f, out / "trace" / f.name)
    tried = {}
    if ok:
        alts = ALTERNATIVES.get(fam, []) if (a.alternatives == "on" or (a.alternatives == "auto" and a.mode != "full")) and not a.family else []
        alts = same_optics(fam, alts, a.metal, scan_lam(data, a.label, extra))
        LV.say("models", "predict", families=[fam] + alts, mode=a.mode)
        LV.say_step("networks", "running")


        later = a.mode == "refine" and bool(alts)

        def predict_args(f, refine, sequence):
            args = [f, "--scan", a.label, "--out", out / f, "--times", "all", "--step", a.step, "--max-windows",
                    a.max_windows or mode["max_windows"], "--k", 3, "--workers", a.workers, "--refine", refine]
            args += ["--sequence", sequence] if sequence else []
            args += variant_args(f, sequence)
            args += ["--growth-prior", "default"] if sequence and a.prior_used else []
            args += ["--like", like] if like else []
            return args + (["--data", data] if data is not None else [])

        first = (0, 0) if later else (mode["refine"], mode["sequence"])
        for f in [fam] + alts:
            sub = out / f
            good = run("predict", predict_args(f, *first)) if f == fam else run_quiet("predict", predict_args(f, *first), f)
            if good and (sub / "summary.json").exists():
                s = json.loads((sub / "summary.json").read_text(encoding="utf-8"))
                tried[f] = next((float(s[k]) for k in ("median_dev5_sequence", "median_dev5_refined", "median_dev5_best") if s.get(k) is not None), float("inf"))
        ok = bool(tried)
        if ok:
            fam = min(tried, key=tried.get)
            LV.say("chosen", "predict", family=fam, deviations={k: LV.rounded(v, 3) for k, v in tried.items()})
            dev_text = ", ".join(f"{k} {v:.1f}" for k, v in tried.items())
            LV.say_step("networks", "done", summary=f"median fit deviation {dev_text}")
            LV.say_step("compare", "done", detail=LV.trace("compare", dict(deviations=tried, chosen=fam)),
                        summary=(f"{fam} fits better ({dev_text})" if len(tried) > 1 else f"{fam}")
                                + ("; its result is shown first" if later else ""))
            if not later and not mode["sequence"]:
                LV.say_step("joint", "skipped", summary=f"mode {a.mode}: the network's solutions and their time path")
            elif not later:
                LV.say_step("joint", "done", summary="run with the network (mode full)")
            status = "ok" if status.startswith("failed at predict") else status
            iterations = _joint_iterations(fam, mode["sequence"])
            if later:


                ok = finish(fam, preliminary=True)
                if ok:
                    LV.say("preliminary", "predict", family=fam, folder=str(out))
                LV.say("refine", "predict", family=fam, refine=mode["refine"], iterations=iterations)


                workers0 = a.workers
                a.workers = _workers_for_memory(a.workers)
                if a.workers < workers0:
                    print(f"      whole-scan refinement with {a.workers} processes (free memory)", flush=True)
                LV.say_step("joint", "running")
                ok_joint = run_quiet("predict", predict_args(fam, mode["refine"], iterations) + ["--no-network-step", "--reuse"], "refine")
                if not ok_joint and a.workers > 1:
                    a.workers = max(1, a.workers // 2)
                    print(f"      whole-scan refinement failed: once more with {a.workers} processes", flush=True)
                    ok_joint = run_quiet("predict", predict_args(fam, mode["refine"], iterations) + ["--no-network-step", "--reuse"], "refine")
                a.joint_note = dict(done=bool(ok_joint), workers=a.workers, iterations=iterations)
                LV.say_step("joint", "done" if ok_joint else "failed",
                            summary=(f"{iterations} iterations with {a.workers} processes" if ok_joint else
                                     "did not finish (see the log of this step): the network's solutions are shown"))
                a.workers = workers0
    if ok:
        finish(fam)
    else:
        write_analysis(fam, status)
    LV.clear(live)
    return 0 if status == "ok" else 1


JOINT_WORKER_BYTES, JOINT_KEEP_BYTES = 0.35e9, 1.0e9


def _available_bytes():
    'Physical memory available now (bytes), or None when it cannot be read (psutil, else the OS directly).'
    try:
        import psutil
        return float(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return float(m.ullAvailPhys)
        else:
            for line in open("/proc/meminfo", encoding="ascii"):
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024.0
    except Exception:
        pass
    return None


def _workers_for_memory(workers):
    'At most the processes the free memory holds for the whole-scan refinement (JOINT_WORKER_BYTES each beyond'
    avail = _available_bytes()
    if avail is None:
        return int(workers)
    return max(1, min(int(workers), int((avail - JOINT_KEEP_BYTES) // JOINT_WORKER_BYTES)))


def _joint_iterations(fam, default):
    "Iterations of the whole-scan refinement for the family's shipped network: its meta 'joint_iterations' when set"
    try:
        import numpy as np
        path = Path(os.environ.get("IAMPRED_MODELS", "")) / fam / "nn" / "model.npz"
        if os.environ.get("IAMPRED_MODELS") and path.is_file():
            with np.load(path, allow_pickle=False) as z:
                n = json.loads(str(z["meta"])).get("joint_iterations")
            if n:
                return int(n)
    except (OSError, ValueError, KeyError):
        pass
    return default


def _reading_of(data, label):
    "The import's reading summary from the scan's marker (None when absent)."
    marker = data / label / "pixmask.json" if data is not None else None
    try:
        return json.loads(marker.read_text(encoding="utf-8")).get("reading") if marker is not None else None
    except (OSError, ValueError):
        return None


def _finish(a, out, data, fam, tried, like, steps, run, run_quiet, status_of, preliminary=False):
    "Copies the chosen network's tables, runs report and present (the main-screen data) and writes analysis.json;"
    for v in ("windows.csv", "summary.json", "evolution.png", "patterns.png", "sequence_record.json"):
        if (out / fam / v).exists():
            shutil.copyfile(out / fam / v, out / v)
    if not preliminary:
        LV.say_step("finish", "running")
        LV.say_step("finish.report", "running")
    ok = run("report", [out])
    if not preliminary:
        LV.say_step("finish.report", "done" if ok else "failed")
        LV.say_step("finish.present", "running")
    if ok:
        print("[4/4] present ...", flush=True)
        args = [out / fam, "--k", 3, "--workers", a.workers] + (["--data", data] if data is not None else [])
        drawn = run_quiet("present", args, "main") and (out / fam / "main" / "main.json").exists()
        if drawn:
            shutil.copytree(out / fam / "main", out / "main", dirs_exist_ok=True)
        if not preliminary:
            LV.say_step("finish.present", "done" if drawn else "failed")
    if not preliminary:
        LV.say_step("finish", "done" if ok else "failed")
    _analysis(a, out, data, fam, tried, like, steps, "preliminary" if (ok and preliminary) else status_of())
    return ok


def _analysis(a, out, data, fam, tried, like, steps, status):
    files = {k: (str(out / v) if (out / v).exists() else None) for k, v in FILES.items()}
    res = dict(status=status, label=a.label, metal=a.metal, substrate=a.substrate, host=host_of(a.metal, a.substrate), family=fam, families_tried=tried, like=like, mode=a.mode, steps=steps, files=files,
               mask=_mask_used(data, a.label), note="development reference: not validated against independent characterisation")
    if status == "preliminary":
        res["preliminary"] = "the network's candidates and time path; the whole-scan refinement is running"


    marker = data / a.label / "pixmask.json" if data is not None else None
    if marker is not None and marker.exists():
        try:
            reading = json.loads(marker.read_text(encoding="utf-8")).get("reading")
        except (OSError, ValueError):
            reading = None
        if reading:
            res["reading"] = reading
    res["workers"] = int(a.workers)
    res["numerics"] = NUM.describe()


    gp = None
    try:
        gp = ((json.loads((out / fam / "summary.json").read_text(encoding="utf-8")).get("sequence_setup") or {})
              .get("growth_prior") or {}).get("prior")
    except (OSError, ValueError, AttributeError):
        gp = None
    res["growth_prior"] = dict(setting=a.growth_prior, requested=bool(a.prior_used and MODES[a.mode]["sequence"]),
                               used=bool(gp and gp.get("active")), prior=gp,
                               default_on=sorted(f"{m}/{s}" for m, s in PRIOR_DEFAULT_ON))


    res["stop_hint"] = SH.for_folder(out)
    if getattr(a, "joint_note", None) and status != "preliminary":
        res["joint_fit"] = a.joint_note
    (out / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"status: {status}; report: {files['report_md']}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
