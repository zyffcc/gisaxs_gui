'Bring a new in-situ scan (raw CBF frames) into the predictor: binned IAM cache + scan.json (+ Si shadow table).'

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import live as LV

HERE = Path(__file__).resolve().parent


DEFAULTS = HERE / "intake_defaults.json"
IAM = HERE.parent
MODEL = IAM.parent
LAM_12KEV = 0.103320165
HC = 1239.8419843320026


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def lit_frames(files, n=30):
    'The frames with counts among n spread over the scan (a frame taken before the beam or the shutter opened has'
    import numpy as np
    from sputtergisaxs.polymer_dev_intake import load_frame
    ids = np.unique(np.linspace(0, len(files) - 1, min(n, len(files))).round().astype(int))
    lit = []
    for i in ids:
        try:
            image, _ = load_frame(files[int(i)])
        except Exception:
            continue
        if float(np.nansum(image)) > 0:
            lit.append(files[int(i)])
    return lit or files


DARK_RAMP_SHARE, BASE_S, RISE = 0.9, 0.3, 1.05
"""A scan whose first bins have no counts (the X-ray shutter opened after the detector started): the first lit bin whose
rate reaches 90 % of the next three is the start of the bare plateau; the onset is the first bin after BASE_S seconds of
plateau where four bins in a row exceed 1.05 x its median rate - the forward model's own rule (tio2_explore.onset), with
the baseline on lit bins instead of 0.3-1.0 s."""


def dark_start_onset(scan_dir):
    '(t_dep, pre_start, pre_end, shutter_open) from the binned cache of a scan that starts with dark bins after 0.3 s,'
    import numpy as np
    root = Path(scan_dir)
    grid = root.parent / "native_grid.npz"
    if not grid.exists():
        return None
    g = np.load(grid)
    af = 0.5 * (g["native_edges"][1:] + g["native_edges"][:-1])
    q = 0.5 * (g["q_edges"][1:] + g["q_edges"][:-1])
    sel = ((af >= 0.10) & (af < 0.35))[:, None] & ((np.abs(q) >= 0.15) & (np.abs(q) < 1.2))[None, :]
    times, rate, total = [], [], []
    for meta in sorted((root / "binned").glob("chunk_*.json")):
        rows = json.loads(meta.read_text(encoding="utf-8"))["rows"]
        with np.load(meta.with_suffix(".npz")) as z:
            Y, E = z["Y"].astype(float), z["E"].astype(float)
        for i, r in enumerate(rows):
            times.append(float(r["time_s"]))
            total.append(float(Y[i].sum()))
            rate.append(float((Y[i] * sel).sum() / max((E[i] * sel).sum(), 1e-12)))
        if times and times[-1] > 30.0:
            break
    times, rate, total = np.array(times), np.array(rate), np.array(total)
    lit = total > 0
    if not lit.any() or not (~lit[(times >= 0.3) & (times < times[np.argmax(lit)])]).any():
        return None
    first = int(np.argmax(lit))
    start = next((i for i in range(first, len(rate) - 3) if rate[i] >= DARK_RAMP_SHARE * np.median(rate[i + 1:i + 4])),
                 first)
    base = (times >= times[start]) & (times <= times[start] + BASE_S)
    r0 = float(np.median(rate[base]))
    above = rate > RISE * r0
    for i in range(len(rate) - 4):
        if times[i] > times[start] + BASE_S and above[i:i + 4].all():
            return float(times[i]), float(times[start]), float(times[i] - 0.05), float(times[first])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--metal", required=True, choices=["Au", "Ag", "Cu"])
    ap.add_argument("--substrate", required=True, choices=["Si", "TiO2", "PS", "PET", "PSBMA", "PNO", "PEO"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rate-nm-s", type=float)
    g.add_argument("--qcm-nm-s", type=float)
    ap.add_argument("--alpha-i", type=float, default=0.40)
    ap.add_argument("--energy-ev", type=float)
    ap.add_argument("--wavelength-nm", type=float)
    known = json.loads(DEFAULTS.read_text(encoding="utf-8")) if DEFAULTS.exists() else {}
    ap.add_argument("--sdd-mm", type=float, default=known.get("sdd_mm"))
    ap.add_argument("--beam-col", type=float, default=known.get("beam_col"))
    ap.add_argument("--beam-row", type=float, default=known.get("beam_row"))
    ap.add_argument("--auto-centre", action="store_true")
    ap.add_argument("--roll", type=float, default=known.get("roll", 0.0))
    ap.add_argument("--mask", default=str(MODEL / "assets" / "p03_2025_pixmask_v1.npy"))
    ap.add_argument("--data")
    ap.add_argument("--t-dep", type=float)
    ap.add_argument("--pre")
    ap.add_argument("--poly-t", type=float)
    ap.add_argument("--t-sput", type=float, help="deposition DURATION after the onset (s): the mass stays at rate x t_sput"
                    " afterwards (not the absolute shutter-close time)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shadow", action="store_true")
    ap.add_argument("--shadow-only", action="store_true", help="only extract the shadow table of an existing intake")
    ap.add_argument("--windows-only", type=float, default=None, metavar="STEP",
                    help="read only the frames the analysis windows of predict --times all --step STEP use (with at "
                         "most 24 and at most 60 windows: the refine and quick modes of iampred.analyze), plus the first "
                         f"{SELECT_FULL_S:g} s and the last seconds (select_frames)")
    ap.add_argument("--auto-mask", choices=["on", "off"], default="on",
                    help="mask the pixels the frames show to be unusable (dead, hot, beamstop) as well (iampred.automask)")
    a = ap.parse_args()
    sys.path.insert(0, str(IAM))
    if a.data:
        os.environ["IAM_DATA"] = str(Path(a.data).resolve())
    import iam_paths as IAMP
    sys.path.insert(0, str(IAMP.DIST))
    import iam as M

    if a.label in M.SCANS and not a.shadow_only:
        raise SystemExit(f"{a.label} is a built-in scan label (iam.SCANS): choose another --label")
    if not a.shadow_only and not Path(a.frames).is_dir():
        raise SystemExit(f"{a.frames}: no such frame folder")
    if a.energy_ev and a.wavelength_nm:
        raise SystemExit("give --energy-ev or --wavelength-nm, not both")
    if a.sdd_mm is None and not a.shadow_only:
        raise SystemExit("give the sample-detector distance --sdd-mm")
    if not a.auto_centre and not a.shadow_only and (a.beam_col is None or a.beam_row is None):
        raise SystemExit("give the beam centre --beam-col and --beam-row (pixels), or --auto-centre")
    if a.substrate not in ("Si", "TiO2") and a.poly_t is None and a.substrate not in M.POLY:
        raise SystemExit(f"{a.substrate}: give the film thickness --poly-t")
    root = IAMP.DATA.resolve()
    mask = Path(a.mask).resolve()
    msha = sha(mask)
    rm = root / "pixmask.json"
    if rm.exists():
        if json.loads(rm.read_text(encoding="utf-8"))["sha256"] != msha:
            raise SystemExit(f"{root} holds caches of another pixel mask: use another --data root")
    elif root.exists() and any(p.is_dir() for p in root.iterdir()):
        raise SystemExit(f"{root} holds caches without a mask marker: use a new, empty --data root")
    else:
        root.mkdir(parents=True, exist_ok=True)
        rm.write_text(json.dumps(dict(file=mask.name, path=str(mask), sha256=msha), indent=1) + "\n", encoding="utf-8")
    if not (root / "native_grid.npz").exists():
        shutil.copyfile(HERE / "native_grid.npz", root / "native_grid.npz")
    out = root / a.label
    if out.exists() and not a.shadow_only:
        raise SystemExit(f"{out} exists: never overwritten")
    lam = a.wavelength_nm or (HC / a.energy_ev if a.energy_ev else LAM_12KEV)
    for ref in (LAM_12KEV, 0.105):
        if abs(lam - ref) <= 1e-4 * ref:
            lam = ref
            break
    energy = HC / lam
    from . import registry as REG
    problem = REG.lam_problem(a.metal, lam)
    if problem:
        raise SystemExit(f"{a.label}: {problem}")
    if a.shadow_only:
        a.shadow = True
    else:
        _intake(a, root, mask, msha, energy, lam, out)
    if a.shadow:
        shadow(a, root)


SELECT_FULL_S, SELECT_TAIL_S, SELECT_MARGIN_S = float(os.environ.get("IAM_SELECT_FULL_S", "30")), 3.0, 0.6
SELECT_MAXW = tuple(int(v) for v in os.environ.get("IAM_SELECT_MAXW", "24,60").split(","))


SLOW_READ_S, SLOW_SHARE = 0.15, 0.5


def reading_summary(intake_dir, run, workers, t_import):
    'Frames read and the time the import spent reading them (dict for the scan marker and the live event).'
    import time as _time
    read_s = wall_s = mb = 0.0
    frames = 0
    for meta in Path(intake_dir).glob("chunk_*.json"):
        try:
            doc = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        frames += len(doc.get("rows", []))
        read_s += float(doc.get("read_seconds") or 0.0)
        wall_s += float(doc.get("wall_seconds") or 0.0)
        mb += float(doc.get("read_bytes") or 0) / 1e6
    per = read_s / frames if frames else None
    share = read_s / wall_s if wall_s > 0 else None
    out = dict(frames_read=frames, frames_total=run.get("received_frames"), workers=int(workers),
               seconds=round(_time.perf_counter() - t_import, 1), read_s_per_frame=None if per is None else round(per, 4),
               read_share=None if share is None else round(share, 3), mb_read=round(mb, 1),
               slow=bool(per is not None and share is not None and per >= SLOW_READ_S and share >= SLOW_SHARE))
    print(f"reading: {frames} of {out['frames_total']} frames in {out['seconds']:.0f} s with {workers} processes; disk "
          f"{out['read_s_per_frame']} s per frame ({100 * (share or 0):.0f} % of the processes' time)"
          + (": the disk is the slow part" if out["slow"] else ""), flush=True)
    LV.emit("reading", **out)
    return out


def _windows(t_dep, t_first, t_last, step, nmax):
    'predict._all_times with the first and last bin time of the scan.'
    import numpy as np
    t_end = t_last - 0.5
    t0 = max(t_dep + 1.0, t_first + 0.5)
    tt = np.arange(np.ceil(t0), t_end, step)
    if nmax and len(tt) > nmax:
        tt = np.arange(np.ceil(t0), t_end, step * int(np.ceil(len(tt) / nmax)))
    return [float(t) for t in tt]


def _bins_and_onsets(out, t_dep_given):
    '(first and last bin time, onset candidates, selection indices, header times) of the import read so far: its bins'
    import numpy as np
    import iam as M
    from tio2_explore import onset
    from sputtergisaxs.polymer_review_data import intake_rows, bin_ids, aggregate
    run, rows = intake_rows(out / "intake")
    shutil.rmtree(out / "binned", ignore_errors=True)
    aggregate(out / "intake", out / "binned", run, rows, bin_ids([r["time_s"] for r in rows], seconds=0.5), nominal_seconds=0.5)
    try:
        grid = M.scan_grid()
        res = M.Result(out, grid)
        t_on = onset(res, grid)[0]
        dark = dark_start_onset(out)
        times = np.asarray(res.time, float)
    finally:
        shutil.rmtree(out / "binned", ignore_errors=True)
    cands = sorted({float(t) for t in (t_on, dark[0] if dark else None, t_dep_given) if t is not None})
    return (float(times[0]), float(times[-1]), cands, np.asarray(run["selection_indices"], int),
            np.array([r["time_s"] for r in rows], float))


def select_frames(a, out, read):
    'Reads the frames --windows-only needs (see above); returns what was read, or None when every frame was read.'
    import numpy as np
    from sputtergisaxs.polymer_dev_intake import frames, load_frame
    n = len(frames(Path(a.frames)))
    _, meta = load_frame(frames(Path(a.frames))[0])
    dt = float(meta.get("period_s") or meta["exposure_s"])
    head = min(n, int(np.ceil((SELECT_FULL_S / dt + 1) / 128.0)) * 128)
    tail = int(np.ceil(SELECT_TAIL_S / dt)) + 1
    if head + tail >= 0.8 * n:
        read()
        return None
    sel = set(range(head)) | set(range(n - tail, n))
    for rnd in range(4):
        read(sorted(sel))
        t_first, t_last, cands, idx, ts = _bins_and_onsets(out, a.t_dep)
        full_end = ts[idx < head].max()
        if not cands or max(cands) + 3.0 > full_end:
            print(f"frames: the onset is not within the first {full_end:.1f} s read in full: reading every frame", flush=True)
            read()
            return None
        wins = sorted({t for c in cands for nm in SELECT_MAXW for t in _windows(c, t_first, t_last, a.windows_only, nm)})
        add = set()
        for t in wins:
            lo, hi = t - 0.5 - SELECT_MARGIN_S, t + 0.5 + SELECT_MARGIN_S
            before, after = idx[ts <= lo], idx[ts >= hi]
            if not len(before) and idx[0] == 0:
                before = idx[:1]
            if not len(after) and idx[-1] == n - 1:
                after = idx[-1:]
            if len(before) and len(after):
                i0, i1 = before.max(), after.min()
                if np.isin(np.arange(i0, i1 + 1), idx).all():
                    continue
            j0 = int(np.floor(np.interp(lo, ts, idx))) - 3
            j1 = int(np.ceil(np.interp(hi, ts, idx))) + 3
            add |= set(range(max(j0, 0), min(j1, n - 1) + 1))
        if not add - sel:
            if add:
                break
            done = dict(frames_read=len(sel), frames=n, windows=len(wins), onsets=cands, step=a.windows_only,
                        max_windows=list(SELECT_MAXW), rounds=rnd + 1)
            print(f"frames: {len(sel)} of {n} read for {len(wins)} windows (onset {', '.join(f'{c:.2f}' for c in cands)} s)",
                  flush=True)
            return done
        sel |= add
    print("frames: the windows could not be covered by a selection: reading every frame", flush=True)
    read()
    return None


def _detector_event(total, keep, auto, geo, a, binning=4):
    "The live view's picture of the detector: the summed frames of the automatic mask and what the mask leaves out,"
    from . import automask as AM
    image, masked = LV.binned_log(total, keep, binning)
    at = lambda v: LV.rounded((float(v) + 0.5) / binning - 0.5, 2)
    fields = dict(binning=binning, col=at(geo["beam_col"]), row=at(geo["beam_row"]),
                  horizon_row=at(AM.horizon_row(geo["beam_row"], a.sdd_mm, a.alpha_i)),
                  **{k: auto.get(k) for k in ("frames_summed", "dead", "hot", "shadow", "kept", "pixels")})
    LV.emit("detector", dict(image=image, masked=masked), **fields)
    LV.trace("detector", fields, dict(image=image, masked=masked))


def _onset_step(out, scan, a):
    "Step import.onset with its detail: the rate of the onset rule's band in every bin, the rule's baseline and the"
    try:
        import iam as M
        from tio2_explore import onset
        grid = M.scan_grid()
        res = M.Result(out, grid)
        t_on, r0, rate = onset(res, grid)
        used = scan.get("t_dep", t_on)
        how = ("given" if a.t_dep is not None else "dark first frames: from the lit bare plateau" if "t_dep" in scan
               else "the rate rises above 1.05 x the 0.3-1.0 s baseline")
        detail = LV.trace("onset", dict(times=[float(t) for t in res.time], rate=[float(r) for r in rate],
                                       baseline=float(r0), base_window=[0.3, 1.0], onset_rule=t_on, onset=used, how=how,
                                       pre_window=[scan.get("pre_start", 0.3), scan.get("pre_end", (used or 1.05) - 0.05)]))
        LV.step("import.onset", "done", detail=detail,
                summary=(f"{used:.2f} s ({how})" if used is not None else "not found in the frames"))
    except Exception as e:
        LV.step("import.onset", "done", summary=f"found when the windows are made ({type(e).__name__})")


def _intake(a, root, mask, msha, energy, lam, out):
    from sputtergisaxs.polymer_dev_intake import prepare, frames
    from sputtergisaxs.polymer_review_data import intake_rows, bin_ids, aggregate
    geo = dict(energy_ev=energy, sdd_mm=a.sdd_mm, alpha_i_deg=a.alpha_i, metal=a.metal, sample_roll_deg=a.roll)
    LV.step("import.centre", "running")
    if a.auto_centre:
        from sputtergisaxs import polymer_dev_intake as PDI
        PDI.RAW_CACHE_N = 30
        try:
            cen = PDI.auto_center(lit_frames(frames(Path(a.frames))), a.sdd_mm, a.alpha_i)
        finally:
            PDI.RAW_CACHE_N = 0
            PDI._RAW_CACHE.clear()
        geo.update(beam_col=float(cen["db_col"]), beam_row=float(cen["db_row"]))
        print(f"auto centre: col {cen['db_col']:.1f} row {cen['db_row']:.1f} (roll kept at {a.roll})")
        LV.emit("centre", col=LV.rounded(cen["db_col"], 2), row=LV.rounded(cen["db_row"], 2))
        LV.step("import.centre", "done", summary=f"found in the frames: column {cen['db_col']:.1f}, row {cen['db_row']:.1f}",
                detail="detector.json" if a.auto_mask == "on" else None)
    else:
        geo.update(beam_col=a.beam_col, beam_row=a.beam_row)
        LV.step("import.centre", "done", summary=f"given: column {a.beam_col:.1f}, row {a.beam_row:.1f}")
    LV.step("import.mask", "running")
    t0 = datetime.now(timezone.utc)
    used, auto = mask, None
    if a.auto_mask == "on":
        import numpy as np
        from . import automask as AM
        picture = {}
        combined, auto = AM.build(a.frames, mask, geo["beam_row"], a.sdd_mm, a.alpha_i,
                                  on_sum=(lambda total: picture.update(total=total)) if (LV.folder() or LV.trace_dir()) else None)
        out.mkdir(parents=True, exist_ok=True)
        used = out / "pixmask_used.npy"
        np.save(used, combined)
        auto.update(file=used.name, sha256=sha(used))
        print(f"automatic mask: {auto['dead']} dead, {auto['hot']} hot, {auto['shadow']} beamstop / holder pixels "
              f"masked besides {mask.name}" + (f" ({auto['base']})" if auto.get("base") else ""))
        if picture:
            try:
                _detector_event(picture.pop("total"), combined, auto, geo, a)
            except Exception:
                pass
        LV.step("import.mask", "done", detail="detector.json",
                summary=f"{auto['dead']} dead, {auto['hot']} hot, {auto['shadow']} beamstop / holder pixels besides {mask.name}")
    else:
        LV.step("import.mask", "done", summary=f"{mask.name} only (no automatic mask)")

    def read(indices=None):
        n_read = len(frames(Path(a.frames))) if indices is None else len(indices)
        with LV.ChunkWatcher(out / "intake", root / "native_grid.npz", n_read):
            for attempt in range(3):
                try:
                    prepare(Path(a.frames), out / "intake", **geo, maskpath=used, workers=a.workers,
                            system_note=f"iampred intake {a.label}", indices=indices)
                    return
                except OSError as e:
                    if attempt == 2:
                        raise
                    print(f"reading the frames failed ({type(e).__name__}: {e}); trying again in 20 s", flush=True)
                    import time
                    time.sleep(20)

    import time as _time
    t_import = _time.perf_counter()
    selection = None
    LV.step("import.read", "running")
    if a.windows_only:
        selection = select_frames(a, out, read)
    else:
        read()
    run, rows = intake_rows(out / "intake")
    ids = bin_ids([r["time_s"] for r in rows], seconds=0.5)
    aggregate(out / "intake", out / "binned", run, rows, ids, nominal_seconds=0.5)
    b = json.loads((out / "binned" / "run.json").read_text(encoding="utf-8"))
    reading = reading_summary(out / "intake", run, a.workers, t_import)
    sel = run.get("selection_indices") or []
    ranges, start = [], None
    for i, v in enumerate(sel):
        if start is None:
            start = v
        if i + 1 == len(sel) or sel[i + 1] != v + 1:
            ranges.append([start, v]); start = None
    LV.step("import.read", "done", detail=LV.trace("reading", dict(reading, selection=selection, frames_read_ranges=ranges)),
            summary=(f"{reading['frames_read']} of {reading['frames_total']} frames in {reading['seconds']:.0f} s, "
                     f"{len(set(ids.tolist()))} bins; disk {reading['read_s_per_frame']} s per frame"
                     + (" (the disk is the slow part)" if reading.get("slow") else "")))
    marker = dict(file=mask.name, path=str(mask), sha256=msha, label=a.label, source=str(Path(a.frames).resolve()),
                  alpha_i_deg=a.alpha_i, energy_ev=energy, metal=a.metal, builder="iampred.intake",
                  started_utc=t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  built_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), frames=run["selected_frames"],
                  received_frames=run["received_frames"], time_first_s=rows[0]["time_s"], time_last_s=rows[-1]["time_s"],
                  geometry=b.get("geometry"), auto_mask=auto, selection=selection,
                  reading=reading)
    (out / "pixmask.json").write_text(json.dumps(marker, indent=1) + "\n", encoding="utf-8", newline="\n")
    qcm = a.qcm_nm_s if a.qcm_nm_s is not None else a.rate_nm_s / 0.6
    scan = dict(metal=a.metal, sub=a.substrate, qcm=qcm, growth=a.label, ai=a.alpha_i, lam=lam,
                source="iampred.intake", rate_note=("QCM reading" if a.qcm_nm_s is not None else "rate at the sample / 0.6"))
    if a.t_dep is not None:
        scan["t_dep"] = a.t_dep
    if a.pre:
        scan["pre_start"], scan["pre_end"] = (float(v) for v in a.pre.split(","))
    if a.t_dep is None and not a.pre:
        dark = dark_start_onset(out)
        if dark:
            scan.update(t_dep=dark[0], pre_start=dark[1], pre_end=dark[2])
            print(f"the first frames are dark (X-ray from {dark[3]:.2f} s): bare plateau from {dark[1]:.2f} s, "
                  f"onset {dark[0]:.2f} s")
    if a.poly_t is not None:
        scan["poly_t"] = a.poly_t
    if a.t_sput is not None:
        scan["t_sput"] = a.t_sput
    (out / "scan.json").write_text(json.dumps(scan, indent=1) + "\n", encoding="utf-8", newline="\n")
    _onset_step(out, scan, a)
    print(f"{a.label}: {run['selected_frames']} frames, {rows[0]['time_s']:.2f}-{rows[-1]['time_s']:.2f} s -> {out}")
    LV.emit("imported", frames=run["selected_frames"], t_first=LV.rounded(rows[0]["time_s"], 3),
            t_last=LV.rounded(rows[-1]["time_s"], 3), onset=LV.rounded(scan.get("t_dep"), 3))


def shadow(a, root):
    'Near-horizon shadow table of every window (Si families with the pre0 background term).'
    env = dict(os.environ, IAM_DATA=str(root))
    env.pop("IAM_SHADOW_BG", None)
    code = ("import sys; sys.path.insert(0, '.'); from iampred import adapter as A; A.register(sys.argv[1]); "
            "import shadow_rows as S; S.extract(sys.argv[1], 'all', sys.argv[2])")
    r = subprocess.run([sys.executable, "-c", code, a.label, str(root)], cwd=str(IAM), env=env)
    if r.returncode:
        raise SystemExit("shadow table extraction failed")


if __name__ == "__main__":
    main()
