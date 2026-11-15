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
    if a.shadow_only:
        a.shadow = True
    else:
        _intake(a, root, mask, msha, energy, lam, out)
    if a.shadow:
        shadow(a, root)


def _intake(a, root, mask, msha, energy, lam, out):
    from sputtergisaxs.polymer_dev_intake import prepare
    from sputtergisaxs.polymer_review_data import intake_rows, bin_ids, aggregate
    geo = dict(energy_ev=energy, sdd_mm=a.sdd_mm, alpha_i_deg=a.alpha_i, metal=a.metal, sample_roll_deg=a.roll)
    if a.auto_centre:
        from sputtergisaxs.polymer_dev_intake import auto_center, frames
        cen = auto_center(frames(Path(a.frames)), a.sdd_mm, a.alpha_i)
        geo.update(beam_col=float(cen["db_col"]), beam_row=float(cen["db_row"]))
        print(f"auto centre: col {cen['db_col']:.1f} row {cen['db_row']:.1f} (roll kept at {a.roll})")
    else:
        geo.update(beam_col=a.beam_col, beam_row=a.beam_row)
    t0 = datetime.now(timezone.utc)
    used, auto = mask, None
    if a.auto_mask == "on":
        import numpy as np
        from . import automask as AM
        combined, auto = AM.build(a.frames, mask, geo["beam_row"], a.sdd_mm, a.alpha_i)
        out.mkdir(parents=True, exist_ok=True)
        used = out / "pixmask_used.npy"
        np.save(used, combined)
        auto.update(file=used.name, sha256=sha(used))
        print(f"automatic mask: {auto['dead']} dead, {auto['hot']} hot, {auto['shadow']} beamstop / holder pixels "
              f"masked besides {mask.name}" + (f" ({auto['base']})" if auto.get("base") else ""))
    prepare(Path(a.frames), out / "intake", **geo, maskpath=used, workers=a.workers, system_note=f"iampred intake {a.label}")
    run, rows = intake_rows(out / "intake")
    ids = bin_ids([r["time_s"] for r in rows], seconds=0.5)
    aggregate(out / "intake", out / "binned", run, rows, ids, nominal_seconds=0.5)
    b = json.loads((out / "binned" / "run.json").read_text(encoding="utf-8"))
    marker = dict(file=mask.name, path=str(mask), sha256=msha, label=a.label, source=str(Path(a.frames).resolve()),
                  alpha_i_deg=a.alpha_i, energy_ev=energy, metal=a.metal, builder="iampred.intake",
                  started_utc=t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  built_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), frames=run["selected_frames"],
                  received_frames=run["received_frames"], time_first_s=rows[0]["time_s"], time_last_s=rows[-1]["time_s"],
                  geometry=b.get("geometry"), auto_mask=auto)
    (out / "pixmask.json").write_text(json.dumps(marker, indent=1) + "\n", encoding="utf-8", newline="\n")
    qcm = a.qcm_nm_s if a.qcm_nm_s is not None else a.rate_nm_s / 0.6
    scan = dict(metal=a.metal, sub=a.substrate, qcm=qcm, growth=a.label, ai=a.alpha_i, lam=lam,
                source="iampred.intake", rate_note=("QCM reading" if a.qcm_nm_s is not None else "rate at the sample / 0.6"))
    if a.t_dep is not None:
        scan["t_dep"] = a.t_dep
    if a.pre:
        scan["pre_start"], scan["pre_end"] = (float(v) for v in a.pre.split(","))
    if a.poly_t is not None:
        scan["poly_t"] = a.poly_t
    if a.t_sput is not None:
        scan["t_sput"] = a.t_sput
    (out / "scan.json").write_text(json.dumps(scan, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{a.label}: {run['selected_frames']} frames, {rows[0]['time_s']:.2f}-{rows[-1]['time_s']:.2f} s -> {out}")


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
