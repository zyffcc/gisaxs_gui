'Redraw OUTDIR/evolution.png from an existing OUTDIR/windows.csv (no model evaluation, no data needed).'

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from . import predict as PR
from . import registry as REG

FRINGE_COLS = ["obs_d_fringe_nm", "obs_fringe_contrast", "obs_fringe_status"]
GRAIN_COLS = ["obs_h_grain_nm", "obs_h_grain_contrast", "obs_h_grain_status"]
YONEDA_COLS = ["obs_yoneda_deg", "obs_yoneda_rel", "obs_yoneda_status", "obs_yoneda_grain_deg", "obs_yoneda_grain_rel",
               "obs_yoneda_grain_status"]


def lines_of(path):
    "Rows of a windows.csv as predict's line dicts: numbers as floats, empty cells left out."
    out = []
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            L = {}
            for k, v in r.items():
                if v is None or v == "":
                    continue
                try:
                    L[k] = float(v)
                except ValueError:
                    L[k] = v
            out.append(L)
    return out


def add_fringe(d, s, cfg, data=None, shadow=None):
    'Read observe.fringe (low q and grain scale) for every window of d/windows.csv (scan and forward settings of'
    import numpy as np
    from . import adapter as A, observe as OB, runner
    site = runner.site_env(cfg, data, shadow)
    scan = s["scan"]
    if scan not in cfg["scans"]:
        cfg = dict(cfg); cfg["scans"] = dict(cfg["scans"])
        cfg["scans"][scan], _ = PR._client_entry(cfg, argparse.Namespace(scan=scan, like=s.get("like")),
                                                 Path(site.get("IAM_DATA", ".")))
    env = dict(REG.scan_env(cfg, scan)); env.update(site)
    A.activate(env)
    import iam as M
    sc = A.get_scan(scan)
    eps = float(os.environ.get("IAM_EPS", "0"))
    sin2_ac = 2.0 * float(np.real(M.dc(str(sc.metal))))
    with open(d / "windows.csv", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        header, rows = list(rd.fieldnames), list(rd)
    if FRINGE_COLS[0] not in header:
        i = header.index("obs_knee_status") + 1 if "obs_knee_status" in header else header.index("n_cells") + 1
        header[i:i] = FRINGE_COLS
    if GRAIN_COLS[0] not in header:
        i = header.index(FRINGE_COLS[-1]) + 1
        header[i:i] = GRAIN_COLS
    header = [c for c in header if c not in YONEDA_COLS]
    i = header.index(GRAIN_COLS[-1]) + 1
    header[i:i] = YONEDA_COLS
    n = k = kg = 0
    for r in rows:
        for c in FRINGE_COLS + GRAIN_COLS + YONEDA_COLS:
            r[c] = ""
        if not r.get("h_nom") or not r.get("t"):
            continue
        w = sc.window(float(r["t"]))
        if not float(np.sum(w["E"])) > 0:
            continue
        of = OB.fringe(sc, w, float(r["h_nom"]), sin2_ac, float(M.K0), eps)
        n += 1
        r["obs_fringe_status"] = of["status"]
        if of.get("d_nm"):
            r["obs_d_fringe_nm"] = round(of["d_nm"], 2); r["obs_fringe_contrast"] = of["contrast"]
            k += of["status"] == "fringe"
        og = OB.fringe(sc, w, float(r["h_nom"]), sin2_ac, float(M.K0), eps, q_band=OB.GRAIN_Q)
        r["obs_h_grain_status"] = og["status"]
        if og.get("d_nm"):
            r["obs_h_grain_nm"] = round(og["d_nm"], 2); r["obs_h_grain_contrast"] = og["contrast"]
            kg += og["status"] == "fringe"
        oy = OB.yoneda(sc, w, sin2_ac, eps); oyg = OB.yoneda(sc, w, sin2_ac, eps, q_band=OB.GRAIN_Q)
        r["obs_yoneda_status"], r["obs_yoneda_grain_status"] = oy["status"], oyg["status"]
        if oy.get("af_deg"):
            r["obs_yoneda_deg"] = round(oy["af_deg"], 4); r["obs_yoneda_rel"] = round(oy["rel"], 3)
        if oyg.get("af_deg"):
            r["obs_yoneda_grain_deg"] = round(oyg["af_deg"], 4); r["obs_yoneda_grain_rel"] = round(oyg["rel"], 3)
    with open(d / "windows.csv", "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    return n, k, kg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--fringe", action="store_true")
    ap.add_argument("--data")
    ap.add_argument("--shadow")
    a = ap.parse_args()
    dirs = [Path(d) for d in a.dirs]
    if a.fringe and len(dirs) != 1:
        raise SystemExit("--fringe: one OUTDIR per call (the forward settings of a scan are fixed per process)")
    for d in dirs:
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        cfg = REG.load(s["family"])
        if a.fringe:
            n, k, kg = add_fringe(d, s, cfg, a.data, a.shadow)
            print(f"{s['scan']}: fringe read in {n} windows, film thickness in {k}, grain height in {kg}")
        PR.evolution_figure(cfg, s["scan"], d, lines_of(d / "windows.csv"), cfg["reported"])
        print("wrote", d / "evolution.png")


if __name__ == "__main__":
    main()
