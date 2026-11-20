'Series joint fit of after-deposition (static) samples: the samples of a thickness series as one growing film.'

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RULES = (("ln_R", 0.10), ("ln_D", 0.05), ("t_b", 0.10), ("height", 0.10), ("ln_hm", 0.03), ("eta", 0.03))
"""Growth quantities of static_nn.REPORTED that may not decrease with the nominal thickness, with their scales."""
ETA_CLOSED = 0.9
"""Above this grain coverage the coverage rule is not applied (a closed film: the grains merge into the base)."""


def _rows(out: Path, label: str) -> list:
    return [r for r in csv.DictReader(open(out / label / "frames.csv", encoding="utf-8", newline="")) if r["label"] == label]


RECORD_DIRS = ("static_fits", "static_fits")
"""Folder of a sample's per-frame records: the research name, and the name of the released package."""


def _records(out: Path, label: str) -> Path:
    for name in RECORD_DIRS:
        if (out / label / name / label).is_dir():
            return out / label / name
    raise SystemExit(f"static_series: no per-frame records of {label} under {out / label}")


def _record(out: Path, label: str, frame: int) -> dict:
    return json.loads((_records(out, label) / label / f"{label}_{frame:03d}.json").read_text(encoding="utf-8"))


class Sample:
    'The frames used for one sample, their fitters, and the split of a parameter vector into morphology (shared by'

    def __init__(self, out: Path, label: str, n_frames: int, rb_late: float):
        import iam_static as IS
        import static_nn as SN
        self.label = label
        rows = sorted(_rows(out, label), key=lambda r: float(r["dev5_refined"]))[:max(n_frames, 1)]
        self.frames = [int(r["frame"]) for r in rows]
        self.scan = IS.StaticScan(label)
        mode = "analytic" if self.scan.ref == "analytic_si" else "ref"
        self.fits = [IS.StaticFit(self.scan, i, mode) for i in self.frames]
        self.nominal = float(self.scan.cfg["nominal_nm"])
        for f in self.fits:
            if rb_late > 0 and self.nominal >= 8 and "rb" in f.names:
                j = f.names.index("rb")
                f.lb[j] = max(f.lb[j], min(rb_late, f.ub[j] - 1e-6))
        names = self.fits[0].names
        self.nuis = [n for n in names if n in SN.SUBS]
        self.morph = [n for n in names if n not in SN.SUBS]
        starts = [_record(out, label, i)["params"] for i in self.frames]
        best = starts[0]
        self.x = np.concatenate([[best[n] for n in self.morph]] + [[st[n] for n in self.nuis] for st in starts]).astype(float)
        f0 = self.fits[0]
        lb = dict(zip(names, f0.lb)); ub = dict(zip(names, f0.ub))
        self.lb = np.concatenate([[lb[n] for n in self.morph]] + [[lb[n] for n in self.nuis] for _ in self.fits]).astype(float)
        self.ub = np.concatenate([[ub[n] for n in self.morph]] + [[ub[n] for n in self.nuis] for _ in self.fits]).astype(float)
        self.x = np.clip(self.x, self.lb + 1e-9, self.ub - 1e-9)

    def frame_x(self, x: np.ndarray, k: int) -> np.ndarray:
        'The fitter vector of frame k (fitter name order) from a sample vector.'
        f = self.fits[k]
        nm, nn = len(self.morph), len(self.nuis)
        d = dict(zip(self.morph, x[:nm]))
        d.update(zip(self.nuis, x[nm + k * nn: nm + (k + 1) * nn]))
        return np.array([d[n] for n in f.names], float)

    def growth(self, x: np.ndarray) -> np.ndarray:
        'static_nn.REPORTED of the sample vector (frame 0), the growth quantities.'
        import static_nn as SN
        f = self.fits[0]
        (_, _, _), m, p = f.model(self.frame_x(x, 0))
        return SN.reported(f, m, p)

    def residual(self, x: np.ndarray) -> np.ndarray:
        "Frames' residual rows scaled to dev5 per cell, averaged over the frames."
        out = []
        for k, f in enumerate(self.fits):
            r = f.resid(self.frame_x(x, k))
            out.append(r / math.sqrt(max(f.ok.sum(), 1) * len(self.fits)))
        return np.concatenate(out)

    def dev5(self, x: np.ndarray) -> float:
        import static_nn as SN
        return float(np.mean([SN.dev5_of(f, self.frame_x(x, k)) for k, f in enumerate(self.fits)]))


def _index():
    import static_nn as SN
    return {n: SN.REPORTED.index(n) for n, _ in RULES}


def rule_rows(q_prev, q, q_next, weight: float) -> np.ndarray:
    "No-decrease rows of a sample's growth quantities q against its neighbours (None at the ends)."
    ix, rows = _index(), []
    sw = math.sqrt(weight)
    for name, scale in RULES:
        j = ix[name]
        for a, b in ((q_prev, q), (q, q_next)):
            if a is None or b is None:
                continue
            if name == "eta" and min(a[j], b[j]) >= ETA_CLOSED:
                rows.append(0.0)
                continue
            rows.append(sw * max(a[j] - b[j], 0.0) / scale)
    return np.asarray(rows, float)


def decreases(qs: list) -> dict:
    'Per rule: how many neighbouring pairs decrease by more than one scale.'
    ix = _index()
    return {name: int(sum(1 for a, b in zip(qs, qs[1:]) if a is not None and b is not None and a[ix[name]] - b[ix[name]] > scale
                          and not (name == "eta" and min(a[ix[name]], b[ix[name]]) >= ETA_CLOSED)))
            for name, scale in RULES}


_W = {}


def _work_init(out, labels, n_frames, rb_late):
    _W["samples"] = {lab: Sample(Path(out), lab, n_frames, rb_late) for lab in labels}


def _refit(args):
    'One sample against its frames and the rules to its neighbours (their growth quantities fixed).'
    from scipy.optimize import least_squares
    label, x0, q_prev, q_next, weight, nfev = args
    s = _W["samples"][label]

    def res(x):
        return np.r_[s.residual(x), rule_rows(q_prev, s.growth(x), q_next, weight)]

    r = least_squares(res, np.clip(x0, s.lb + 1e-9, s.ub - 1e-9), bounds=(s.lb, s.ub), x_scale="jac", diff_step=1e-3,
                      max_nfev=nfev)
    return label, r.x, s.dev5(r.x), s.growth(r.x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default="", help="samples of the series (default: every sample in OUT with frames.csv)")
    ap.add_argument("--frames", type=int, default=1, help="frames per sample (the best refined ones)")
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--nfev", type=int, default=6, help="least-squares steps per sample and sweep")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--weight", type=float, default=1.0, help="weight of the no-decrease rows (dev5 units)")
    ap.add_argument("--rb-late", type=float, default=0.8)
    a = ap.parse_args(argv)
    out = Path(a.out)
    labels = [x for x in a.labels.split(",") if x] or sorted(p.name for p in out.iterdir() if (p / "frames.csv").is_file())
    t0 = time.time()
    _work_init(out, labels, a.frames, a.rb_late)
    samples = sorted(_W["samples"].values(), key=lambda s: (s.nominal, s.label))
    if len({s.scan.ref == "analytic_si" for s in samples}) > 1:
        raise SystemExit("static_series: some samples have a bare-substrate reference and some not - one series per kind")
    order = [s.label for s in samples]
    xs = {s.label: s.x.copy() for s in samples}
    q = {s.label: s.growth(xs[s.label]) for s in samples}
    d0 = {s.label: s.dev5(xs[s.label]) for s in samples}
    before = decreases([q[lab] for lab in order])
    print(f"series {order}: dev5 {[round(d0[lab], 2) for lab in order]}, decreases {before}", flush=True)
    dev = dict(d0)
    for sweep in range(a.sweeps):
        for parity in (0, 1):
            jobs = []
            for k, lab in enumerate(order):
                if k % 2 != parity:
                    continue
                q_prev = q[order[k - 1]] if k > 0 else None
                q_next = q[order[k + 1]] if k + 1 < len(order) else None
                jobs.append((lab, xs[lab], q_prev, q_next, a.weight, a.nfev))
            if not jobs:
                continue
            if a.workers > 1 and len(jobs) > 1:
                with ProcessPoolExecutor(min(a.workers, len(jobs)), initializer=_work_init,
                                         initargs=(str(out), [j[0] for j in jobs], a.frames, a.rb_late)) as ex:
                    results = list(ex.map(_refit, jobs))
            else:
                results = [_refit(j) for j in jobs]
            for lab, x, d, g in results:
                xs[lab], dev[lab], q[lab] = x, d, g
        print(f"sweep {sweep + 1}: dev5 {[round(dev[lab], 2) for lab in order]}, "
              f"decreases {decreases([q[lab] for lab in order])} ({time.time() - t0:.0f} s)", flush=True)
    after = decreases([q[lab] for lab in order])

    import iam_static as IS
    import static_nn as SN
    for s in samples:
        lab = s.label
        records = _records(out, lab)
        rec_dir = records / lab
        keep = out / lab / (records.name + "_independent")
        if not keep.exists():
            shutil.copytree(records, keep)
        if not (out / lab / "frames_independent.csv").exists():
            shutil.copy2(out / lab / "frames.csv", out / lab / "frames_independent.csv")
        rows = list(csv.DictReader(open(out / lab / "frames_independent.csv", encoding="utf-8", newline="")))
        mode = "analytic" if s.scan.ref == "analytic_si" else "ref"
        for row in rows:
            if row["label"] != lab:
                continue
            i = int(row["frame"])
            f = s.fits[s.frames.index(i)] if i in s.frames else IS.StaticFit(s.scan, i, mode)
            if i in s.frames:
                x = s.frame_x(xs[lab], s.frames.index(i))
            else:
                own = json.loads((keep / lab / f"{lab}_{i:03d}.json").read_text(encoding="utf-8"))["params"]
                d = dict(zip(s.morph, xs[lab][:len(s.morph)]))
                x = np.clip(np.array([d.get(n, own[n]) for n in f.names], float), f.lb + 1e-9, f.ub - 1e-9)
            old = json.loads((keep / lab / f"{lab}_{i:03d}.json").read_text(encoding="utf-8"))
            rec, mu, P, m, p = IS.record(f, s.scan, lab, i, x, dict(start="series", nfev=a.nfev * a.sweeps), old.get("candidates", []),
                                         t0, mode)
            rec["network"] = old.get("network")
            rec["series"] = dict(samples=order, rules=[n for n, _ in RULES], weight=a.weight, frames=s.frames)
            (rec_dir / f"{lab}_{i:03d}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
            np.savez_compressed(rec_dir / f"{lab}_{i:03d}_mu.npz", mu=mu.astype(np.float32), sub=P["sub"].astype(np.float32))
            row["dev5_refined"] = rec["dev_per_cell"]["0.05"]
            for kk in ("h_m_model", "t_b", "T", "R", "d_hex", "eta", "rb"):
                row[f"{kk}_refined"] = round(float(rec[kk]), 4)
        with open(out / lab / "frames.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    names = [n for n, _ in RULES]
    ix = _index()
    summary = dict(samples=order, nominal_nm=[s.nominal for s in samples], frames=a.frames, sweeps=a.sweeps, nfev=a.nfev,
                   weight=a.weight, rules=names, eta_closed=ETA_CLOSED, seconds=round(time.time() - t0, 1),
                   dev5_independent={lab: round(d0[lab], 4) for lab in order}, dev5_series={lab: round(dev[lab], 4) for lab in order},
                   decreases_independent=before, decreases_series=after,
                   growth_series={lab: {n: round(float(q[lab][ix[n]]), 4) for n in names} for lab in order},
                   note="development reference: the samples of the series taken as one growth path (no-decrease rules)")
    (out / "series.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("dev5_independent", "dev5_series", "decreases_independent", "decreases_series",
                                               "seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
