'Shadow rows below the sample horizon: the growing beam-path background of the in-situ scans (2026-11-08, si_horizon'

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import iam_paths as IAMP

NAME = "shadow_v1.npz"
SHADOW_ROWS = (1272, 1346)
CROP_COLS = (220, 1330)
EXCL_COLS = ((487, 494), (981, 988), (977, 993), (1022, 1038))


STOP = ()
BAND_EDGES = (-0.06, -0.10, -0.15, -0.21, -0.28)

AF_MAX = -0.03
NCOL = 5
NCOL_HI, QSPLIT = 15, 0.25
RHO_MIN = 0.03
HALO_TMIN, HALO_QMIN, HALO_NMAX = 20.0, 0.25, 4.0
TEST = ()
CAL = ()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def nbands():
    return len(BAND_EDGES) - 1


def shadow_root():
    'Folder holding <label>/shadow_v1.npz: IAM_SHADOW_DIR if set (tests, side tables), else IAM_DATA (bundled).'
    return Path(os.environ["IAM_SHADOW_DIR"]) if os.environ.get("IAM_SHADOW_DIR") else IAMP.DATA


def q_edges():
    import iam as M
    return np.asarray(M.scan_grid().q_edges, float)


def pixel_af_qy(geometry, rows, cols):
    "alpha_f (deg) and q_y (nm^-1) of detector pixels: the construction of the toolkit's AtomicMap / FeatureMap."
    g = geometry
    rr, cc = np.meshgrid(np.asarray(rows, float), np.asarray(cols, float), indexing="ij")
    L, p = float(g["sdd_mm"]), float(g["pixel_mm"])
    u = np.stack([np.full(rr.shape, L), (cc - float(g["db_col"])) * p, (float(g["db_row"]) - rr) * p])
    u /= np.linalg.norm(u, axis=0)
    ai, roll = np.radians(float(g["alpha_i_deg"])), np.radians(float(g.get("sample_roll_deg", 0.0)))
    ns = np.array([-np.sin(ai), np.cos(ai) * np.sin(roll), np.cos(ai) * np.cos(roll)])
    xs = np.array([np.cos(ai), np.sin(ai) * np.sin(roll), np.sin(ai) * np.cos(roll)])
    ys = np.cross(ns, xs)
    af = np.degrees(np.arcsin(np.clip(np.einsum("i,ijk->jk", ns, u), -1, 1)))
    k = 2 * np.pi / (1239.841984 / float(g["energy_ev"]))
    qy = np.einsum("i,ijk->jk", ys, k * (u - np.array([1.0, 0.0, 0.0])[:, None, None]))
    return af, qy


def rho_spec(af, qy, ai=0.40, energy_ev=12000.0):
    'Angular distance (deg) from the specular direction, in the sample frame: exit direction (cos af cos 2th,'
    k = 2 * np.pi / (1239.841984 / energy_ev)
    a, b = np.radians(np.asarray(af, float)), np.radians(ai)
    s2 = np.clip(np.abs(np.asarray(qy, float)) / (k * np.cos(a)), 0, 1)
    d = np.sqrt((np.cos(a) * np.sqrt(1 - s2 * s2) - np.cos(b)) ** 2 + (np.cos(a) * s2) ** 2 + (np.sin(a) - np.sin(b)) ** 2)
    return np.degrees(2 * np.arcsin(np.clip(0.5 * d, 0, 1)))


def shadow_cells(geometry, mask, qe):
    'Static shadow pixel map: (rows, cols, cell index (side, band, q) or -1, rho of every pixel).'
    rows, cols = np.arange(*SHADOW_ROWS), np.arange(*CROP_COLS)
    af, qy = pixel_af_qy(geometry, rows, cols)
    nq, nb = len(qe) - 1, nbands()
    qa = np.searchsorted(qe, np.abs(qy), side="right") - 1
    band = (nb - 1) - (np.searchsorted(np.asarray(BAND_EDGES)[::-1], af, side="right") - 1)
    ok = (qa >= 0) & (qa < nq) & (band >= 0) & (band < nb) & (af < AF_MAX) & (af >= BAND_EDGES[-1])
    ok &= np.asarray(mask, bool)[np.ix_(rows, cols)]
    for c0, c1 in EXCL_COLS:
        ok[:, (cols >= c0) & (cols < c1)] = False
    for (r0, r1), (c0, c1) in STOP:
        ok[np.ix_((rows >= r0) & (rows < r1), (cols >= c0) & (cols < c1))] = False
    side = (qy > 0).astype(int)
    cell = np.where(ok, (side * nb + np.clip(band, 0, nb - 1)) * nq + np.clip(qa, 0, nq - 1), -1)
    return rows, cols, cell, rho_spec(af, qy, float(geometry["alpha_i_deg"]), float(geometry["energy_ev"]))


def cell_rho(cell, rho, nq):
    'Mean rho of the shadow pixels of every (side, band, q) cell (nan where empty).'
    size = 2 * nbands() * nq
    c = cell.ravel(); g = c >= 0
    n = np.bincount(c[g], minlength=size).astype(float)
    s = np.bincount(c[g], weights=rho.ravel()[g], minlength=size)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan).reshape(2, nbands(), nq), n.reshape(2, nbands(), nq)


def smooth_rate(Y, E, ncol=NCOL):
    'Rate of ncol-column moving sums along the last axis (q); a column whose window has no exposure takes the'
    Y, E = np.asarray(Y, float), np.asarray(E, float)
    k = np.ones(ncol)
    Ys = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), -1, Y)
    Es = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), -1, E)
    r = np.where(Es > 0, Ys / np.where(Es > 0, Es, 1.0), np.nan)
    flat = r.reshape(-1, r.shape[-1])
    j = np.arange(r.shape[-1])
    for row in flat:
        g = np.isfinite(row)
        if g.any() and not g.all():
            gi = j[g]
            row[~g] = row[gi[np.abs(j[~g][:, None] - gi[None, :]).argmin(1)]]
    return flat.reshape(r.shape)


def window_ratio(Yw, Ew, Yp, Ep, ncol=NCOL):
    'S_w / S_pre per side and q column from (2, nband, nq) sums: all bands, ncol moving sums, floor 1.'
    rw = smooth_rate(np.sum(Yw, 1), np.sum(Ew, 1), ncol)
    rp = smooth_rate(np.sum(Yp, 1), np.sum(Ep, 1), ncol)
    return np.maximum(rw / rp, 1.0)


def smooth_rate_q(Y, E, q, ncol=NCOL, ncol_hi=NCOL_HI, qsplit=QSPLIT):
    'smooth_rate with ncol columns where |q| < qsplit and ncol_hi columns where |q| >= qsplit (last axis = q).'
    hi = np.abs(np.asarray(q, float)) >= qsplit
    return np.where(hi, smooth_rate(Y, E, ncol_hi), smooth_rate(Y, E, ncol))


def window_add(Yw, Ew, Yp, Ep, cfac, q, aB=1.0):
    'IAM_SHADOW_BG=flat (physics_20261109/si_bg): additive increment of the band next to the horizon (band 0,'
    d0 = smooth_rate_q(Yw[:, 0], Ew[:, 0], q) - cfac * smooth_rate_q(Yp[:, 0], Ep[:, 0], q)
    return (d0 if aB == 1.0 else aB * d0)[:, None, :]


def window_pre0(Yw, Ew, Yp, Ep, cfac, q, aB=1.0):
    'IAM_SHADOW_BG=pre0 (control arm): a_B (S0_w / S0_pre - cfac) of band 0, no floor (2, nq); iam4.Model4c family'
    d = smooth_rate_q(Yw[:, 0], Ew[:, 0], q) / smooth_rate_q(Yp[:, 0], Ep[:, 0], q) - cfac
    return d if aB == 1.0 else aB * d


def halo_template(Yw, Ew, Yp, Ep, rho_band, n, af, q, ai=0.40, energy_ev=12000.0, ncol=NCOL):
    'Report-only halo shape (2, n_af, nq), counts / pixel / s: the band-0 increment dS0 (>= 0) of each column'
    d0 = np.maximum(smooth_rate(Yw[:, 0], Ew[:, 0], ncol) - smooth_rate(Yp[:, 0], Ep[:, 0], ncol), 0.0)
    rho = np.maximum(rho_spec(np.asarray(af)[:, None], np.asarray(q)[None, :], ai, energy_ev), RHO_MIN)
    r0 = rho_band[:, 0, :]
    r0 = np.where(np.isfinite(r0), r0, np.nanmean(r0))
    return d0[:, None, :] * (rho[None] / r0[:, None, :]) ** (-float(n))


def halo_exponent(dS, rho_band, q, qmin=HALO_QMIN, nmax=HALO_NMAX):
    'n of dS ~ rho^-n from the shadow bands: least squares of ln dS on ln rho over the columns q >= qmin whose'
    xs, ys = [], []
    for w in range(dS.shape[0]):
        for s in range(2):
            for j in np.flatnonzero(np.asarray(q) >= qmin):
                d, r = dS[w, s, :, j], rho_band[s, :, j]
                if np.all(d > 0) and np.all(np.isfinite(r)):
                    lx, ly = np.log(r), np.log(d)
                    xs.append(lx - lx.mean()); ys.append(ly - ly.mean())
    if not xs:
        return 0.0, 0
    x, y = np.concatenate(xs), np.concatenate(ys)
    slope = float(np.sum(x * y) / max(np.sum(x * x), 1e-300))
    return float(np.clip(-slope, 0.0, nmax)), len(xs)


class Shadow:
    'The sha-checked shadow table of one scan, aligned with the bins of iam.Scan.res (matched by bin name).'

    def __init__(self, label, res, root=None):
        root = Path(root) if root else shadow_root()
        p = root / label / NAME
        side = p.with_name(NAME + ".sha256")
        if not p.exists() or not side.exists():
            raise FileNotFoundError(f"{p} (+ .sha256): run shadow_rows.py extract {label}")
        self.sha256 = sha(p)
        if side.read_text(encoding="utf-8").split()[0] != self.sha256:
            raise ValueError(f"{p}: sha256 differs from {side.name}")
        with np.load(p, allow_pickle=False) as z:
            names = [str(v) for v in z["bins"]]
            Y, E, have = z["Y"], z["E"], z["have"]
            self.rho_band, self.n_halo = z["rho_band"], float(z["n_halo"])
            self.geometry = json.loads(str(z["geometry"]))
        pos = {n: i for i, n in enumerate(names)}
        nr = len(res.rows)
        self.Y = np.zeros((nr,) + Y.shape[1:]); self.E = np.zeros((nr,) + E.shape[1:]); self.have = np.zeros(nr, bool)
        for i, r in enumerate(res.rows):
            j = pos.get(r.get("frame"))
            if j is not None and have[j]:
                self.Y[i], self.E[i], self.have[i] = Y[j], E[j], True

    def sums(self, idx):
        idx = np.asarray(idx, int)
        miss = idx[~self.have[idx]]
        if len(miss):
            raise KeyError(f"shadow table lacks {len(miss)} bins of this window (extract with --windows all or joint)")
        return self.Y[idx].sum(0), self.E[idx].sum(0)

    def window_terms(self, idx, pre_idx, mode, q, cfac=1.0, aB=1.0):
        'Window-dictionary keys of IAM_SHADOW_BG: shadow_add (2, 1, nq) for flat, shadow_pre0 (2, nq) for pre0.'
        Yw, Ew = self.sums(idx)
        Yp, Ep = self.sums(pre_idx)
        if mode == "flat":
            out = dict(shadow_add=window_add(Yw, Ew, Yp, Ep, float(cfac), q, float(aB)))
        elif mode == "pre0":
            out = dict(shadow_pre0=window_pre0(Yw, Ew, Yp, Ep, float(cfac), q, float(aB)))
        else:
            raise ValueError(f"IAM_SHADOW_BG={mode}: use flat or pre0")
        if not all(np.all(np.isfinite(v)) for v in out.values()):
            raise ValueError("shadow term not finite (a side without shadow pixels?)")
        return out


def _toolkit():
    sys.path.insert(0, str(IAMP.DIST))
    from sputtergisaxs.polymer_dev_intake import load_frame
    from sputtergisaxs.stage5_input import AtomicMap
    return load_frame, AtomicMap


def cache_inputs(label):
    'Intake run (geometry, raw folder), binned rows (bin name, time, raw members), pixel mask (sha checked).'
    d = IAMP.DATA / label
    run = json.loads((d / "intake" / "run.json").read_text(encoding="utf-8"))
    raw = Path(run["input_source"])
    if not raw.exists():
        raw = IAMP.RAW / "timescans" / raw.parent.name / "p2m"
    name = raw.parent.name.lower()
    if any(name.startswith(t) for t in TEST) or any(name.startswith(c) for c in CAL):
        raise SystemExit(f"{label}: TEST / CAL scan, not read")
    brun = json.loads((d / "binned" / "run.json").read_text(encoding="utf-8"))
    rows = []
    for c in brun["chunks"]:
        rows += json.loads((d / "binned" / (c["file"][:-4] + ".json")).read_text(encoding="utf-8"))["rows"]
    pm = json.loads((d / "pixmask.json").read_text(encoding="utf-8"))
    mp = Path(pm["path"])
    if not mp.exists():
        mp = HERE.parent / "assets" / pm["file"]
    if sha(mp) != pm["sha256"]:
        raise ValueError(f"{mp}: sha256 differs from the cache marker")
    return run, raw, rows, np.load(mp).astype(bool), pm


def select_bins(label, which, times=None, half=0.5):
    'Bin names to read: the pre-state window and the joint windows (joint_fit.WINDOWS, half width 0.5 s) as iam.Scan'
    os.environ.pop("IAM_SHADOW_BG", None)
    import iam as M
    import joint_fit as JF
    s = M.Scan(label)
    names = [r["frame"] for r in s.res.rows]
    if which == "all":
        return names, s
    idx = set(int(i) for i in s.pre_idx)
    if which == "joint":
        for t in (times if times else JF.WINDOWS[label]):
            idx |= set(np.flatnonzero((s.res.time >= t - half) & (s.res.time < t + half)).tolist())
    return [names[i] for i in sorted(idx)], s


def extract(label, which="joint", out_root=None, times=None, half=0.5):
    load_frame, AtomicMap = _toolkit()
    run, raw, rows, mask, pm = cache_inputs(label)
    g = run["geometry"]
    qe = q_edges()
    nq, nb = len(qe) - 1, nbands()
    want, s = select_bins(label, which, times, half)
    shape = mask.shape
    rr, cc, cell, rho = shadow_cells(g, mask, qe)
    rho_band, npx = cell_rho(cell, rho, nq)
    crop = [max(0, int(round(g["db_row"] - 939.0))), min(shape[0], int(round(g["db_row"] + 31.0))),
            max(0, int(round(g["db_col"] - 556.3))), min(shape[1], int(round(g["db_col"] + 553.7)))]
    amap = AtomicMap(g, shape, g["alpha_i_deg"], crop=crop, exclude_rings=((2.1, .13),))
    sub = (slice(*SHADOW_ROWS), slice(*CROP_COLS))
    cflat = cell.ravel(); cok = cflat >= 0
    size = 2 * nb * nq
    names = [r["frame"] for r in s.res.rows]
    pos = {n: i for i, n in enumerate(names)}
    meta = {r["frame"]: r for r in rows}
    Y = np.zeros((len(names), 2, nb, nq)); E = np.zeros_like(Y); have = np.zeros(len(names), bool)
    closure = np.full(len(names), np.nan); nframes = np.zeros(len(names), int)
    t0 = time.time(); nread = 0
    for nm in want:
        i = pos[nm]
        ya, ea, ys, es = 0.0, 0.0, np.zeros(size), np.zeros(size)
        for fr in meta[nm]["raw_members"]:
            im, h = load_frame(raw / fr)
            ex = float(h["exposure_s"])
            a, b = amap.project(im, ex, mask)
            ya, ea = ya + a, ea + b
            v = im[sub].ravel()
            good = cok & np.isfinite(v) & (v >= 0)
            ys += np.bincount(cflat[good], weights=v[good], minlength=size)
            es += np.bincount(cflat[good], minlength=size) * ex
            nread += 1
        Y[i], E[i], have[i], nframes[i] = ys.reshape(2, nb, nq), es.reshape(2, nb, nq), True, len(meta[nm]["raw_members"])
        cy, ce = s.res.Y[i].astype(float), s.res.E[i].astype(float)
        closure[i] = max(float(np.max(np.abs(ya - cy) / np.maximum(np.abs(cy), 1.0))),
                         float(np.max(np.abs(ea - ce) / np.maximum(np.abs(ce), 1e-12))))
    import joint_fit as JF
    tw = [t for t in JF.WINDOWS.get(label, []) if t >= HALO_TMIN]
    dS, ok_n = [], True
    Yp, Ep = Y[s.pre_idx].sum(0), E[s.pre_idx].sum(0)
    for t in tw:
        idx = np.flatnonzero((s.res.time >= t - 0.5) & (s.res.time < t + 0.5))
        if not have[idx].all():
            ok_n = False; break
        Yw, Ew = Y[idx].sum(0), E[idx].sum(0)
        dS.append(np.stack([smooth_rate(Yw[:, b], Ew[:, b]) - smooth_rate(Yp[:, b], Ep[:, b]) for b in range(nb)], 1))
    n_halo, n_cols = halo_exponent(np.array(dS), rho_band, 0.5 * (qe[:-1] + qe[1:])) if ok_n and dS else (0.0, 0)
    root = Path(out_root) if out_root else shadow_root()
    (root / label).mkdir(parents=True, exist_ok=True)
    p = root / label / NAME
    info = dict(label=label, which=which, raw=str(raw), pixmask_sha256=pm["sha256"], rows=list(SHADOW_ROWS), cols=list(CROP_COLS),
                excl_cols=[list(v) for v in EXCL_COLS], stop=[list(map(list, v)) for v in STOP], band_edges=list(BAND_EDGES),
                af_max=AF_MAX, frames_read=nread, seconds=round(time.time() - t0, 1), n_halo=n_halo, n_halo_columns=n_cols,
                halo_windows=tw, closure_max=float(np.nanmax(closure)) if have.any() else None, source_sha256=sha(__file__),
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    np.savez_compressed(p, Y=Y, E=E, have=have, bins=np.array(names), time=s.res.time, nframes=nframes, closure=closure,
                        rho_band=rho_band, npx=npx, q_edges=qe, n_halo=n_halo, geometry=json.dumps(g), info=json.dumps(info))
    p.with_name(NAME + ".sha256").write_text(f"{sha(p)}  {NAME}\n", encoding="utf-8", newline="\n")
    print(f"{label}: {int(have.sum())} bins, {nread} frames, {info['seconds']} s, closure max {info['closure_max']:.2e}, "
          f"n_halo {n_halo:.3f} ({n_cols} columns) -> {p}", flush=True)
    return info


def table_sha(z):
    'sha256 of the table content (Y, E, have, bins, rho_band, n_halo; float64 / bool / str bytes): reproducible'
    h = hashlib.sha256()
    for k in ("Y", "E", "have", "bins", "rho_band", "n_halo"):
        a = np.asarray(z[k])
        h.update(a.astype(str).tobytes() if a.dtype.kind == "U" else np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def check(label, out_root=None, ref=None):
    'G-S1 read closure: above the horizon the same reader / frames reproduce the cache bins (rel <= 1e-6, float32'
    _, AtomicMap = _toolkit()
    root = Path(out_root) if out_root else shadow_root()
    p = root / label / NAME
    if sha(p) != p.with_name(NAME + ".sha256").read_text(encoding="utf-8").split()[0]:
        raise ValueError(f"{p}: sha256 side file")
    z = np.load(p, allow_pickle=False)
    g = json.loads(str(z["geometry"]))
    out = dict(label=label, sha256=sha(p), arrays_sha256=table_sha(z))
    have = z["have"]
    out["G_S1_read_closure_max_rel"] = float(np.nanmax(z["closure"][have]))
    out["G_S1_pass"] = bool(out["G_S1_read_closure_max_rel"] <= 1e-6)

    shape = (1679, 1475)
    crop = [450, 1420, 220, 1330]
    amap = AtomicMap(g, shape, g["alpha_i_deg"], crop=crop, exclude_rings=())
    qe = z["q_edges"]
    import iam as M
    ne = np.asarray(M.scan_grid().native_edges, float)
    rows, cols = np.arange(crop[0], crop[1]), np.arange(crop[2], crop[3])
    af, qy = pixel_af_qy(g, rows, cols)
    qa = np.searchsorted(qe, np.abs(qy), side="right") - 1
    fa = np.searchsorted(ne, af, side="right") - 1
    good = (qa >= 0) & (qa < len(qe) - 1) & (fa >= 0) & (fa < len(ne) - 1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    mine = np.full(shape[0] * shape[1], -1, np.int64); theirs = mine.copy()
    mine[np.ravel_multi_index((rr[good], cc[good]), shape)] = \
        ((qy[good] > 0).astype(int) * (len(ne) - 1) + fa[good]) * (len(qe) - 1) + qa[good]
    theirs[amap.indices] = amap.bins
    out["G_S2_geometry_pixels"] = int(len(amap.indices))
    out["G_S2_mismatched_pixels"] = int(np.sum(mine != theirs))
    out["G_S2_pass"] = bool(np.array_equal(mine, theirs))


    if ref:
        rb = json.loads(Path(ref).read_text(encoding="utf-8"))
        a_, b_ = raw_below_levels(label, rb["bands"])
        af_rb = np.array(rb["af"]); sel = (af_rb < -0.06) & (af_rb > -0.15)
        stored = [float(np.nanmean(np.array(v, dtype=float)[sel])) for v in rb["scans"][label]["0.3"]]
        out["G_S3a_panel_rerun"] = dict(rerun=a_, stored=stored, max_abs=float(np.max(np.abs(np.array(a_) - stored))))
        out["G_S3b_levels"] = dict(module=b_, panel=a_, rel=[x / y - 1 for x, y in zip(b_, a_)])
        out["G_S3_pass"] = bool(out["G_S3a_panel_rerun"]["max_abs"] < 1e-6 and all(abs(x / y - 1) <= 0.02 for x, y in zip(b_, a_)))
    out["n_halo"] = float(z["n_halo"])
    out["info"] = json.loads(str(z["info"]))
    return out


def raw_below_levels(label, bands, k0=7, nfr=20):
    "Pre-state shadow level per |q_y| band, rows alpha_f -0.15 ... -0.06, frames k0 .. k0 + nfr - 1: (a) the panel's"
    import fabio
    load_frame, _ = _toolkit()
    run, raw, _, mask, _ = cache_inputs(label)
    g = run["geometry"]
    d = raw.parent.name
    rows = np.arange(1150, 1346)
    af0 = np.degrees(np.arctan((g["db_row"] - rows) * g["pixel_mm"] / g["sdd_mm"])) - g["alpha_i_deg"]
    qy0 = 2 * np.pi / 0.103320165 * np.sin(np.arctan((np.arange(1475) - g["db_col"]) * g["pixel_mm"] / g["sdd_mm"]))
    badc = np.zeros(1475, bool); badc[487:494] = True; badc[981:988] = True; badc[977:993] = True; badc[1022:1038] = True
    acc = bad = None; ys = es = 0.0
    rr, cc = np.arange(*SHADOW_ROWS), np.arange(*CROP_COLS)
    afm, qym = pixel_af_qy(g, rr, cc)
    stat = mask[np.ix_(rr, cc)] & (afm < -0.06) & (afm >= -0.15)
    for c0, c1 in EXCL_COLS:
        stat[:, (cc >= c0) & (cc < c1)] = False
    acc_m = np.zeros(stat.shape); exp_m = np.zeros(stat.shape)
    for k in range(k0, k0 + nfr):
        f = raw / f"{d}_{k:05d}.cbf"
        a = fabio.open(str(f)).data[1150:1346].astype(np.int64)
        b = (a < 0) | (a >= 55245)
        acc = a if acc is None else acc + a
        bad = b if bad is None else (bad | b)
        im, h = load_frame(f)
        v = im[np.ix_(rr, cc)]
        okv = stat & np.isfinite(v) & (v >= 0)
        acc_m += np.where(okv, v, 0.0); exp_m += okv * float(h["exposure_s"])
    rate = np.where(bad, np.nan, acc / (nfr * 0.022)); rate[:, badc] = np.nan
    gap = (rows >= 1255) & (rows <= 1271)
    sel = (af0 < -0.06) & (af0 > -0.15)
    pan, mod = [], []
    for q0, q1 in bands:
        cs = (np.abs(qy0) >= q0) & (np.abs(qy0) < q1)
        v = np.nanmean(rate[:, cs], 1); v[gap] = np.nan
        pan.append(float(np.nanmean(v[sel])))
        m = (np.abs(qym) >= q0) & (np.abs(qym) < q1)
        mod.append(float(acc_m[m].sum() / exp_m[m].sum()))
    return pan, mod


def maskcount(label, ref_columns=None):
    'D2 gate (synthesis 2.2): valid pixels per (side, alpha_f row, q column) from the mask file and the geometry alone'
    _, AtomicMap = _toolkit()
    run, _, _, mask, pm = cache_inputs(label)
    g = run["geometry"]
    shape = mask.shape
    amap = AtomicMap(g, shape, g["alpha_i_deg"], crop=[450, 1420, 220, 1330], exclude_rings=((2.1, .13),))
    one = np.ones(shape)
    _, n_mask = amap.project(one, 1.0, mask)
    _, n_geo = amap.project(one, 1.0, None)
    irun = json.loads((IAMP.DATA / label / "intake" / "run.json").read_text(encoding="utf-8"))
    le, same, eq = True, True, []
    for c in irun["chunks"]:
        with np.load(IAMP.DATA / label / "intake" / c["file"]) as z:
            P = z["P"][0].astype(float)
        le &= bool(np.all(P <= n_mask + 1e-9))
        same &= bool(np.array_equal(P > 0, n_mask > 0))
        eq.append(float(np.mean(P[n_mask > 0] == n_mask[n_mask > 0])))
    import iam as M
    grid = M.scan_grid()
    af, q = np.asarray(grid.native_af, float), np.asarray(grid.q, float)
    rows = (af >= 0.03) & (af <= 0.8)
    complete = [dict(side=sd, q=float(q[j]), rows_valid=int((n_mask[sd][rows, j] > 0).sum()))
                for sd in (0, 1) for j in range(len(q)) if q[j] < 0.6 and (n_mask[sd][rows, j] > 0).sum() >= rows.sum() - 11]
    lowq = {}
    for sd in (0, 1):
        for j in range(2):
            zm = af[(n_mask[sd][:, j] == 0) & (n_geo[sd][:, j] > 0) & rows]
            zg = af[(n_geo[sd][:, j] == 0) & rows]
            lowq[f"side{sd}_q{q[j]:.4f}"] = dict(rows_emptied_by_mask=int(len(zm)),
                                                 af_emptied=[float(zm.min()), float(zm.max())] if len(zm) else None,
                                                 rows_without_pixels=[round(float(v), 4) for v in zg])
    out = dict(label=label, pixmask_sha256=pm["sha256"], chunks_checked=len(eq), closure_P_le_mask=le, closure_zero_pattern=same,
               closure_cells_equal_min=min(eq), cells_mask_zero_in_domain=int(((n_mask == 0) & (n_geo > 0) & (af >= 0.03)[None, :, None]).sum()),
               complete_columns=complete, n_complete=len(complete), lowq_columns=lowq)
    if ref_columns:
        rc = json.loads(Path(ref_columns).read_text(encoding="utf-8"))[label]["columns"]
        a_ = {(c["side"], round(c["q"], 4)) for c in rc}; b_ = {(c["side"], round(c["q"], 4)) for c in complete}
        out["complete_columns_vs_ref"] = dict(identical=a_ == b_, only_ref=sorted(a_ - b_), only_mask=sorted(b_ - a_))
    out["pass"] = bool(le and same)
    return out


def stop_profile(label):
    'Column profile of the pre-state shadow (frames of the pre-state bins): columns darker than 0.3 x the row'
    load_frame, _ = _toolkit()
    run, raw, rows, mask, _ = cache_inputs(label)
    os.environ.pop("IAM_SHADOW_BG", None)
    import iam as M
    s = M.Scan(label)
    meta = {r["frame"]: r for r in rows}
    acc, ex, bad = 0.0, 0.0, None
    for i in s.pre_idx:
        for fr in meta[s.res.rows[i]["frame"]]["raw_members"]:
            im, h = load_frame(raw / fr)
            a = im[slice(*SHADOW_ROWS), slice(*CROP_COLS)]
            b = ~np.isfinite(a) | (a < 0)
            acc = acc + np.where(b, 0.0, a); ex += float(h["exposure_s"]); bad = b if bad is None else bad | b
    rate = np.where(mask[slice(*SHADOW_ROWS), slice(*CROP_COLS)] & ~bad, acc / ex, np.nan)
    med = np.nanmedian(rate, axis=1, keepdims=True)
    dark = np.nan_to_num(rate < 0.3 * med)
    cols = np.arange(*CROP_COLS)
    return dict(label=label, dark_columns=cols[dark.mean(0) >= 0.3].tolist(),
                column_mean=dict(zip(map(int, cols[::5]), [float(v) for v in np.nanmean(rate, 0)[::5]])))


def selftest():
    'Data-free checks: moving-sum ratio (exact sums, nearest fill, floor 1, S_w = S_pre -> 1), halo anchor (template'
    Y = np.array([[0.0, 2, 4, 6, 8, 10, 0], [5, 0, 0, 0, 0, 0, 7]])
    E = np.array([[0.0, 1, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 0, 1]])
    r = smooth_rate(Y, E, 3)

    e1 = float(np.max(np.abs(r[0] - [2, 3, 4, 6, 8, 9, 10]))) + float(np.max(np.abs(r[1] - [5, 5, 5, 5, 7, 7, 7])))
    nb, nq = nbands(), 7
    Ep = np.ones((2, nb, nq)); Yp = np.ones((2, nb, nq)) * np.linspace(1, 3, nq)
    Yw = Yp.copy(); Yw[0] *= 2.0; Yw[1] *= 0.5
    rat = window_ratio(Yw, Ep, Yp, Ep, 3)
    e2 = float(np.max(np.abs(rat[0] - 2))) + float(np.max(np.abs(rat[1] - 1)))
    e3 = float(np.max(np.abs(window_ratio(Yp, Ep, Yp, Ep, 3) - 1)))
    q, af = np.linspace(0.3, 0.9, nq), np.array([0.05, 0.1, 0.2])
    rb = np.empty((2, nb, nq)); rb[:] = rho_spec(np.full(nq, 0.05), q)[None, None, :]
    Yw2 = Yp.copy(); Yw2[:, 0] += 3.0
    T = halo_template(Yw2, Ep, Yp, Ep, rb, 1.7, af, q, ncol=1)
    e4 = float(np.max(np.abs(T[:, 0, :] - 3.0) / 3.0))
    rho_b = np.ones((2, nb, nq)) * np.linspace(0.45, 0.65, nb)[None, :, None]
    dS = np.ones((3, 1, 1, 1)) * (5.0 * rho_b ** -1.3)[None]
    n_est, ncols = halo_exponent(dS, rho_b, q, qmin=0.0)
    e5 = abs(n_est - 1.3)
    e6 = float(rho_spec(np.array([0.4]), np.array([0.0]))[0])
    g0 = dict(sdd_mm=3000.0, pixel_mm=0.172, db_col=700.5, db_row=1300.5, alpha_i_deg=0.4, sample_roll_deg=0.0, energy_ev=12000.0)
    af1, qy1 = pixel_af_qy(g0, [1250], [700.5 - 100, 700.5 + 100])
    e7 = abs(qy1[0, 0] + qy1[0, 1]) + abs(af1[0, 0] - af1[0, 1])
    e6 += abs(float(rho_spec(np.array([0.4 + 0.25]), np.array([0.0]))[0]) - 0.25)


    nq2 = 40
    q2 = np.linspace(0.05, 1.0, nq2)
    Ep2 = np.ones((2, nb, nq2)); Yp2 = np.full((2, nb, nq2), 4.0)
    Yw2 = Yp2.copy(); Yw2[0, 0] = 10.0 + np.where(np.arange(nq2) % 2, 1.0, -1.0); Yw2[1, 0] = 2.0; Yw2[:, 1:] = 99.0
    fa = window_add(Yw2, Ep2, Yp2, Ep2, 1.25, q2)
    hi2 = q2 >= QSPLIT
    e8 = float(np.max(np.abs(fa[1, 0] - (2.0 - 1.25 * 4.0))))
    alt, jj = np.where(np.arange(nq2) % 2, 1.0, -1.0), np.arange(nq2)
    ex_hi, ex_lo = 5.0 + np.convolve(alt, np.ones(15), "same") / 15, 5.0 + np.convolve(alt, np.ones(5), "same") / 5
    s_hi, s_lo = hi2 & (jj >= 7) & (jj <= nq2 - 8), ~hi2 & (jj >= 2)
    e9 = float(max(np.max(np.abs(fa[0, 0, s_hi] - ex_hi[s_hi])), np.max(np.abs(fa[0, 0, s_lo] - ex_lo[s_lo]))))
    e10 = float(np.max(np.abs(window_add(Yw2, Ep2, Yp2, Ep2, 1.25, q2, 0.7) - 0.7 * fa)))
    p0 = window_pre0(Yw2, Ep2, Yp2, Ep2, 1.25, q2)
    e11 = float(np.max(np.abs(p0[1] - (2.0 / 4.0 - 1.25))))
    e12 = float(np.max(np.abs(window_add(Yp2, Ep2, Yp2, Ep2, 1.0, q2))))
    ok = e1 < 1e-12 and e2 < 1e-12 and e3 == 0.0 and e4 < 1e-12 and e5 < 1e-9 and ncols > 0 and e6 < 1e-9 and e7 < 1e-12
    ok = ok and fa.shape == (2, 1, nq2) and e8 < 1e-12 and e9 < 1e-12 and e10 < 1e-12 and e11 < 1e-12 and e12 == 0.0
    print("shadow_rows selftest", "ALL PASS" if ok else "FAILED", dict(smooth=e1, ratio=e2, neutral=e3, halo_anchor=e4,
                                                                      n_recovery=e5, rho_spec=e6, geometry_symmetry=e7,
                                                                      flat_unclipped=e8, flat_qsplit=e9, flat_aB=e10,
                                                                      pre0=e11, flat_neutral=e12))
    return ok


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "check", "maskcount", "stop"])
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--windows", default="joint", choices=["joint", "pre", "all"])
    ap.add_argument("--out", default=None, help="root for shadow_v1.npz (default IAM_DATA)")
    ap.add_argument("--ref", default=None, help="check: raw_below.json (G-S3); maskcount: yoneda_columns.json of the panel")
    ap.add_argument("--json", default=None, help="write the gate / count results here")
    ap.add_argument("--times", type=float, nargs="+", default=None, help="extract: window centres (s) instead of joint_fit.WINDOWS")
    ap.add_argument("--half", type=float, default=0.5, help="extract: window half width (s) with --times")
    a = ap.parse_args()
    os.environ.pop("IAM_SHADOW_BG", None)
    res = {}
    for lab in a.labels:
        if a.cmd == "extract":
            res[lab] = extract(lab, a.windows, a.out, a.times, a.half)
        elif a.cmd == "check":
            res[lab] = check(lab, a.out, a.ref)
        elif a.cmd == "maskcount":
            res[lab] = maskcount(lab, a.ref)
        else:
            res[lab] = stop_profile(lab)
    txt = json.dumps(res, indent=1, default=float)
    if a.json:
        Path(a.json).write_text(txt + "\n", encoding="utf-8", newline="\n")
    print(txt if a.cmd != "extract" else "", flush=True)


if __name__ == "__main__":
    main()
