'Did the deposition stop before the end of the scan although no duration was given? (2026-11-24, stop_hint_20261124)'

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

SLOPE_LIMIT = 0.15
RATIO_LIMIT = 1.5
MIN_HALF = 3
MIN_SPAN = 1.3
MIN_BEFORE, MIN_AFTER = 2, 3
DEV_COLUMNS = ("seq_dev5", "refined_dev5", "best_dev5")


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x


def dev_column(rows):
    'The deviation of the solution shown: the first of DEV_COLUMNS with a value in at least half of the windows.'
    for c in DEV_COLUMNS:
        v = [_num(r.get(c)) for r in rows]
        if v and np.mean(np.isfinite(v)) >= 0.5:
            return c
    return None


def hinge(t, L):
    "Break time t_b of ln L = a + b min(ln t, ln t_b), least squares over the readings' own times (from the second"
    x, y = np.log(t), np.log(L)
    best = (math.inf, float(t[0]))
    for k in range(1, len(t)):
        z = np.minimum(x, x[k])
        A = np.c_[np.ones_like(z), z]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(np.sum((A @ coef - y) ** 2))
        if sse < best[0] - 1e-12:
            best = (sse, float(t[k]))
    return best[1]


def hint(rows, t_sput=None):
    "The check and its numbers; 'hint' True when both signs are there and no duration was given."
    out = dict(hint=False, slope_limit=SLOPE_LIMIT, ratio_limit=RATIO_LIMIT)
    if t_sput is not None:
        return dict(out, checked=False, reason="duration given")
    t = np.array([_num(r.get("t_rel")) for r in rows])
    L = np.array([_num(r.get("obs_L_knee_nm")) for r in rows])
    knee = np.array([str(r.get("obs_knee_status") or "knee") == "knee" for r in rows])
    m = np.isfinite(t) & (t > 0) & np.isfinite(L) & (L > 0) & knee
    if m.sum() < MIN_HALF + 1:
        return dict(out, checked=False, reason="too few lateral-scale readings", readings=int(m.sum()))
    tk, Lk = t[m], L[m]
    order = np.argsort(tk)
    tk, Lk = tk[order], Lk[order]
    half = tk >= 0.5 * tk.max()
    if half.sum() < MIN_HALF or tk[half].max() / tk[half].min() < MIN_SPAN:
        return dict(out, checked=False, reason="the readings do not span enough time", readings=int(m.sum()))
    slope = float(np.polyfit(np.log(tk[half]), np.log(Lk[half]), 1)[0])
    out.update(checked=True, readings=int(m.sum()), slope=round(slope, 4), stalls=bool(slope < SLOPE_LIMIT))
    if not out["stalls"]:
        return out
    tb = hinge(tk, Lk)
    h = np.array([_num(r.get("h_nom")) for r in rows])
    hs = np.isfinite(h) & np.isfinite(t)
    h_b = float(np.interp(tb, t[hs], h[hs])) if hs.sum() >= 2 else None
    out.update(t_stop_s=round(tb, 2), h_stop_nm=None if h_b is None else round(h_b, 2))
    col = dev_column(rows)
    if col is None:
        return dict(out, reason="no deviation column")
    d = np.array([_num(r.get(col)) for r in rows])
    ok = np.isfinite(d) & np.isfinite(t)
    before, after = d[ok & (t <= tb)], d[ok & (t > tb)]
    out.update(dev_column=col, n_before=int(before.size), n_after=int(after.size))
    if before.size < MIN_BEFORE or after.size < MIN_AFTER:
        return dict(out, reason="too few windows before or after the break")
    mb, ma = float(np.median(before)), float(np.median(after))
    out.update(dev_before=round(mb, 3), dev_after=round(ma, 3), ratio=round(ma / mb, 3) if mb > 0 else None)
    out["hint"] = bool(mb > 0 and ma / mb >= RATIO_LIMIT)
    return out


def t_sput_of(scan_json):
    'The duration of a scan.json dict (None when not given).'
    v = (scan_json or {}).get("t_sput")
    return None if v is None else float(v)


def for_folder(d):
    "The check of a result folder (predict's summary.json and windows.csv). The duration is that of the imported"
    d = Path(d)
    try:
        summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        with open(d / "windows.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except (OSError, ValueError):
        return dict(hint=False, checked=False, reason="no result tables")
    if not summ.get("new_scan"):
        return dict(hint=False, checked=False, reason="built-in scan")
    return hint(rows, t_sput_of(summ.get("scan_json")))
