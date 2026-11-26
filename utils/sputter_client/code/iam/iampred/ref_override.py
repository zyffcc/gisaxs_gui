'Reference override of a scan (2026-11-25, results/iam/fringe_refs_20261125; moved here from references.py on'

from __future__ import annotations

from . import references as RF


def load_scan(cfg, scan):
    out, shas = RF.load_scan(cfg, scan)
    ov = (cfg["scans"].get(scan) or {}).get("override")
    if not ov:
        return out, shas
    rep, s2 = RF.load_scan(dict(cfg, scans={scan: {k: v for k, v in ov.items() if k != "t_min"}}), scan)
    t_min = float(ov["t_min"])
    out = [r for r in out if r["t"] < t_min] + [r for r in rep if r["t"] >= t_min]
    shas = dict(shas)
    shas.update(s2)
    return sorted(out, key=lambda r: r["t"]), shas


def load(cfg):
    refs, shas = {}, {}
    for sc in cfg["scans"]:
        refs[sc], s = load_scan(cfg, sc)
        shas.update({f"{sc}/{k}": v for k, v in s.items()})
    return refs, shas
