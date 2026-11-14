'Reference solutions of a family (the adopted fits): read from the record files copied into iampred/refs/.'

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFS = HERE / "refs"


def _path(p):
    p = Path(p)
    return p if p.is_absolute() else REFS / p


def load_scan(cfg, scan):
    'References of one scan of the family (sorted by time) and the sha256 of the record files used.'
    sc = cfg["scans"][scan]
    kind = sc.get("kind", cfg.get("ref_kind", "joint"))
    out, shas = [], {}
    if kind == "joint":
        p = _path(sc["ref"])
        raw = p.read_bytes()
        shas[p.name] = hashlib.sha256(raw).hexdigest()
        rec = json.loads(raw.decode("utf-8"))
        stage = sc.get("stage") or cfg.get("ref_stage") or list(rec["stages"])[-1]
        st = rec["stages"][stage]
        ns = len(rec["shared_names"])
        x = st["x"]
        xs, xl = x[:ns], x[ns:]
        nl = len(rec["local_names"])
        shared = dict(zip(rec["shared_names"], xs))
        for k, t in enumerate(rec["windows"]):
            p_ = dict(zip(rec["local_names"], xl[k * nl:(k + 1) * nl]))
            for nm, v in shared.items():
                if nm not in ("lnkap", "dt"):
                    p_.setdefault(nm, v)
            d = st["windows"][k]
            out.append(dict(t=float(t), params=p_, h_m=float(d["h_m"]), dev5=float(d["dev5_per_cell"]),
                            source=f"{p.name}:{stage}"))
    elif kind == "window":
        for f in sorted(_path(sc["ref"]).glob(sc.get("glob", f"{scan}_*.json"))):
            if f.name.endswith("_mu.json"):
                continue
            raw = f.read_bytes()
            r = json.loads(raw.decode("utf-8"))
            if r.get("scan") != scan or "params" not in r:
                continue
            shas[f.name] = hashlib.sha256(raw).hexdigest()
            out.append(dict(t=float(r["t"]), params=dict(r["params"]), h_m=float(r["h_m"]),
                            dev5=float(r.get("dev5_per_cell", float("nan"))), source=f.name))
    else:
        raise ValueError(f"unknown reference kind {kind}")
    windows = sc.get("windows")
    if windows:
        out = [r for r in out if any(abs(r["t"] - float(w)) < 1e-6 for w in windows)]
    return sorted(out, key=lambda r: r["t"]), shas


def load(cfg):
    refs, shas = {}, {}
    for sc in cfg["scans"]:
        refs[sc], s = load_scan(cfg, sc)
        shas.update({f"{sc}/{k}": v for k, v in s.items()})
    return refs, shas
