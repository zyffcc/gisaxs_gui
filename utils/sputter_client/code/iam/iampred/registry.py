'Family registry: one JSON file per family in iampred/families/ describes everything the predictor needs to know about'

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
IAM_DIR = HERE.parent
FAM_DIR = HERE / "families"

REQUIRED = ("family", "fitter", "env", "scans", "reported")
DEFAULTS = dict(
    context_step=[[25.0, 1.0], [1e9, 5.0]],
    prior=dict(near=0.55, interp=0.2, cross=0.1, uniform=0.15, sigma_u=0.06, sigma_mass=0.15, t_jitter=1.0,
               balance="metal"),
    noise=dict(bank=True, bank_amp=[0.4, 1.2], smooth=0.05, bank_sigma=2.0),
    features=dict(af=[0.03, 1.0], pool=2),
    scalars=["ln_hm_nominal", "t_rel"],
    metals=["Au", "Ag"],
)


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(name):
    'Family config by name (file families/<name>.json) or path; defaults filled in, hashes added.'
    p = Path(name)
    if not p.suffix:
        p = FAM_DIR / f"{name}.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    miss = [k for k in REQUIRED if k not in raw]
    if miss:
        raise ValueError(f"{p}: missing keys {miss}")
    cfg = _merge(DEFAULTS, raw)
    core = {k: v for k, v in cfg.items() if k not in NOT_HASHED}
    cfg["family_hash"] = hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    cfg["forward_hash"] = forward_hash()
    cfg["_path"] = str(p)
    return cfg


NOT_HASHED = ("description", "needs", "notes", "monotone")
PRED_FILES = ("adapter.py", "features.py", "generate.py", "references.py")


def pred_hash():
    'sha256 of the iampred code that defines the simulated samples and the network inputs.'
    h = hashlib.sha256()
    for nm in PRED_FILES:
        h.update(nm.encode()); h.update((HERE / nm).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


FORWARD_FILES = ("iam.py", "iam2.py", "iam4.py", "iam_fit2.py", "iam_fit4.py", "iam_fit_t.py", "iam_paths.py",
                 "shadow_rows.py", "tio2_dwba.py", "tio2_explore.py", "underlayer.py")


def forward_hash():
    h = hashlib.sha256()
    for nm in FORWARD_FILES:
        f = IAM_DIR / nm
        if f.exists():
            h.update(nm.encode()); h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def families():
    return sorted(p.stem for p in FAM_DIR.glob("*.json"))


def scan_env(cfg, scan):
    "IAM_* environment of one scan: family env + the scan's overrides."
    env = {k: str(v) for k, v in cfg["env"].items()}
    env.update({k: str(v) for k, v in (cfg["scans"][scan].get("env") or {}).items()})
    return env


def env_groups(cfg):
    'Scans grouped by identical environment (the IAM modules read IAM_* at import: one process per group).'
    groups = {}
    for sc in cfg["scans"]:
        key = json.dumps(scan_env(cfg, sc), sort_keys=True)
        groups.setdefault(key, []).append(sc)
    return [(json.loads(k), v) for k, v in groups.items()]
