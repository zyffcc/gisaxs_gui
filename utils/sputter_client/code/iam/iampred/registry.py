'Family registry: one JSON file per family in iampred/families/ describes everything the predictor needs to know about'

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
IAM_DIR = HERE.parent
FAM_DIR = HERE / "families"

REQUIRED = ("family", "fitter", "env", "scans", "reported")
LAM_12KEV = 0.103320165
LAM_TOL = 0.05
"""Relative difference between a new scan's wavelength and its family's (the wavelength its network was trained at)
up to which the family's own optics are used (2026-11-19): the network gives the starting solutions, the forward model
renders and fits at the scan's own wavelength (predict sets IAM_LAM_NM to it). Beyond it - and for Au at any energy but
12 keV (its hand optics are 12 keV values next to the L3 edge at 11.92 keV) - the scan is analysed with the tabulated
optics of xray_optics.py at its own energy (IAM_OPTICS=table, 2026-11-20): the networks still give the starts, the
joint fit decides (analyze --mode auto refines). Refused only outside the optics table (OPTICS_KEV)."""
OPTICS_KEV = (4.0, 30.0)
"""Energy range (keV) of the tabulated optics (xray_optics_table.npz)."""


def model_lam(cfg):
    "The wavelength (nm) a family's network was trained at."
    return float((cfg.get("env") or {}).get("IAM_LAM_NM", LAM_12KEV))


def lam_problem(metal, lam):
    "Why a scan of ``metal`` at wavelength ``lam`` (nm) cannot be analysed ('' when it can): an energy outside the"
    kev = 1.239841984 / float(lam)
    if not OPTICS_KEV[0] <= kev <= OPTICS_KEV[1]:
        return (f"this scan is at {kev:.3f} keV (wavelength {float(lam):.5f} nm); the optics are tabulated from "
                f"{OPTICS_KEV[0]:g} to {OPTICS_KEV[1]:g} keV")
    return ""


def lam_outside(metal, lam, cfg=None):
    "The scan is away from its networks' wavelength (more than LAM_TOL; Au: anything but 12 keV): it is analysed with"
    if cfg is not None and cfg.get("energy_range_kev"):
        lo, hi = (float(v) for v in cfg["energy_range_kev"])
        return not lo * (1 - 1e-6) <= 1.239841984 / float(lam) <= hi * (1 + 1e-6)
    model = model_lam(cfg) if cfg is not None else (0.105 if metal == "Cu" else LAM_12KEV)
    if metal == "Au":
        return abs(float(lam) / LAM_12KEV - 1) > 1e-4
    return abs(float(lam) / model - 1) > LAM_TOL


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


KINPUT_FILES = ("kinput.py",)


def pred_hash(cfg=None):
    'sha256 of the iampred code that defines the simulated samples and the network inputs; for a family whose features'
    h = hashlib.sha256()
    kfz = bool(cfg) and (cfg.get("features") or {}).get("axis") == "kfz"
    for nm in PRED_FILES + (KINPUT_FILES if kfz else ()):
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


def network_family(cfg):
    "The family whose network gives this family's candidates: its own, or the one named by 'network_from'"
    return cfg.get("network_from") or cfg["family"]


def map_params(src_names, x, dst_names, start, lb, ub):
    'A parameter vector given in src_names order, in dst_names order: a shared name keeps its value, a name only in'
    import numpy as np
    src = dict(zip(src_names, (float(v) for v in x)))
    v = [src.get(nm, float(start[i])) for i, nm in enumerate(dst_names)]
    new = [nm for nm in dst_names if nm not in src]
    dropped = [nm for nm in src_names if nm not in set(dst_names)]
    return np.clip(np.array(v, float), np.asarray(lb, float), np.asarray(ub, float)), new, dropped


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
