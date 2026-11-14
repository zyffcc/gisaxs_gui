'Forward-model adapter: everything the predictor does with the IAM code goes through here.'

from __future__ import annotations

import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_STATE = {}
MASS = "ln_mass_factor"


def activate(env):
    'Set the IAM environment of this process (first call only) and import the forward model.'
    key = json.dumps(env, sort_keys=True)
    if "env" in _STATE:
        if _STATE["env"] != key:
            raise RuntimeError("this process already runs another IAM configuration (IAM modules freeze env at import)")
        return
    if any(m in sys.modules for m in ("iam", "iam_fit4", "iam_fit_t", "iam_fit2")):
        raise RuntimeError("iam was imported before the family environment was set")
    stray = [k for k in os.environ if k.startswith("IAM_") and k not in env]
    for k in stray:
        del os.environ[k]
    os.environ.update({k: str(v) for k, v in env.items()})
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    sys.path.insert(0, str(HERE.parent))
    _STATE["env"] = key


def register(label):
    'Register a scan that is not in iam.SCANS from DATA/<label>/scan.json (written by iampred.intake).'
    import iam as M
    import iam_paths as IAMP
    p = IAMP.DATA / label / "scan.json"
    if label in M.SCANS and p.exists() and M.SCANS[label].get("source") != "iampred.intake":
        raise KeyError(f"{label} is a built-in scan label and also has {p}: choose another label for the new scan")
    if label not in M.SCANS:
        if not p.exists():
            raise KeyError(f"{label}: not in iam.SCANS and no {p} (run iampred.intake)")
        M.SCANS[label] = {k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items() if k != "rate_note"}


def get_scan(label):
    'iam.Scan of a label (registered from its scan.json when needed).'
    import iam as M
    register(label)
    return M.Scan(label)


def fitter(cfg):
    import importlib
    return importlib.import_module(cfg["fitter"])


class Context:
    'One real window of a scan (exposure, mask, measured pre-state, shadow terms): the instrument part of a sample.'

    def __init__(self, cfg, scan, t):
        import iam as M
        F = fitter(cfg)
        self.cfg, self.scan, self.t = cfg, scan, float(t)
        self.f = F.Fit(scan, float(t))
        self.s = scan
        self.w = self.f.w
        self.t_mean = float(self.w["t_mean"])
        self.h_nom_ctx = self.h_nominal(self.t)
        self.names = [q[0] for q in self.f.free]
        mf = cfg.get("mass_factor")
        self.lb = np.r_[self.f.lb, [mf[0]] if mf else []].astype(float)
        self.ub = np.r_[self.f.ub, [mf[1]] if mf else []].astype(float)
        self.pnames = self.names + ([MASS] if mf else [])
        self.M = M


    def h_nominal(self, t):
        s = self.s
        dep = max(float(t) + (self.t_mean - self.t) - float(s.t_dep), 0.0)
        if "t_sput" in s.cfg:
            dep = min(dep, float(s.cfg["t_sput"]))
        return float(s.rate_h * dep)

    def h_true(self, t, lnmf):
        return max(self.h_nominal(t), 0.002) * math.exp(lnmf)

    def render(self, x, t=None, parts=False):
        'Forward model of the full parameter vector x (pnames order) at morphology time t (default: context time).'
        t = self.t if t is None else float(t)
        x = np.asarray(x, float)
        nf = len(self.names)
        if len(self.pnames) > nf:
            h = self.h_true(t, x[nf])
        else:
            h = self.h_nominal(t) if t != self.t else self.h_nom_ctx
        self.w["h_m"] = h
        out, m, p = self.f.model(x[:nf], parts=parts)
        return out, m, p, h

    def dev5(self, mu):
        d = self.M.dev5(self.w, mu, self.s.V_pre)
        return float(d.sum() / max(self.f.ok.sum(), 1)), d

    def reported(self, m, p, h):
        "The family's reported quantities from config expressions (evaluated on the morphology object m, the full"
        ns = dict(m=m, p=p, w=self.w, h=h, np=np, log=np.log, exp=np.exp, sqrt=np.sqrt, pi=np.pi, min=min, max=max,
                  float=float, sum=np.sum, abs=abs)
        out = []
        for r in self.cfg["reported"]:
            try:
                out.append(float(eval(r["expr"], {"__builtins__": {}}, ns)))
            except Exception:
                out.append(float("nan"))
        return np.array(out, float)


class ScanContexts:
    'All contexts of one scan, built once per worker process; the raw bins are dropped afterwards.'

    def __init__(self, cfg, scan_label, times):
        import iam as M
        self.label = scan_label
        self.scan = get_scan(scan_label)
        self.ctx = [Context(cfg, self.scan, t) for t in times]
        if hasattr(self.scan, "res"):
            del self.scan.res
        gc.collect()


def x_from_reference(ctx, params, lnmf=None):
    'Full parameter vector (ctx.pnames order) from a reference parameter dict (missing names: fitter start value).'
    import importlib
    F = importlib.import_module(ctx.cfg["fitter"])
    start = {q[0]: q[3] for q in F.PAR}
    fill = ctx.cfg.get("fill", {})
    v = []
    for nm in ctx.names:
        v.append(params.get(nm, fill.get(nm, start[nm])))
    x = np.clip(np.array(v, float), ctx.lb[:len(ctx.names)], ctx.ub[:len(ctx.names)])
    if len(ctx.pnames) > len(ctx.names):
        x = np.r_[x, np.clip(0.0 if lnmf is None else lnmf, ctx.lb[-1], ctx.ub[-1])]
    return x
