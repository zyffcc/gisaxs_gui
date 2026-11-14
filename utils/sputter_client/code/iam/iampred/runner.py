'Process pools per environment group: the IAM modules freeze IAM_* at import, so every group of scans with the same'

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

from . import registry as REG

HERE = Path(__file__).resolve().parent


def site_env(cfg, data=None, shadow=None, work=None):
    "Machine paths: explicit arguments, else the current environment, else the family's 'data_local' (relative to"
    env = {}
    if data:
        env["IAM_DATA"] = str(Path(data).resolve())
    else:
        d = os.environ.get("IAM_DATA") or cfg.get("data_local")
        if d:
            p = Path(d)
            env["IAM_DATA"] = str(p if p.is_absolute() else (HERE.parent / p).resolve())
    s = shadow or os.environ.get("IAM_SHADOW_DIR")
    if s:
        env["IAM_SHADOW_DIR"] = str(Path(s).resolve())
    w = work or os.environ.get("IAM_WORK")
    if w:
        env["IAM_WORK"] = str(Path(w).resolve())
    for k in ("IAM_GIMAP", "IAM_RAW", "IAM_DIST", "TZ"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def _init(env):
    from . import adapter
    adapter.activate(env)


def run(cfg, tasks, fn, workers, data=None, shadow=None, work=None):
    'Run fn(task) for tasks whose second element is the scan label; results in task order.'
    base = site_env(cfg, data, shadow, work)
    groups = REG.env_groups(cfg)
    res = [None] * len(tasks)
    ctx = mp.get_context("spawn")
    for env, scans in groups:
        idx = [i for i, t in enumerate(tasks) if t[1] in scans]
        if not idx:
            continue
        e = dict(env); e.update(base)
        if workers <= 0:
            _init(e)
            for i in idx:
                res[i] = fn(tasks[i])
            continue
        n = max(1, min(workers, len(idx)))
        with ctx.Pool(n, initializer=_init, initargs=(e,), maxtasksperchild=None) as pool:
            for i, r in zip(idx, pool.map(fn, [tasks[i] for i in idx], chunksize=1)):
                res[i] = r
    return res
