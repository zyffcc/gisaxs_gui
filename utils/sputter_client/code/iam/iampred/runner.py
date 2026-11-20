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


_INIT_ERROR = None
"""Why this worker could not be set up (None: it was): said by the first task it gets."""


def _init(env):
    'Pool initializer. A failure is kept for the tasks (2026-11-19): an initializer that raises makes the pool start'
    global _INIT_ERROR
    try:
        from . import adapter
        adapter.activate(env)
    except BaseException as e:
        _INIT_ERROR = f"{type(e).__name__}: {e}"


class Guard:
    'A task function for a multiprocessing pool that cannot hang it (2026-11-19). SystemExit in a worker (the IAM'

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        if _INIT_ERROR is not None:
            raise RuntimeError(f"worker set-up failed: {_INIT_ERROR}")
        try:
            return self.fn(*args)
        except SystemExit as e:
            raise RuntimeError(str(e.code if e.code is not None else "SystemExit")) from None


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
            for i, r in zip(idx, pool.map(Guard(fn), [tasks[i] for i in idx], chunksize=1)):
                res[i] = r
    return res
