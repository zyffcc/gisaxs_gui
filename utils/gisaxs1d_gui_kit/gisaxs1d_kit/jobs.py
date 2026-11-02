"""Job handlers for a host application that runs background work in its own job processes (GIMaP JobRunner).

A handler has the signature handler(payload, report, is_cancelled) -> JSON-serialisable value, is named by
'module:function' and runs in a process started by the host:

    JobRequest(handler='gisaxs1d_kit.jobs:run_fit1d_job', payload=fit_payload(q, intensity, sigma, composition='sphere,vc'))

report(completed, total, message, details) and is_cancelled() are provided by the host (GIMaP:
src/gimap/integrations/jobs/local_process.py). The payload is a plain dict (lists, no numpy arrays), so it crosses
the process boundary as JSON. NaN / inf cannot travel in such a payload: fit_payload replaces non-finite points of
the curve by null and the handler drops them.

  run_fit1d_job     payload = arguments of gisaxs1d_kit.core.fit_curve (+ 'operation': 'fit' | 'preview')
                    value   = result dict (schema gisaxs1d_gui_result_v1)
  run_fit_many_job  payload = arguments of core.fit_many; progress per curve; stops between curves when cancelled
  run_series_job    payload = arguments of core.run_series; progress = frames written

The job process needs the kit folder on its import path. When the host cannot arrange that, pass
payload['kit_home'] and use the tiny loader shipped in gimap/fit1d_predictor.py (it inserts the folder into
sys.path and then calls these functions).
"""
from __future__ import annotations

import math
import os


def _finite_or_none(values):
    out = []
    for v in values:
        v = float(v)
        out.append(v if math.isfinite(v) else None)
    return out


def fit_payload(q, intensity, sigma=None, **settings):
    """JSON-safe payload for run_fit1d_job from arrays and the settings of core.fit_curve."""
    plain = lambda a: _finite_or_none(a.tolist() if hasattr(a, 'tolist') else a)
    payload = dict(settings)
    payload.update(q=plain(q), intensity=plain(intensity), sigma=None if sigma is None else plain(sigma))
    return payload


def _restore(args):
    """Curve arrays with null back to NaN (the fit drops non-finite points)."""
    out = dict(args)
    for key in ('q', 'intensity', 'sigma'):
        if out.get(key) is not None:
            out[key] = [float('nan') if v is None else v for v in out[key]]
    return out


def _setup(payload):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(name, '1')
    if payload.get('kit_home'):
        os.environ['GISAXS1D_KIT_HOME'] = str(payload['kit_home'])
    from gisaxs1d_kit import core
    return core


def _clean(value):
    """Everything JSON-serialisable without NaN (job results are checked with allow_nan=False)."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, 'tolist'):
        return _clean(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def run_fit1d_job(payload, report=None, is_cancelled=None):
    core = _setup(payload)
    if report is not None:
        report(0, 1, 'Loading the fitting code', {})
    if is_cancelled is not None and is_cancelled():
        raise RuntimeError('cancelled before start')
    args = _restore({k: v for k, v in payload.items() if k not in ('kit_home', 'operation')})
    value = core.preview_curve(args) if payload.get('operation') == 'preview' else core.fit_curve(args)
    if report is not None:
        report(1, 1, f"chi2 per point {value['summary']['chi2_per_point']:.3g}", {'status': value['summary']['status']})
    return _clean(value)


def run_fit_many_job(payload, report=None, is_cancelled=None):
    core = _setup(payload)
    args = {k: v for k, v in payload.items() if k != 'kit_home'}
    args['curves'] = [_restore(c) for c in args.get('curves') or []]
    return _clean(core.fit_many(args, report, is_cancelled))


def run_series_job(payload, report=None, is_cancelled=None):
    core = _setup(payload)
    return _clean(core.run_series({k: v for k, v in payload.items() if k != 'kit_home'}, report, is_cancelled))
