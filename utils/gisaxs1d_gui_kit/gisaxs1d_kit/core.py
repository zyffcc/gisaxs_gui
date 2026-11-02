"""The operations of the kit, executed in the worker process (or in a job process of the host application).

This module imports the fitting code (research.fit1d, research.series1d): do not import it into a GUI process;
use gisaxs1d_kit.client.Fit1DClient or gisaxs1d_kit.jobs there.

Arguments of fit_curve / preview_curve (a plain dict, JSON-serialisable):
  q, intensity          arrays of the same length
  sigma                 array or None (None: Poisson errors assumed, flagged)
  q_unit                'nm^-1' (default) or 'A^-1'
  fold                  what to do with negative q: 'abs' (default: |q|, both sides together), 'positive', 'negative'
  q_range               [lo, hi] in the unit of q (after folding), optional
  composition           e.g. 'sphere,vc' | 'vc,vc,vc' | 'sphere' | 'auto:vc:3' (shapes: sphere, vc, rc)
  mode                  'gisaxs' (default: structure factor and low-q term on) | 'saxs'
  qz                    out-of-plane momentum in nm^-1 for spheres / random cylinders in GISAXS mode (default 0;
                        ignored in SAXS mode)
  policy                'fast' (default) | 'auto' | 'robust' | 'preview'
  model                 network: catalogue name ('general', 'au_np_si', ...) or a directory; default: the catalogue default
  sigma_floor           relative systematic error added to sigma in quadrature (default 0; 0.03 for bright real curves)
  constraints           {'pop1': {'R_nm': 2.5}, 'pop2': {'D_nm': [5, 8]}, 'nu': 2, 'background': 0} (fix or range, nm)
  lowq, lowq_nu, structure_factor, dwba     passed to research.fit1d.fit1d unchanged (see its docstring)
  include_raw           also return the complete fit1d result under 'raw'
Result: gisaxs1d_kit.results (schema gisaxs1d_gui_result_v1).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from gisaxs1d_kit import _paths, results

Q_FACTOR = {'nm^-1': 1.0, 'A^-1': 10.0, '1/nm': 1.0, '1/A': 10.0, 'nm-1': 1.0, 'A-1': 10.0}
COMPOSITIONS = ('sphere', 'vc', 'sphere,vc', 'sphere,vc,vc', 'vc,vc', 'vc,vc,vc', 'sphere,sphere', 'sphere,sphere,vc',
                'auto:vc:3', 'auto:sphere,vc:3')
POLICIES = ('fast', 'auto', 'robust', 'preview')
_READY = threading.Lock()
_STATE = {}


def ensure_ready():
    """Put the fitting code on sys.path and make sure the form-factor tables exist (unpacked once)."""
    with _READY:
        if _STATE.get('ready'):
            return _STATE
        app = _paths.app_dir()
        if str(app) not in sys.path:
            sys.path.insert(0, str(app))
        from research.fit1d import schema
        packed = _paths.packed_tables()
        directory = Path(schema.tables_dir())
        if packed is not None and not all((directory/name).is_file() for name in schema.TABLE_FILES):
            home = str(_paths.kit_home())
            if home not in sys.path:
                sys.path.insert(0, home)
            from tablepack import unpack           # KIT/tablepack.py: lossless .fft -> .npy, checksummed (once, ~2 s)
            unpack(packed, directory)
        schema.check_tables()
        _STATE.update(ready=True, app=str(app), fit1d_version=schema.VERSION)
        return _STATE


def info():
    state = ensure_ready()
    from research.fit1d import schema
    import scipy
    catalogue = _paths.model_catalogue()
    return dict(kit_version=_paths.KIT_VERSION, fit1d_version=state['fit1d_version'], kit_home=str(_paths.kit_home()),
                code=state['app'], models=catalogue, compositions=list(COMPOSITIONS), policies=list(POLICIES),
                q_units=['nm^-1', 'A^-1'], tables_dir=str(schema.tables_dir()),
                python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__, pid=os.getpid())


def prepare_curve(args):
    """(q in nm^-1, intensity, sigma or None, q unit, factor) from the arguments; folding and cropping applied."""
    unit = str(args.get('q_unit') or 'nm^-1')
    if unit not in Q_FACTOR:
        raise ValueError(f'q_unit must be nm^-1 or A^-1, got {unit!r}')
    q = np.asarray(args['q'], float).ravel()
    y = np.asarray(args['intensity'], float).ravel()
    s = None if args.get('sigma') is None else np.asarray(args['sigma'], float).ravel()
    if q.shape != y.shape or (s is not None and s.shape != y.shape):
        raise ValueError(f'q, intensity and sigma must have the same length (got {q.size}, {y.size}'
                         + (f', {s.size})' if s is not None else ')'))
    fold = str(args.get('fold') or 'abs')
    if fold == 'positive':
        keep = q > 0
    elif fold == 'negative':
        keep = q < 0
    elif fold == 'abs':
        keep = q != 0
    else:
        raise ValueError(f"fold must be 'abs', 'positive' or 'negative', got {fold!r}")
    q, y, s = np.abs(q[keep]), y[keep], (None if s is None else s[keep])
    if args.get('q_range') is not None:
        lo, hi = sorted(float(v) for v in args['q_range'])
        keep = (q >= lo) & (q <= hi)
        q, y, s = q[keep], y[keep], (None if s is None else s[keep])
    order = np.argsort(q, kind='stable')
    q, y, s = q[order], y[order], (None if s is None else s[order])
    return q*Q_FACTOR[unit], y, s, unit, Q_FACTOR[unit]


def _fit(args, policy=None):
    ensure_ready()
    from research.fit1d import fit1d
    q, y, s, unit, factor = prepare_curve(args)
    name, directory = _paths.model_dir(args.get('model'))
    extra = {k: args[k] for k in ('lowq', 'lowq_nu', 'structure_factor', 'dwba', 'max_steps', 'heads') if args.get(k) is not None}
    if 'lowq_nu' in args and args['lowq_nu'] is None:
        extra['lowq_nu'] = None                                   # None = fit the exponent
    t0 = time.perf_counter()
    mode = args.get('mode') or 'gisaxs'
    fit = fit1d(q, y, s, composition=args.get('composition') or 'sphere,vc', mode=mode,
                qz=float(args.get('qz') or 0.0) if mode == 'gisaxs' else 0.0,        # SAXS has no qz: a left-over value is ignored
                policy=policy or args.get('policy') or 'fast', model_dir=directory,
                sigma_floor=float(args.get('sigma_floor') or 0.0), constraints=args.get('constraints') or None,
                source=args.get('source') or None, **extra)
    return results.gui_result(fit, model_name=name, q_unit=unit, q_factor=factor, seconds=time.perf_counter()-t0,
                              include_raw=bool(args.get('include_raw')), kit_version=_paths.KIT_VERSION)


def load_file(args):
    """Read one curve file with the readers of fit1d: a yoneda_cut .npz (both sides merged, sigma = max(Poisson,
    left/right difference); its metadata carry qz and the geometry) or a text file with columns q, I[, sigma]
    (comments and DPDAK headers are skipped). args: path, q_unit ('nm^-1' | 'A^-1', text files), side, sigma_mode.
    Returns {'q' (nm^-1), 'intensity', 'sigma' or None, 'q_unit': 'nm^-1', 'qz' or None, 'meta', 'label'}."""
    ensure_ready()
    from research.fit1d import load_input
    path = Path(str(args['path'])).expanduser()
    q, y, s, meta = load_input(path, q_unit=args.get('q_unit') or 'nm^-1', side=args.get('side') or 'both',
                               sigma_mode=args.get('sigma_mode') or 'side')
    qz = meta.get('qz_at_yoneda_mean') if isinstance(meta, dict) else None
    keep = {k: v for k, v in (meta or {}).items() if isinstance(v, (str, int, float, bool)) or v is None}
    return dict(q=q, intensity=y, sigma=s, q_unit='nm^-1', qz=None if qz is None else float(qz), meta=keep, label=path.name)


def fit_curve(args):
    return _fit(args)


def preview_curve(args):
    """Network proposals only (no iteration, no uncertainties): the instant first estimate."""
    return _fit(args, policy='preview')


def fit_many(args, report=None, is_cancelled=None):
    """Independent fits of several curves with the same settings.

    args: the settings of fit_curve plus 'curves': [{'q': ..., 'intensity': ..., 'sigma': ..., 'label': ...,
    and optionally any setting to override for this curve (qz, composition, ...)}, ...]; 'include_curves'
    (default False) keeps the plotted arrays of every result. Returns {'results': [...], 'n': ..., 'done': ...,
    'cancelled': bool, 'seconds': ...}; a curve that fails gives {'ok': False, 'error': {...}} and does not stop
    the others."""
    curves = list(args.get('curves') or [])
    shared = {k: v for k, v in args.items() if k not in ('curves', 'include_curves')}
    keep_curves = bool(args.get('include_curves'))
    out, t0, cancelled = [], time.perf_counter(), False
    for i, curve in enumerate(curves):
        if is_cancelled is not None and is_cancelled():
            cancelled = True
            break
        try:
            res = _fit(dict(shared, **curve))
            if not keep_curves:
                res.pop('curves', None)
        except Exception as exc:                                    # noqa: BLE001 - one bad curve must not stop a batch
            res = results.error_result(exc, _paths.KIT_VERSION)
        res['label'] = curve.get('label', i)
        out.append(res)
        if report is not None:
            ok = res.get('ok')
            report(i+1, len(curves), f"{i+1}/{len(curves)} {res['label']}: " +
                   (f"chi2/pt {res['summary']['chi2_per_point']:.3g}" if ok else f"failed ({res['error']['message']})"),
                   dict(index=i, label=str(res['label']), ok=bool(ok)))
    return dict(results=out, n=len(curves), done=len(out), cancelled=cancelled, seconds=time.perf_counter()-t0)


def run_series(args, report=None, is_cancelled=None):
    """Tracked fit of a series with research.series1d (frame-to-frame warm starts, fallbacks, backfill).

    args: files (series npz, directory or glob), recipe (path or dict; sections as GIMaP's in-situ recipe, see
    research/series1d/README_zh.md), mode ('review' | 'live' | 'batch'), out (output directory), workers, resume,
    frames ('A:B'), input_format, replay_hz. Progress is reported from the number of frames written to
    out/frames.jsonl. Cancelling means stopping the process that runs this function; resume=True continues.
    Returns {'summary': ..., 'out': ...}."""
    ensure_ready()
    from research.series1d import run
    out = Path(args.get('out') or Path.cwd()/'series1d_out').expanduser()
    frames_path = out/'frames.jsonl'
    box = {}

    def target():
        try:
            box['summary'] = run(args['files'], args['recipe'], mode=args.get('mode') or 'review', out=out,
                                 workers=int(args.get('workers') or 1), resume=bool(args.get('resume')),
                                 frames=args.get('frames'), input_format=args.get('input_format') or 'auto',
                                 replay_hz=float(args.get('replay_hz') or 10.0), quiet=True)
        except BaseException as exc:                                # noqa: BLE001 - re-raised in the caller's thread
            box['error'] = exc
    thread = threading.Thread(target=target, name='series1d', daemon=True)
    thread.start()
    last = -1
    while thread.is_alive():
        thread.join(0.5)
        if report is not None and frames_path.is_file():
            try:
                with open(frames_path, 'rb') as f:
                    n = max(sum(1 for _ in f)-1, 0)                   # first line = header
            except OSError:
                continue
            if n != last:
                last = n
                report(n, 0, f'{n} frames', dict(frames=n))
    if 'error' in box:
        raise box['error']
    return dict(summary=box['summary'], out=str(out))


OPERATIONS = dict(info=lambda args, report=None, is_cancelled=None: info(),
                  fit=lambda args, report=None, is_cancelled=None: fit_curve(args),
                  preview=lambda args, report=None, is_cancelled=None: preview_curve(args),
                  load=lambda args, report=None, is_cancelled=None: load_file(args),
                  fit_many=fit_many, series=run_series)
