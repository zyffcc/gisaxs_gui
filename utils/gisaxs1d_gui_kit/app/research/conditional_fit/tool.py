"""Fit a 1D GISAXS Yoneda cut (or a SAXS curve) with a supplied or auto-ranked composition.

    python -m research.conditional_fit.tool CUT.npz                      # both sides, recentred, auto:vc:3
    python -m research.conditional_fit.tool CUT.npz --composition vc,vc
    python -m research.conditional_fit.tool curve.csv --mode saxs --composition sphere

Pipeline: (cut files) estimate the q origin from left/right symmetry and merge
both sides -> q-rescaling into the network's length-scale window -> network
proposals at three intensity normalisations (instant) -> NNLS amplitudes ->
refinement (FastFit, parallel) from the best network heads plus random
Latin-hypercube starts, then D-scan restarts, in a wide physical domain ->
BIC ranking with "indistinguishable" flags -> physical parameters with 1-sigma
errors, domain-bound flags and per-population curves.

Cut files: the q column is q_par = sqrt(qx^2+qy^2) by default (--q-column qy
selects the qy component alone).

physics='v2' (default): every ranked candidate is evaluated in physics v2
(number-weighted sizes, Hosemann paracrystal, table engine). The cond_v1 network
proposes in the legacy codec; its proposals are converted to the refinement
codec and their amplitudes re-solved with the v2 engine before ranking, so a
result never carries legacy conventions. physics='legacy' ranks the legacy
proposals as they are.

Parallel refinement uses a 'spawn' process pool (Windows, macOS, Linux) that is
kept for later fits and closed at exit (close_pool()). The workers import this
module only, never the caller's __main__, so fit_curve can be called with
workers > 1 from any script, `python - <<EOF` or a notebook. BLAS is limited to
one thread in the workers and, during a fit, in this process.

Composition: comma list of shapes (sphere|rc|vc, repeats allowed, <=4), or
'auto[:shapes][:Kmax]', e.g. 'auto:vc:3' = 1..3 standing-cylinder populations
(default for GISAXS: exact for Yoneda cuts because the qz factor of a standing
cylinder is constant along the cut; the usual Mueller-Buschbaum-style model),
'auto:sphere:3' (default for SAXS), 'auto' = all 34. The curve alone does not
identify multi-population compositions; auto mode answers "how many
populations of the chosen shape" and flags statistically indistinguishable
alternatives. Spheres and random cylinders in GISAXS need an explicit qz.

Low-q term: C/(1+(q/sigma_res)^nu). Default nu=2 (Lorentzian, the usual
"resolution function" of such fits); --lowq-nu free lets nu vary inside
FREE_NU = [1.5, 4] (an unbounded exponent absorbs almost any low-q feature).

Scale covariance: form factors depend on qR, qh and structure factors on qD,
so with q' = s q the model parameters satisfy R = s R', h = s h', D = s D' and
sigma_res = sigma_res'/s (widths are relative).
"""
from __future__ import annotations

import argparse
import atexit
import functools
import json
import os
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from research.identifiability.run import COMBOS
from research.conditional_fit import pipeline
from research.conditional_fit.codec import CODECS, LEGACY, convert

SHAPES = {'sphere': 1, 's': 1, 'rc': 2, 'random_cylinder': 2, 'vc': 3, 'vertical_cylinder': 3, 'cylinder': 3}
NAMES = {1: 'sphere', 2: 'random_cylinder', 3: 'vertical_cylinder'}
Q_WINDOW = (1.5, 4.3)   # q_max range of the training grids (nm^-1)
D_SCAN = (0.02, 0.06, 0.12, 0.2, 0.3, 0.45, 0.6, 0.8)
DEFAULT_COMPOSITION = {'gisaxs': 'auto:vc:3', 'saxs': 'auto:sphere:3'}
FREE_NU = (1.5, 4.0)    # range of the low-q exponent when it is fitted (lowq_nu=None)


def parse_composition(text):
    text = text.strip().lower()
    if text.startswith('auto'):
        parts = text.split(':')
        allowed = {SHAPES[t] for t in parts[1].split(',')} if len(parts) > 1 and parts[1] else {1, 2, 3}
        kmax = int(parts[2]) if len(parts) > 2 else 4
        return [c for c in range(34) if set(int(t) for t in COMBOS[c] if t) <= allowed
                and int(np.sum(COMBOS[c] > 0)) <= kmax]
    types = sorted(SHAPES[t.strip()] for t in text.split(','))
    if not 1 <= len(types) <= 4:
        raise ValueError('A composition has 1..4 populations')
    padded = types+[0]*(4-len(types))
    return [int(np.flatnonzero((COMBOS == padded).all(axis=1))[0])]


def choose_scale(q):
    qmax = float(np.max(q))
    return 1.0 if Q_WINDOW[0] <= qmax <= Q_WINDOW[1] else 3.5/qmax


def load_curve(path, side='both', q_column='q_par', recenter=True, sigma_mode='poisson'):
    """Returns q, I, sigma, meta. yoneda_cut npz (+json) or csv/txt with columns q, I[, sigma].

    q_column (cut files only): 'q_par' = sqrt(qx^2+qy^2), the in-plane momentum (default), or 'qy'.
    side='both' merges the two sides of a cut after estimating the q origin from
    their symmetry (research.real_data.symmetry) and rescaling the left side by the
    fitted left/right intensity ratio; meta['recentering'] records both.
    """
    path = Path(path)
    meta = {}
    if path.suffix == '.npz':
        side_meta = path.with_suffix('.json')
        if side_meta.exists():
            meta = json.loads(side_meta.read_text())
        with np.load(path) as f:
            wanted = [f'{sd}_{q_column}' for sd in (('left', 'right') if side == 'both' else (side,))]
            if any(key not in f.files for key in wanted):
                columns = sorted({k.split('_', 1)[1] for k in f.files if k.startswith(('left_', 'right_'))})
                raise KeyError(f'{path.name} has no q column {q_column!r} for side {side!r} (columns: {columns}); '
                               f'choose it with q_column / --q-column')
            if side != 'both':
                return f[f'{side}_{q_column}'], f[f'{side}_intensity'], f[f'{side}_sigma'], meta
            left = [f[f'left_{k}'] for k in (q_column, 'intensity', 'sigma')]
            right = [f[f'right_{k}'] for k in (q_column, 'intensity', 'sigma')]
        from research.real_data.symmetry import estimate_center_shift, merge_sides, side_difference_sigma
        shift = estimate_center_shift(left[0], left[1], right[0], right[1]) if recenter else \
            dict(delta=0.0, ratio_left_over_right=1.0)
        q, y, s, sides = merge_sides(*left, *right, delta=shift['delta'], ratio=shift['ratio_left_over_right'])
        meta['recentering'] = shift
        if sigma_mode == 'side':
            # data-driven sigma: noise plus left/right systematics, never below the Poisson estimate
            sig, table = side_difference_sigma(left[0], left[1], right[0], right[1], delta=shift['delta'],
                                               ratio=shift['ratio_left_over_right'])
            s = np.maximum(sig(q, y), s)
            meta['sigma'] = dict(mode='side_difference', table=table)
        return q, y, s, meta
    table = np.loadtxt(path, delimiter=',' if path.suffix == '.csv' else None, comments='#')
    q, y = table[:, 0], table[:, 1]
    s = table[:, 2] if table.shape[1] > 2 else np.full_like(y, np.nan)
    return q, y, s, meta


def _limit_threads():
    """One BLAS thread per worker: oversubscribed BLAS threads made pooled fits ~90x slower."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def _single_threaded(func):
    """Run func with one BLAS thread in this process (restored afterwards). Multi-threaded BLAS makes
    these small least-squares problems an order of magnitude slower; the results are the same."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            return func(*args, **kwargs)
        with threadpool_limits(1):
            return func(*args, **kwargs)
    return wrapper


def _refine_task(args):
    """Pool task (module level, picklable arguments): refine one start."""
    q, y, s, cand, max_nfev, fixed, mode, tables_dir = args
    if tables_dir is not None:
        # a spawned worker imports the modules afresh: read the form-factor tables where the parent does
        from research.fast_inference import tables
        if str(tables.CACHE) != tables_dir:
            tables.CACHE = Path(tables_dir)
    return pipeline.refine(q, y, s, cand, max_nfev=max_nfev, fixed=fixed, **mode)


def _exit_with_parent():
    """In a pool worker: a daemon thread that ends the worker as soon as its parent process is gone.
    The pool is kept between fits, so a parent that is killed (SIGKILL, a timeout) would otherwise leave
    its idle workers behind for good."""
    import threading
    from multiprocessing import parent_process
    parent = parent_process()
    if parent is None:
        return

    def watch():
        parent.join()       # returns when the parent process has ended, however it ended
        os._exit(1)
    threading.Thread(target=watch, name='exit-with-parent', daemon=True).start()


def _worker_init():
    """Start-up of a refinement worker: one BLAS thread, exit when the parent is gone."""
    _limit_threads()
    _exit_with_parent()


_POOL = {}      # {'workers': n, 'executor': ProcessPoolExecutor}: the pool of this process, reused by later fits


def _executor(workers):
    """'spawn' process pool with one BLAS thread per worker, created on first use and kept for later fits
    (starting spawn workers costs about a second); a different worker count replaces it."""
    if _POOL.get('workers') != workers:
        close_pool()
        _POOL.update(workers=workers, executor=ProcessPoolExecutor(workers, mp_context=get_context('spawn'),
                                                                   initializer=_worker_init))
    return _POOL['executor']


@atexit.register
def close_pool():
    """Stop the worker processes kept by fit_curve (also runs at interpreter exit)."""
    executor = _POOL.pop('executor', None)
    _POOL.clear()
    if executor is not None:
        executor.shutdown(wait=True)


@contextmanager
def _main_not_reimported():
    """While worker processes are started: keep 'spawn' from re-importing the caller's __main__ in them.

    A spawned worker normally imports the parent's main module first. The tasks of this module reference
    importable modules only, so the workers do not need it, and a main module that cannot be imported
    again (python - <<EOF, a script without an `if __name__ == '__main__':` guard) would kill every
    worker. multiprocessing reads sys.modules['__main__'] when it starts a process; it sees an empty
    stand-in here. Not used when this module is the program itself (its workers need it as __main__)."""
    if __name__ == '__main__':
        yield
        return
    main = sys.modules.get('__main__')
    sys.modules['__main__'] = types.ModuleType('__main__')
    try:
        yield
    finally:
        if main is None:
            sys.modules.pop('__main__', None)
        else:
            sys.modules['__main__'] = main


def _run_jobs(jobs, workers):
    """Refine every job, in order: in this process (workers <= 1) or in the spawn pool."""
    if workers <= 1:
        return [_refine_task(job) for job in jobs]
    try:
        executor = _executor(workers)
        with _main_not_reimported():        # the executor starts its processes while tasks are submitted
            futures = [executor.submit(_refine_task, job) for job in jobs]
        return [future.result() for future in futures]
    except BrokenProcessPool as error:
        close_pool()
        raise RuntimeError('fit_curve: a worker process of the refinement pool stopped unexpectedly (killed or '
                           'crashed); the pool was closed. workers=1 fits in this process.') from error


def _nu_box(codec):
    """(u_lo, u_hi) of coordinate 31 for a free low-q exponent inside FREE_NU."""
    lo, hi = codec.lo_hi
    return tuple(float(np.clip((v-lo[31])/(hi[31]-lo[31]), 0.0, 1.0)) for v in FREE_NU)


def _start(cand_or_u, combo, codec, lowq_nu, source, nu_box=None):
    """A refinement start in `codec`, with the low-q exponent pinned when requested, or kept inside
    nu_box = (u_lo, u_hi) when it is free."""
    if isinstance(cand_or_u, pipeline.Candidate):
        u = convert(cand_or_u.u, cand_or_u.codec, codec)
    else:
        u = convert(np.asarray(cand_or_u, float), LEGACY, codec)
    if lowq_nu is not None:
        lo, hi = codec.lo_hi
        u[31] = np.clip((lowq_nu-lo[31])/(hi[31]-lo[31]), 0.0, 1.0)
    elif nu_box is not None:
        u[31] = np.clip(u[31], *nu_box)
    return pipeline.Candidate(combo, u, np.zeros(6), np.zeros(0), np.inf, 0, source, codec=codec)


def _evaluated(q, y, sigma, start, engine, mode):
    """An unrefined candidate: NNLS amplitudes of start.u in start.codec with the given physics engine."""
    q, y, sigma = pipeline._clean_inputs(q, y, sigma)
    fit = pipeline.make_fit(q, y, sigma, start.combo, codec=start.codec, engine=engine, **mode)
    coef, prediction, residual = fit.solve(fit.matrix_from(*fit.full(start.u)))
    return pipeline.Candidate(start.combo, start.u, coef, prediction, float(residual@residual), len(y), start.source,
                              codec=start.codec, info=dict(engine=engine))


def d_scan_starts(cand):
    """Restarts that move one population's paracrystal distance across its range.

    The weak population's D sits in a narrow basin that heads and random starts
    tend to miss; scanning D_fraction per population from the current best
    solution and refining locally recovers it (synthetic end-to-end test)."""
    out = []
    for j, typ in enumerate(COMBOS[cand.combo]):
        if not typ:
            continue
        for g in D_SCAN:
            u = cand.u.copy()
            u[7*j+4] = g
            out.append(pipeline.Candidate(cand.combo, u, cand.coef, cand.prediction, cand.chi2, cand.n_points,
                                          f'dscan{j}:{g}', codec=cand.codec))
    return out


@_single_threaded
def fit_curve(q, y, sigma=None, *, composition=None, mode='gisaxs', qz=None, model_dir=None,
              refine=True, refine_heads=None, random_starts=None, d_scan=True, max_nfev=80, domain='extended',
              lowq_nu=2.0, relative_floor=0.0, delta=10.0, workers=None, q_scale='auto', seed=0, components_top=5,
              physics='v2'):
    """Returns a dict with ranked results; see module docstring.

    physics='v2': every result has engine 'v2' and v2 parameter names (sigma_R_rel_number_weighted,
    w_rel_hosemann); 'legacy': multishape_v1 conventions. lowq_nu=None fits the low-q exponent inside FREE_NU.
    workers: processes of the refinement pool (default: up to all cores; 1 = this process only)."""
    t0 = time.perf_counter()
    q, y = np.asarray(q, float), np.asarray(y, float)
    sigma = np.sqrt(np.maximum(np.abs(y), 1.0)) if sigma is None or not np.isfinite(sigma).all() else np.asarray(sigma, float)
    keep = np.isfinite(q) & np.isfinite(y) & (q > 0) & np.isfinite(sigma) & (sigma > 0)
    q, y, sigma = q[keep], y[keep], sigma[keep]
    sigma_fit = np.sqrt(sigma**2+(relative_floor*np.abs(y))**2)
    s = choose_scale(q) if q_scale == 'auto' else float(q_scale)
    qs = q*s
    if mode == 'saxs' and qz:
        raise ValueError('SAXS mode has qz = 0')
    physics_mode = dict(qz=0.0 if mode == 'saxs' else float(qz or 0.0)*s,
                        sf_on=mode != 'saxs', res_on=mode != 'saxs')
    composition = composition or DEFAULT_COMPOSITION[mode]
    combos = parse_composition(composition) if isinstance(composition, str) else list(composition)
    model_dir = Path(model_dir or Path(__file__).resolve().parents[3]/'models'/'cond_v1_a')
    proposer = pipeline.Proposer(model_dir)
    proposals = pipeline.propose(proposer, qs, y, sigma_fit, combos, **physics_mode)
    nu = lowq_nu if physics_mode['res_on'] else None
    fixed = [31] if nu is not None else []
    free_nu = physics_mode['res_on'] and nu is None
    codec = CODECS[domain]
    nu_box = _nu_box(codec) if free_nu else None
    if physics == 'v2':
        # The network proposes in legacy physics. Rank its proposals as v2 candidates: same parameters in the
        # refinement codec (nu pinned or boxed like the refinement starts), amplitudes by NNLS with the v2
        # engine. A legacy-convention candidate must never be returned under physics='v2'.
        candidates = [_evaluated(qs, y, sigma_fit, _start(c, combo, codec, nu, c.source, nu_box), 'v2', physics_mode)
                      for combo, cands in proposals.items() for c in cands]
    else:
        candidates = [c for cands in proposals.values() for c in cands]
        if free_nu:     # a free exponent stays inside FREE_NU also for the unrefined proposals
            box = _nu_box(LEGACY)
            candidates = [c if box[0] <= c.u[31] <= box[1] else
                          _evaluated(qs, y, sigma_fit, _start(c, c.combo, LEGACY, None, c.source, box), 'legacy', physics_mode)
                          for c in candidates]
    t_network = time.perf_counter()-t0
    if refine:
        refine_mode = dict(physics_mode, codec=codec, engine=physics)
        if free_nu:
            refine_mode.update(lower={31: nu_box[0]}, upper={31: nu_box[1]})
        tables_dir = None
        if physics == 'v2':
            from research.fast_inference import tables
            tables_dir = str(tables.CACHE)
        many = len(combos) > 3
        refine_heads = refine_heads or (2 if many else 4)
        random_starts = (2 if many else 4) if random_starts is None else random_starts
        starts = []
        for combo, cands in proposals.items():
            starts += [_start(c, combo, codec, nu, c.source, nu_box) for c in cands[:refine_heads]]
            for k, u in enumerate(qmc.LatinHypercube(32, seed=seed+combo).random(random_starts)):
                starts.append(_start(u, combo, codec, nu, f'lhs{k}', nu_box))
        jobs = [(qs, y, sigma_fit, c, max_nfev, fixed, refine_mode, tables_dir) for c in starts]
        workers = workers or min(len(jobs)*2, os.cpu_count() or 1)
        stage1 = _run_jobs(jobs, workers)
        candidates += stage1
        # stage 2: D-scan restarts from the best refined solution of each composition (SF on only)
        if physics_mode['sf_on'] and d_scan:
            best = {}
            for c in stage1:
                if c.combo not in best or c.chi2 < best[c.combo].chi2:
                    best[c.combo] = c
            jobs = [(qs, y, sigma_fit, start, max_nfev, fixed, refine_mode, tables_dir)
                    for c in best.values() for start in d_scan_starts(c)]
            candidates += _run_jobs(jobs, workers)
    ranked = pipeline.rank(candidates, delta=delta)
    results = []
    for i, entry in enumerate(ranked):
        c = entry['candidate']
        phys = c.physical()
        for particle in phys['particles']:
            for key in ('R_nm', 'R_number_mean_nm', 'D_nm', 'w_hosemann_nm', 'h_nm', 'R_nm_err', 'D_nm_err', 'h_nm_err'):
                # lengths scale with the q rescaling; relative widths do not
                if particle.get(key) is not None:
                    particle[key] *= s
        phys['low_q_sigma_res_nm_inv'] /= s
        at_bound = [pipeline.coordinate_name(k) for k in pipeline.boundary_coordinates(c, physics_mode) if k not in fixed]
        result = dict(composition=[NAMES[int(t)] for t in COMBOS[c.combo] if t],
                      combo=int(c.combo), source=c.source, refined=c.source.startswith('refined'),
                      chi2=c.chi2, chi2_per_point=c.chi2/c.n_points,
                      bic=entry['bic'], delta_bic=entry['delta_bic'], indistinguishable_from_best=entry['indistinguishable'],
                      parameters=phys, parameters_at_domain_bound=at_bound, prediction=c.prediction.tolist(),
                      u=c.u.tolist(), codec=getattr(c.codec, 'name', 'legacy'), engine=c.info.get('engine', 'legacy'))
        if i < components_top:
            result['components'] = {k: v.tolist() for k, v in pipeline.components(qs, c, **physics_mode).items()}
        results.append(result)
    return dict(q=q.tolist(), intensity=y.tolist(), sigma=sigma.tolist(), sigma_fit=sigma_fit.tolist(),
                mode=mode, qz=qz, q_scale=s, refinement_domain=domain, physics=physics, lowq_nu=nu, relative_floor=relative_floor,
                delta_bic=delta, composition_request=composition, network_seconds=t_network,
                total_seconds=time.perf_counter()-t0, n_compositions=len(combos), results=results,
                note=('Compositions flagged indistinguishable fit about as well (delta BIC <= threshold); the '
                      'curve alone cannot decide between them. physics=v2: number-weighted Gaussian sizes, '
                      'Hosemann paracrystal w = w_rel*D; physics=legacy: intensity-weighted sizes, sigma_D model '
                      'parameter (w = sqrt(2 pi) sigma_D D). The network proposes in legacy physics; under '
                      'physics=v2 its proposals are re-evaluated in v2 before ranking (every result is v2). '
                      'A free low-q exponent (lowq_nu null) is fitted inside [1.5, 4]. Errors are '
                      '1-sigma Gauss-Newton estimates scaled by the reduced chi2.'))


def plot(result, path, top=3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    q, y = np.array(result['q']), np.array(result['intensity'])
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, gridspec_kw=dict(height_ratios=(3, 1)))
    pos = y > 0
    ax.loglog(q[pos], y[pos], '.', color='#52514e', ms=2, label='data')
    for r, colour in zip(result['results'][:top], ('#2a78d6', '#eb6834', '#1baf7a')):
        p = np.array(r['prediction'])
        label = '+'.join(r['composition'])+f" chi2/pt={r['chi2_per_point']:.2f}"+(' *' if r['indistinguishable_from_best'] else '')
        ax.loglog(q, p, lw=1.2, color=colour, label=label)
        bx.semilogx(q, (y-p)/np.array(result['sigma_fit']), '.', ms=1, color=colour)
    best = result['results'][0]
    for name, curve in best.get('components', {}).items():
        curve = np.array(curve)
        if np.any(curve > 0):
            ax.loglog(q, np.where(curve > 0, curve, np.nan), ':', lw=0.8, color='#52514e')
    ax.legend(fontsize=7)
    bx.set_xlabel('q (nm$^{-1}$)')
    bx.set_ylabel('residual/σ')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('curve', type=Path)
    p.add_argument('--side', default='both', choices=('both', 'left', 'right'))
    p.add_argument('--no-recenter', action='store_true')
    p.add_argument('--q-column', default='q_par', choices=('q_par', 'qy'),
                   help='q column of a cut file: q_par = sqrt(qx^2+qy^2) (default) or qy')
    p.add_argument('--sigma', default='poisson', choices=('poisson', 'side'),
                   help='side: max(Poisson, left/right-difference sigma) for two-sided cuts')
    p.add_argument('--composition', default=None, help="default: auto:vc:3 (gisaxs) / auto:sphere:3 (saxs)")
    p.add_argument('--mode', default='gisaxs', choices=('gisaxs', 'saxs'))
    p.add_argument('--qz', type=float, default=None, help='defaults to the cut metadata qz_at_yoneda_mean')
    p.add_argument('--model-dir', type=Path, default=None)
    p.add_argument('--no-refine', action='store_true')
    p.add_argument('--lowq-nu', default='2', help="Lorentzian exponent of the low-q term; 'free' to fit it in [1.5, 4]")
    p.add_argument('--random-starts', type=int, default=None)
    p.add_argument('--relative-floor', type=float, default=0.0)
    p.add_argument('--physics', default='v2', choices=('v2', 'legacy'),
                   help='v2: number-weighted sizes + Hosemann paracrystal (table engine); legacy: multishape_v1')
    p.add_argument('--domain', default='extended', choices=('extended', 'legacy'),
                   help='refinement parameter box; legacy = network training domain')
    p.add_argument('--q-scale', default='auto')
    p.add_argument('--q-min', type=float, default=None)
    p.add_argument('--out', type=Path, default=None)
    a = p.parse_args()
    q, y, s, meta = load_curve(a.curve, a.side, a.q_column, recenter=not a.no_recenter, sigma_mode=a.sigma)
    if 'recentering' in meta:
        r = meta['recentering']
        print(f"{a.q_column} origin shift {r['delta']:.5f} nm^-1 ({r.get('delta_in_q_steps', 0):.2f} points); "
              f"left/right mismatch {r.get('mismatch_before', float('nan')):.3f} -> {r.get('mismatch_after', float('nan')):.3f}; "
              f"left/right ratio {r['ratio_left_over_right']:.3f}")
    if a.q_min:
        keep = q >= a.q_min
        q, y, s = q[keep], y[keep], s[keep]
    qz = a.qz if a.qz is not None else meta.get('qz_at_yoneda_mean')
    result = fit_curve(q, y, s, composition=a.composition, mode=a.mode, qz=qz, model_dir=a.model_dir,
                       refine=not a.no_refine, relative_floor=a.relative_floor, q_scale=a.q_scale, domain=a.domain,
                       lowq_nu=None if a.lowq_nu == 'free' else float(a.lowq_nu), random_starts=a.random_starts,
                       physics=a.physics)
    result['input'] = dict(file=str(a.curve), side=a.side, recentering=meta.get('recentering'))
    out = a.out or a.curve.with_name(a.curve.stem+f'_{a.side}_fit')
    Path(str(out)+'.json').write_text(json.dumps(result, indent=1))
    plot(result, str(out)+'.png')
    for r in result['results'][:5]:
        print(('*' if r['indistinguishable_from_best'] else ' '), '+'.join(r['composition']),
              f"chi2/pt={r['chi2_per_point']:.3f}", f"dBIC={r['delta_bic']:.1f}", r['source'],
              ('AT DOMAIN BOUND: '+', '.join(r['parameters_at_domain_bound'])) if r['parameters_at_domain_bound'] else '')


if __name__ == '__main__':
    main()
