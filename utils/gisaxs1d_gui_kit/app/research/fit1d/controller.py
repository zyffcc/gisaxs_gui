"""Adaptive per-composition fitting controller of fit1d (PLAN_zh.md §3, M2).

Everything runs in the curve's own CurveCodec(q_min, q_max) wrapped by the constraint codec, with the
product physics v2 only: TabFit(weighting='number', sf='hosemann'), qz for spheres/random cylinders.

Stages (stop_stage = last stage that ran):
  S0  network proposals: the 8 cond_v2 heads, projected into the constrained box, amplitudes by NNLS
  S1  starts = the best `heads` (default 2) network heads, plus (K >= 2) one start built greedily without
      the network (gridsearch.Aligner.pursuit). Every start: (R, D) grid alignment of each sphere /
      standing-cylinder population (gridsearch.Aligner.align, 2 sweeps) -> first_steps LM evaluations
      (default 8) -> one more alignment sweep -> first_steps more evaluations when the sweep improved chi2.
      DEV (160 non-rc rows, 2026-10-01): chi2 within the same-basin tolerance of the LM-from-truth optimum
      for 65 % (plain 2 heads x 40) -> 82 % (aligned 2 heads + pursuit); align=False, pursuit=False restore
      the plain behaviour
  S2  heads whose chi2 is still falling (relative drop of the running-minimum chi2 over the last 2
      evaluations > 1 % while the LM budget was exhausted) and whose chi2 is within 1.25 x the best
      continue in warm-started chunks of first_steps evaluations, up to max_steps in total
  S3  robust search (policy 'robust', or 'auto' when a state signal fired): all 8 heads (the S1 heads
      again, now in one uninterrupted run) and 16 Latin-hypercube starts (seeded), max_steps evaluations
      each, every start as align -> 12 evaluations -> align -> the rest (aligned_lm). Note: LM end points of these problems are chaotic (NNLS active-set changes, bounds): a start
      vector that differs by 1e-16, or an NNLS solver that differs by 1e-12, can end in another basin
      (DEV row 86, head 7: chi2 331.5 vs 14073.7). An external "8 heads x 40 steps" run (FastFitterV2) is
      therefore not reproduced bit by bit, and S3 can end above or below it on single curves
  S4  D-scan restarts from the best solution so far: each population's D coordinate set to each value
      of tool.D_SCAN, max_steps evaluations each (only with the structure factor on and D not fixed)
State signals, evaluated after S2 on the refined heads (PLAN_zh.md §3.3):
  still_falling      the best refined head was still falling when it reached max_steps
  heads_disagree     the two best refined heads have chi2 within 10 % but a reportable parameter
                     (R_nm, D_nm of a population; populations matched by type, then sorted by R) differs
                     by > 2 sigma (larger of the two 1-sigma, each scaled by sqrt(max(1, chi2/dof)))
  size_at_bound      a free R, h or D coordinate of a population with amplitude > 0 at a bound (within
                     BOUND_TOL = 1e-3 of the box width: the trust-region optimiser keeps its iterates
                     strictly inside the box, so a coordinate pressed against a bound stops ~1e-5 short)
  zero_amplitude     the NNLS amplitude of a population is 0
  no_head_fits       (fit1d 1.4, from S0) no network proposal describes the curve: for every head the chi2
                     per point with a 3 % relative error added to sigma (HEAD_FIT_FLOOR; amplitudes as
                     solved) is >= HEAD_FIT_LIMIT = 2. DEV, 281 curves (2026-10-06): the fast path reaches
                     chi2_ok on 95 % of the curves where a head fits and on 62 % of the others; the signal
                     covers about 9 of 10 fast-path misses and fires on half of the curves
Relative widths or w at a bound and a large chi2 do not trigger S3 (they only set flags).
The result of a composition is the lowest chi2 over everything evaluated (S0..S4), so the robust
stages never replace a solution by a worse one.
After the last stage, alternatives() compares the best solution with every other finished LM solution
whose chi2 lies within the PLAN same-basin tolerance max(25, 0.02 N) max(1, chi2/N): an R_nm or D_nm
that differs by > MULTIMODAL_SIGMA (5, tuned on DEV) sigma is reported as multimodal (flag
weakly_constrained), with the alternative values. With S0-S2 only, the other refined head is the only
possible alternative; the check is informative mainly after the robust stages.
Uncertainties: FastFit.run returns pinv(J^T J), which gives zero variance along directions the data do
not constrain at all (a population with amplitude 0, an exactly degenerate direction); scaled_covariance
marks those coordinates as unconstrained (infinite variance) instead.
Parallelism: workers > 1 runs S3/S4 jobs in a 'spawn' process pool (module-level worker function, one
BLAS thread per worker, the table directory of the parent passed to every worker); workers = 1 is fully
serial and deterministic. No pool is created, and the jobs run serially, inside a child process (e.g. a
script without the __main__ guard that a spawn worker re-imports) and when the main script is not a file
(piped to "python -"): in both cases a pool would respawn failing workers forever.
"""
from __future__ import annotations

import atexit
import itertools
import os
import time
from pathlib import Path
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls

from research.conditional_fit.fastfit import FastFit, powerlaw_u
from research.conditional_fit.tool import D_SCAN
from research.fast_inference.tabfit import TabFit
from research.multishape_v1 import physics
from research.multishape_v1.physics import COMBOS

FALL_THRESHOLD = 0.01        # relative drop over the last 2 LM evaluations that counts as "still falling"
CONTINUE_WITHIN = 1.25       # S2 continues only heads with chi2 <= this x the best
HEADS_CHI2_CLOSE = 0.10      # heads_disagree needs relative chi2 difference below this
HEAD_FIT_FLOOR = 0.03        # no_head_fits: relative error added to sigma in quadrature
HEAD_FIT_LIMIT = 2.0         # no_head_fits: chi2 per point (with the floor) of the best network proposal
DISAGREE_SIGMA = 2.0
LHS_STARTS = 16
BOUND_TOL = 1e-3             # fraction of the box width that counts as "at the bound" (see module docstring)
DWBA_STEPS = 30                 # LM iterations per chunk of the DWBA refinement stage S5
DWBA_MAX_STEPS = 120            # ... continued in chunks while the chi2 is still falling
ALIGN_FIRST_STEPS = 12          # LM evaluations between the two (R, D) alignments of a start
NULL_VARIANCE = 1e-18        # variance (u^2) at or below this is not a measurement: the direction is unconstrained
NULL_COMPONENT = 1e-3        # eigenvector component above which a coordinate belongs to an unconstrained direction
MULTIMODAL_DCHI2 = (25.0, 0.02)   # PLAN same-basin chi2 tolerance: max(25, 0.02 N) max(1, chi2/N)
# A reportable parameter of a chi2-equivalent solution that differs by more than this many sigma is multimodal.
# Tuned on DEV (140 K = 1 sphere/vc rows < 1000, policy robust, 2026-10-01; "wrong" = R off the truth by more
# than max(10 %, 3 sigma)): 2 sigma flags 33/45 wrong and 32/95 correct R; 5 sigma 31/45 and 17/95; 10 sigma
# 28/45 and 12/95. 5 keeps the catch rate and halves the flags on correct results.
MULTIMODAL_SIGMA = 5.0
RIDGE = 1e-8                 # the ridge of FastFit.solve
_KEYS = itertools.count()


# ---------------------------------------------------------------- physics object
class TracedFit(TabFit):
    """TabFit (physics v2) that records chi2 of every LM function evaluation and can hold B fixed.

    FastFit.run evaluates the model at a trial point through full() -> matrix_from() -> solve(); the
    Jacobian columns call solve() without full(). full() arms a flag so the next solve() is logged."""

    def __init__(self, q, y, sigma, types, *, qz, sf_on, res_on, codec, lowq, background=None, dwba=None):
        super().__init__(q, y, sigma, types, qz=qz, sf_on=sf_on, res_on=res_on, codec=codec,
                         weighting='number', sf='hosemann', lowq=lowq, dwba=dwba)
        self.background = None if background is None else float(background)
        self.trace = []
        self._armed = False
        self._inv_sigma = 1.0/self.sigma
        self._set_target()

    def _set_target(self):
        target = self.y if self.background is None else self.y-self.background
        self._response = target*self._inv_sigma

    def full(self, u):
        self._armed = True
        return super().full(u)

    def _nnls(self, m):
        """Ridged NNLS of FastFit.solve (same objective: unit-scaled columns, ridge 1e-8) solved on the
        k x k Cholesky factor of the normal matrix instead of the N x k design (about 2x faster; agrees
        with FastFit.solve to ~1e-12 relative). Falls back to FastFit.solve if the factorisation fails."""
        design = m*self._inv_sigma[:, None]
        gram = design.T@design
        rhs = design.T@self._response
        scale = np.sqrt(np.diag(gram))
        valid = scale > 1e-25
        coef = np.zeros(m.shape[1])
        if valid.any():
            sv = scale[valid]
            g = gram[np.ix_(valid, valid)]/np.outer(sv, sv)+RIDGE*np.eye(int(valid.sum()))
            try:
                lower = np.linalg.cholesky(g)
                c = solve_triangular(lower, rhs[valid]/sv, lower=True, check_finite=False)
                x = nnls(lower.T, c, maxiter=2000)[0]
            except (np.linalg.LinAlgError, RuntimeError, ValueError):
                return None
            if not np.all(np.isfinite(x)):
                return None
            coef[valid] = x/sv
        return coef

    def solve(self, m):
        m_used = m
        if self.background is not None:
            m_used = m.copy()
            m_used[:, 4] = 0.0
        coef = self._nnls(m_used)
        if coef is None:                                     # rare: exact FastFit path
            y = self.y
            try:
                if self.background is not None:
                    self.y = y-self.background
                coef = FastFit.solve(self, m_used)[0]
            finally:
                self.y = y
        if self.background is not None:
            coef[4] = self.background
        prediction = m@coef if self.background is not None else m_used@coef
        residual = (prediction-self.y)*self._inv_sigma
        if self._armed:
            self.trace.append(float(residual@residual))
            self._armed = False
        return coef, prediction, residual


@dataclass
class Setup:
    """Everything a (worker) process needs to rebuild the fit of one composition."""
    q: np.ndarray
    y: np.ndarray
    s: np.ndarray
    combo: int
    qz: float
    sf_on: bool
    res_on: bool
    lowq: str                      # 'lorentz' | 'powerlaw' | 'none'
    codec: object                  # constraints.ConstrainedCodec
    fixed: tuple
    lower: dict = field(default_factory=dict)
    upper: dict = field(default_factory=dict)
    background: float | None = None
    u31: float | None = None       # power-law exponent coordinate start (or fixed value)
    key: str = ''
    dwba: object = None            # research.fast_inference.tabfit.YonedaDWBA: spheres use the 4-term DWBA (stage S5)

    @property
    def types(self):
        return COMBOS[self.combo]

    def make_fit(self):
        return TracedFit(self.q, self.y, self.s, self.types, qz=self.qz,
                         sf_on=self.sf_on, res_on=self.res_on, codec=self.codec,
                         lowq='powerlaw' if self.lowq == 'powerlaw' else 'lorentz', background=self.background,
                         dwba=self.dwba)

    def prepare(self, u):
        """Start vector: power-law exponent coordinate set/clipped to its box."""
        u = np.array(u, float, copy=True)
        if self.res_on and self.lowq == 'powerlaw':
            if 31 in self.fixed:
                u[31] = self.u31
            else:
                u[31] = float(np.clip(self.u31, self.lower.get(31, 0.0), self.upper.get(31, 1.0)))
        return u

    def project(self, u_base):
        """A base-codec vector (network head) projected into the constrained box."""
        return self.prepare(self.codec.encode(self.codec.base.decode(u_base)))

    def free_coords(self):
        """Free nonlinear u indices (the rule of FastFit.__init__ minus the fixed ones)."""
        active = physics.active_mask(self.types).astype(bool)
        active[[6, 13, 20, 27, 28, 29]] = False
        if not self.sf_on:
            for j in range(4):
                active[[7*j+4, 7*j+5]] = False
        if not self.res_on:
            active[[30, 31]] = False
        elif self.lowq == 'powerlaw':
            active[30] = False
        return [int(c) for c in np.flatnonzero(active) if int(c) not in set(self.fixed)]


# ---------------------------------------------------------------- one LM run
def falling(trace, status):
    """Still falling: budget exhausted (status 0) and running-min chi2 dropped > 1 % over the last 2 evals."""
    t = list(trace)
    if len(t) > 1 and t[-1] in t[:-1]:
        t = t[:-1]                           # FastFit.run re-evaluates the returned point: not a new step
    if status != 0 or len(t) < 3:
        return False
    best = np.minimum.accumulate(t)
    return bool((best[-3]-best[-1]) > FALL_THRESHOLD*max(best[-1], 1e-300))


def lm_run(fit, setup, u0, max_nfev):
    """One FastFit.run with the chi2 trace; returns a plain dict."""
    fit.trace = []
    ev0 = fit.evaluations
    r = fit.run(u0, fixed=setup.fixed, lower=setup.lower, upper=setup.upper, max_nfev=int(max_nfev))
    trace = list(fit.trace)
    return dict(u=r['u'], coef=r['coef'], chi2=float(r['chi2']), cov_u=r['cov_u'], coords=list(r['coords']),
                status=int(r['status']), nfev=len(trace), evaluations=int(fit.evaluations-ev0),
                falling=falling(trace, r['status']))


def make_aligner(fit, setup):
    """(R, D) grid-search helper of a fit (research.fit1d.gridsearch): screening on a rebinned copy of the curve."""
    from research.fit1d.fastgrid import FastAligner
    from research.fit1d.gridsearch import screen_curve
    vq, vy, vs = screen_curve(setup.q, setup.y, setup.s)
    screen = fit if len(vq) == len(setup.q) else replace(setup, q=vq, y=vy, s=vs, key=setup.key+'-screen').make_fit()
    return FastAligner(fit, screen, fixed=setup.fixed)


def _sum_runs(a, b):
    """LM result b that continued result a: counters added."""
    out = dict(b)
    out['nfev'] = a['nfev']+b['nfev']
    out['evaluations'] = a['evaluations']+b['evaluations']
    return out


def aligned_lm(fit, setup, aligner, u0, max_nfev, first=ALIGN_FIRST_STEPS):
    """align -> LM(first) -> align -> LM(rest): one uninterrupted robust start with (R, D) alignment."""
    u1 = aligner.align(u0, sweeps=2)[0]
    first = int(min(first, max_nfev))
    r = lm_run(fit, setup, u1, first)
    if max_nfev > first:
        u2 = aligner.align(r['u'], sweeps=1)[0]
        r = _sum_runs(r, lm_run(fit, setup, u2, max_nfev-first))
    return r


_WORKER_FITS: dict = {}
_WORKER_ERROR: list = []


def _worker_init(tables_dir=None):
    """Pool initializer: one BLAS thread, and the parent's table directory (a spawn worker would otherwise
    fall back to the default directory when the parent was pointed elsewhere). It must not raise (a failing
    initializer makes the pool respawn workers forever): errors are kept and raised by the first job."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:          # pragma: no cover
        pass
    try:
        from research.fit1d import schema
        schema.check_tables(tables_dir)
    except Exception as exc:     # noqa: BLE001 - reported through _worker_job
        _WORKER_ERROR.append(f'{type(exc).__name__}: {exc}')


def _worker_job(args):
    """Module-level pool task: (setup, u0, max_nfev) -> lm_run result (fit cached per setup key)."""
    if _WORKER_ERROR:
        raise RuntimeError(f'fit1d worker could not start: {_WORKER_ERROR[0]}')
    setup, u0, max_nfev = args[:3]
    aligned = len(args) > 3 and args[3]
    fit = _WORKER_FITS.get(setup.key)
    if fit is None:
        _WORKER_FITS.clear()
        fit = _WORKER_FITS[setup.key] = setup.make_fit()
        _WORKER_FITS[setup.key+'#aligner'] = None
    if not aligned:
        return lm_run(fit, setup, u0, max_nfev)
    aligner = _WORKER_FITS.get(setup.key+'#aligner')
    if aligner is None:
        aligner = _WORKER_FITS[setup.key+'#aligner'] = make_aligner(fit, setup)
    return aligned_lm(fit, setup, aligner, u0, max_nfev)


def _worker_probe(_=None):
    """Module-level pool task (tests, diagnostics): the worker's table directory, BLAS threads, start error."""
    from research.fast_inference import tables
    threads = None
    try:
        from threadpoolctl import threadpool_info
        threads = max((int(i.get('num_threads', 1)) for i in threadpool_info()), default=None)
    except ImportError:          # pragma: no cover
        pass
    return dict(pid=os.getpid(), tables_dir=str(tables.CACHE), blas_threads=threads,
                error=_WORKER_ERROR[0] if _WORKER_ERROR else None)


def _worker_effective_workers(workers):
    """Module-level pool task (tests): what effective_workers(workers) is inside a worker (always 1)."""
    return effective_workers(workers)


_POOLS: dict = {}


def close_pool(workers, tables_dir=None):
    """Terminate and forget one pool (get_pool key)."""
    pool = _POOLS.pop((int(workers), None if tables_dir is None else str(Path(tables_dir))), None)
    if pool is not None:
        pool.terminate()
        pool.join()


def in_child_process():
    """True inside a multiprocessing child (pool worker), including the phase in which a spawn child
    re-imports the main script (parent_process() is still None then): never start a pool from there."""
    import multiprocessing
    return (multiprocessing.parent_process() is not None
            or bool(getattr(multiprocessing.current_process(), '_inheriting', False)))


def spawn_safe_main():
    """False when spawn children could not start because they would re-run a main script that is not a
    file (a script piped to "python -": __main__.__file__ is '<stdin>'); the pool would respawn forever."""
    import sys
    main = sys.modules.get('__main__')
    if getattr(getattr(main, '__spec__', None), 'name', None) is not None:      # python -m package
        return True
    path = getattr(main, '__file__', None)
    return path is None or os.path.exists(path)            # no __file__: interactive session, nothing re-run


def get_pool(workers, tables_dir=None):
    """Process-wide 'spawn' pool per (worker count, table directory); created lazily, closed at exit."""
    key = (int(workers), None if tables_dir is None else str(Path(tables_dir)))
    pool = _POOLS.get(key)
    if pool is None:
        from multiprocessing import get_context
        pool = get_context('spawn').Pool(int(workers), initializer=_worker_init, initargs=(key[1],))
        _POOLS[key] = pool
    return pool


@atexit.register
def shutdown_pools():
    for pool in _POOLS.values():
        pool.terminate()
    _POOLS.clear()


def effective_workers(workers):
    """Workers actually used: 1 inside a child process (no nested pools) or when the main script cannot be
    re-imported by spawn children (spawn_safe_main), else the request."""
    return 1 if workers <= 1 or in_child_process() or not spawn_safe_main() else int(workers)


def run_jobs(jobs, workers, serial_fits, tables_dir=None):
    """jobs: [(setup, u0, max_nfev)]. workers == 1: in-process with the composition's own fit objects."""
    if effective_workers(workers) <= 1 or len(jobs) <= 1:
        out = []
        for job in jobs:
            s, u0, n = job[:3]
            fit = serial_fits[s.key]
            if len(job) > 3 and job[3]:
                key = s.key+'#aligner'
                if serial_fits.get(key) is None:
                    serial_fits[key] = make_aligner(fit, s)
                out.append(aligned_lm(fit, s, serial_fits[key], u0, n))
            else:
                out.append(lm_run(fit, s, u0, n))
        return out
    return get_pool(workers, tables_dir).map(_worker_job, jobs, chunksize=1)


# ---------------------------------------------------------------- uncertainties
def scaled_covariance(cov_u, chi2, dof):
    """Covariance of the free coordinates: (J^T J)^-1 max(1, chi2/dof), unconstrained coordinates marked.

    FastFit.run returns pinv(J^T J) chi2/dof. pinv assigns zero variance to directions with a vanishing
    Jacobian (population with amplitude 0, low-q term with C = 0, exact degeneracies), which would read as
    a perfectly known parameter. Coordinates with variance <= NULL_VARIANCE, or with a component in an
    eigen-direction of such variance, are unconstrained: their rows/columns are zeroed and the diagonal
    set to inf (propagate() then returns inf for every quantity that depends on them)."""
    if cov_u is None or not np.size(cov_u):
        return None
    cov = np.array(cov_u, float)
    if 0 < chi2 < dof:
        cov = cov*(dof/chi2)
    if not np.all(np.isfinite(cov)):
        null = np.ones(len(cov), bool)
    else:
        null = ~(np.diag(cov) > NULL_VARIANCE)
        if not null.all():
            values, vectors = np.linalg.eigh(0.5*(cov+cov.T))
            for k in np.flatnonzero(values <= NULL_VARIANCE):
                null |= np.abs(vectors[:, k]) > NULL_COMPONENT
    if null.any():
        cov[null, :] = 0.0
        cov[:, null] = 0.0
        cov[null, null] = np.inf
    return cov


def gradient(func, u, coords, step=1e-6):
    """d func(u) / d u[coords] (one-sided, inward at the box edges); func returns a 1-D array."""
    base = np.asarray(func(u), float)
    grad = np.zeros((len(base), len(coords)))
    for i, c in enumerate(coords):
        h = step if u[c] <= .5 else -step
        u2 = u.copy()
        u2[c] += h
        grad[:, i] = (np.asarray(func(u2), float)-base)/h
    return base, grad


def propagate(func, u, coords, cov):
    """(values, 1-sigma) of func(u) under the covariance of u[coords] (scaled_covariance).

    sigma is NaN when no covariance is available and inf for a quantity that depends on an unconstrained
    coordinate (infinite diagonal of cov)."""
    if cov is None or not len(coords):
        values = np.asarray(func(u), float)
        return values, np.full(len(values), np.nan)
    values, grad = gradient(func, u, coords)
    null = np.isinf(np.diag(cov))
    finite = np.where(np.isfinite(cov), cov, 0.0) if null.any() else cov
    var = np.einsum('ij,jk,ik->i', grad, finite, grad)
    sigma = np.sqrt(np.where(var >= 0, var, np.nan))
    if null.any():
        sigma = np.where(np.any(grad[:, null] != 0, axis=1), np.inf, sigma)
    return values, sigma


# ---------------------------------------------------------------- composition controller
class CompositionRun:
    """S0..S4 for one composition (see module docstring)."""

    def __init__(self, setup, heads_u, *, n_heads=2, first_steps=8, max_steps=40, seed=0, align=True, pursuit=True):
        self.setup = setup
        self.setup.key = self.setup.key or f'{os.getpid()}-{next(_KEYS)}-{setup.combo}'
        self.fit = setup.make_fit()
        self.heads_u = np.asarray(heads_u, float)
        self.n_heads = int(max(0, min(n_heads, len(self.heads_u))))
        self.first_steps, self.max_steps = int(first_steps), int(max_steps)
        self.seed = int(seed)
        self.candidates = []            # every evaluated solution
        self.heads = []                 # S0 candidates, best first
        self.tracks = []                # refined heads (S1/S2): current state per head
        self.stage_seconds = {}
        self.stages_run = []
        self.signals = {}
        self.fired = []
        self.n = len(setup.y)
        self.free = setup.free_coords()
        self.born = None                # set by dwba_refine()
        searchable = bool(np.any(np.isin(setup.types, (1, 3))))
        self.align = bool(align) and searchable
        self.pursuit = bool(pursuit) and self.align and int(np.sum(setup.types > 0)) >= 2
        self._aligner = None

    # -- bookkeeping ------------------------------------------------------------
    def _add(self, stage, source, r):
        cand = dict(stage=stage, source=source, **r)
        self.candidates.append(cand)
        return cand

    @property
    def best(self):
        # lowest chi2; on exact ties an LM solution (it carries a covariance) before an S0 proposal, then
        # the earliest (stable order)
        return min(self.candidates, key=lambda c: (c['chi2'], c['stage'] == 'S0'))

    def k_params(self, cand):
        """Free nonlinear coordinates + positive linear coefficients (B excluded when fixed)."""
        coef = np.asarray(cand['coef'])
        linear = int(np.sum(coef[:4] > 0))+int(coef[5] > 0 and self.setup.res_on)
        if self.setup.background is None:
            linear += int(coef[4] > 0)
        return len(self.free)+linear

    def bic(self, cand):
        return cand['chi2']+self.k_params(cand)*np.log(self.n)

    def dof(self, cand):
        return max(self.n-self.k_params(cand), 1)

    def at_bound(self, cand):
        """[(u index, 'lower'|'upper')] of free coordinates on an edge of their box."""
        out = []
        u = cand['u']
        for c in self.free:
            lo, hi = self.setup.lower.get(c, 0.0), self.setup.upper.get(c, 1.0)
            x = (u[c]-lo)/max(hi-lo, 1e-300)
            if x < BOUND_TOL:
                out.append((c, 'lower'))
            elif x > 1-BOUND_TOL:
                out.append((c, 'upper'))
        return out

    @property
    def aligner(self):
        if self._aligner is None:
            self._aligner = make_aligner(self.fit, self.setup)
        return self._aligner

    # -- stages -----------------------------------------------------------------
    def stage0(self, network_seconds=0.0):
        t0 = time.perf_counter()
        ev0 = self.fit.evaluations
        cands = []
        sig = np.asarray(self.setup.s, float)
        shrink = sig/np.sqrt(sig*sig+(HEAD_FIT_FLOOR*np.abs(np.asarray(self.setup.y, float)))**2)   # residual / sigma_eff
        for h, u in enumerate(self.heads_u):
            uc = self.setup.project(u)
            self.fit.trace = []
            coef, prediction, residual = self.fit.solve(self.fit.matrix_from(*self.fit.full(uc)))
            eff = residual*shrink
            cands.append(self._add('S0', f'head{h}', dict(u=uc, coef=coef, chi2=float(residual@residual),
                                                          cov_u=None, coords=[], status=None, nfev=1,
                                                          evaluations=1, falling=False,
                                                          head_fit=float(eff@eff)/max(self.n, 1))))
        self.fit.evaluations = ev0+len(cands)
        self.heads = sorted(cands, key=lambda c: c['chi2'])
        self.stage_seconds['S0'] = network_seconds+time.perf_counter()-t0
        self.stages_run.append('S0')

    def stage12(self):
        t0 = time.perf_counter()
        self.tracks = []
        first = min(self.first_steps, self.max_steps)
        starts = [(cand['source'], cand['u']) for cand in self.heads[:self.n_heads]]
        align_seconds = 0.0
        if self.pursuit and self.heads:
            ta = time.perf_counter()
            starts.append(('pursuit', self.aligner.pursuit(self.heads[0]['u'])))
            align_seconds += time.perf_counter()-ta
        for source, u0 in starts:
            if self.align:
                # (R, D) grid alignment before and once after the first LM chunk (research.fit1d.gridsearch)
                ta = time.perf_counter()
                u0 = self.aligner.align(u0, sweeps=2)[0]
                align_seconds += time.perf_counter()-ta
            r = lm_run(self.fit, self.setup, u0, first)
            steps = first
            if self.align and self.max_steps > first:
                ta = time.perf_counter()
                u1, _, improved = self.aligner.align(r['u'], sweeps=1)
                align_seconds += time.perf_counter()-ta
                if improved:
                    n = min(self.first_steps, self.max_steps-steps)
                    r = _sum_runs(r, lm_run(self.fit, self.setup, u1, n))
                    steps += n
            track = self._add('S1', f"{source}+lm", r)
            track['steps'] = steps
            self.tracks.append(track)
        self.stage_seconds['S1'] = time.perf_counter()-t0
        self.stage_seconds['align'] = align_seconds
        self.stages_run.append('S1')
        t0 = time.perf_counter()
        ran = False
        while True:
            best = min(t['chi2'] for t in self.tracks) if self.tracks else np.inf
            active = [i for i, t in enumerate(self.tracks) if t['falling'] and t['steps'] < self.max_steps
                      and t['chi2'] <= CONTINUE_WITHIN*best]
            if not active:
                break
            ran = True
            for i in active:
                t = self.tracks[i]
                n = min(self.first_steps, self.max_steps-t['steps'])
                r = lm_run(self.fit, self.setup, t['u'], n)
                new = self._add('S2', t['source'].split('+')[0]+'+lm', r)
                new['steps'] = t['steps']+n
                new['nfev_total'] = t.get('nfev_total', t['nfev'])+r['nfev']
                self.tracks[i] = new
        if ran:
            self.stage_seconds['S2'] = time.perf_counter()-t0
            self.stages_run.append('S2')

    def compute_signals(self, reportable):
        """State signals after S2 (see module docstring). reportable(cand) -> list of population dicts
        with 'type', 'R_nm', 'R_nm_err', 'D_nm', 'D_nm_err', 'amplitude'."""
        sig = dict(still_falling=False, heads_disagree=False, size_at_bound=False, zero_amplitude=False, no_head_fits=False)
        detail = {}
        fits = [c['head_fit'] for c in self.heads if c.get('head_fit') is not None]
        if fits:
            detail['best_head_chi2_per_point_with_floor'] = float(min(fits))
            sig['no_head_fits'] = bool(min(fits) >= HEAD_FIT_LIMIT)
        if not self.tracks:
            self.fired = [k for k, v in sig.items() if v]
            self.signals = dict(sig, detail=detail)
            return self.fired
        ordered = sorted(self.tracks, key=lambda t: t['chi2'])
        top = ordered[0]
        if top['falling'] and top['steps'] >= self.max_steps:
            sig['still_falling'] = True
        if len(ordered) >= 2:
            a, b = ordered[0], ordered[1]
            close = abs(b['chi2']-a['chi2']) < HEADS_CHI2_CLOSE*max(min(a['chi2'], b['chi2']), 1e-300)
            if close:
                diffs = _compare(reportable(a), reportable(b))
                worst = max(diffs, key=lambda d: d['n_sigma']) if diffs else None
                if worst is not None and worst['n_sigma'] > DISAGREE_SIGMA:
                    sig['heads_disagree'] = True
                    detail['heads_disagree'] = worst
        coef = np.asarray(top['coef'])
        types = self.setup.types
        live = {j for j in range(4) if types[j] and coef[j] > 0}
        bounds = [(c, side) for c, side in self.at_bound(top) if c < 28 and c//7 in live and c % 7 in (0, 2, 4)]
        if bounds:
            sig['size_at_bound'] = True
            detail['size_at_bound'] = [[int(c), side] for c, side in bounds]
        zero = [int(j) for j in range(4) if types[j] and not coef[j] > 0]
        if zero:
            sig['zero_amplitude'] = True
            detail['zero_amplitude_slots'] = zero
        self.fired = [k for k, v in sig.items() if v]
        self.signals = dict(sig, detail=detail)
        return self.fired

    def robust_jobs(self):
        """S3 starts: every head (best first; the S1/S2 heads again, in one uninterrupted run of max_steps
        evaluations: chunked warm starts can end slightly above a straight run) and seeded LHS starts."""
        from scipy.stats import qmc
        jobs, labels = [], []
        for cand in self.heads:
            jobs.append((self.setup, cand['u'], self.max_steps, self.align))
            labels.append(f"{cand['source']}+lm")
        lhs = qmc.LatinHypercube(32, seed=self.seed+int(self.setup.combo)).random(LHS_STARTS)
        for k, u in enumerate(lhs):
            jobs.append((self.setup, self.setup.prepare(u), self.max_steps, self.align))
            labels.append(f'lhs{k}+lm')
        return jobs, labels

    def dscan_jobs(self):
        """S4 starts: each population's D coordinate over D_SCAN from the best solution so far."""
        if not self.setup.sf_on:
            return [], []
        parent = self.best['u']
        jobs, labels = [], []
        for j, typ in enumerate(self.setup.types):
            if not typ or 7*j+4 in self.setup.fixed:
                continue
            for g in D_SCAN:
                u = parent.copy()
                u[7*j+4] = g
                jobs.append((self.setup, u, self.max_steps))
                labels.append(f'dscan{j}:{g}+lm')
        return jobs, labels

    def absorb(self, stage, labels, results, seconds):
        for label, r in zip(labels, results):
            self._add(stage, label, r)
        self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0)+seconds
        if stage not in self.stages_run:
            self.stages_run.append(stage)

    def alternatives(self, reportable):
        """Finished LM solutions that are chi2-equivalent to the best one but differ in R_nm or D_nm.

        Compared: the final state of every refined head and every S3/S4 result, with
        chi2 - chi2_best <= max(25, 0.02 N) max(1, chi2_best/N) (PLAN_zh.md §1 same-basin tolerance).
        Returns [dict(parameter 'popN.key', value, alternative, sigma, n_sigma, delta_chi2, stage, source)]
        for differences above MULTIMODAL_SIGMA; populations matched as in the heads_disagree signal, types
        with a different number of active populations are skipped."""
        best = self.best
        if best.get('cov_u') is None:
            return []
        scale = max(1.0, best['chi2']/max(self.n, 1))
        tolerance = max(MULTIMODAL_DCHI2[0], MULTIMODAL_DCHI2[1]*self.n)*scale
        finished = list(self.tracks)+[c for c in self.candidates if c['stage'] in ('S3', 'S4')]
        mine = None
        out = []
        for cand in finished:
            delta = cand['chi2']-best['chi2']
            if cand is best or cand.get('cov_u') is None or not delta <= tolerance:
                continue
            if mine is None:
                mine = reportable(best)
            for d in _compare(mine, reportable(cand), same_count=True):
                if d['n_sigma'] > MULTIMODAL_SIGMA:
                    out.append(dict(parameter=f"{d['pop']}.{d['key']}", value=d['a'], alternative=d['b'],
                                    sigma=d['sigma'], n_sigma=d['n_sigma'], delta_chi2=float(delta),
                                    stage=cand['stage'], source=cand['source']))
        return out

    def profile_stage(self, polish=0, rounds=2, budget=None, exchange=None, exchange_steps=16):
        """Stage P: conditional (R, D) profile alternatives and population exchanges of the best solution
        (research.fit1d.profile). A refined alternative that beats the best becomes the best, and the profile
        is repeated for it (at most `rounds` times). budget / exchange: profile starts / population exchanges
        refined per round (None: all). Returns the number of alternatives added."""
        from research.fit1d import profile
        if not self.align or not self.candidates:
            return 0
        t0 = time.perf_counter()
        total = 0
        for _ in range(int(rounds)):
            before = self.best
            total += profile.alternatives(self, polish=polish, budget=budget, exchange=exchange, exchange_steps=exchange_steps)
            if self.best is before:
                break
        self.stage_seconds['P'] = self.stage_seconds.get('P', 0.0)+time.perf_counter()-t0
        if 'P' not in self.stages_run:
            self.stages_run.append('P')
        return total

    def dwba_refine(self, dwba, steps=DWBA_STEPS):
        """S5: refit the best Born solution with the four-term DWBA sphere model (LM from the Born optimum).

        The Born stages (network proposals, LM, robust search) use the table engine with one qz; for spheres on
        a substrate that biases R (BornAgain round trip, PROJECT_LOG 2026-10-01). After this stage the run
        reports the DWBA solution; the Born optimum is kept in self.born for the result's born_reference."""
        t0 = time.perf_counter()
        born = self.best
        setup = replace(self.setup, dwba=dwba, key=f'{self.setup.key}-dwba')
        fit = setup.make_fit()
        if born.get('cov_u') is None:               # preview: evaluate the proposal in DWBA, no refinement
            fit.trace = []
            coef, _, residual = fit.solve(fit.matrix_from(*fit.full(born['u'])))
            r = dict(u=born['u'], coef=coef, chi2=float(residual@residual), cov_u=None, coords=[], status=None,
                     nfev=1, evaluations=1, falling=False)
        else:
            r = lm_run(fit, setup, born['u'], steps)
            total = steps
            while r['falling'] and total < DWBA_MAX_STEPS:        # the Born optimum can be far from the DWBA one
                r = _sum_runs(r, lm_run(fit, setup, r['u'], steps))
                total += steps
        self.born = dict(chi2=float(born['chi2']), u=np.array(born['u'], float), coef=np.array(born['coef'], float),
                         stage=born['stage'], source=born['source'], setup=self.setup, fit=self.fit,
                         evaluations=self.evaluations())
        self.setup, self.fit = setup, fit
        self.candidates = [dict(stage='S5', source=f"{born['source']}+dwba", **r)]
        self.tracks = []
        self.stage_seconds['S5'] = time.perf_counter()-t0
        self.stages_run.append('S5')

    @property
    def stop_stage(self):
        # 'P' (profile alternatives) is a check on the finished fit, not a search stage: not a stop stage
        stages = [s for s in self.stages_run if s != 'P']
        return stages[-1] if stages else None

    def evaluations(self):
        out = {}
        for c in self.candidates:
            out.setdefault(c['stage'], dict(lm_function_evaluations=0, model_evaluations=0, runs=0))
            out[c['stage']]['lm_function_evaluations'] += int(c['nfev'])
            out[c['stage']]['model_evaluations'] += int(c['evaluations'])
            out[c['stage']]['runs'] += 1
        total = dict(lm_function_evaluations=sum(v['lm_function_evaluations'] for v in out.values()),
                     model_evaluations=sum(v['model_evaluations'] for v in out.values()),
                     runs=sum(v['runs'] for v in out.values()))
        return dict(total=total, by_stage=out)


def _compare(pops_a, pops_b, same_count=False):
    """Reportable-parameter differences of two solutions: populations with amplitude > 0 matched by type,
    then R order. sigma = the larger of the two 1-sigma (inf = unconstrained: no disagreement can be
    claimed; None = not available). same_count: skip a type whose number of active populations differs."""
    out = []
    for typ in sorted({p['type'] for p in pops_a} | {p['type'] for p in pops_b}):
        la = sorted([p for p in pops_a if p['type'] == typ and p['amplitude'] > 0], key=lambda p: p['R_nm'])
        lb = sorted([p for p in pops_b if p['type'] == typ and p['amplitude'] > 0], key=lambda p: p['R_nm'])
        if same_count and len(la) != len(lb):
            continue
        for pa, pb in zip(la, lb):
            for key in ('R_nm', 'D_nm'):
                va, vb = pa.get(key), pb.get(key)
                if va is None or vb is None:
                    continue
                errs = [e for e in (pa.get(f'{key}_err'), pb.get(f'{key}_err')) if e is not None and not np.isnan(e)]
                if not errs:
                    continue
                sigma = max(errs)
                if np.isinf(sigma):
                    continue
                n_sigma = abs(va-vb)/sigma if sigma > 0 else (np.inf if va != vb else 0.0)
                out.append(dict(type=typ, key=key, pop=pa.get('pop'), a=float(va), b=float(vb), sigma=float(sigma),
                                n_sigma=float(n_sigma)))
    return out


def powerlaw_box(rule):
    """(fixed?, start u31, lower, upper) for the power-law exponent coordinate from a constraint rule."""
    if rule is None:
        return False, powerlaw_u(3.0), {}, {}
    if rule[0] == 'fixed':
        return True, powerlaw_u(rule[1]), {}, {}
    lo, hi = powerlaw_u(rule[1]), powerlaw_u(rule[2])
    return False, float(np.clip(powerlaw_u(3.0), lo, hi)), {31: lo}, {31: hi}
