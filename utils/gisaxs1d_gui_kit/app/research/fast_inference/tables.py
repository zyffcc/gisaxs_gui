"""Precomputed size-averaged form factors (sphere, disc = standing-cylinder cross section, sinc^2 height factor).

For a Gaussian size distribution with relative width s (4-sigma, truncated at 0,
renormalised) the averaged normalised form factor depends only on x = q*R and s:
    F(x, s) = sum_i w_i(s) t_i^p f(x t_i) / sum_i w_i(s) t_i^p,
with Gauss-Legendre nodes t_i over [max(1-4s, 0), 1+4s] (in units of R), f the
single-particle form factor (sphere: [3(sin x - x cos x)/x^3]^2, disc/standing
cylinder cross section: [2 J1(x)/x]^2) and p the weighting power:
p = 0 reproduces the legacy 'intensity' convention of multishape_v1, p = 6
(sphere) / 4 (cylinder cross section at fixed height) the number-weighted
distribution (SasView/BornAgain convention). One table lookup therefore replaces
the size quadrature for a whole curve.

Tables: x on a uniform fine grid up to X_FINE, then log-spaced to X_MAX (the
polydisperse average is smooth there); s on a uniform grid. Bilinear
interpolation in (x, s) of log F.

Files: ff_<type>_<weighting>_v2.npy in the directory CACHE (default
~/.cache/gisaxs1d_tables, or the environment variable GISAXS1D_TABLES when it is
set at import; CACHE is read at every table() call, so it can also be reassigned).
table() never builds: a missing table is a FileNotFoundError that names the file,
the directory and the build command. Build once, explicitly (works on Windows:
'spawn' pool, one BLAS thread per worker):

    python3 -m research.fast_inference.tables --build [--weighting number|intensity|both]
                                              [--workers N] [--types 1 2 3] [--dir DIR] [--force]

The size average is integrated with a composite Gauss-Legendre rule whose panel
count grows with x (>= 1.5 panels of 16 nodes per oscillation period), so large
x (big particles at high q) is not aliased; the frozen 64-node rule of
multishape_v1 aliases for x*width >~ 150 and must not be used there.
Type 2 here means the random-cylinder height factor sinc^2(x/2), x = q h
(number weighting power 2); the random-cylinder radial factor is the disc table.
"""
from __future__ import annotations

import os
import sys
import time
import types
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import j1

from research.multishape_v1 import physics

X_FINE, DX_FINE = 200.0, 0.004
X_MAX, N_LOG = 8000.0, 2500
S_MIN, S_MAX, DS = 0.02, 0.60, 0.005
ORDER = 64
ENV_VAR = 'GISAXS1D_TABLES'
DEFAULT_CACHE = Path.home()/'.cache'/'gisaxs1d_tables'
CACHE = Path(os.environ[ENV_VAR]).expanduser() if os.environ.get(ENV_VAR) else DEFAULT_CACHE
WEIGHTINGS = ('number', 'intensity')
BUILD_COMMAND = 'python3 -m research.fast_inference.tables --build --weighting {weighting}'


def _sphere(x):
    x = np.asarray(x, float)
    safe = np.where(np.abs(x) < .1, 1.0, x)
    amp = 3*(np.sin(safe)-safe*np.cos(safe))/safe**3
    return np.where(np.abs(x) < .1, 1-x*x/10+x**4/280-x**6/15120, amp)**2


def _disc(x):
    x = np.asarray(x, float)
    safe = np.where(np.abs(x) < 1e-4, 1.0, x)
    return np.where(np.abs(x) < 1e-4, 1-x*x/8+x**4/192, 2*j1(safe)/safe)**2


def _sinc_half(x):
    return np.sinc(np.asarray(x, float)/(2*np.pi))**2    # [sin(x/2)/(x/2)]^2


SHAPES = {1: (_sphere, 6), 3: (_disc, 4), 2: (_sinc_half, 2)}
PANEL_NODES = 16


def x_grid():
    fine = np.arange(0.0, X_FINE+DX_FINE/2, DX_FINE)
    coarse = np.geomspace(X_FINE, X_MAX, N_LOG)[1:]
    return np.r_[fine, coarse]


def s_grid():
    return np.round(np.arange(S_MIN, S_MAX+DS/2, DS), 6)


def _gaussian_rule(s, n_panels):
    """Composite Gauss-Legendre nodes/weights of the 4-sigma truncated Gaussian in t = r/R."""
    lo, hi = max(1-4*s, 0.0), 1+4*s
    x, w = np.polynomial.legendre.leggauss(PANEL_NODES)
    edges = np.linspace(lo, hi, n_panels+1)
    half = (edges[1:]-edges[:-1])/2
    t = (edges[:-1, None]+half[:, None]*(x[None, :]+1)).ravel()
    weights = (half[:, None]*w[None, :]).ravel()*np.exp(-0.5*((t-1)/s)**2)
    return t, weights


def _nodes(s):
    t, w = physics._gaussian(1.0, s, ORDER)   # the frozen 64-node rule (reference for small x only)
    return t, w


def average(f, s, p, x):
    """Accurate size average for an array x (panels scale with the largest x)."""
    width = 8*s
    n_panels = int(np.clip(np.ceil(1.5*np.max(x)*width/(2*np.pi)), 2, 6000))
    t, w = _gaussian_rule(s, n_panels)
    w = w*t**p
    w = w/w.sum()
    out = np.empty(len(x))
    step = max(1, int(2e7//len(t)))
    for start in range(0, len(x), step):
        out[start:start+step] = w@f(np.outer(t, x[start:start+step]))
    return out


def _build_row(args):
    typ, p, s = args
    f = SHAPES[typ][0]
    xs = x_grid()
    row = np.empty(len(xs))
    edges = np.r_[0, np.geomspace(1.0, X_MAX, 60)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (xs >= lo) & ((xs < hi) if hi < X_MAX else (xs <= hi))
        if sel.any():
            row[sel] = average(f, s, p, xs[sel])
    return row


def _one_blas_thread():
    """Pool initializer: one BLAS thread per worker (oversubscribed BLAS threads make pooled work far slower)."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


@contextmanager
def _main_not_reimported():
    """While the pool starts its workers: keep 'spawn' from re-importing the caller's __main__ in them
    (the worker function lives in this importable module; a main module that cannot be imported again,
    e.g. python - <<EOF or a script without a main guard, would kill every worker). multiprocessing reads
    sys.modules['__main__'] when it starts a process; it sees an empty stand-in here. Not used when this
    module is the program itself (python -m research.fast_inference.tables: the workers need it as __main__)."""
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


def build_rows(typ, weighting, widths=None, workers=None):
    """Rows log F(x_grid(), s) (float32) for the relative widths `widths` (default: all of s_grid()).

    workers > 1: multiprocessing 'spawn' pool (macOS, Linux, Windows), module-level worker _build_row,
    one BLAS thread per worker; the workers do not import the caller's __main__."""
    if weighting not in WEIGHTINGS:
        raise ValueError(f'weighting must be one of {WEIGHTINGS}, not {weighting!r}')
    p = 0 if weighting == 'intensity' else SHAPES[typ][1]
    tasks = [(typ, p, float(s)) for s in (s_grid() if widths is None else widths)]
    workers = min(int(workers or os.cpu_count() or 1), len(tasks))
    if workers > 1:
        from multiprocessing import get_context
        with _main_not_reimported():
            pool = get_context('spawn').Pool(workers, initializer=_one_blas_thread)
        with pool:
            rows = pool.map(_build_row, tasks, chunksize=1)
    else:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            rows = [_build_row(task) for task in tasks]
        else:
            with threadpool_limits(1):      # same one-thread rule in this process, restored afterwards
                rows = [_build_row(task) for task in tasks]
    return np.log(np.maximum(np.array(rows), 1e-300)).astype(np.float32)


def build(typ, weighting, workers=None):
    """The full table of one shape and weighting (minutes; explicit use only, see the module docstring)."""
    return build_rows(typ, weighting, workers=workers)


def table_path(typ, weighting='intensity', directory=None):
    return Path(CACHE if directory is None else directory)/f'ff_{typ}_{weighting}_v2.npy'


def _how_to_get(weighting):
    return (f'Tables are never built on the fly. Copy the ff_*_{weighting}_v2.npy files into that directory, '
            f'point the environment variable {ENV_VAR} to a directory that holds them, or build them once '
            f'(explicit, takes minutes) from the portable_project directory with: '
            f'{BUILD_COMMAND.format(weighting=weighting)}')


@lru_cache(maxsize=16)
def _load(directory, typ, weighting):
    path = table_path(typ, weighting, directory)
    if not path.is_file():
        raise FileNotFoundError(f'form-factor table {path.name} not found in {directory} (directory: {ENV_VAR} if '
                                f'set, else {DEFAULT_CACHE}). {_how_to_get(weighting)}')
    values = np.load(path)
    expected = (len(s_grid()), len(x_grid()))
    if values.shape != expected:
        raise ValueError(f'form-factor table {path.name} in {directory} has shape {values.shape}, the code expects '
                         f'{expected} (stale grid). Rebuild it (the stale file is replaced) with: '
                         f'{BUILD_COMMAND.format(weighting=weighting)}')
    return values


def table(typ, weighting='intensity'):
    """log F(x_grid(), s_grid()) of one shape, loaded from CACHE (read at call time) and kept in memory.

    Never builds: FileNotFoundError names the missing file, the directory, GISAXS1D_TABLES and the build
    command; ValueError for a file whose shape does not match the grids of this code."""
    return _load(str(CACHE), typ, weighting)


table.cache_clear = _load.cache_clear
table.cache_info = _load.cache_info


def build_tables(weightings=('number',), types=(1, 2, 3), directory=None, workers=None, force=False, log=print):
    """Build and save the missing (or, with force, all) tables; returns {file name: seconds or 'kept'}."""
    directory = Path(CACHE if directory is None else directory)
    directory.mkdir(parents=True, exist_ok=True)
    expected = (len(s_grid()), len(x_grid()))
    done = {}
    for weighting in weightings:
        for typ in types:
            path = table_path(typ, weighting, directory)
            if path.is_file() and not force:
                try:
                    current = np.load(path, mmap_mode='r').shape == expected
                except (OSError, ValueError):
                    current = False
                if current:
                    done[path.name] = 'kept'
                    log(f'{path}: present with the current grid, kept (use --force to rebuild)')
                    continue
            started = time.perf_counter()
            log(f'{path.name}: building {expected[0]} rows x {expected[1]} points ...')
            values = build_rows(typ, weighting, workers=workers)
            partial = path.with_name(path.name+'.partial')
            with open(partial, 'wb') as handle:
                np.save(handle, values)
            os.replace(partial, path)
            done[path.name] = time.perf_counter()-started
            log(f'{path}: written in {done[path.name]:.0f} s')
    table.cache_clear()
    return done


class FormTable:
    """Vectorised bilinear lookup of log F(x, s)."""

    def __init__(self, typ, weighting='intensity'):
        self.logf = table(typ, weighting)
        self.xs = x_grid()
        self.ss = s_grid()
        self.n_fine = int(round(X_FINE/DX_FINE))+1
        self.log_lo, self.log_hi = np.log(X_FINE), np.log(X_MAX)

    def _x_index(self, x):
        x = np.clip(np.abs(x), 0.0, X_MAX*(1-1e-12))
        fine = x < X_FINE
        pos = np.where(fine, x/DX_FINE,
                       self.n_fine-1+(np.log(np.maximum(x, X_FINE))-self.log_lo)/(self.log_hi-self.log_lo)*(N_LOG-1))
        i = np.clip(np.floor(pos).astype(int), 0, len(self.xs)-2)
        return i, pos-i

    def __call__(self, x, s):
        """F(x, s) for an array x and a scalar relative width s.

        Beyond X_MAX the polydisperse average is a smooth power law: continue log F
        linearly in log x with the slope of the last table interval."""
        s = float(np.clip(s, S_MIN, S_MAX))
        ks = (s-S_MIN)/DS
        k = int(min(np.floor(ks), len(self.ss)-2))
        a = ks-k
        x = np.abs(np.asarray(x, float))
        i, b = self._x_index(x)
        row0 = self.logf[k]
        row1 = self.logf[k+1]
        v0 = row0[i]*(1-b)+row0[i+1]*b
        v1 = row1[i]*(1-b)+row1[i+1]*b
        value = v0*(1-a)+v1*a
        beyond = x > X_MAX
        if np.any(beyond):
            last = row0[-1]*(1-a)+row1[-1]*a
            prev = row0[-2]*(1-a)+row1[-2]*a
            slope = (last-prev)/(np.log(self.xs[-1])-np.log(self.xs[-2]))
            value = np.where(beyond, last+slope*(np.log(np.maximum(x, X_MAX))-np.log(self.xs[-1])), value)
        return np.exp(value)


def check(n=400, seed=0):
    """Worst/median lnRMSE of table curves vs an independent accurate average (more panels)."""
    rng = np.random.default_rng(seed)
    worst = {}
    for typ in (1, 3, 2):
        for weighting in ('intensity', 'number'):
            ft = FormTable(typ, weighting)
            f, power = SHAPES[typ]
            p = 0 if weighting == 'intensity' else power
            errs = []
            for _ in range(n):
                q = np.sort(np.exp(rng.uniform(np.log(1e-3), np.log(4.5), 400)))
                r = np.exp(rng.uniform(np.log(0.5), np.log(200 if typ != 2 else 500)))
                s = rng.uniform(S_MIN, 0.5)
                x = q*r
                t, w = _gaussian_rule(s, int(np.clip(np.ceil(12*x.max()*8*s/(2*np.pi)), 4, 8000)))
                w = w*t**p
                w = w/w.sum()
                ref = w@f(np.outer(t, x))
                ok = ref > 1e-10*ref.max()
                errs.append(np.sqrt(np.mean((np.log(ft(x, s)[ok])-np.log(ref[ok]))**2)))
            worst[f'{typ}_{weighting}'] = (round(float(np.max(errs)), 5), float(np.median(errs)))
    return worst


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Build or list the size-averaged form-factor tables (physics v2).')
    parser.add_argument('--build', action='store_true', help='build the missing tables (explicit; takes minutes)')
    parser.add_argument('--weighting', default='number', choices=WEIGHTINGS+('both',))
    parser.add_argument('--workers', type=int, default=None, help='pool processes (default: all cores)')
    parser.add_argument('--types', type=int, nargs='+', default=[1, 2, 3], choices=(1, 2, 3),
                        help='1 sphere, 2 sinc^2 height factor, 3 disc (default: all)')
    parser.add_argument('--dir', type=Path, default=None, help=f'target directory (default: {ENV_VAR} or {DEFAULT_CACHE})')
    parser.add_argument('--force', action='store_true', help='rebuild tables that are already present')
    args = parser.parse_args(argv)
    directory = Path(CACHE if args.dir is None else args.dir)
    weightings = WEIGHTINGS if args.weighting == 'both' else (args.weighting,)
    if args.build:
        build_tables(weightings, tuple(args.types), directory, args.workers, args.force)
    expected = (len(s_grid()), len(x_grid()))
    missing = []
    for weighting in weightings:
        for typ in args.types:
            path = table_path(typ, weighting, directory)
            if not path.is_file():
                state = 'MISSING'
            else:
                state = 'ok' if np.load(path, mmap_mode='r').shape == expected else 'STALE (grid changed)'
            if state != 'ok':
                missing.append(path.name)
            print(f'{state:8s} {path}')
    if missing and not args.build:
        print(f'{len(missing)} table(s) not usable; build with: '
              f'{BUILD_COMMAND.format(weighting=args.weighting)}')
    return 1 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())
