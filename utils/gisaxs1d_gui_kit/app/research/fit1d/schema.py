"""Result schema 'gisaxs1d_result_v1' of fit1d: conventions, provenance, populations, CSV and plot.

Top-level keys (all JSON-serialisable; NaN/inf are written as null):
  schema, fit1d_version, conventions, population_order, input, settings, provenance, candidate_set,
  results, data, notes, timing
Per composition (results[i], ranked by BIC, i = 0 is the lowest BIC):
  rank, composition, composition_text, combo, chi2, chi2_per_point, n_points, dof, k_params, bic,
  delta_bic, in_candidate_set (delta_bic <= threshold), indistinguishable (in the candidate set together
  with at least one other composition), populations, background_B, background_fixed, low_q, flags, stop_stage,
  stages_run, signals, stage_seconds, evaluations, winner, u, curves (model and components on data.q)
Per population: pop, slot, type, R_nm, R_nm_err, R_number_mean_nm, sigma_R_rel, sigma_R_rel_err,
  D_nm, D_nm_err, w_rel, w_rel_err, w_nm, w_nm_err, h_nm, h_nm_err, sigma_h_rel, sigma_h_rel_err,
  amplitude, inactive, intensity_fraction, q_star_nm_inv, q_star_nm_inv_err, Rg_nm, Rg_nm_err, fixed,
  constraints
  inactive = the NNLS amplitude is 0: the population does not contribute to the model, its parameters are
  where the optimiser left them, carry no information, have no errors (null) and are all listed in
  flags.weakly_constrained.
Errors (*_err) are null when not available: fixed parameter, preview (no covariance), inactive population,
or a direction the data do not constrain at all (listed in flags.weakly_constrained with reason no_error).
low_q: fitted values (sigma_res_nm_inv, a free nu, a free powerlaw_n) are null when C = 0 (inactive: true);
the last optimiser values are kept under 'unconstrained_values'.
Time-dependent values live only in 'timing' and in each result's 'stage_seconds'.
"""
from __future__ import annotations

import csv
import hashlib
import math
import os
import platform
from pathlib import Path

import numpy as np

SCHEMA = 'gisaxs1d_result_v1'
VERSION = 'fit1d 1.4.0 (M2: (R, D) alignment + pursuit, stage P: profile alternatives + population exchanges, ensemble ambiguity, DWBA stage S5; 2026-10-06)'
WORK_DIR = Path(__file__).resolve().parents[3]
TYPE_NAMES = {1: 'sphere', 2: 'random_cylinder', 3: 'vertical_cylinder'}
SHORT = {1: 'sphere', 2: 'rc', 3: 'vc'}

CONVENTIONS = dict(
    physics='v2 (research.fast_inference.tabfit.TabFit, weighting=number, sf=hosemann)',
    units=dict(q='nm^-1', lengths='nm', relative_widths='dimensionless (fraction of the mean)'),
    size_distribution=('number-weighted Gaussian of relative width sigma (sigma_R_rel, sigma_h_rel), '
                       'truncated to [max(R(1-4 sigma), 0), R(1+4 sigma)] and renormalised; R_nm is the '
                       'distribution parameter (equal to the number mean for sigma <= 0.25), R_number_mean_nm '
                       'the mean of the truncated distribution'),
    structure_factor=('Hosemann 1D paracrystal S(q) = (1-phi^2)/(1-2 phi cos(qD)+phi^2), '
                      'phi = exp(-q^2 w^2/2), w = w_rel * D (nearest-neighbour distance D, spread w); '
                      'S = 1 when the structure factor is off (SAXS default)'),
    shapes=dict(sphere=('Born (sphere_model=born): |q| = sqrt(q_par^2 + qz^2) with the stated qz (GISAXS) or q '
                        '(SAXS). DWBA (sphere_model=dwba4, stage S5): four-term DWBA amplitude of a sphere sitting on '
                        'the substrate, A = sum_j c_j exp(-i qz_j R) f(sqrt(q_par^2+qz_j^2) R) with (c_j, qz_j) = '
                        '(1, kf+ki), (r_i, kf-ki), (r_f, ki-kf), (r_i r_f, -(kf+ki)), r = Fresnel reflection '
                        'coefficients of the substrate at alpha_i and alpha_f (settings.dwba), local monodisperse '
                        'approximation; validated against BornAgain (results/physics_check)'),
                random_cylinder=('isotropic orientation average, |q| = sqrt(q_par^2 + qz^2); radius R, '
                                 'height h, independent number-weighted distributions'),
                vertical_cylinder=('cross section only, q_par (the qz height factor is constant along a '
                                   'Yoneda cut and absorbed by the amplitude)')),
    model='I(q) = sum_j A_j F_j(q) S_j(q) + B + C g(q); A_j, B, C >= 0 by NNLS (variable projection)',
    low_q=dict(lorentz='g(q) = 1/(1+(q/sigma_res)^nu), nu fixed at lowq_nu (default 2) unless freed',
               powerlaw='g(q) = (max(q, q_min)/q_min)^-n, n in [1, 4]',
               none='no low-q term (SAXS default)'),
    derived=dict(q_star='2 pi / D',
                 Rg=('scattering-weighted radius of gyration of the number distribution, <.> over the '
                     'truncated Gaussian: sphere sqrt(3/5 <R^8>/<R^6>); vertical cylinder cross-section '
                     '(in-plane Guinier exp(-q^2 Rc^2/2)) sqrt(<R^6>/(2 <R^4>)); random cylinder '
                     'sqrt(<R^6>/(2 <R^4>) + <h^4>/(12 <h^2>))'),
                 intensity_fraction='sum over the data q of A_j F_j S_j divided by the sum of the total model',
                 w_nm='w_rel * D'),
    uncertainties=('1-sigma from the Gauss-Newton covariance of the free nonlinear coordinates (amplitudes '
                   'profiled out), scaled by sqrt(max(1, chi2/dof)); null when not available (preview, '
                   'fixed, inactive population with amplitude 0, or a direction the data do not constrain); '
                   'parameters at a bound are not reliable. The errors describe the local minimum only: '
                   'flags.alternatives lists chi2-equivalent solutions with different R_nm/D_nm'),
    bic='chi2 + k ln(N), k = free nonlinear coordinates + positive linear coefficients (B excluded if fixed)',
    composition=('the curve alone does not determine the composition (K >= 2); compositions within '
                 'delta BIC <= threshold of the lowest are marked indistinguishable'),
)
POPULATION_ORDER = ('populations pop1, pop2, ... follow the canonical composition order: by type code '
                    '(1 sphere, 2 random cylinder, 3 vertical cylinder), then by slot; populations of the '
                    'same type are interchangeable (their order carries no physical meaning)')

CSV_COLUMNS = ('rank', 'composition', 'combo', 'indistinguishable', 'in_candidate_set', 'delta_bic', 'bic', 'chi2', 'chi2_per_point',
               'n_points', 'dof', 'stop_stage', 'flags', 'population', 'slot', 'type', 'R_nm', 'R_nm_err',
               'R_number_mean_nm', 'sigma_R_rel', 'sigma_R_rel_err', 'D_nm', 'D_nm_err', 'w_rel', 'w_rel_err',
               'w_nm', 'h_nm', 'h_nm_err', 'sigma_h_rel', 'amplitude', 'intensity_fraction', 'q_star_nm_inv',
               'Rg_nm', 'fixed', 'background_B', 'low_q_form', 'low_q_C', 'low_q_sigma_res_nm_inv', 'low_q_nu',
               'low_q_powerlaw_n', 'mode', 'qz_nm_inv')


# ---------------------------------------------------------------- JSON helpers
def jsonable(obj):
    """Plain Python types only; non-finite floats -> None."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == 'f':
            values = obj.astype(float)
            if np.all(np.isfinite(values)):
                return values.tolist()
            return [v if math.isfinite(v) else None for v in values.tolist()] if values.ndim == 1 else \
                [jsonable(v) for v in values]
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ---------------------------------------------------------------- provenance
_SHA: dict = {}
_GIT: dict = {}
_TABLES_OK: dict = {}


def sha256_file(path, chunk=1 << 20):
    """SHA-256 of a file, computed once per process (keyed by path, size and mtime)."""
    path = Path(path)
    if not path.exists():
        return None
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _SHA:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(chunk), b''):
                h.update(block)
        _SHA[key] = h.hexdigest()
    return _SHA[key]


def git_head(repo=WORK_DIR):
    """HEAD commit of the work/ repository (read from .git without a subprocess; cached), or None."""
    repo = Path(repo)
    if str(repo) in _GIT:
        return _GIT[str(repo)]
    value = None
    try:
        git = repo/'.git'
        head = (git/'HEAD').read_text(encoding='utf-8').strip()
        if head.startswith('ref:'):
            ref = head.split(None, 1)[1]
            path = git/ref
            if path.exists():
                value = path.read_text(encoding='utf-8').strip()
            elif (git/'packed-refs').exists():
                for line in (git/'packed-refs').read_text(encoding='utf-8').splitlines():
                    if line.endswith(' '+ref):
                        value = line.split()[0]
        else:
            value = head
    except OSError:
        value = None
    _GIT[str(repo)] = value
    return value


def tables_dir():
    from research.fast_inference import tables
    return Path(tables.CACHE)


TABLE_FILES = tuple(f'ff_{t}_number_v2.npy' for t in (1, 2, 3))
BUILD_COMMAND = 'python3 -m research.fast_inference.tables --build --weighting number'


def check_tables(directory=None):
    """Select the table directory and raise FileNotFoundError (with instructions) unless the three number
    tables are present there. Returns the directory.

    directory: explicit path (spawn workers get the parent's directory this way); None -> the environment
    variable GISAXS1D_TABLES if set, else the default ~/.cache/gisaxs1d_tables. Tables are never built on
    the fly (building takes minutes on many cores)."""
    from research.fast_inference import tables
    default = Path(getattr(tables, 'DEFAULT_CACHE', None) or _TABLES_OK.setdefault('default', Path(tables.CACHE)))
    if directory is None:
        env = os.environ.get('GISAXS1D_TABLES')
        directory = Path(env).expanduser() if env else default
    directory = Path(directory)
    if Path(tables.CACHE) != directory:
        tables.CACHE = directory
        tables.table.cache_clear()
    stamp = []
    for name in TABLE_FILES:
        try:
            st = (directory/name).stat()
            stamp.append((st.st_size, st.st_mtime_ns))
        except OSError:
            stamp.append(None)
    key = (str(directory), tuple(stamp))
    if _TABLES_OK.get('key') == key:
        return directory
    shape = (len(tables.s_grid()), len(tables.x_grid()))
    missing = []
    for name in TABLE_FILES:
        path = directory/name
        try:
            ok = path.exists() and np.load(path, mmap_mode='r').shape == shape
        except (OSError, ValueError):
            ok = False
        if not ok:
            missing.append(name)
    if missing:
        command = getattr(tables, 'BUILD_COMMAND', BUILD_COMMAND)
        command = command.format(weighting='number') if '{weighting}' in command else command
        raise FileNotFoundError(
            f'fit1d: physics tables missing or invalid in {directory}: {", ".join(missing)}. Copy the three '
            f'number-weighted tables ({", ".join(TABLE_FILES)}, 73.7 MB) into that directory, or point the '
            f'environment variable GISAXS1D_TABLES to a directory that has them, or build them once from the '
            f'portable_project directory with "{command}" (explicit, takes minutes). '
            'fit1d never builds tables itself.')
    _TABLES_OK['key'] = key
    return directory


_SOURCES: dict = {}


def sources_sha256():
    """SHA-256 over the fit1d source files (*.py, sorted by name; each as name + NUL + bytes), once per
    process: pins down the code that ran even when the package is not committed."""
    if 'value' not in _SOURCES:
        h = hashlib.sha256()
        for path in sorted(Path(__file__).resolve().parent.glob('*.py')):
            h.update(path.name.encode('utf-8')+b'\0'+path.read_bytes())
        _SOURCES['value'] = h.hexdigest()
    return _SOURCES['value']


def provenance(model_dir):
    import scipy
    directory = tables_dir()
    package = Path(__file__).resolve().parent
    return dict(fit1d_version=VERSION, git_head=git_head(), git_repository=str(WORK_DIR),
                git_note=('git_head is the HEAD commit of the repository, not a statement that these sources are '
                          'committed: fit1d_sources_sha256 and ood_stats_sha256 pin down the code and thresholds '
                          'that ran'),
                fit1d_sources_sha256=sources_sha256(), ood_stats_sha256=sha256_file(package/'ood_stats.json'),
                model_dir=str(model_dir), weights_sha256=sha256_file(Path(model_dir)/'weights_numpy.npz'),
                tables_dir=str(directory), tables_sha256={n: sha256_file(directory/n) for n in TABLE_FILES},
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)


# ---------------------------------------------------------------- populations
_GL = np.polynomial.legendre.leggauss(96)


def truncated_moments(s, powers):
    """E[t^k] of the number distribution in t = r/R: Gaussian(1, s) truncated to [max(1-4s, 0), 1+4s]."""
    lo, hi = max(1-4*s, 0.0), 1+4*s
    x, w = _GL
    t = lo+(hi-lo)*(x+1)/2
    dens = w*np.exp(-0.5*((t-1)/max(s, 1e-12))**2)
    total = dens.sum()
    return [float(dens@t**k/total) for k in powers]


def number_mean(r, s):
    return r*truncated_moments(s, [1])[0]


def radius_of_gyration(typ, r, s_r, h=None, s_h=None):
    m4, m6, m8 = truncated_moments(s_r, [4, 6, 8])
    if typ == 1:
        return r*math.sqrt(0.6*m8/m6)
    if typ == 3:
        return r*math.sqrt(m6/(2*m4))
    h2, h4 = truncated_moments(s_h, [2, 4])
    return math.sqrt(r*r*m6/(2*m4)+h*h*h4/(12*h2))


def population_values(p, j, typ, sf_on):
    """Physical quantities of slot j as a flat dict (order fixed by the keys below)."""
    r, s = float(p['R'][j]), float(p['sigma_R'][j])
    out = dict(R_nm=r, sigma_R_rel=s, R_number_mean_nm=number_mean(r, s))
    if sf_on:
        d, w = float(p['D'][j]), float(p['sigma_D'][j])
        out.update(D_nm=d, w_rel=w, w_nm=w*d, q_star_nm_inv=2*math.pi/d)
    if typ == 2:
        out.update(h_nm=float(p['h'][j]), sigma_h_rel=float(p['sigma_h'][j]))
        out['Rg_nm'] = radius_of_gyration(typ, r, s, out['h_nm'], out['sigma_h_rel'])
    else:
        out['Rg_nm'] = radius_of_gyration(typ, r, s)
    return out


def populations(codec, types, u, coef, coords, cov, sf_on, components_sum=None, total_sum=None):
    """Population dicts with 1-sigma errors (controller.propagate) for one solution."""
    from research.fit1d.constraints import population_labels, FIELD_TO_KEY
    from research.fit1d.controller import propagate
    u = np.asarray(u, float)
    p = codec.decode(u)
    out = []
    for label, j, typ in population_labels(types):
        keys = list(population_values(p, j, typ, sf_on))
        func = lambda uu, j=j, typ=typ: np.array(list(population_values(codec.decode(uu), j, typ, sf_on).values()))
        values, errors = propagate(func, u, coords, cov)
        pop = dict(pop=label, slot=j, type=TYPE_NAMES[typ])
        fixed = [FIELD_TO_KEY[f] for f in ('R', 'sigma_R', 'h', 'sigma_h', 'D', 'sigma_D')
                 if hasattr(codec, 'is_fixed') and codec.is_fixed(j, f)]
        exact = population_values(p, j, typ, sf_on)
        inactive = not coef[j] > 0
        for key, err in zip(keys, errors):
            pop[key] = exact[key]
            pop[f'{key}_err'] = None if inactive or key in fixed or not np.isfinite(err) or not err > 0 else float(err)
        for key in ('R_nm', 'sigma_R_rel', 'D_nm', 'w_rel', 'h_nm', 'sigma_h_rel'):
            pop.setdefault(key, None)
            pop.setdefault(f'{key}_err', None)
        for key in ('w_nm', 'q_star_nm_inv'):
            pop.setdefault(key, None)
        pop['amplitude'] = float(coef[j])
        pop['inactive'] = bool(inactive)
        pop['intensity_fraction'] = (float(components_sum[j]/total_sum)
                                     if components_sum is not None and total_sum and total_sum > 0 else None)
        pop['fixed'] = fixed
        pop['constraints'] = codec.describe(j) if hasattr(codec, 'describe') else {}
        out.append(pop)
    return out


# ---------------------------------------------------------------- CSV and plot
def _fmt(v):
    if v is None:
        return ''
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return ';'.join(str(x) for x in v)
    return str(v)


def csv_rows(result):
    rows = []
    settings = result.get('settings', {})
    for res in result['results']:
        low = res.get('low_q') or {}
        base = dict(rank=res['rank'], composition=res['composition_text'], combo=res['combo'],
                    indistinguishable=res['indistinguishable'], in_candidate_set=res['in_candidate_set'], delta_bic=res['delta_bic'], bic=res['bic'],
                    chi2=res['chi2'], chi2_per_point=res['chi2_per_point'], n_points=res['n_points'], dof=res['dof'],
                    stop_stage=res['stop_stage'], flags=res['flags'].get('raised', []),
                    background_B=res['background_B'], low_q_form=low.get('form'), low_q_C=low.get('C'),
                    low_q_sigma_res_nm_inv=low.get('sigma_res_nm_inv'), low_q_nu=low.get('nu'),
                    low_q_powerlaw_n=low.get('powerlaw_n'), mode=settings.get('mode'),
                    qz_nm_inv=settings.get('qz_nm_inv'))
        for pop in res['populations']:
            row = dict(base, population=pop['pop'], slot=pop['slot'], type=pop['type'])
            for key in CSV_COLUMNS:
                if key in pop and key not in ('type',):
                    row[key] = pop[key]
            rows.append([_fmt(row.get(key)) for key in CSV_COLUMNS])
    return rows


def to_csv(result, path):
    """One row per population of every reported composition; columns in CSV_COLUMNS order."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(csv_rows(result))
    return str(path)


def plot(result, path, top=1):
    """Data, total model, per-population components, background/low-q, and residual/sigma panel."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = result['data']
    q, y, s = (np.asarray(data[k], float) for k in ('q', 'intensity', 'sigma'))
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True, gridspec_kw=dict(height_ratios=(3, 1)))
    pos = y > 0
    ax.loglog(q[pos], y[pos], '.', color='#52514e', ms=2.5, label='data', zorder=1)
    colours = ('#2a78d6', '#eb6834', '#1baf7a', '#8b5cf6')
    comp_colours = ('#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22')
    for i, res in enumerate(result['results'][:max(top, 1)]):
        model = np.asarray(res['curves']['model'], float)
        tag = '' if not res['indistinguishable'] or len(result['results']) == 1 else ' (indistinguishable)'
        ax.loglog(q, np.where(model > 0, model, np.nan), lw=1.4, color=colours[i % 4],
                  label=f"{res['composition_text']}  chi2/pt={res['chi2_per_point']:.2f}{tag}", zorder=3)
        bx.semilogx(q, (y-model)/s, '.', ms=1.5, color=colours[i % 4])
        if i == 0:
            for k, (name, curve) in enumerate(res['curves']['components'].items()):
                curve = np.asarray(curve, float)
                if np.any(curve > 0):
                    ax.loglog(q, np.where(curve > 0, curve, np.nan), '--', lw=0.9,
                              color=comp_colours[k % len(comp_colours)], label=name, zorder=2)
    ax.set_ylabel('intensity')
    lo = np.nanmin(y[pos]) if pos.any() else 1.0
    ax.set_ylim(bottom=max(lo/3, 1e-300))
    ax.legend(fontsize=7, loc='lower left')
    best = result['results'][0]
    settings = result.get('settings', {})
    ax.set_title(f"fit1d {settings.get('mode', '')}, policy {settings.get('policy', '')}, "
                 f"stop {best['stop_stage']}, flags: {', '.join(best['flags'].get('raised', [])) or 'none'}",
                 fontsize=8)
    bx.axhline(0, color='#999999', lw=0.8)
    bx.set_xlabel('q (nm$^{-1}$)')
    bx.set_ylabel('(I - model)/sigma')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
