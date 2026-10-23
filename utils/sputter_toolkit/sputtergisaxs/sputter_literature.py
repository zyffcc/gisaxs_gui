"""Literature method of a sputter series and its comparison with published growth (summary key 'literature').

The P03 in-situ GISAXS literature (Schwartzkopf 2015, Bulut 2024, Gensch 2019/2021) does not fit R: it takes
the nominal thickness delta (a QCM rate times the time) and the spacing D = 2*pi/q* of the direct peak and
computes R by mass balance for flat-based hemispheres (h=R) on a hexagonal lattice:

    delta = (2/3)*pi*R^3 / ((sqrt(3)/2)*D^2) = 2.418*R^3/D^2,  coverage = (2*pi/sqrt(3))*(R/D)^2,

and calls the first delta where 2R = D the percolation threshold. Those studies go on reporting D, R and 2R/D
after it; this tool keeps D = 2*pi/q* after the percolation but gives no R there (molecular dynamics, Abr16,
finds that R loses its island meaning once a network forms). This module works per window of
structure_physical.csv when a deposition rate is given:

- D_lit_nm = 2*pi/q* (D_corr_nm) only where a resolved peak exists. Peaks that probably belong to the host
  are marked (lit_status host_peak) and not used: a peak before the deposition starts, a peak more than
  host_peak_factor above every literature D(delta) at its delta. A peak more than a factor 2 above the median of
  the preceding peaks, beyond any growth since t0 (D cannot grow faster than delta), is a jump: host_peak when it
  sits at the host spacing, else outlier. A peak more than a factor 2 from the trend (outlier) or after a long
  stretch without a peak (late_peak) is not used either; the late peaks are compared with the end of the trend.
- D_lit_trend_nm: non-decreasing L1 fit (pool-adjacent-violators with block medians) of ln D_lit in time.
- R_lit_nm = (delta*D_lit^2/2.418)^(1/3), two_R_over_D_lit, coverage_lit, and their trend values.
- percolation: the first delta after nucleation where the trend of 2R/D reaches 1. Its 68 % interval holds the scatter of the
  windows (a moving-block bootstrap around their local median, added to the trend and fitted again) and the
  error of the rate and t0 (rate_error, t0_error_s): the lower end with a higher rate and an earlier t0, the
  upper end with a lower rate and a later t0. When 2R/D may reach 1 within these errors but need not, the
  percolation is 'not determined'. After a percolation R_lit is not given (lit_status after_percolation).
- the network's own 2R/D and coverage (from R_nm/D_nm of the physical path) and eta_net = t_eq_nm/delta.
- checks against configs/literature_reference_v1.json, each {id, title, verdict, measured, expected,
  references, note, notes, values}; verdict 'consistent', 'inconsistent' or 'not testable'.

The literature-method R agrees with the literature by construction (same mass balance): the independent tests
are the shape of D(delta), the percolation delta against the published band and the network's 2R/D (with the
coverage that follows from it) and mass balance. Without a rate only the monotony of D is tested (in time);
everything else says it needs a rate.

All sentences are made from the templates below (plain {field}, values formatted before), so a program that
shows them can translate each one (TEMPLATES lists them).

Example:
    from sputtergisaxs.sputter_literature import literature_table
    columns, summary = literature_table(times, rows, metal='Ag', polymer='PS', rate_nm_s=0.05)
"""
from __future__ import annotations
from pathlib import Path
import json, math
import numpy as np

CONFIG_FILE = Path(__file__).resolve().parents[1] / 'configs' / 'literature_reference_v1.json'
HEX = (2. / 3.) * math.pi / (math.sqrt(3.) / 2.)     # 2.418: delta = HEX * R^3 / D^2
COVERAGE = 2. * math.pi / math.sqrt(3.)               # coverage = COVERAGE * (R/D)^2
MIN_WINDOWS = 10          # windows with a usable metal peak for the literature method
MIN_CHECK = 5             # windows for a check of the network values
CHECK_FROM_NM = 1.        # network checks use island windows from this delta on (after nucleation)
OUTLIER = math.log(2.)    # a peak this far (in ln) from the trend is not used
HOST_MATCH = math.log(1.35)
JUMP_WINDOWS = 5          # a jump is measured against the median of this many preceding peaks
GAP_MIN_S = 20.           # a stretch without a peak this long (or two modulation periods) ends the peak phase
BOOTSTRAP = 200
NEAR_ONE = .05            # 2R/D within 5 % of 1: the crossing is sensitive to the rate and t0 there
DROP_MIN = math.log(1.1)  # a drop of the binned median D below 10 % is not counted
PEAK = ('bilateral', 'one_side')
NOT_USED = ('host_peak', 'host_before', 'host_match', 'outlier', 'jump', 'late_peak')
LIT_COLUMNS = ('delta_nm', 'D_lit_nm', 'D_lit_trend_nm', 'R_lit_nm', 'R_lit_trend_nm', 'two_R_over_D_lit',
               'two_R_over_D_lit_trend', 'coverage_lit', 'two_R_over_D_net', 'coverage_net', 'eta_net', 'lit_status',
               'lit_reason')

# --------------------------------------------------------------------------------------------- texts --
NEEDS_RATE = ('needs a deposition rate: the literature method computes R from the nominal thickness '
              'delta = rate*(t - t0); without it only the monotony of D is tested (in time)')
TOO_FEW = 'fewer than {minimum} windows with a resolved peak of the metal: no literature-method values'
METHOD = ('literature method of the P03 in-situ GISAXS studies: D = 2*pi/q* of the direct peak, '
          'R = (delta*D^2/2.418)^(1/3) for hemispheres (h=R) on a hexagonal lattice, coverage = '
          '(2*pi/sqrt(3))*(R/D)^2; percolation where the trend of 2R/D reaches 1; after it this tool keeps '
          'D = 2*pi/q* but gives no R (Abr16; the P03 studies go on reporting R and 2R/D)')
REASON = dict(
    no_data='fewer than 1000 counts in the window: no value',
    before_deposition='before the start of the deposition (delta = 0): no island value',
    no_peak='no resolved peak: D and R not given',
    host_peak='peak far above every literature D(delta) at this delta: probably the host, not used',
    host_before='peak before the start of the deposition: the host, not used',
    host_match='peak at the spacing of the host peak and far from the trend of D: probably the host, not used',
    outlier='peak more than a factor 2 from the trend of D: another feature, not used',
    jump=('peak more than a factor 2 above the preceding peaks, beyond any growth since t0: a jump, not growth; '
          'not used'),
    late_peak='peak after a long stretch without a peak: another structure, not used',
    after_percolation='after the percolation of the literature method: D = 2*pi/q* kept, R not given (film)',
    no_rate='no deposition rate: only D is given',
    islands='')
TITLE = dict(
    D_monotone='D (= 2*pi/q*) does not decrease with delta',
    percolation='Percolation by the literature method against the published band',
    TS_onset='TS onset of the physical path against the published band',
    network_two_R_over_D='Network 2R/D and coverage against the literature',
    network_mass_balance='Network mass balance eta = t_eq/delta',
    TS_film_thickness='TS film thickness grows with delta')
M_NEEDS_RATE = 'needs a deposition rate'
M_FEW_PEAKS = 'fewer than {minimum} windows with a resolved peak of the metal'
M_MONO_DROP = ('Spearman rho = {rho} over {windows} windows; the binned median D drops by {drop}% between delta '
               '{low} and {high} nm, beyond the scatter')
M_MONO_DROP_T = ('Spearman rho = {rho} over {windows} windows; the binned median D drops by {drop}% between '
                 '{low} and {high} s, beyond the scatter')
M_MONO_OK = ('Spearman rho = {rho} over {windows} windows; no drop of the binned median D beyond the scatter '
             '(largest {drop}%)')
E_MONO = ('D grows or stays with delta after nucleation (delta > {start} nm); only during nucleation a slight '
          'decrease is reported')
E_MONO_T = 'D grows or stays with time during the deposition; only during nucleation a slight decrease is reported'
N_MONO_NOT_USED = '{count} peak windows were not used (host peak, outlier, jump or late peak; see lit_status).'
N_MONO_NO_RATE = 'Without a rate the nucleation phase is not known: early windows are included.'
N_LATE = ('{count} peak windows after a stretch without a peak (t = {low}-{high} s) were not used: their median '
          '2*pi/q* is {median} nm, {percent}% of the end of the trend ({end} nm).')
M_PERC = ('delta = {delta} nm at t = {time} s (68% interval {low}-{high} nm with the noise, the rate '
          '±{rate_error}% and t0 ±{t0_error} s)')
M_PERC_UNDET = ('not determined: within the noise, the rate ±{rate_error}% and t0 ±{t0_error} s, 2R/D reaches 1 '
                'from delta = {low} nm or not before the peak phase ends (delta = {last} nm, t = {time} s)')
M_PERC_START = ('delta <= {delta} nm: 2R/D is at or above 1 from the first window with a peak after nucleation '
                '(t = {time} s)')
M_PERC_NONE = ('2R/D stays below 1 (at most {ratio}) up to delta = {delta} nm (t = {time} s), where the peak '
               'phase ends, also with a {rate_error}% higher rate and t0 {t0_error} s earlier')
E_BAND = ('percolation at delta = {low}-{high} nm for {host} (published values: {count}, {gisaxs} of them by GISAXS; '
          'each with its printed ± or ±{tol}%)')
E_BAND_SCALED = ('percolation at delta = {low}-{high} nm for {host} (published values: {count}, {gisaxs} of them by '
                 'GISAXS; each with its printed ± or ±{tol}%); scaled to {rate} nm/s with chi = 0.07-0.17: '
                 '{scaled_low}-{scaled_high} nm')
E_NO_BAND = 'no published percolation threshold for {host}'
N_PERC_INTERVAL = ('68% interval: a moving-block bootstrap of the scatter of the windows around their local median, '
                   'added to the trend of D and fitted again, evaluated with the rate {rate_error}% higher and t0 '
                   '{t0_error} s earlier for the lower end and the other way round for the upper end (unless given, an '
                   'engineering choice; the toolkit options --rate-error and --t0-error set them). A systematic error '
                   'of 2*pi/q* is not in it.')
N_PERC_NEAR = ('The trend of 2R/D stays within 5% of 1 from delta = {low} to {high} nm: there a small change of the '
               'rate or t0 moves the crossing far.')
N_EARLIER = 'Earlier than every published value of this band.'
N_LATER = 'Later than every published value of this band.'
N_BOUND = 'Only a bound is measured, and it overlaps the band: the comparison is not decided.'
N_ONE_VALUE = 'Only one published value for this class: the comparison rests on it and its ±{tol}%.'
N_SHAPE_CROSS = ('With H/R = {f} (Cu on PS-b-PEO), R is {factor} times this R and 2R/D reaches 1 at '
                 'delta = {delta} nm.')
N_SHAPE_NONE = 'With H/R = {f} (Cu on PS-b-PEO), R is {factor} times this R and 2R/D stays below 1 in the peak phase.'
N_SHAPE_OPPOSITE = ('With f = 1 and with f = {f} the percolation is outside the band on opposite sides (earlier with '
                    'f = 1, later with the published f): the direction depends on the island shape.')
N_SHAPE_DEPENDS = ('With f = 1 and with f = {f} the comparison gives different verdicts: it depends on the island '
                   'shape and is not decided.')
N_SHAPE_ALSO = 'The same holds with f = {f}.'
M_TS = 'TS onset at t = {time} s, delta = {delta} nm (onsets within the slack: {low}-{high} nm)'
M_TS_NONE_SLACK = 'no TS onset on the path; onsets from delta = {low} nm are within the slack'
M_TS_NONE = 'no TS onset on the path up to delta = {delta} nm'
N_TS_MEANING = ('The TS onset is the first window the path gives to the TS network (coalescence or film '
                'candidate); a connected film is not proven.')
N_TS_NOT_EST = 'The onset is not established: no TS window at all is within the slack.'
M_NET_SIZE = ('2R/D median {median} (10-90%: {low}-{high}), coverage median {c_median} (10-90%: {c_low}-{c_high}) '
              'over {windows} island windows with delta >= 1 nm')
M_NET_FEW = 'fewer than {minimum} island windows with network R and D at delta >= 1 nm'
E_NET = ('2R/D {low}-{high} after nucleation (mass-balance studies: Au 0.8-0.95, Ag 0.66-0.95; independent: Cu on '
         'graphene 0.55-0.7), 1 at percolation; coverage {c_low}-{c_high} at delta >= 1 nm (0.5-0.65 at 1 nm, 0.58 '
         'at 3 nm for Ag on SiO2, 0.76-0.96 at 4 nm for Au on polymers), 0.56-0.9 at percolation')
N_NET_INDEPENDENT = ('The network R comes from the shape of the scattering, not from the nominal thickness: an '
                     'independent test.')
N_COVERAGE = ('coverage = (2*pi/sqrt(3))*(R/D)^2 = 0.907*(2R/D)^2 for equal hemispheres on a hexagonal lattice: it '
              'follows from 2R/D, so the two are one test.')
N_NET_ONE_BAND = ('Only one of the two published ranges holds the network value; the published ranges of 2R/D and '
                  'coverage do not map onto each other exactly.')
M_ETA = ('median {median} (10-90%: {low}-{high}) over {windows} island windows with delta >= 1 nm; with the t_eq '
         'band {band_low}-{band_high}')
E_ETA = ('1 by construction in the literature method; embedding in glassy polymers holds 0.6-6%; sticking below 1 '
         'is possible on polymers but not measured for sputtering; below {threshold} is flagged (engineering '
         'choice)')
N_ETA_LOW = ('Even the upper end of the t_eq band holds less than half of the nominal metal: most of the deposited '
             'metal is not in the islands the network sees, or the rate, t0 or the hemisphere condition is off.')
N_ETA_HIGH = 'The islands hold more metal than was deposited: check the rate and t0.'
N_ETA_SHARED = 'eta uses the same network R and D as the 2R/D check: the two are not independent of each other.'
M_FILM = ('network film thickness {t_low}-{t_high} nm (metal-equivalent {m_low}-{m_high} nm) while delta grows '
          'from {d_low} to {d_high} nm: slope {slope}')
M_FILM_FEW = 'fewer than 5 TS windows over at least 2 nm of delta'
M_FILM_NOT_ID = 'no value: the network film thickness is not identifiable'
E_FILM = 'after percolation the metal-equivalent film thickness grows about linearly with delta (slope near 1)'
N_FILM = ('The TS film thickness is not determined: in simulation the network posterior equals its prior, so a '
          'constant value says nothing about the sample.')
NOTE_CONSTRUCTION = ('The literature-method R agrees with the literature by construction: it uses the same mass '
                     'balance with the nominal thickness. The independent tests are the shape of D(delta), the '
                     'percolation delta against the published band, and the 2R/D and mass balance of the network.')
NOTE_SHAPE = ('R scales as f^(-1/3) with the shape factor f = H/R (here f = 1): the literature reports flatter Ag on '
              'oxides and f = 2.75 for Cu on PS-b-PEO.')
NOTE_RATE = ('delta = rate*(t - t0) with the given rate ({rate} nm/s), e.g. from a QCM; GISAXS does not measure it. '
             'The percolation interval allows the rate ±{rate_error}% and t0 ±{t0_error} s.')
NOTE_T0 = ('t0 not given: assumed 0 s (the first frame); delta is an upper bound if the deposition started at or '
           'after the first frame, and each 1 s of error in t0 shifts delta by {rate} nm.')
NOTE_D_PEAK = 'D = 2*pi/q* can be off by more than 40% for D > 10 nm once coalescence starts (Hod17).'
NOTE_STICKING = ('The condensation (sticking) coefficient of these metals on these polymers is not known; the '
                 'literature method assumes that all deposited metal stays on the surface.')
NOTE_CHAMBER = ('Published thresholds come from other chambers and rates; the rate dependence (delta_c ~ F^-chi, '
                'chi 0.07-0.17) is weaker than differences between chambers.')
NOTE_CU = 'Cu: the literature method needs no network, but the TS onset and the network checks use the Ag model.'
NOTE_HOST_PEAKS = '{count} peak windows probably belong to the host (median D {D} nm) and are not used.'
HOST_UNKNOWN = 'Polymer type not given: the band of all polymers is used for {metal}.'
HOST_NO_DATA = 'No published percolation data for {polymer}: the band of {host} is used.'
HOST_INCLUDED = 'The band of {host} includes published values for {polymer}.'
TEMPLATES = (NEEDS_RATE, TOO_FEW, METHOD, *REASON.values(), *TITLE.values(), M_NEEDS_RATE, M_FEW_PEAKS, M_MONO_DROP,
             M_MONO_DROP_T, M_MONO_OK, E_MONO, E_MONO_T, N_MONO_NOT_USED, N_MONO_NO_RATE, N_LATE, M_PERC,
             M_PERC_UNDET, M_PERC_START, M_PERC_NONE, E_BAND, E_BAND_SCALED, E_NO_BAND, N_PERC_INTERVAL, N_PERC_NEAR,
             N_EARLIER, N_LATER, N_BOUND, N_ONE_VALUE, N_SHAPE_CROSS, N_SHAPE_NONE, N_SHAPE_OPPOSITE, N_SHAPE_DEPENDS,
             N_SHAPE_ALSO, M_TS, M_TS_NONE_SLACK, M_TS_NONE, N_TS_MEANING, N_TS_NOT_EST, M_NET_SIZE, M_NET_FEW, E_NET,
             N_NET_INDEPENDENT, N_COVERAGE, N_NET_ONE_BAND, M_ETA, E_ETA, N_ETA_LOW, N_ETA_HIGH, N_ETA_SHARED, M_FILM,
             M_FILM_FEW, M_FILM_NOT_ID, E_FILM, N_FILM, NOTE_CONSTRUCTION, NOTE_SHAPE, NOTE_RATE, NOTE_T0,
             NOTE_D_PEAK, NOTE_STICKING, NOTE_CHAMBER, NOTE_CU, NOTE_HOST_PEAKS, HOST_UNKNOWN, HOST_NO_DATA,
             HOST_INCLUDED)
"""Every sentence this module writes into the summary and the table (filled with str.format)."""
OXIDES = {'si', 'sio2', 'siox', 'sio', 'silicon', 'tio2', 'al2o3', 'oxide', 'glass', 'quartz', 'a-c', 'ito', 'sapphire'}
UNKNOWN_POLYMER = {'', 'polymer', 'unknown', 'none'}


def fmt(x):
    """Numbers in the sentences: three significant digits, no exponent below 1e5."""
    return '–' if x is None else f'{x:.3g}'


def load_config(path=CONFIG_FILE):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# ------------------------------------------------------------------------------------------ host class --
def host_class(metal, polymer, config):
    """(label, keys of config['percolation'], note) of the published band that fits metal and host."""
    model = metal if metal in ('Ag', 'Au', 'Cu') else 'Ag'
    name = (polymer or '').strip()
    key = name.lower().replace(' ', '')
    if key in OXIDES:
        label = {'Au': 'Au/Si', 'Ag': 'Ag/oxides', 'Cu': 'Cu/oxides'}[model]
        return label, [label], ''
    if key in UNKNOWN_POLYMER:
        keys = ['Au/PS', 'Au/other polymers'] if model == 'Au' else [f'{model}/polymers']
        return (f'{model}/polymers', keys, HOST_UNKNOWN.format(metal=model))
    if model == 'Au':
        label = 'Au/PS' if key in ('ps', 'polystyrene') else 'Au/other polymers'
    else:
        label = f'{model}/polymers'
    systems = ' '.join(e.get('system', '') for e in config['percolation'].get(label, [])).lower()
    words = set(systems.replace(',', ' ').replace('(', ' ').replace(')', ' ').split())
    note = (HOST_INCLUDED if key in words else HOST_NO_DATA).format(polymer=name, host=label)
    return label, [label], note


def tolerance(config):
    """Relative ± given to a published threshold printed without one (config 'band_tolerance')."""
    return float(config.get('band_tolerance', {}).get('relative', .1))


def band(keys, config, rate_nm_s=None):
    """Published thresholds of the class: (entries, low, high, rate-scaled low, high).

    Each value counts with its published ± (plus_minus_nm), or ± tolerance(config) of itself when none is
    printed, so that a class of one value is not a band of zero width.
    """
    entries = [e for k in keys for e in config['percolation'].get(k, [])]
    tol = tolerance(config)
    values = []
    for e in entries:
        if e.get('range_nm'):
            values += list(e['range_nm'])
        elif e.get('delta_c_nm'):
            pm = e.get('plus_minus_nm') or tol * e['delta_c_nm']
            values += [e['delta_c_nm'] - pm, e['delta_c_nm'] + pm]
    if not values:
        return entries, None, None, None, None
    scaled = []
    chi = config['rate_dependence']['chi_range']
    if rate_nm_s:
        for e in entries:
            if e.get('rate_nm_s') and e.get('delta_c_nm'):
                scaled += [e['delta_c_nm'] * (rate_nm_s / e['rate_nm_s']) ** (-c) for c in chi]
    low, high = round(min(values), 6), round(max(values), 6)
    return entries, low, high, (min(scaled) if scaled else None), (max(scaled) if scaled else None)


def reference_envelope(delta, curves):
    """Largest literature D at delta over all curves (log-log interpolation, ends held)."""
    best = None
    for c in curves:
        x = np.log([p[0] for p in c['points']]); y = np.log([p[1] for p in c['points']])
        value = float(np.exp(np.interp(math.log(max(delta, 1e-6)), x, y)))
        best = value if best is None else max(best, value)
    return best


# ------------------------------------------------------------------------------------------- fitting --
def isotonic_median(y):
    """Non-decreasing L1 fit of a sequence: pool-adjacent-violators with block medians."""
    blocks = []
    for v in y:
        blocks.append([np.array([v], float), float(v)])
        while len(blocks) > 1 and blocks[-2][1] > blocks[-1][1]:
            b = blocks.pop(); a = blocks.pop()
            values = np.concatenate([a[0], b[0]])
            blocks.append([values, float(np.median(values))])
    return np.concatenate([np.full(len(v), m) for v, m in blocks]) if blocks else np.array([])


def _step(times, fit_t, fit_v):
    """The fit at each time: the value of the last fitted window at or before it (the first before the first)."""
    idx = np.searchsorted(fit_t, times, side='right') - 1
    return fit_v[np.clip(idx, 0, len(fit_v) - 1)]


def ratio(delta, D):
    """2R/D of the literature method: 2*(delta/(2.418*D))^(1/3)."""
    return 2. * (delta / (HEX * D)) ** (1. / 3.)


def crossing(deltas, D_trend):
    """(delta where 2R/D first reaches 1, index) along the windows; (None, None) when it never does."""
    r = ratio(deltas, D_trend)
    above = np.flatnonzero(r >= 1.)
    if not len(above):
        return None, None
    k = int(above[0])
    if k == 0:
        return float(deltas[0]), 0
    r0, r1 = r[k - 1], r[k]
    if r1 > r0:
        return float(deltas[k - 1] + (1. - r0) / (r1 - r0) * (deltas[k] - deltas[k - 1])), k
    return float(deltas[k]), k


def local_scatter(y, half):
    """Scatter of each value around the median of its +-half neighbours (the noise of the windows).

    The residuals of the isotonic fit itself are no measure of the noise: a rising series is fitted window by
    window (blocks of one), with residuals of zero.
    """
    y = np.asarray(y, float)
    return np.array([y[i] - np.median(y[max(0, i - half):i + half + 1]) for i in range(len(y))])


def bootstrap_trends(t_fit, e, fit, t_span, block, rng, n=BOOTSTRAP):
    """D trends at t_span of n refits with residuals resampled in moving blocks."""
    out = []
    m = len(e); block = max(1, min(block, m))
    starts_max = m - block + 1
    for _ in range(n):
        pieces = []
        while sum(len(p) for p in pieces) < m:
            s = int(rng.integers(0, starts_max)); pieces.append(e[s:s + block])
        sample = np.concatenate(pieces)[:m]
        out.append(np.exp(_step(t_span, t_fit, isotonic_median(fit + sample))))
    return out


def jumps(t, D, idx, t0):
    """Peaks (of idx, in time order) more than a factor 2 above the median of the preceding kept peaks, beyond the
    growth delta allows since t0 (D grows at most as fast as delta in every published D(delta))."""
    kept, out = [], []
    for i in idx:
        if len(kept) >= 3:
            prev = kept[-JUMP_WINDOWS:]
            ref, t_ref = float(np.median(D[prev])), float(np.median(t[prev]))
            growth = (t[i] - t0) / (t_ref - t0) if t_ref - t0 > 0 else math.inf
            if math.log(D[i] / ref) > OUTLIER + math.log(max(1., growth)):
                out.append(i)
                continue
        kept.append(i)
    return out


def theil_sen(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    dx = np.subtract.outer(x, x); dy = np.subtract.outer(y, y)
    pair = dx > 0
    return float(np.median(dy[pair] / dx[pair])) if pair.any() else None


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    if len(x) < 3 or np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def binned_drop(position, logD, bins_max=12, per_bin=8):
    """Largest drop of the binned median ln D beyond twice its standard error: (drop, low, high, significant)."""
    n = len(logD)
    bins = min(bins_max, n // per_bin)
    if bins < 3:
        return None
    edges = np.linspace(0, n, bins + 1).astype(int)
    med, se, pos = [], [], []
    for a, b in zip(edges, edges[1:]):
        v = logD[a:b]
        mad = 1.4826 * np.median(np.abs(v - np.median(v)))
        med.append(float(np.median(v))); se.append(1.2533 * mad / math.sqrt(len(v))); pos.append(float(np.median(position[a:b])))
    best = (0., pos[0], pos[0], False)
    for i in range(bins):
        for j in range(i + 1, bins):
            drop = med[i] - med[j]
            significant = drop > max(DROP_MIN, 2 * math.hypot(se[i], se[j]))
            if (significant, drop) > (best[3], best[0]):
                best = (drop, pos[i], pos[j], significant)
    return best


def _quantiles(values):
    v = np.asarray([x for x in values if x is not None], float)
    if not len(v):
        return None, None, None
    return float(np.median(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90))


def _check(cid, verdict, measured, expected, refs, notes=(), **values):
    notes = [n for n in notes if n]
    return dict(id=cid, title=TITLE[cid], verdict=verdict, measured=measured, expected=expected, references=list(refs),
                note=' '.join(notes), notes=notes, values=values)


def _overlap(lo, hi, blo, bhi):
    return (hi is None or hi >= blo) and (lo is None or lo <= bhi)


# ------------------------------------------------------------------------------------- percolation --
def _crossing_at(t_sp, trend_sp, rate, t0, f=1., start=0.):
    """(delta, index) of the crossing of 2R/D = 1 with this rate, t0 and shape factor f (2R/D ~ f^(-1/3)), searched
    from delta > start on (after nucleation)."""
    d = rate * (t_sp - t0)
    ok = d > max(start, 0.)
    if not ok.any():
        return None, None
    return crossing(d[ok], trend_sp[ok] * f)


def percolation_estimate(t_sp, trend_sp, boots, rate, t0, rate_error, t0_error, f=1., start=0.):
    """The percolation with its 68 % interval over the noise (boots) and the rate and t0 errors.

    The crossing comes earliest with a higher rate and an earlier t0 (every delta larger) and latest with a lower
    rate and a later t0; the lower end is the 16 % quantile of the resampled crossings at the first setting, the
    upper end the 84 % quantile at the second (infinite when 2R/D does not reach 1 in more resamples). Only windows
    with delta > start count: a shift of t0 changes the delta of the first windows many times over, and no
    published percolation comes before the end of nucleation. state: 'reached', 'first_window' (at or above 1 from
    the first counted window with a peak: an upper bound), 'not_determined' (it may reach 1 within the errors but
    need not) or 'not_reached' (not even at the first setting).
    """
    early, late = (rate * (1. + rate_error), t0 - t0_error), (rate * (1. - rate_error), t0 + t0_error)

    def quantile(setting, q, rounding):
        values = sorted(math.inf if c is None else c
                        for c in (_crossing_at(t_sp, b, *setting, f, start)[0] for b in boots))
        return values[int(rounding(q * (len(values) - 1)))]

    point, k = _crossing_at(t_sp, trend_sp, rate, t0, f, start)
    lo, hi = quantile(early, .16, math.floor), quantile(late, .84, math.ceil)
    if not math.isfinite(lo):
        state = 'not_reached'
    elif point is None or not math.isfinite(hi):
        state = 'not_determined'
    else:
        state = 'first_window' if k == 0 else 'reached'
    if state == 'first_window':
        lo = 0.
    elif state == 'reached':  # the interval holds the point itself
        lo, hi = min(lo, point), max(hi, point)
    to_time = lambda value, setting: setting[1] + value / setting[0] if math.isfinite(value) else None  # noqa: E731
    d = rate * (t_sp - t0)
    counted = d > max(start, 0.)
    r = ratio(d[counted], trend_sp[counted] * f)
    near = d[counted][np.abs(r - 1.) <= NEAR_ONE]
    return dict(state=state, delta_nm=point, index=k, time_s=t0 + point / rate if point is not None else None,
                low_nm=lo if math.isfinite(lo) else None, high_nm=hi if math.isfinite(hi) else None,
                low_s=to_time(lo, early) if state != 'first_window' else t0, high_s=to_time(hi, late),
                max_ratio=float(r.max()) if len(r) else None,
                near_one_delta_nm=[float(near.min()), float(near.max())] if len(near) >= 3 else None)


# ------------------------------------------------------------------------------------------- the table --
def literature_table(times, rows, *, metal, polymer=None, rate_nm_s=None, t0_s=None, onset=None, period=None,
                     teq_factor=None, config=None, seed=0, rate_error=None, t0_error_s=None, film_identifiable=False):
    """(columns per window, summary) of the literature method for one series sorted by time.

    rows: one dict per window with D_corr_nm, feature_status, no_data (bool), kind ('islands', 'film',
    'pre_nucleation', 'no_data', ...), R_nm, D_nm, t_eq_nm (physical path), film_thickness / metal_fraction
    (TS network, film windows) and module_path. onset: the TS-onset part of the physical summary. rate_error
    (relative) and t0_error_s widen the percolation interval (defaults in the config's 'uncertainty': a t0 that
    is not given, and so assumed at the first frame, gets the larger error).
    film_identifiable: the TS film thickness is a measurement (it is not with the current network: the film
    check is then not testable).
    """
    cfg = config or load_config()
    n = len(rows)
    t = np.array([np.nan if x is None else x for x in times], float)
    model = metal if metal in ('Ag', 'Au', 'Cu') else 'Ag'
    t0 = t0_s if t0_s is not None else 0.
    has_rate = rate_nm_s is not None and rate_nm_s > 0
    errors = cfg.get('uncertainty', {})
    rate_error = float(errors.get('rate_relative', .1) if rate_error is None else rate_error)
    if t0_error_s is None:  # an assumed t0 (the first frame) is less sure than a given one
        t0_error_s = errors.get('t0_s', 2.) if t0_s is None else errors.get('t0_given_s', .5)
    t0_error_s = float(t0_error_s)
    if not 0. <= rate_error < 1. or t0_error_s < 0.:
        raise ValueError('rate error must be in [0, 1) and t0 error >= 0 s')
    delta = rate_nm_s * (t - t0) if has_rate else np.full(n, np.nan)
    D = np.array([r.get('D_corr_nm') if (r.get('D_corr_nm') and r.get('feature_status') in PEAK and not r.get('no_data'))
                  else np.nan for r in rows], float)
    status = ['no_data' if r.get('no_data') else 'no_peak' for r in rows]
    started = (t >= t0) if (has_rate or t0_s is not None) else np.ones(n, bool)
    for i in range(n):
        if status[i] != 'no_data' and not started[i]:
            status[i] = 'host_before' if np.isfinite(D[i]) else 'before_deposition'
        elif np.isfinite(D[i]) and status[i] == 'no_peak':
            status[i] = 'islands'
    curves = cfg['D_curves']
    if has_rate:
        factor = cfg.get('host_peak_factor', 2.)
        for i in np.flatnonzero(np.array(status) == 'islands'):
            if delta[i] > 0 and D[i] > factor * reference_envelope(delta[i], curves):
                status[i] = 'host_peak'
            elif delta[i] <= 0:
                status[i] = 'host_before'
    # peak phase: from the first usable peak to the last one before a long stretch without a peak
    gap = max(GAP_MIN_S, 2 * (period or 0.))
    usable = [i for i in range(n) if status[i] == 'islands' and np.isfinite(t[i])]
    for a, b in zip(usable, usable[1:]):
        if t[b] - t[a] > gap:
            for k in usable[usable.index(b):]:
                status[k] = 'late_peak'
            break
    columns = [{k: None for k in LIT_COLUMNS} for _ in range(n)]
    trend = np.full(n, np.nan)
    host = [i for i in range(n) if status[i] in ('host_peak', 'host_before')]
    D_host = float(np.median(D[host])) if len(host) >= 2 else None

    def host_or(i, other):
        return 'host_match' if (D_host and abs(math.log(D[i] / D_host)) <= HOST_MATCH) else other

    # jumps far above the preceding peaks (an isotonic fit would absorb them), then peaks far from the trend
    for i in jumps(t, D, [i for i in range(n) if status[i] == 'islands'], t0):
        status[i] = host_or(i, 'jump')
    fit_idx = [i for i in range(n) if status[i] == 'islands']
    t_fit = fit = None
    if len(fit_idx) >= MIN_WINDOWS:
        for _ in range(2):  # fit, leave out peaks far from it, fit again
            y = np.log(D[fit_idx]); fit = isotonic_median(y); t_fit = t[fit_idx]
            far = [i for i, r in zip(fit_idx, y - fit) if abs(r) > OUTLIER]
            if not far:
                break
            for i in far:
                status[i] = host_or(i, 'outlier')
            fit_idx = [i for i in fit_idx if status[i] == 'islands']
            if len(fit_idx) < MIN_WINDOWS:
                break
        y = np.log(D[fit_idx]); fit = isotonic_median(y); t_fit = t[fit_idx]
    enough = len(fit_idx) >= MIN_WINDOWS
    span = np.zeros(n, bool)
    if enough:
        span = (t >= t_fit[0]) & (t <= t_fit[-1]) & np.array([s != 'no_data' for s in status])
        trend[span] = np.exp(_step(t[span], t_fit, fit))
    # percolation
    perc = dict(state=None, delta_nm=None, time_s=None, interval_delta_nm=None, interval_time_s=None, reached=False,
                bound=None, max_two_R_over_D_trend=None, last_delta_nm=None, last_time_s=None, nominal_delta_nm=None,
                near_one_delta_nm=None, rate_error=rate_error, t0_error_s=t0_error_s,
                method=N_PERC_INTERVAL.format(rate_error=fmt(100 * rate_error), t0_error=fmt(t0_error_s)))
    t_perc = None
    shape = None
    if enough and has_rate:
        idx_span = np.flatnonzero(span)
        t_sp, trend_sp = t[idx_span], trend[idx_span]
        spacing = float(np.median(np.diff(t_fit))) if len(t_fit) > 1 else 0.5
        block = max(3, int(round(max(period or 0., 2.5) / max(spacing, 1e-3))))
        residual = local_scatter(np.log(D[fit_idx]), max(2, block // 2))
        boots = bootstrap_trends(t_fit, residual, fit, t_sp, block, np.random.default_rng(seed))
        start = cfg.get('nucleation_delta_nm', .4)
        est = percolation_estimate(t_sp, trend_sp, boots, rate_nm_s, t0, rate_error, t0_error_s, start=start)
        positive = delta[idx_span] > start
        perc.update(state=est['state'], nominal_delta_nm=est['delta_nm'], max_two_R_over_D_trend=est['max_ratio'],
                    near_one_delta_nm=est['near_one_delta_nm'],
                    last_delta_nm=float(delta[idx_span][positive][-1]) if positive.any() else None,
                    last_time_s=float(t_sp[positive][-1]) if positive.any() else None)
        if est['state'] in ('reached', 'first_window'):
            t_perc = est['time_s']
            perc.update(delta_nm=est['delta_nm'], time_s=t_perc, reached=True,
                        bound='upper' if est['state'] == 'first_window' else None)
        elif est['state'] == 'not_reached':
            perc['bound'] = 'lower'
        if est['state'] != 'not_reached':
            perc.update(interval_delta_nm=[est['low_nm'], est['high_nm']], interval_time_s=[est['low_s'], est['high_s']])
        f = (cfg['shape'].get(model) or {}).get('f')
        if model == 'Cu' and f:
            shape = dict(f=f, **percolation_estimate(t_sp, trend_sp, boots, rate_nm_s, t0, rate_error, t0_error_s,
                                                     f=f, start=start))
    # per-window values
    for i in range(n):
        c = columns[i]
        c['delta_nm'] = float(delta[i]) if has_rate and np.isfinite(delta[i]) else None
        if t_perc is not None and t[i] > t_perc and status[i] == 'islands':
            status[i] = 'after_percolation'
        islands = status[i] == 'islands'
        after = status[i] == 'after_percolation'
        if (not has_rate and islands) or after:
            c['D_lit_nm'] = float(D[i])  # 2*pi/q* stays the spacing after the percolation
        if span[i] and status[i] != 'no_data':
            c['D_lit_trend_nm'] = float(trend[i])
        if has_rate and span[i] and delta[i] > 0:
            c['two_R_over_D_lit_trend'] = float(ratio(delta[i], trend[i]))
            if not (t_perc is not None and t[i] > t_perc):
                c['R_lit_trend_nm'] = float((delta[i] * trend[i] ** 2 / HEX) ** (1. / 3.))
        if has_rate and islands and delta[i] > 0:
            R = (delta[i] * D[i] ** 2 / HEX) ** (1. / 3.)
            c.update(D_lit_nm=float(D[i]), R_lit_nm=float(R), two_R_over_D_lit=float(2 * R / D[i]),
                     coverage_lit=float(COVERAGE * (R / D[i]) ** 2))
        r = rows[i]
        if r.get('kind') == 'islands' and r.get('R_nm') and r.get('D_nm'):
            c['two_R_over_D_net'] = 2. * r['R_nm'] / r['D_nm']
            c['coverage_net'] = COVERAGE * (r['R_nm'] / r['D_nm']) ** 2
        if has_rate and r.get('t_eq_nm') is not None and delta[i] > 0:
            c['eta_net'] = r['t_eq_nm'] / delta[i]
        key = status[i]
        shown = {'host_before': 'host_peak', 'host_match': 'host_peak', 'jump': 'outlier'}.get(key, key)
        if key == 'islands' and not has_rate:
            key = 'no_rate'
        c['lit_status'] = shown
        c['lit_reason'] = REASON.get(key, '')
    summary = _summary(cfg, columns, rows, status, t, delta, D, trend, span, fit_idx, host, D_host, perc, shape,
                       has_rate, rate_nm_s, t0_s, model, metal, polymer, onset, teq_factor, enough, film_identifiable)
    return columns, summary


# ------------------------------------------------------------------------------------------- checks --
def _late_peaks(t, D, trend, status):
    """(count, first time, last time, median D, end of the trend) of the late peaks, or None."""
    late = [i for i, s in enumerate(status) if s == 'late_peak']
    ends = np.flatnonzero(np.isfinite(trend))
    if not late or not len(ends):
        return None
    return len(late), float(t[late[0]]), float(t[late[-1]]), float(np.median(D[late])), float(trend[ends[-1]])


def _monotone(t, delta, D, trend, status, fit_idx, has_rate, start):
    refs = ['Sch15', 'Bulut24', 'GenTh23', 'Hod17', 'Sch21', 'Abr16']
    usable = sorted((i for i in fit_idx if not has_rate or delta[i] > start), key=lambda i: t[i])
    expected = E_MONO.format(start=fmt(start)) if has_rate else E_MONO_T
    not_used = [i for i, s in enumerate(status) if s in NOT_USED]
    late = _late_peaks(t, D, trend, status)
    values = {}
    notes = [N_MONO_NOT_USED.format(count=len(not_used)) if not_used else '', '' if has_rate else N_MONO_NO_RATE]
    if late:
        count, low, high, median, end = late
        notes.append(N_LATE.format(count=count, low=fmt(low), high=fmt(high), median=fmt(median),
                                   percent=fmt(100 * median / end), end=fmt(end)))
        values['late_peaks'] = dict(count=count, time_s=[low, high], median_nm=median, trend_end_nm=end,
                                    ratio=median / end)
    if len(usable) < MIN_WINDOWS:
        return _check('D_monotone', 'not testable', M_FEW_PEAKS.format(minimum=MIN_WINDOWS), expected, refs, notes,
                      not_used=len(not_used), **values)
    position = delta[usable] if has_rate else t[usable]
    logD = np.log(D[usable])
    rho = spearman(position, logD)
    found = binned_drop(position, logD)
    if found is None:
        return _check('D_monotone', 'not testable', M_FEW_PEAKS.format(minimum=24), expected, refs, notes,
                      rho=rho, windows=len(usable), not_used=len(not_used), **values)
    drop, low, high, significant = found
    pct = fmt(100 * (1 - math.exp(-drop))) if drop > 0 else '0'
    if significant:
        measured = (M_MONO_DROP if has_rate else M_MONO_DROP_T).format(rho=fmt(rho), windows=len(usable), drop=pct,
                                                                       low=fmt(low), high=fmt(high))
    else:
        measured = M_MONO_OK.format(rho=fmt(rho), windows=len(usable), drop=pct)
    return _check('D_monotone', 'inconsistent' if significant else 'consistent', measured, expected, refs, notes,
                  rho=rho, windows=len(usable), drop_fraction=1 - math.exp(-drop) if drop > 0 else 0.,
                  drop_between=[low, high], drop_axis='delta_nm' if has_rate else 'time_s', not_used=len(not_used),
                  **values)


def _band_expected(label, entries, low, high, s_low, s_high, rate, tol):
    if low is None:
        return E_NO_BAND.format(host=label)
    gisaxs = sum(1 for e in entries if str(e.get('method', '')).startswith('GISAXS'))
    fields = dict(low=fmt(low), high=fmt(high), host=label, count=len(entries), gisaxs=gisaxs, tol=fmt(100 * tol))
    if s_low is not None:
        return E_BAND_SCALED.format(rate=fmt(rate), scaled_low=fmt(s_low), scaled_high=fmt(s_high), **fields)
    return E_BAND.format(**fields)


def _against_band(lo, hi, blo, bhi, one_sided):
    """(verdict, note, side): side 'earlier' or 'later' when the measured interval lies outside the band."""
    if blo is None:
        return 'not testable', '', None
    if _overlap(lo, hi, blo, bhi):
        return ('not testable', N_BOUND, None) if one_sided else ('consistent', '', None)
    if hi is not None and hi < blo:
        return 'inconsistent', N_EARLIER, 'earlier'
    return 'inconsistent', N_LATER, 'later'


def _measured_interval(est_state, perc_like):
    """(low, high, one_sided) of a percolation (perc dict or a shape estimate) for the band comparison."""
    if est_state == 'not_reached':
        return perc_like['last'], None, True
    if est_state == 'first_window':
        return 0., perc_like['high'], True
    if est_state == 'not_determined':
        return perc_like['low'], None, True
    return perc_like['low'], perc_like['high'], False


def _percolation_check(perc, shape, label, refs, entries, blo, bhi, expected, host_note, shape_note, tol):
    notes = [host_note]
    errors = dict(rate_error=fmt(100 * perc['rate_error']), t0_error=fmt(perc['t0_error_s']))
    state = perc['state']
    if state is None:
        return _check('percolation', 'not testable', M_FEW_PEAKS.format(minimum=MIN_WINDOWS), expected, refs,
                      notes + [shape_note])
    low, high = (perc['interval_delta_nm'] or [None, None])
    if state == 'not_reached':
        measured = M_PERC_NONE.format(ratio=fmt(perc['max_two_R_over_D_trend']), delta=fmt(perc['last_delta_nm']),
                                      time=fmt(perc['last_time_s']), **errors)
    elif state == 'first_window':
        measured = M_PERC_START.format(delta=fmt(perc['delta_nm']), time=fmt(perc['time_s']))
    elif state == 'not_determined':
        measured = M_PERC_UNDET.format(low=fmt(low), last=fmt(perc['last_delta_nm']), time=fmt(perc['last_time_s']),
                                       **errors)
    else:
        measured = M_PERC.format(delta=fmt(perc['delta_nm']), time=fmt(perc['time_s']), low=fmt(low), high=fmt(high),
                                 **errors)
    notes.insert(0, perc['method'])
    near = perc.get('near_one_delta_nm')
    if near and near[1] >= 1.25 * near[0]:
        notes.append(N_PERC_NEAR.format(low=fmt(near[0]), high=fmt(near[1])))
    lo, hi, one_sided = _measured_interval(state, dict(low=low, high=high, last=perc['last_delta_nm']))
    verdict, why, side = _against_band(lo, hi, blo, bhi, one_sided)
    values = dict(measured_interval_nm=[lo, hi], band_nm=[blo, bhi], host_class=label, state=state)
    if shape is not None:  # Cu: the verdict with f = 1 and with the published shape factor
        s_lo, s_hi, s_one = _measured_interval(shape['state'], dict(low=shape['low_nm'], high=shape['high_nm'],
                                                                   last=perc['last_delta_nm']))
        s_verdict, _s_why, s_side = _against_band(s_lo, s_hi, blo, bhi, s_one)
        values['shape'] = dict(f=shape['f'], state=shape['state'], delta_nm=shape['delta_nm'],
                               measured_interval_nm=[s_lo, s_hi], verdict=s_verdict)
        f = fmt(shape['f'])
        if s_verdict != verdict:
            verdict, why = 'not testable', N_SHAPE_DEPENDS.format(f=f)
        elif verdict == 'inconsistent' and side != s_side:
            why = N_SHAPE_OPPOSITE.format(f=f)
        elif verdict == 'inconsistent':
            why = why + ' ' + N_SHAPE_ALSO.format(f=f)
    if len(entries) == 1:
        notes.append(N_ONE_VALUE.format(tol=fmt(100 * tol)))
    return _check('percolation', verdict, measured, expected, refs, notes + [shape_note, why], **values)


def _onset_check(onset, rate, t0, last_delta, label, refs, blo, bhi, expected):
    onset = onset or {}
    when, interval = onset.get('film_onset_s'), onset.get('film_onset_interval_s')
    notes = [N_TS_MEANING]
    to_delta = lambda s: rate * max(0., s - t0)  # noqa: E731
    if when is not None:
        d = to_delta(when)
        lo, hi = (to_delta(interval[0]), to_delta(interval[1])) if interval else (d, d)
        measured = M_TS.format(time=fmt(when), delta=fmt(d), low=fmt(lo), high=fmt(hi))
        if onset.get('film_onset_no_TS_within_slack'):
            notes.append(N_TS_NOT_EST)
            hi = None
        verdict, why, _side = _against_band(lo, hi, blo, bhi, hi is None)
        return _check('TS_onset', verdict, measured, expected, refs, notes + [why], delta_nm=d, interval_nm=[lo, hi])
    if interval:
        lo = to_delta(interval[0])
        verdict, why, _side = _against_band(lo, None, blo, bhi, True)
        return _check('TS_onset', verdict, M_TS_NONE_SLACK.format(low=fmt(lo)), expected, refs, notes + [why],
                      interval_nm=[lo, None])
    verdict, why, _side = _against_band(last_delta, None, blo, bhi, True)
    return _check('TS_onset', verdict, M_TS_NONE.format(delta=fmt(last_delta)), expected, refs, notes + [why],
                  interval_nm=[last_delta, None])


def _network_checks(cfg, columns, rows, delta, has_rate, teq_factor):
    out = []
    ratio_band = cfg['two_R_over_D_test_from_1nm']; cover_band = cfg['coverage_test_from_1nm']
    eta_cfg = cfg['mass_balance']
    eta_refs = sorted({r for note in eta_cfg['notes'] for r in note['refs']} | set(eta_cfg['refs']))
    size_refs = sorted(set(ratio_band['refs']) | set(cover_band['refs']))
    if not has_rate:
        for cid, refs in (('network_two_R_over_D', size_refs), ('network_mass_balance', eta_refs)):
            out.append(_check(cid, 'not testable', M_NEEDS_RATE, '', refs))
        return out
    late = [i for i in range(len(rows)) if delta[i] >= CHECK_FROM_NM]
    (lo_r, hi_r), (lo_c, hi_c) = ratio_band['range'], cover_band['range']
    expected = E_NET.format(low=fmt(lo_r), high=fmt(hi_r), c_low=fmt(lo_c), c_high=fmt(hi_c))
    notes = [N_NET_INDEPENDENT, N_COVERAGE]
    pairs = [(columns[i]['two_R_over_D_net'], columns[i]['coverage_net']) for i in late
             if columns[i]['two_R_over_D_net'] is not None]
    if len(pairs) < MIN_CHECK:
        out.append(_check('network_two_R_over_D', 'not testable', M_NET_FEW.format(minimum=MIN_CHECK), expected,
                          size_refs, notes))
    else:
        med, p10, p90 = _quantiles([p[0] for p in pairs])
        c_med, c10, c90 = _quantiles([p[1] for p in pairs])
        in_ratio, in_cover = lo_r <= med <= hi_r, lo_c <= c_med <= hi_c
        verdict = 'consistent' if (in_ratio or in_cover) else 'inconsistent'
        out.append(_check('network_two_R_over_D', verdict,
                          M_NET_SIZE.format(median=fmt(med), low=fmt(p10), high=fmt(p90), c_median=fmt(c_med),
                                            c_low=fmt(c10), c_high=fmt(c90), windows=len(pairs)),
                          expected, size_refs, notes + ([N_NET_ONE_BAND] if in_ratio != in_cover else []),
                          median=med, p10=p10, p90=p90, coverage_median=c_med, coverage_p10=c10, coverage_p90=c90,
                          windows=len(pairs), band=[lo_r, hi_r], coverage_band=[lo_c, hi_c]))
    eta = [columns[i]['eta_net'] for i in late if columns[i]['eta_net'] is not None]
    threshold = eta_cfg['warning_threshold']
    expected = E_ETA.format(threshold=fmt(threshold))
    if len(eta) < MIN_CHECK:
        out.append(_check('network_mass_balance', 'not testable', M_NET_FEW.format(minimum=MIN_CHECK), expected,
                          eta_refs, [N_ETA_SHARED]))
        return out
    med, p10, p90 = _quantiles(eta)
    f = teq_factor or 1.
    verdict, note = 'consistent', ''
    if med * f < threshold:
        verdict, note = 'inconsistent', N_ETA_LOW
    elif med / f > 1.5:
        verdict, note = 'inconsistent', N_ETA_HIGH
    out.append(_check('network_mass_balance', verdict, M_ETA.format(median=fmt(med), low=fmt(p10), high=fmt(p90),
                      windows=len(eta), band_low=fmt(med / f), band_high=fmt(med * f)), expected, eta_refs,
                      [note, N_ETA_SHARED], median=med, p10=p10, p90=p90, windows=len(eta), band_factor=f))
    return out


def _film_check(rows, delta, has_rate, identifiable):
    refs = ['Sch15', 'Bulut24']
    if not has_rate:
        return _check('TS_film_thickness', 'not testable', M_NEEDS_RATE, E_FILM, refs, [N_FILM])
    if not identifiable:  # the expectation stays for a film model that measures the thickness
        return _check('TS_film_thickness', 'not testable', M_FILM_NOT_ID, E_FILM, refs, [N_FILM])
    pts = [(delta[i], r['film_thickness'], r['film_thickness'] * (r.get('metal_fraction') or 1.))
           for i, r in enumerate(rows) if r.get('kind') == 'film' and r.get('film_thickness') and np.isfinite(delta[i])]
    if len(pts) < 5 or max(p[0] for p in pts) - min(p[0] for p in pts) < 2.:
        return _check('TS_film_thickness', 'not testable', M_FILM_FEW, E_FILM, refs)
    d, th, metal = (np.array(v) for v in zip(*pts))
    slope = theil_sen(d, metal)
    verdict = 'consistent' if slope is not None and 0.3 <= slope <= 1.5 else 'inconsistent'
    measured = M_FILM.format(t_low=fmt(np.percentile(th, 10)), t_high=fmt(np.percentile(th, 90)),
                             m_low=fmt(np.percentile(metal, 10)), m_high=fmt(np.percentile(metal, 90)),
                             d_low=fmt(d.min()), d_high=fmt(d.max()), slope=fmt(slope))
    return _check('TS_film_thickness', verdict, measured, E_FILM, refs, slope=slope, windows=len(pts))


def _summary(cfg, columns, rows, status, t, delta, D, trend, span, fit_idx, host, D_host, perc, shape, has_rate, rate,
             t0_s, model, metal, polymer, onset, teq_factor, enough, film_identifiable):
    label, keys, host_note = host_class(metal, polymer, cfg)
    entries, blo, bhi, s_lo, s_hi = band(keys, cfg, rate if has_rate else None)
    tol = tolerance(cfg)
    expected = _band_expected(label, entries, blo, bhi, s_lo, s_hi, rate, tol)
    refs = sorted({e['ref'] for e in entries})
    start = cfg.get('nucleation_delta_nm', .4)
    checks = [_monotone(t, delta, D, trend, status, list(fit_idx), has_rate, start)]
    shape_note = ''
    if shape is not None:
        factor = shape['f'] ** (-1. / 3.)
        shape_note = (N_SHAPE_CROSS.format(f=fmt(shape['f']), factor=fmt(factor), delta=fmt(shape['delta_nm']))
                      if shape['delta_nm'] is not None else N_SHAPE_NONE.format(f=fmt(shape['f']), factor=fmt(factor)))
    if has_rate:
        checks.append(_percolation_check(perc, shape, label, refs, entries, blo, bhi, expected, host_note, shape_note,
                                         tol))
        last = float(np.nanmax(delta)) if np.isfinite(delta).any() else None
        checks.append(_onset_check(onset, rate, t0_s or 0., last, label, refs, blo, bhi, expected))
    else:
        checks.append(_check('percolation', 'not testable', M_NEEDS_RATE, expected, refs, [host_note]))
        checks.append(_check('TS_onset', 'not testable', M_NEEDS_RATE, expected, refs, [N_TS_MEANING]))
    checks += _network_checks(cfg, columns, rows, delta, has_rate, teq_factor)
    checks.append(_film_check(rows, delta, has_rate, film_identifiable))
    counts = {s: sum(1 for c in columns if c['lit_status'] == s) for s in sorted({c['lit_status'] for c in columns})}
    notes = [NOTE_CONSTRUCTION, NOTE_SHAPE]
    if has_rate:
        notes.append(NOTE_RATE.format(rate=fmt(rate), rate_error=fmt(100 * perc['rate_error']),
                                      t0_error=fmt(perc['t0_error_s'])))
        if t0_s is None:
            notes.append(NOTE_T0.format(rate=fmt(rate)))
    notes += [NOTE_D_PEAK, NOTE_STICKING, NOTE_CHAMBER]
    if host_note:
        notes.append(host_note)
    if host:
        notes.append(NOTE_HOST_PEAKS.format(count=len(host), D=fmt(D_host if D_host else float(np.nanmedian(D[host])))))
    if metal == 'Cu':
        notes.append(NOTE_CU)
    used = set(refs) | {r for c in checks for r in c['references']} | {'Sch15', 'Bulut24', 'GenTh23', 'Hod17', 'Jam19',
                                                                        'Sch17', 'Thran99', 'Nittler12', 'Gra13',
                                                                        'Abr16'}
    curves = [c for c in cfg['D_curves'] if c['metal'] == model]
    used |= {c['ref'] for c in curves} | ({'Schap21'} if model == 'Cu' else set())
    status_word = 'ok' if (enough and has_rate) else ('needs_rate' if not has_rate else 'too_few_windows')
    fit_t = [float(t[i]) for i in fit_idx]
    island_windows = sum(1 for c in columns if c['R_lit_nm'] is not None)
    return dict(
        status=status_word, island_windows=island_windows,
        default_ready=status_word == 'ok' and island_windows >= MIN_WINDOWS,
        message=NEEDS_RATE if not has_rate else ('' if enough else TOO_FEW.format(minimum=MIN_WINDOWS)),
        method=METHOD, metal=metal, polymer=polymer or None, host_class=label, host_note=host_note,
        rate_nm_s=rate if has_rate else None, t0_s=t0_s if t0_s is not None else (0. if has_rate else None),
        t0_assumed=has_rate and t0_s is None, windows=dict(used=len(fit_idx), **counts),
        peak_phase_s=[fit_t[0], fit_t[-1]] if fit_t else None, host_peak_D_nm=D_host,
        percolation=perc if has_rate else None,
        band=dict(host_class=label, classes=keys, low_nm=blo, high_nm=bhi, rate_scaled_nm=[s_lo, s_hi] if s_lo else None,
                  tolerance=tol, chi_range=cfg['rate_dependence']['chi_range'], entries=entries,
                  caveat=cfg['rate_dependence']['caveat']),
        reference_curves=curves,
        two_R_over_D_bands=[b for b in cfg['two_R_over_D'] if b['metal'] in (None, model)],
        two_R_over_D_test=cfg['two_R_over_D_test_from_1nm'], coverage_bands=cfg['coverage'],
        coverage_at_percolation=cfg['coverage_at_percolation'], coverage_test=cfg['coverage_test_from_1nm'],
        eta_warning=cfg['mass_balance']['warning_threshold'], shape=cfg['shape'].get(model),
        checks=checks, notes=notes,
        references={k: cfg['references'][k] for k in sorted(used) if k in cfg['references']},
        config=cfg['version'])


__all__ = ['HEX', 'LIT_COLUMNS', 'TEMPLATES', 'band', 'crossing', 'host_class', 'isotonic_median', 'jumps',
           'literature_table', 'load_config', 'percolation_estimate', 'ratio', 'reference_envelope']
