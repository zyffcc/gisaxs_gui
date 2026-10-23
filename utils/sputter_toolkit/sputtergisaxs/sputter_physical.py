"""Physical path of a sputter series and thickness substitutes (structure_physical.csv).

structure.csv applies the stage rule to every window on its own, so the stage,
the network (island or TS film) and the R/D source can switch back and forth
from one window to the next. This module reads that table together with the
network outputs of both experts (binned/frames.csv, binned/frames_alt.csv) and
writes one physical reading of the whole series:

- stage path: the rule stages on the ordinal scale pre_nucleation(0) <
  islands(1) < coalescence(2) < film(3), fitted non-decreasing by
  pool-adjacent-violators with block medians (an L1 fit: path values stay on
  the scale, and the TS onset minimises the number of resolved windows that
  disagree with it). Windows with an unresolved stage cast no vote; they take
  the path value of the previous resolved window (of the next one at the start)
  and are marked stage_imputed. Windows with almost no counts (shutter closed,
  beam off) get no stage and no values. Windows where the beam is probably
  partly or fully off the sample (beam_off_sample, below) cast no vote either:
  they carry the path value of their neighbours and are marked stage_imputed.
  Path stage 'film' is the rule's coalescence-or-film candidate: a connected
  film is not proven.
- module path: TS_layer for the path stages the toolkit routes to the TS
  network (the LATE set), else polymer_embed. Every value comes from the network
  of the path module, also where the per-window rule chose the other one.
- R and D: the network estimate in every window whose path stage is islands
  (grade P; no switching between refined and network values); none in
  pre-nucleation windows. R_trend_nm / D_trend_nm: a local log-linear fit
  (Theil-Sen) of ln R / ln D against time over the island windows within a
  centred span of max(modulation period, 2.5 s), evaluated at the window, so it
  follows growth also when the span is long and is not biased at the ends.
- intensity modulation period: autocorrelation of the detrended log intensity;
  the shortest maximum whose double lag is also a maximum, so that two unequal
  dips per repeat give the dip spacing in every scan. The sample stage is
  translated back and forth during the deposition (against beam damage), and
  near the ends of its travel the footprint can reach the sample edge or leave
  the sample: the intensity dips once or twice per cycle. The summary keeps the
  key rotation_period_s (and rotation_detection) for older readers;
  modulation_period_s and modulation_detection are the neutral aliases.
- beam off the sample (sputter_offsample): a dip whose intensity drops below
  half of the level within one modulation period on both sides (without a
  modulation: below a fifth within 10 s) is flagged beam_off_sample with a
  beam_off_reason, together with the windows next to it that stay below 80 %
  of their level (the whole dip, not only its bottom). They keep their values
  in the table but cast no stage vote and are left out of the R/D trends, the
  reversal counts (a flagged window breaks the R/D zig-zag sequences: its
  neighbours are not joined), the mass continuity and the literature method's
  trend of D and its checks of the network.
- thickness substitutes: the island-volume equivalent thickness t_eq of the
  visible particles (flat-based hemispheres h=R on a hexagonal lattice) and,
  when a deposition rate is given, the nominal thickness
  rate*clip(t - t0, 0, t_end - t0): the shutter is closed by hand and a scan can
  go on after the deposition, so with a given end t_end the nominal thickness
  stays at rate*(t_end - t0) after it (without t_end the deposition lasts to the
  end of the scan). The TS film thickness is not identifiable (in simulation the
  posterior equals the prior), so film windows say so instead of giving a number.
- the literature method (sputter_literature): R from the nominal thickness and
  D = 2*pi/q* by mass balance, its percolation (with an interval over the noise
  and the errors of the rate and t0), and checks against published growth
  (summary key 'literature', columns LIT_COLUMNS).

Every note is made from the templates below (NOTE_TEMPLATES, plain {field}),
so a program that shows them can translate each one.

Example:
    from sputtergisaxs.sputter_physical import write_physical
    write_physical("OUT", rate_nm_s=0.05, t0_s=1.0, t_end_s=120.0)
"""
from __future__ import annotations
from pathlib import Path
import csv, json, math
import numpy as np

from .sputter_sequence import LATE
from .sputter_literature import LIT_COLUMNS, literature_table
from .sputter_offsample import TEMPLATES as OFF_TEMPLATES, beam_off_sample

STAGE_ORDINAL = {'pre_nucleation_candidate': 0, 'separate_islands_candidate': 1, 'coalescence_candidate': 2,
                 'coalescence_or_film_candidate': 3, 'continuous_film_candidate': 3}
PATH_STAGES = ('pre_nucleation', 'islands', 'coalescence', 'film')
RULE_STAGE_OF = ('pre_nucleation_candidate', 'separate_islands_candidate', 'coalescence_candidate',
                 'coalescence_or_film_candidate')
PEAK = ('bilateral', 'one_side')
ROTATION = dict(start_s=5., step_s=.5, detrend_s=60., min_lag_s=2., max_lag_s=60., min_acf=.3, min_harmonic_acf=.1,
                min_cycles=3., unequal_dips=.05)
MIN_SPAN_S = 2.5
MIN_TREND_POINTS = 4
NO_DATA_COUNTS = 1000.
ZIGZAG_STEP = math.log(1.2)
ONSET_SLACK = 2
HEX_TEQ = (2. / 3.) * math.pi / (math.sqrt(3.) / 2.)

ROTATION_METHOD = ('autocorrelation of ln(intensity_rate) on the uniform (0.5 s) windows after the first 5 s, minus a '
                   'centred 60 s local linear fit; local maxima >= 0.1 at lags 2-60 s; the period is the shortest maximum '
                   'whose double lag is also a maximum, with >= 0.3 at one of the two (>= 0.3 on its own when the '
                   'double lag is beyond 60 s; else the first maximum >= 0.3), with >= 3 full cycles in the series')
ROTATION_MEANING = ('period of the intensity modulation (the spacing of its dips), probably from the back-and-forth '
                    'translation of the sample stage during the deposition (against beam damage; near the ends of its '
                    'travel the footprint can reach the sample edge or leave the sample); the data alone do not prove '
                    'the source. When the pattern repeats exactly only after twice this period (two unequal dips, '
                    'full_repeat_s), one full back-and-forth cycle of the stage may take full_repeat_s')
FILM_STATUS = ("not determined by GISAXS: the TS film thickness and metal fraction are not identifiable "
               "(in simulation the network's posterior equals its prior); see L_TS_nm for the lateral texture length")
T_EQ_NOTE = ('t_eq = (2/3)*pi*R^3*exp(3*sigma^2) / ((sqrt(3)/2)*D^2): island-volume equivalent thickness of the '
             'visible particles, conditional on flat-based hemispheres (h=R) on a hexagonal lattice; not the '
             'deposited amount. Band t_eq/f..t_eq*f with f = exp(sqrt(9*eR^2 + 4*eD^2)), eR and eD = ln(1 + typical '
             'relative error) of the network R and D.')
NOMINAL_NOTE = ('nominal deposited thickness = rate * (t - t0) from the given deposition rate (e.g. QCM); not '
                'measured by GISAXS.')
T0_ASSUMED_NOTE = (' t0 not given: assumed 0 s = the first frame; the start of the deposition is not known from the '
                   'data (each 1 s of error in t0 shifts the nominal thickness by {rate} nm).')
NO_RATE_NOTE = 'no deposition rate given: nominal thickness not computed'
T_END_NOTE = ('deposition ended at t_end = {t_end} s (given): after it the nominal thickness stays at '
              'rate * (t_end - t0) = {thickness} nm.')
T_END_ASSUMED_NOTE = ('t_end not given: the deposition is assumed to last to the end of the scan (the shutter is '
                      'closed by hand, and a scan can go on after it).')
ONSET_MEANING = ('first window of the TS-network part of the path (rule stage coalescence or coalescence-or-film '
                 'candidate); a connected film is not proven')
ALT_MISSING = ('the network of the path module was not run on this window in this folder '
               '(binned/frames_alt.csv missing; run analyze again on this folder to add it)')
FILM_REASON = 'film-type (TS layer) window on the physical path: island R/D not defined (see L_TS_nm)'
PRE_REASON = 'pre-nucleation on the physical path: no island estimate'
UNSTAGED_REASON = 'no window of the series has a resolved stage: no island estimate'
NO_DATA_REASON = f'fewer than {NO_DATA_COUNTS:g} counts in the window (shutter closed or beam off): no stage, no values'
NO_PEAK_REASON = 'no resolved peak: D not reported'
NO_ESTIMATE = 'no estimate'
D_OUTSIDE = 'outside simulated 90% range of D/(2pi/q*)'
NOTE_MONOTONE = ('Model assumption: during deposition the TS onset is monotone (no return from film to islands); the stage '
                 'path is the non-decreasing L1 fit of the rule stages, and unresolved windows cast no vote.')
NOTE_FILM_STAGE = ("stage_path 'film' is the rule's coalescence-or-film candidate (a connected film is not proven); film=true "
                   'marks every window on the TS network (coalescence or film).')
NOTE_TREND = ('R_trend_nm / D_trend_nm: local log-linear (Theil-Sen) fit of ln R / ln D over the island windows in a '
              'centred span of {span} s (max of the modulation period and 2.5 s), evaluated at each window.')
NOTE_GRADE_P = ('R_nm / D_nm are network estimates in every window whose path stage is islands (grade P); refined values '
                'stay in refined_R_nm / refined_D_nm. Pre-nucleation windows have no island values.')
NOTE_TS_FILM = ('TS film windows: no thickness or metal fraction is reported (not identifiable); L_TS_nm is the lateral '
                'texture length, not an island spacing.')
NOTE_CU = 'Cu approximated with the Ag model, for reference only (errors not validated).'
NOTE_IMPUTED = ('{count} windows with an unresolved rule stage, or without a vote because the beam may be off the '
                'sample, carry the stage of the neighbouring resolved window (stage_imputed=true; '
                'stage_path_counts_imputed).')
NOTE_NO_DATA = '{count} windows with fewer than {counts} counts have no stage and no values.'
NOTE_NOT_ESTABLISHED = ('TS onset not established: no TS window at all disagrees with {no_ts} resolved windows, within {slack} '
                        'of the {best} at the chosen onset ({onset} s); the series may stay in the island phase.')
NOTE_LATE_ONSET = ('No TS onset on the path, but onsets from {start} s disagree with at most {most} resolved windows: a late '
                   'TS onset is not excluded.')
NOTE_NO_FILM = ('The path reaches coalescence but never the film stage: film_onset_s is the onset of TS-network '
                '(coalescence) windows, not of a continuous film.')
NOTE_DIPS = ('The intensity dips every {period} s with unequal depths; the pattern repeats exactly after {repeat} s, which '
             'may be one full back-and-forth cycle of the sample stage (one dip at each end of its travel).')
NOTE_OFF_SAMPLE = ('{count} windows in intensity dips: {core} below {limit}% of the level within one modulation period on '
                   'both sides and their neighbours in the same dip below {release}%. The beam is probably partly or '
                   'fully off the sample while the stage is translated (beam_off_sample=true). They keep their values '
                   'but cast no stage vote and are left out of the R/D trends, the reversal counts (the R/D zig-zags '
                   'are counted between them), the mass continuity and the literature method.')
NOTE_OFF_EXTREME = ('{count} windows in intensity dips without a periodic modulation: {core} below {limit}% of the level '
                    'within {span} s on both sides and their neighbours in the same dip below {release}%. The beam was '
                    'probably off the sample or interrupted (beam_off_sample=true). They keep their values but cast no '
                    'stage vote and are left out of the R/D trends, the reversal counts (the R/D zig-zags are counted '
                    'between them), the mass continuity and the literature method.')
NOTE_SPAN = ('The smoothing span ({span} s) covers the whole island phase ({start}-{end} s): R_trend_nm / D_trend_nm are '
             'one log-linear fit over it; the per-window R_nm / D_nm (grade P) show more.')
NOTE_TEMPLATES = (T_EQ_NOTE, NOMINAL_NOTE, NOMINAL_NOTE + T0_ASSUMED_NOTE, NO_RATE_NOTE, T_END_NOTE,
                  T_END_ASSUMED_NOTE, FILM_STATUS, ONSET_MEANING, ALT_MISSING, FILM_REASON, PRE_REASON, UNSTAGED_REASON,
                  NO_DATA_REASON, NO_PEAK_REASON, NO_ESTIMATE, D_OUTSIDE, NOTE_MONOTONE, NOTE_FILM_STAGE, NOTE_TREND, NOTE_GRADE_P, NOTE_TS_FILM, NOTE_CU, NOTE_IMPUTED,
                  NOTE_NO_DATA, NOTE_NOT_ESTABLISHED, NOTE_LATE_ONSET, NOTE_NO_FILM, NOTE_DIPS, NOTE_SPAN,
                  NOTE_OFF_SAMPLE, NOTE_OFF_EXTREME, *OFF_TEMPLATES)
"""Every note and reason this module writes (filled with str.format; the literature texts are in
sputter_literature.TEMPLATES)."""
NEW_COLUMNS = ('stage_path', 'module_path', 'film', 'path_changed', 'R_trend_nm', 'D_trend_nm', 't_eq_nm', 't_eq_lo_nm',
               't_eq_hi_nm', 'nominal_thickness_nm', 'film_thickness_status', 'stage_imputed', 'beam_off_sample',
               'beam_off_reason') + LIT_COLUMNS


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- stage path --
def _lower_median(counts, n):
    k, acc = (n + 1) // 2, 0
    for level, c in enumerate(counts):
        acc += c
        if acc >= k:
            return level
    return len(counts) - 1


def isotonic_levels(values, levels=4):
    """Non-decreasing L1 fit of integer levels by pool-adjacent-violators (block lower medians)."""
    blocks = []
    for v in values:
        counts = [0] * levels; counts[v] = 1
        blocks.append([counts, 1, v])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
            c2, n2, _ = blocks.pop(); c1, n1, _ = blocks.pop()
            counts = [a + b for a, b in zip(c1, c2)]
            blocks.append([counts, n1 + n2, _lower_median(counts, n1 + n2)])
    return [level for _, n, level in blocks for _ in range(n)]


def stage_path(stages):
    """Ordinal path value per window (None everywhere when no window has a resolved stage)."""
    ordinals = [STAGE_ORDINAL.get(s) for s in stages]
    resolved = [i for i, o in enumerate(ordinals) if o is not None]
    if not resolved:
        return [None] * len(stages)
    fit = dict(zip(resolved, isotonic_levels([ordinals[i] for i in resolved])))
    out, last = [], fit[resolved[0]]
    for i in range(len(stages)):
        last = fit.get(i, last)
        out.append(last)
    return out


def is_late(ordinal):
    return ordinal is not None and RULE_STAGE_OF[ordinal] in LATE


def onset_disagreements(stages):
    """Resolved windows disagreeing with a single islands->TS step placed before window k (k = 0..n; n = no TS)."""
    late = [None if STAGE_ORDINAL.get(s) is None else is_late(STAGE_ORDINAL[s]) for s in stages]
    before_film = np.cumsum([0] + [1 if v is True else 0 for v in late])
    after_island = np.cumsum([0] + [1 if v is False else 0 for v in reversed(late)])[::-1]
    return before_film + after_island


# ------------------------------------------------ intensity modulation period --
def local_linear_trend(y, half):
    """Least-squares line through the finite values of y within +-half samples, evaluated at each sample.

    Unlike a rolling median it is not biased at the ends of a rising or falling series, and it scales a periodic
    component instead of distorting it. NaN where fewer than 2 finite values lie in the window.
    """
    i = np.arange(len(y), dtype=float); ok = np.isfinite(y)
    sums = [np.concatenate([[0.], np.cumsum(np.where(ok, v, 0.))]) for v in (ok * 1., i, i * i, y, i * y)]
    lo = np.clip(np.arange(len(y)) - half, 0, len(y)); hi = np.clip(np.arange(len(y)) + half + 1, 0, len(y))
    s0, s1, s2, sy, sxy = (c[hi] - c[lo] for c in sums)
    det = s0 * s2 - s1 * s1
    with np.errstate(invalid='ignore', divide='ignore'):
        slope = np.where(det > 0, (s0 * sxy - s1 * sy) / det, 0.)
        level = (sy - slope * s1) / s0
    return np.where(s0 >= 2, level + slope * i, np.nan)


def _acf_maxima(acf, kmin, kmax, floor):
    """(lag index, refined lag in windows, value) of the local ACF maxima >= floor at lags kmin..kmax."""
    out = []
    for k in range(max(kmin, 1), min(kmax, len(acf) - 2) + 1):
        a, b, c = acf[k - 1], acf[k], acf[k + 1]
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)) or not (b >= a and b > c) or b < floor:
            continue
        curve = a - 2 * b + c
        out.append((k, k + (0.5 * (a - c) / curve if curve < 0 else 0.), float(b)))
    return out


def _psd_peak(z, valid, step, span, cfg):
    """Cross-check: the strongest periodogram line of the detrended series in the searched period band."""
    x = np.where(valid, z, 0.)
    power = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x), step)
    periods = np.divide(1., freq, out=np.full_like(freq, np.inf), where=freq > 0)
    band = (freq > 0) & (periods >= cfg['min_lag_s']) & (periods <= cfg['max_lag_s']) & (span / periods >= cfg['min_cycles'])
    total = float(power[freq > 0].sum())
    if not band.any() or total <= 0:
        return dict(psd_peak_period_s=None, psd_power_fraction=None)
    i = np.flatnonzero(band)[int(np.argmax(power[band]))]
    return dict(psd_peak_period_s=round(float(periods[i]), 3), psd_power_fraction=round(float(power[i]) / total, 4))


def rotation_period(times, rates, nominal_seconds, cfg=ROTATION):
    """(modulation period_s or None, detection record) from the intensity modulation of the 0.5 s windows."""
    late = [round(n, 6) for t, n in zip(times, nominal_seconds) if t is not None and t >= cfg['start_s'] and n]
    step = max(set(late), key=late.count) if late else cfg['step_s']
    pts = [(t, r) for t, r, n in zip(times, rates, nominal_seconds)
           if t is not None and r is not None and r > 0 and t >= cfg['start_s'] and n is not None and abs(n - step) < 1e-6]
    record = dict(method=ROTATION_METHOD, meaning=ROTATION_MEANING, acf_peak=None, cycles=None, window_s=step,
                  full_repeat_s=None)
    if len(pts) < 20:
        record['reason'] = 'fewer than 20 uniform windows after the first 5 s'
        return None, record
    t_first = pts[0][0]
    idx = np.array([int(round((t - t_first) / step)) for t, _ in pts])
    y = np.full(int(idx.max()) + 1, np.nan); y[idx] = np.log([r for _, r in pts])
    z = y - local_linear_trend(y, int(round(cfg['detrend_s'] / 2 / step))); valid = np.isfinite(z)
    z = z - np.mean(z[valid]); var = float(np.mean(z[valid] ** 2))
    kmin, kmax = int(round(cfg['min_lag_s'] / step)), int(round(cfg['max_lag_s'] / step))
    acf = np.full(kmax + 2, np.nan); acf[0] = 1.
    for k in range(1, min(kmax + 2, len(z))):
        pair = valid[:-k] & valid[k:]
        if pair.sum() >= 10 and var > 0:
            acf[k] = float(np.mean(z[:-k][pair] * z[k:][pair]) / var)
    span = (len(y) - 1) * step
    maxima = _acf_maxima(acf, kmin, kmax, cfg['min_harmonic_acf'])
    record['acf_local_maxima'] = [[round(lag * step, 3), round(value, 3)] for _, lag, value in maxima]
    record.update(_psd_peak(z, valid, step, span, cfg))

    def accept(lag, value, rule, double=None):
        period = lag * step
        record.update(acf_peak=round(value, 4), cycles=round(span / period, 2), lag_windows=int(round(lag)), rule=rule)
        if double is not None:
            record['double_lag_acf'] = round(double[2], 4)
            if double[2] >= value + cfg['unequal_dips']:
                record['full_repeat_s'] = round(double[1] * step, 3)
        return round(float(period), 3), record

    for _, lag, value in maxima:  # shortest repeating spacing first
        if span / (lag * step) < cfg['min_cycles']:
            break
        if 2 * lag <= kmax:
            double = next((m for m in maxima if abs(m[1] - 2 * lag) <= max(1.5, 0.1 * lag)), None)
            if double is not None and max(value, double[2]) >= cfg['min_acf']:
                return accept(lag, value, 'shortest maximum repeated at the double lag', double)
        elif value >= cfg['min_acf']:
            return accept(lag, value, 'maximum >= 0.3 (double lag beyond the searched lags)')
    for _, lag, value in maxima:  # no repeated maximum: the first strong one
        if span / (lag * step) < cfg['min_cycles']:
            break
        if value >= cfg['min_acf']:
            return accept(lag, value, 'first maximum >= 0.3 (no maximum at its double lag)')
    record['reason'] = 'no autocorrelation maximum >= 0.3 (or repeated maximum) with >= 3 full cycles at lags 2-60 s'
    return None, record


modulation_period = rotation_period
"""The neutral name: the modulation comes from the back-and-forth stage translation (rotation_period stays for
older callers)."""


# ------------------------------------------------------------------ smoothing --
def rolling_log_trend(times, values, span, min_points=MIN_TREND_POINTS, at=None):
    """Local log-linear trend: Theil-Sen fit of ln(value) vs time over windows within +-span/2, at each window.

    Windows without a value stay None, unless at[i] is true: the trend is then evaluated at their time from the
    values around them (a window left out of the fit still lies on the trend line; None when no value is within
    the span). With fewer than min_points values in the span the local median is used.
    """
    out = [None] * len(values)
    pts = sorted((t, i, math.log(v)) for i, (t, v) in enumerate(zip(times, values))
                 if t is not None and v is not None and v > 0)
    if not pts:
        return out
    ts = np.array([p[0] for p in pts]); ls = np.array([p[2] for p in pts])
    have = {i for _, i, _ in pts}
    extra = [(times[i], i) for i in range(len(values))
             if at is not None and at[i] and i not in have and times[i] is not None]
    for t0, i in [(t, i) for t, i, _ in pts] + extra:
        a = int(np.searchsorted(ts, t0 - span / 2, side='left')); b = int(np.searchsorted(ts, t0 + span / 2, side='right'))
        if b <= a:
            continue
        tt, yy = ts[a:b] - t0, ls[a:b]
        if b - a >= min_points and tt[-1] > tt[0]:
            dt = np.subtract.outer(tt, tt); dy = np.subtract.outer(yy, yy)
            pair = dt > 0  # every pair of windows once
            slope = float(np.median(dy[pair] / dt[pair])) if pair.any() else 0.
            out[i] = float(math.exp(np.median(yy - slope * tt)))
        else:
            out[i] = float(math.exp(np.median(yy)))
    return out


def t_eq(R, D, sigma=None):
    """Island-volume equivalent thickness (nm) of hemispheres (h=R) on a hexagonal lattice of spacing D."""
    if R is None or D is None or R <= 0 or D <= 0:
        return None
    return HEX_TEQ * R ** 3 * math.exp(3 * (sigma or 0.) ** 2) / D ** 2


# ----------------------------------------------------------------- reversals --
def zigzags(values, breaks=None):
    """Steps |d ln x| > ln 1.2 reversed to within 20 % of the earlier value within the next 1-3 values.

    Windows without a value are skipped (their neighbours are joined). A window with breaks[i] true ends the sequence
    instead: the counts of the stretches between such windows are added, so leaving a window out never joins values
    from both sides of it (growth across the gap, or the flanks of a dip, would make new steps).
    """
    if breaks is not None:
        total, stretch = 0, []
        for value, stop in zip(values, breaks):
            if stop:
                total += zigzags(stretch); stretch = []
            else:
                stretch.append(value)
        return total + zigzags(stretch)
    v = np.log([x for x in values if x is not None and x > 0])
    count = 0
    d = np.diff(v)
    for i in np.flatnonzero(np.abs(d) > ZIGZAG_STEP):
        for k in range(1, 4):
            j = i + 1 + k
            if j >= len(v):
                break
            if np.sign(v[j] - v[i + 1]) == -np.sign(d[i]) and abs(v[j] - v[i]) < ZIGZAG_STEP:
                count += 1
                break
    return count


def _runs(seq):
    out = []
    for x in seq:
        if x is not None and x != '' and (not out or out[-1] != x):
            out.append(x)
    return out


def stage_reversals(ordinals):
    """A -> B -> A excursions among the resolved (non-None) stage values."""
    runs = _runs(ordinals)
    return sum(1 for i in range(1, len(runs) - 1) if runs[i - 1] == runs[i + 1])


def module_reverts(modules):
    runs = _runs(modules)
    return sum(1 for a, b in zip(runs, runs[1:]) if a == 'TS_layer' and b == 'polymer_embed')


# ------------------------------------------------------------- physical rows --
def _network(fr):
    """Mode-1 network values of one frames.csv / frames_alt.csv row."""
    return dict(R=_num(fr.get('mode1_R_number_mean_nm')), D=_num(fr.get('mode1_D_nm')),
                sigma=_num(fr.get('mode1_radius_sigma_log')), burial=_num(fr.get('mode1_burial_base_depth_nm')),
                thickness=_num(fr.get('mode1_polymer_thickness_nm')), L=_num(fr.get('mode1_spectral_domain_nm')),
                probability=_num(fr.get('mode1_probability')), film_thickness=_num(fr.get('mode1_film_thickness_nm')),
                metal_fraction=_num(fr.get('mode1_metal_fraction')))


def _kind(no_data, module, ordinal):
    if no_data:
        return 'no_data'
    if module == 'TS_layer':
        return 'film'
    return {0: 'pre_nucleation', 1: 'islands'}.get(ordinal, 'unstaged')


def _size_values(kind, net, peak, errR, errD):
    """R_* and D_* columns of one window on the path."""
    values = {}
    for name, err in (('R', errR), ('D', errD)):
        value = None
        if kind != 'islands':
            reason = dict(no_data=NO_DATA_REASON, film=FILM_REASON, pre_nucleation=PRE_REASON).get(kind, UNSTAGED_REASON)
        elif net is None:
            reason = ALT_MISSING
        elif name == 'D' and not peak:
            reason = NO_PEAK_REASON
        else:
            value = net[name]; reason = '' if value is not None else NO_ESTIMATE
        e = err['typical_rel_scatter']; has = value is not None
        values.update({f'{name}_nm': value, f'{name}_source': 'network' if has else None,
                       f'{name}_grade': 'P' if has else None, f'{name}_reason': reason,
                       f'{name}_typical_rel_error': e if has else None,
                       f'{name}_lo_nm': value / (1 + e) if has else None, f'{name}_hi_nm': value * (1 + e) if has else None,
                       f'{name}_P90_rel_error': err['P90_rel_error'] if has else None,
                       f'{name}_order_accuracy': err['order_accuracy_1p2x'] if has else None})
    return values


def _onset(module_path, stages, t, n):
    """TS onset on the path and every onset (or 'no TS') within ONSET_SLACK disagreeing windows of the best."""
    onset = next((i for i, m in enumerate(module_path) if m == 'TS_layer'), None)
    counts = onset_disagreements(stages)
    best = int(counts.min()) if len(counts) else 0
    near = [k for k in range(n) if counts[k] <= best + ONSET_SLACK and t[k] is not None]
    no_ts = int(counts[n])
    return onset, dict(
        film_onset_s=t[onset] if onset is not None else None,
        film_onset_interval_s=[t[near[0]], t[near[-1]]] if near else None,
        film_onset_meaning=ONSET_MEANING,
        film_onset_disagreeing_windows=int(counts[onset]) if onset is not None else no_ts,
        film_onset_min_disagreeing_windows=best,
        film_onset_no_TS_disagreeing_windows=no_ts,
        film_onset_no_TS_within_slack=no_ts <= best + ONSET_SLACK,
        film_onset_slack_windows=ONSET_SLACK)


def _onset_notes(extras, path_counts):
    notes, onset, interval = [], extras['film_onset_s'], extras['film_onset_interval_s']
    if onset is not None and extras['film_onset_no_TS_within_slack']:
        notes.append(NOTE_NOT_ESTABLISHED.format(no_ts=extras['film_onset_no_TS_disagreeing_windows'], slack=ONSET_SLACK,
                                                 best=extras['film_onset_min_disagreeing_windows'], onset=f'{onset:.1f}'))
    if onset is None and interval is not None:
        notes.append(NOTE_LATE_ONSET.format(start=f'{interval[0]:.1f}',
                                            most=extras['film_onset_min_disagreeing_windows'] + ONSET_SLACK))
    if onset is not None and not path_counts.get('film'):
        notes.append(NOTE_NO_FILM)
    return notes


def polymer_of(frames):
    """The polymer named in the analysis (system_note 'metal/polymer' of binned/frames.csv), or None."""
    note = next((fr.get('system_note') for fr in frames if fr.get('system_note')), '') or ''
    return (note.split('/', 1)[1].strip() or None) if '/' in note else None


def physical_table(structure, frames, alternate, *, rate_nm_s=None, t0_s=None, reliability=None, polymer_type=None,
                   rate_error=None, t0_error_s=None, t_end_s=None):
    """(rows of new column values per structure row, summary extras) for one series. rate_error (relative) and
    t0_error_s: errors of the rate and t0 in the literature method's percolation interval (default: the config's).
    t_end_s: the end of the deposition (s), after which the nominal thickness stays constant (None: the end of the
    scan)."""
    if reliability is None:
        from .sputter_struct import reliability as load
        reliability = load()
    rel = reliability['parameters']
    n = len(structure)
    frame_by_key = {fr['frame']: fr for fr in frames}
    alt_by_key = {fr['frame']: fr for fr in alternate} if alternate is not None else None
    times = [_num(r.get('time_s')) for r in structure]
    order = sorted(range(n), key=lambda i: (times[i] is None, times[i] or 0., i))
    srt = [structure[i] for i in order]
    t = [times[i] for i in order]
    frame_rows = [frame_by_key.get(r['frame'], {}) for r in srt]
    metal = (srt[0].get('metal') if srt else None) or 'Ag'
    model_metal = 'Ag' if metal == 'Cu' else metal
    no_data = [(_num(f.get('selected_counts')) is not None and _num(f.get('selected_counts')) < NO_DATA_COUNTS)
               for f in frame_rows]
    rule_stages = [r.get('stage') or '' for r in srt]
    stages = ['' if nd else s for s, nd in zip(rule_stages, no_data)]
    # The intensity modulation (the back-and-forth stage translation) and the windows where the beam is probably
    # partly or fully off the sample: they keep their values but cast no stage vote.
    rates = [_num(f.get('intensity_rate')) for f in frame_rows]
    period, detection = rotation_period(t, rates, [_num(f.get('nominal_bin_seconds')) for f in frame_rows])
    off, off_reasons, off_record = beam_off_sample(t, rates, period, no_data=no_data,
                                                   yoneda=[_num(f.get('Yoneda_deg')) for f in frame_rows])
    votes = ['' if o else s for s, o in zip(stages, off)]
    path = [None if nd else o for o, nd in zip(stage_path(votes), no_data)]
    imputed = [o is not None and STAGE_ORDINAL.get(s) is None for o, s in zip(path, votes)]  # carried, not voted
    rule_module = [r.get('module') or '' for r in srt]
    module_path = [('' if nd else rule_module[i]) if o is None else ('TS_layer' if is_late(o) else 'polymer_embed')
                   for i, (o, nd) in enumerate(zip(path, no_data))]
    errR = rel['R'][model_metal]['network']; errD = rel['D'][model_metal]['network']
    band = rel.get('D_over_direct', {}).get('band_90', [0.587, 1.75])
    out, kinds, film_net = [], [], []
    for i, r in enumerate(srt):
        fr = frame_rows[i]
        o = path[i]; rule_o = STAGE_ORDINAL.get(stages[i])
        kind = _kind(no_data[i], module_path[i], o); kinds.append(kind)
        same = module_path[i] == (fr.get('module') or rule_module[i])
        source = fr if same else (alt_by_key.get(r['frame']) if alt_by_key is not None else None)
        net = _network(source) if source and kind != 'no_data' else None
        peak = r.get('feature_status') in PEAK
        island_module = module_path[i] == 'polymer_embed'
        row = dict(stage_path=PATH_STAGES[o] if o is not None else '', module_path=module_path[i],
                   film=module_path[i] == 'TS_layer', stage_imputed=imputed[i], beam_off_sample=off[i],
                   beam_off_reason=off_reasons[i],
                   path_changed=(not no_data[i]) and (module_path[i] != rule_module[i]
                                                      or (rule_o is not None and o is not None and rule_o != o)))
        values = _size_values(kind, net, peak, errR, errD)
        D_corr = _num(r.get('D_corr_nm'))
        if values['D_nm'] is not None and D_corr:
            ratio = values['D_nm'] / D_corr
            values['D_over_direct'] = ratio
            values['D_direct_check'] = 'consistent' if band[0] <= ratio <= band[1] else D_OUTSIDE
        else:
            values['D_over_direct'] = values['D_direct_check'] = None
        shape = net if kind == 'islands' else None
        values.update(network_R_nm=net['R'] if net and island_module else None,
                      network_D_nm=net['D'] if net and island_module and peak else None,
                      network_mode1_probability=net['probability'] if net else None,
                      L_TS_nm=net['L'] if net and kind == 'film' else None,
                      sigma_log=shape['sigma'] if shape else None, spacing_disorder=None,
                      burial_nm_conditional=shape['burial'] if shape else None,
                      thickness_nm_conditional=shape['thickness'] if shape else None,
                      shape_source='network' if values['R_nm'] is not None else None,
                      TS_metal_fraction_conditional=None, TS_film_thickness_nm_conditional=None)
        row.update(values)
        row['film_thickness_status'] = FILM_STATUS if kind == 'film' else ''
        out.append(row)
        film_net.append((net['film_thickness'], net['metal_fraction']) if net and kind == 'film' else (None, None))
    # Smoothed sizes over the modulation period, without the windows off the sample (the trend is evaluated there too).
    span = max(period or 0., MIN_SPAN_S)
    for name in ('R', 'D'):
        trend = rolling_log_trend(t, [row[f'{name}_nm'] if k == 'islands' and not o else None
                                      for row, k, o in zip(out, kinds, off)], span,
                                  at=[k == 'islands' and o and row[f'{name}_nm'] is not None
                                      for row, k, o in zip(out, kinds, off)])
        for row, value in zip(out, trend):
            row[f'{name}_trend_nm'] = value
    # Thickness substitutes.
    eR, eD = math.log1p(errR['typical_rel_scatter']), math.log1p(errD['typical_rel_scatter'])
    factor = math.exp(math.sqrt(9 * eR ** 2 + 4 * eD ** 2))
    t0 = t0_s if t0_s is not None else 0.
    deposited = t_end_s - t0 if t_end_s is not None else math.inf  # seconds of deposition: rate*clip(t - t0, 0, ...)
    for row, ti, k in zip(out, t, kinds):
        value = t_eq(row['R_nm'], row['D_nm'], row['sigma_log']) if k == 'islands' else None
        row.update(t_eq_nm=value, t_eq_lo_nm=value / factor if value is not None else None,
                   t_eq_hi_nm=value * factor if value is not None else None)
        row['nominal_thickness_nm'] = (rate_nm_s * min(max(0., ti - t0), deposited)
                                       if rate_nm_s is not None and ti is not None else None)
    # TS onset and its interval.
    onset, onset_extras = _onset(module_path, votes, t, n)
    # The literature method and the comparison with published growth.
    lit_rows = [dict(D_corr_nm=_num(r.get('D_corr_nm')), feature_status=r.get('feature_status'), no_data=nd, kind=k,
                     R_nm=row['R_nm'], D_nm=row['D_nm'], t_eq_nm=row['t_eq_nm'], module_path=row['module_path'],
                     film_thickness=film[0], metal_fraction=film[1], beam_off=o)
                for r, nd, k, row, film, o in zip(srt, no_data, kinds, out, film_net, off)]
    columns, literature = literature_table(t, lit_rows, metal=metal, polymer=polymer_type or polymer_of(frames),
                                           rate_nm_s=rate_nm_s, t0_s=t0_s, onset=onset_extras, period=period,
                                           teq_factor=factor, rate_error=rate_error, t0_error_s=t0_error_s,
                                           t_end_s=t_end_s)
    for row, values in zip(out, columns):
        row.update(values)
    # Mass continuity between islands and the nominal thickness.
    continuity = None
    if rate_nm_s is not None:
        stop = onset if onset is not None else n
        idx = [i for i in range(stop) if out[i]['t_eq_nm'] is not None and not off[i]][-5:]
        if idx:
            teq = float(np.median([out[i]['t_eq_nm'] for i in idx]))
            nominal = float(np.median([out[i]['nominal_thickness_nm'] for i in idx]))
            continuity = dict(time_s=float(np.median([t[i] for i in idx])), t_eq_nm=teq, nominal_nm=nominal,
                              ratio=teq / nominal if nominal > 0 else None, windows=len(idx),
                              scope=('median over the last 5 island windows with t_eq before the TS onset' if onset is not None
                                     else 'median over the last 5 island windows with t_eq (no TS onset on the path)')
                              + '; windows where the beam may be off the sample are left out')
    # Back-and-forth counts over the same windows before and after: the windows off the sample are left out of all.
    # The stages and modules on either side of such a window are joined (a category has no level a dip could bias);
    # the R/D sequences break there (zigzags(..., off): values from both sides of a dip are not joined).
    def kept(values):
        return [None if o else v for v, o in zip(values, off)]

    before = dict(stage=stage_reversals(kept([STAGE_ORDINAL.get(s) for s in rule_stages])),
                  module_revert=module_reverts(kept(rule_module)),
                  R_zigzag=zigzags([_num(r.get('R_nm')) for r in srt], off),
                  D_zigzag=zigzags([_num(r.get('D_nm')) for r in srt], off))
    after = dict(stage=stage_reversals(kept(path)), module_revert=module_reverts(kept(module_path)),
                 R_zigzag=zigzags([row['R_nm'] for row in out], off), D_zigzag=zigzags([row['D_nm'] for row in out], off))
    after_trend = dict(R_zigzag=zigzags([row['R_trend_nm'] for row in out], off),
                       D_zigzag=zigzags([row['D_trend_nm'] for row in out], off))
    path_counts = {name: sum(1 for o in path if o == j) for j, name in enumerate(PATH_STAGES)}
    imputed_counts = {name: sum(1 for o, m in zip(path, imputed) if m and o == j) for j, name in enumerate(PATH_STAGES)}
    extras = dict(onset_extras,
        rotation_period_s=period, rotation_detection=detection, smoothing_span_s=span,
        modulation_period_s=period, modulation_detection=detection,
        beam_off_sample=off_record, beam_off_sample_windows=sum(off),
        stage_path_counts=path_counts, stage_path_counts_imputed=imputed_counts,
        stage_imputed_windows=sum(imputed), no_data_windows=sum(no_data),
        path_changed_windows=sum(bool(row['path_changed']) for row in out),
        alternate_network=('binned/frames_alt.csv' if alternate is not None else ALT_MISSING),
        reversals=dict(before=before, after=after, after_trend=after_trend,
                       definition=('before: per-window table (structure.csv: rule stage and module, R_nm/D_nm); after: '
                                   'physical path (stage_path, module_path) and its per-window R_nm/D_nm (grade P); '
                                   'after_trend: the smoothed R_trend_nm/D_trend_nm. stage: A->B->A excursions among '
                                   'resolved stages; module_revert: TS_layer -> polymer_embed changes; R/D zig-zag: a '
                                   'step |d ln x| > ln 1.2 reversed to within 20 % within the next 1-3 values. Windows '
                                   'where the beam may be off the sample (beam_off_sample) are left out of every count: '
                                   'the stages and modules on either side of them are joined, the R/D zig-zags are '
                                   'counted within the stretches between them')),
        thickness=dict(rate_nm_s=rate_nm_s, t0_s=t0_s if t0_s is not None else (0. if rate_nm_s is not None else None),
                       t0_assumed=rate_nm_s is not None and t0_s is None,
                       t_eq_note=T_EQ_NOTE, t_eq_band_factor=factor,
                       nominal_note=(NO_RATE_NOTE if rate_nm_s is None else
                                     NOMINAL_NOTE + (T0_ASSUMED_NOTE.format(rate=f'{rate_nm_s:g}') if t0_s is None else '')),
                       t_end_s=t_end_s, t_end_given=t_end_s is not None,
                       t_end_note=('' if rate_nm_s is None else T_END_ASSUMED_NOTE if t_end_s is None else
                                   T_END_NOTE.format(t_end=f'{t_end_s:g}', thickness=f'{rate_nm_s * deposited:.3g}')),
                       film_thickness_note=FILM_STATUS, mass_continuity=continuity),
        notes=[NOTE_MONOTONE, NOTE_FILM_STAGE, NOTE_TREND.format(span=f'{span:g}'), NOTE_GRADE_P, NOTE_TS_FILM]
        + ([NOTE_CU] if metal == 'Cu' else []),
        literature=literature)
    if extras['stage_imputed_windows']:
        extras['notes'].append(NOTE_IMPUTED.format(count=extras['stage_imputed_windows']))
    if extras['no_data_windows']:
        extras['notes'].append(NOTE_NO_DATA.format(count=extras['no_data_windows'], counts=f'{NO_DATA_COUNTS:g}'))
    if off_record['windows']:
        fields = dict(count=off_record['windows'], core=off_record['core_windows'],
                      limit=f"{100 * off_record['limit_fraction']:g}", release=f"{100 * off_record['release_fraction']:g}")
        extras['notes'].append(NOTE_OFF_SAMPLE.format(**fields) if period is not None
                               else NOTE_OFF_EXTREME.format(span=f"{off_record['span_s']:g}", **fields))
    extras['notes'] += _onset_notes(extras, path_counts)
    if detection.get('full_repeat_s'):
        extras['notes'].append(NOTE_DIPS.format(period=f'{period:g}', repeat=f"{detection['full_repeat_s']:g}"))
    island_t = [ti for row, ti, k in zip(out, t, kinds) if k == 'islands' and row['R_nm'] is not None and ti is not None]
    extras['island_phase_s'] = [island_t[0], island_t[-1]] if island_t else None
    if island_t and span >= island_t[-1] - island_t[0]:
        extras['notes'].append(NOTE_SPAN.format(span=f'{span:g}', start=f'{island_t[0]:.1f}', end=f'{island_t[-1]:.1f}'))
    rows = [None] * n
    for k, i in enumerate(order):
        rows[i] = out[k]
    return rows, extras


def _cell(v):
    if v is None:
        return ''
    if isinstance(v, bool):
        return 'true' if v else 'false'
    return v


def write_physical(out, *, binned=None, rate_nm_s=None, t0_s=None, polymer_type=None, rate_error=None,
                   t0_error_s=None, t_end_s=None):
    """structure_physical.csv and structure_physical_summary.json from OUT/structure.csv (no frame reading).

    polymer_type names the host for the literature band (default: the polymer named in binned/frames.csv);
    rate_error (relative) and t0_error_s widen the literature percolation interval (default: the config's);
    t_end_s is the end of the deposition (the nominal thickness and delta stay constant after it; default: the
    end of the scan).
    """
    from .sputter_struct import summary, reliability
    out = Path(out); binned = Path(binned) if binned is not None else out / 'binned'
    if rate_nm_s is not None and not rate_nm_s > 0:
        raise ValueError('deposition rate must be positive (nm/s)')
    if t_end_s is not None and not t_end_s > (t0_s if t0_s is not None else 0.):
        raise ValueError('the end of the deposition t_end must be after its start t0 (s)')
    structure = _read_csv(out / 'structure.csv')
    frames = _read_csv(binned / 'frames.csv')
    alternate = _read_csv(binned / 'frames_alt.csv') if (binned / 'frames_alt.csv').exists() else None
    rel = reliability()
    new, extras = physical_table(structure, frames, alternate, rate_nm_s=rate_nm_s, t0_s=t0_s, reliability=rel,
                                 polymer_type=polymer_type, rate_error=rate_error, t0_error_s=t0_error_s,
                                 t_end_s=t_end_s)
    base = list(structure[0].keys()) if structure else []
    keys = base + [k for k in NEW_COLUMNS if k not in base]
    rows = []
    for r, v in zip(structure, new):
        row = dict(r)
        row.update({k: _cell(x) for k, x in v.items()})
        rows.append({k: row.get(k, '') for k in keys})
    path = out / 'structure_physical.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    metal = structure[0].get('metal') if structure else None
    numeric = [dict(module=v['module_path'], R_grade=v['R_grade'], D_grade=v['D_grade'], R_nm=v['R_nm'], D_nm=v['D_nm'],
                    D_direct_check=v['D_direct_check']) for v in new]
    s = summary(numeric, metal)
    for name in ('R', 'D'):
        s[name]['physical_P'] = sum(v[f'{name}_grade'] == 'P' for v in new)
    s['TS_windows_rule'] = sum(r.get('module') == 'TS_layer' for r in structure)
    s['grades']['P'] = 'network estimate on the physical path (monotone stage path, one network per window, no refined/network switching)'
    s.update(extras)
    (out / 'structure_physical_summary.json').write_text(json.dumps(s, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return path
