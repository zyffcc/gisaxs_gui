"""Windows of a sputter series in which the beam is probably partly or fully off the sample.

During the deposition the sample stage is translated back and forth, so that the beam does not damage one spot.
Near the ends of its travel the grazing-incidence footprint (millimetres long) can reach the sample edge or leave
the sample: the scattered intensity then drops, once or twice per back-and-forth cycle. That is the periodic
intensity modulation sputter_physical.rotation_period finds (the summary keeps the key rotation_period_s for older
readers; modulation_period_s is the neutral alias). A window recorded while the beam is partly off the sample sees
less of the sample and possibly its edge: its values are not representative of the film.

The rule uses the data only (binned/frames.csv: intensity_rate, the counts per second of the analysed detector
region; Yoneda_deg):

1. The on-sample level of a window is the OFF_SAMPLE['quantile'] (90 %) quantile of ln(intensity_rate) over the other
   windows with data within one modulation period before it, and the same after it; the lower of the two sides counts.
   One full period on each side holds the plateau of the neighbouring cycles, so a dip does not hide its own level,
   while a monotone rise (or fall) of the intensity during the growth keeps the lower side at the window's own level
   and is never a dip. A step that stays (an attenuator, a change of the shutter) is not a dip either.
2. The core of a dip: with a detected modulation a window below OFF_SAMPLE['fraction'] (50 %) of that level. More
   than half of the scattering is missing, as when more than half of the footprint has left the sample (the
   scattered intensity scales with the illuminated sample area). Growth changes the intensity far more slowly, and
   the counting noise of a window with >= 1000 counts is a few percent.
3. The whole dip is flagged, not only its bottom: the footprint leaves the sample and comes back over several
   windows, so the windows on either side of a core have lost part of the sample too (the per-window sizes there
   are biased low, and a cut at the core alone puts flags right at the threshold, where noise decides). From each
   core the flag extends to the contiguous windows with data on both sides while their intensity stays below
   OFF_SAMPLE['release'] (80 %) of their level: a fifth of the scattering missing. A window on the sample sits near
   its level (the 90 % quantile lies about 1.3 noise widths above the median: about 0.9-0.95 of the level with a
   few percent of noise), so 80 % is several noise widths below it; and only a dip with a core is extended, never a
   lone shallow one.
4. Without a detected modulation only an extreme dip has a core: below OFF_SAMPLE['extreme_fraction'] (20 %) of the
   level within OFF_SAMPLE['extreme_span_s'] (10 s) on each side; a single moderate dip is not attributed to the
   stage. Such a window probably saw no sample at all, or the beam was interrupted. Its flanks are extended in the
   same way (below 80 % of their level).
5. A window needs OFF_SAMPLE['min_side_windows'] (3) windows with data on each side; the first and last windows of
   a series cannot be judged (counted as not evaluated) and stop the extension of a dip. Windows with fewer than 1000
   counts (no data) are neither judged nor used for a level; a dip extends across them.
6. Corroboration (recorded, it does not decide): binned/frames.csv holds no separate Yoneda or specular intensity,
   but its Yoneda ridge position. A flagged window without a Yoneda ridge, while the ridge is found in at least
   OFF_SAMPLE['yoneda_share'] (50 %) of the windows that are not flagged, says so in its reason.
7. Sensitivity (recorded): how many windows the rule would flag with the core limit and the release level each
   moved by OFF_SAMPLE['sensitivity_step'] (0.1) down and up.

The physical path keeps these windows in its tables with their values, but they cast no stage vote, and the R/D
trends, the reversal (zig-zag) counts, the mass continuity and the literature method's trend of D and its checks of
the network leave them out.

Example:
    from sputtergisaxs.sputter_offsample import beam_off_sample
    flags, reasons, record = beam_off_sample(times, intensity_rates, period_s)
"""
from __future__ import annotations
import math
import numpy as np

OFF_SAMPLE = dict(
    fraction=.5,          # with a modulation: a core below half the on-sample level, more than half the scattering missing
    extreme_fraction=.2,  # without one: only a core below a fifth of the level (no sample seen, or no beam)
    release=.8,           # a dip goes on to the windows next to its core while they stay below 80 % of the level
    quantile=.9,          # the on-sample level: a high quantile, robust to dips (up to ~90 % of a side) and spikes
    min_span_s=2.,        # the shortest side span (the modulation search starts at 2 s lags)
    extreme_span_s=10.,   # side span without a modulation: holds on-sample windows around a short interruption
    min_side_windows=3,   # windows with data needed on each side for a level
    yoneda_share=.5,      # the Yoneda ridge is found in at least this share of the other windows: its absence counts
    sensitivity_step=.1,  # the record says how many windows the core limit and the release +- this step would flag
)
OFF_METHOD = ('the on-sample level of a window is the 90% quantile of ln(intensity_rate) over the other windows with '
              'data within one modulation period before it and within one after it, the lower of the two; with a '
              'detected modulation a window below 50% of its level is the core of a dip, without one only a window '
              'below 20% of the level within 10 s on each side; the dip is flagged with the contiguous windows on both '
              'sides of its core while they stay below 80% of their level; at least 3 windows with data on each side, '
              'windows with < 1000 counts are neither judged nor used; a missing Yoneda ridge (found in >= 50% of the '
              'other windows) is recorded as corroboration')
OFF_MEANING = ('the beam is probably partly or fully off the sample: the sample stage is translated back and forth '
               'during the deposition (against beam damage), and near the ends of its travel the footprint can reach '
               'the sample edge or leave the sample; the data alone do not prove the cause')
OFF_SCOPE = ('kept in the tables with their values; no stage vote (neutral: the path carries the stage of the '
             'neighbouring windows, stage_imputed); left out of R_trend_nm / D_trend_nm, the reversal counts (the R/D '
             'zig-zags are counted within the stretches between them), the mass continuity and the literature '
             'method\'s trend of D and its checks of the network')
OFF_REASON = ('intensity {percent}% of the level within one modulation period ({period} s) before and after (below '
              '{limit}%): the beam is probably partly or fully off the sample (stage translation)')
OFF_FLANK = ('intensity {percent}% of the level within one modulation period ({period} s) before and after, next to a '
             'window below {limit}% in the same dip (below {release}%): the beam is probably partly off the sample '
             '(stage translation)')
OFF_EXTREME_REASON = ('intensity {percent}% of the level within {span} s before and after, without a periodic '
                      'modulation (below {limit}%): the beam was probably off the sample or interrupted')
OFF_EXTREME_FLANK = ('intensity {percent}% of the level within {span} s before and after, without a periodic '
                     'modulation, next to a window below {limit}% in the same dip (below {release}%): the beam was '
                     'probably partly off the sample or interrupted')
OFF_YONEDA = '; no Yoneda ridge in this window, unlike most other windows'
TEMPLATES = (OFF_REASON, OFF_REASON + OFF_YONEDA, OFF_FLANK, OFF_FLANK + OFF_YONEDA, OFF_EXTREME_REASON,
             OFF_EXTREME_REASON + OFF_YONEDA, OFF_EXTREME_FLANK, OFF_EXTREME_FLANK + OFF_YONEDA)
"""Every reason this module writes (filled with str.format)."""


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def intensity_ratios(times, rates, span, *, no_data=None, cfg=OFF_SAMPLE):
    """(ratios, used): each window's intensity over its on-sample level (None where it is not judged) and the
    indices of the windows with data in time order."""
    n = len(times)
    no_data = list(no_data) if no_data is not None else [False] * n
    ratios = [None] * n
    used = [i for i in range(n) if _num(times[i]) is not None and (_num(rates[i]) or 0.) > 0 and not no_data[i]]
    used.sort(key=lambda i: float(times[i]))
    ts = np.array([float(times[i]) for i in used]); ys = np.log([float(rates[i]) for i in used])
    need = cfg['min_side_windows']
    for k, i in enumerate(used):
        a = int(np.searchsorted(ts, ts[k] - span, side='left')); b = int(np.searchsorted(ts, ts[k] + span, side='right'))
        lo = int(np.searchsorted(ts, ts[k], side='left')); hi = int(np.searchsorted(ts, ts[k], side='right'))
        before, after = ys[a:lo], ys[hi:b]
        if len(before) < need or len(after) < need:
            continue
        level = min(float(np.quantile(before, cfg['quantile'])), float(np.quantile(after, cfg['quantile'])))
        ratios[i] = math.exp(ys[k] - level)
    return ratios, used


def dips(ratios, used, limit, release):
    """(flags, core): the cores below ``limit`` and, from each, the contiguous windows of ``used`` (time order) on both
    sides while their ratio stays below ``release`` (a window that is not judged ends the dip)."""
    core = [r is not None and r < limit for r in ratios]
    flags = list(core)
    for k, i in enumerate(used):
        if not core[i]:
            continue
        for step in (-1, 1):
            j = k + step
            while 0 <= j < len(used) and ratios[used[j]] is not None and ratios[used[j]] < release:
                flags[used[j]] = True
                j += step
    return flags, core


def beam_off_sample(times, rates, period=None, *, no_data=None, yoneda=None, cfg=OFF_SAMPLE):
    """(flags, reasons, record) per window: True where the beam is probably partly or fully off the sample.

    times (s) and rates (intensity_rate, counts/s) per window, in any order; period: the detected modulation period
    (s) or None; no_data: windows with too few counts (never judged, never a level); yoneda: the Yoneda ridge
    position per window (None where none was found), for the corroboration only.
    """
    n = len(times)
    modulated = period is not None and period > 0
    span = max(float(period), cfg['min_span_s']) if modulated else cfg['extreme_span_s']
    limit = cfg['fraction'] if modulated else cfg['extreme_fraction']
    release = cfg['release']
    ratios, used = intensity_ratios(times, rates, span, no_data=no_data, cfg=cfg)
    flags, core = dips(ratios, used, limit, release)
    reasons = [''] * n
    values = dict(limit=f'{100 * limit:g}', release=f'{100 * release:g}')
    for i in range(n):
        if flags[i]:
            fields = dict(percent=f'{100 * ratios[i]:.0f}', **values)
            if modulated:
                reasons[i] = (OFF_REASON if core[i] else OFF_FLANK).format(period=f'{span:.3g}', **fields)
            else:
                reasons[i] = (OFF_EXTREME_REASON if core[i] else OFF_EXTREME_FLANK).format(span=f'{span:.3g}', **fields)
    others = [i for i in used if not flags[i]]
    share = None
    missing = 0
    if yoneda is not None and others:
        share = sum(_num(yoneda[i]) is not None for i in others) / len(others)
        if share >= cfg['yoneda_share']:
            for i in range(n):
                if flags[i] and _num(yoneda[i]) is None:
                    reasons[i] += OFF_YONEDA
                    missing += 1
    step = cfg['sensitivity_step']
    sensitivity = [dict(limit_fraction=round(a, 4), release_fraction=round(b, 4), windows=sum(dips(ratios, used, a, b)[0]))
                   for a, b in ((limit - step, release), (limit + step, release), (limit, release - step),
                                (limit, release + step)) if 0 < a < b]
    judged = [r for r in ratios if r is not None]
    record = dict(windows=sum(flags), core_windows=sum(core), evaluated=len(judged), not_evaluated=len(used) - len(judged),
                  rule='modulation' if modulated else 'extreme_dip_only', limit_fraction=limit,
                  release_fraction=release, span_s=span, quantile=cfg['quantile'], min_side_windows=cfg['min_side_windows'],
                  yoneda_missing=missing, yoneda_found_share=round(share, 4) if share is not None else None,
                  lowest_ratio=round(min(judged), 4) if judged else None, sensitivity=sensitivity,
                  method=OFF_METHOD, meaning=OFF_MEANING, scope=OFF_SCOPE)
    return flags, reasons, record


__all__ = ['OFF_EXTREME_FLANK', 'OFF_EXTREME_REASON', 'OFF_FLANK', 'OFF_REASON', 'OFF_SAMPLE', 'OFF_YONEDA', 'TEMPLATES',
           'beam_off_sample', 'dips', 'intensity_ratios']
