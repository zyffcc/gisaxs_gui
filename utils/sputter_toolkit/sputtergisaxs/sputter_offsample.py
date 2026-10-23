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
2. With a detected modulation a window is flagged when its intensity is below OFF_SAMPLE['fraction'] (50 %) of
   that level: more than half of the scattering is missing, as when more than half of the footprint has left the
   sample (the scattered intensity scales with the illuminated sample area). Growth changes the intensity far
   more slowly, and the counting noise of a window with >= 1000 counts is a few percent.
3. Without a detected modulation only an extreme dip is flagged: below OFF_SAMPLE['extreme_fraction'] (20 %) of the
   level within OFF_SAMPLE['extreme_span_s'] (10 s) on each side; a single moderate dip is not attributed to the
   stage. Such a window probably saw no sample at all, or the beam was interrupted.
4. A window needs OFF_SAMPLE['min_side_windows'] (3) windows with data on each side; the first and last windows of
   a series cannot be judged (counted as not evaluated). Windows with fewer than 1000 counts (no data) are neither
   judged nor used for a level.
5. Corroboration (recorded, it does not decide): binned/frames.csv holds no separate Yoneda or specular intensity,
   but its Yoneda ridge position. A flagged window without a Yoneda ridge, while the ridge is found in at least
   OFF_SAMPLE['yoneda_share'] (50 %) of the windows that are not flagged, says so in its reason.

The physical path keeps these windows in its tables with their values, but they cast no stage vote, and the R/D
trends, the reversal (zig-zag) counts and the literature method's trend of D leave them out.

Example:
    from sputtergisaxs.sputter_offsample import beam_off_sample
    flags, reasons, record = beam_off_sample(times, intensity_rates, period_s)
"""
from __future__ import annotations
import math
import numpy as np

OFF_SAMPLE = dict(
    fraction=.5,          # with a modulation: below half the on-sample level, more than half the scattering is missing
    extreme_fraction=.2,  # without one: only a dip to below a fifth of the level (no sample seen, or no beam)
    quantile=.9,          # the on-sample level: a high quantile, robust to dips (up to ~90 % of a side) and spikes
    min_span_s=2.,        # the shortest side span (the modulation search starts at 2 s lags)
    extreme_span_s=10.,   # side span without a modulation: holds on-sample windows around a short interruption
    min_side_windows=3,   # windows with data needed on each side for a level
    yoneda_share=.5,      # the Yoneda ridge is found in at least this share of the other windows: its absence counts
)
OFF_METHOD = ('the on-sample level of a window is the 90% quantile of ln(intensity_rate) over the other windows with '
              'data within one modulation period before it and within one after it, the lower of the two; with a '
              'detected modulation a window below 50% of its level is flagged, without one only a window below 20% of '
              'the level within 10 s on each side; at least 3 windows with data on each side, windows with < 1000 '
              'counts are neither judged nor used; a missing Yoneda ridge (found in >= 50% of the other windows) is '
              'recorded as corroboration')
OFF_MEANING = ('the beam is probably partly or fully off the sample: the sample stage is translated back and forth '
               'during the deposition (against beam damage), and near the ends of its travel the footprint can reach '
               'the sample edge or leave the sample; the data alone do not prove the cause')
OFF_SCOPE = ('kept in the tables with their values; no stage vote (neutral); left out of R_trend_nm / D_trend_nm, the '
             'reversal (zig-zag) counts and the literature method\'s trend of D')
OFF_REASON = ('intensity {percent}% of the level within one modulation period ({period} s) before and after (below '
              '{limit}%): the beam is probably partly or fully off the sample (stage translation)')
OFF_EXTREME_REASON = ('intensity {percent}% of the level within {span} s before and after, without a periodic '
                      'modulation (below {limit}%): the beam was probably off the sample or interrupted')
OFF_YONEDA = '; no Yoneda ridge in this window, unlike most other windows'
TEMPLATES = (OFF_REASON, OFF_REASON + OFF_YONEDA, OFF_EXTREME_REASON, OFF_EXTREME_REASON + OFF_YONEDA)
"""Every reason this module writes (filled with str.format)."""


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def beam_off_sample(times, rates, period=None, *, no_data=None, yoneda=None, cfg=OFF_SAMPLE):
    """(flags, reasons, record) per window: True where the beam is probably partly or fully off the sample.

    times (s) and rates (intensity_rate, counts/s) per window, in any order; period: the detected modulation period
    (s) or None; no_data: windows with too few counts (never judged, never a level); yoneda: the Yoneda ridge
    position per window (None where none was found), for the corroboration only.
    """
    n = len(times)
    no_data = list(no_data) if no_data is not None else [False] * n
    flags, reasons, ratios = [False] * n, [''] * n, [None] * n
    used = [i for i in range(n) if _num(times[i]) is not None and (_num(rates[i]) or 0.) > 0 and not no_data[i]]
    used.sort(key=lambda i: float(times[i]))
    modulated = period is not None and period > 0
    span = max(float(period), cfg['min_span_s']) if modulated else cfg['extreme_span_s']
    limit = cfg['fraction'] if modulated else cfg['extreme_fraction']
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
        if ratios[i] < limit:
            flags[i] = True
    values = dict(limit=f'{100 * limit:g}')
    for i in range(n):
        if flags[i]:
            percent = f'{100 * ratios[i]:.0f}'
            reasons[i] = (OFF_REASON.format(percent=percent, period=f'{span:.3g}', **values) if modulated else
                          OFF_EXTREME_REASON.format(percent=percent, span=f'{span:.3g}', **values))
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
    record = dict(windows=sum(flags), evaluated=sum(r is not None for r in ratios),
                  not_evaluated=len(used) - sum(r is not None for r in ratios),
                  rule='modulation' if modulated else 'extreme_dip_only', limit_fraction=limit, span_s=span,
                  quantile=cfg['quantile'], min_side_windows=need, yoneda_missing=missing,
                  yoneda_found_share=round(share, 4) if share is not None else None,
                  lowest_ratio=round(min(r for r in ratios if r is not None), 4) if any(r is not None for r in ratios)
                  else None, method=OFF_METHOD, meaning=OFF_MEANING, scope=OFF_SCOPE)
    return flags, reasons, record


__all__ = ['OFF_EXTREME_REASON', 'OFF_REASON', 'OFF_SAMPLE', 'OFF_YONEDA', 'TEMPLATES', 'beam_off_sample']
