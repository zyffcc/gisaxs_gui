"""Per-frame tracking fit of a series (PLAN_zh.md M3).

Per frame (Tracker.process):
  cut -> both sides merged with the reference shift/ratio -> sigma = max(Poisson, left/right difference)
  -> skip rule (causal) -> warm start from the previous accepted frame: LM with fitting.track_steps
     evaluations (default 6) in the frame's own CurveCodec wrapped by ConstrainedCodec (u converted through
     physical parameters when the q window changed); the LM continues in chunks of track_steps up to
     fitting.track_max_steps while the chi2 is still falling or above the jump threshold
  -> population states (on/off) -> state signals -> fallback fit1d(policy fitting.fallback_policy = 'auto'),
     started fresh; the fallback solution replaces the tracked one only when its chi2 is lower by more than
     the same-basin tolerance max(25, 0.02 N) max(1, chi2/N) (among chi2-equivalent solutions the one that
     continues the previous frame is kept).
First frame: fit1d(policy fitting.first_policy = 'robust'), populations of one class sorted by R.
All tracking is Born (table engine, physics v2). model.sphere_model = 'dwba' runs the fit1d DWBA stage on the
first frame and on fallback frames only; its result is reported in the row's 'dwba' block, the tracked
columns stay Born (a DWBA LM step costs 0.3-0.6 s per frame).

A frame that cannot be read or cut (damaged file, another image shape) is written as a skipped row that names
the file; the series goes on. Skipped rows carry skip_kind: unreadable | min_counts | median | points |
fit_failed.

State signals of a tracked frame (any of them asks for a fallback):
  chi2_jump       chi2/point > fitting.chi2_jump x the median of the last 10 unflagged frames
  still_falling   the LM was still falling (> 1 % over the last 2 evaluations) at track_max_steps
  size_at_bound   a free R, h or D coordinate of an 'on' population reached a bound of its box that it did
                  not touch in the previous accepted frame (only populations that are significant in this
                  frame: one that is fading out wanders, and that alone must not start a fallback)
  chi2_drift      the chi2 baseline (median of the last 10 unflagged frames) is more than
                  fitting.chi2_drift (1.5) x its lowest value since the last fresh fit: the track fits clearly
                  worse than when the best solution was last searched (a slow drift never makes a chi2_jump);
                  the fresh fit uses fitting.drift_policy
No fallback is started within fitting.fallback_cooldown frames of the last one unless chi2/point is more than
twice the chi2/point of that fallback frame (a lasting bad regime is fitted once, not on every frame); the
row then carries the event 'fallback_suppressed(cooldown)'.
  refresh         (fitting.refresh_every = N > 0, default off) N frames after the last fresh fit a fresh
                  fit1d(policy fitting.first_policy) is run and compared like a fallback: a drift check for
                  tracks that sit in a worse solution without any signal (a smooth series never raises one).

Second start: a solution that rests on one frame only is not trusted as the only start of the next frame.
That is the case after an accepted fallback (a fresh solution won on this frame; whether it also describes
the following frames is not known yet) and after a suspicious frame: final chi2/point above the jump
threshold, or fewer points than 60 % of the median of the last 10 fitted frames (a side of the cut is
missing; possibly a damaged frame). The row stores the last solution before such frames (fit.reseed), the
next fitted frame is tracked from both starts and the lower chi2 is kept (event 'reseeded_from_frame_index').
A fallback solution therefore stays only when it is also the better start on the next frame.

Population states: a population is 'off' when its NNLS amplitude is 0 or when removing it raises chi2 by
less than OFF_DCHI2 = 9 (it is not significant at 3 sigma: a free population can always fit some noise, so
"exactly 0" alone would miss a population that has vanished; a larger threshold switches off populations
that are present). 'off' never starts a fallback. The
nonlinear coordinates of an off population are frozen at the values of the last frame in which it was
significant (removal cost above the 'on' bar below; reported as null, the frozen values are listed under
'frozen'; fit.anchor holds them while a population is on but not significant: the shape of such a population
may already be a fit to noise, and frozen there it came back again and again), its amplitude is 0 by definition. While off, its column stays in the
non-negative amplitude solve with the frozen parameters: when it gains more than chi2 9 there, the population
is refitted with free parameters; it is 'on' again when removing the refitted population would raise chi2 by
more than the 4-sigma quantile of chi2 with as many degrees of freedom as it has free parameters (26.8 for
amplitude, R, sigma_R, D and w): a refitted shape fits noise too, and with the bar of one parameter a
population that is absent came back on 0.5 % of the frames of the synthetic series and started fallbacks.
The same bar holds for a population that was off and is 'on' in a fresh solution. Every
fitting.off_probe_every-th off frame (default 3) the (R, D) grid of that population is searched (fit1d
gridsearch); it re-enters there when the chi2 falls by more than the same-basin tolerance (a grid search has
many trials: a population that comes back with other parameters has to be clearly needed).

Identity: populations are slots (pop1.. in the fit1d population order). A tracked LM step never exchanges
slots. A fresh solution (fallback, live backfill, batch chunk start) is matched to the previous accepted frame
by Hungarian assignment on (type, ln R, ln D); only populations of the same type and constraints can be
exchanged, and the identity is kept unless another assignment is strictly better.

Live mode: the calibration (qy-origin shift, left/right ratio, Yoneda row when the recipe gives none) is
made on the first usable frame when it arrives and stored in that row ('calibration'). Asynchronous work runs
in the second process and is written as additional lines of the same frame index (state 'backfilled'; the
last line of a frame index in frames.jsonl is the valid one):
  fallback   the tracked solution is written at once with state 'pending_fallback'; the fallback is
             backfilled. The first frame of a live run is a network proposal (fit1d 'preview') refined by the
             tracker's LM, written as 'pending_fallback'; the robust first-frame fit is backfilled. While one
             piece of asynchronous work is pending no second one is started (event 'fallback_suppressed').
  re-track   when a backfill is accepted, the frames that were written since (tracked from the provisional
             solution) are processed again in the second process as a continuation of the accepted solution,
             exactly as review mode would process them, and rewritten (row entry 'retrack': from = index of
             the accepted frame); frames that arrived meanwhile follow in further steps; when the re-track
             is within 3 frames of the current frame the main process tracks those itself (about 15 ms
             each) and continues from the rewritten rows. A fallback that was
             suppressed while another one was pending, and would have been allowed afterwards, is recovered
             the same way after a rejected backfill.
  resume     Tracker.resume_live() submits a restored 'pending_fallback' row again and continues an
             interrupted re-track.

Everything a later frame depends on is updated in _advance(row) from the JSON form of the row, so
restore(rows) (resume, batch re-seed, live re-track) rebuilds exactly the state of an uninterrupted run.
"""
from __future__ import annotations

import json
import time

import numpy as np

from research.fit1d import flags as fl
from research.fit1d.api import fit1d, single_thread
from research.fit1d.controller import lm_run, make_aligner
from research.fit1d.schema import jsonable
from research.series1d.annotate import WINDOW, Annotator
from research.series1d.inputs import Cutter, InputError, complete_live, curve_from_sides, total_counts
from research.series1d.model import (SIZE_FIELDS, ModelSpec, at_bound, canonical_order, coordinate_name,
                                     identity_points, low_q, match_identities, permute_result, populations,
                                     tolerance)

ROW_SCHEMA = 'gisaxs1d_series_row_v1'
VALUE_KEYS = ('R_nm', 'R_number_mean_nm', 'sigma_R_rel', 'D_nm', 'w_rel', 'w_nm', 'h_nm', 'sigma_h_rel')
ERROR_KEYS = ('R_nm_err', 'sigma_R_rel_err', 'D_nm_err', 'w_rel_err', 'h_nm_err', 'sigma_h_rel_err')
FROZEN_KEYS = ('R_nm', 'sigma_R_rel', 'D_nm', 'w_rel', 'h_nm', 'sigma_h_rel')
BASE_KEYS = ('schema', 'index', 'frame', 'file', 'time_s', 'total_counts', 'cut')
MIN_POINTS = 10
COOLDOWN_FACTOR = 2.0
LIVE_FIRST_POLICY = 'preview'   # live: provisional first frame = network proposal + tracker LM; robust fit backfilled
OFF_DCHI2 = 9.0              # a population whose removal raises chi2 by less than this is 'off' (3 sigma, 1 parameter)
REENTRY_DCHI2 = 9.0          # chi2 gain of an off population at its frozen parameters that starts a free refit
ON_PROBABILITY = 6.33e-5     # ... and it is 'on' again when the refitted population is significant at this level (4 sigma)
POINTS_FRACTION = 0.6        # a frame with fewer points than this x the median of the last 10 is suspicious
MIN_POINTS_HISTORY = 3
MAX_RETRACK_STEPS = 8        # live: asynchronous steps of one re-track before the main process finishes it
LOCAL_RETRACK_FRAMES = 3     # live: a re-track that is this close to the current frame is finished by the main process


def roundtrip(row):
    """The JSON form of a row (what frames.jsonl stores): plain types, non-finite floats as null."""
    return json.loads(json.dumps(jsonable(row), allow_nan=False))


def light_copy(row):
    """A row whose annotation fields can be rewritten without touching the stored row."""
    return dict(row, flags=dict(row['flags']), populations=[dict(p) for p in row['populations']])


def _sum_runs(a, b):
    out = dict(b)
    out['nfev'] = a['nfev']+b['nfev']
    out['evaluations'] = a['evaluations']+b['evaluations']
    return out


def seed_of(row):
    """The solution of a stored row as a start for another frame."""
    fit = row['fit']
    return dict(u=np.array(fit['u'], float), coef=np.array(fit['coef'], float), q_min=fit['q_min'],
                q_max=fit['q_max'], index=row['index'], off_age={int(k): int(v) for k, v in fit['off'].items()},
                anchor=np.array(fit.get('anchor') or fit['u'], float), weak={})


def _seed_json(seed):
    out = dict(u=[float(v) for v in seed['u']], coef=[float(v) for v in seed['coef']], q_min=float(seed['q_min']),
               q_max=float(seed['q_max']), index=int(seed['index']),
               off={str(j): int(a) for j, a in sorted(seed['off_age'].items())})
    if seed.get('anchor') is not None and not np.array_equal(seed['anchor'], seed['u']):
        out['anchor'] = [float(v) for v in seed['anchor']]
    return out


def _seed_from_json(seed):
    return dict(u=np.array(seed['u'], float), coef=np.array(seed['coef'], float), q_min=seed['q_min'],
                q_max=seed['q_max'], index=int(seed['index']),
                off_age={int(k): int(v) for k, v in seed['off'].items()},
                anchor=np.array(seed.get('anchor') or seed['u'], float), weak={})


class Tracker:
    def __init__(self, recipe, source, calibration, cutter=None, live=None, first_policy=None):
        """live: None (synchronous fallbacks) or an object with submit(payload) and retrack(task), both
        returning a handle (ready(), get()).
        first_policy: fit1d policy of the first fitted frame (default fitting.first_policy; batch chunks after
        the first one start with fitting.chunk_start_policy)."""
        self.recipe, self.source, self.cal = recipe, source, dict(calibration)
        self.spec = ModelSpec.from_recipe(recipe)
        self.cfg = recipe['fitting']
        self.first_policy = first_policy or self.cfg['first_policy']
        self.series = recipe['series']
        self.cutter = cutter if cutter is not None else Cutter(source, recipe, calibration)
        self.live = live
        self.history = {}                 # index -> the valid stored row (JSON form)
        self.written = {}                 # live: index -> time the first line of the frame was produced
        self.last_index = None            # index of the last processed frame
        self.pending = None               # live: the outstanding asynchronous work (fallback or re-track)
        self.suppressed = []              # live: (index, chi2/point) of fallbacks not started while one was pending
        self.async_errors = []            # live: failures of asynchronous work (reported in the summary notes)
        self.dwba = self._dwba_base()
        self._reset()

    def _reset(self):
        self.annot = Annotator(self.cfg)
        self.prev = None                  # last accepted solution: u, coef, q_min, q_max, index, weak
        self.off_age = {}                 # {slot: consecutive off frames - 1}
        self.bound_state = set()          # size coordinates at a bound in the last accepted frame
        self.totals = []                  # total counts of the earlier fitted frames (skip rule)
        self.npoints = []                 # number of points of the last fitted frames
        self.last_fallback = None         # (index, chi2/point) of the last frame whose fallback ran
        self.last_fresh = None            # index of the last frame with a fresh fit (first frame, fallback, refresh)
        self.drift_ref = None             # lowest chi2 baseline since the last fresh fit (chi2_drift signal)
        self.reseed = None                # second start for the next fitted frame (after a suspicious frame)

    # ------------------------------------------------------------------ settings
    def _dwba_base(self):
        if self.spec.sphere_model != 'dwba':
            return None
        m, g = self.recipe['model'], self.recipe['experiment_setup']
        if g is None:
            from research.series1d.recipe import RecipeError
            raise RecipeError("recipe: model.sphere_model: 'dwba' needs experiment_setup (wavelength_nm, alpha_i_deg)")
        spec = dict(wavelength_nm=g['wavelength_nm'], alpha_i_deg=g['alpha_i_deg'])
        if m['substrate_delta'] is not None:
            spec.update(substrate_delta=m['substrate_delta'], substrate_beta=m['substrate_beta'] or 0.0)
        else:
            spec['substrate'] = m['substrate']
        return spec

    def _dwba(self, cutinfo):
        if self.dwba is None:
            return None
        alpha = cutinfo.get('alpha_f_deg') or self.recipe['cut']['alpha_f_deg'] \
            or self.recipe['experiment_setup'].get('file_yoneda_alpha_f_deg')
        if alpha is None:
            from research.series1d.recipe import RecipeError
            raise RecipeError("recipe: model.sphere_model: 'dwba' needs the exit angle of the cut "
                              '(cut.alpha_f_deg, or yoneda_alpha_f_deg in the geometry file)')
        return dict(self.dwba, alpha_f_deg=float(alpha))

    def _qz(self, cutinfo):
        if self.spec.mode != 'gisaxs':
            return 0.0
        if self.spec.qz is not None:
            return float(self.spec.qz)
        qz = cutinfo.get('qz_vacuum')
        if qz is None:
            if not self.spec.needs_qz():
                return 0.0
            raise InputError('model.qz is not given and the input carries no qz of the cut (no left_qz/right_qz '
                             'arrays): give model.qz in nm^-1 (the vacuum qz at the Yoneda row; 0 only on purpose)')
        return float(qz)

    def chi2_reference(self):
        return self.annot.baseline()

    # ------------------------------------------------------------------ frame loop
    def process(self, i):
        """Fit frame i of the source; returns its row (JSON types) and advances the state."""
        from research.series1d.recipe import RecipeError
        t0 = time.perf_counter()
        number, name = self.source.info(i)
        base = dict(schema=ROW_SCHEMA, index=int(i), frame=int(number), file=name,
                    time_s=self.series['time_step_s']*(number-self.series['time_origin_frame']),
                    total_counts=None, cut=self.cutter.state() or {})
        sides = curve = reason = None
        cutinfo = {}
        try:
            sides, cutinfo = self.cutter.sides(i)
        except RecipeError:
            raise
        except Exception as exc:          # noqa: BLE001 - a damaged or foreign file must not end the series
            reason = ('unreadable', f'cannot read or cut {name}: {type(exc).__name__}: {exc}')
        if sides is not None:
            total = total_counts(sides)
            base.update(total_counts=total, cut=dict(cutinfo, **(self.cutter.state() or {})))
            reason = self._skip_reason(total)
        t_cut = time.perf_counter()
        if reason is None and self.cal.get('pending'):
            # live: the calibration is made on the first usable frame, at its arrival (frames <= t only)
            cal, sides = complete_live(self.cal, self.cutter, self.source, self.recipe, i, sides)
            self.cal = roundtrip(cal)
            base['calibration'] = dict(self.cal)
            base['cut'] = dict(cutinfo, **(self.cutter.state() or {}))
        if reason is None:
            curve = curve_from_sides(sides, self.cal['delta'], self.cal['ratio'], self.recipe['model']['sigma'])
            if len(curve[0]) < MIN_POINTS:
                reason = ('points', f'only {len(curve[0])} valid points')
        t_curve = time.perf_counter()
        if reason is not None:
            row = self._skipped_row(base, *reason)
        else:
            q, y, s = curve
            qz = self._qz(cutinfo)
            try:
                with single_thread():
                    if self.prev is None:
                        row = self._first(base, q, y, s, qz, cutinfo)
                    else:
                        row = self._track(base, q, y, s, qz, cutinfo)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                if isinstance(exc, (RecipeError, InputError)):
                    raise
                row = self._skipped_row(base, 'fit_failed', f'fit failed: {type(exc).__name__}: {exc}')
        t1 = time.perf_counter()
        row['timing'] = dict(row.get('timing') or {}, cut_s=t_cut-t0, curve_s=t_curve-t_cut, fit_s=t1-t_curve,
                             total_s=t1-t0)
        row = roundtrip(row)
        self._advance(row)
        self.history[row['index']] = row
        self.written[row['index']] = t1
        self.last_index = row['index']
        if self.pending is not None and self.pending['kind'] == 'fallback' \
                and self.pending['row'].get('index') == row['index']:
            self.pending['row'] = row                 # the annotated JSON form is what the backfill starts from
        return row

    def _skip_reason(self, total):
        """(kind, text) when the frame is skipped for its counts, else None."""
        skip = self.series['skip']
        if not np.isfinite(total):
            return 'min_counts', 'the total counts of the cut are not finite'
        if total < skip['min_counts']:
            return 'min_counts', f'total counts {total:.6g} < series.skip.min_counts {skip["min_counts"]:g}'
        if self.totals and skip['fraction'] > 0:
            median = float(np.median(self.totals))
            if total < skip['fraction']*median:
                return 'median', (f'total counts {total:.6g} < {skip["fraction"]:g} x running median of the earlier '
                                  f'frames ({median:.6g})')
        return None

    def _skipped_row(self, base, kind, reason):
        return dict(base, mode='skipped', state='final', n_points=0, chi2=None, chi2_per_point=None, qz_nm_inv=None,
                    populations=[], background_B=None, low_q=None, skip_reason=reason, skip_kind=kind,
                    flags=dict(fallback=False, fallback_accepted=False, pending_fallback=False, signals=[],
                               at_bound=[], weakly_constrained=[], weakly_constrained_reportable=[], off=[],
                               events=[]), fit=None)

    def _advance(self, row):
        """State update from the JSON form of a row (the only place where the frame-to-frame state changes)."""
        if row.get('calibration'):
            self.cal = dict(row['calibration'])
            self.cutter.calibrated(self.cal)
        self.cutter.restore(row.get('cut'))
        if row['mode'] != 'skipped':
            fit = row['fit']
            weak = {}
            self.totals.append(row['total_counts'])
            self.npoints = (self.npoints+[int(row['n_points'])])[-WINDOW:]
            self.prev = dict(u=np.array(fit['u'], float), coef=np.array(fit['coef'], float), q_min=fit['q_min'],
                             q_max=fit['q_max'], index=row['index'], chi2_per_point=row['chi2_per_point'], weak=weak,
                             anchor=np.array(fit.get('anchor') or fit['u'], float))
            self.off_age = {int(k): int(v) for k, v in fit['off'].items()}
            self.bound_state = {int(c) for c in fit['bound_state']}
            self.reseed = _seed_from_json(fit['reseed']) if fit.get('reseed') else None
            if row['flags']['fallback']:
                self.last_fallback = (row['index'], row['chi2_per_point'])
            if row['flags']['fallback'] or row['mode'] == 'first' or row['state'] == 'pending_fallback':
                self.last_fresh = row['index']
                self.drift_ref = None
        self.annot.annotate(row)
        if row['mode'] != 'skipped' and self.last_fresh != row['index']:
            baseline = self.annot.baseline()
            if baseline is not None:
                self.drift_ref = baseline if self.drift_ref is None else min(self.drift_ref, baseline)
        if row['mode'] != 'skipped' and self.cfg['continuity'] is not None:
            for pop in row['populations']:
                for key in pop.get('weak', ()):
                    smoothed = pop.get(f'{key}_smoothed')
                    if smoothed is not None and smoothed != pop.get(key):
                        self.prev['weak'][(pop['slot'], key)] = smoothed
        return row

    def restore(self, rows):
        """Rebuild the state from stored rows (the last line of each frame index is the valid one)."""
        latest = {}
        for row in rows:
            latest[row['index']] = row
        self._reset()
        self.history = {}
        for index in sorted(latest):
            self._advance(light_copy(latest[index]))
            self.history[index] = latest[index]
        self.last_index = max(latest) if latest else None
        return len(latest)

    def rebuild(self):
        """The state as a function of the stored rows (after rows were rewritten)."""
        self._reset()
        for index in sorted(self.history):
            self._advance(light_copy(self.history[index]))

    # ------------------------------------------------------------------ LM pieces
    def _lm(self, fit, setup, u0, reference):
        """Adaptive LM: track_steps evaluations, continued while falling / above the jump threshold."""
        step, limit = self.cfg['track_steps'], self.cfg['track_max_steps']
        n = len(setup.y)
        r = lm_run(fit, setup, u0, step)
        steps, runs = step, 1
        while steps < limit and r['status'] == 0 and \
                (r['falling'] or (reference is not None and r['chi2']/n > self.cfg['chi2_jump']*reference)):
            k = min(step, limit-steps)
            r = _sum_runs(r, lm_run(fit, setup, r['u'], k))
            steps += k
            runs += 1
        r['steps'], r['lm_runs'] = steps, runs
        return r

    def _frozen(self, off):
        return [c for j in sorted(off) for c in self.spec.slot_coords(j)]

    def _feedback(self, u0, codec, prev, off):
        """Continuity feedback: weakly constrained parameters start from their smoothed values."""
        cont = self.cfg['continuity']
        if not cont or not cont['feedback'] or not prev.get('weak'):
            return u0
        names = {'R_nm': ('R', 0), 'sigma_R_rel': ('sigma_R', 1), 'h_nm': ('h', 2), 'sigma_h_rel': ('sigma_h', 3),
                 'D_nm': ('D', 4), 'w_rel': ('sigma_D', 5)}
        p = {k: np.array(v, float, copy=True) for k, v in codec.decode(u0).items()}
        touched = set()
        for (slot, key), value in sorted(prev['weak'].items()):
            if slot in off:
                continue
            field_, offset = names[key]
            p[field_][slot] = value
            touched.update({7*slot+offset} | ({7*slot+4} if key == 'R_nm' and self.spec.sf_on else set()))
        if not touched:
            return u0
        u1 = codec.encode(p)
        out = u0.copy()
        for c in sorted(touched):
            out[c] = u1[c]
        return out

    def _track_from(self, q, y, s, qz, prev, off_age, reference):
        """One tracked solution from a previous solution: (r, setup, off_age, events)."""
        spec = self.spec
        n = len(q)
        codec = spec.codec(q[0], q[-1])
        codec_prev = spec.codec(prev['q_min'], prev['q_max'])
        u_prev = spec.convert(prev['u'], codec_prev, codec)
        # the values at which every population was last significant (what an off population is frozen at)
        u_anchor = spec.convert(prev['anchor'], codec_prev, codec) if prev.get('anchor') is not None else u_prev
        off = set(off_age)
        setup = spec.setup(q, y, s, qz, self._frozen(off))
        fit = setup.make_fit()
        r = self._lm(fit, setup, self._feedback(u_prev, codec, prev, off), reference)
        events = []
        runs = r['lm_runs']
        # populations that were off: amplitude back at the frozen parameters -> refit them free
        back = [j for j in sorted(off) if r['coef'][j] > 0]
        if back:
            gains = self._gains(fit, r, back, off-set(back))
            back = [j for j in back if gains[j] > REENTRY_DCHI2]
        if back:
            setup2 = spec.setup(q, y, s, qz, self._frozen(off-set(back)))
            fit2 = setup2.make_fit()
            r2 = self._lm(fit2, setup2, r['u'], reference)
            runs += r2['lm_runs']
            r2, dropped = self._prune(fit2, r2, off-set(back), u_anchor, n, setup2, back)
            for j in back:
                if j in dropped:                        # its amplitude came back, but not significantly
                    events.append(f'{spec.labels[j]}:return_not_significant')
            if not set(back) <= set(dropped):
                for j in back:
                    if j not in dropped:
                        off.discard(j)
                        events.append(f'{spec.labels[j]}:on')
                for j in dropped:
                    off.add(j)
                    if j not in back:
                        events.append(f'{spec.labels[j]}:off')
                r, setup, fit = r2, setup2, fit2
        # periodic (R, D) grid probe for populations that stay off
        every = self.cfg['off_probe_every']
        for j in sorted(off):
            if j in back or not every or (off_age.get(j, 0)+1) % every:
                continue
            if spec.types[j] not in (1, 3):
                continue
            setup3 = spec.setup(q, y, s, qz, self._frozen(off-{j}))
            fit3 = setup3.make_fit()
            aligner = make_aligner(fit3, setup3)
            start = np.array(r['u'], float)
            start[spec.slot_coords(j)] = u_anchor[spec.slot_coords(j)]
            now = aligner.chi2(start)
            u_probe, found = aligner.search_population(start, j, now)
            if found < now-tolerance(n, now):
                r3 = self._lm(fit3, setup3, u_probe, reference)
                runs += r3['lm_runs']
                r3, dropped = self._prune(fit3, r3, off-{j}, u_anchor, n, setup3, (j,))
                if j not in dropped and r3['chi2'] < r['chi2']-tolerance(n, r['chi2']):
                    for k in dropped:
                        off.add(k)
                        events.append(f'{spec.labels[k]}:off')
                    r, setup, fit = r3, setup3, fit3
                    off.discard(j)
                    events.append(f'{spec.labels[j]}:on(probe)')
        # populations without a significant amplitude: off, frozen at the last active values
        r, dropped = self._prune(fit, r, off, u_anchor, n)
        for j in dropped:
            off.add(j)
            events.append(f'{spec.labels[j]}:off')
        u = np.array(r['u'], float)
        coef = np.array(r['coef'], float)
        for j in off:
            u[spec.slot_coords(j)] = u_anchor[spec.slot_coords(j)]
            coef[j] = 0.0
        r = dict(r, u=u, coef=coef, lm_runs=runs, anchor=self._anchor(setup, r, u, u_anchor, off))
        return r, setup, {j: (off_age[j]+1 if j in off_age else 0) for j in sorted(off)}, events

    def _anchor(self, setup, r, u, u_anchor, off):
        """u with the coordinates of every population at the values where it was last significant: an 'on'
        population that is not significant in this frame (removing it raises chi2 by less than its
        _on_threshold: its shape may be a fit to noise) keeps its earlier anchor."""
        anchor = np.array(u, float)
        if u_anchor is None:
            return anchor
        removal = r.get('removal') or {}
        for j in self.spec.slots:
            if j in off or removal.get(j, np.inf) < self._on_threshold(setup, j):
                anchor[self.spec.slot_coords(j)] = u_anchor[self.spec.slot_coords(j)]
        return anchor

    def _gains(self, fit, r, slots, excluded):
        """{slot: chi2 increase when its column is removed} at the solution r (amplitudes re-solved; the
        columns of `excluded` slots are absent in both solves)."""
        m = fit.matrix_from(*fit.full(np.asarray(r['u'], float)))
        fit._armed = False
        for j in excluded:
            m[:, j] = 0.0
        res = fit.solve(m)[2]
        base = float(res@res)
        out = {}
        for j in slots:
            m2 = m.copy()
            m2[:, j] = 0.0
            res2 = fit.solve(m2)[2]
            out[j] = float(res2@res2)-base
        return out

    def _on_threshold(self, setup, j):
        """Chi2 increase (on removal) that an off population needs to be 'on' again: the ON_PROBABILITY
        (4 sigma) quantile of chi2 with as many degrees of freedom as the population has free parameters
        (its amplitude and its free nonlinear coordinates; 26.8 for 5, 16 for the amplitude alone). A
        population whose shape was refitted to the frame fits some noise, so the bar of one parameter and
        3 sigma is too low (hysteresis: it stays on down to OFF_DCHI2)."""
        from scipy.stats import chi2 as chi2_distribution
        free = set(setup.free_coords())
        k = 1+sum(c in free for c in self.spec.slot_coords(j))
        return float(chi2_distribution.isf(ON_PROBABILITY, k))

    def _prune(self, fit, r, off, u_frozen, n, setup=None, entering=()):
        """Populations that the data do not need are switched off: (r with their amplitude 0, [slots]).

        Greedy: the 'on' population whose removal raises chi2 least is removed while the total increase
        stays below its threshold (amplitudes re-solved without it; the nonlinear parameters of the others
        are kept): OFF_DCHI2 for a population that was on, _on_threshold for one that is `entering` from
        'off'. A population with NNLS amplitude 0 is the limiting case (increase 0)."""
        spec = self.spec
        u = np.array(r['u'], float)
        on = [j for j in spec.slots if j not in off]
        m = fit.matrix_from(*fit.full(u))
        fit._armed = False
        for j in off:
            m[:, j] = 0.0
        coef, _, res = fit.solve(m)
        chi2_0 = chi2 = float(res@res)
        limit = {j: (self._on_threshold(setup, j) if j in entering else OFF_DCHI2) for j in on}
        dropped, removal = [], None
        while on:
            trials = []
            for j in on:
                m2 = m.copy()
                m2[:, j] = 0.0
                c2, _, res2 = fit.solve(m2)
                trials.append((float(res2@res2), j, c2, m2))
            if removal is None:                         # chi2 increase when one population is removed
                removal = {t[1]: (t[0]-chi2_0 if coef[t[1]] > 0 else 0.0) for t in trials}
            removable = [t for t in trials if not coef[t[1]] > 0 or t[0]-chi2_0 < limit[t[1]]]
            if not removable:
                break
            chi2_j, j, c2, m2 = min(removable, key=lambda t: (t[0], t[1]))
            dropped.append(j)
            on.remove(j)
            m, coef, chi2 = m2, c2, chi2_j
        r = dict(r, removal=removal or {})
        if not dropped and not off:
            return r, dropped
        coef = np.array(coef, float)
        for j in list(off)+dropped:
            coef[j] = 0.0
            if u_frozen is not None:
                u[spec.slot_coords(j)] = u_frozen[spec.slot_coords(j)]
        return dict(r, u=u, coef=coef, chi2=chi2), dropped

    @staticmethod
    def _dwba_block(top, dwba):
        return dict(chi2_per_point=top['chi2_per_point'], born_chi2_per_point=top['born_reference']['chi2_per_point'],
                    geometry=dwba, populations={int(p['slot']): {k: p.get(k) for k in
                                                                 ('R_nm', 'R_nm_err', 'D_nm', 'D_nm_err', 'sigma_R_rel',
                                                                  'w_rel', 'amplitude')} for p in top['populations']})

    def _fresh(self, q, y, s, qz, policy, cutinfo):
        """fit1d started fresh + Born LM polish in the tracker's setup: (r, setup, meta)."""
        spec = self.spec
        dwba = self._dwba(cutinfo) if policy != 'preview' else None       # a preview has no DWBA refinement
        res = fit1d(q, y, s, **spec.fit1d_kwargs(qz, policy, dwba))
        top = res['results'][0]
        setup = spec.setup(q, y, s, qz)
        fit = setup.make_fit()
        r = self._lm(fit, setup, np.array(top['u'], float), None)
        r['_fit'] = fit
        meta = dict(policy=policy, stages_run=top['stages_run'],
                    signals=top['signals'].get('fired', []), chi2_per_point=top['chi2_per_point'],
                    sphere_model=top.get('sphere_model', 'born'),
                    _timing=dict(fit1d_s=res['timing']['total_seconds']))
        if dwba is not None and top.get('born_reference') is not None:
            meta['dwba'] = self._dwba_block(top, dwba)
        return r, setup, meta

    def _settle_fresh(self, r, setup, fit, meta, u_ref, perm, previous_off):
        """A fresh solution in the series' slot order: (r, off_age, meta). u_ref: frozen values for off slots;
        previous_off: {slot: age} of the frame the solution is compared with."""
        spec = self.spec
        r = permute_result(r, perm)
        if 'dwba' in meta:
            source = meta['dwba']['populations']
            meta = dict(meta, dwba=dict(meta['dwba'], populations={t: source[s_] for t, s_ in perm.items()}))
        meta = dict(meta, identity_permutation=[[int(t), int(s_)] for t, s_ in sorted(perm.items()) if t != s_])
        r, dropped = self._prune(fit, r, set(), u_ref, len(setup.y), setup, tuple(previous_off))
        r = dict(r, anchor=self._anchor(setup, r, r['u'], u_ref, set(dropped)))
        off_age, changes = {}, []
        for j in spec.slots:
            if j in dropped:
                off_age[j] = previous_off[j]+1 if j in previous_off else 0
                if u_ref is not None and j not in previous_off:
                    changes.append(f'{spec.labels[j]}:off(fresh)')
            elif j in previous_off:
                changes.append(f'{spec.labels[j]}:on(fresh)')
        return r, off_age, dict(meta, state_changes=changes)

    # ------------------------------------------------------------------ first frame, tracked frame
    def _first(self, base, q, y, s, qz, cutinfo):
        policy = self.first_policy
        provisional = self.live is not None and policy == 'robust'
        r, setup, meta = self._fresh(q, y, s, qz, LIVE_FIRST_POLICY if provisional else policy, cutinfo)
        fit = self._permuted_fit(setup, None, r)
        pruned, _ = self._prune(fit, r, set(), None, len(q))            # order by R among the populations that count
        perm = canonical_order(self.spec, setup.codec, pruned['u'], pruned['coef'])
        r, off_age, meta = self._settle_fresh(r, setup, fit, meta, None, perm, {})
        row = self._solution_row(base, r, setup, 'first', off_age, [], [], first=meta)
        if provisional:
            self._submit(row, q, y, s, qz, cutinfo, policy, ['first_frame'], None)
        return row

    def _drifted(self, reference):
        """The chi2 baseline has risen by fitting.chi2_drift above its lowest value since the last fresh fit."""
        factor = self.cfg['chi2_drift']
        return bool(factor and reference is not None and self.drift_ref is not None
                    and reference > factor*self.drift_ref)

    def _few_points(self, n):
        return len(self.npoints) >= MIN_POINTS_HISTORY and n < POINTS_FRACTION*float(np.median(self.npoints))

    def _suspicious(self, n, chi2pp, reference, few=None):
        """A frame whose solution is not trusted as the only start of the next frame."""
        if reference is not None and chi2pp > self.cfg['chi2_jump']*reference:
            return True
        return bool(self._few_points(n) if few is None else few)

    def _good_seed(self):
        """The last solution before the current run of suspicious frames."""
        if self.reseed is not None:
            return self.reseed
        return dict(self.prev, off_age=dict(self.off_age))

    def _track(self, base, q, y, s, qz, cutinfo):
        n = len(q)
        reference = self.chi2_reference()
        good = self._good_seed()
        r, setup, off_age, events = self._track_from(q, y, s, qz, self.prev, self.off_age, reference)
        if self.reseed is not None:                   # the frame before was suspicious: second start
            seed = self.reseed
            r2, setup2, off2, events2 = self._track_from(q, y, s, qz, seed, seed['off_age'], reference)
            if r2['chi2'] < r['chi2']:
                r, setup, off_age = r2, setup2, off2
                events = events2+[f"reseeded_from_frame_index:{seed['index']}"]
        chi2pp = r['chi2']/n
        signals = []
        if reference is not None and chi2pp > self.cfg['chi2_jump']*reference:
            signals.append('chi2_jump')
        if r['falling'] and r['steps'] >= self.cfg['track_max_steps']:
            signals.append('still_falling')
        # a population that is not significant in this frame (it is fading out) raises no bound signal
        removal = r.get('removal') or {}
        if any(removal.get(c//7, np.inf) >= self._on_threshold(setup, c//7)
               for c in set(self._size_bounds(setup, r, off_age))-self.bound_state):
            signals.append('size_at_bound')
        if self._drifted(reference):
            signals.append('chi2_drift')
        fallback = None
        every = self.cfg['refresh_every']
        refresh = bool(every) and self.last_fresh is not None and base['index']-self.last_fresh >= every
        if refresh:
            signals.append('refresh')
        allowed = refresh or self._fallback_allowed(base['index'], chi2pp)
        if signals and not allowed:
            events = events+['fallback_suppressed(cooldown)']
        if signals and allowed:
            policy = self.cfg['first_policy'] if refresh else \
                (self.cfg['drift_policy'] if signals == ['chi2_drift'] else self.cfg['fallback_policy'])
            if self.live is not None:
                row = self._solution_row(base, r, setup, 'track', off_age, signals, events)
                if self._suspicious(n, chi2pp, reference):
                    row['fit']['reseed'] = _seed_json(good)
                if self.pending is None:
                    self._submit(row, q, y, s, qz, cutinfo, policy, signals, (good, self._few_points(n)))
                else:
                    row['flags']['events'].append('fallback_suppressed(asynchronous work is pending)')
                    self.suppressed.append((base['index'], chi2pp))
                return row
            t0 = time.perf_counter()
            rf, setup_f, meta = self._fresh(q, y, s, qz, policy, cutinfo)
            rf, off_f, meta, accepted = self._compare(r, rf, setup_f, meta, self.prev, n)
            fallback = dict(meta, accepted=accepted, tracked_chi2_per_point=chi2pp)
            fallback['_timing'] = dict(meta['_timing'], fallback_s=time.perf_counter()-t0)
            if accepted:
                r, setup, off_age = rf, setup_f, off_f
        mode = 'fallback' if fallback is not None and fallback['accepted'] else 'track'
        row = self._solution_row(base, r, setup, mode, off_age, signals, events, fallback=fallback)
        if mode == 'fallback' or self._suspicious(n, r['chi2']/n, reference):
            row['fit']['reseed'] = _seed_json(good)
        return row

    def _compare(self, r, rf, setup_f, meta, prev, n):
        """Fresh solution rf against the tracked r: slots matched to the previous frame, accepted when better
        than the same-basin tolerance."""
        spec = self.spec
        codec_prev = spec.codec(prev['q_min'], prev['q_max'])
        u_ref = spec.convert(prev['anchor'] if prev.get('anchor') is not None else prev['u'], codec_prev, setup_f.codec)
        perm = match_identities(spec, [identity_points(spec, codec_prev, prev['u'])],
                                [identity_points(spec, setup_f.codec, rf['u'], rf['coef'])])
        rf, off_f, meta = self._settle_fresh(rf, setup_f, self._permuted_fit(setup_f, perm, rf), meta, u_ref, perm,
                                             self.off_age)
        accepted = bool(rf['chi2'] < r['chi2']-tolerance(n, min(rf['chi2'], r['chi2'])))
        return rf, off_f, meta, accepted

    @staticmethod
    def _permuted_fit(setup, perm, r):
        """The fit object of a fresh solution (populations of one class share type and constraints, so the
        same object evaluates the permuted solution)."""
        return r.pop('_fit', None) or setup.make_fit()

    def _size_bounds(self, setup, r, off_age):
        return sorted(c for c, _ in at_bound(setup, r['u'], r['coef'])
                      if c < 28 and c % 7 in SIZE_FIELDS and c//7 not in off_age)

    def _fallback_allowed(self, index, chi2pp):
        if self.last_fallback is None:
            return True
        last_index, last_chi2 = self.last_fallback
        return not (index-last_index <= self.cfg['fallback_cooldown'] and chi2pp <= COOLDOWN_FACTOR*last_chi2)

    # ------------------------------------------------------------------ batch stitching
    def retrack_from(self, i, seed_row, old_row):
        """Frame i tracked from the solution of another row (batch stitching: the frames before a chunk
        border followed backwards from a better chunk start). No signals, no fallback, no state change;
        the row keeps the frame entries of old_row."""
        t0 = time.perf_counter()
        self.cutter.restore(old_row.get('cut'))
        sides, cutinfo = self.cutter.sides(i)
        q, y, s = curve_from_sides(sides, self.cal['delta'], self.cal['ratio'], self.recipe['model']['sigma'])
        seed = seed_of(seed_row)
        with single_thread():
            r, setup, off_age, _ = self._track_from(q, y, s, self._qz(cutinfo), seed, seed['off_age'],
                                                    old_row.get('chi2_baseline'))
        row = self._solution_row({k: old_row[k] for k in BASE_KEYS}, r, setup, 'track', off_age, [], [])
        row['timing'] = dict(total_s=time.perf_counter()-t0)
        return roundtrip(row)

    # ------------------------------------------------------------------ rows
    def _solution_row(self, base, r, setup, mode, off_age, signals, events, fallback=None, first=None):
        spec = self.spec
        u = np.asarray(r['u'], float)
        coef = np.asarray(r['coef'], float)
        n = len(setup.y)
        pops_raw = populations(spec, setup, r)
        bounds = at_bound(setup, u, coef)
        names = [f'{coordinate_name(c, spec)}@{side}' for c, side in bounds if c >= 28 or c//7 not in off_age]
        weak, _ = fl.weakly_constrained([p for p in pops_raw if p['slot'] not in off_age], names)
        pops = []
        for pop in pops_raw:
            j, label = pop['slot'], pop['pop']
            on = j not in off_age
            item = dict(pop=label, slot=int(j), type=pop['type'], state='on' if on else 'off',
                        amplitude=float(coef[j]) if on else 0.0)
            for key in VALUE_KEYS+ERROR_KEYS:
                item[key] = pop.get(key) if on else None
            item['weak'] = [k for k in fl.POPULATION_KEYS if f'{label}.{k}' in weak] if on else []
            if not on:
                item['frozen'] = {k: pop.get(k) for k in FROZEN_KEYS if pop.get(k) is not None}
                item['off_frames'] = int(off_age[j])+1
            pops.append(item)
        flags = dict(fallback=fallback is not None, fallback_accepted=bool(fallback and fallback.get('accepted')),
                     pending_fallback=False, signals=list(signals), at_bound=names, weakly_constrained=weak,
                     weakly_constrained_reportable=fl.reportable_only(weak),
                     off=[spec.labels[j] for j in sorted(off_age)], events=list(events))
        fit = dict(u=u, coef=coef, q_min=float(setup.q[0]), q_max=float(setup.q[-1]), steps=r.get('steps'),
                   nfev=int(r['nfev']), evaluations=int(r['evaluations']), status=int(r['status']),
                   falling=bool(r['falling']), lm_runs=int(r.get('lm_runs', 1)),
                   off={str(j): int(a) for j, a in sorted(off_age.items())},
                   bound_state=self._size_bounds(setup, r, off_age))
        if r.get('anchor') is not None and not np.array_equal(r['anchor'], u):
            fit['anchor'] = np.asarray(r['anchor'], float)      # where an insignificant population was last significant
        row = dict(base, mode=mode, state='final', n_points=n, chi2=float(r['chi2']), chi2_per_point=float(r['chi2'])/n,
                   qz_nm_inv=float(setup.qz), populations=pops, background_B=float(coef[4]),
                   low_q=low_q(spec, setup, u, coef), flags=flags, fit=fit)
        meta = fallback if fallback is not None else first
        if meta is not None:
            meta = dict(meta)
            dwba = meta.pop('dwba', None)
            row['timing'] = meta.pop('_timing', {})
            row['fit']['fallback' if fallback is not None else 'first'] = meta
            if dwba is not None and (fallback is None or fallback.get('accepted')):
                row['dwba'] = dict(dwba, populations=[dict(pop=spec.labels[j], **dwba['populations'][j])
                                                      for j in spec.slots])
        return row

    # ------------------------------------------------------------------ live: asynchronous fallback
    def _submit(self, row, q, y, s, qz, cutinfo, policy, signals, good):
        dwba = self._dwba(cutinfo)
        kwargs = self.spec.fit1d_kwargs(qz, policy, dwba)
        handle = self.live.submit(dict(q=q, y=y, s=s, kwargs=kwargs))
        self.pending = dict(kind='fallback', handle=handle, q=q, y=y, s=s, qz=qz, row=row, policy=policy, dwba=dwba,
                            good=good, submitted=time.perf_counter())
        self.suppressed = []
        self.last_fresh = row['index']
        row['state'] = 'pending_fallback'
        row['flags']['pending_fallback'] = True
        row['flags']['signals'] = list(dict.fromkeys(list(row['flags']['signals'])+list(signals)))
        row['fit']['pending'] = dict(policy=policy)

    def _submit_retrack(self, start, stop, origin, step):
        """Frames [start, stop) processed again in the second process as a continuation of the stored rows
        before `start` (synchronous fallbacks there, as in review mode)."""
        restore = [self.history[j] for j in sorted(self.history) if j < start]
        task = (self.recipe, self.source.spec(), self.cal, int(start), int(stop), restore, None)
        self.pending = dict(kind='retrack', handle=self.live.retrack(task), start=int(start), stop=int(stop),
                            origin=int(origin), step=int(step), submitted=time.perf_counter())

    def poll(self, wait=False):
        """Rows to append for finished asynchronous work (a backfilled frame, or re-tracked frames); [] when
        nothing is ready. wait=True blocks until the pending work is done."""
        p = self.pending
        if p is None or not (wait or p['handle'].ready()):
            return []
        self.pending = None
        return self._finish_fallback(p) if p['kind'] == 'fallback' else self._finish_retrack(p)

    def _finish_fallback(self, p):
        t0 = time.perf_counter()
        provisional = roundtrip(p['row'])
        try:
            top, seconds = p['handle'].get()
        except Exception as exc:          # noqa: BLE001 - the provisional result stays, the failure is recorded
            new = dict(provisional, state='backfilled')
            new['fit'] = dict(provisional['fit'], fallback=dict(policy=p['policy'], accepted=False, asynchronous=True,
                                                                error=f'{type(exc).__name__}: {exc}'))
            new['fit'].pop('pending', None)
            new['flags'] = dict(provisional['flags'], fallback=True, fallback_accepted=False, pending_fallback=False)
            new['timing'] = dict(provisional.get('timing') or {}, backfill_s=time.perf_counter()-p['submitted'])
            self.history[new['index']] = new
            self.suppressed = []
            self.rebuild()
            return [new]
        spec = self.spec
        q, y, s, qz = p['q'], p['y'], p['s'], p['qz']
        n = len(q)
        with single_thread():
            setup = spec.setup(q, y, s, qz)
            fit = setup.make_fit()
            rf = self._lm(fit, setup, np.array(top['u'], float), None)
        meta = dict(policy=p['policy'], stages_run=top['stages_run'],
                    signals=top['signals'].get('fired', []), chi2_per_point=top['chi2_per_point'],
                    sphere_model=top.get('sphere_model', 'born'), asynchronous=True)
        if p['dwba'] is not None and top.get('born_reference') is not None:
            meta['dwba'] = self._dwba_block(top, p['dwba'])
        codec = spec.codec(provisional['fit']['q_min'], provisional['fit']['q_max'])
        u_ref = np.array(provisional['fit'].get('anchor') or provisional['fit']['u'], float)
        tracked_chi2 = provisional['chi2']
        # the provisional row defines the slots (later frames were already written in its order)
        perm = match_identities(spec, [identity_points(spec, codec, provisional['fit']['u'], provisional['fit']['coef'])],
                                [identity_points(spec, setup.codec, rf['u'], rf['coef'])])
        rf, off_f, meta = self._settle_fresh(rf, setup, fit, meta, u_ref, perm,
                                             {int(k): int(v) for k, v in provisional['fit']['off'].items()})
        accepted = bool(rf['chi2'] < tracked_chi2-tolerance(n, min(rf['chi2'], tracked_chi2)))
        wall = time.perf_counter()-p['submitted']
        index, last = provisional['index'], self.last_index
        meta = dict(meta, accepted=accepted, tracked_chi2_per_point=tracked_chi2/n)
        baseline = provisional.get('chi2_baseline')
        if accepted:
            meta['retrack_to'] = int(last)              # the frames written since are tracked again
            base = {k: provisional[k] for k in BASE_KEYS}
            if 'calibration' in provisional:
                base['calibration'] = provisional['calibration']
            new = self._solution_row(base, rf, setup, provisional['mode'] if provisional['mode'] == 'first' else 'fallback',
                                     off_f, provisional['flags']['signals'], provisional['flags']['events'], fallback=meta)
            if p['good'] is not None:                            # the next frame gets both starts
                new['fit']['reseed'] = _seed_json(p['good'][0])
        else:
            new = dict(provisional)
            block = meta.pop('dwba', None)
            new['fit'] = dict(provisional['fit'], fallback=meta)
            new['fit'].pop('pending', None)
            new['flags'] = dict(provisional['flags'], fallback=True, fallback_accepted=False)
            if block is not None:                       # the DWBA result of this frame, in the series' slot order
                new['dwba'] = dict(block, populations=[dict(pop=spec.labels[j], **block['populations'][j])
                                                       for j in spec.slots])
        new['flags']['pending_fallback'] = False
        new['flags']['fallback'] = True
        new['state'] = 'backfilled'
        new = roundtrip(new)
        # the causal annotations of the frame stay as they were written; the chi2 flag follows the new chi2
        new['chi2_baseline'] = baseline
        new['flags']['chi2_jump'] = bool(baseline is not None and new['chi2_per_point'] > self.cfg['chi2_jump']*baseline)
        new['flags']['change_points'] = provisional['flags'].get('change_points', [])
        for pop, old in zip(new['populations'], provisional['populations']):
            for key, value in old.items():
                if key.endswith('_smoothed') and key not in pop:
                    pop[key] = pop.get(key[:-len('_smoothed')]) if accepted else value
        new['timing'] = dict(provisional.get('timing') or {}, backfill_s=wall, fit1d_s=seconds,
                             backfill_polish_s=time.perf_counter()-t0)
        self.history[index] = new
        suppressed, self.suppressed = self.suppressed, []
        if accepted and last > index:
            self._submit_retrack(index+1, last+1, index, 1)
            return [new]
        self.rebuild()
        if not accepted:
            # a fallback that was not started while this one was pending and would be allowed now
            for j, chi2pp in suppressed:
                if j in self.history and self.history[j]['state'] == 'final' \
                        and not (j-index <= self.cfg['fallback_cooldown'] and chi2pp <= COOLDOWN_FACTOR*new['chi2_per_point']):
                    self._submit_retrack(j, last+1, j-1, 1)
                    break
        return [new]

    def _finish_retrack(self, p):
        try:
            rows = p['handle'].get()
        except Exception as exc:          # noqa: BLE001 - the rows written before stay valid
            self.async_errors.append(f"re-track of frames {p['start']}..{p['stop']-1}: {type(exc).__name__}: {exc}")
            self.rebuild()
            return []
        out = self._retracked(rows, p['origin'], p['submitted'])
        last, done = self.last_index, p['stop']-1
        if last-done > LOCAL_RETRACK_FRAMES and p['step'] < MAX_RETRACK_STEPS:
            out[-1]['retrack']['next'] = [done+1, int(last)]
            self._submit_retrack(done+1, last+1, p['origin'], p['step']+1)
            return out
        if last > done:
            # the few frames that arrived meanwhile are tracked again here, so that the re-track ends at the
            # current frame and this process continues from it
            out[-1]['retrack']['next'] = [done+1, int(last)]
            local = Tracker(self.recipe, self.source, self.cal, cutter=self.cutter)
            local.restore([self.history[j] for j in sorted(self.history) if j <= done])
            out += self._retracked([local.process(i) for i in range(done+1, last+1)], p['origin'], time.perf_counter())
        out[-1]['retrack']['last'] = True
        self.rebuild()
        return out

    def _retracked(self, rows, origin, since):
        """Rows of a re-track as lines of frames.jsonl (state 'backfilled', entry 'retrack')."""
        now = time.perf_counter()
        out = []
        for row in rows:
            old = self.history.get(row['index']) or {}
            row = dict(row, state='backfilled', retrack={'from': int(origin)})
            timing = dict(row.get('timing') or {}, retrack_s=now-self.written.get(row['index'], since))
            timing.update({k: v for k, v in (old.get('timing') or {}).items()
                           if k in ('arrival_s', 'done_s', 'latency_s', 'backlog')})
            row['timing'] = timing
            self.history[row['index']] = row
            out.append(row)
        return out

    def resume_live(self):
        """Live resume: asynchronous work that was pending when the run stopped is started again."""
        if self.live is None or not self.history or self.pending is not None:
            return
        last = self.last_index
        for j in sorted(self.history):
            row = self.history[j]
            if row.get('state') != 'pending_fallback':
                continue
            state = self.cutter.state()
            self.cutter.restore(row.get('cut'))
            sides, cutinfo = self.cutter.sides(j)
            self.cutter.restore(state)
            q, y, s = curve_from_sides(sides, self.cal['delta'], self.cal['ratio'], self.recipe['model']['sigma'])
            policy = (row['fit'].get('pending') or {}).get('policy') or \
                (self.first_policy if row['mode'] == 'first' else self.cfg['fallback_policy'])
            good = None
            if row['fit'].get('reseed'):      # suspicious for its points when the chi2 was not flagged
                good = (_seed_from_json(row['fit']['reseed']), not row['flags'].get('chi2_jump'))
            self._submit(roundtrip(row), q, y, s, self._qz(cutinfo), cutinfo, policy, [], good)
            self.last_fresh = max(self.last_fresh or 0, j)
            return
        for k in sorted(self.history, reverse=True):
            row = self.history[k]
            meta = (row.get('fit') or {}).get('fallback') or {}
            if row.get('state') != 'backfilled' or 'retrack' in row or not meta.get('accepted'):
                continue
            if meta.get('retrack_to', k) > k:
                done = [j for j in range(k+1, last+1) if (self.history.get(j) or {}).get('retrack', {}).get('from') == k]
                if not any(self.history[j]['retrack'].get('last') for j in done):
                    start = max(done)+1 if done else k+1
                    if start <= last:
                        self._submit_retrack(start, last+1, k, 1)
            return
