"""Run modes of series1d: live, review, batch, and a folder of independent curves.

frames.jsonl: line 1 is a header (recipe, calibration, input identity); every further line is one row
(schema gisaxs1d_series_row_v1), appended and flushed as soon as the frame is done. A frame can have more
than one line (live: 'backfilled' rows; batch: rows rewritten by a later stitch): the last line of a frame
index is the valid one.

review  one process, frames in order, synchronous fallbacks; the calibration may use a reference frame of
        the recipe (tracking.center 'frame N').
live    the same tracker, causal: nothing is read before its arrival time. Before the first frame only the
        setup is prepared (tables, network, the second process, for detector frames the cut geometry with
        the layout of the first file); frame i "arrives" at t0 + i / --replay-hz (replay of existing files;
        0 = as fast as possible) and the calibration is made on the first usable frame when it arrives
        (stored in that row). Fallbacks, the robust fit of the first frame and the re-tracking after an
        accepted backfill run in a second process (spawn) and are backfilled. Two processes in total.
batch   chunks of series.chunk_frames frames (default 96, independent of --workers) in --workers processes
        (this one and a spawn pool, one BLAS thread each). Chunk k > 0 starts series.warmup_frames (15)
        frames early with a fresh fit1d(policy fitting.chunk_start_policy = 'robust') and tracks up to its
        own frames: the warm-up builds the chunk-local state (chi2 baseline, fallback cooldown), its last
        series.overlap_frames (5) frames are the overlap that is compared with the previous chunk. The
        parent stitches the chunks in frame order (stitch_chunk):
          identities   Hungarian matching on (ln R, ln D) summed over the overlap frames (with no fitted
                       overlap frame: against the last fitted row before the chunk)
          reseed       the fresh start is worse than the rows before it by more than the same-basin
                       tolerance (median chi2 difference over all warm-up frames, so that one unusual frame
                       does not decide), or it is chi2-equivalent but another solution
                       (a reportable parameter differs by more than 3 sigma): the chunk is recomputed as a
                       continuation of the rows before it (continuity wins)
          switch       the fresh start is better by more than the tolerance: it is treated like an accepted
                       fallback. The frames before the border are tracked backwards from the chunk start
                       and replaced as far back as that track is better on balance (backward_block: one
                       contiguous block, back to the lowest cumulative chi2 difference); the first row of
                       the new solution becomes mode 'fallback' with the event 'basin_switch', so a series
                       has one switch where the two solutions meet instead of a jump at the chunk border
          state check  a chunk starts without the history of the series (skip-rule median, chi2 baseline,
                       fallback cooldown, second starts after suspicious frames). Every row of the chunk is
                       checked against the true state of the rows before it; from the first row whose
                       decisions (skipped or not, chi2_jump signal, fallback run or suppressed, suspicious)
                       differ, the rest of the chunk is recomputed sequentially from the true state.
        Results do not depend on --workers: every chunk is a pure function of (recipe, calibration, frames)
        and all stitching is done by the parent in frame order. The causal annotations (chi2 baseline,
        change points, smoothing) are those of the final sequence of rows.
resume  --resume continues an interrupted run from frames.jsonl: the tracker state is rebuilt by replaying
        the stored rows (Tracker.restore). In review and batch mode the result equals an uninterrupted run
        bit for bit, apart from the 'timing' entries (batch: every line carries the chunk whose stitch wrote
        it; the lines of an unfinished stitch are dropped and that chunk is recomputed). Live: a pending
        fallback is submitted again and an interrupted re-track is continued; the result is not bit-identical
        (backfill times depend on the wall clock). The header must match (recipe hash, input, mode).
curves  (--input-format dpdak|csv) one fit1d per file, no tracking; batch mode distributes the files.
"""
from __future__ import annotations

import copy
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np

from research.fit1d import schema as fit_schema
from research.series1d import recipe as rcp
from research.series1d.inputs import Cutter, InputError, NullCutter, calibrate, open_source, prepare_live, reopen
from research.series1d.model import ModelSpec, identity_points, match_identities, tolerance
from research.series1d.tracker import ROW_SCHEMA, Tracker, light_copy, roundtrip

HEADER_SCHEMA = 'gisaxs1d_series_header_v1'
VERSION = 'series1d 1.1.0 (M3; 2026-10-01)'
MAX_WORKERS = 64
STOP_NOT_BETTER = 8          # batch, backward re-track: frames in a row without a new lowest chi2 sum that end it
STOP_WORSE = 10.0            # ... or the chi2 sum risen by this many same-basin tolerances above its lowest value
SAME_SOLUTION_SIGMA = 3.0    # batch: chi2-equivalent chunk starts are the same solution within this many sigma


# ---------------------------------------------------------------------------- frames.jsonl
def read_frames(path):
    """(header or None, rows, byte length of the valid part). A cut-off last line is dropped."""
    path = Path(path)
    if not path.exists():
        return None, [], 0
    header, rows, valid = None, [], 0
    with open(path, 'rb') as f:
        for raw in f:
            if not raw.endswith(b'\n'):
                break
            try:
                obj = json.loads(raw.decode('utf-8'))
            except (ValueError, UnicodeDecodeError):
                break
            if header is None:
                if obj.get('schema') != HEADER_SCHEMA:
                    raise InputError(f'{path}: the first line is not a series1d header')
                header = obj
            else:
                rows.append(obj)
            valid += len(raw)
    return header, rows, valid


def latest_rows(rows):
    """The valid row of every frame index (the last line wins), in index order."""
    latest = {}
    for row in rows:
        latest[row['index']] = row
    return [latest[i] for i in sorted(latest)]


class FramesWriter:
    def __init__(self, path, header=None, keep_bytes=None):
        self.path = Path(path)
        if header is not None:
            self.f = open(self.path, 'w', encoding='utf-8')
            self.write(header)
        else:
            if keep_bytes is not None:
                with open(self.path, 'r+b') as f:
                    f.truncate(keep_bytes)
            self.f = open(self.path, 'a', encoding='utf-8')

    def write(self, obj):
        self.f.write(json.dumps(obj, allow_nan=False)+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


# ---------------------------------------------------------------------------- pool workers
_PROC: dict = {}


def _init_worker(tables_dir, model_dir=None):
    """Pool initializer: one BLAS thread, the parent's table directory, network weights loaded once.
    It must not raise (a failing initializer makes the pool respawn workers forever)."""
    try:
        from threadpoolctl import threadpool_limits
        _PROC['limit'] = threadpool_limits(1)
    except ImportError:          # pragma: no cover
        pass
    try:
        fit_schema.check_tables(tables_dir)
        from research.fit1d.api import get_fitter
        get_fitter(model_dir)
    except Exception as exc:     # noqa: BLE001 - reported by the first job
        _PROC['error'] = f'{type(exc).__name__}: {exc}'


def _check_worker():
    if _PROC.get('error'):
        raise RuntimeError(f'series1d worker could not start: {_PROC["error"]}')


def _probe(_=None):
    """Pool task for tests: pid, BLAS threads and table directory of a worker."""
    from research.fast_inference import tables
    threads = None
    try:
        from threadpoolctl import threadpool_info
        threads = max((int(i.get('num_threads', 1)) for i in threadpool_info()), default=None)
    except ImportError:          # pragma: no cover
        pass
    return dict(pid=os.getpid(), blas_threads=threads, tables_dir=str(tables.CACHE), error=_PROC.get('error'))


def _source_for(spec):
    key = json.dumps(spec, sort_keys=True, default=str)
    if _PROC.get('source_key') != key:
        _PROC['source_key'], _PROC['source'] = key, reopen(spec)
        _PROC.pop('cutter', None)
    return _PROC['source']


def _cutter_for(source, recipe, calibration):
    key = json.dumps([rcp.sha256(recipe), calibration], sort_keys=True, default=str)
    if _PROC.get('cutter_key') != key or 'cutter' not in _PROC:
        _PROC['cutter_key'], _PROC['cutter'] = key, Cutter(source, recipe, calibration)
    cutter = _PROC['cutter']
    cutter.restore(dict(alpha_f_deg=calibration.get('yoneda_alpha_f_deg')))       # a fresh chunk starts at the reference row
    return cutter


def chunk_task(task):
    """Rows of frames [start, stop) of one chunk: a fresh tracker, or one restored from `restore` rows.
    A pure function of its arguments (module-level: runs in spawn workers and in the parent alike)."""
    _check_worker()
    recipe, source_spec, calibration, start, stop, restore, first_policy = task
    source = _source_for(source_spec)
    tracker = Tracker(recipe, source, calibration, cutter=_cutter_for(source, recipe, calibration),
                      first_policy=first_policy)
    if restore:
        tracker.restore(restore)
    return [tracker.process(i) for i in range(start, stop)]


def fallback_task(payload):
    """Live fallback in the second process: fit1d started fresh -> (essentials of the best result, seconds)."""
    _check_worker()
    from research.fit1d.api import fit1d
    t0 = time.perf_counter()
    res = fit1d(payload['q'], payload['y'], payload['s'], **payload['kwargs'])
    top = res['results'][0]
    keys = ('u', 'coef', 'stages_run', 'signals', 'chi2_per_point', 'sphere_model', 'born_reference', 'populations')
    return {k: top.get(k) for k in keys}, time.perf_counter()-t0


def curve_task(task):
    """One independent curve file -> row."""
    _check_worker()
    recipe, source_spec, i = task
    return independent_row(recipe, _source_for(source_spec), i)


def make_pool(workers, model_dir=None):
    from multiprocessing import get_context
    return get_context('spawn').Pool(int(workers), initializer=_init_worker,
                                     initargs=(str(fit_schema.tables_dir()), model_dir))


class LiveFallback:
    """The second process of live mode (one spawn worker, started and warmed up before the first frame)."""

    def __init__(self, model_dir=None):
        self.pool = make_pool(1, model_dir)
        info = self.pool.apply(_probe)
        if info.get('error'):
            self.close()
            raise RuntimeError(f'series1d live worker could not start: {info["error"]}')

    def submit(self, payload):
        return self.pool.apply_async(fallback_task, (payload,))

    def retrack(self, task):
        """Frames processed again as a continuation of stored rows (chunk_task with restore rows)."""
        return self.pool.apply_async(chunk_task, (task,))

    def close(self):
        self.pool.terminate()
        self.pool.join()


# ---------------------------------------------------------------------------- independent curves
def independent_row(recipe, source, i):
    from research.fit1d.api import fit1d
    t0 = time.perf_counter()
    spec = ModelSpec.from_recipe(recipe, independent=True)
    number, name = source.info(i)
    base = dict(schema=ROW_SCHEMA, index=int(i), frame=int(number), file=name,
                time_s=recipe['series']['time_step_s']*(number-recipe['series']['time_origin_frame']),
                total_counts=None, cut={})
    try:
        q, y, s, meta = source.curve(i)
        res = fit1d(q, y, s, **spec.fit1d_kwargs(spec.qz or 0.0, recipe['fitting']['fallback_policy']),
                    source=dict(file=name))
    except (ValueError, OSError) as exc:
        return roundtrip(dict(base, mode='skipped', state='final', n_points=0, chi2=None, chi2_per_point=None,
                              qz_nm_inv=None, populations=[], background_B=None, low_q=None,
                              skip_reason=f'{type(exc).__name__}: {exc}',
                              flags=dict(fallback=False, fallback_accepted=False, pending_fallback=False, signals=[],
                                         at_bound=[], weakly_constrained=[], weakly_constrained_reportable=[],
                                         off=[], events=[], chi2_jump=False, change_points=[]), fit=None,
                              timing=dict(total_s=time.perf_counter()-t0)))
    top = res['results'][0]
    order = sorted(top['populations'], key=lambda p: (p['type'], p['inactive'], p['R_nm']))
    pops = []
    for k, pop in enumerate(order, 1):
        on = not pop['inactive']
        label = f'pop{k}'
        item = dict(pop=label, slot=int(pop['slot']), type=pop['type'], state='on' if on else 'off',
                    amplitude=pop['amplitude'] if on else 0.0)
        for key in ('R_nm', 'R_number_mean_nm', 'sigma_R_rel', 'D_nm', 'w_rel', 'w_nm', 'h_nm', 'sigma_h_rel',
                    'R_nm_err', 'sigma_R_rel_err', 'D_nm_err', 'w_rel_err', 'h_nm_err', 'sigma_h_rel_err'):
            item[key] = pop.get(key) if on else None
        weak = top['flags']['weakly_constrained']
        item['weak'] = [key for key in ('R_nm', 'sigma_R_rel', 'D_nm', 'w_rel', 'h_nm', 'sigma_h_rel')
                        if f"{pop['pop']}.{key}" in weak] if on else []
        pops.append(item)
    rename = {p['pop']: f'pop{k}' for k, p in enumerate(order, 1)}

    def relabel(names):
        return [rename.get(n.split('.')[0], n.split('.')[0])+n[len(n.split('.')[0]):] for n in names]

    flags = dict(fallback='S3' in top['stages_run'], fallback_accepted=top['winner']['stage'] in ('S3', 'S4'),
                 pending_fallback=False, signals=top['signals'].get('fired', []),
                 at_bound=relabel(top['flags']['at_bound']),
                 weakly_constrained=relabel([n for n in top['flags']['weakly_constrained']
                                             if n.split('.')[0] in rename and not
                                             next(p for p in order if p['pop'] == n.split('.')[0])['inactive']]),
                 off=[rename[p['pop']] for p in order if p['inactive']], events=[], chi2_jump=False,
                 change_points=[], quality=top['flags']['quality'], ood=top['flags']['ood'],
                 sigma_assumed=top['flags']['sigma_assumed'], fit1d_raised=top['flags']['raised'])
    flags['weakly_constrained_reportable'] = [n for n in flags['weakly_constrained']
                                              if n.split('.')[-1] in ('R_nm', 'D_nm')]
    low = top.get('low_q') or {}
    row = dict(base, mode='independent', state='final', n_points=top['n_points'], chi2=top['chi2'],
               chi2_per_point=top['chi2_per_point'], qz_nm_inv=res['settings']['qz_nm_inv'], populations=pops,
               background_B=top['background_B'],
               low_q=dict(form=res['settings']['lowq'], C=low.get('C'), sigma_res_nm_inv=low.get('sigma_res_nm_inv'),
                          nu=low.get('nu'), powerlaw_n=low.get('powerlaw_n')),
               flags=flags, composition=top['composition_text'], candidate_set=res['candidate_set'],
               indistinguishable=top['indistinguishable'],
               fit=dict(u=top['u'], coef=top['coef'], q_min=res['input']['q_min_nm_inv'],
                        q_max=res['input']['q_max_nm_inv'], stop_stage=top['stop_stage'],
                        stages_run=top['stages_run'], off={}, bound_state=[]),
               chi2_baseline=None, timing=dict(fit1d_s=res['timing']['total_seconds'], total_s=time.perf_counter()-t0))
    return roundtrip(row)


# ---------------------------------------------------------------------------- batch stitching
def permute_row(row, perm, spec):
    """A row with its populations relabelled: perm[target slot] = source slot."""
    if row['mode'] == 'skipped' or all(t == s for t, s in perm.items()):
        return row
    row = copy.deepcopy(row)
    labels = spec.labels
    target_of = {s: t for t, s in perm.items()}
    rename = {labels[s]: labels[t] for t, s in perm.items()}

    def relabel(names):
        out = []
        for name in names:
            head = name.split('.')[0].split(':')[0].split('@')[0]
            out.append(rename[head]+name[len(head):] if head in rename else name)
        return out

    def move(u, coef):
        u, coef = np.array(u, float), np.array(coef, float)
        u2, c2 = u.copy(), coef.copy()
        for target, source in perm.items():
            u2[7*target:7*target+7] = u[7*source:7*source+7]
            c2[target] = coef[source]
        return u2.tolist(), c2.tolist()

    def move_off(off):
        return {str(target_of[int(j)]): a for j, a in sorted(off.items(), key=lambda kv: target_of[int(kv[0])])}

    fit = row['fit']
    if fit.get('anchor'):
        fit['anchor'] = move(fit['anchor'], fit['coef'])[0]
    fit['u'], fit['coef'] = move(fit['u'], fit['coef'])
    fit['off'] = move_off(fit['off'])
    fit['bound_state'] = sorted(7*target_of[c//7]+c % 7 for c in fit['bound_state'])
    if fit.get('reseed'):
        seed = fit['reseed']
        if seed.get('anchor'):
            seed['anchor'] = move(seed['anchor'], seed['coef'])[0]
        seed['u'], seed['coef'] = move(seed['u'], seed['coef'])
        seed['off'] = move_off(seed['off'])
    for pop in row['populations']:
        pop['slot'] = int(target_of[pop['slot']])
        pop['pop'] = rename[pop['pop']]
    row['populations'].sort(key=lambda p: p['slot'])
    for key in ('at_bound', 'weakly_constrained', 'weakly_constrained_reportable', 'off', 'events', 'change_points'):
        if key in row['flags']:
            row['flags'][key] = relabel(row['flags'][key])
    if 'dwba' in row:
        for pop in row['dwba']['populations']:
            pop['pop'] = rename[pop['pop']]
        row['dwba']['populations'].sort(key=lambda p: p['pop'])
    return row


def _points(spec, row):
    codec = spec.codec(row['fit']['q_min'], row['fit']['q_max'])
    on = [0.0 if str(j) in row['fit']['off'] else 1.0 for j in range(4)]
    return identity_points(spec, codec, row['fit']['u'], on)


def _last_fitted(rows, before):
    for j in range(min(before, len(rows))-1, -1, -1):
        if rows[j]['mode'] != 'skipped':
            return rows[j]
    return None


def same_solution(pairs):
    """Two tracks of the same frames are one solution: the populations have the same state and no reportable
    parameter (R_nm, D_nm; not weakly constrained, with a 1-sigma) differs by more than SAME_SOLUTION_SIGMA
    combined sigma in the median over the frames."""
    z, states = {}, {}
    for a, b in pairs:
        for pa, pb in zip(a['populations'], b['populations']):
            states.setdefault(pa['pop'], []).append(pa['state'] == pb['state'])
            if pa['state'] != 'on' or pb['state'] != 'on':
                continue
            for key in ('R_nm', 'D_nm'):
                ea, eb = pa.get(f'{key}_err'), pb.get(f'{key}_err')
                if pa.get(key) is None or pb.get(key) is None or not ea or not eb \
                        or key in pa.get('weak', ()) or key in pb.get('weak', ()):
                    continue
                z.setdefault((pa['pop'], key), []).append(abs(pa[key]-pb[key])/float(np.hypot(ea, eb)))
    if any(np.mean(v) < 0.5 for v in states.values()):
        return False
    return all(float(np.median(v)) <= SAME_SOLUTION_SIGMA for v in z.values())


def backward_block(tracker, rows, candidates, start, seed):
    """Batch, better chunk start: the rows before the border replaced by the track of the chunk start.

    candidates: the chunk's rows of the frames before the border (slots already matched); earlier frames
    are tracked backwards, each from the candidate of the next frame (seed: the first fitted row of the
    chunk, used when there is no candidate). Going back from the border the chi2 differences (candidate -
    stored row) are summed; the block reaches back to the frame where that sum is lowest, i.e. as far as
    the new track is better on balance (a single frame where it is not better does not end it). A new
    lowest sum counts when it is lower than the last one by more than the same-basin tolerance; the walk
    ends at the start of the series, after STOP_NOT_BETTER frames without a new lowest sum, or when the sum
    has risen by more than STOP_WORSE tolerances above it. Returns the replacement rows in frame order."""
    cands = {r['index']: r for r in candidates if r['mode'] != 'skipped'}
    later = seed
    visited, total, lowest, keep, misses = [], 0.0, 0.0, 0, 0
    for j in range(start-1, -1, -1):
        old = rows[j]
        if old['mode'] == 'skipped':
            continue
        cand = cands.get(j)
        if cand is None:
            if later is None:
                break
            cand = tracker.retrack_from(j, later, old)
        later = cand
        visited.append(cand)
        total += cand['chi2']-old['chi2']
        tol = tolerance(old['n_points'], min(cand['chi2'], old['chi2']))
        if total < lowest-tol:
            lowest, keep, misses = total, len(visited), 0
        else:
            misses += 1
            if misses >= STOP_NOT_BETTER or total-lowest > STOP_WORSE*tol:
                break
    return [copy.deepcopy(r) for r in sorted(visited[:keep], key=lambda r: r['index'])]


def normalise_block(rows, block, start):
    """Rows of a backward block in the forward sense: off ages and on/off events follow the frame order."""
    before = _last_fitted(rows, block[0]['index'])
    for row in block:
        previous = before['fit']['off'] if before is not None else {}
        ages = {j: (int(previous[j])+1 if j in previous else 0) for j in sorted(row['fit']['off'], key=int)}
        row['fit']['off'] = {str(j): int(a) for j, a in ages.items()}
        events = [f'retracked_backward(chunk start at frame index {start})']
        for pop in row['populations']:
            was_off = str(pop['slot']) in previous
            if pop['state'] == 'off':
                pop['off_frames'] = ages[str(pop['slot'])]+1
                if before is not None and not was_off:
                    events.append(f"{pop['pop']}:off")
            elif was_off:
                events.append(f"{pop['pop']}:on")
        row['flags']['events'] = events
        if row['mode'] == 'first':                              # the fresh fit of the chunk start
            row['mode'] = 'track'
            row['fit']['chunk_start'] = row['fit'].pop('first', None)
        before = row
    return block


def mark_switch(row, before, start, policy, old=None):
    """The first row of the better solution: reported like an accepted fallback (mode, flags, counted).
    before: the last fitted row of the old solution (None: the block reaches the start of the series);
    old: the row this one replaces."""
    row['flags']['events'] = list(row['flags']['events'])+[f'basin_switch(better chunk start at frame index {start})']
    if before is None:
        if old is not None and old['mode'] == 'first':          # the whole beginning of the series was replaced
            row['mode'] = 'first'
            row['fit']['first'] = dict(old['fit'].get('first') or {}, superseded_by_chunk_start=int(start))
        return row
    if not row['flags']['fallback']:
        row['fit']['fallback'] = dict(policy=policy, accepted=True, origin='batch_chunk_start',
                                      chunk_start_index=int(start), chi2_per_point=row['chi2_per_point'],
                                      tracked_chi2_per_point=None if old is None else old['chi2_per_point'],
                                      signals=[], stages_run=[])
    row['mode'] = 'fallback'
    row['flags']['fallback'] = True
    row['flags']['fallback_accepted'] = True
    return row


def stitch_chunk(spec, rows, chunk_rows, start, overlap_frames):
    """Decision for a chunk (warm-up and overlap frames first) against the final rows before it; the
    comparison uses the overlap_frames frames before `start`.
    Returns (own rows relabelled or None for a re-seed, rows before the border relabelled, note)."""
    own = [r for r in chunk_rows if r['index'] >= start]
    overlap = [r for r in chunk_rows if r['index'] < start]
    shared = [(rows[r['index']], r) for r in overlap
              if r['mode'] != 'skipped' and rows[r['index']]['mode'] != 'skipped']
    pairs = [(a, b) for a, b in shared if b['index'] >= start-overlap_frames] or shared
    note = dict(start=int(start), overlap_frames_compared=len(pairs), frames_compared=len(shared), permutation=[],
                decision='keep', reseed=False)
    if not pairs:
        # no fitted frame in common: identities from the last fitted row before the chunk
        ref, new = _last_fitted(rows, start), next((r for r in chunk_rows if r['mode'] != 'skipped'), None)
        perm = {j: j for j in spec.slots}
        if ref is not None and new is not None:
            perm = match_identities(spec, [_points(spec, ref)], [_points(spec, new)])
        note.update(decision='no_overlap', permutation=[[int(t), int(s)] for t, s in sorted(perm.items()) if t != s])
        return [permute_row(r, perm, spec) for r in own], [], note
    perm = match_identities(spec, [_points(spec, a) for a, _ in pairs], [_points(spec, b) for _, b in pairs])
    # which track is better: the median over all frames the two tracks share (warm-up and overlap)
    pairs = [(a, permute_row(b, perm, spec)) for a, b in shared]
    delta = float(np.median([b['chi2']-a['chi2'] for a, b in pairs]))
    n = int(np.median([a['n_points'] for a, _ in pairs]))
    reference = float(np.median([a['chi2'] for a, _ in pairs]))
    tol = tolerance(n, reference)
    note.update(chi2_difference_median=delta, tolerance=tol,
                permutation=[[int(t), int(s)] for t, s in sorted(perm.items()) if t != s])
    if delta > tol:
        note.update(decision='reseed', reseed=True, reason='the fresh start is worse than the continuation')
        return None, [], note
    if delta >= -tol:
        if same_solution(pairs):
            return [permute_row(r, perm, spec) for r in own], [], note
        note.update(decision='reseed', reseed=True,
                    reason='the fresh start is a chi2-equivalent but different solution: the continuation is kept')
        return None, [], note
    note.update(decision='switch', basin_switch=True)
    return [permute_row(r, perm, spec) for r in own], [permute_row(r, perm, spec) for r in overlap], note


def state_mismatch(state, row, first_own, local_seed):
    """Why the decisions taken for a chunk row in its chunk-local state differ from those of a sequential
    run (state: tracker replaying the final rows before it), or None when they agree.
    first_own: the row is the first one of the chunk; local_seed: the chunk had a second start for it."""
    if row['mode'] == 'skipped':
        if row.get('skip_kind') == 'median' and state._skip_reason(row['total_counts']) is None:
            return 'skip rule'
        return None
    if state._skip_reason(row['total_counts']) is not None:
        return 'skip rule'
    if row['mode'] == 'first':
        return None
    cfg = state.cfg
    index = row['index']
    tracked = (row['fit'].get('fallback') or {}).get('tracked_chi2_per_point')
    tracked = row['chi2_per_point'] if tracked is None else tracked
    reference = state.chi2_reference()
    signals = row['flags']['signals']
    if ('chi2_jump' in signals) != bool(reference is not None and tracked > cfg['chi2_jump']*reference):
        return 'chi2 baseline'
    if ('chi2_drift' in signals) != state._drifted(reference):
        return 'chi2 drift'
    every = cfg['refresh_every']
    refresh = bool(every) and state.last_fresh is not None and index-state.last_fresh >= every
    if ('refresh' in signals) != refresh:
        return 'refresh'
    if signals and bool(row['flags']['fallback']) != bool(refresh or state._fallback_allowed(index, tracked)):
        return 'fallback cooldown'
    if bool(row['fit'].get('reseed')) != bool(row['mode'] == 'fallback'
                                              or state._suspicious(row['n_points'], row['chi2_per_point'], reference)):
        return 'suspicious frame'
    if first_own and (state.reseed is not None) != bool(local_seed):
        return 'second start'
    return None


# ---------------------------------------------------------------------------- run
def _header(recipe, mode, source, calibration):
    return dict(schema=HEADER_SCHEMA, series1d_version=VERSION, mode=mode, recipe_sha256=rcp.sha256(recipe),
                recipe=recipe, source=source.describe(), calibration=calibration, created_unix=time.time())


def _check_resume(header, recipe, mode, source, path):
    problems = []
    if header.get('recipe_sha256') != rcp.sha256(recipe):
        problems.append('the recipe differs')
    if header.get('mode') != mode:
        problems.append(f"the mode differs ({header.get('mode')} there, {mode} now)")
    if header.get('source') != roundtrip(source.describe()):
        problems.append('the input files differ')
    if problems:
        raise InputError(f'--resume: {path} was written by another run ({"; ".join(problems)}); '
                         'use a new --out directory or remove --resume to start again')


def warm_up(recipe):
    """Load the network weights and tables before the first frame (not counted as frame time)."""
    from research.fit1d.api import get_fitter
    fit_schema.check_tables()
    get_fitter(recipe['model'].get('model_dir'))


def run(files, recipe, *, mode='review', out=None, workers=1, replay_hz=10.0, resume=False, input_format='auto',
        frames=None, stop_after=None, quiet=True, finalize=True):
    """Run a series; returns the summary dict (also written to out/summary.json).

    files: series npz, directory or glob; recipe: path or dict; stop_after: stop after this many frames
    (tests: an interrupted run to be continued with resume=True)."""
    from research.series1d import outputs
    t_run = time.perf_counter()
    if isinstance(recipe, (str, os.PathLike)):
        recipe_path = Path(recipe).expanduser()
        recipe = rcp.load(recipe_path)
    else:
        recipe = rcp.normalise(recipe)
    workers = int(max(1, min(int(workers or 1), MAX_WORKERS)))
    if mode not in rcp.MODES:
        raise rcp.RecipeError(f'mode must be one of {rcp.MODES}, got {mode!r}')
    source = open_source(files, input_format, frames, recipe['model']['q_unit'])
    rcp.check_mode(recipe, mode, source.kind)
    out = Path(out).expanduser() if out is not None else Path.cwd()/'series1d_out'
    out.mkdir(parents=True, exist_ok=True)
    frames_path = out/'frames.jsonl'
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    warm_up(recipe)

    header, stored, valid = (read_frames(frames_path) if resume else (None, [], 0))
    info = dict(mode=mode, workers=workers, resumed_frames=0, interrupted=False, live=None, batch=None, notes=[])
    from research.fit1d.controller import effective_workers
    pools_ok = effective_workers(2) == 2
    if not pools_ok and (workers > 1 or mode == 'live'):
        # inside a child process, or the main script is not a file ("python -"): spawn workers cannot start
        workers = 1
        info['workers'] = 1
        info['notes'].append('no worker process could be started here (child process, or a main script that is '
                             'not a file): everything ran in one process' +
                             (' and live fallbacks ran synchronously' if mode == 'live' else ''))
    live = None
    tracked_live = mode == 'live' and source.kind != 'curves'
    try:
        if tracked_live and pools_ok:
            live = LiveFallback(recipe['model'].get('model_dir'))
        t_start = time.perf_counter()
        cutter = None
        if header is not None:
            _check_resume(header, recipe, mode, source, frames_path)
            calibration = header['calibration']
        else:
            if tracked_live:
                # only the setup, nothing that depends on the counts of a frame: the calibration is made on
                # the first usable frame when it arrives (Tracker.process)
                calibration, cutter = prepare_live(source, recipe)
            else:
                calibration, cutter = calibrate(source, recipe, with_cutter=True)
            calibration = roundtrip(calibration)
            header = roundtrip(_header(recipe, mode, source, calibration))
            stored, valid = [], None
        if tracked_live:
            t_start = time.perf_counter()                   # frame i arrives at t_start + i / replay_hz
        if source.kind == 'curves':
            rows = _run_curves(recipe, source, mode, workers, frames_path, header, stored, valid, stop_after, info, say)
        elif mode == 'batch':
            rows = _run_batch(recipe, source, calibration, workers, frames_path, header, stored, valid, stop_after,
                              info, say)
        else:
            rows = _run_sequential(recipe, source, calibration, cutter, live, replay_hz if mode == 'live' else 0.0,
                                   t_start, frames_path, header, stored, valid, stop_after, info, say)
    finally:
        if live is not None:
            live.close()
    info['wall_seconds'] = time.perf_counter()-t_run
    info['processing_seconds'] = time.perf_counter()-t_start
    if info.get('calibration') is not None:                 # live: completed on the first usable frame
        header = dict(header, calibration=info.pop('calibration'))
    summary = outputs.summarise(recipe, header, rows, info, n_total=len(source))
    if finalize:
        outputs.write_all(out, recipe, header, rows, summary)
    return summary


def _open_writer(frames_path, header, stored, valid):
    if valid is None:
        return FramesWriter(frames_path, header=header)
    return FramesWriter(frames_path, keep_bytes=valid)


def _run_sequential(recipe, source, calibration, cutter, live, hz, t_start, frames_path, header, stored, valid,
                    stop_after, info, say):
    tracker = Tracker(recipe, source, calibration, cutter=cutter, live=live)
    rows = {r['index']: r for r in latest_rows(stored)}
    done = tracker.restore(stored) if stored else 0
    info['resumed_frames'] = done
    writer = _open_writer(frames_path, header, stored, valid)
    latency, backlog, backfill, retrack = [], [], [], []

    def backfilled(wait=False):
        for new in tracker.poll(wait):
            rows[new['index']] = new
            if 'retrack' in new:
                retrack.append(new['timing']['retrack_s'])
            else:
                backfill.append(new['timing']['backfill_s'])
            writer.write(new)

    try:
        n = len(source)
        if live is not None and stored:
            tracker.resume_live()                           # pending fallback / interrupted re-track
        for i in range(done, n):
            if stop_after is not None and i >= stop_after:
                info['interrupted'] = True
                break
            arrival = t_start+(i-done)/hz if hz > 0 else None
            while True:
                if live is not None:
                    backfilled()
                if arrival is None:
                    break
                wait = arrival-time.perf_counter()
                if wait <= 0:
                    break
                time.sleep(min(wait, 0.002))
            row = tracker.process(i)
            if arrival is not None:
                done_t = time.perf_counter()
                arrived = min(int(np.floor((done_t-t_start)*hz))+1, n-done)
                row['timing'].update(arrival_s=arrival-t_start, done_s=done_t-t_start, latency_s=done_t-arrival,
                                     backlog=max(arrived-(i-done+1), 0))
                latency.append(row['timing']['latency_s'])
                backlog.append(row['timing']['backlog'])
            rows[row['index']] = row
            writer.write(row)
            if not i % 50:
                say(f'frame {i+1}/{n}: {row["mode"]} chi2/pt {row["chi2_per_point"]}')
        while live is not None and tracker.pending is not None and not info['interrupted']:
            backfilled(wait=True)
    finally:
        writer.close()
    if tracker.cal != calibration:
        info['calibration'] = tracker.cal
    info['notes'].extend(tracker.async_errors)
    if tracker.cal.get('at_search_edge'):
        info['notes'].append('the left/right symmetry search for the qy origin ended at the edge of its range: '
                             'check experiment_setup.direct_beam_col (the centre start value)')
    if live is not None:
        info['live'] = dict(replay_hz=hz, processes=2, backfills=len(backfill),
                            backfill_p95_s=float(np.percentile(backfill, 95)) if backfill else None,
                            backfill_max_s=float(np.max(backfill)) if backfill else None,
                            retracked_rows=len(retrack),
                            retrack_p95_s=float(np.percentile(retrack, 95)) if retrack else None,
                            retrack_max_s=float(np.max(retrack)) if retrack else None)
        if hz > 0 and latency:
            info['live'].update(frames=len(latency), latency_p50_s=float(np.percentile(latency, 50)),
                                latency_p95_s=float(np.percentile(latency, 95)), latency_max_s=float(np.max(latency)),
                                max_backlog=int(np.max(backlog)),
                                frames_over_300ms=int(np.sum(np.asarray(latency) > 0.3)))
    return [rows[i] for i in sorted(rows)]


def _run_curves(recipe, source, mode, workers, frames_path, header, stored, valid, stop_after, info, say):
    rows = latest_rows(stored)
    done = len(rows)
    info['resumed_frames'] = done
    writer = _open_writer(frames_path, header, stored, valid)
    n = len(source) if stop_after is None else min(len(source), stop_after)
    info['interrupted'] = n < len(source)
    pool = None
    try:
        todo = list(range(done, n))
        if mode == 'batch' and workers > 1 and len(todo) > 1:
            pool = make_pool(min(workers, len(todo)), recipe['model'].get('model_dir'))
            results = pool.imap(curve_task, [(recipe, source.spec(), i) for i in todo])
        else:
            results = (independent_row(recipe, source, i) for i in todo)
        for row in results:
            rows.append(row)
            writer.write(row)
            say(f'{row["file"]}: {row["mode"]} chi2/pt {row["chi2_per_point"]}')
    finally:
        writer.close()
        if pool is not None:
            pool.terminate()
            pool.join()
    return rows


def _chunk_of(row):
    return (row.get('batch') or {}).get('chunk')


def _run_batch(recipe, source, calibration, workers, frames_path, header, stored, valid, stop_after, info, say):
    spec = ModelSpec.from_recipe(recipe)
    cfg = recipe['fitting']
    n = len(source)
    size, overlap = recipe['series']['chunk_frames'], recipe['series']['overlap_frames']
    warmup = max(overlap, recipe['series']['warmup_frames'])
    starts = list(range(0, n, size))
    # resume: a chunk is complete when the line of its last frame, written by its own stitch, is stored; the
    # lines of a later (unfinished) stitch - own rows and rows of earlier frames it rewrote - are dropped
    done_chunks = 0
    for k, start in enumerate(starts):
        last = min(start+size, n)-1
        if not any(r['index'] == last and _chunk_of(r) == k for r in stored):
            break
        done_chunks = k+1
    kept = [r for r in stored if _chunk_of(r) is not None and _chunk_of(r) < done_chunks]
    if stored and len(kept) != len(stored):
        writer = FramesWriter(frames_path, header=header)       # rewrite the kept lines in their order
        for row in kept:
            writer.write(row)
    else:
        writer = _open_writer(frames_path, header, stored, valid)
    rows = latest_rows(kept)
    info['resumed_frames'] = len(rows)
    # the true state of the series (skip median, chi2 baseline, cooldown, annotations) follows the final rows
    state = Tracker(recipe, source, calibration, cutter=NullCutter())
    state.restore(rows)
    todo = [k for k in range(done_chunks, len(starts))
            if stop_after is None or starts[k] < stop_after]
    info['interrupted'] = bool(stop_after is not None and stop_after < n)
    tasks = [(recipe, source.spec(), calibration, max(0, starts[k]-warmup) if k else 0, min(starts[k]+size, n), None,
              cfg['chunk_start_policy'] if k else None) for k in todo]
    stitches = []
    counts = dict(reseeded_chunks=0, basin_switches=0, state_mismatches=0, frames_retracked_backward=0,
                  frames_recomputed=0)
    helper = []

    def parent_tracker():
        if not helper:
            helper.append(Tracker(recipe, source, calibration, cutter=_cutter_for(source, recipe, calibration)))
        return helper[0]

    pool = None
    t0 = time.perf_counter()
    try:
        if workers > 1 and len(tasks) > 1:
            # --workers processes in total: the pool computes the later chunks while this process computes
            # the first one (no waiting for the pool to start)
            pool = make_pool(min(workers-1, len(tasks)-1), recipe['model'].get('model_dir'))
            later = pool.imap(chunk_task, tasks[1:])
            results = itertools.chain([chunk_task(tasks[0])], later)
        else:
            results = (chunk_task(task) for task in tasks)
        for k, chunk_rows in zip(todo, results):
            start, stop = starts[k], min(starts[k]+size, n)
            block, switch = [], False
            if k == 0:
                own, note, local_seed, verify = chunk_rows, dict(start=0, decision='first', reseed=False, permutation=[]), None, False
            else:
                own, candidates, note = stitch_chunk(spec, rows, chunk_rows, start, overlap)
                before_own = next((r for r in reversed(chunk_rows) if r['index'] < start and r['mode'] != 'skipped'), None)
                local_seed = bool(before_own and before_own['fit'].get('reseed'))
                verify = own is not None
                if own is None:                                 # continuation of the rows before the chunk
                    counts['reseeded_chunks'] += 1
                    counts['frames_recomputed'] += stop-start
                    own = chunk_task((recipe, source.spec(), calibration, start, stop, list(rows), None))
                elif note['decision'] == 'switch':
                    switch = True
                    counts['basin_switches'] += 1
                    seed = next((r for r in candidates+own if r['mode'] != 'skipped'), None)
                    block = backward_block(parent_tracker(), rows, candidates, start, seed)
            if block:
                block = normalise_block(rows, block, start)
                first = block[0]['index']
                mark_switch(block[0], _last_fitted(rows, first), start, cfg['chunk_start_policy'], old=rows[first])
                counts['frames_retracked_backward'] += len(block)
                note.update(switch_at=int(first), frames_replaced=len(block))
                replaced = {r['index']: r for r in block}
                state.restore(rows[:first])
                for j in range(first, start):
                    row = dict(replaced[j]) if j in replaced else light_copy(rows[j])
                    state._advance(row)                         # the annotations of the final sequence
                    if j in replaced or row != rows[j]:
                        row['batch'] = dict(chunk=k)
                        rows[j] = row
                        writer.write(row)
                state.history = {r['index']: r for r in rows}
            position = 0
            while position < len(own):
                row = own[position]
                why = state_mismatch(state, row, position == 0, local_seed) if verify else None
                if why is not None:
                    # the chunk-local state led to another decision than the series state: the rest of the
                    # chunk is recomputed as a continuation of the final rows
                    verify = False
                    counts['state_mismatches'] += 1
                    counts['frames_recomputed'] += stop-row['index']
                    note.update(state_mismatch=dict(index=int(row['index']), reason=why))
                    if switch and not block:                    # nothing of the better start is left
                        switch = False
                        note.update(decision='switch_cancelled')
                    own = own[:position]+chunk_task((recipe, source.spec(), calibration, row['index'], stop,
                                                     list(rows), None))
                    continue
                row = dict(row, batch=dict(chunk=k))
                if position == 0 and k and row['mode'] != 'skipped' and verify:
                    row['flags'] = dict(row['flags'], events=list(row['flags']['events'])+['chunk_start(fresh)'])
                if switch and not block and row['mode'] != 'skipped':
                    switch = False                              # no row before the border was replaced: the switch is here
                    row['flags'], row['fit'] = dict(row['flags']), dict(row['fit'])
                    mark_switch(row, _last_fitted(rows, row['index']), start, cfg['chunk_start_policy'])
                    note.update(switch_at=int(row['index']), frames_replaced=0)
                state._advance(row)
                state.history[row['index']] = row
                rows.append(row)
                writer.write(row)
                position += 1
            stitches.append(note)
            say(f'chunk {k+1}/{len(starts)} (frames {start}..{stop-1}) done, {note["decision"]}')
    finally:
        writer.close()
        if pool is not None:
            pool.terminate()
            pool.join()
    info['batch'] = dict(chunk_frames=size, overlap_frames=overlap, warmup_frames=warmup, chunks=len(starts),
                         chunks_computed=len(todo),
                         stitches=stitches, pool_seconds=time.perf_counter()-t0, **counts)
    return rows
