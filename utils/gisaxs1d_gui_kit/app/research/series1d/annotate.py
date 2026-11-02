"""Causal annotation of the rows of a series: chi2 baseline, change points, continuity smoothing.

Everything here is a pure function of the sequence of rows seen so far (frames <= t), so it can be replayed
from frames.jsonl (--resume) and re-run over the merged rows of a batch run.

chi2 baseline   median chi2/point of the last 10 fitted frames that were not flagged chi2_jump (at least 3
                needed). flags.chi2_jump = chi2/point > fitting.chi2_jump (default 2) x baseline. Three
                flagged frames in a row are taken as a new regime: they become the baseline (they keep
                their flag), so a lasting change does not flag the rest of the series.
change points   per reportable parameter (R_nm, D_nm) of every population that is 'on', not weakly
                constrained and has a 1-sigma: z = (value - median of the last 5 accepted values) / scale,
                scale = max(1-sigma of the frame, scatter of the last 20 accepted values) sqrt(1 + 1.57/n);
                the scatter is 1.4826 MAD / sqrt(2) of their successive differences, so that a slow ramp does
                not widen it (with the MAD of the values, 7 % of 6-sigma steps on ramps were missed).
                A change point is flagged (flags.change_points, e.g. 'pop1.R_nm') when |z| exceeds the
                single-frame threshold (default 4.5) or exceeds the two-frame threshold (default 3) in two
                consecutive frames with the same sign; the history of that parameter then restarts.
                A lone deviation above the two-frame threshold is not added to the history.
continuity      (fitting.continuity = {alpha, feedback}) exponential smoothing value_smoothed of the
                parameters flagged weakly constrained in the current frame, only on frames whose chi2 is
                normal (no chi2_jump): ln-space for lengths, linear for relative widths,
                s_t = (1 - alpha) s_(t-1) + alpha x_t. Well-constrained parameters, frames with a chi2 jump,
                the first frame after 'off' and change points reset the smoother to the raw value, so
                value_smoothed == value there. The raw value is always reported unchanged.
"""
from __future__ import annotations

import math

import numpy as np

WINDOW = 10
MIN_BASELINE = 3
REGIME_FRAMES = 3
CP_KEYS = ('R_nm', 'D_nm')
SMOOTH_KEYS = ('R_nm', 'D_nm', 'sigma_R_rel', 'w_rel', 'h_nm', 'sigma_h_rel')
CP_MIN_HISTORY = 4


class Annotator:
    def __init__(self, fitting):
        self.factor = float(fitting['chi2_jump'])
        self.z_two, self.z_one = (float(v) for v in fitting['change_point_sigma'])
        self.continuity = fitting.get('continuity')
        self.base = []            # chi2/point of the last unflagged frames
        self.run = []             # chi2/point of the current run of flagged frames
        self.cp = {}              # parameter -> dict(values=[...], z=last suspect z or None, held=value)
        self.smooth = {}          # parameter -> smoothed value

    def baseline(self):
        return float(np.median(self.base)) if len(self.base) >= MIN_BASELINE else None

    def is_jump(self, chi2_per_point):
        b = self.baseline()
        return bool(b is not None and chi2_per_point is not None and chi2_per_point > self.factor*b)

    # ------------------------------------------------------------------
    def annotate(self, row):
        """Adds chi2_baseline, flags.chi2_jump, flags.change_points and the *_smoothed values to a row
        (in place; fields written by an earlier annotation are overwritten) and advances the state."""
        flags = row['flags']
        row['chi2_baseline'] = self.baseline()
        flags['chi2_jump'] = False
        flags['change_points'] = []
        if row['mode'] in ('skipped', 'independent') or row.get('chi2_per_point') is None:
            return row
        chi = float(row['chi2_per_point'])
        jump = self.is_jump(chi)
        flags['chi2_jump'] = jump
        if jump:
            self.run.append(chi)
            if len(self.run) >= REGIME_FRAMES:
                self.base, self.run = list(self.run), []
        else:
            self.run = []
            self.base = (self.base+[chi])[-WINDOW:]
        for pop in row['populations']:
            for key in CP_KEYS:
                name = f"{pop['pop']}.{key}"
                if self._change_point(name, pop, key):
                    flags['change_points'].append(name)
            if self.continuity is not None:
                for key in SMOOTH_KEYS:
                    self._smooth(f"{pop['pop']}.{key}", pop, key, jump, f"{pop['pop']}.{key}" in flags['change_points'])
        return row

    def _change_point(self, name, pop, key):
        value, err = pop.get(key), pop.get(f'{key}_err')
        if pop.get('state') != 'on' or value is None:
            self.cp.pop(name, None)                       # the history restarts when the population returns
            return False
        if err is None or not err > 0 or key in pop.get('weak', ()):
            return False
        h = self.cp.setdefault(name, dict(values=[], z=None, held=None))
        values = h['values']
        if len(values) < CP_MIN_HISTORY:
            values.append(float(value))
            return False
        recent = values[-5:]
        ref = float(np.median(recent))
        # frame-to-frame scatter from the successive differences of the accepted values (a slow trend does
        # not count as scatter): 1.4826 MAD(differences) / sqrt(2)
        steps = np.diff(np.asarray(values[-2*WINDOW:], float))
        spread = 1.4826*float(np.median(np.abs(steps-np.median(steps))))/math.sqrt(2.0) if len(steps) >= 3 else 0.0
        scale = max(float(err), spread)*math.sqrt(1.0+1.57/len(recent))
        z = (float(value)-ref)/scale
        two = h['z'] is not None and abs(z) > self.z_two and abs(h['z']) > self.z_two and z*h['z'] > 0
        if abs(z) > self.z_one or two:
            h['values'] = ([h['held']] if two and h['held'] is not None else [])+[float(value)]
            h['z'], h['held'] = None, None
            return True
        if abs(z) > self.z_two:
            h['z'], h['held'] = z, float(value)           # suspect: wait for the next frame
        else:
            h['z'], h['held'] = None, None
            h['values'] = (values+[float(value)])[-2*WINDOW:]
        return False

    def _smooth(self, name, pop, key, jump, changed):
        value = pop.get(key)
        if pop.get('state') != 'on' or value is None:
            self.smooth.pop(name, None)
            pop[f'{key}_smoothed'] = None
            return
        value = float(value)
        weak = key in pop.get('weak', ())
        previous = self.smooth.get(name)
        if weak and not jump and not changed and previous is not None:
            a = float(self.continuity['alpha'])
            if key.endswith('_nm') and value > 0 and previous > 0:
                smoothed = math.exp((1-a)*math.log(previous)+a*math.log(value))
            else:
                smoothed = (1-a)*previous+a*value
        else:
            smoothed = value
        self.smooth[name] = smoothed
        pop[f'{key}_smoothed'] = smoothed
