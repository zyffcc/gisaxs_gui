"""Time groups, and in each the minima and the phase-locked fit of both sides.

The usable windows (on the sample, with counts in the band) are put into groups of ``group_seconds`` from the first
of them; each side's counts and exposure are summed over a group. A first pass fits each cut's baseline and finds
the static detector rows (``minima.static_cells``), which are then left out; the second pass locates the minima
(``minima.find_minima``) and makes the phase-locked fit (``phase.fit``).

A minimum of side 0 and one of side 1 are the same when they lie within ``SIDE_TOL_DEG`` (or three times their
combined error); they are then averaged (``sides`` "both"). One within ``AGREE_DEG`` of the other side's but beyond
that tolerance is kept as ``sides`` "disagree" (it gives no height); one without a partner is a one-side minimum
("0" or "1").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .minima import AGREE_DEG, SD_FLOOR_DEG, baseline, find_minima, static_cells
from .optics import q_inside, qz_vacuum
from .phase import fit as phase_fit
from .phase import AC_MIN_PERIODS, heights_grid, side_check, trend_accept

SIDE_TOL_DEG = 0.012


@dataclass
class Group:
    index: int
    windows: np.ndarray
    time: float
    t_lo: float
    t_hi: float
    raw_frames: int
    found: dict = field(default_factory=dict)
    phase_sides: dict = field(default_factory=dict)
    phase: object = None
    """(H, error) of the accepted phase-locked fit, else None."""
    phase_note: str = ""
    detections: list = field(default_factory=list)
    cuts: dict = field(default_factory=dict)
    """side -> (counts, exposure) of the band per cell"""
    height: object = None
    status: str = ""
    bound_nm: object = None


@dataclass
class Detection:
    group: int
    time: float
    af: float
    sd: float
    sides: str
    per_side: dict
    q: float = float("nan")
    lnq: float = float("nan")
    sd_lnq: float = float("inf")
    track: int = -1
    order: int = 0
    accepted: bool = False
    static: bool = False

    @property
    def usable(self) -> bool:
        """Its position may give a height (both sides agree, or one side only)."""
        return self.sides != "disagree"


def make_groups(scan, group_seconds: float) -> list:
    usable = np.flatnonzero(scan.usable)
    if len(usable) == 0:
        return []
    start = float(scan.time[usable[0]])
    keys = np.floor((scan.time[usable] - start) / group_seconds + 1e-9).astype(int)
    groups = []
    for key in np.unique(keys):
        members = usable[keys == key]
        groups.append(Group(index=len(groups), windows=members, time=float(np.mean(scan.time[members])),
                            t_lo=float(np.min(scan.first[members])), t_hi=float(np.max(scan.last[members])),
                            raw_frames=int(scan.raw_frames[members].sum())))
    return groups


def merge_sides(found: dict) -> list:
    """[(af, sd, sides, per_side)] of one group from each side's minima."""
    a, b = list(found.get(0, [])), list(found.get(1, []))
    pairs = sorted(((abs(m.af - n.af), i, j) for i, m in enumerate(a) for j, n in enumerate(b)
                    if abs(m.af - n.af) <= AGREE_DEG), key=lambda p: p[0])
    used_a, used_b, out = set(), set(), []
    for distance, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        m, n = a[i], b[j]
        per_side = {0: (m.af, m.sd), 1: (n.af, n.sd)}
        if distance <= max(SIDE_TOL_DEG, 3 * float(np.hypot(m.sd, n.sd))):
            w0, w1 = 1 / m.sd ** 2, 1 / n.sd ** 2
            af = (w0 * m.af + w1 * n.af) / (w0 + w1)
            sd = max(float(np.sqrt(1 / (w0 + w1))), distance / 2, SD_FLOOR_DEG)
            out.append((af, sd, "both", per_side))
        else:
            out.append(((m.af + n.af) / 2, max(distance / 2, m.sd, n.sd), "disagree", per_side))
    for side, minima, used in ((0, a, used_a), (1, b, used_b)):
        for i, m in enumerate(minima):
            if i not in used:
                out.append((m.af, m.sd, str(side), {side: (m.af, m.sd)}))
    return sorted(out, key=lambda item: -item[0])


def _in_ranges(af, ranges) -> np.ndarray:
    inside = np.zeros(len(af), bool)
    for lo, hi in ranges:
        inside |= (af > lo) & (af < hi)
    return inside


def _cells(af, ranges: dict, usable: dict, side: int, static=None) -> np.ndarray:
    cells = _in_ranges(af, ranges.get(side, [])) & usable[side]
    return cells if static is None else cells & ~static[side]


def detect(scan, groups: list, ranges: dict, usable: dict, alpha_i: float, k: float, rng) -> dict:
    """The minima of every group (in ``group.detections``); returns {side: static cells}."""
    af = scan.grid.cell_af
    qz = qz_vacuum(af, alpha_i, k)
    static = {}
    for side in (0, 1):
        cells = _cells(af, ranges, usable, side)
        residuals = np.full((len(groups), len(af)), np.nan)
        for g in groups:
            y = scan.counts[g.windows, side].sum(axis=0)
            e = scan.exposure[g.windows, side].sum(axis=0)
            g.cuts[side] = (y, e)
            ok = cells & (y > 0) & (e > 0)
            if ok.sum() >= 3:
                residuals[g.index, ok] = np.log(y[ok] / e[ok]) - baseline(af[ok], y[ok], e[ok], qz[ok])
        static[side] = static_cells(residuals) & cells
    for g in groups:
        for side in (0, 1):
            y, e = g.cuts[side]
            cells = _cells(af, ranges, usable, side, static)
            base = np.zeros(len(af))
            ok = cells & (y > 0) & (e > 0)
            if ok.sum() >= 3:
                base[ok] = baseline(af[ok], y[ok], e[ok], qz[ok])
            g.found[side] = find_minima(af, y, e, base, ranges.get(side, []), usable[side] & ~static[side], rng)
        for af_value, sd, sides, per_side in merge_sides(g.found):
            g.detections.append(Detection(g.index, g.time, float(af_value), float(sd), sides, per_side))
    return static


def phase_pass(scan, groups: list, ranges: dict, usable: dict, static: dict, alpha_i: float, alpha_c: float,
               k: float) -> None:
    """The phase-locked fit of every group with ``alpha_c`` (``group.phase``: (H, error) when accepted)."""
    af = scan.grid.cell_af
    ln_qz = np.log(qz_vacuum(af, alpha_i, k))
    q_in = q_inside(af, alpha_i, alpha_c, k)
    grid = heights_grid()
    for g in groups:
        for side in (0, 1):
            y, e = g.cuts[side]
            cells = _cells(af, ranges, usable, side, static)
            g.phase_sides[side] = phase_fit(q_in[cells], ln_qz[cells], y[cells], e[cells], grid)
    checked = [side_check(g.phase_sides.get(0), g.phase_sides.get(1)) for g in groups]
    passed = [value is not None for value, _reason in checked]
    if not groups:
        return
    kept = trend_accept([g.time for g in groups], [v[0] if v else np.nan for v, _r in checked], passed)
    used = np.flatnonzero(_cells(af, ranges, usable, 0, static) | _cells(af, ranges, usable, 1, static))
    mid = float((af[used[0]] + af[used[-1]]) / 2) if len(used) else 1.0
    for g, (value, reason), keep, ok in zip(groups, checked, kept, passed):
        g.phase = value if keep else None
        g.phase_alpha_c, g.phase_mid_deg = alpha_c, mid
        g.phase_note = reason if not ok else ("" if keep else "off the trend of the accepted groups")


def phase_cuts(scan, groups: list, ranges: dict, usable: dict, static: dict, alpha_i: float, k: float) -> list:
    """[(H, af, ln q_z, counts, exposure)] of both sides of the groups with an accepted phase-locked fit that holds at
    least ``AC_MIN_PERIODS`` periods on both sides (fewer leave alpha_c and H too free)."""
    af = scan.grid.cell_af
    ln_qz = np.log(qz_vacuum(af, alpha_i, k))
    out = []
    for g in groups:
        sides = [g.phase_sides.get(side) or {} for side in (0, 1)]
        if g.phase is None or min(side.get("periods", 0.0) for side in sides) < AC_MIN_PERIODS:
            continue
        for side in (0, 1):
            y, e = g.cuts[side]
            cells = _cells(af, ranges, usable, side, static)
            out.append((g.phase[0], af[cells], ln_qz[cells], y[cells], e[cells]))
    return out


__all__ = ["Detection", "Group", "SIDE_TOL_DEG", "detect", "make_groups", "merge_sides", "phase_cuts", "phase_pass"]
