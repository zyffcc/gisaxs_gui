"""Follow the minima through time and give each track its order.

While the metal grows, the order-n minimum of a layer of height H, at Q_n = 2 pi n / H (Q inside the layer), moves
to lower alpha_f. Q uses the alpha_c of the tracking (the given one, else the bulk value).

1. Static minima: a minimum with others within ``STATIC_TOL_DEG`` over a run of at least ``STATIC_SECONDS`` (no
   more than ``LINK_GROUPS`` groups missing) stays where it is: a detector or film feature, left out.
2. Linking (minima seen on both sides only): in time order, a minimum continues the track whose last minimum is at
   most ``LINK_GROUPS`` groups earlier and lies between ``LINK_UP_DEG`` below and ``LINK_DOWN_DEG`` above it (the
   closest wins; one minimum per track and group); otherwise it starts a track.
3. Acceptance: at least ``MIN_POINTS`` minima and a net descent of at least ``MIN_DESCENT_DEG``. A minimum seen on
   one side only (or placed differently by the two sides) then joins an accepted track in a group where it has none,
   when it lies within ``ATTACH_DEG`` of the track's line.
4. Orders, in this sequence (each step only for tracks without an order):
   a. the phase-locked fit: n = H_phase Q / (2 pi) at the track's minima in groups where the fit is accepted, the
      median within ``PHASE_TOL`` of an integer (at least ``PHASE_POINTS`` minima);
   b. when no track has an order yet: the first accepted track is order 1 when it entered at the top of the clean
      range — its first minimum within ``ENTRY_ZONE_DEG`` of the top, or a straight line of 1/Q (proportional to H)
      over its first ``ENTRY_POINTS`` minima puts it there at most ``ENTRY_EXTRAPOLATE_S`` earlier — at least
      ``ENTRY_BEFORE_GROUPS`` groups after the first group (fringe-free groups before it);
   c. a track coexisting with one of known order: n = n_ref median(Q / Q_ref), within ``RATIO_TOL`` of an integer;
   d. a track with no such overlap (e.g. split by a module gap): n = H Q / (2 pi) at (up to) five of its minima, with
      H from a straight line through the known heights within ``EXTRAPOLATE_S`` (tracks whose order does not itself
      rest on an extrapolation, and the accepted phase-locked groups): the median within ``EXTRAPOLATE_TOL`` of an
      integer and every estimate within 0.5 of it.
   Steps c and d repeat until nothing changes; a track left without an order is ambiguous.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .optics import q_inside

STATIC_TOL_DEG = 0.01
STATIC_SECONDS = 20.0
LINK_GROUPS = 3
LINK_DOWN_DEG = 0.08
LINK_UP_DEG = 0.01
MIN_POINTS = 4
MIN_DESCENT_DEG = 0.05
PHASE_TOL = 0.25
PHASE_POINTS = 3
ENTRY_ZONE_DEG = 0.12
ENTRY_BEFORE_GROUPS = 2
ENTRY_EXTRAPOLATE_S = 30.0
ENTRY_POINTS = 8
RATIO_TOL = 0.25
EXTRAPOLATE_S = 30.0
EXTRAPOLATE_TOL = 0.3
ATTACH_DEG = 0.02
EXTRAPOLATED = "H extrapolated from the known heights"


@dataclass
class Track:
    points: list = field(default_factory=list)
    status: str = "open"
    """accepted, short, stationary"""
    order: int = 0
    order_source: str = ""
    entered: bool = False
    entry_time: float = float("nan")

    @property
    def start(self):
        return self.points[0]

    @property
    def end(self):
        return self.points[-1]

    def by_group(self) -> dict:
        return {p.group: p for p in self.points}

    def summary(self) -> dict:
        return {"order": self.order or None, "order_source": self.order_source or None, "status": self.status,
                "entered_at_top": self.entered,
                "entry_time_s": round(self.entry_time, 3) if math.isfinite(self.entry_time) else None,
                "minima": len(self.points), "both_sides": sum(p.sides == "both" for p in self.points),
                "t_start_s": round(self.start.time, 3), "t_end_s": round(self.end.time, 3),
                "af_start_deg": round(self.start.af, 4), "af_end_deg": round(self.end.af, 4)}


def annotate(groups: list, alpha_i: float, alpha_c: float, k: float) -> None:
    """ln Q of each minimum and its error, with the alpha_c of the tracking."""
    for g in groups:
        for d in g.detections:
            q = float(q_inside(d.af, alpha_i, alpha_c, k))
            low = float(q_inside(d.af - d.sd, alpha_i, alpha_c, k))
            high = float(q_inside(d.af + d.sd, alpha_i, alpha_c, k))
            d.q = q
            d.lnq = math.log(q) if q > 0 else float("nan")
            d.sd_lnq = (math.log(max(high, 1e-9)) - math.log(max(low, 1e-9))) / 2 if q > 0 else float("inf")


def mark_static(groups: list, group_seconds: float) -> int:
    detections = [d for g in groups for d in g.detections]
    if not detections:
        return 0
    af = np.array([d.af for d in detections])
    time = np.array([d.time for d in detections])
    gap = (LINK_GROUPS + 0.5) * group_seconds
    for i, d in enumerate(detections):
        times = np.sort(time[np.abs(af - d.af) <= STATIC_TOL_DEG])
        breaks = np.flatnonzero(np.diff(times) > gap)
        starts, ends = np.r_[0, breaks + 1], np.r_[breaks, len(times) - 1]
        for s, e in zip(starts, ends):
            if times[s] - 1e-6 <= d.time <= times[e] + 1e-6 and times[e] - times[s] >= STATIC_SECONDS:
                d.static = True
    return sum(d.static for d in detections)


def link(groups: list) -> list:
    tracks: list = []
    for g in groups:
        taken = set()
        for d in sorted((d for d in g.detections if d.sides == "both" and not d.static and math.isfinite(d.lnq)),
                        key=lambda d: -d.af):
            best = None
            for i, track in enumerate(tracks):
                last = track.end
                if i in taken or not 0 < g.index - last.group <= LINK_GROUPS:
                    continue
                if last.af - LINK_DOWN_DEG <= d.af <= last.af + LINK_UP_DEG:
                    distance = abs(d.af - last.af)
                    if best is None or distance < best[0]:
                        best = (distance, i)
            if best is None:
                tracks.append(Track(points=[d]))
                taken.add(len(tracks) - 1)
            else:
                tracks[best[1]].points.append(d)
                taken.add(best[1])
    return tracks


def accept(tracks: list) -> None:
    for track in tracks:
        descent = track.start.af - track.end.af
        if len(track.points) < MIN_POINTS:
            track.status = "short"
        elif descent < MIN_DESCENT_DEG:
            track.status = "stationary"
        else:
            track.status = "accepted"


def attach(tracks: list, groups: list) -> None:
    """One-side (and disagreeing) minima of an accepted track's groups without a point of it: those within
    ``ATTACH_DEG`` of the track's line between its neighbouring minima."""
    taken = {id(p) for t in tracks for p in t.points}
    for track in tracks:
        if track.status != "accepted":
            continue
        have = track.by_group()
        times = np.array([p.time for p in track.points])
        afs = np.array([p.af for p in track.points])
        added = []
        for g in groups[track.start.group:track.end.group + 1]:
            if g.index in have:
                continue
            expected = float(np.interp(g.time, times, afs))
            near = [d for d in g.detections if d.sides != "both" and not d.static and id(d) not in taken
                    and abs(d.af - expected) <= ATTACH_DEG and math.isfinite(d.lnq)]
            if near:
                d = min(near, key=lambda d: abs(d.af - expected))
                added.append(d)
                taken.add(id(d))
        track.points = sorted(track.points + added, key=lambda p: p.time)


def _entry(track, top_af: float, alpha_i: float, alpha_c: float, k: float) -> float:
    """The time the track's minimum was at the top of the range (its first minimum's time when it is there), or NaN:
    1/Q of a minimum is H / (2 pi n), so a straight line of 1/Q over its first minima goes back to 1/Q at the top."""
    if track.start.af >= top_af - ENTRY_ZONE_DEG:
        return track.start.time
    first = [p for p in track.points if p.q > 0][:ENTRY_POINTS]
    t = np.array([p.time for p in first])
    inverse = np.array([1 / p.q for p in first])
    if len(first) < 3 or np.ptp(t) <= 0:
        return float("nan")
    slope, offset = np.polyfit(t, inverse, 1)
    if slope <= 0:
        return float("nan")
    entry = (1 / float(q_inside(top_af, alpha_i, alpha_c, k)) - offset) / slope
    return entry if 0 <= track.start.time - entry <= ENTRY_EXTRAPOLATE_S else float("nan")


def _known_heights(tracks: list, groups: list) -> list:
    """(time, H) of the tracks whose order does not rest on an extrapolation, and of the phase-locked groups."""
    points = [(p.time, 2 * math.pi * t.order / p.q) for t in tracks
              if t.order and t.order_source != EXTRAPOLATED for p in t.points if p.q > 0 and p.usable]
    points += [(g.time, g.phase[0]) for g in groups if g.phase is not None]
    return points


def _by_ratio(track, assigned: list):
    for ref in assigned:
        mine, theirs = track.by_group(), ref.by_group()
        shared = sorted(mine.keys() & theirs.keys())
        if len(shared) < 2:
            continue
        ratio = ref.order * float(np.median([mine[g].q / theirs[g].q for g in shared]))
        if round(ratio) >= 1 and abs(ratio - round(ratio)) <= RATIO_TOL:
            return int(round(ratio)), f"coexists with order {ref.order}"
    return None


def _by_extrapolation(track, known: list):
    if len(known) < 3:
        return None
    known = np.array(known)
    estimates = []
    for p in sorted(track.points, key=lambda p: min(abs(known[:, 0] - p.time)))[:5]:
        near = np.abs(known[:, 0] - p.time) <= EXTRAPOLATE_S
        if near.sum() < 3 or np.ptp(known[near, 0]) <= 0:
            continue
        h = float(np.polyval(np.polyfit(known[near, 0], known[near, 1], 1), p.time))
        estimates.append(h * p.q / (2 * math.pi))
    if len(estimates) < 3:
        return None
    ratio = float(np.median(estimates))
    n = int(round(ratio))
    if n >= 1 and abs(ratio - n) <= EXTRAPOLATE_TOL and all(abs(e - n) <= 0.5 for e in estimates):
        return n, EXTRAPOLATED
    return None


def assign_orders(tracks: list, groups: list, top_af: float, alpha_i: float, alpha_c: float, k: float) -> None:
    accepted = sorted((t for t in tracks if t.status == "accepted"), key=lambda t: t.start.time)
    phase = {g.index: g.phase for g in groups if g.phase is not None}
    for track in accepted:
        estimates = [phase[p.group][0] * p.q / (2 * math.pi) for p in track.points if p.group in phase]
        if len(estimates) >= PHASE_POINTS:
            n = float(np.median(estimates))
            if round(n) >= 1 and abs(n - round(n)) <= PHASE_TOL:
                track.order, track.order_source = int(round(n)), "phase-locked fit"
    for track in accepted:
        track.entry_time = float(_entry(track, top_af, alpha_i, alpha_c, k))
        track.entered = bool(math.isfinite(track.entry_time) and len(groups) > ENTRY_BEFORE_GROUPS
                             and track.entry_time >= groups[ENTRY_BEFORE_GROUPS].time - 1e-9)
    if accepted and not any(t.order for t in accepted) and accepted[0].entered:
        accepted[0].order, accepted[0].order_source = 1, "entered at the top first"
    changed = True
    while changed:
        changed = False
        for track in accepted:
            if track.order:
                continue
            found = _by_ratio(track, [t for t in accepted if t.order]) \
                or _by_extrapolation(track, _known_heights(accepted, groups))
            if found:
                track.order, track.order_source = found
                changed = True


def build(groups: list, top_af: float, group_seconds: float, alpha_i: float, alpha_c: float, k: float) -> list:
    """All tracks (accepted and left out) with their orders; each minimum learns its track and order."""
    mark_static(groups, group_seconds)
    tracks = [t for t in link(groups) if len(t.points) >= 2]
    accept(tracks)
    attach(tracks, groups)
    assign_orders(tracks, groups, top_af, alpha_i, alpha_c, k)
    for index, track in enumerate(tracks):
        for p in track.points:
            p.track = index
            p.accepted = track.status == "accepted"
            p.order = track.order if p.accepted else 0
    return tracks


def settings() -> dict:
    return {"static_tol_deg": STATIC_TOL_DEG, "static_seconds": STATIC_SECONDS, "link_groups": LINK_GROUPS,
            "link_down_deg": LINK_DOWN_DEG, "link_up_deg": LINK_UP_DEG, "min_points": MIN_POINTS,
            "min_descent_deg": MIN_DESCENT_DEG, "phase_tol": PHASE_TOL, "entry_zone_deg": ENTRY_ZONE_DEG,
            "entry_before_groups": ENTRY_BEFORE_GROUPS, "entry_extrapolate_s": ENTRY_EXTRAPOLATE_S,
            "ratio_tol": RATIO_TOL, "extrapolate_s": EXTRAPOLATE_S, "extrapolate_tol": EXTRAPOLATE_TOL}


__all__ = ["ENTRY_ZONE_DEG", "Track", "accept", "attach", "annotate", "assign_orders", "build", "link", "mark_static",
           "settings"]
