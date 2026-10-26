"""Heights from the tracked minima: the critical angle, H of every time group, its status and the growth rate.

H_n = 2 pi n / (kz(alpha_i) + kz(alpha_f)) inside the layer (``optics.height_nm``). alpha_c: given on the command
line, else fitted when two orders coexist in at least ``AC_FIT_MIN_GROUPS`` groups (the alpha_c at which the orders
give the same H; its error from Delta chi^2 = 1, scaled by the reduced chi^2 when that is above 1), else the bulk
value of the metal with an assumed error of ``AC_BULK_ERROR`` (islands may be less dense than the bulk). A fit at the
edge of its range, or outside ``AC_PLAUSIBLE`` times the bulk value, is not used.

Orders that coexist, and the minima and the accepted phase-locked fit of a group, must give the same H within
``ORDER_REL_TOL`` (or three times their combined error); when more than a third of such pairs do not, the orders are
not trusted and every track is reported with an unknown order.

A group's H is the accepted phase-locked fit (``phase.py``) when it holds ``PHASE_FIRST_PERIODS`` periods or more;
otherwise the inverse-variance mean over its minima of known order (the error scaled by the scatter of the orders
when that is larger), or the phase-locked fit when there are none. Its 68% interval combines the position errors
(bootstrap; the curvature of the phase-locked chi^2), the error of alpha_c and the baseline's systematic error
(``BASELINE_SYSTEMATIC``, by the number of periods in range). Statuses: ``measured`` (a minimum seen
on both sides, or the phase-locked fit), ``one_side_only``, ``sides_disagree`` (the two sides put the
minimum at different places), ``ambiguous_order`` (moving minima whose order cannot be told), ``below_range`` (before
the first order entered the top of the range: H is below 2 pi / Q at the top, given as a bound), ``not_detected``.
"""

from __future__ import annotations

import math

import numpy as np

from .optics import height_nm, q_inside

AC_FIT_MIN_GROUPS = 3
AC_PLAUSIBLE = (0.6, 1.25)
AC_BULK_ERROR = 0.02
"""Assumed error of the bulk alpha_c (deg): islands may be less dense than the bulk metal."""
AC_GRID = (0.0, 1.0, 0.0005)
AC_STEP = 0.0002
AC_MAX_CHI2_DOF = 5.0
"""A fit of coexisting orders whose reduced chi^2 is above this is not used: the orders do not fit one alpha_c."""
ORDER_REL_TOL = 0.08
BASELINE_SYSTEMATIC = ((0.75, 0.08), (1.25, 0.04), (2.0, 0.02), (float("inf"), 0.015))
"""Relative systematic error of H by the number of fringe periods in the clean range (H span(Q) / 2 pi): with one
period or less the smooth baseline and the fringe cannot be told apart well and the minimum moves; with many periods
a fringe whose amplitude fades with q_z (a spread of heights) still moves the minima by about 1% (estimated from
synthetic cuts with power-law backgrounds; ``tests/test_sputter_fringes.py``)."""
PHASE_FIRST_PERIODS = 2.0
"""With at least this many periods in range the accepted phase-locked fit gives a group's H (the whole fringe, without
the minima's bias); with fewer, the minima of known order do."""
STATUSES = ("measured", "one_side_only", "sides_disagree", "ambiguous_order", "below_range", "not_detected")


def _height(d, alpha_i: float, alpha_c: float, alpha_c_error: float, k: float) -> tuple:
    h = float(height_nm(d.order, d.af, alpha_i, alpha_c, k))
    by_af = abs(float(height_nm(d.order, d.af - d.sd, alpha_i, alpha_c, k))
                - float(height_nm(d.order, d.af + d.sd, alpha_i, alpha_c, k))) / 2
    by_ac = 0.0
    if alpha_c_error > 0:
        by_ac = abs(float(height_nm(d.order, d.af, alpha_i, alpha_c + alpha_c_error, k))
                    - float(height_nm(d.order, d.af, alpha_i, max(alpha_c - alpha_c_error, 0.0), k))) / 2
    return h, float(math.hypot(by_af, by_ac))


def baseline_systematic(height: float, span_q) -> float:
    """Relative systematic error of H (``span_q``: the clean range's span of Q, nm^-1; 0 without it)."""
    if not span_q:
        return 0.0
    periods = height * span_q / (2 * math.pi)
    return next(error for limit, error in BASELINE_SYSTEMATIC if periods < limit)


def _pairs(groups: list) -> list:
    """[(group, detection a, detection b)] of two different orders in one group (both usable)."""
    out = []
    for g in groups:
        seen = {}
        for d in g.detections:
            if d.order > 0 and d.usable:
                seen.setdefault(d.order, d)
        orders = sorted(seen)
        out += [(g, seen[a], seen[b]) for i, a in enumerate(orders) for b in orders[i + 1:]]
    return out


def fit_alpha_c(groups: list, alpha_i: float, k: float, bulk) -> dict:
    pairs = _pairs(groups)
    used = len({g.index for g, _a, _b in pairs})
    record = {"fitted": False, "groups_with_two_orders": used, "pairs": len(pairs)}
    if used < AC_FIT_MIN_GROUPS:
        record["reason"] = f"two orders coexist in {used} groups (at least {AC_FIT_MIN_GROUPS} needed)"
        return record

    n_a, af_a, sd_a = (np.array([getattr(a, key) for _g, a, _b in pairs]) for key in ("order", "af", "sd"))
    n_b, af_b, sd_b = (np.array([getattr(b, key) for _g, _a, b in pairs]) for key in ("order", "af", "sd"))

    def heights(n, af, sd, ac):
        h = height_nm(n, af, alpha_i, ac, k)
        return h, np.abs(height_nm(n, af - sd, alpha_i, ac, k) - height_nm(n, af + sd, alpha_i, ac, k)) / 2

    def chi2(ac: float) -> float:
        ha, sa = heights(n_a, af_a, sd_a, ac)
        hb, sb = heights(n_b, af_b, sd_b, ac)
        value = float(np.sum((ha - hb) ** 2 / (sa * sa + sb * sb)))
        return value if np.isfinite(value) else np.inf

    high = min(AC_GRID[1], float(min(af_a.min(), af_b.min())) - 0.02)  # below every minimum's alpha_f
    grid = np.arange(AC_GRID[0], high, AC_GRID[2])
    values = np.array([chi2(x) for x in grid])
    best = int(np.argmin(values)) if len(grid) else 0
    if len(grid) < 3 or best in (0, len(grid) - 1) or not np.isfinite(values[best]):
        record["reason"] = "the fit runs to the edge of its range"
        return record
    from scipy.optimize import minimize_scalar

    found = minimize_scalar(chi2, bounds=(grid[best - 1], grid[best + 1]), method="bounded")
    ac, low = float(found.x), float(found.fun)
    dof = max(len(pairs) - 1, 1)
    scale = max(1.0, low / dof)
    edges = []
    for direction in (-1, 1):  # walk out from the minimum until chi^2 has risen by the (scaled) unit
        x = ac
        while AC_GRID[0] < x < high and chi2(x) <= low + scale:
            x += direction * AC_STEP
        edges.append(min(max(x, AC_GRID[0]), high))
    error = max((edges[1] - edges[0]) / 2, AC_STEP)
    record.update(alpha_c_deg=ac, error_deg=error, chi2_dof=low / dof)
    if edges[0] <= AC_GRID[0] or edges[1] >= high:
        record["reason"] = "the fit's interval reaches the edge of its range"
    elif low / dof > AC_MAX_CHI2_DOF:
        record["reason"] = (f"the coexisting orders do not fit one alpha_c (reduced chi^2 {low / dof:.1f} above "
                            f"{AC_MAX_CHI2_DOF:g})")
    elif bulk is not None and not AC_PLAUSIBLE[0] * bulk <= ac <= AC_PLAUSIBLE[1] * bulk:
        record["reason"] = (f"the fitted alpha_c {ac:.3f} deg is outside {AC_PLAUSIBLE[0]}-{AC_PLAUSIBLE[1]} times the "
                            f"bulk value {bulk:.3f} deg: the orders may be wrong")
    else:
        record["fitted"] = True
    return record


def orders_consistent(groups: list, alpha_i: float, alpha_c: float, alpha_c_error: float, k: float) -> dict:
    """Coexisting orders, and the minima against the accepted phase-locked fit, must give the same H."""
    checked = bad = 0
    for g in groups:
        heights = {}
        for d in g.detections:
            if d.order > 0 and d.usable:
                heights.setdefault(d.order, _height(d, alpha_i, alpha_c, alpha_c_error, k))
        values = list(heights.values())
        if g.phase is not None:
            values.append(g.phase)
        for i, (ha, sa) in enumerate(values):
            for hb, sb in values[i + 1:]:
                checked += 1
                if abs(ha - hb) > max(ORDER_REL_TOL * (ha + hb) / 2, 3 * math.hypot(sa, sb)):
                    bad += 1
    return {"pairs_checked": checked, "pairs_inconsistent": bad, "consistent": not (checked and bad * 3 > checked)}


def _phase_height(g, alpha_i: float, alpha_c: float, alpha_c_error: float, k: float, span_q) -> dict:
    """The accepted phase-locked H of a group with ``alpha_c`` (scaled from the alpha_c of the fit by Q at the middle
    of the range when they differ)."""
    h, error = g.phase
    fitted_with = getattr(g, "phase_alpha_c", alpha_c)
    if abs(fitted_with - alpha_c) > 1e-9:
        mid = getattr(g, "phase_mid_deg", 1.0)
        h *= float(q_inside(mid, alpha_i, fitted_with, k) / q_inside(mid, alpha_i, alpha_c, k))
    error = math.hypot(error, h * _ac_relative(alpha_i, alpha_c, alpha_c_error, k), h * baseline_systematic(h, span_q))
    return {"H_nm": h, "error_nm": error, "method": "phase-locked fit", "both_sides": True}


def heights_at(groups: list, alpha_i: float, alpha_c: float, alpha_c_error: float, k: float, span_q=None) -> dict:
    """{group index: height} with ``alpha_c`` (H, its 68% error, the orders of the minima, the method, both sides):
    the accepted phase-locked fit when it holds ``PHASE_FIRST_PERIODS`` periods, else the minima of known order, else
    the accepted phase-locked fit."""
    out = {}
    for g in groups:
        values = {}
        for d in g.detections:
            if d.order > 0 and d.usable and d.order not in values:
                values[d.order] = (_height(d, alpha_i, alpha_c, alpha_c_error, k), d)
        phase = _phase_height(g, alpha_i, alpha_c, alpha_c_error, k, span_q) if g.phase is not None else None
        if phase is not None and (not values or (span_q and phase["H_nm"] * span_q / (2 * math.pi)
                                                 >= PHASE_FIRST_PERIODS)):
            out[g.index] = dict(phase, orders=sorted(values))
            continue
        if values:
            h = np.array([v[0][0] for v in values.values()])
            s = np.array([max(v[0][1], 1e-6) for v in values.values()])
            w = 1 / s ** 2
            mean = float(np.sum(w * h) / np.sum(w))
            error = float(1 / np.sqrt(np.sum(w)))
            if len(h) > 1:
                scatter = float(np.sum(w * (h - mean) ** 2)) / (len(h) - 1)
                error *= math.sqrt(max(1.0, scatter))
            error = math.hypot(error, mean * baseline_systematic(mean, span_q))
            out[g.index] = {"H_nm": mean, "error_nm": error, "orders": sorted(values), "method": "minima",
                            "both_sides": any(d.sides == "both" for _v, d in values.values())}
    return out


def group_heights(groups: list, alpha_i: float, alpha_c: float, alpha_c_error: float, k: float, span_q=None) -> None:
    heights = heights_at(groups, alpha_i, alpha_c, alpha_c_error, k, span_q)
    for g in groups:
        g.height = heights.get(g.index)


def _ac_relative(alpha_i: float, alpha_c: float, alpha_c_error: float, k: float) -> float:
    """Relative change of H over the clean range's middle when alpha_c changes by its error (for the phase fit)."""
    if alpha_c_error <= 0:
        return 0.0
    af = 1.0
    h = float(height_nm(1, af, alpha_i, alpha_c, k))
    return abs(float(height_nm(1, af, alpha_i, alpha_c + alpha_c_error, k))
               - float(height_nm(1, af, alpha_i, max(alpha_c - alpha_c_error, 0.0), k))) / (2 * h)


def statuses(groups: list, tracks: list, bound_nm) -> None:
    """``group.status`` and, for ``below_range``, ``group.bound_nm``."""
    entry = next((t.entry_time for t in tracks if t.status == "accepted" and t.order == 1 and t.entered), None)
    for g in groups:
        g.bound_nm = None
        in_tracks = [d for d in g.detections if d.accepted]
        if g.height is not None:
            g.status = "measured" if g.height["both_sides"] else "one_side_only"
        elif any(d.order == 0 for d in in_tracks):
            g.status = "ambiguous_order"
        elif any(d.order > 0 and not d.usable for d in in_tracks):
            g.status = "sides_disagree"
        elif entry is not None and g.time < entry:
            g.status = "below_range"
            g.bound_nm = bound_nm
        else:
            g.status = "not_detected"


def growth_fit(points: list) -> dict:
    """H = v (t - t0) through [(t, H, error)] (weighted)."""
    record = {"fitted": False, "groups": len(points)}
    if len(points) < 4 or np.ptp([p[0] for p in points]) < 3.0:
        record["reason"] = "fewer than 4 measured groups over at least 3 s"
        return record
    t, h, s = (np.array(column, dtype=float) for column in zip(*points))
    w = 1 / np.maximum(s, 1e-6) ** 2
    X = np.stack([np.ones_like(t), t], axis=1)
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    a, b = cov @ (X.T @ (w * h))
    chi2 = float(np.sum(w * (h - a - b * t) ** 2))
    dof = len(t) - 2
    cov = cov * max(1.0, chi2 / dof)
    if b == 0:
        record["reason"] = "no slope"
        return record
    t0 = -a / b
    grad = np.array([-1 / b, a / b ** 2])  # d t0 / d (a, b)
    record.update(fitted=True, v_nm_s=float(b), v_error_nm_s=float(math.sqrt(cov[1, 1])), t0_s=float(t0),
                  t0_error_s=float(math.sqrt(grad @ cov @ grad)), H_at_t0_scan_nm=float(a),
                  chi2_dof=chi2 / dof, t_first_s=float(t.min()), t_last_s=float(t.max()))
    return record


def growth_points(groups: list, heights: dict = None) -> list:
    heights = heights if heights is not None else {g.index: g.height for g in groups}
    return [(g.time, heights[g.index]["H_nm"], heights[g.index]["error_nm"]) for g in groups
            if heights.get(g.index) is not None]


def upper_bound_nm(top_af: float, alpha_i: float, alpha_c: float, k: float):
    """2 pi / Q at the top of the clean range: H below it puts the first minimum above the range."""
    q = float(q_inside(top_af, alpha_i, alpha_c, k))
    return 2 * math.pi / q if q > 0 else None


__all__ = ["AC_BULK_ERROR", "AC_FIT_MIN_GROUPS", "AC_PLAUSIBLE", "BASELINE_SYSTEMATIC", "ORDER_REL_TOL",
           "PHASE_FIRST_PERIODS", "STATUSES",
           "baseline_systematic", "fit_alpha_c", "group_heights", "growth_fit", "growth_points", "heights_at",
           "orders_consistent", "statuses", "upper_bound_nm"]
