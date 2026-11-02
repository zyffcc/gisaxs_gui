"""Left/right consistency of a Yoneda cut (isotropic in-plane samples scatter symmetrically in qy).

Both sides are compared on the overlap of their |q| ranges: the right side is
interpolated (log-log, positive values) onto the left q points, z is the
difference over the combined sigma, smoothed with a running median to ignore
isolated hot pixels. Runs of |z| above `threshold` are reported as disputed
regions, together with the smallest q above which the sides agree. A global
scale ratio is estimated from the agreeing region and reported separately (a
small overall mismatch can come from detector efficiency or the footprint).
Disputed regions are candidates for masking, e.g. beamstop flare, parasitic
scattering or the specular rod on one side; they are not proof of an artifact.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def compare_sides(q_left, i_left, s_left, q_right, i_right, s_right, *, threshold=5.0, window=5,
                  relative_floor=0.02):
    ql, il, sl = (np.asarray(v, float) for v in (q_left, i_left, s_left))
    qr, ir, sr = (np.asarray(v, float) for v in (q_right, i_right, s_right))
    order = np.argsort(qr)
    qr, ir, sr = qr[order], ir[order], sr[order]
    lo, hi = max(ql.min(), qr.min()), min(ql.max(), qr.max())
    use = (ql >= lo) & (ql <= hi) & (il > 0)
    good_r = ir > 0
    if use.sum() < 3 or good_r.sum() < 3:
        return dict(status='insufficient_overlap')
    q = ql[use]
    interp = np.exp(np.interp(np.log(q), np.log(qr[good_r]), np.log(ir[good_r])))
    s_interp = np.interp(q, qr, sr)
    # relative floor: interpolation between right-side points is not exact
    sigma = np.sqrt(sl[use]**2+s_interp**2+(relative_floor*interp)**2)
    z = (il[use]-interp)/sigma
    zs = median_filter(z, size=window, mode='nearest')
    bad = np.abs(zs) > threshold
    regions = []
    start = None
    for k, flag in enumerate(np.r_[bad, False]):
        if flag and start is None:
            start = k
        elif not flag and start is not None:
            regions.append(dict(q_from=float(q[start]), q_to=float(q[k-1]), points=int(k-start),
                                sign='left>right' if np.median(z[start:k]) > 0 else 'right>left',
                                max_abs_z=float(np.abs(z[start:k]).max())))
            start = None
    agree = ~bad
    ratio = float(np.median(il[use][agree]/interp[agree])) if agree.any() else float('nan')
    last_bad = np.flatnonzero(bad)
    q_agree_from = float(q[last_bad[-1]+1]) if len(last_bad) and last_bad[-1]+1 < len(q) else (float(q[0]) if not len(last_bad) else None)
    return dict(status='ok', overlap=[float(lo), float(hi)], points=int(use.sum()), threshold=threshold,
                disputed_fraction=float(bad.mean()), regions=regions, median_ratio_left_over_right=ratio,
                q_agree_from=q_agree_from, rms_z=float(np.sqrt(np.mean(z**2))))


def check_cut(cut, **kwargs):
    """compare_sides on a yoneda_cut result (uses qy)."""
    left, right = cut['left'], cut['right']
    return compare_sides(left['qy'], left['intensity'], left['sigma'], right['qy'], right['intensity'],
                         right['sigma'], **kwargs)


def _log_mismatch(qs, ys, delta, lo, hi):
    """rms log difference between the two sides after moving the qy origin by delta (signed q)."""
    q2 = qs-delta
    left, right = q2 < 0, q2 > 0
    a_q, a_y = np.abs(q2[left]), ys[left]
    b_q, b_y = q2[right], ys[right]
    oa, ob = np.argsort(a_q), np.argsort(b_q)
    a_q, a_y, b_q, b_y = a_q[oa], a_y[oa], b_q[ob], b_y[ob]
    use = (a_q >= lo) & (a_q <= hi) & (a_y > 0)
    good = b_y > 0
    if use.sum() < 5 or good.sum() < 5:
        return np.inf, np.nan
    bi = np.exp(np.interp(np.log(a_q[use]), np.log(b_q[good]), np.log(b_y[good])))
    diff = np.log(a_y[use])-np.log(bi)
    return float(np.sqrt(np.mean((diff-np.median(diff))**2))), float(np.exp(np.median(diff)))


def estimate_center_shift(q_left, i_left, q_right, i_right, *, window=None, max_shift=None, steps=801):
    """Shift of the qy origin that makes the two sides of a cut agree best.

    Returns delta in q units in the signed convention (left side negative): the
    true qy=0 lies at signed q = delta. The mismatch is the rms of the log ratio
    after removing a global left/right scale factor (reported as ratio_left_over_right).
    """
    ql, il, qr, ir = (np.asarray(v, float) for v in (q_left, i_left, q_right, i_right))
    qs = np.r_[-ql, qr]
    ys = np.r_[il, ir]
    step = float(np.median(np.abs(np.diff(np.sort(qr)))))
    if window is None:
        window = (3*step, 0.3*min(ql.max(), qr.max()))
    max_shift = max_shift or 20*step
    grid = np.linspace(-max_shift, max_shift, steps)
    values = np.array([_log_mismatch(qs, ys, g, *window)[0] for g in grid])
    k = int(np.argmin(values))
    delta = float(grid[k])
    if 0 < k < len(grid)-1 and np.isfinite(values[k-1:k+2]).all():   # parabolic refinement
        a, b, c = values[k-1:k+2]
        denominator = a-2*b+c
        if denominator > 0:
            delta = float(grid[k]+0.5*(a-c)/denominator*(grid[1]-grid[0]))
    before, _ = _log_mismatch(qs, ys, 0.0, *window)
    after, ratio = _log_mismatch(qs, ys, delta, *window)
    return dict(delta=delta, delta_in_q_steps=delta/step, mismatch_before=before, mismatch_after=after,
                ratio_left_over_right=ratio, window=list(window), at_search_edge=bool(k in (0, len(grid)-1)))


def merge_sides(q_left, i_left, s_left, q_right, i_right, s_right, delta=0.0, ratio=1.0):
    """One curve from both sides: move the origin by delta, rescale the left side by 1/ratio."""
    qs = np.r_[-np.asarray(q_left, float), np.asarray(q_right, float)]-delta
    ys = np.r_[np.asarray(i_left, float)/ratio, np.asarray(i_right, float)]
    ss = np.r_[np.asarray(s_left, float)/ratio, np.asarray(s_right, float)]
    side = np.r_[np.zeros(len(q_left), int), np.ones(len(q_right), int)]
    keep = np.abs(qs) > 0
    order = np.argsort(np.abs(qs[keep]))
    return np.abs(qs[keep])[order], ys[keep][order], ss[keep][order], side[keep][order]


def side_difference_sigma(q_left, i_left, q_right, i_right, delta=0.0, ratio=1.0, bins=24, min_points=6):
    """Data-driven noise estimate from the left/right difference of a recentred cut.

    Each left point is compared with the right side interpolated (log-log) to
    the same |q|; in log-spaced q bins the robust spread (MAD x 1.4826) of the
    differences, normalised for the interpolation weights, estimates the per-point
    sigma of one side.
    Returns a function sigma(q, intensity) that interpolates the relative sigma
    between bin centres (flat outside), plus the table. Assumes both sides carry
    the same noise level and that interpolation error is small compared with it.
    """
    qs = np.r_[-np.asarray(q_left, float), np.asarray(q_right, float)]-delta
    ys = np.r_[np.asarray(i_left, float)/ratio, np.asarray(i_right, float)]
    left, right = qs < 0, qs > 0
    a_q, a_y = np.abs(qs[left]), ys[left]
    b_q, b_y = qs[right], ys[right]
    ob = np.argsort(b_q)
    b_q, b_y = b_q[ob], b_y[ob]
    good = b_y > 0
    use = (a_q >= b_q[good].min()) & (a_q <= b_q[good].max()) & (a_y > 0)
    a_q, a_y = a_q[use], a_y[use]
    lq, lb = np.log(b_q[good]), b_y[good]
    b_i = np.exp(np.interp(np.log(a_q), lq, np.log(lb)))
    # interpolation averages two right-side points with weights (1-t, t), which lowers
    # their variance to (t^2+(1-t)^2) sigma^2; normalise the difference accordingly
    j = np.clip(np.searchsorted(lq, np.log(a_q)), 1, len(lq)-1)
    t = np.clip((np.log(a_q)-lq[j-1])/(lq[j]-lq[j-1]), 0, 1)
    rel = (a_y-b_i)/np.sqrt(a_y*b_i)/np.sqrt(1+t**2+(1-t)**2)
    edges = np.geomspace(a_q.min(), a_q.max()*1.0001, bins+1)
    centres, values = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (a_q >= lo) & (a_q < hi)
        if k.sum() >= min_points:
            centres.append(np.sqrt(lo*hi))
            values.append(1.4826*np.median(np.abs(rel[k]-np.median(rel[k]))))
    centres, values = np.array(centres), np.array(values)

    def sigma(q, intensity):
        return np.interp(np.log(np.asarray(q, float)), np.log(centres), values)*np.abs(np.asarray(intensity, float))
    return sigma, dict(q=centres.tolist(), relative_sigma=values.tolist())
