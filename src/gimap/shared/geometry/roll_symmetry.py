"""Bilateral angular profiles for absolute roll about the incident beam."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks, peak_widths
from scipy.stats import theilslopes

from .q_mapping import lab_rays, transmission_q


def _ring_axis(beta, intensity, pixel_angle):
    """Reflect the measured azimuth bins about each trial axis, using fixed common coverage."""
    edges = np.arange(-111., 111.01, .5)
    total, _ = np.histogram(beta, edges, weights=intensity.astype(float))
    count, _ = np.histogram(beta, edges)
    profile = total / count.clip(min=1)
    profile[count < 3] = np.nan
    centres = (edges[1:] + edges[:-1])/2
    chi = np.arange(.25, 100., .5)
    angles = np.linspace(-8., 8., 161)
    right = np.interp(angles[:, None] + chi, centres, profile)
    left = np.interp(angles[:, None] - chi, centres, profile)
    # Do not interpolate over missing angular bins or allow the candidate angle to
    # reduce its residual by rotating part of the usable data outside the detector.
    use = np.all(np.isfinite(right + left), axis=0)
    if use.sum() < 40:
        return None
    right, left = right[:, use], left[:, use]
    difference = right - left
    loss = np.mean(difference**2, axis=1)
    i = int(np.argmin(loss))
    if i < 2 or i >= len(loss)-2:
        return None
    correlation = np.corrcoef(left[i], right[i])[0, 1] if min(np.std(left[i]), np.std(right[i])) > 0 else 0
    if not np.isfinite(correlation) or correlation < .55:
        return None
    # Between-sector scatter allows for true arm asymmetry. Include a pixel and half
    # an angular bin in the error, and reject a flat or multiply located minimum.
    sensitivity = (difference[i+1] - difference[i-1])/.2
    information = sensitivity @ sensitivity
    if information <= 0 or max(loss[0], loss[-1]) <= 1.1 * loss[i]:
        return None
    score = sensitivity * difference[i]
    block_score = np.array([part.sum() for part in np.array_split(score, min(12, len(score)//4))])
    error = np.sqrt(np.sum(block_score**2)) / information
    error = max(.25, pixel_angle, float(error))
    competitive = angles[loss <= loss[i] + max(1e-14, loss[i]*.03)]
    if competitive[-1] - competitive[0] > 6 or error > 3:
        return None
    return float(angles[i]), max(error, (competitive[-1] - competitive[0])/2)


def fibre_symmetry(frame, valid, g):
    """At fixed |q| the two pixels at lab azimuth ψ ± β have equal |χ| for a fibre."""
    if min(frame.shape) < 32 or not valid.any():
        return None
    stride = max(1, int(np.ceil(np.sqrt(frame.size/1500000))))
    row, col = np.nonzero(valid[::stride, ::stride])
    if len(row) < 100:
        return None
    row, col = row*stride, col*stride
    along, right, up = lab_rays(g, (col+.5-g.beam_center_x_px)*g.pixel_size_x_m,
                               (g.beam_center_y_px-row-.5)*g.pixel_size_y_m)
    q, _ = transmission_q(along, right, up, g.wavevector_inv_angstrom)
    beta = np.degrees(np.arctan2(right, up))
    above = np.abs(beta) < 78
    # Use the sign of the lab transverse coordinate to test bilateral detector coverage.
    if not above.any() or min(np.sum(beta < -15), np.sum(beta > 15)) < 100:
        return None
    intensity = frame[row, col]
    dq = g.wavevector_inv_angstrom * min(g.pixel_size_x_m, g.pixel_size_y_m)/g.distance_m
    bins = min(1400, max(32, int(q[above].max()/(1.5*dq))))
    edges = np.linspace(0., q[above].max(), bins+1)
    n, _ = np.histogram(q[above], edges)
    total, _ = np.histogram(q[above], edges, weights=intensity[above].astype(float))
    radial = gaussian_filter1d(total/n.clip(min=1), 1.)
    background = median_filter(radial, size=max(11, bins//18*2+1), mode="nearest")
    residual = radial - background
    noise = 1.4826*np.median(np.abs(np.diff(residual)))
    peaks, props = find_peaks(residual, prominence=max(noise*4, np.ptp(residual)*.025, 1e-12))
    if not len(peaks):
        return None
    widths = peak_widths(residual, peaks, rel_height=.5)[0]
    values, errors = [], []
    for j in np.argsort(props["prominences"])[::-1][:8]:
        peak = peaks[j]
        if not 4 <= peak < bins-3 or n[peak] < 15:
            continue
        centre = .5*(edges[peak] + edges[peak+1])
        width = np.clip(widths[j], 2, .10*peak)*(edges[1]-edges[0])
        band = np.abs(q-centre) <= width*.7
        pixel_angle = np.degrees(np.arctan2(dq, centre))
        estimate = _ring_axis(beta[band], intensity[band], pixel_angle)
        if estimate is not None:
            values.append(estimate[0])
            errors.append(estimate[1])
    if not values:
        return None
    weights = np.asarray(errors)**-2
    value = np.average(values, weights=weights)
    return float(value), float(max(np.min(errors), np.sqrt(np.average((values-value)**2, weights=weights))))


def rod_symmetry(frame, valid, g):
    """Axis of a resolved specular rod, with an independent intercept for beam-centre offset."""
    if min(frame.shape) < 32:
        return None
    height, width = frame.shape
    half = max(8, int(.04*width))
    left = max(0, int(g.beam_center_x_px)-half)
    right = min(width, int(g.beam_center_x_px)+half+1)
    if right-left < 2*half-2:
        return None
    # The specular rod is the narrow central peak above the sample horizon. Dark
    # beamstop shadows / diffuse lobes do not establish its direction.
    crossing = g.with_sample_roll(0.).horizon_row()
    rows = np.arange(height)
    above = ((crossing-rows-.5)*np.sign(g.in_plane_axes()[1][2]) > 3)
    if above.sum() < 30:
        return None
    strip_rows = np.array_split(rows[above], min(56, above.sum()//3))
    xs, ys = [], []
    for rows in strip_rows:
        slab = frame[rows, left:right]
        mask = valid[rows, left:right]
        n = mask.sum(axis=0)
        line = np.where(mask, slab, 0.).sum(axis=0) / n.clip(min=1)
        if np.sum(n > 0) < .8*line.size:
            continue
        smooth = gaussian_filter1d(line, 1.)
        base = np.median(smooth)
        peak = int(smooth.argmax())
        amplitude = smooth[peak]-base
        if amplitude <= max(6*1.4826*np.median(np.abs(smooth-base)), 1e-12):
            continue
        half_max = smooth > base+amplitude/2
        if half_max.sum() > .25*line.size or peak < 2 or peak > line.size-3:
            continue
        idx = np.arange(max(0, peak-3), min(line.size, peak+4))
        centre = np.average(idx, weights=(smooth[idx]-base).clip(min=0)) + left + .5
        _, x, y = lab_rays(g, (centre-g.beam_center_x_px)*g.pixel_size_x_m,
                           (g.beam_center_y_px-rows.mean()-.5)*g.pixel_size_y_m)
        xs.append(x)
        ys.append(y)
    if len(xs) < 8 or np.ptp(ys) < .10*height*g.pixel_size_y_m:
        return None
    slope, intercept, low, high = theilslopes(xs, ys)
    error = max(np.degrees(g.pixel_size_x_m/np.ptp(ys)), np.degrees(np.arctan(high)-np.arctan(low))/3.92)
    value = np.degrees(np.arctan(slope))
    if abs(intercept) > max(3*g.pixel_size_x_m, .015*width*g.pixel_size_x_m) or abs(value) > 8 or error > 3:
        return None
    return float(value), float(error)
