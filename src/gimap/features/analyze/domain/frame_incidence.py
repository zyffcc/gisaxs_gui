"""External incidence angle from observed detector features and a calibrated direct beam."""

from dataclasses import asdict, dataclass
from itertools import combinations
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from src.gimap.shared.geometry.detector_geometry import roll_transverse
from src.gimap.shared.geometry.q_inverse import positions_of_rays
from src.gimap.shared.geometry.sample_roll import fit_horizon_trace

from .gisaxs import locate_yoneda


@dataclass(frozen=True)
class IncidenceEstimate:
    deg: float
    error_deg: float
    position_px: tuple[float, float]
    details: dict


@dataclass(frozen=True)
class IncidenceMeasurement:
    horizon: IncidenceEstimate | None
    specular: IncidenceEstimate | None
    yoneda: IncidenceEstimate | None
    deg: float | None
    error_deg: float | None
    method: str
    used_estimators: tuple[str, ...]
    roll_deg: float
    critical_angle_deg: float | None
    yoneda_material: str | None = None
    material_wavelength_angstrom: float | None = None
    material_optics_error: str | None = None

    def as_dict(self):
        record = asdict(self)
        if self.yoneda_material:
            record.update(critical_angle_source="material optical constants",
                          material_optics_source="Henke, Gullikson & Davis (CXRO), tabulated bulk density")
        return record


def _arc_contrast(frame, valid, geometry, scattering_angle, radial_fwhm, angular_pixel):
    """Peak above radial background along a cone of constant |q|, relative to its axis.

    A Bragg arc can have a sharp axial peak and exceed the straight side strips
    simply because the ring curves away from those strips. Sample its actual cone
    on both sides, 1.5–3 radial widths from the specular ray.
    """
    transverse = max(radial_fwhm, 2*angular_pixel)
    sectors = np.array([np.linspace(-1., 1., 9)*angular_pixel,
                        np.linspace(-3., -1.5, 9)*transverse,
                        np.linspace(1.5, 3., 9)*transverse])
    phi = sectors[None, :, :] / math.sin(scattering_angle)
    # Three radial samples at the peak and on each side for a local linear background.
    offsets = np.array([-1.75, -1.5, -1.25, -.25, 0., .25, 1.25, 1.5, 1.75])
    angles = (scattering_angle+offsets*radial_fwhm)[:, None, None]
    across, up = roll_transverse(np.sin(angles)*np.sin(phi), np.sin(angles)*np.cos(phi),
                                 -geometry.sample_roll_rad)
    x, y = positions_of_rays(geometry, np.array([np.broadcast_to(np.cos(angles), up.shape), across, up]))
    keep = (x >= 0) & (x < frame.shape[1]) & (y >= 0) & (y < frame.shape[0])
    i, j = np.where(keep, y, 0.).astype(int), np.where(keep, x, 0.).astype(int)
    keep &= valid[i, j]
    if np.any(keep.reshape(3, 3, 3, 9).sum(axis=(1, 3)) < 14):
        return None
    profiles = np.nanmedian(np.where(keep, frame[i, j], np.nan).reshape(3, 3, 3, 9), axis=(1, 3))
    net = profiles[1] - (profiles[0]+profiles[2])/2
    if net[0] <= 0.:
        return None
    return float(max(net[1:])/net[0])


def _specular(frame, valid, g):
    """Look for an isolated peak on the calculated specular ray above both side strips."""
    pitch = min(g.pixel_size_x_m, g.pixel_size_y_m) / g.distance_m
    # 0–3° incidence, with at most one sample per pixel and a bounded working array.
    theta = np.linspace(0., math.radians(6), max(2, min(4*max(frame.shape), int(math.radians(6)/pitch))+1))
    across, up = roll_transverse(np.zeros_like(theta), np.sin(theta), -g.sample_roll_rad)
    x, y = positions_of_rays(g, np.array([np.cos(theta), across, up]))
    dx, dy = x[1]-x[0], y[1]-y[0]
    norm = math.hypot(dx, dy)
    if not np.isfinite(norm) or norm <= 0.:
        return None
    offsets = np.arange(-18, 19)
    xx, yy = x[:, None]-dy/norm*offsets, y[:, None]+dx/norm*offsets
    rows, cols = frame.shape
    inside = (xx >= 0.) & (xx < cols) & (yy >= 0.) & (yy < rows)
    i = np.where(inside, yy, 0.).astype(int)
    j = np.where(inside, xx, 0.).astype(int)
    usable = inside & valid[i, j]
    samples = frame[i, j]

    def strip(region):
        mask = usable[:, region]
        count = mask.sum(axis=1)
        intensity = np.where(mask, samples[:, region], 0.).sum(axis=1, dtype=float)/count.clip(min=1)
        intensity[count < .7*region.sum()] = np.nan
        return intensity

    centre = strip(abs(offsets) <= 2)
    flank = np.maximum(strip(offsets <= -8), strip(offsets >= 8))
    excess = centre - flank
    # Search continuous stretches only. A detector gap cannot create half a spot.
    finite = np.isfinite(excess)
    bounds = np.flatnonzero(np.diff(np.r_[False, finite, False])).reshape(-1, 2)
    candidates = []
    for start, stop in bounds:
        if stop-start < 11:
            continue
        profile = excess[start:stop]
        smoothed = gaussian_filter1d(profile, 1.)
        differences = np.diff(profile)
        noise = 1.4826*np.median(abs(differences-np.median(differences)))/math.sqrt(2)
        peaks, props = find_peaks(smoothed, prominence=max(6*noise, 1e-12))
        widths = peak_widths(smoothed, peaks)[0]
        for peak, width, prominence in zip(peaks, widths, props["prominences"]):
            index = start+peak
            if (index*norm < 5 or width > .45*index or smoothed[peak] < max(6*noise, 1e-12)
                    or min(peak, stop-index-1) < max(width, 2)):
                continue
            # The brightest transverse column must lie on the predicted specular axis.
            line = np.where(usable[index], samples[index], -np.inf)
            if np.max(line[abs(offsets) <= 3]) < np.max(line[abs(offsets) >= 5]):
                continue
            # Count distinct detector pixels before axial smoothing. An isolated hot
            # pixel has an apparent width after interpolation and smoothing.
            row, column = int(y[index]), int(x[index])
            region = np.s_[max(0, row-3):row+4, max(0, column-3):column+4]
            patch = np.where(valid[region], frame[region]-flank[index], 0.).clip(min=0.)
            support = patch > max(2*noise, .15*float(patch.max()))
            if (support.any(axis=0).sum() < 2 or support.any(axis=1).sum() < 2
                    or patch.max() > .55*patch.sum()):
                continue
            arc_ratio = _arc_contrast(frame, valid, g, theta[index], width*theta[1], theta[1])
            if arc_ratio is None or arc_ratio >= .5:
                continue  # the half-height contour extends around a diffraction arc
            candidates.append((float(prominence), index, width, arc_ratio))
    candidates.sort(reverse=True)
    if not candidates or (len(candidates) > 1 and candidates[1][0] >= .75*candidates[0][0]):
        return None
    prominence, index, width, arc_ratio = candidates[0]
    step_deg = math.degrees(theta[1])/2
    error = step_deg * max(1., width/2.355)
    return IncidenceEstimate(math.degrees(theta[index])/2, error, (float(x[index]), float(y[index])),
                             {"scattering_angle_deg": math.degrees(theta[index]), "prominence": prominence,
                              "fwhm_scattering_deg": width*step_deg*2, "arc_to_axis_ratio": arc_ratio})


def _incidence_at_band(g, row, critical_deg):
    # n·r̂=sin αc. The beam column of a tilted/rolled detector can have a transverse component.
    dy = (g.beam_center_y_px-row)*g.pixel_size_y_m
    along, across, up = g.rolled_detector_axes()[1]
    along, across, up = g.distance_m + dy*along, dy*across, dy*up
    plane = math.hypot(along, up)
    sine = math.hypot(plane, across)*math.sin(math.radians(critical_deg))/plane
    if abs(sine) > 1.:
        return math.nan
    return math.degrees(math.atan2(up, along)-math.asin(sine))


def _combine(estimates):
    present = [(name, e) for name, e in estimates.items() if e is not None and e.deg > 0.]
    if not present:
        return None, None, "unavailable", ()
    if any(e is not None and e.deg < -3*e.error_deg for e in estimates.values()):
        # A sharp edge on the wrong side of the calibrated direct beam conflicts
        # with a positive-angle assignment. Preserve both observations for review.
        return None, None, "disagreement", ()
    # A unique largest mutually agreeing group. A tied choice requires a human assignment.
    for size in range(len(present), 0, -1):
        groups = [group for group in combinations(present, size)
                  if all(abs(a.deg-b.deg) <= 3*math.hypot(a.error_deg, b.error_deg)
                         for (_, a), (_, b) in combinations(group, 2))]
        if not groups:
            continue
        if len(groups) != 1:
            return None, None, "disagreement", ()
        names, values = zip(*groups[0])
        angles = np.array([e.deg for e in values])
        errors = np.array([e.error_deg for e in values])
        deg = float(np.average(angles, weights=errors**-2))
        # Same calibrated beam for all features. Preserve the best individual uncertainty;
        # inflate for excess inter-method scatter without assigning a weak edge equal weight.
        variance_factor = np.sum(((angles-deg)/errors)**2)/max(1, size-1)
        error = max(float(min(errors)), float(np.sqrt(variance_factor/np.sum(errors**-2))))
        if deg <= 3*error:
            return None, None, "unavailable", ()
        return deg, error, names[0] if size == 1 else "combined", names


def measure_incidence(data, geometry, *, valid, technique="giwaxs", roll_deg=None, critical_angle_deg=None):
    """Measure independently of the typed incidence, using positive αc only when specified."""
    frame = np.asarray(data)
    if frame.ndim != 2 or np.shape(valid) != frame.shape:
        raise ValueError("Incidence measurement needs a two-dimensional frame and matching validity mask.")
    keep = np.asarray(valid, dtype=bool) & np.isfinite(frame)
    g = geometry.with_incidence(0.).with_sample_roll(geometry.sample_roll_deg if roll_deg is None else roll_deg)
    critical = g.critical_angle_deg if critical_angle_deg is None else float(critical_angle_deg)
    if not 0. <= critical < 90.:
        raise ValueError("The critical angle must be in [0, 90) degrees.")
    trace = fit_horizon_trace(frame, keep, g)
    horizon = None
    if trace is not None:
        horizon = IncidenceEstimate(trace.incidence_deg, trace.incidence_error_deg, trace.intercept_px,
                                    {"roll_deg": trace.roll.deg, "roll_error_deg": trace.roll.error_deg,
                                     "near_beam_deg": trace.near_beam_incidence_deg})
    specular = _specular(frame, keep, g)
    yoneda = None
    if technique == "gisaxs" and min(frame.shape) >= 32:
        band = locate_yoneda(frame, keep, g, search_max_deg=3.+critical, require_peak=True)
        if band is not None:
            row = band.row
            if horizon is not None:
                # A Yoneda peak's illuminated flank supplies a roll direction but does
                # not locate the surface. Exclude edges within the peak and its smoothing span.
                edge = g.with_incidence(horizon.deg).with_sample_roll(trace.roll.deg).horizon_row()
                if abs(edge-row) <= 3*band.error_rows + 2*max(1.3, frame.shape[0]/200):
                    horizon = None
            if critical > 0:
                measured = _incidence_at_band(g, row, critical)
                error = abs(_incidence_at_band(g, row+band.error_rows, critical)-measured)
                if math.isfinite(measured) and math.isfinite(error) and error > 0:
                    yoneda = IncidenceEstimate(measured, error, (g.beam_center_x_px, row),
                        {"critical_angle_deg": critical, "row_error_px": band.error_rows,
                         "side_columns": band.side_columns, "search_rows": band.search_rows})
    result = _combine(dict(horizon=horizon, specular=specular, yoneda=yoneda))
    return IncidenceMeasurement(horizon, specular, yoneda, *result, g.sample_roll_deg, critical or None)
