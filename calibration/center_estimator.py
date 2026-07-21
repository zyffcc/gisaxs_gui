from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .preprocessing import AnalysisImage
from .radial_profile import calculate_radial_profile


@dataclass
class CenterProposal:
    x_px: float
    y_px: float
    score: float
    method: str
    support: float = 0.0


def _beamstop_proposals(source_data: np.ndarray) -> list[CenterProposal]:
    """Find GISAXS direct/reflected beamstops along the intense beam column.

    A beamstop is an unusually dark, compact region embedded in an illuminated
    vertical streak.  Opening removes the thin support wires while retaining
    the 20-100 px beamstop bodies.  When two aligned stops are present, the
    lower one is the direct-beam center and receives the strongest support.
    """
    try:
        import cv2
    except Exception:
        return []
    source = np.asarray(source_data)
    if source.ndim != 2 or min(source.shape) < 128:
        return []
    finite_positive = np.isfinite(source) & (source > 0)
    log_image = np.zeros(source.shape, dtype=np.float32)
    log_image[finite_positive] = np.log1p(source[finite_positive].astype(np.float32, copy=False))
    counts = np.maximum(1, finite_positive.sum(axis=0))
    column_score = log_image.sum(axis=0) / counts
    column_score = gaussian_filter1d(column_score.astype(np.float32), 3.0)
    margin = max(10, source.shape[1] // 20)
    beam_x = float(margin + np.argmax(column_score[margin:-margin]))

    # log(1 + counts) < 3 corresponds to a dark stop for photon-counting
    # detectors while remaining independent of the bright scattering scale.
    low = (log_image < 3.0).astype(np.uint8)
    low = cv2.morphologyEx(
        low, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(low, 8)
    height, width = source.shape
    found: list[tuple[float, float, int]] = []
    for index in range(1, count):
        x, y, component_w, component_h, area = map(int, stats[index])
        center_x, center_y = map(float, centroids[index])
        if not (0.35 * height < center_y < 0.96 * height):
            continue
        if abs(center_x - beam_x) > max(55.0, 0.05 * width):
            continue
        if not (15 <= component_w <= 120 and 15 <= component_h <= 120 and area >= 180):
            continue
        aspect = component_w / max(component_h, 1)
        if not 0.35 <= aspect <= 2.8:
            continue
        found.append((center_x, center_y, area))
    if not found:
        return []
    # Merge threshold fragments and keep aligned beamstops only.
    found.sort(key=lambda item: item[1])
    unique: list[tuple[float, float, int]] = []
    for item in found:
        if not any(np.hypot(item[0] - old[0], item[1] - old[1]) < 20 for old in unique):
            unique.append(item)
    aligned = [item for item in unique if abs(item[0] - np.median([v[0] for v in unique])) < 35]
    if not aligned:
        return []
    direct = max(aligned, key=lambda item: item[1])
    proposals = [CenterProposal(direct[0], direct[1], 0.0, "direct beamstop", 1.0)]
    for item in aligned:
        if item is not direct:
            proposals.append(CenterProposal(item[0], item[1], 0.0, "reflected beamstop", 0.45))
    return proposals[:3]


def _profile_sharpness(signal: np.ndarray, valid: np.ndarray, x: float, y: float) -> float:
    profile = calculate_radial_profile(signal, valid, x, y)
    smooth = gaussian_filter1d(profile.intensity, 1.1)
    usable = (profile.counts >= 3) & (profile.coverage >= 0.015)
    if usable.sum() < 15:
        return -1e9
    values = smooth[usable]
    noise = float(np.median(np.abs(values - np.median(values)))) + 1e-5
    peaks, props = find_peaks(smooth, prominence=max(2.0 * noise, 1e-5), distance=3)
    if not len(peaks):
        return float(np.percentile(np.abs(np.diff(values)), 95) / noise)
    good = [idx for idx, peak in enumerate(peaks) if usable[peak]]
    prominences = np.asarray(props["prominences"])[good] if good else np.array([0.0])
    top = np.sort(prominences)[-10:]
    return float(np.sum(top / noise) + min(len(top), 8) * 0.5)


def _concentric_arc_proposals(
    signal: np.ndarray,
    valid: np.ndarray,
    stride: int,
) -> list[CenterProposal]:
    """Estimate centers from partial powder arcs, including off-detector centers.

    The image gradient at a powder-ring edge points along a radius of that
    ring. Intersections of many such gradient lines therefore vote for the
    common center even when no complete ring, direct beam, or center pixel is
    visible. Detector-gap boundaries are removed before voting so multi-module
    mosaics do not dominate the estimate.
    """
    try:
        import cv2
    except Exception:
        return []
    if min(signal.shape) < 48 or np.count_nonzero(valid) < 500:
        return []

    eroded = cv2.erode(
        np.asarray(valid, dtype=np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ).astype(bool)
    if np.count_nonzero(eroded) < 300:
        return []
    source = np.asarray(signal, dtype=np.float32)
    smooth = cv2.GaussianBlur(source, (0, 0), 1.0)
    grad_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(grad_x, grad_y)
    signal_cutoff = float(np.percentile(source[eroded], 94.0))
    gradient_cutoff = float(np.percentile(magnitude[eroded], 75.0))
    selected = eroded & (source > signal_cutoff) & (magnitude > max(gradient_cutoff, 1e-6))
    ys, xs = np.nonzero(selected)
    if xs.size < 180:
        return []

    # Strong edges contribute the most useful normals, but retain broad
    # azimuthal/radial sampling instead of keeping only the brightest ring.
    if xs.size > 6500:
        weights = magnitude[ys, xs]
        keep = np.argpartition(weights, -6500)[-6500:]
        xs, ys = xs[keep], ys[keep]
    norms = np.maximum(magnitude[ys, xs].astype(np.float64), 1e-9)
    directions_x = grad_x[ys, xs].astype(np.float64) / norms
    directions_y = grad_y[ys, xs].astype(np.float64) / norms

    pair_count = min(500_000, max(100_000, xs.size * 90))
    rng = np.random.default_rng(0)
    first = rng.integers(0, xs.size, pair_count)
    second = rng.integers(0, xs.size, pair_count)
    x1, y1 = xs[first].astype(float), ys[first].astype(float)
    x2, y2 = xs[second].astype(float), ys[second].astype(float)
    dx1, dy1 = directions_x[first], directions_y[first]
    dx2, dy2 = directions_x[second], directions_y[second]
    denominator = dx1 * dy2 - dy1 * dx2
    usable = np.abs(denominator) > 0.20
    parameter = np.zeros_like(denominator)
    parameter[usable] = (
        (x2[usable] - x1[usable]) * dy2[usable]
        - (y2[usable] - y1[usable]) * dx2[usable]
    ) / denominator[usable]
    intersections_x = x1 + parameter * dx1
    intersections_y = y1 + parameter * dy1

    height, width = signal.shape
    inside = usable & np.isfinite(intersections_x) & np.isfinite(intersections_y)
    inside &= (intersections_x >= -width) & (intersections_x <= 2.0 * width)
    inside &= (intersections_y >= -height) & (intersections_y <= 2.0 * height)
    if np.count_nonzero(inside) < 250:
        return []
    bin_size = float(max(7, round(0.012 * max(height, width))))
    x_edges = np.arange(-width, 2.0 * width + bin_size, bin_size)
    y_edges = np.arange(-height, 2.0 * height + bin_size, bin_size)
    votes, _, _ = np.histogram2d(
        intersections_y[inside], intersections_x[inside], bins=(y_edges, x_edges)
    )
    votes = votes.astype(np.float32)
    if not np.any(votes > 0):
        return []
    local_max = votes == cv2.dilate(votes, np.ones((3, 3), dtype=np.uint8))
    peak_rows, peak_cols = np.nonzero(local_max)
    ordering = np.argsort(votes[peak_rows, peak_cols])[::-1]

    provisional: list[tuple[float, float, float, float]] = []
    for position in ordering:
        row, column = int(peak_rows[position]), int(peak_cols[position])
        vote = float(votes[row, column])
        if vote < max(4.0, 0.12 * float(votes.max())):
            break
        x = 0.5 * (x_edges[column] + x_edges[column + 1])
        y = 0.5 * (y_edges[row] + y_edges[row + 1])
        if any(np.hypot(x - old_x, y - old_y) < 1.7 * bin_size for old_x, old_y, _, _ in provisional):
            continue
        sharpness = _profile_sharpness(signal, valid, x, y)
        provisional.append((x, y, vote, sharpness))
        # Weak, highly partial rings often form a lower-vote but much sharper
        # mode than detector seams. Keep enough modes for the independent
        # radial-sharpness test to recover that case.
        if len(provisional) >= 24:
            break
    if not provisional:
        return []

    max_vote = max(item[2] for item in provisional)
    finite_sharpness = [max(0.0, item[3]) for item in provisional if np.isfinite(item[3])]
    max_sharpness = max(finite_sharpness, default=1.0)
    # Refine the most plausible histogram modes by a small sharpness grid. The
    # voting bin remains deliberately broad to collect weak, broken arcs.
    provisional.sort(
        key=lambda item: 0.55 * item[2] / max(max_vote, 1e-6)
        + 0.45 * max(0.0, item[3]) / max(max_sharpness, 1e-6),
        reverse=True,
    )
    refined: list[tuple[float, float, float, float]] = []
    for x, y, vote, sharpness in provisional[:7]:
        best = (sharpness, x, y)
        for offset_y in (-bin_size, 0.0, bin_size):
            for offset_x in (-bin_size, 0.0, bin_size):
                trial_x, trial_y = x + offset_x, y + offset_y
                trial_score = _profile_sharpness(signal, valid, trial_x, trial_y)
                if trial_score > best[0]:
                    best = (trial_score, trial_x, trial_y)
        refined.append((best[1], best[2], vote, best[0]))
    max_sharpness = max((max(0.0, item[3]) for item in refined), default=1.0)
    proposals: list[CenterProposal] = []
    for x, y, vote, sharpness in refined:
        vote_quality = vote / max(max_vote, 1e-6)
        sharpness_quality = max(0.0, sharpness) / max(max_sharpness, 1e-6)
        # Radial sharpness is the independent physical check; intersection
        # votes alone are easily inflated by module/chip boundaries.
        support = min(0.95, 0.15 + 0.15 * vote_quality + 0.70 * sharpness_quality)
        proposals.append(CenterProposal(
            x * stride,
            y * stride,
            sharpness,
            "concentric arc gradients",
            support,
        ))
    proposals.sort(key=lambda item: (item.support, item.score), reverse=True)
    return proposals[:5]


def estimate_center_candidates(
    analysis: AnalysisImage,
    metadata_center: tuple[float, float] | None = None,
    max_preview_pixels: int = 180_000,
    source_data: np.ndarray | None = None,
) -> list[CenterProposal]:
    height, width = analysis.signal.shape
    stride = max(1, int(np.ceil(np.sqrt(analysis.signal.size / max_preview_pixels))))
    signal = analysis.signal[::stride, ::stride]
    valid = analysis.valid[::stride, ::stride]
    sh, sw = signal.shape
    proposals: list[CenterProposal] = _beamstop_proposals(source_data) if source_data is not None else []
    arc_stride = max(1, int(np.ceil(np.sqrt(analysis.signal.size / 600_000))))
    arc_proposals = _concentric_arc_proposals(
        analysis.signal[::arc_stride, ::arc_stride],
        analysis.valid[::arc_stride, ::arc_stride],
        arc_stride,
    )
    proposals.extend(arc_proposals)
    if arc_proposals:
        # The profile of a very short arc changes abruptly within a single
        # voting cell. Sample its immediate neighbourhood so the subsequent
        # standard match can select the correct sub-cell geometry.
        best_arc = max(arc_proposals, key=lambda item: (item.support, item.score))
        local_step = max(12.0, 0.006 * max(height, width))
        for offset_y in (-local_step, 0.0, local_step):
            for offset_x in (-local_step, 0.0, local_step):
                if offset_x == 0.0 and offset_y == 0.0:
                    continue
                proposals.append(CenterProposal(
                    best_arc.x_px + offset_x,
                    best_arc.y_px + offset_y,
                    best_arc.score,
                    "concentric arc gradients",
                    max(0.0, best_arc.support - 0.035),
                ))
    seeds = [(0.5 * (sw - 1), 0.5 * (sh - 1), "image center")]
    if metadata_center and all(np.isfinite(metadata_center)):
        seeds.append((metadata_center[0] / stride, metadata_center[1] / stride, "metadata"))
    # Rotational phase correlation supplies an independent symmetry estimate.
    try:
        import cv2
        source = signal.astype(np.float32, copy=False)
        shift, response = cv2.phaseCorrelate(np.flipud(np.fliplr(source)), source)
        symmetry_x = 0.5 * (sw - 1 + shift[0])
        symmetry_y = 0.5 * (sh - 1 + shift[1])
        if response > 0.01:
            seeds.append((symmetry_x, symmetry_y, "180-degree symmetry"))
    except Exception:
        pass

    best_x, best_y, best_score, best_method = seeds[0][0], seeds[0][1], -1e9, "radial sharpness"
    for x, y, method in seeds:
        score = _profile_sharpness(signal, valid, x, y)
        support = 0.70 if method == "metadata" else (0.20 if method == "180-degree symmetry" else 0.0)
        proposals.append(CenterProposal(x * stride, y * stride, score, method, support))
        if score > best_score:
            best_x, best_y, best_score, best_method = x, y, score, method

    bounds = (-1.0 * sw, 2.0 * sw, -1.0 * sh, 2.0 * sh)
    for fraction in (0.18, 0.07, 0.025):
        step = fraction * max(sh, sw)
        local_best = (best_score, best_x, best_y)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                x = float(np.clip(best_x + dx * step, bounds[0], bounds[1]))
                y = float(np.clip(best_y + dy * step, bounds[2], bounds[3]))
                score = _profile_sharpness(signal, valid, x, y)
                if score > local_best[0]:
                    local_best = (score, x, y)
        best_score, best_x, best_y = local_best
    proposals.append(CenterProposal(best_x * stride, best_y * stride, best_score, "multi-ring radial sharpness", 0.0))
    proposals.sort(key=lambda item: (item.support, item.score), reverse=True)
    unique: list[CenterProposal] = []
    for item in proposals:
        if not any(np.hypot(item.x_px - old.x_px, item.y_px - old.y_px) < 3.0 for old in unique):
            unique.append(item)
    return unique[:14]
