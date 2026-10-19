"""Threshold/connected-component measurements, with explicit calibration."""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import asdict
import heapq
import math

import numpy as np
from scipy import ndimage

from .models import SemExtractionSettings, SemImage, SemMeasurements, SemParticle


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise CancelledError("SEM extraction cancelled")


def otsu_threshold(image: np.ndarray) -> float:
    values = image[np.isfinite(image)]
    if not values.size:
        raise ValueError("SEM image has no finite pixels")
    low, high = float(values.min()), float(values.max())
    if low == high:
        return low
    counts, edges = np.histogram(values, bins=256, range=(low, high))
    centers = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(counts).astype(float)
    moment = np.cumsum(counts * centers)
    other = weight[-1] - weight
    denominator = weight * other
    score = np.zeros(256)
    good = denominator > 0
    score[good] = (moment[-1] * weight[good] - moment[good] * weight[-1]) ** 2 / denominator[good]
    return float(centers[np.argmax(score)])


def _validate(settings, shape):
    if settings.nm_per_pixel is None or not np.isfinite(settings.nm_per_pixel) or settings.nm_per_pixel <= 0:
        raise ValueError("A positive explicit calibration in nm/pixel is required")
    if settings.min_area_px < 1:
        raise ValueError("Minimum particle area must be at least one pixel")
    if settings.max_area_px is not None and settings.max_area_px < settings.min_area_px:
        raise ValueError("Maximum area must be at least the minimum area")
    if not 0 <= settings.min_circularity <= 1:
        raise ValueError("Minimum circularity must be between zero and one")
    if settings.threshold is not None and not np.isfinite(settings.threshold):
        raise ValueError("Manual threshold must be finite")
    if settings.peak_min_distance_px < 1:
        raise ValueError("Peak separation must be at least one pixel")
    height, width = shape
    roi = settings.roi_xyxy or (0, 0, width, height)
    if (len(roi) != 4 or any(int(v) != v for v in roi)
            or not (0 <= roi[0] < roi[2] <= width and 0 <= roi[1] < roi[3] <= height)):
        raise ValueError("ROI must have integer pixel bounds x0,y0,x1,y1 inside the image")
    return tuple(int(v) for v in roi)


def _split(mask: np.ndarray, separation: int, *, cancelled=None) -> np.ndarray:
    """Priority-flood watershed descending from EDT maxima, restricted to foreground."""
    _check_cancelled(cancelled)
    distance = ndimage.distance_transform_edt(mask)
    _check_cancelled(cancelled)
    maxima = mask & (distance == ndimage.maximum_filter(distance, size=2 * separation + 1))
    plateaus, count = ndimage.label(maxima)
    markers = np.zeros(mask.shape, np.int32)
    for index, region in enumerate(ndimage.find_objects(plateaus), 1):
        _check_cancelled(cancelled)
        if region is None:
            continue
        local = np.argwhere(plateaus[region] == index)
        center = local[np.argmin(np.sum((local - local.mean(axis=0)) ** 2, axis=1))]
        y, x = center + np.array([s.start for s in region])
        markers[y, x] = index
    if not count:
        return np.zeros(mask.shape, np.int32)
    # Keeping the floating distance and restricting every flood step to the mask
    # avoids background paths and quantization artifacts at narrow contacts.
    labels = markers.copy()
    queued = labels > 0
    queue = []
    serial = 0
    height, width = mask.shape
    for y, x in np.argwhere(queued):
        heapq.heappush(queue, (-distance[y, x], serial, int(y), int(x), int(labels[y, x])))
        serial += 1
    visited = 0
    while queue:
        if visited % 1024 == 0:
            _check_cancelled(cancelled)
        visited += 1
        _, _, y, x, label = heapq.heappop(queue)
        for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not queued[ny, nx]:
                queued[ny, nx] = True
                labels[ny, nx] = label
                heapq.heappush(queue, (-distance[ny, nx], serial, ny, nx, label))
                serial += 1
    _check_cancelled(cancelled)
    return labels


def _perimeter(mask):
    """Marching-squares perimeter through the halfway intensity contour (pixel units)."""
    padded = np.pad(mask.astype(np.uint8), 1)
    codes = padded[:-1, :-1] + 2 * padded[:-1, 1:] + 4 * padded[1:, :-1] + 8 * padded[1:, 1:]
    diagonal = math.sqrt(0.5)
    lookup = np.array([0, diagonal, diagonal, 1, diagonal, 1, 2 * diagonal, diagonal,
                       diagonal, 2 * diagonal, 1, diagonal, 1, diagonal, diagonal, 0])
    return float(lookup[codes].sum())


def extract_particles(image: SemImage, settings: SemExtractionSettings, *, cancelled=None) -> SemMeasurements:
    _check_cancelled(cancelled)
    raw = image.image
    x0, y0, x1, y1 = _validate(settings, raw.shape)
    cropped = raw[y0:y1, x0:x1]
    threshold = otsu_threshold(cropped) if settings.threshold is None else float(settings.threshold)
    _check_cancelled(cancelled)
    finite = np.isfinite(cropped)
    foreground = (cropped > threshold if settings.bright_particles else cropped <= threshold) & finite
    _check_cancelled(cancelled)
    labels = (_split(foreground, settings.peak_min_distance_px, cancelled=cancelled) if settings.split_touching
              else ndimage.label(foreground, structure=np.ones((3, 3)))[0])
    _check_cancelled(cancelled)
    particles = []
    nmpp = float(settings.nm_per_pixel)
    for label, bounds in enumerate(ndimage.find_objects(labels), 1):
        _check_cancelled(cancelled)
        if bounds is None:
            continue
        local = labels[bounds] == label
        rows, columns = np.nonzero(local)
        area = int(rows.size)
        perimeter = _perimeter(local)
        _check_cancelled(cancelled)
        circularity = min(1.0, 4 * math.pi * area / perimeter ** 2) if perimeter else 0.0
        border = any(s.start == 0 or s.stop == limit for s, limit in zip(bounds, cropped.shape))
        reasons = []
        if settings.exclude_border and border:
            reasons.append("touches_roi_border")
        if area < settings.min_area_px:
            reasons.append("area_below_minimum")
        if settings.max_area_px is not None and area > settings.max_area_px:
            reasons.append("area_above_maximum")
        if circularity < settings.min_circularity:
            reasons.append("circularity_below_minimum")
        radius = math.sqrt(area / math.pi) * nmpp
        particles.append(SemParticle(
            label=label, radius_nm=radius, diameter_nm=2 * radius,
            area_px=area, area_nm2=area * nmpp ** 2,
            centroid_x_px=float(columns.mean() + bounds[1].start + x0 + 0.5),
            centroid_y_px=float(rows.mean() + bounds[0].start + y0 + 0.5),
            circularity=circularity, accepted=not reasons, rejection_reason=";".join(reasons),
        ))
    _check_cancelled(cancelled)
    full_labels = np.zeros(raw.shape, np.int32)
    full_labels[y0:y1, x0:x1] = labels
    accepted = [particle.radius_nm for particle in particles if particle.accepted]
    recorded = asdict(settings)
    recorded.update({"roi_xyxy": [x0, y0, x1, y1], "applied_threshold": threshold,
                     "threshold_method": "otsu_256_bins" if settings.threshold is None else "manual",
                     "radius_definition": "sqrt(area_px/pi) * nm_per_pixel",
                     "centroid_convention": "pixel corners integer; pixel centers x=column+0.5,y=row+0.5",
                     "perimeter_method": "marching_squares_halfway_contour",
                     "segmentation_method": "inverse_distance_marker_watershed" if settings.split_touching else "8_connected_components"})
    result = SemMeasurements(
        radius_nm=np.asarray(accepted), particles=tuple(particles),
        source={"path": image.source_path, "kind": "sem_image", **dict(image.metadata)},
        settings=recorded, labels=full_labels, image=raw,
        summary={"detected_count": len(particles), "accepted_count": len(accepted),
                 "rejected_count": len(particles) - len(accepted), "invalid_pixel_count": int((~finite).sum()),
                 "review_required": True, "method": "threshold_area_equivalent_radius"},
    )
    _check_cancelled(cancelled)
    return result
