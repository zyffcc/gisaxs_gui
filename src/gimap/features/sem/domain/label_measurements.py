"""Measure externally segmented instances in the original image pixel grid."""

from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np
from scipy import ndimage

from .models import SemInstanceSegmentation, SemMeasurements, SemParticle
from .segmentation import _check_cancelled, _perimeter, _validate


def measure_labels(image, labels, settings, metadata=None, *, cancelled=None):
    """Preserve instance IDs and measure their full original-resolution support.

    Labels supplied by an adapter already have the exact image shape and zero
    outside the requested ROI. No threshold or connected-component operation is
    applied here: adjacent instances remain separate and retain their IDs.
    """
    _check_cancelled(cancelled)
    raw = image.image
    x0, y0, x1, y1 = _validate(settings, raw.shape)
    instance = SemInstanceSegmentation(labels, metadata or {})
    labels = instance.labels
    if labels.shape != raw.shape:
        raise ValueError("SEM instance-label shape must match the original image shape")
    if (np.any(labels[:y0]) or np.any(labels[y1:])
            or np.any(labels[y0:y1, :x0]) or np.any(labels[y0:y1, x1:])):
        raise ValueError("SEM instance labels must be zero outside the selected ROI")
    _check_cancelled(cancelled)
    cropped = labels[y0:y1, x0:x1]
    # Dense temporary IDs bound scipy's object-index storage by object count,
    # even when a source label uses an ID near the int32 maximum.
    source_ids, inverse = np.unique(cropped, return_inverse=True)
    _check_cancelled(cancelled)
    dense = inverse.reshape(cropped.shape).astype(np.int32)
    if source_ids[0] != 0:
        dense += 1
        source_ids = np.r_[0, source_ids]
    particles = []
    nmpp = float(settings.nm_per_pixel)
    for temporary_id, bounds in enumerate(ndimage.find_objects(dense), 1):
        _check_cancelled(cancelled)
        if bounds is None:
            continue
        local = dense[bounds] == temporary_id
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
            label=int(source_ids[temporary_id]), radius_nm=radius, diameter_nm=2 * radius,
            area_px=area, area_nm2=area * nmpp ** 2,
            centroid_x_px=float(columns.mean() + bounds[1].start + x0 + 0.5),
            centroid_y_px=float(rows.mean() + bounds[0].start + y0 + 0.5),
            circularity=circularity, accepted=not reasons, rejection_reason=";".join(reasons),
        ))
    accepted = [particle.radius_nm for particle in particles if particle.accepted]
    recorded = asdict(settings)
    recorded.update({
        "roi_xyxy": [x0, y0, x1, y1], "applied_threshold": None,
        "threshold_method": "not_applied_to_instance_labels",
        "radius_definition": "sqrt(area_px/pi) * nm_per_pixel",
        "centroid_convention": "pixel corners integer; pixel centers x=column+0.5,y=row+0.5",
        "perimeter_method": "marching_squares_halfway_contour",
        "segmentation_method": settings.method,
        "measurement_grid": "original_image_pixels",
        "instance_segmentation": dict(instance.metadata),
    })
    _check_cancelled(cancelled)
    result = SemMeasurements(
        radius_nm=np.asarray(accepted), particles=tuple(particles),
        source={"path": image.source_path, "kind": "sem_image", **dict(image.metadata),
                "instance_segmentation": dict(instance.metadata)},
        settings=recorded, labels=labels, image=raw,
        summary={"detected_count": len(particles), "accepted_count": len(accepted),
                 "rejected_count": len(particles) - len(accepted),
                 "invalid_pixel_count": int((~np.isfinite(raw[y0:y1, x0:x1])).sum()),
                 "review_required": True, "method": f"{settings.method}_area_equivalent_radius"},
    )
    _check_cancelled(cancelled)
    return result
