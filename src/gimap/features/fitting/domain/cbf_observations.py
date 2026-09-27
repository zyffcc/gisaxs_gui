"""Measured column means and uncertainty from the preprocessed analysis image."""

import numpy as np


def column_observations(image, q_mesh, region, *, selection_mask=None):
    if np.shape(image) != np.shape(q_mesh):
        raise ValueError("CBF image and q grid must have matching shapes")
    r0, r1, x0, x1 = region
    if not (0 <= r0 <= r1 < np.shape(image)[0] and 0 <= x0 <= x1 < np.shape(image)[1]):
        raise ValueError("CBF pixel region is outside the detector")
    data = np.asarray(image, float)[r0 : r1 + 1, x0 : x1 + 1]
    q = np.asarray(q_mesh, float)[r0 : r1 + 1, x0 : x1 + 1]
    valid = np.isfinite(data) & np.isfinite(q)
    if np.any(data[valid] < 0):
        raise ValueError("CBF invalid detector codes must be masked during preprocessing")
    invalid_columns = ~valid.any(axis=0)
    if selection_mask is not None:
        valid &= selection_mask[r0 : r1 + 1, x0 : x1 + 1]
    count = valid.sum(axis=0)
    total = np.where(valid, data, 0).sum(axis=0)
    means = total / np.maximum(count, 1)
    # Independent Poisson counts approximation; not a model-fit quality score.
    sigma = np.sqrt(np.maximum(total, 1)) / np.maximum(count, 1)
    q_mean = np.where(valid, q, 0).sum(axis=0) / np.maximum(count, 1)
    keep = count > 0
    if keep.sum() < 8:
        raise ValueError("Too few measured CBF columns remain outside detector gaps")
    metadata = dict(
        source="native_cbf_columns",
        invalid_columns=int(invalid_columns.sum()),
        measured_columns=int(keep.sum()),
        pixel_region=list(region),
        valid_pixel_counts=count[keep].tolist(),
        intensity_unit="counts_per_pixel",
        uncertainty="Poisson sum/count approximation; additional relative noise is a working tolerance",
        sampling="Native measured columns; no interpolated points across detector gaps",
    )
    return q_mean[keep], means[keep], sigma[keep], metadata
