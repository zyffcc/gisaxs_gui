"""Count-derived inputs for the effective-host v2 candidate, not a release.

The frozen feature tool is unchanged. Example::

    x = from_counts(Y, N, 12000., .4, 'Au', host_category=None)
    encoded = prepare({k: np.stack([x[k]]) for k in INPUT_KEYS}, 'rowimage')

All four modes share features, support, optional masks and metadata. No layer
density is supplied or inferred by this adapter. Optional references retain
their own pixel/exposure denominators through ``stage5_input.model_input``.
"""
import numpy as np
from .stage5_input import model_input

MODES = ('features', 'qimage', 'hybrid', 'rowimage')
HOST_CLASSES = ('Si', 'oxide_particles', 'polymer')
INPUT_KEYS = ('image', 'pre', 'features', 'mandatory', 'optional', 'known')
FEATURE_SCALE = np.array([1., 1.] * 4 + [.4] + [2.] * 4 + [3.] * 4 + [1.] * 3)


def from_counts(Y, N, energy_ev, alpha_i_deg, metal, *, host_category=None,
                preY=None, preN=None, thickness_nm=None, qcm_rate=None):
    """No optional layer is treated as known by default; zero thickness allowed."""
    x = model_input(Y, N, energy_ev, alpha_i_deg, metal, preY=preY, preN=preN)
    x['optional'] = np.array([-1., 0., 0.], np.float32)
    if host_category not in (None, 'unknown'):
        if host_category not in HOST_CLASSES:
            raise ValueError('host_category must be Si/oxide_particles/polymer/unknown')
        x['optional'][0] = HOST_CLASSES.index(host_category)
        x['known'][0] = True
    for flag, index, value in ((1, 1, thickness_nm), (3, 2, qcm_rate)):
        if value is None:
            continue
        if not np.isfinite(value) or value < 0 or (flag == 3 and value == 0):
            raise ValueError('Thickness must be nonnegative; QCM rate positive')
        x['optional'][index] = value
        x['known'][flag] = True
    return x


def _masked_image(image):
    a = np.asarray(image, np.float64)
    if a.ndim != 4 or a.shape[1:] != (2, 32, 70):
        raise ValueError('Expected batch of 2x32x70 count-derived images')
    mask = a[:, 1] != 0
    if not np.isin(a[:, 1], [0., 1.]).all():
        raise ValueError('Binary image support required')
    value = np.where(mask, a[:, 0], 0.)
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError('Supported log1p rates must be finite and nonnegative')
    return np.stack([value, mask.astype(float)], axis=1)


def row_normalize(image, pre):
    """One q80 per supported exit-angle row; holes stay holes.

    The reference uses the *current* row denominator, preserving its relative
    change. Independent normalisation of a supplied reference would erase it.
    This transforms encoder intensities only, never photon counts or errors.
    """
    image, pre = _masked_image(image), _masked_image(pre)
    rate = np.expm1(image[:, 0]); reference = np.expm1(pre[:, 0])
    mask = image[:, 1].astype(bool); den = np.ones(rate.shape[:2])
    for b in range(len(rate)):
        for row in range(32):
            if mask[b, row].any():
                q80 = float(np.quantile(rate[b, row, mask[b, row]], .8))
                den[b, row] = q80 if q80 > 0 else 1.
    image[:, 0] = np.log1p(rate / den[:, :, None]) * image[:, 1]
    pre[:, 0] = np.log1p(reference / den[:, :, None]) * pre[:, 1]
    return image, pre


def prepare(batch, mode):
    """Return NumPy encoder inputs; unknown placeholders cannot carry signal.

    ``vector`` is 40 masked/scaled feature coordinates + 14 metadata for the
    features/hybrid branch, or metadata alone for the two image branches.
    ``image`` has current/reference intensity+support channels when used.
    This schema is separate from the old 25-coordinate physics posterior.
    """
    if mode not in MODES:
        raise ValueError('Unknown v2 input mode')
    mandatory = np.asarray(batch['mandatory'], np.float64)
    if mandatory.ndim != 2 or mandatory.shape[1] != 5 or not np.isfinite(mandatory).all():
        raise ValueError('Expected finite batch of five mandatory coordinates')
    n = len(mandatory); flags = np.asarray(batch['known'])
    if not np.isin(flags, [0., 1.]).all():
        raise ValueError('Binary optional presence flags required')
    known = flags.astype(bool)
    optional = np.asarray(batch['optional'], np.float64)
    features = np.asarray(batch['features'], np.float64)
    if known.shape != (n, 4) or optional.shape != (n, 3) or features.shape != (n, 40):
        raise ValueError('v2 batch coordinate dimensions differ')
    active = np.column_stack([known[:, 0], known[:, 1], known[:, 3]])
    optional = np.where(active, optional, 0.)
    if not np.isfinite(optional).all():
        raise ValueError('Known optional input must be finite')
    cat = np.zeros((n, 3))
    for b in np.flatnonzero(known[:, 0]):
        value = optional[b, 0]
        if value not in (0., 1., 2.):
            raise ValueError('Known host category must use v2 class IDs')
        cat[b, int(value)] = 1.
    if np.any(known[:, 1] & (optional[:, 1] < 0)) or np.any(known[:, 3] & (optional[:, 2] <= 0)):
        raise ValueError('Invalid known thickness or QCM rate')
    scales = np.column_stack([np.log1p(optional[:, 1] / 30.),
                              np.where(known[:, 3], np.log(np.where(known[:, 3], optional[:, 2], 1.)), 0.)])
    metadata = np.column_stack([mandatory, cat, scales, known.astype(float)])
    fmask = features[:, 20:].copy()
    if not np.isin(fmask, [0., 1.]).all():
        raise ValueError('Binary feature presence required')
    fmask[~known[:, 2], 13:17] = 0.
    f = np.where(fmask != 0, features[:, :20], 0.) / FEATURE_SCALE
    if not np.isfinite(f).all():
        raise ValueError('Observed feature must be finite')
    vector = np.column_stack([f, fmask, metadata]) if mode in ('features', 'hybrid') else metadata
    image = None
    if mode != 'features':
        image = _masked_image(batch['image'])
        p = np.asarray(batch['pre'], np.float64)
        if p.shape != image.shape or len(image) != n:
            raise ValueError('Reference/current batch dimensions differ')
        pre = _masked_image(np.where(known[:, 2, None, None, None], p, 0.))
        if mode == 'rowimage':
            image, pre = row_normalize(image, pre)
        image = np.concatenate([image, pre], axis=1).astype(np.float32)
    return dict(image=image, vector=vector.astype(np.float32), metadata=metadata.astype(np.float32),
                input_mode=mode, schema='effective_v2_input_1', layer_density_output=False)
