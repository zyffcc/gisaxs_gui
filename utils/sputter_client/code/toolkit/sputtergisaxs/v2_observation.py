'Private v2 pixel-frame count contract; no model or structural release.'
from dataclasses import dataclass
import numpy as np
from .stage5_measurement import NATIVE_AF, Q, counts_profiles, describe
from .v2_input import from_counts

COUNT_CONTRACT = 'pixel_frames_shape_reference_1'
TEMPLATE_CONTRACT = 'dimensionless_shape_with_inferred_fluence_gain_1'
SHAPE = (2, len(NATIVE_AF), len(Q))


@dataclass(frozen=True)
class CountFrame:
    Y: np.ndarray
    P: np.ndarray
    photon_counts_known: bool = False
    exposure_s: object = None

    def __post_init__(self):
        y, p = np.asarray(self.Y, float), np.asarray(self.P, float)
        if (y.shape != SHAPE or p.shape != SHAPE or not np.isfinite([y, p]).all()
            or np.any(y < 0) or np.any(p < 0) or np.any((p == 0) & (y != 0))
            or np.any(p != np.rint(p))):
            raise ValueError('Original nonnegative counts and integer pixel-frame support required')
        if not isinstance(self.photon_counts_known, (bool, np.bool_)):
            raise ValueError('Explicit photon-count semantics flag required')
        if self.exposure_s is not None and (isinstance(self.exposure_s, (bool, np.bool_))
            or not np.isscalar(self.exposure_s) or not np.isfinite(self.exposure_s) or self.exposure_s <= 0):
            raise ValueError('Exposure metadata, when known, must be positive')
        y, p = y.copy(), p.copy(); y.setflags(write=False); p.setflags(write=False)
        object.__setattr__(self, 'Y', y); object.__setattr__(self, 'P', p)


def project_frame(mapping, image, *, mask=None, photon_counts_known=False, exposure_s=None):
    'Use existing AtomicMap rays/bins; collect P directly without an exposure.'
    im = np.asarray(image, float)
    if im.shape != mapping.shape or tuple(mapping.atomic_shape) != SHAPE:
        raise ValueError('Image differs from the original geometry map')
    valid = np.isfinite(im) & (im >= 0)
    if mask is not None:
        if np.shape(mask) != im.shape: raise ValueError('Wrong detector mask shape')
        valid &= np.asarray(mask, bool)
    keep = valid.ravel()[mapping.indices]; bins = mapping.bins[keep]
    y = np.bincount(bins, weights=im.ravel()[mapping.indices[keep]], minlength=np.prod(SHAPE)).reshape(SHAPE)
    p = np.bincount(bins, minlength=np.prod(SHAPE)).reshape(SHAPE)
    return CountFrame(y, p, photon_counts_known, exposure_s)


def from_rate_support(Y, N, exposures_s, *, photon_counts_known=False):
    'Private cache unit bridge with proven homogeneous header exposure only.'
    e = np.asarray(exposures_s, float)
    if (e.ndim != 1 or not len(e) or not np.isfinite(e).all() or np.any(e <= 0)
        or not np.all(e == e[0])):
        raise ValueError('Known identical header exposures required for this cache bridge')
    p = np.asarray(N, float) / e[0]
    if not np.isfinite(p).all() or np.max(abs(p - np.rint(p))) > 1e-8:
        raise ValueError('Seconds support does not prove integer pixel-frame support')
    return CountFrame(Y, np.rint(p), photon_counts_known, float(e[0]))


def template_shape(rate, valid):
    "Remove only a template's scalar count unit/dose, retain spectral shape."
    r, v = np.asarray(rate, float), np.asarray(valid)
    if r.shape != v.shape or not np.isin(v, [0, 1]).all():
        raise ValueError('Binary template support required')
    v = v.astype(bool)
    if not np.isfinite(r[v]).all() or np.any(r[v] < 0):
        raise ValueError('Finite nonnegative supported template rate required')
    scale = float(np.quantile(r[v], .8)) if v.any() else 0.
    if scale <= 0 and np.any(v & (r > 0)):
        scale = float(np.quantile(r[v & (r > 0)], .8))
    if scale <= 0: scale = 1.
    return np.where(v, r / scale, 0.), v, scale


def frame_input(full, public, energy_ev, alpha_i_deg, metal, *, pre=None,
                host_category=None, thickness_nm=None, qcm_rate=None):
    'Full encoder image + public features, optional shape-only reference.'
    if not isinstance(full, CountFrame) or not isinstance(public, CountFrame):
        raise ValueError('Explicit pixel-frame observations required')
    if (np.any(public.P > full.P) or np.any(public.Y > full.Y)
        or np.any((full.P == public.P) & (full.Y != public.Y))):
        raise ValueError('Public photons/support must be a subset of full observation')
    x = from_counts(full.Y, full.P, energy_ev, alpha_i_deg, metal,
                    host_category=host_category, thickness_nm=thickness_nm, qcm_rate=qcm_rate)
    hy, hn, vy, vn, _, _ = counts_profiles(public.Y, public.P)
    d = describe(hy, hn, vy, vn)
    x['features'] = np.array(d['features'], copy=True)

    x['features'][13:17] = 0.; x['features'][33:37] = 0.
    if pre is not None:
        if not isinstance(pre, CountFrame): raise ValueError('Explicit reference pixel-frame observation required')
        _, _, _, _, pic, scale = counts_profiles(pre.Y, pre.P)
        rate = np.expm1(pic[0].astype(float)) * scale
        shape, valid, _ = template_shape(rate, pic[1])
        x['pre'] = np.stack([np.log1p(shape), valid.astype(float)]).astype(np.float32)
        x['known'][2] = True

    x.pop('atomic_exposure_sum', None); x.pop('mean_rate_not_second_normalized', None)
    for key in ('Y', 'N', 'VY', 'VN'): x.pop(key, None)
    x.update(count_support_contract=COUNT_CONTRACT, template_contract=TEMPLATE_CONTRACT,
             geometry=dict(energy_ev=float(energy_ev),alpha_i_deg=float(alpha_i_deg),metal=metal),
             intensity_unit='detector_units_per_pixel_frame', support_unit='pixel_frames',
             atomic_pixel_frames=float(full.P.sum()), exposure_s=full.exposure_s,
             photon_count_statistics_known=bool(full.photon_counts_known and public.photon_counts_known),
             reference_photon_statistics_known=bool(pre.photon_counts_known) if pre is not None else None,
             reference_role='shape_only_no_verified_relative_fluence' if pre is not None else 'not_provided',
             full_atomic_Y=full.Y,full_atomic_P=full.P,
             public_atomic_Y=public.Y,public_atomic_P=public.P,
             public_Y=hy,public_N=hn,public_VY=vy,public_VN=vn,
             descriptor=d, structural_release=False)
    return x


def ppc_features(frame, *, AF_band=None, estimate_stat_se=True):
    'Count-semantic guard for the public observation or one predicted draw.'
    from .v2_ppc import extract
    if not isinstance(frame,CountFrame):raise ValueError('Explicit pixel-frame observation required')
    x=extract(frame.Y,frame.P,AF_band=AF_band,
              estimate_stat_se=bool(estimate_stat_se and frame.photon_counts_known))
    x.update(count_support_contract=COUNT_CONTRACT,
             photon_count_statistics_known=bool(frame.photon_counts_known))
    return x
