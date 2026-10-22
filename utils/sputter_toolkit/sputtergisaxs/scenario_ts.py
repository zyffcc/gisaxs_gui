"""S4: a Teubner--Strey-type effective two-phase scattering scenario.

This is a positive lateral TS spectrum and a finite-layer DWBA depth response,
not an island model or a verified connected voxel morphology.  The same metal
phase fraction enters the optical reference and the fluctuation variance.
The finite network layer sits at 0 < z < thickness_nm above the effective host.

Lengths are nm and reciprocal lengths nm^-1.  A domain scale is a spectral
parameter, never a metal-island centre spacing, diameter, height, or gap.
No posterior, inverse calibration, or topology certification is implied.
"""
from dataclasses import dataclass

import numpy as np

from .effective_host import HostOptics, chi, field_arrays, field_at
from .formfactor import _gauss_legendre
from .optics import nd, wavelength_nm


@dataclass(frozen=True)
class TSParams:
    """Conditional TS spectrum and homogeneous mean network-layer parameters.

    width_rel is the analytic lateral FWHM/q_peak before detector response.
    fraction is both the mean volume fraction in the represented slab and
    the expected metal area fraction of a stationary planar phase cut.  It
    is not independently inferred surface coverage.  depth_corr_nm sets an
    exponential vertical covariance and does not assert connected topology.
    """
    domain_nm: float = 15.
    width_rel: float = .5
    fraction: float = .45
    thickness_nm: float = 8.
    depth_corr_nm: float = 5.

    def validate(self):
        vals = [self.domain_nm, self.width_rel, self.fraction,
                self.thickness_nm, self.depth_corr_nm]
        if not np.isfinite(vals).all():
            raise ValueError('Finite TS parameters required')
        if not 3 <= self.domain_nm <= 200:
            raise ValueError('TS domain scale outside 3--200 nm implementation domain')
        if not .03 <= self.width_rel <= 1.35:
            raise ValueError('TS FWHM/q_peak outside .03--1.35 implementation domain')
        if not .01 <= self.fraction <= .99:
            raise ValueError('TS metal fraction outside .01--.99 implementation domain')
        if not .2 <= self.thickness_nm <= 60:
            raise ValueError('TS effective network thickness outside .2--60 nm')
        if not .2 <= self.depth_corr_nm <= 120:
            raise ValueError('TS vertical correlation outside .2--120 nm')

    @property
    def q_peak_nm_inv(self):
        return 2 * np.pi / self.domain_nm

    @property
    def width_nm_inv(self):
        return self.width_rel * self.q_peak_nm_inv


def coefficients(parameters):
    """Positive 1/(a2+c1*q^2+c2*q^4), with analytic prescribed peak/FWHM.

    Write its denominator as (q^2-q0^2)^2+b^2.  The strict b>0 prevents
    singular shells.  The implemented width range keeps both half maxima
    at positive q, so width_rel has an unambiguous meaning.
    """
    parameters.validate()
    s = parameters.q_peak_nm_inv ** 2
    r = parameters.width_rel
    b = s * r * np.sqrt(1 - r*r/4)
    return {'a2': s*s + b*b, 'c1': -2*s, 'c2': 1.,
            'q0_squared': s, 'b': b}


def ts_psd(qpar_nm_inv, parameters):
    """Nonnegative TS lateral PSD, integral d^2q/(2*pi)^2 equal to one.

    The returned PSD has units nm^2.  The separate phase variance is
    fraction*(1-fraction); it is included by metal_intensity exactly once.
    """
    q = np.asarray(qpar_nm_inv, dtype=float)
    if not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError('Finite nonnegative lateral q required')
    c = coefficients(parameters)
    s, b = c['q0_squared'], c['b']
    normalizer = 4*np.pi*b / (np.pi/2 + np.arctan(s/b))
    return normalizer / ((q*q-s)**2+b*b)


def build_context(host, parameters, metal='Au', energy_ev=12000., alpha_i_deg=.4):
    """A few-layer reference with the same mean metal fraction as scattering.

    This simplified S4 reference represents a sharp effective host with NC
    interface roughness.  Graded-host transition_nm is unsupported here and
    is rejected explicitly; callers may select an explicitly sharp effective
    host variant.  The network itself has no invented internal interfaces.
    """
    if not isinstance(host, HostOptics):
        raise TypeError('HostOptics required')
    host.validate()
    parameters.validate()
    if host.transition_nm:
        raise ValueError('S4 few-layer reference requires a sharp effective host variant')
    if metal not in ('Au', 'Ag'):
        raise ValueError('S4 currently supports Au or Ag')
    if not np.isfinite([energy_ev, alpha_i_deg]).all() or energy_ev <= 0 or alpha_i_deg <= 0:
        raise ValueError('Finite positive energy and incidence angle required')
    edges = np.unique(np.array([-host.thickness_nm-host.oxide_nm,
                                -host.thickness_nm, 0., parameters.thickness_nm]))
    widths = np.diff(edges)
    mid = (edges[:-1]+edges[1:])/2
    base = np.zeros(len(widths), complex)
    base[(mid < 0) & (mid > -host.thickness_nm)] = host.decrement
    base[mid < -host.thickness_nm] = nd('SiO2', energy_ev=energy_ev)
    fraction = np.where(mid > 0, parameters.fraction, 0.)
    post = base + fraction*(nd(metal, energy_ev=energy_ev)-base)
    topdown = edges[::-1]
    sigma = np.zeros(len(widths)+2)
    for z, rough in [(0., host.roughness_nm),
                     (-host.thickness_nm, host.buried_roughness_nm)]:
        ix = np.flatnonzero(topdown == z)
        if len(ix):
            sigma[int(ix[0])+1] = max(sigma[int(ix[0])+1], rough)
    return {
        'host': host, 'parameters': parameters, 'metal': metal,
        'E': float(energy_ev), 'ai': float(alpha_i_deg), 'edges': edges,
        'base': base, 'phi': fraction, 'thick': widths[::-1].tolist(),
        'sigma': sigma.tolist(),
        'pre_nds': [0j]+base[::-1].tolist()+[nd('Si', energy_ev=energy_ev)],
        'post_nds': [0j]+post[::-1].tolist()+[nd('Si', energy_ev=energy_ev)],
        'mass_nm': parameters.fraction*parameters.thickness_nm,
        'scenario': 'S4', 'connected_topology_verified': False,
        'measurement_hint': {
            'q_peak_nm_inv': parameters.q_peak_nm_inv,
            'width_nm_inv': parameters.width_nm_inv,
            'total_thickness_nm': float(widths.sum())},
        'model_definition': 'positive lateral TS PSD times finite-layer DWBA exponential depth covariance',
        'sharp_effective_host': True,
    }


def _expmoment(z):
    """Integral_0^1 exp(z*x) dx with its analytic critical limit."""
    z = np.asarray(z, complex)
    out = np.empty_like(z)
    small = abs(z) < 1e-5
    zz = z[small]
    out[small] = 1 + zz/2 + zz*zz/6 + zz**3/24 + zz**4/120
    out[~small] = np.expm1(z[~small])/z[~small]
    return out


def _divided_moment(a, b):
    """Stable first divided difference of expmoment, including equal arguments."""
    a, b = np.broadcast_arrays(np.asarray(a, complex), np.asarray(b, complex))
    d = a-b
    out = np.empty_like(d)
    close = abs(d) < 1e-5
    out[~close] = (_expmoment(a[~close])-_expmoment(b[~close]))/d[~close]
    if close.any():
        # Actual critical-limit evaluation, not a new optical discretisation.
        t, w = _gauss_legendre(16)
        m = (a[close]+b[close])/2
        x = d[close, None]*t/2
        sinhc = 1 + x*x/6 + x**4/120
        out[close] = (np.exp(m[:, None]*t)*sinhc*t)@w
    return out


def _depth_covariance(u, v, thickness_nm, depth_corr_nm):
    """Exact four-wave double integral of exp(-abs(z-z')/depth_corr_nm).

    u/v are physical vertical wave numbers, not externally supplied q_z.
    Two triangular integrals give a positive covariance in exact arithmetic.
    No depth binning is used in this path.
    """
    L = thickness_nm
    a = 1j*np.asarray(u, complex)*L
    b = -1j*np.conjugate(np.asarray(v, complex))*L
    r = L/depth_corr_nm
    return L*L*(_divided_moment(a+b, a-r)+_divided_moment(a+b, b-r))


def _positive_depth_reference(context, af_deg, order):
    """Positive GL covariance evaluation, used for critical-wave fallback/reference.

    Subdivision resolves the real field phase and vertical covariance cusp.
    This is a bounded fallback/optional reference for the simplified S4 model,
    not the old full native S3 certification chain.
    """
    p = context['parameters']
    af = np.atleast_1d(af_deg).astype(float)
    inc, rec = field_arrays(context, af)
    kz_i = inc['kz'][1, 0]
    kz_f = rec['kz'][1]
    result = np.empty(len(af))
    for i, aa in enumerate(af):
        phase = abs(kz_i.real)+abs(kz_f[i].real)
        max_width = min(p.depth_corr_nm, np.pi/max(phase, 1e-12))
        cells = max(1, int(np.ceil(p.thickness_nm/max_width)))
        t, w = _gauss_legendre(max(4, int(order)))
        edges = np.linspace(0, p.thickness_nm, cells+1)
        z = (edges[:-1, None]+np.diff(edges)[:, None]*t).ravel()
        weights = (np.diff(edges)[:, None]*w).ravel()
        f = field_at(context, [aa], z)[0]*weights
        # Exact action of the SAME positive exponential kernel on sorted GL
        # points, without an O(N^2) matrix.  Short vertical correlation and a
        # thick layer can otherwise allocate hundreds of MB for a single AF.
        left = np.zeros(len(z), complex)
        decay = np.exp(-np.diff(z)/p.depth_corr_nm)
        for j in range(1, len(z)):
            left[j] = decay[j-1]*(left[j-1]+f[j-1])
        result[i] = float(np.real(np.vdot(f, f+2*left)))
    return result


def depth_response(context, af_deg, *, order=4, reference=False):
    """Positive integrated |DWBA field|^2 covariance response, units nm^2.

    The fast path analytically integrates the homogeneous network layer's four
    waves.  A critical/cancellation fallback uses stable field_at, and the
    explicit reference flag uses that positive quadrature throughout.  order
    affects the fallback/reference only; ordinary fast evaluations are exact
    within this effective separable covariance model.
    """
    af = np.atleast_1d(af_deg).astype(float)
    if not np.isfinite(af).all() or np.any(af <= 0):
        raise ValueError('Finite positive exit angles required')
    if reference:
        return _positive_depth_reference(context, af, order)
    p = context['parameters']
    inc, rec = field_arrays(context, af)
    ki = inc['kz'][1, 0]
    kf = rec['kz'][1]
    di, ui = inc['T'][1, 0], inc['R'][1, 0]
    df, uf = rec['T'][1], rec['R'][1]
    u = np.array([ki+kf, ki-kf, -ki+kf, -ki-kf]).T
    c = np.array([di*df, di*uf, ui*df, ui*uf]).T
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        integral = _depth_covariance(u[:, :, None], u[:, None, :],
                                     p.thickness_nm, p.depth_corr_nm)
        terms = c[:, :, None]*np.conjugate(c[:, None, :])*integral
        response = np.real(terms.sum(axis=(1, 2)))
        abs_sum = abs(terms).sum(axis=(1, 2))
    critical = (abs(ki*p.thickness_nm) < 1e-4) | (abs(kf*p.thickness_nm) < 1e-4)
    bad = ~np.isfinite(response) | critical | (response < 0)
    bad |= (response < 1e-8*np.maximum(abs_sum, 1e-300))
    if np.any(bad):
        response[bad] = _positive_depth_reference(context, af[bad], max(8, int(order)))
    if not np.isfinite(response).all() or np.any(response < 0):
        raise ValueError('Nonfinite or negative TS depth response')
    return response


def metal_intensity(context, af_deg, qpar_nm_inv, *, order=4, reference=False):
    """S4 diffuse intensity on exact q_parallel[AF,q] or explicit separable q.

    Absolute units follow the existing effective forward k^4/(16*pi^2)
    convention; callers supply the count-rate gain once.  Detector response,
    empirical host transport/backgrounds and photon noise are applied outside.
    """
    af = np.atleast_1d(af_deg).astype(float)
    q = np.asarray(qpar_nm_inv, float)
    if q.ndim == 1:
        q = np.broadcast_to(q, (len(af), len(q)))
    elif q.ndim != 2 or q.shape[0] != len(af):
        raise ValueError('Expected q[q] or exact q_parallel[AF,q]')
    p = context['parameters']
    spectrum = ts_psd(q, p)
    depth = depth_response(context, af, order=order, reference=reference)
    contrast = abs(chi(nd(context['metal'], energy_ev=context['E'])))**2
    k = 2*np.pi/wavelength_nm(context['E'])
    return k**4/(16*np.pi**2)*contrast*p.fraction*(1-p.fraction)*depth[:, None]*spectrum


def labels(context):
    """Applicable conditional labels only; all island quantities are missing."""
    p = context['parameters']
    return {
        'scenario': 'S4', 'merged_phase_assumed': True,
        'connected_topology_verified': False,
        'D_nm': None, 'diameter_nm': None, 'height_nm': None, 'gap_nm': None,
        'island_quantity_missing_reason': 'not applicable to effective two-phase S4',
        'coverage_fraction': p.fraction,
        'coverage_definition': 'conditional expected metal area fraction of a stationary planar phase cut; not measured top-surface coverage',
        'features': {'D_corr_intrinsic_nm': p.domain_nm,
                     'width_rel_intrinsic': p.width_rel},
        'parameter_intervals_calibrated': False,
    }


def conditional_geometry_spec(context):
    """Small, serializable recipe for a later *conditional* GRF phase sample.

    A later renderer may use lateral ts_psd times vertical
    2*depth_corr_nm/(1+(q_z*depth_corr_nm)^2) as the Gaussian spectrum, then
    threshold at the desired phase fraction.  Thresholding changes the PSD;
    that geometry requires its own forward check and measured connectivity
    before any certified network export.  This function does not create data.
    """
    p = context['parameters']
    return {
        'scenario': 'S4', 'conditional_sample': True,
        'spectrum': 'TS lateral PSD times exponential vertical covariance',
        'domain_nm': p.domain_nm, 'width_rel': p.width_rel,
        'phase_fraction': p.fraction, 'thickness_nm': p.thickness_nm,
        'depth_corr_nm': p.depth_corr_nm,
        'z_support_nm': [0., p.thickness_nm],
        'thresholding_changes_spectrum': True,
        'connected_topology_verified': False,
        'certified_geometry_export': False,
    }
