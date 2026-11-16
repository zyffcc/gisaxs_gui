"""Independent forward model ("truth") of a GISAXS horizontal cut at the Yoneda exit angle.

Written for the physics checks of GIMaP (Physics and Methods, sections 9.10 to 9.12). It shares no code with GIMaP
or the fit1d kit, so fits of its curves with GIMaP's models test GIMaP's approximations, not its implementation.

Units: q in nm^-1, lengths in nm, angles in degrees at the interface.

Physics, from the particle to the detector:
  amplitude    F(q_par, q_z; R) of a full or truncated sphere whose base sits on the substrate (z = 0),
               computed by integrating discs over z (exact for the full sphere: V f(|q| R) exp(i q_z R)).
  DWBA         four terms of a particle on a substrate (Renaud, Lazzari & Leroy 2009, Surf. Sci. Rep. 64, 255):
               A = F(q_par, kf+ki) + r_i F(q_par, kf-ki) + r_f F(q_par, ki-kf) + r_i r_f F(q_par, -(kf+ki)),
               ki,f = k sin(alpha_i,f), r = Fresnel reflection coefficient of the substrate.
               'born_qz' keeps the first term only; 'born_qy' the first term at q_z = 0.
  sizes        number-weighted Gaussian of R (truncated at 0), Gauss-Legendre rule over +-4 sigma.
  interference 1D paracrystal (Hosemann): S = (1-|p|^2) / (1 - 2|p| cos(q D) + |p|^2), |p| = exp(-q^2 s_nn^2 / 2).
               'lma': I = <|A|^2> S.  'da': I = <|A|^2> + |<A>|^2 (S - 1).
  central      a sharp term at q_par = 0 (specular/transmitted beam tail, long-range roughness) and
               a resolvable roughness term c (1 + (q xi)^2)^-(1+H) (self-affine, K-correlation-like).
  resolution   R(q) = (1-eta) Gauss(sigma_g) + eta Lorentz(gamma), normalised; the measured curve is R * I.
  noise        Poisson counts with a chosen peak count.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.special import j1

# Si at 12.12 keV (lambda 0.1023 nm): delta 3.28e-6, beta 5e-8 (CXRO); delta ~ lambda^2, beta ~ lambda^4.
SI_DELTA_REF, SI_BETA_REF, LAMBDA_REF = 3.28e-6, 5.0e-8, 0.1023


@dataclass(frozen=True)
class Geometry:
    wavelength_nm: float = 0.1023
    alpha_i_deg: float = 0.40
    alpha_f_deg: float | None = None          # None: the critical angle of the substrate (Yoneda)
    delta: float | None = None
    beta: float | None = None
    film_delta: float = 0.0                   # optional layer between the particles and the substrate
    film_beta: float = 0.0
    film_thickness: float = 0.0               # nm; 0 = particles directly on the substrate

    def index(self):
        s = (self.wavelength_nm / LAMBDA_REF) ** 2
        d = SI_DELTA_REF * s if self.delta is None else self.delta
        b = SI_BETA_REF * s * s if self.beta is None else self.beta
        return d, b

    @property
    def k(self):
        return 2 * np.pi / self.wavelength_nm

    @property
    def alpha_f(self):
        d, _ = self.index()
        return np.degrees(np.sqrt(2 * d)) if self.alpha_f_deg is None else self.alpha_f_deg

    def fresnel(self, alpha_deg):
        """Reflection coefficient of the surface the particles sit on (Parratt for an optional single layer)."""
        d, b = self.index()
        a = np.radians(alpha_deg)
        s2 = np.sin(a) ** 2
        kz0 = self.k * np.sin(a)
        ks = self.k * np.sqrt(s2 - 2 * d + 2j * b + 0j)
        if self.film_thickness <= 0:
            return (kz0 - ks) / (kz0 + ks)
        k1 = self.k * np.sqrt(s2 - 2 * self.film_delta + 2j * self.film_beta + 0j)
        r01, r12 = (kz0 - k1) / (kz0 + k1), (k1 - ks) / (k1 + ks)
        ph = np.exp(2j * k1 * self.film_thickness)
        return (r01 + r12 * ph) / (1 + r01 * r12 * ph)

    def terms(self):
        """(coefficient, q_z) of the four DWBA terms."""
        ki = self.k * np.sin(np.radians(self.alpha_i_deg))
        kf = self.k * np.sin(np.radians(self.alpha_f))
        ri, rf = self.fresnel(self.alpha_i_deg), self.fresnel(self.alpha_f)
        return [(1.0 + 0j, kf + ki), (ri, kf - ki), (rf, ki - kf), (ri * rf, -(kf + ki))]

    @property
    def qz_vacuum(self):
        return self.terms()[0][1]


@dataclass(frozen=True)
class Film:
    R: float = 4.0                 # number-mean radius of the sphere (nm)
    sR: float = 0.2                # relative standard deviation of R (number distribution)
    shape: float = 2.0             # height / R: 2 full sphere, 1 hemisphere, between: truncated sphere
    cyl_h: float = 0.0             # > 0: standing cylinders of this fixed height (nm) instead of spheres
    D: float = 14.0                # mean nearest-neighbour distance (nm); <= 0 switches interference off
    s_nn: float = 0.2              # relative standard deviation of the neighbour distance (Hosemann w/D)
    coupling: str = 'lma'          # 'lma' or 'da'
    scattering: str = 'dwba'       # 'dwba', 'born_qz', 'born_qy'


@dataclass(frozen=True)
class Instrument:
    sigma_g: float = 0.0           # Gaussian core of the resolution (nm^-1, standard deviation)
    eta: float = 0.0               # Lorentzian fraction of the resolution
    gamma: float = 0.0             # Lorentzian half width (nm^-1)
    central: float = 0.0           # integrated sharp intensity at q = 0 (x 1 nm^-1), relative to the particle peak
    rough: float = 0.0             # amplitude of the resolvable roughness term, relative to the particle peak
    xi: float = 100.0              # roughness correlation length (nm)
    hurst: float = 0.7
    background: float = 0.0        # constant, relative to the particle term at its maximum
    peak_counts: float = 2e4       # Poisson counts at the particle peak of the measured curve (<= 0: no noise)


@dataclass(frozen=True)
class Case:
    geometry: Geometry = field(default_factory=Geometry)
    film: Film = field(default_factory=Film)
    instrument: Instrument = field(default_factory=Instrument)

    def with_(self, **kw):
        g = {k[2:]: v for k, v in kw.items() if k.startswith('g_')}
        f = {k[2:]: v for k, v in kw.items() if k.startswith('f_')}
        i = {k[2:]: v for k, v in kw.items() if k.startswith('i_')}
        return Case(replace(self.geometry, **g), replace(self.film, **f), replace(self.instrument, **i))


# --- particle amplitudes ------------------------------------------------------------------------------------------

def _sphere_f(x):
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-3
    xs = np.where(small, 1.0, x)
    f = 3 * (np.sin(xs) - xs * np.cos(xs)) / xs ** 3
    return np.where(small, 1 - x * x / 10, f)


def _disc(x):
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-6
    xs = np.where(small, 1.0, x)
    return np.where(small, 1.0, 2 * j1(xs) / xs)


_ZNODES, _ZWEIGHTS = np.polynomial.legendre.leggauss(96)


def amplitude(q_par, qz, R, shape=2.0):
    """Form-factor amplitude F (nm^3) of a sphere of radius R truncated to height shape*R, base at z = 0.

    q_par [Q], R [K] -> complex [K, Q]."""
    q_par = np.asarray(q_par, float)[None, :]
    R = np.atleast_1d(np.asarray(R, float))[:, None]
    # Sign: the DWBA matrix element <psi_f|V|psi_i> with psi_i = e^{-i ki z} + r_i e^{+i ki z} and the
    # time-reversed final state gives terms of the form  integral V e^{-i q_par r_par} e^{-i Q z}  (Renaud et al. 2009).
    if abs(shape - 2.0) < 1e-12:
        q = np.hypot(q_par, qz)
        return (4 / 3) * np.pi * R ** 3 * _sphere_f(q * R) * np.exp(-1j * qz * R)
    # truncated sphere: centre at z0 = (shape - 1) R; discs of radius sqrt(R^2 - (z - z0)^2) for z in [0, shape R]
    H = shape * R
    z = 0.5 * H[..., None] * (_ZNODES + 1)                        # [K, 1, N]
    w = 0.5 * H[..., None] * _ZWEIGHTS
    r2 = np.clip(R[..., None] ** 2 - (z - (shape - 1) * R[..., None]) ** 2, 0, None)
    r = np.sqrt(r2)
    phase = np.pi * r2 * np.exp(-1j * qz * z) * w                  # [K, 1, N]
    out = np.empty((R.shape[0], q_par.shape[1]), complex)
    for a in range(0, q_par.shape[1], 256):                        # chunks over q keep memory small
        qc = q_par[:, a:a + 256, None]
        out[:, a:a + 256] = np.sum(phase * _disc(qc * r), axis=-1)
    return out


def cylinder_amplitude(q_par, qz, R, H):
    """Standing cylinder (radius R, height H, base at z = 0): pi R^2 H [2 J1(q R)/(q R)] sinc(Q H/2) e^{-i Q H/2}."""
    q_par = np.asarray(q_par, float)[None, :]
    R = np.atleast_1d(np.asarray(R, float))[:, None]
    x = 0.5 * qz * H
    sinc = np.sinc(x / np.pi)
    return np.pi * R ** 2 * H * _disc(q_par * R) * sinc * np.exp(-1j * x)


def size_rule(R, sR, n=48):
    """Nodes and number weights of a Gaussian of mean R and relative width sR, truncated at 0 (+-4 sigma)."""
    if sR <= 0:
        return np.array([R]), np.array([1.0])
    x, w = np.polynomial.legendre.leggauss(n)
    lo, hi = max(R * (1 - 4 * sR), 1e-3), R * (1 + 4 * sR)
    r = 0.5 * (hi - lo) * (x + 1) + lo
    g = w * np.exp(-0.5 * ((r - R) / (sR * R)) ** 2)
    return r, g / g.sum()


def paracrystal(q, D, s_nn):
    if D <= 0:
        return np.ones_like(q)
    p = np.exp(-0.5 * (q * s_nn * D) ** 2)
    return (1 - p * p) / np.maximum(1 - 2 * p * np.cos(q * D) + p * p, 1e-15)


def particle_intensity(q_par, case: Case):
    """Particle scattering of the film per particle (arbitrary absolute scale), before resolution."""
    g, f = case.geometry, case.film
    q_abs = np.abs(q_par)
    radii, weights = size_rule(f.R, f.sR)
    if f.scattering == 'dwba':
        terms = g.terms()
    elif f.scattering == 'born_qz':
        terms = [(1.0 + 0j, g.qz_vacuum)]
    elif f.scattering == 'born_qy':
        terms = [(1.0 + 0j, 0.0)]
    else:
        raise ValueError(f.scattering)
    if f.cyl_h > 0:
        A = sum(c * cylinder_amplitude(q_abs, qz, radii, f.cyl_h) for c, qz in terms)
    else:
        A = sum(c * amplitude(q_abs, qz, radii, f.shape) for c, qz in terms)      # [K, Q]
    mean_sq = np.einsum('k,kq->q', weights, np.abs(A) ** 2)
    S = paracrystal(q_abs, f.D, f.s_nn)
    if f.coupling == 'lma':
        return mean_sq * S
    mean_amp = np.einsum('k,kq->q', weights, A)
    return mean_sq + np.abs(mean_amp) ** 2 * (S - 1)


# --- instrument -------------------------------------------------------------------------------------------------------

def resolution_kernel(dq, inst: Instrument):
    """Normalised resolution function on a uniform grid of spacing dq (odd length, centred)."""
    width = max(inst.sigma_g * 8, inst.gamma * 60 if inst.eta > 0 else 0, dq)
    n = int(np.ceil(width / dq))
    x = np.arange(-n, n + 1) * dq
    k = np.zeros_like(x)
    if inst.sigma_g > 0:
        k += (1 - inst.eta) * np.exp(-0.5 * (x / inst.sigma_g) ** 2) / (np.sqrt(2 * np.pi) * inst.sigma_g)
    if inst.eta > 0 and inst.gamma > 0:
        k += inst.eta * (inst.gamma / np.pi) / (x * x + inst.gamma ** 2)
    if k.sum() <= 0:
        k = np.zeros_like(x)
        k[n] = 1 / dq
    return x, k / (k.sum() * dq)


def measured_curve(q_out, case: Case, rng=None, oversample=8):
    """Measured horizontal cut at the positive q_out (nm^-1): truth -> resolution -> background -> counts.

    Returns dict(q, I, sigma, I_true, parts) with I in counts (or the noiseless model when peak_counts <= 0)."""
    inst = case.instrument
    q_out = np.asarray(q_out, float)
    dq = (q_out[1] - q_out[0]) / oversample
    _, kern = resolution_kernel(dq, inst)
    half = len(kern) // 2
    q_hi = q_out.max() + (half + 2) * dq
    grid = np.arange(-q_hi, q_hi + dq / 2, dq)                    # symmetric grid including q = 0
    particles = particle_intensity(grid, case)
    ref = float(particles[np.abs(grid) >= q_out.min()].max())     # the particle peak (outside the beam stop)
    rough = inst.rough * ref * (1 + (grid * inst.xi) ** 2) ** (-(1 + inst.hurst))
    smeared_p = np.convolve(particles, kern, mode='same') * dq
    smeared_r = np.convolve(rough, kern, mode='same') * dq
    # a sharp term at q = 0 smeared by the resolution is the resolution function itself
    x_full = grid
    _, kern_at = resolution_kernel(dq, inst)
    central = inst.central * ref * np.interp(x_full, np.arange(-half, half + 1) * dq, kern_at, left=0, right=0) * dq
    if inst.central > 0 and len(kern) == 1:
        central = np.zeros_like(grid)
    total = smeared_p + smeared_r + central
    bg = inst.background * float(particles.max())
    pick = lambda arr: np.interp(q_out, grid, arr)
    I_true = pick(total) + bg
    parts = dict(particles=pick(smeared_p), particles_unsmeared=pick(particles), rough=pick(smeared_r),
                 central=pick(central), background=np.full_like(q_out, bg))
    if inst.peak_counts <= 0:
        return dict(q=q_out, I=I_true, sigma=np.maximum(1e-3 * I_true, 1e-12), I_true=I_true, parts=parts)
    scale = inst.peak_counts / parts['particles'].max()           # counting statistics fixed by the particle peak
    rng = np.random.default_rng(0) if rng is None else rng
    counts = rng.poisson(I_true * scale).astype(float)
    return dict(q=q_out, I=counts, sigma=np.sqrt(np.maximum(counts, 1.0)), I_true=I_true * scale,
                parts={k: v * scale for k, v in parts.items()}, scale=scale)


def number_to_intensity_mean(R, sR, power=6):
    """Mean and relative width of R weighted by R^power (V^2 weighting for power 6)."""
    r, w = size_rule(R, sR, n=200)
    wi = w * r ** power
    wi /= wi.sum()
    m = float(np.sum(wi * r))
    return m, float(np.sqrt(np.sum(wi * (r - m) ** 2)) / m)
