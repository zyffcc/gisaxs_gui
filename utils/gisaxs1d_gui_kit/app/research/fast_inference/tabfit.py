"""Table-based physics for fast, accurate fitting over wide size domains.

TabPhysics has the same interface as conditional_fit.fastfit.Physics, so
FastFit's factor caching, variable projection and covariance are reused; only
the slot factors are replaced by lookups in research.fast_inference.tables:

  sphere:            F_sphere(|q| R, sR)                       |q| = sqrt(q_par^2 + qz^2)
  standing cylinder: F_disc(q_par R, sR)                       (qz factor constant along the cut)
  random cylinder:   int_0^1 dc F_disc(|q| R sqrt(1-c^2), sR) F_sinc(|q| h c, sh)

Conventions (constructor flags):
  weighting='intensity' reproduces multishape_v1; 'number' uses number-weighted
  Gaussian size distributions (SasView/BornAgain convention).
  sf='legacy' uses the v1 paracrystal damping exp(-pi q^2 (sD D)^2); sf='hosemann'
  the standard exp(-q^2 w^2 / 2) with w = sD * D (sD then is the relative
  nearest-neighbour spread w/D).
Orientation quadrature for random cylinders adapts to q*h so it is not aliased.

DWBA-lite for spheres on a substrate (option dwba=YonedaDWBA(...)): the Born |F(sqrt(q_par^2+qz^2))|^2
with one fixed qz is replaced by the four-term DWBA amplitude of a particle sitting on the substrate
(centre at height R), in the local monodisperse approximation:
  A(q_par; R) = sum_j c_j exp(-i qz_j R) f(sqrt(q_par^2+qz_j^2) R),
  (c_j, qz_j) = (1, kf+ki), (r_i, kf-ki), (r_f, ki-kf), (r_i r_f, -(kf+ki)),  ki,f = k sin(alpha_i,f),
r = Fresnel reflection coefficient of the substrate, f the normalised sphere amplitude; the size average
<|A|^2> uses number weights times V^2 (the same 4-sigma truncated Gaussian as the tables) with a composite
Gauss-Legendre rule. Checked against BornAgain 23 (monodisperse R = 5 nm on Si, lambda 0.1023 nm,
alpha_i 0.3765 deg): identical shape to <0.5% away from form-factor minima (research/physics_check).
Standing cylinders need no change (their DWBA terms share the in-plane factor); random cylinders keep qz.
"""
from __future__ import annotations

import numpy as np

from research.conditional_fit import fastfit
from research.conditional_fit.codec import LEGACY
from research.fast_inference.tables import FormTable
from research.multishape_v1 import physics


class YonedaDWBA:
    """Four-term DWBA of a particle on a substrate for a cut at exit angle alpha_f (degrees).

    delta, beta: substrate refractive index decrement / absorption at the wavelength (Si at 12.12 keV:
    3.28e-6, 5e-8; delta scales as lambda^2, see si_index())."""

    def __init__(self, wavelength_nm, alpha_i_deg, alpha_f_deg, delta, beta=0.0):
        self.wavelength_nm, self.alpha_i_deg, self.alpha_f_deg = float(wavelength_nm), float(alpha_i_deg), float(alpha_f_deg)
        self.delta, self.beta = float(delta), float(beta)
        k = 2*np.pi/self.wavelength_nm
        ai, af = np.radians(self.alpha_i_deg), np.radians(self.alpha_f_deg)
        ki, kf = k*np.sin(ai), k*np.sin(af)
        ri, rf = self.fresnel(ai), self.fresnel(af)
        self.terms = [(1.0+0j, kf+ki), (ri, kf-ki), (rf, ki-kf), (ri*rf, -(kf+ki))]
        self.qz_vacuum = kf+ki

    def fresnel(self, alpha):
        k = 2*np.pi/self.wavelength_nm
        kz = k*np.sin(alpha)
        ks = k*np.sqrt(np.sin(alpha)**2-2*self.delta+2j*self.beta+0j)
        return complex((kz-ks)/(kz+ks))

    def as_dict(self):
        return dict(model='dwba4_particle_on_substrate', wavelength_nm=self.wavelength_nm, alpha_i_deg=self.alpha_i_deg,
                    alpha_f_deg=self.alpha_f_deg, substrate_delta=self.delta, substrate_beta=self.beta)


def si_index(wavelength_nm):
    """Approximate Si refractive index decrement and absorption at a wavelength (delta ~ lambda^2, beta ~ lambda^4
    scaled from 12.12 keV; adequate for the Fresnel coefficients of a Yoneda cut)."""
    scale = (wavelength_nm/0.1023)**2
    return 3.28e-6*scale, 5.0e-8*scale**2


DWBA_POINTS_PER_PERIOD, DWBA_MIN_POINTS = 48, 256
DWBA_INCOHERENT_R_NM = 60.0   # above this R(1+4 sigma) the DWBA cross terms are averaged out (incoherent sum)


def _sphere_amplitude(x):
    safe = np.where(np.abs(x) < .1, 1.0, x)
    amp = 3*(np.sin(safe)-safe*np.cos(safe))/safe**3
    return np.where(np.abs(x) < .1, 1-x*x/10+x**4/280-x**6/15120, amp)


class TabPhysics:
    def __init__(self, q, order=None, angles=None, qz=0.0, weighting='intensity', sf='legacy', dwba=None):
        self.q = np.abs(np.asarray(q, float))
        self.qz = float(qz)
        self.qt = np.hypot(self.q, self.qz)
        self.weighting, self.sf = weighting, sf
        self.dwba = dwba

        self.sphere = FormTable(1, weighting)
        self.disc = FormTable(3, weighting)
        self.sinc = FormTable(2, weighting)
        self.min_angles = int(angles or 64)
        self._angle_cache = {}

    def _angles(self, h_hi):
        n = int(np.clip(np.ceil(2*self.qt.max()*h_hi/np.pi), self.min_angles, 4096))
        n = int(2**np.ceil(np.log2(n)))      # few distinct sizes -> cache hits
        if n not in self._angle_cache:
            xa, wa = physics.quadrature(n)
            c, ow = (xa+1)/2, wa/2
            s = np.sqrt(1-c**2)
            self._angle_cache[n] = (c, ow, self.qt[None, :]*s[:, None], self.qt[None, :]*c[:, None])
        return self._angle_cache[n]

    def sphere_dwba(self, r, sr):
        """<|A|^2> over the number-weighted size distribution (weights t^6).

        The four terms pair up: qz = +-(kf+ki) share |q+| and qz = +-(ki-kf) share |q-|, so
        A = P+(R) f(|q+| R) + P-(R) f(|q-| R) with per-size phase factors
        P+ = exp(-i q+ R) + r_i r_f exp(i q+ R),  P- = r_i exp(i q- R) + r_f exp(-i q- R)   (q+- = kf+ki, ki-kf)
        and |A|^2 = |P+|^2 f+^2 + |P-|^2 f-^2 + 2 Re(P+ conj P-) f+ f-: two real amplitudes per size node."""
        from research.fast_inference.tables import _gaussian_rule
        (c1, qp), (ci, _), (cf, _), (cif, _) = self.dwba.terms
        qm = -self.dwba.terms[1][1]
        if r*(1+4*sr) > DWBA_INCOHERENT_R_NM:
            # Large particles: the relative phases exp(2 i k_f R) of the terms run over several periods across the
            # size distribution, so the cross terms average out; the two |q| families add incoherently (tables).
            a0, b0 = 1+abs(ci*cf)**2, abs(ci)**2+abs(cf)**2
            return a0*self.sphere(np.hypot(self.q, qp)*r, sr)+b0*self.sphere(np.hypot(self.q, qm)*r, sr)
        # The size-averaged factor is smooth in q on the scale 2 pi / R_max: evaluate it on a uniform grid with
        # DWBA_POINTS_PER_PERIOD points per period (at least DWBA_MIN_POINTS) and interpolate log values.
        period = 2*np.pi/(r*(1+4*sr))
        n_grid = int(np.clip(np.ceil((self.q.max()-self.q.min())/period*DWBA_POINTS_PER_PERIOD), DWBA_MIN_POINTS, len(self.q)))
        grid = self.q if n_grid >= len(self.q) else np.linspace(self.q.min(), self.q.max(), n_grid)
        q_plus, q_minus = np.hypot(grid, qp), np.hypot(grid, qm)
        x_max = float(q_plus.max())*r*(1+4*sr)
        n_panels = int(np.clip(np.ceil(1.5*x_max*8*sr/(2*np.pi)), 2, 400))
        t, w = _gaussian_rule(max(sr, 1e-3), n_panels)
        w = w*t**(6 if self.weighting == 'number' else 0)
        w = w/w.sum()
        radii = r*t
        p_plus = np.exp(-1j*qp*radii)+cif*np.exp(1j*qp*radii)
        p_minus = ci*np.exp(1j*qm*radii)+cf*np.exp(-1j*qm*radii)
        a_pp, a_mm = w*np.abs(p_plus)**2, w*np.abs(p_minus)**2
        a_pm = 2*w*np.real(p_plus*np.conj(p_minus))
        out = np.zeros(len(grid))
        step = max(1, int(2e6//max(len(grid), 1)))
        for start in range(0, len(radii), step):
            rr = radii[start:start+step, None]
            fp = _sphere_amplitude(q_plus[None, :]*rr)
            fm = _sphere_amplitude(q_minus[None, :]*rr)
            sl = slice(start, start+step)
            out += a_pp[sl]@(fp*fp)+a_mm[sl]@(fm*fm)+a_pm[sl]@(fp*fm)
        if grid is self.q:
            return out
        from scipy.interpolate import CubicSpline
        return np.exp(CubicSpline(grid, np.log(np.maximum(out, 1e-300)))(self.q))

    def radial(self, typ, r, sr):
        if typ == 1:
            if self.dwba is not None:
                return self.sphere_dwba(r, sr)
            return self.sphere(self.qt*r, sr)
        if typ == 3:
            return self.disc(self.q*r, sr)
        # random cylinder: radial factor on the orientation grid; the grid is chosen in height()
        return ('rc_radial', r, sr)

    def height(self, h, sh):
        return ('rc_height', h, sh)

    def combine(self, typ, radial, height):
        if typ != 2:
            return radial
        _, r, sr = radial
        _, h, sh = height
        c, ow, qs, qc = self._angles(h*(1+4*sh))
        return ow@(self.disc(qs*r, sr)*self.sinc(qc*h, sh))

    def structure(self, distance, sd):
        q = self.q
        if self.sf == 'hosemann':
            lp = -0.5*(q*sd*distance)**2
        else:
            lp = -np.pi*q*q*(distance*sd)**2
        phi = np.exp(lp)
        sf = -np.expm1(2*lp)/np.maximum(np.expm1(lp)**2+4*phi*np.sin(0.5*q*distance)**2, 1e-30)
        limit = (sd**2) if self.sf == 'hosemann' else 2*np.pi*sd*sd
        return np.where(q == 0, limit, sf)


class TabFit(fastfit.FastFit):
    """FastFit with table physics. Extra keywords: weighting, sf (see module docstring)."""

    def __init__(self, q, y, sigma, types, order=None, angles=None, *, qz=0.0, sf_on=True, res_on=True,
                 codec=LEGACY, weighting='intensity', sf='legacy', lowq='lorentz', dwba=None):
        super().__init__(q, y, sigma, types, qz=qz, sf_on=sf_on, res_on=res_on, codec=codec, lowq=lowq)
        self.ph = TabPhysics(self.q, qz=qz, weighting=weighting, sf=sf, dwba=dwba)
    # Random-cylinder radial/height factors are lazy tuples, recombined by combine(); FastFit's
    # perturbed_column already recombines on any radial/height change and reuses ff for D/sD.


def check_against_fastfit(n=100, seed=0):
    """lnRMSE between TabFit (intensity, legacy SF) and FastFit columns inside the validated v1 domain."""
    from research.multishape_v1.data import native_grid
    rng = np.random.default_rng(seed)
    worst = {1: 0.0, 2: 0.0, 3: 0.0}
    for _ in range(n):
        q, mask, _, _ = native_grid(rng)
        q = q[mask]
        u = rng.uniform(size=32)
        p = physics.decode(u)
        ref_ph = fastfit.Physics(q)
        tab_ph = TabPhysics(q)
        for typ in (1, 2, 3):
            ref = ref_ph.combine(typ, ref_ph.radial(typ, p['R'][0], p['sigma_R'][0]),
                                 ref_ph.height(p['h'][0], p['sigma_h'][0]) if typ == 2 else None)
            tab = tab_ph.combine(typ, tab_ph.radial(typ, p['R'][0], p['sigma_R'][0]),
                                 tab_ph.height(p['h'][0], p['sigma_h'][0]) if typ == 2 else None)
            worst[typ] = max(worst[typ], float(np.sqrt(np.mean((np.log(tab)-np.log(ref))**2))))
    return worst
