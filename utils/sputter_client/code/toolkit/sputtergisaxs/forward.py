'Layered-DWBA forward model for sputtered metal on Si / TiO2-Si / polymer-Si.'
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.special import ndtri

from . import optics
from .formfactor import volume, cap_profile, hemiellipsoid_profile, _gauss_legendre
from .interference import radial_paracrystal
from .multilayer import amplitudes, fields_at_bottom
from .roughness import psd_k_correlation


_JX = np.linspace(0.0, 400.0, 200001)
from scipy.special import j1 as _j1
_JY = np.ones_like(_JX)
_JY[1:] = 2.0 * _j1(_JX[1:]) / _JX[1:]


def _jinc(x):
    return np.interp(x, _JX, _JY, right=0.0)


NBINS = 7


def lognormal_moment(s_log: float, k: int, nbins: int = NBINS) -> float:
    'E[(a/a_med)^k] for the discretised (quantile-bin) lognormal used by MorphState.size_bins.'
    p = (np.arange(nbins) + 0.5) / nbins
    return float(np.mean(np.exp(k * s_log * ndtri(p))))


@dataclass
class SampleParams:
    metal: str = "Au"
    substrate: str = "TiO2"
    layer_thickness: float = 10.0
    layer_density: float = 1.0
    sio2_thickness: float = 1.5
    sigma_top: float = 0.4
    xi_top: float = 20.0
    hurst_top: float = 0.5
    sigma_si: float = 0.3
    xi_si: float = 30.0
    hurst_si: float = 0.5
    conformality: float = 0.0
    sub_gain: float = 1.0
    pore_xi: float = 3.0
    pore_gain: float = 1.0
    alpha_i_deg: float = 0.40
    energy_ev: float = 12000.0

    def to_dict(self):
        return asdict(self)


@dataclass
class MorphState:
    n_islands: float = 0.0
    a_med: float = 2.0
    s_log: float = 0.3
    aspect: float = 0.8
    omega_rel: float = 0.3
    film_thickness: float = 0.0
    film_density: float = 1.0
    film_sigma: float = 0.3
    film_xi: float = 10.0
    film_hurst: float = 0.5
    infil_mass: float = 0.0
    infil_depth: float = 5.0
    infil_decay: float = 0.0
    void_xi: float = 3.0
    void_gain: float = 1.0
    surf_sigma: float = 0.0
    surf_xi: float = 50.0
    surf_hurst: float = 0.5
    shape: str = "cap"
    use_avg_material: bool = True

    def to_dict(self):
        return asdict(self)


    def size_bins(self, nbins: int = NBINS):
        p = (np.arange(nbins) + 0.5) / nbins
        a = self.a_med * np.exp(self.s_log * ndtri(p))
        return a, self.aspect * a, np.full(nbins, 1.0 / nbins)

    def D(self) -> float:
        return float(np.sqrt(2.0 / (np.sqrt(3.0) * self.n_islands))) if self.n_islands > 0 else np.inf

    def mean_volume(self, nbins: int = NBINS) -> float:
        a, h, w = self.size_bins(nbins)
        return float(np.sum(w * volume(a, h, self.shape)))

    def coverage(self, nbins: int = NBINS) -> float:
        a, h, w = self.size_bins(nbins)
        return float(self.n_islands * np.pi * np.sum(w * a * a))

    def mass_thickness(self, nbins: int = NBINS) -> float:
        return float(self.n_islands * self.mean_volume(nbins) + self.film_thickness * self.film_density
                     + self.infil_mass)


INFIL_SLABS = 6


def infil_fmax(sample: SampleParams) -> float:
    'Largest metal volume fraction inside the interlayer: porosity for TiO2, 0.3 for polymers.'
    return max(1.0 - sample.layer_density, 0.05) if sample.substrate == "TiO2" else 0.3


def infil_profile(sample: SampleParams, state: MorphState | None):
    'Metal inside the interlayer as slabs from the top down: ([(thickness, volume fraction)], remaining host nm).'
    L = sample.layer_thickness
    if state is None or state.infil_mass <= 0 or L <= 0 or sample.substrate == "Si":
        return [], L
    fmax = infil_fmax(sample)
    M = min(state.infil_mass, fmax * L)
    if state.infil_decay <= 0:
        zi = min(state.infil_depth, L)
        return [(zi, min(M / zi, fmax))], L - zi
    Ld = state.infil_decay
    Z = min(L, 3.0 * Ld)
    if fmax * Z <= M:
        Z = min(L, M / fmax)
        return [(Z, min(M / Z, fmax))], L - Z
    dz = Z / INFIL_SLABS
    zc = (np.arange(INFIL_SLABS) + 0.5) * dz
    shape = np.exp(-zc / Ld)
    lo, hi = 0.0, fmax / shape[-1]
    for _ in range(50):
        A = 0.5 * (lo + hi)
        if np.minimum(A * shape, fmax).sum() * dz > M:
            hi = A
        else:
            lo = A
    f = np.minimum(0.5 * (lo + hi) * shape, fmax)
    return [(dz, float(fk)) for fk in f], L - Z


def _stack(sample: SampleParams, state: MorphState | None, nbins: int):
    'Return (nds, thicknesses, sigmas, roughness descriptors, index of island zone or None).'
    E = sample.energy_ev
    metal = optics.nd(sample.metal, energy_ev=E)
    nds, thick, sig, rough = [0j], [], [0.0], []
    iz = None
    if state is not None and state.n_islands > 0:
        a, h, w = state.size_bins(nbins)
        H = float(h.max())
        fv = min(state.n_islands * state.mean_volume(nbins) / H, 0.95) if state.use_avg_material else 0.0
        nds.append(optics.mix(metal, 0j, fv))
        thick.append(H)
        sig.append(0.0)
        rough.append((state.surf_sigma, state.surf_xi, state.surf_hurst, "m") if state.surf_sigma > 0 else None)
        iz = 1
    if state is not None and state.film_thickness > 0:
        nds.append(metal * state.film_density)
        thick.append(state.film_thickness)
        sig.append(state.film_sigma)
        rough.append((state.film_sigma, state.film_xi, state.film_hurst, "m"))
    if sample.substrate != "Si" and sample.layer_thickness > 0:
        host = optics.nd(sample.substrate, sample.layer_density, energy_ev=E)
        L = sample.layer_thickness
        slabs, rest = infil_profile(sample, state)
        if slabs:
            for k, (d, f) in enumerate(slabs):
                nds.append(host + f * metal)
                thick.append(d)
                sig.append(sample.sigma_top if k == 0 else 0.0)
                rough.append((sample.sigma_top, sample.xi_top, sample.hurst_top, "s") if k == 0 else None)
            if rest > 0.05:
                nds.append(host)
                thick.append(rest)
                sig.append(0.0)
                rough.append(None)
        else:
            nds.append(host)
            thick.append(L)
            sig.append(sample.sigma_top)
            rough.append((sample.sigma_top, sample.xi_top, sample.hurst_top, "s"))
    nds.append(optics.nd("SiO2", energy_ev=E))
    thick.append(sample.sio2_thickness)
    if sample.substrate == "Si":
        sig.append(sample.sigma_top)
        rough.append((sample.sigma_top, sample.xi_top, sample.hurst_top, "s"))
    else:
        sig.append(0.0)
        rough.append(None)
    nds.append(optics.nd("Si", energy_ev=E))
    sig.append(sample.sigma_si)
    rough.append((sample.sigma_si, sample.xi_si, sample.hurst_si, "s"))
    return nds, thick, sig, rough, iz


def simulate(sample: SampleParams, state: MorphState | None, ang: dict, nbins: int = NBINS,
             nz: int | None = None, lma_kappa: float | None = None, resolution_bins: float = 0.0,
             return_particle_moments: bool = False) -> dict:
    "Intensity components on the pixel grid described by 'ang' (from Geometry.angles)."
    shape = ang["alpha_f"].shape
    af = ang["alpha_f"].ravel()
    qpar = ang["qpar"].ravel()
    E = sample.energy_ev
    k = 2 * np.pi / optics.wavelength_nm(E)
    ai = np.radians(sample.alpha_i_deg)
    nds, thick, sig, rough, iz = _stack(sample, state, nbins)
    amp_i = amplitudes(k, np.array([ai]), nds, thick, sig)
    amp_f = amplitudes(k, np.maximum(af, 1e-6), nds, thick, sig)
    above = af > 0
    pref = (k * k / (4 * np.pi)) ** 2


    Ip = np.zeros_like(af)
    moments = None
    if return_particle_moments:
        moments = {'F2': np.zeros_like(af), 'coherent': np.zeros_like(af),
                   'qpar': qpar.reshape(shape), 'before_interference_and_resolution': True}
    if iz is not None:
        a_s, h_s, w_s = state.size_bins(nbins)
        Di, Ui = fields_at_bottom(amp_i, iz)
        Df, Uf = fields_at_bottom(amp_f, iz)
        kzi = amp_i["kz"][iz][0]
        kzf = amp_f["kz"][iz]
        mnd = optics.nd(sample.metal, energy_ev=E)
        dn2 = -2.0 * ((mnd - nds[iz]).real) + 2j * (mnd - nds[iz]).imag
        if nz is None:
            qzmax = float(np.max(np.abs(kzf))) + abs(kzi)
            nz_eff = int(np.clip(np.ceil(0.5 * qzmax * h_s.max()) + 8, 10, 96))
        else:
            nz_eff = nz
        t, wz = _gauss_legendre(nz_eff)
        prof = cap_profile if state.shape == "cap" else hemiellipsoid_profile
        Fsum = np.zeros(af.size, dtype=np.complex128)
        F2sum = np.zeros(af.size, dtype=np.float64)
        F2_each = [] if lma_kappa is not None else None
        for a, h, w in zip(a_s, h_s, w_s):
            r = prof(a, h, t)
            z = t * h
            A = (wz * h * np.pi * r * r)[:, None] * _jinc(np.outer(r, qpar))


            Ef = np.exp(-1j * np.outer(z, kzf))
            Efi = 1.0 / Ef
            ei = np.exp(-1j * kzi * z)
            eii = 1.0 / ei
            F = (Di * Df * np.einsum("kp,kp,k->p", A, Ef, ei)
                 + Ui * Df * np.einsum("kp,kp,k->p", A, Ef, eii)
                 + Di * Uf * np.einsum("kp,kp,k->p", A, Efi, ei)
                 + Ui * Uf * np.einsum("kp,kp,k->p", A, Efi, eii))
            Fsum += w * F
            F2 = np.abs(F) ** 2
            F2sum += w * F2
            if F2_each is not None:
                F2_each.append(F2)
        D = state.D()
        if return_particle_moments:
            common = pref * abs(dn2) ** 2 * state.n_islands
            moments['F2'] = common * F2sum
            moments['coherent'] = common * np.abs(Fsum) ** 2
            moments['F2'][~above] = 0.0
            moments['coherent'][~above] = 0.0
        om = state.omega_rel * D
        if lma_kappa is None:
            S = radial_paracrystal(qpar, D, om)
            Ip = state.n_islands * (F2sum + np.abs(Fsum) ** 2 * (S - 1.0))
        else:
            Ip = np.zeros_like(af)
            for (a, w, F2) in zip(a_s, w_s, F2_each):
                Ds = D * (a / state.a_med) ** lma_kappa
                Ip += w * F2 * radial_paracrystal(qpar, Ds, state.omega_rel * Ds)
            Ip *= state.n_islands
        Ip = pref * abs(dn2) ** 2 * np.maximum(Ip, 0.0)


    Ir = np.zeros_like(af)
    G, P, keep = [], [], []
    for j in range(1, len(nds)):
        desc = rough[j - 1]
        if desc is None or desc[0] <= 0:
            continue
        s, xi, hh, kind = desc

        Pi = amp_i["T"][j][0] + amp_i["R"][j][0]
        Pf = amp_f["T"][j] + amp_f["R"][j]
        dn2 = -2.0 * (nds[j] - nds[j - 1]).real + 2j * (nds[j] - nds[j - 1]).imag
        qz_up = amp_i["kz"][j - 1][0] + amp_f["kz"][j - 1]
        dw = np.exp(-0.5 * np.abs(qz_up) ** 2 * s ** 2)
        g = dn2 * Pi * Pf * dw
        G.append(g * np.sqrt(sample.sub_gain) if kind == "s" else g)
        P.append(psd_k_correlation(qpar, s, xi, hh))
        keep.append(j)
    if G:
        c = sample.conformality
        for m in range(len(G)):
            Ir += np.abs(G[m]) ** 2 * P[m]
            for l in range(m + 1, len(G)):
                if c > 0:
                    cross = c ** (l - m) * np.sqrt(P[m] * P[l])
                    Ir += 2.0 * np.real(G[m] * np.conj(G[l])) * cross
        Ir = pref * np.maximum(Ir, 0.0)


    Iv = np.zeros_like(af)
    porosity = 1.0 - sample.layer_density
    if sample.substrate == "TiO2" and porosity > 0.02 and sample.layer_thickness > 0:
        j0 = next(j for j in range(1, len(nds)) if thick[j - 1] > 0 and rough[j - 1] is not None and rough[j - 1][-1] == "s")
        skel = optics.nd("TiO2", energy_ev=E)
        metal = optics.nd(sample.metal, energy_ev=E)
        slabs, rest = infil_profile(sample, state)
        if slabs:
            segs = [(j0 + k, d, min(f / porosity, 1.0)) for k, (d, f) in enumerate(slabs)]
            if rest > 0.05:
                segs.append((j0 + len(slabs), rest, 0.0))
        else:
            segs = [(j0, sample.layer_thickness, 0.0)]
        q3 = np.sqrt(qpar ** 2 + np.real(amp_f["kz"][j0] + amp_i["kz"][j0][0]) ** 2)
        P3 = 8 * np.pi * sample.pore_xi ** 3 / (1 + (q3 * sample.pore_xi) ** 2) ** 2
        for j, d, x in segs:
            C = (1 - porosity) * porosity * abs(2 * (skel - x * metal)) ** 2 + porosity * x * (1 - x) * abs(2 * metal) ** 2
            Q = amp_i["kz"][j][0] + amp_f["kz"][j]
            imq = np.maximum(np.imag(Q), 1e-9)
            depth = (1 - np.exp(-2 * imq * d)) / (2 * imq)
            Iv += C * np.abs(amp_i["T"][j][0] * amp_f["T"][j]) ** 2 * depth * P3
        Iv = pref * sample.pore_gain * Iv


    if state is not None and state.film_thickness > 0 and state.film_density < 0.98:
        jf = 2 if iz is not None else 1
        rho = state.film_density
        metal = optics.nd(sample.metal, energy_ev=E)
        Cf = rho * (1 - rho) * abs(2 * metal) ** 2
        q3f = np.sqrt(qpar ** 2 + np.real(amp_f["kz"][jf] + amp_i["kz"][jf][0]) ** 2)
        P3f = 8 * np.pi * state.void_xi ** 3 / (1 + (q3f * state.void_xi) ** 2) ** 2
        Q = amp_i["kz"][jf][0] + amp_f["kz"][jf]
        imq = np.maximum(np.imag(Q), 1e-9)
        depth = (1 - np.exp(-2 * imq * state.film_thickness)) / (2 * imq)
        Iv = Iv + pref * state.void_gain * Cf * np.abs(amp_i["T"][jf][0] * amp_f["T"][jf]) ** 2 * depth * P3f

    Ip[~above] = 0.0
    Ir[~above] = 0.0
    Iv[~above] = 0.0
    Ir = Ir + Iv
    Ip = Ip.reshape(shape)
    Ir = Ir.reshape(shape)


    sig_rc = (resolution_bins, resolution_bins) if np.isscalar(resolution_bins) else tuple(resolution_bins)
    if max(sig_rc) > 0 and len(shape) == 2:
        from scipy.ndimage import gaussian_filter
        Ip = gaussian_filter(Ip, sig_rc, mode="nearest")
        Ir = gaussian_filter(Ir, sig_rc, mode="nearest")
    result = {"particles": Ip, "roughness": Ir, "total": Ip + Ir, "stack": {"nd": nds, "thickness": thick}}
    if return_particle_moments:
        moments['F2'] = moments['F2'].reshape(shape)
        moments['coherent'] = moments['coherent'].reshape(shape)
        result['particle_moments'] = moments
    return result
