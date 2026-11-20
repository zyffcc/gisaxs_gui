'Shared dense-underlayer module (proposal 2026-11-08, gaps_20261108 SYNTHESIS 1.3 A / S2b).'

from __future__ import annotations

import os

import numpy as np

import iam as M
import iam2 as M2
from iam import RE, LAM, GH_X, GH_W, dc, rho_e, stack, s_para, lma_D, pref

GL4_U, GL4_W = M2.GL4_U, M2.GL4_W
S_VOID = 0.25
S_BUR, A_BUR = 0.25, 1.0
KD_BUR, OM_BUR = 1.07, 0.35
NBUR = int(os.environ.get("IAM_NBUR", "24"))
DZ_BUR = 0.5


def _supported(m):
    if hasattr(m, "cb"):
        raise NotImplementedError("underlayer terms: Model4c / GrainsB only (GrainsC, Model4d/e/f not supported)")


def _alpha(mdl):
    return np.r_[mdl.ai, np.maximum(mdl.af_c, 1e-4)]


def _zeros(mdl):
    return np.zeros((len(mdl.af_c), len(mdl.q)))


def _nodes(kzs, A, B, ts, js):
    'Total fields at the 4 Gauss-Legendre nodes of the contiguous layers js (stack indices): E (nodes, 1 + n_af),'
    E, z, w = [], [], []
    top = 0.0
    for j in js:
        for u, wu in zip(GL4_U, GL4_W):
            dz = u * ts[j]
            E.append(A[j] * np.exp(1j * kzs[j] * dz) + B[j] * np.exp(-1j * kzs[j] * dz))
            z.append(top + dz); w.append(wu * ts[j])
        top += ts[j]
    return np.array(E), np.array(z), np.array(w)


def _pmat(E, wz):
    'Field-product weights (n_af, nodes) as in iam4.Model4.grains_sliced.'
    return (E[:, 1:] * (E[:, 0] * wz)[:, None]).T


def void_pop(Rv, omv, fv, q, s=S_VOID):
    'Columnar voids of area fraction fv: radii / weights, number density, hexagonal spacing, paracrystal S.'
    Rj, wj = M.ln_nodes(Rv, s, GH_X, GH_W)
    Rm = float(np.sum(wj * Rj))
    n = fv / float(np.sum(wj * np.pi * Rj ** 2))
    D = Rm * np.sqrt(2 * np.pi / (np.sqrt(3) * max(fv, 1e-3)))
    S = s_para(np.asarray(q)[None, :], lma_D(D, Rj, Rm)[:, None], omv)
    return dict(Rj=Rj, wj=wj, n=n, D=D, S=S)


def void_term(mdl, m, slab0, convention="full"):
    'C2: dsigma/dOmega (n_af, nq) of the columnar voids in the base slices of slab0 (unsmeared, like the grain term).'
    if mdl.void is None:
        return 0.0
    _supported(m)
    nt = m.n_top if getattr(m, "n_top", None) is not None else len(slab0)
    nb = len(slab0) - nt
    if nb <= 0 or getattr(m, "t_b", 0.0) <= 1e-6 or m.rb >= 1.0 - 1e-9:
        return _zeros(mdl)
    Rv, omv = float(mdl.void[0]), float(mdl.void[1])
    sv = float(mdl.void[2]) if len(mdl.void) > 2 and mdl.void[2] is not None else S_VOID
    Lz = float(mdl.void[3]) if len(mdl.void) > 3 and mdl.void[3] is not None else None
    kv = float(mdl.void[4]) if len(mdl.void) > 4 and mdl.void[4] is not None else None
    fv = 1.0 - m.rb if kv is None else kv * (1.0 - m.rb)
    if fv <= 1e-12:
        return _zeros(mdl)
    pop = void_pop(Rv, omv, fv, mdl.q, sv)
    layers = [(mdl.nu_isl * d, t) for d, t in slab0] + mdl._below()
    kzs, A, B, ts = stack(_alpha(mdl), layers, mdl.d_sub)
    E, z, wz = _nodes(kzs, A, B, ts, range(nt + 1, nt + nb + 1))
    P = _pmat(E, wz)
    if Lz is None:
        Pe = P.sum(1, keepdims=True)
        acc = M.incoherent_sum(Pe, pop["Rj"][:, None], mdl.q, pop["n"] * pop["wj"], pop["S"])
    else:
        C = np.exp(-np.abs(z[:, None] - z[None, :]) / max(Lz, 1e-6))
        Qz = np.real(np.einsum("an,nm,am->a", P, C, np.conj(P)))
        sig2 = np.sum((pop["n"] * pop["wj"])[:, None] * M.sig_disk(pop["Rj"], mdl.q) ** 2 * pop["S"], axis=0)
        acc = Qz[:, None] * sig2[None, :]
    rho = rho_e(mdl.s.metal, mdl.f1) - (mdl._fill_rho(m) if hasattr(mdl, "_fill_rho") else 0.0)
    if convention == "avg":
        rho = rho * m.rb
    elif kv is not None:
        rho = rho * (m.rb / (1.0 - fv))
    return RE ** 2 * abs(rho) ** 2 * acc


def buried_pop(h_bur, Rb, q, s=S_BUR, a=A_BUR):
    Rj, wj = M.ln_nodes(Rb, s, GH_X, GH_W)
    Hj = a * Rj
    V = (2 * np.pi / 3) * Rj ** 2 * Hj
    n = max(float(h_bur), 0.0) / float(np.sum(wj * V))
    Rm = float(np.sum(wj * Rj))
    D = KD_BUR / np.sqrt(max(n, 1e-30))
    S = s_para(np.asarray(q)[None, :], lma_D(D, Rj, Rm)[:, None], OM_BUR)
    return dict(Rj=Rj, wj=wj, Hj=Hj, n=n, D=D, S=S, Hmax=float(np.max(Hj)))


def _hemi_area_int(R, H, u0, u1):
    'Integral over depth u in [u0, u1] of the cross-section pi R^2 (1 - (u / H)^2) of an inverted hemispheroid.'
    a, b = np.clip(u0, 0.0, H), np.clip(u1, 0.0, H)
    return np.pi * R ** 2 * ((b - a) - (b ** 3 - a ** 3) / (3 * H ** 2))


def n_slices(H):
    return NBUR if M.SMOOTH else max(4, int(np.ceil(H / DZ_BUR)))


def buried_geometry(mdl, host):
    'Composite stack replacing host[0] (the film the clusters hang into): [film z_bur] + N composite slices over the'
    if not host:
        raise ValueError("C3a (buried clusters) needs a host film under the base: IAM_PFILM=1 on a polymer scan")
    h_bur, Rb, zb = (float(v) for v in mdl.buried[:3])
    pop = buried_pop(h_bur, Rb, mdl.q, float(mdl.buried[3]) if len(mdl.buried) > 3 else S_BUR)
    d_f, t_f = host[0]
    H = pop["Hmax"]
    if zb < 0 or zb + H > t_f + 1e-9:
        raise ValueError(f"C3a: clusters span {zb:.2f}-{zb + H:.2f} nm below the host top, film {t_f:.2f} nm")
    N = n_slices(H)
    e = np.linspace(0.0, H, N + 1)
    area = _hemi_area_int(pop["Rj"][:, None], pop["Hj"][:, None], e[None, :-1], e[None, 1:])
    phi = np.clip(pop["n"] * (pop["wj"] @ area) / np.diff(e), 0.0, 1.0)
    dM = mdl.nu_isl * dc(mdl.s.metal, f1_au=mdl.f1)
    comp = [(d_f + p * (dM - d_f), H / N) for p in phi]
    layers = [(d_f, zb)] + comp + [(d_f, max(t_f - zb - H, 0.0))] + list(host[1:])
    return layers, phi, pop, N


def buried_layers(mdl, host):
    return buried_geometry(mdl, host)[0]


def buried_fluor(mdl, slab, phi, Ise, h_surf):
    'Fluorescence (n_af,) of the surface metal (slab rows: grains + base, mass h_surf) and of the buried clusters (the'
    host = mdl._host()
    below, phib, pop, N = buried_geometry(mdl, host)
    h_bur = float(mdl.buried[0]) if pop["n"] > 0 else 0.0
    if h_surf + h_bur <= 0:
        return np.zeros(len(mdl.af))
    rows = list(slab) + below[: N + 1]
    phi_all = np.r_[np.asarray(phi, float), 0.0, phib]
    Ise_all = M.layer_mid_int(mdl.a_all, list(slab) + below, mdl.d_sub)[: len(rows)]
    dmf = dc(mdl.s.metal, fl=True)
    tail_fl = list(mdl.sub_layers_fl)
    if not tail_fl:
        raise ValueError("C3a fluorescence: no host film at the fluorescence energy (sub_layers_fl)")
    dpf, tpf = tail_fl[0]
    zb, H = float(mdl.buried[2]), H_of(below, N)
    rows_fl = ([(p * mdl.nu_sub * dmf, t) for p, (d, t) in zip(phi, slab)] + [(dpf, zb)]
               + [(dpf + p * (mdl.nu_sub * dmf - dpf), H / N) for p in phib])
    tail = [(dpf, max(tpf - zb - H, 0.0))] + tail_fl[1:]
    return mdl.fluor(M.Morph(1.0, 1.0, 0.1, 0.3, h_surf + h_bur), rows, phi_all, Ise_all, rows_fl=rows_fl, tail_fl=tail)


def H_of(below, N):
    'Total thickness of the N composite slices of a buried_geometry stack.'
    return float(sum(t for _, t in below[1: N + 1]))


def buried_term(mdl, m, slab0, convention="full"):
    'C3a: dsigma/dOmega (n_af, nq) of the buried clusters (fields of the composite slices; unsmeared).'
    if mdl.buried is None:
        return 0.0
    _supported(m)
    host = mdl._host()
    below, phi, pop, N = buried_geometry(mdl, host)
    if pop["n"] <= 0:
        return _zeros(mdl)
    layers = [(mdl.nu_isl * d, t) for d, t in slab0] + below
    kzs, A, B, ts = stack(_alpha(mdl), layers, mdl.d_sub)
    j0 = len(slab0) + 2
    E, u, wz = _nodes(kzs, A, B, ts, range(j0, j0 + N))
    P = _pmat(E, wz)
    if convention == "avg":
        P = P * np.repeat(1.0 - phi, len(GL4_U))[None, :]
    r = M.hemi_radius(u, pop["Rj"], pop["Hj"])
    acc = M.incoherent_sum(P, r, mdl.q, pop["n"] * pop["wj"], pop["S"])
    c = rho_e(mdl.s.metal, mdl.f1) - host[0][0] / pref(LAM)
    return RE ** 2 * abs(c) ** 2 * acc


def top_term(mdl, m, host, d_sub=None, rough=False):
    'TOP: the Model4c grain term (grains, small islands, large clusters on the dense base of m; plus C2 / C3a when set)'
    old = mdl.host, mdl.d_sub
    mdl.host = list(host)
    if d_sub is not None:
        mdl.d_sub = d_sub
    try:
        slab0, _ = m.slab(mdl.s.metal, f1_au=mdl.f1)
        out = mdl.grains_sliced(m, slab0)
        if rough:
            out = out + mdl.base_rough(m, [(mdl.nu_isl * d, t) for d, t in slab0])
        return out
    finally:
        mdl.host, mdl.d_sub = old


def top_model(src, nu_isl=1.0):
    'A Model4c sharing the grids / geometry / metal of another model (e.g. the TiO2 Model2) for top_term.'
    import iam4 as M4
    mdl = M4.Model4c.__new__(M4.Model4c)
    for k in ("s", "f1", "af", "q", "ai", "d_sub"):
        setattr(mdl, k, getattr(src, k))
    mdl.af_c = getattr(src, "af_c", src.af)
    mdl.a_all = np.r_[mdl.ai, np.maximum(mdl.af_c, 1e-4)]
    mdl.nu_isl, mdl.pfilm, mdl.host = float(nu_isl), False, []
    mdl.void = mdl.buried = mdl.rough = None
    return mdl


def parts(mdl, m, slab0):
    out = {}
    if mdl.void is not None:
        out["void"] = void_term(mdl, m, slab0)
    if mdl.buried is not None:
        out["bur"] = buried_term(mdl, m, slab0)
    return out


def describe(mdl, m):
    'Morphology descriptors of the switched-on placements (for fit records).'
    d = {}
    if mdl.void is not None:
        kv = float(mdl.void[4]) if len(mdl.void) > 4 and mdl.void[4] is not None else None
        fv = max(1.0 - m.rb, 0.0) if getattr(m, "t_b", 0.0) > 1e-6 else 0.0
        if kv is not None:
            fv *= kv
        pop = void_pop(float(mdl.void[0]), float(mdl.void[1]), max(fv, 1e-9), mdl.q)
        d.update(void_fraction=fv, Rv=float(mdl.void[0]), omv=float(mdl.void[1]), D_v=float(pop["D"]) if fv > 0 else None,
                 Lz=(float(mdl.void[3]) if len(mdl.void) > 3 and mdl.void[3] is not None else None))
        if kv is not None:
            d["kv"] = kv
    if mdl.buried is not None:
        _, phi, pop, N = buried_geometry(mdl, mdl._host())
        d.update(h_bur=float(mdl.buried[0]), Rbur=float(mdl.buried[1]), zbur=float(mdl.buried[2]), n_bur=pop["n"],
                 D_bur=float(pop["D"]) if pop["n"] > 0 else None, H_bur=pop["Hmax"], phi_bur_max=float(np.max(phi)))
    return d


def _bare(metal="Ag", sub="PNO", t_film=23.2, pfilm=True, af=None, q=(0.3, 0.8, 1.2)):
    import types
    import iam4 as M4
    mdl = M4.Model4c.__new__(M4.Model4c)
    mdl.s = types.SimpleNamespace(metal=metal, sub=sub)
    mdl.f1, mdl.ai, mdl.nu_isl = M.AU_F1_12, 0.40, 1.0
    mdl.af_c = np.linspace(0.02, 1.0, 50) if af is None else np.asarray(af, float)
    mdl.af = mdl.af_c.copy(); mdl.a_all = np.r_[mdl.ai, mdl.af_c]
    mdl.q = np.asarray(q, float)
    mdl.d_sub = dc("Si")
    mdl.sub_layers = [(dc(sub), t_film)] if sub in M.POLY else []
    mdl.pfilm = pfilm
    return mdl


def _grains(rb=0.8, fb=0.5, h=6.0):
    import iam4 as M4
    return M4.GrainsB(3.0, 0.25, 0.6, 0.7, 0.3, h, fb=fb, rb=rb, RL=20.0, aL=0.25, fL=0.02, Rs=1.0, as_=1.0, cs=0.2,
                      oms=0.35)


def selftest():
    import iam4 as M4
    res, ok = {}, True

    def rec(name, good, val):
        nonlocal ok
        res[name] = (bool(good), float(val))
        ok &= bool(good)
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: {float(val):.3g}")


    mdl = _bare()
    m = _grains(); slab0, _ = m.slab("Ag")
    leg = list(mdl.flat_layers()) if mdl.pfilm else []
    rec("off: _below equals the legacy film list", mdl._below() == leg and mdl._host() == leg, 0.0)
    g_off = M4.Model4b.grains_sliced(mdl, m, slab0)
    rec("off: Model4c.grains_sliced bit-identical to Model4b", np.array_equal(mdl.grains_sliced(m, slab0), g_off), 0.0)

    t1 = top_term(mdl, m, mdl.sub_layers)
    mdl0 = _bare(pfilm=False)
    t0 = top_term(mdl0, m, [])
    rec("TOP: host = film equals IAM_PFILM grain term (bit)", np.array_equal(t1, g_off) and mdl.host is None, 0.0)
    rec("TOP: host = [] equals legacy grain term (bit)", np.array_equal(t0, M4.Model4b.grains_sliced(mdl0, m, slab0)), 0.0)

    mdl.void = (2.0, 0.3)
    m1 = _grains(rb=1.0); s1, _ = m1.slab("Ag")
    rec("C2: rb = 1 gives zero", float(np.max(np.abs(void_term(mdl, m1, s1)))) == 0.0, 0.0)
    m2 = _grains(fb=0.0); s2, _ = m2.slab("Ag")
    rec("C2: no base gives zero", float(np.max(np.abs(void_term(mdl, m2, s2)))) == 0.0, 0.0)
    v_full, v_avg = void_term(mdl, m, slab0), void_term(mdl, m, slab0, convention="avg")
    rec("C2: avg / full = rb^2", float(np.max(np.abs(v_avg / v_full - m.rb ** 2))) < 1e-12 and np.all(v_full > 0),
        float(np.max(np.abs(v_avg / v_full - m.rb ** 2))))
    mdl.void = (2.0, 0.3, None, 1e15)
    e_long = float(np.max(np.abs(void_term(mdl, m, slab0) / v_full - 1)))
    rec("C2: vertical correlation L_z -> inf equals the full column", e_long < 1e-10, e_long)
    mdl.void = (2.0, 0.3, None, 1e-9)
    kz_, A_, B_, ts_ = stack(_alpha(mdl), [(mdl.nu_isl * d, t) for d, t in slab0] + mdl._below(), mdl.d_sub)
    E_, _, w_ = _nodes(kz_, A_, B_, ts_, range(m.n_top + 1, len(slab0) + 1))
    inc = np.sum(np.abs(_pmat(E_, w_)) ** 2, axis=1)
    pop_ = void_pop(2.0, 0.3, 1 - m.rb, mdl.q)
    ref_ = RE ** 2 * abs(rho_e("Ag")) ** 2 * inc[:, None] * np.sum((pop_["n"] * pop_["wj"])[:, None]
                                                                    * M.sig_disk(pop_["Rj"], mdl.q) ** 2 * pop_["S"], axis=0)[None, :]
    e_short = float(np.max(np.abs(void_term(mdl, m, slab0) / ref_ - 1)))
    rec("C2: L_z -> 0 equals the node-incoherent sum", e_short < 1e-10, e_short)
    mdl.void = (2.0, 0.3)
    mb = _bare(pfilm=False); mb.nu_isl = 0.0; mb.d_sub = 0j; mb.void = (2.0, 0.3, 0.0)
    vb = void_term(mb, m, slab0)
    kz = M.K0 * np.sin(np.radians(_alpha(mb)))
    Q = kz[0] + kz[1:]
    vz = np.abs((np.exp(1j * Q * m.t_b) - 1) / (1j * Q)) ** 2
    pop = void_pop(2.0, 0.3, 1 - m.rb, mb.q, 0.0)
    sig = np.sum(pop["wj"][:, None] * (2 * np.pi * pop["Rj"][:, None] ** 2 * M.jinc(mb.q[None, :] * pop["Rj"][:, None])) ** 2
                 * pop["S"], axis=0)
    ref = RE ** 2 * abs(rho_e("Ag")) ** 2 * pop["n"] * vz[:, None] * sig[None, :]

    rec("C2: Born limit = analytic cylinders x |int exp(iQz)|^2", float(np.max(np.abs(vb / ref - 1))) < 1e-4,
        float(np.max(np.abs(vb / ref - 1))))


    mdl.void = (2.0, 0.3, None, None, 1.0)
    e_k1 = float(np.max(np.abs(void_term(mdl, m, slab0) / v_full - 1)))
    rec("C2 amp: kv = 1 equals the term without amplitude", e_k1 < 1e-12, e_k1)
    mdl.void = (2.0, 0.3, None, None, 0.0)
    rec("C2 amp: kv = 0 gives zero", float(np.max(np.abs(void_term(mdl, m, slab0)))) == 0.0, 0.0)
    mdl.void = (2.0, 0.3, None, None, 1e-9)
    e_k0 = float(np.max(void_term(mdl, m, slab0) / v_full))
    rec("C2 amp: kv -> 0 is continuous", e_k0 < 1e-6, e_k0)
    mdl.void = (2.0, 0.3, None, None, 0.5)
    fv_ = 0.5 * (1 - m.rb)
    r_ca = float(np.max(np.abs(void_term(mdl, m, slab0) / void_term(mdl, m, slab0, convention="avg") * (1 - fv_) ** 2 - 1)))
    rec("C2 amp: full / avg = 1 / (1 - f_v)^2", r_ca < 1e-12, r_ca)
    d_ = describe(mdl, m)
    rec("C2 amp: descriptor f_v = kv (1 - rb)", abs(d_["void_fraction"] - fv_) < 1e-15 and d_["kv"] == 0.5,
        abs(d_["void_fraction"] - fv_))
    mdl.void = None

    mdl.buried = (0.6, 1.5, 2.0)
    lay, phi, pop, N = buried_geometry(mdl, mdl._host())
    mass = float(np.sum(phi * pop["Hmax"] / N))
    rec("C3a: composite metal mass = h_bur", abs(mass / 0.6 - 1) < 1e-12, abs(mass / 0.6 - 1))
    tsum = sum(t for _, t in lay[: N + 2])
    rec("C3a: film thickness unchanged", abs(tsum - 23.2) < 1e-12, abs(tsum - 23.2))
    rec("C3a: _below uses the composite, _host the bare film", mdl._below() == lay and mdl._host() == leg, 0.0)
    mdl.buried = (0.0, 1.5, 2.0)
    kz_a, A_a, _, _ = stack(_alpha(mdl), [(mdl.nu_isl * d, t) for d, t in slab0] + mdl._below(), mdl.d_sub)
    kz_b, A_b, _, _ = stack(_alpha(mdl), [(mdl.nu_isl * d, t) for d, t in slab0] + leg, mdl.d_sub)
    dE = float(np.max(np.abs(A_a[len(slab0)] - A_b[len(slab0)])))
    rec("C3a: zero mass: fields at the film top equal the plain film", dE < 1e-12, dE)
    rec("C3a: zero mass gives zero", float(np.max(np.abs(buried_term(mdl, m, slab0)))) == 0.0, 0.0)
    mv = _bare(t_film=30.0); mv.sub_layers = [(0j, 30.0)]; mv.d_sub = 0j; mv.nu_isl = 0.0
    mv.buried = (1e-9, 1.5, 2.0, 0.0)
    bt = buried_term(mv, M4.GrainsB(3.0, 0.25, 0.6, 0.7, 0.3, 1e-9, fb=0.0), [(0j, 1.0)])
    pb = buried_pop(1e-9, 1.5, mv.q, 0.0)
    kz = M.K0 * np.sin(np.radians(_alpha(mv)))
    Qz = kz[0] + kz[1:]
    F = M.ff_sep(mv.q, Qz, 1.5, 1.5)
    refb = RE ** 2 * abs(rho_e("Ag")) ** 2 * pb["n"] * np.abs(F) ** 2 * pb["S"][0][None, :]
    err = float(np.max(np.abs(bt / refb - 1)))
    rec("C3a: Born limit = iam.ff_sep (inverted hemispheroid)", err < 1e-5, err)

    mc = _bare(metal="Cu", t_film=40.0); mc.eps_fl, mc.nu_sub = 1.0, 1.0
    mc.sub_layers_fl, mc.d_sub_fl = [(dc("PNO", fl=True), 40.0)], dc("Si", fl=True)
    h = 0.2
    mc.buried = None
    sl = [(mc.nu_isl * 0.05 * dc("Cu"), 0.2)]
    I0 = M.layer_mid_int(mc.a_all, sl + mc.sub_layers, mc.d_sub)[:1]
    f_surf = mc.fluor(M.Morph(1.0, 1.0, 0.1, 0.3, h), sl, np.array([0.05]), I0)
    mc.buried = (h - 1e-6, 0.6, 0.0)
    slb = [(mc.nu_isl * 5e-6 * dc("Cu"), 0.2)]
    I0b = M.layer_mid_int(mc.a_all, slb + mc.sub_layers, mc.d_sub)[:1]
    f_bur = buried_fluor(mc, slb, np.array([5e-6]), I0b, 1e-6)
    above = mc.af >= 0.3
    err = float(np.max(np.abs(f_bur / f_surf - 1)[above]))
    rec("C3a: Cu fluorescence of buried mass ~ surface (tiny clusters at the top, alpha_f >= 0.3)", err < 0.03, err)
    try:
        ms = _bare(sub="Si", pfilm=False); ms.buried = (0.5, 1.5, 0.0); buried_layers(ms, ms._host()); si_ok = False
    except ValueError:
        si_ok = True
    rec("C3a: no host film refused", si_ok, 0.0)
    try:
        mc = M4.GrainsC(3.0, 0.25, 0.6, 0.7, 0.3, 6.0, fb=0.5, rb=0.8, cb=0.5); mdl.buried = None; mdl.void = (2.0, 0.3)
        sc, _ = mc.slab("Ag"); void_term(mdl, mc, sc); gc_ok = False
    except NotImplementedError:
        gc_ok = True
    mdl.void = None
    rec("GrainsC refused", gc_ok, 0.0)

    mp = _bare(metal="Cu", t_film=40.0); mp.rho_top = rho_e("PNO"); mp.void = mp.buried = mp.host = None
    gp = M4.GrainsB(1.5, 0.25, 0.4, 0.5, 0.3, 4.0, fb=0.9, rb=0.6)
    gp.fill_d = dc("PNO")
    sl, _ = gp.slab("Cu")
    nb = len(sl) - gp.n_top
    db = sl[-1][0]
    rec("PFILL: base decrement = rb d_M + (1 - rb) d_P", abs(db - (0.6 * dc("Cu") + 0.4 * dc("PNO"))) < 1e-15,
        float(abs(db - (0.6 * dc("Cu") + 0.4 * dc("PNO")))))
    mp._fill_dt = (1 - gp.rb) * gp.t_b
    vol = sum(t for _, t in mp._below()) + gp.t_b * (1 - gp.rb)
    rec("PFILL: polymer volume kept (film below + filled share of the base)", abs(vol - 40.0) < 1e-12, abs(vol - 40.0))
    mp.dom = (25.0, 8.0, 0.0)
    rec("PDOM: A_d = 0 is off", mp.domain_term(gp, sl) is None, 0.0)
    mp.dom = (25.0, 8.0, 0.1); d1 = mp.domain_term(gp, sl)
    mp.dom = (25.0, 8.0, 0.2); d2 = mp.domain_term(gp, sl)
    e_lin = float(np.max(np.abs(d2 / d1 - 2.0)))
    rec("PDOM: linear in A_d", e_lin < 1e-12, e_lin)
    mp.dom = (25.0, 8.0, 0.1); rho_p, mp.rho_top = mp.rho_top, 0.0
    dv = mp.domain_term(gp, sl)
    mp.rho_top = rho_p
    want = (abs(rho_e("Cu", mp.f1) - rho_e("PNO")) / abs(rho_e("Cu", mp.f1))) ** 2
    e_c = float(np.max(np.abs(d1 / dv / want - 1)))
    rec("PDOM: contrast against the polymer = |rho_M - rho_P|^2 / |rho_M|^2 at fixed fields", e_c < 1e-12, e_c)
    mp._fill_dt, mp.dom = 0.0, None
    print("UNDERLAYER SELFTEST " + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
