"""Versioned research-only multi-shape inverse-model physics and bounded codec.

Units: q in nm^-1, R/h/D in nm, sigma_R/h/D are relative standard deviations.
Independent nonnegative amplitudes multiply normalized particle form factors.
Sphere=1, isotropic random cylinder=2, vertical cylinder cross-section=3.
Every active particle has its paracrystal structure factor ON in this stage.
All Gaussian size averages are truncated to positive [mu-4*sigma,mu+4*sigma]
and renormalized, including vertical cylinders (a new explicit physics version).
R/h averages are independent; isotropic RC orientation integrates cos(alpha)
uniformly on [0,1]. This module never imports production or training artifacts.

Canonical types have shape [...,4]: ascending active IDs then zero padding.
u has shape [...,32]: [logR,sigma_R,logh,sigma_h,D_fraction,sigma_D,logA]*4
then [logB,logC,logSigmaRes,nuRes], each normalized to [0,1]. D_fraction
interpolates LINEARLY from max(3,2.002*R) to 80 nm. Padding and non-RC height
coordinates are ignored. The 34 combinations are multisets of length 1..4.

decode returns arrays R/sigma_R/h/sigma_h/D/sigma_D/A of shape [...,4],
and B/C/sigma_res/nu_res of shape [...]. encode is the inverse dictionary API.
np_basis returns [...,Q,6]: four unit-amplitude particle columns (padding zero),
constant background, resolution. np_forward returns [...,Q]. tf_forward takes
q[B,Q],u[B,32],types[B,4] and returns [B,Q]; only u needs a gradient.
"""

from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np
from scipy.special import j1

VERSION = "multishape_gaussian4sigma_independent_amplitudes_sf_on_v1"
DEFAULT_ORDER = 64
DEFAULT_ANGLES = 128
Q_VALIDATED_MAX = 4.3
DOMAIN = dict(
    R=[1.0, 10.0],
    sigma_R=[0.08, 0.5],
    h=[3.0, 60.0],
    sigma_h=[0.08, 0.5],
    D="linear fraction from max(3, 2.002*R) to80 nm",
    sigma_D=[0.05, 0.4],
    A=[100.0, 10000.0],
    B=[0.1, 10.0],
    C=[1000.0, 1e6],
    sigma_res=[0.004, 0.05],
    nu_res=[1.5, 10.0],
    units=dict(q="nm^-1", R_h_D="nm", widths="relative standard deviation"),
    structure_factor="ON for every active slot",
    gaussian_truncation="positive four-sigma, renormalized",
    quadrature=dict(distribution=DEFAULT_ORDER, orientation=DEFAULT_ANGLES),
    audit="24 preregistered physics probes vs256/512, worst full-curve lnRMSE 0.0004014; not proof over every continuous parameter",
)
SLOT_NAMES = ("logR", "sigma_R", "logh", "sigma_h", "D_fraction", "sigma_D", "logA")
GLOBAL_NAMES = ("logB", "logC", "logSigmaRes", "nuRes")
NAMES = tuple(f"slot{i}_{name}" for i in range(4) for name in SLOT_NAMES) + GLOBAL_NAMES
SLOT_LOW = np.array([np.log(1.0), 0.08, np.log(3.0), 0.08, 0.0, 0.05, np.log(100.0)])
SLOT_HIGH = np.array([np.log(10.0), 0.5, np.log(60.0), 0.5, 1.0, 0.4, np.log(10000.0)])
LO = np.r_[np.tile(SLOT_LOW, 4), np.log(0.1), np.log(1000.0), np.log(0.004), 1.5]
HI = np.r_[np.tile(SLOT_HIGH, 4), np.log(10.0), np.log(1e6), np.log(0.05), 10.0]
COMBOS = np.array(
    [
        list(c) + [0] * (4 - k)
        for k in range(1, 5)
        for c in combinations_with_replacement((1, 2, 3), k)
    ],
    dtype=np.int32,
)


def active_mask(types):
    types = np.asarray(types)
    if types.shape[-1] != 4 or not np.isin(types, [0, 1, 2, 3]).all():
        raise ValueError("types must end with four IDs in 0..3")
    slots = np.repeat((types > 0)[..., None], 7, axis=-1).astype(float)
    slots[..., 2] = slots[..., 3] = types == 2
    return np.concatenate(
        [slots.reshape(*types.shape[:-1], 28), np.ones((*types.shape[:-1], 4))], axis=-1
    )


def decode(u):
    u = np.asarray(u, dtype=float)
    if u.shape[-1] != 32:
        raise ValueError("u must end with 32 coordinates")
    z = LO + u * (HI - LO)
    slot = z[..., :28].reshape(*z.shape[:-1], 4, 7)
    r = np.exp(slot[..., 0])
    lower = np.maximum(3.0, 2.002 * r)
    return dict(
        R=r,
        sigma_R=slot[..., 1],
        h=np.exp(slot[..., 2]),
        sigma_h=slot[..., 3],
        D=lower + (80.0 - lower) * slot[..., 4],
        sigma_D=slot[..., 5],
        A=np.exp(slot[..., 6]),
        B=np.exp(z[..., 28]),
        C=np.exp(z[..., 29]),
        sigma_res=np.exp(z[..., 30]),
        nu_res=z[..., 31],
    )


def encode(parameters):
    p = {key: np.asarray(value, float) for key, value in parameters.items()}
    r = p["R"]
    if r.shape[-1] != 4:
        raise ValueError("Particle arrays must end in four slots, including padded defaults")
    lower = np.maximum(3.0, 2.002 * r)
    slot = np.stack(
        [
            np.log(r),
            p["sigma_R"],
            np.log(p["h"]),
            p["sigma_h"],
            (p["D"] - lower) / (80.0 - lower),
            p["sigma_D"],
            np.log(p["A"]),
        ],
        axis=-1,
    )
    globals_ = np.stack(
        [np.log(p["B"]), np.log(p["C"]), np.log(p["sigma_res"]), p["nu_res"]], axis=-1
    )
    z = np.concatenate([slot.reshape(*r.shape[:-1], 28), globals_], axis=-1)
    return (z - LO) / (HI - LO)


@lru_cache(maxsize=16)
def quadrature(n):
    return np.polynomial.legendre.leggauss(n)


def _gaussian(mu, width, order):
    sigma = mu * width
    lo, hi = max(mu - 4 * sigma, 0.0), mu + 4 * sigma
    x, w = quadrature(order)
    nodes = lo + (x + 1) * (hi - lo) / 2
    weights = w * np.exp(-0.5 * ((nodes - mu) / sigma) ** 2)
    return nodes, weights / weights.sum()


def _radial(x):
    safe = np.where(abs(x) < 1e-4, 1.0, x)
    return np.where(abs(x) < 1e-4, 1 - x * x / 8 + x**4 / 192, 2 * j1(safe) / safe)


def np_particle(q, typ, r, sr, h, sh, distance, sd, *, order=None, angles=None):
    """Unit particle form * active SF, independent of any other q points."""
    order = DEFAULT_ORDER if order is None else int(order)
    angles = DEFAULT_ANGLES if angles is None else int(angles)
    q = np.abs(np.asarray(q, float))
    if q.ndim != 1:
        raise ValueError("np_particle expects one q vector")
    rr, wr = _gaussian(r, sr, order)
    x = rr[:, None] * q[None, :]
    if typ == 1:
        safe = np.where(abs(x) < 0.1, 1.0, x)
        amplitude = 3 * (np.sin(safe) - safe * np.cos(safe)) / safe**3
        amplitude = np.where(abs(x) < 0.1, 1 - x * x / 10 + x**4 / 280 - x**6 / 15120, amplitude)
        result = wr @ (amplitude**2)
    elif typ == 3:
        result = wr @ (_radial(x) ** 2)
    elif typ == 2:
        hh, wh = _gaussian(h, sh, order)
        xx, ww = quadrature(angles)
        orientation, ow = (xx + 1) / 2, ww / 2
        fm = np.sum(
            wr[:, None, None]
            * _radial(
                rr[:, None, None] * np.sqrt(1 - orientation**2)[None, :, None] * q[None, None, :]
            )
            ** 2,
            axis=0,
        )
        hm = np.sum(
            wh[:, None, None]
            * np.sinc(
                hh[:, None, None] * orientation[None, :, None] * q[None, None, :] / (2 * np.pi)
            )
            ** 2,
            axis=0,
        )
        result = np.sum(ow[:, None] * fm * hm, axis=0)
    else:
        raise ValueError("Active particle type must be 1,2 or3")
    lp = -np.pi * q * q * (distance * sd) ** 2
    phi = np.exp(lp)
    sf = -np.expm1(2 * lp) / np.maximum(
        np.expm1(lp) ** 2 + 4 * phi * np.sin(0.5 * q * distance) ** 2, 1e-30
    )
    # The analytic q->0 limit is 2*pi*sd^2, avoiding a 0/0 placeholder.
    sf = np.where(q == 0, 2 * np.pi * sd * sd, sf)
    return result * sf


def _batch_inputs(q, u, types):
    u, types = np.asarray(u, float), np.asarray(types, int)
    if u.shape[-1] != 32 or types.shape[-1] != 4 or u.shape[:-1] != types.shape[:-1]:
        raise ValueError("u[...,32] and types[...,4] require matching batch axes")
    if not np.isfinite(u).all() or not np.isin(types, [0, 1, 2, 3]).all():
        raise ValueError("Nonfinite u or invalid type ID")
    q = np.abs(np.asarray(q, float))
    if q.ndim == 1:
        q = np.broadcast_to(q, (*u.shape[:-1], len(q)))
    if q.shape[:-1] != u.shape[:-1] or not np.isfinite(q).all():
        raise ValueError("q must have matching batch axes or be one shared q vector")
    return q, u, types


def np_basis(q, u, types, *, order=None, angles=None):
    q, u, types = _batch_inputs(q, u, types)
    shape = q.shape
    qq, uu, tt = q.reshape(-1, shape[-1]), u.reshape(-1, 32), types.reshape(-1, 4)
    result = np.zeros((*qq.shape, 6), float)
    for b in range(len(uu)):
        p = decode(uu[b])
        for j, typ in enumerate(tt[b]):
            if typ:
                result[b, :, j] = np_particle(
                    qq[b],
                    typ,
                    *[p[k][j] for k in ("R", "sigma_R", "h", "sigma_h", "D", "sigma_D")],
                    order=order,
                    angles=angles,
                )
        result[b, :, 4] = 1.0
        result[b, :, 5] = 1 / (1 + (qq[b] / p["sigma_res"]) ** p["nu_res"])
    return result.reshape(*shape, 6)


def np_forward(q, u, types, *, order=None, angles=None):
    basis = np_basis(q, u, types, order=order, angles=angles)
    p = decode(u)
    coefficients = np.concatenate([p["A"], p["B"][..., None], p["C"][..., None]], axis=-1)
    return np.sum(basis * coefficients[..., None, :], axis=-1)


def tf_forward(q, u, types, *, order=None, angles=None):
    """Differentiable batched physics; RC work is restricted to RC rows per slot."""
    import tensorflow as tf

    order = DEFAULT_ORDER if order is None else int(order)
    angles = DEFAULT_ANGLES if angles is None else int(angles)
    u = tf.convert_to_tensor(u)
    if not u.dtype.is_floating:
        u = tf.cast(u, tf.float32)
    dtype = u.dtype
    q = tf.abs(tf.cast(q, dtype))
    types = tf.convert_to_tensor(types, dtype=tf.int32)
    z = tf.constant(LO, dtype) + u * tf.constant(HI - LO, dtype)
    slots = tf.reshape(z[:, :28], [-1, 4, 7])
    xn, wn = quadrature(order)
    xn, wn = tf.constant(xn, dtype), tf.constant(wn, dtype)
    xa, wa = quadrature(angles)
    orientation, ow = tf.constant((xa + 1) / 2, dtype), tf.constant(wa / 2, dtype)

    def gaussian(mu, width):
        sigma = mu * width
        lo, hi = tf.maximum(mu - 4 * sigma, 0.0), mu + 4 * sigma
        nodes = lo[:, None] + (xn[None, :] + 1) * (hi - lo)[:, None] / 2
        weights = wn[None, :] * tf.exp(-0.5 * ((nodes - mu[:, None]) / sigma[:, None]) ** 2)
        return nodes, weights / tf.reduce_sum(weights, axis=1, keepdims=True)

    def radial(x):
        safe = tf.where(abs(x) < 1e-4, tf.ones_like(x), x)
        return tf.where(
            abs(x) < 1e-4, 1 - x * x / 8 + x**4 / 192, 2 * tf.math.special.bessel_j1(safe) / safe
        )

    def particle_form(qv, sv, typ):
        r, sr, h, sh, eta, sd, _ = tf.unstack(sv, axis=-1)
        r, h = tf.exp(r), tf.exp(h)
        lower = tf.maximum(tf.constant(3.0, dtype), 2.002 * r)
        distance = lower + (80.0 - lower) * eta
        rr, wr = gaussian(r, sr)
        if typ == 1:
            x = rr[:, :, None] * qv[:, None, :]
            safe = tf.where(abs(x) < 0.1, tf.ones_like(x), x)
            amp = 3 * (tf.sin(safe) - safe * tf.cos(safe)) / safe**3
            amp = tf.where(abs(x) < 0.1, 1 - x * x / 10 + x**4 / 280 - x**6 / 15120, amp)
            form = tf.reduce_sum(wr[:, :, None] * amp**2, axis=1)
        elif typ == 3:
            x = rr[:, :, None] * qv[:, None, :]
            form = tf.reduce_sum(wr[:, :, None] * radial(x) ** 2, axis=1)
        else:
            hh, wh = gaussian(h, sh)
            x = (
                rr[:, :, None, None]
                * tf.sqrt(1 - orientation**2)[None, None, :, None]
                * qv[:, None, None, :]
            )
            fm = tf.reduce_sum(wr[:, :, None, None] * radial(x) ** 2, axis=1)
            x = hh[:, :, None, None] * orientation[None, None, :, None] * qv[:, None, None, :] / 2
            safe = tf.where(abs(x) < 1e-4, tf.ones_like(x), x)
            sinc = tf.where(abs(x) < 1e-4, 1 - x * x / 6 + x**4 / 120, tf.sin(safe) / safe)
            hm = tf.reduce_sum(wh[:, :, None, None] * sinc**2, axis=1)
            form = tf.reduce_sum(ow[None, :, None] * fm * hm, axis=1)
        lp = -tf.constant(np.pi, dtype) * qv * qv * (distance * sd)[:, None] ** 2
        phi = tf.exp(lp)
        sf = -tf.math.expm1(2 * lp) / tf.maximum(
            tf.math.expm1(lp) ** 2 + 4 * phi * tf.sin(0.5 * qv * distance[:, None]) ** 2,
            tf.constant(1e-30, dtype),
        )
        sf = tf.where(qv == 0, 2 * tf.constant(np.pi, dtype) * sd[:, None] ** 2, sf)
        return form * sf

    total = tf.zeros_like(q)
    for j in range(4):
        for typ in (1, 2, 3):
            indices = tf.where(types[:, j] == typ)[:, 0]

            def compute(indices=indices, typ=typ, j=j):
                qs = tf.gather(q, indices)
                ss = tf.gather(slots[:, j, :], indices)
                contribution = particle_form(qs, ss, typ) * tf.exp(ss[:, 6])[:, None]
                return tf.scatter_nd(
                    indices[:, None], contribution, tf.shape(q, out_type=indices.dtype)
                )

            total = total + tf.cond(tf.size(indices) > 0, compute, lambda: tf.zeros_like(q))
    b, c, rs, nu = tf.unstack(z[:, 28:], axis=-1)
    total = (
        total
        + tf.exp(b)[:, None]
        + tf.exp(c)[:, None] / (1 + (q / tf.exp(rs)[:, None]) ** nu[:, None])
    )
    return total
