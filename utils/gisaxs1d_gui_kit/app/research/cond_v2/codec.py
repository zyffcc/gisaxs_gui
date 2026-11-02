"""Per-curve parameter codec for cond_v2 (scale-free across data).

All size ranges are tied to the curve's own q window [q_min, q_max]:
  R, h in [0.5/q_max, 5/q_min] (log)      -> any structure whose features touch the window,
                                           including large ones that only show a Guinier tail
  D = 2.002 R * (D_MAX_RATIO)^u (log)     -> nearest-neighbour distance 2R ... 40R
  w_rel = w/D in [0.02, 0.5]              -> Hosemann spread (physics sf='hosemann')
  sigma_R, sigma_h in [0.03, 0.5]         -> relative widths (number-weighted, physics weighting='number')
  low-q Lorentzian width in [q_min/10, q_max/3] (log), exponent nu in [1, 6] (default fixed 2)
Slot layout is the 32-vector used everywhere else: [logR, sR, logh, sh, D, w_rel, logA] x 4 +
[logB, logC, logWidth, nu]; the linear entries (A, B, C) are placeholders solved by NNLS.
The object exposes decode/encode/lo_hi like conditional_fit.codec.Codec, so FastFit/TabFit accept it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

D_MAX_RATIO = 20.0
WIDTH = (0.03, 0.5)
W_REL = (0.02, 0.5)
NU = (1.0, 6.0)


@dataclass(frozen=True)
class CurveCodec:
    q_min: float
    q_max: float
    name: str = 'curve_v2'

    @property
    def size_range(self):
        return 0.5/self.q_max, 5.0/self.q_min

    @property
    def lo_hi(self):
        r_lo, r_hi = np.log(self.size_range[0]), np.log(self.size_range[1])
        slot_lo = [r_lo, WIDTH[0], r_lo, WIDTH[0], 0.0, W_REL[0], 0.0]
        slot_hi = [r_hi, WIDTH[1], r_hi, WIDTH[1], 1.0, W_REL[1], 1.0]
        glob_lo = [0.0, 0.0, np.log(self.q_min/10), NU[0]]
        glob_hi = [1.0, 1.0, np.log(self.q_max/3), NU[1]]
        return np.r_[np.tile(slot_lo, 4), glob_lo], np.r_[np.tile(slot_hi, 4), glob_hi]

    def decode(self, u):
        u = np.asarray(u, float)
        lo, hi = self.lo_hi
        z = lo+u*(hi-lo)
        slot = z[..., :28].reshape(*z.shape[:-1], 4, 7)
        r = np.exp(slot[..., 0])
        distance = 2.002*r*D_MAX_RATIO**slot[..., 4]
        ones = np.ones(z.shape[:-1])
        return dict(R=r, sigma_R=slot[..., 1], h=np.exp(slot[..., 2]), sigma_h=slot[..., 3], D=distance,
                    sigma_D=slot[..., 5], A=np.ones_like(r), B=ones, C=ones,
                    sigma_res=np.exp(z[..., 30]), nu_res=z[..., 31])

    def encode(self, p):
        p = {k: np.asarray(v, float) for k, v in p.items()}
        r = p['R']
        frac = np.log(np.maximum(p['D'], 2.002*r)/(2.002*r))/np.log(D_MAX_RATIO)
        shape = r.shape[:-1]
        slot = np.stack([np.log(r), p['sigma_R'], np.log(p['h']), p['sigma_h'], frac, p['sigma_D'],
                         np.zeros_like(r)], -1)
        glob = np.stack([np.zeros(shape), np.zeros(shape), np.log(p['sigma_res']), p['nu_res']], -1)
        z = np.concatenate([slot.reshape(*shape, 28), glob], -1)
        lo, hi = self.lo_hi
        return np.clip((z-lo)/np.where(hi > lo, hi-lo, 1.0), 0.0, 1.0)
