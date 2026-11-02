"""Parameter codecs mapping u in [0,1]^32 to physical parameters.

LEGACY is the frozen multishape_v1 codec (the network's training domain).
EXTENDED widens every nonlinear range for refinement, so a real curve is not
forced into the training box; network proposals are converted through
physical parameters (legacy.decode -> extended.encode). EXTENDED maps the
paracrystal distance logarithmically between max(D_floor, 2.002 R) and D_max,
which keeps relative resolution at small and large D alike.

Slot layout is unchanged: [logR, sigma_R, logh, sigma_h, D_fraction, sigma_D,
logA] x 4 then [logB, logC, logSigmaRes, nuRes]; relative widths stay relative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research.multishape_v1 import physics


@dataclass(frozen=True)
class Codec:
    name: str
    R: tuple
    sigma_R: tuple
    h: tuple
    sigma_h: tuple
    D_floor: float
    D_max: float
    sigma_D: tuple
    A: tuple
    B: tuple
    C: tuple
    sigma_res: tuple
    nu_res: tuple
    log_D: bool

    @property
    def lo_hi(self):
        slot_lo = [np.log(self.R[0]), self.sigma_R[0], np.log(self.h[0]), self.sigma_h[0], 0.0, self.sigma_D[0], np.log(self.A[0])]
        slot_hi = [np.log(self.R[1]), self.sigma_R[1], np.log(self.h[1]), self.sigma_h[1], 1.0, self.sigma_D[1], np.log(self.A[1])]
        lo = np.r_[np.tile(slot_lo, 4), np.log(self.B[0]), np.log(self.C[0]), np.log(self.sigma_res[0]), self.nu_res[0]]
        hi = np.r_[np.tile(slot_hi, 4), np.log(self.B[1]), np.log(self.C[1]), np.log(self.sigma_res[1]), self.nu_res[1]]
        return lo, hi

    def decode(self, u):
        if self.name == 'legacy':
            return physics.decode(u)
        u = np.asarray(u, float)
        lo, hi = self.lo_hi
        z = lo+u*(hi-lo)
        slot = z[..., :28].reshape(*z.shape[:-1], 4, 7)
        r = np.exp(slot[..., 0])
        lower = np.maximum(self.D_floor, 2.002*r)
        frac = slot[..., 4]
        distance = lower*(self.D_max/lower)**frac if self.log_D else lower+(self.D_max-lower)*frac
        return dict(R=r, sigma_R=slot[..., 1], h=np.exp(slot[..., 2]), sigma_h=slot[..., 3], D=distance,
                    sigma_D=slot[..., 5], A=np.exp(slot[..., 6]), B=np.exp(z[..., 28]), C=np.exp(z[..., 29]),
                    sigma_res=np.exp(z[..., 30]), nu_res=z[..., 31])

    def encode(self, p):
        if self.name == 'legacy':
            return physics.encode(p)
        p = {k: np.asarray(v, float) for k, v in p.items()}
        r = p['R']
        lower = np.maximum(self.D_floor, 2.002*r)
        if self.log_D:
            frac = np.log(np.maximum(p['D'], lower)/lower)/np.log(self.D_max/lower)
        else:
            frac = (p['D']-lower)/(self.D_max-lower)
        slot = np.stack([np.log(r), p['sigma_R'], np.log(p['h']), p['sigma_h'], frac, p['sigma_D'], np.log(p['A'])], -1)
        glob = np.stack([np.log(p['B']), np.log(p['C']), np.log(p['sigma_res']), p['nu_res']], -1)
        z = np.concatenate([slot.reshape(*r.shape[:-1], 28), glob], -1)
        lo, hi = self.lo_hi
        return np.clip((z-lo)/(hi-lo), 0.0, 1.0)


LEGACY = Codec('legacy', R=(1.0, 10.0), sigma_R=(0.08, 0.5), h=(3.0, 60.0), sigma_h=(0.08, 0.5), D_floor=3.0,
               D_max=80.0, sigma_D=(0.05, 0.4), A=(100.0, 1e4), B=(0.1, 10.0), C=(1e3, 1e6),
               sigma_res=(0.004, 0.05), nu_res=(1.5, 10.0), log_D=False)

EXTENDED = Codec('extended', R=(0.5, 200.0), sigma_R=(0.03, 0.5), h=(1.0, 500.0), sigma_h=(0.03, 0.5), D_floor=1.0,
                 D_max=2000.0, sigma_D=(0.02, 0.45), A=(1e-6, 1e12), B=(1e-6, 1e12), C=(1e-6, 1e12),
                 sigma_res=(1e-4, 0.2), nu_res=(1.0, 12.0), log_D=True)

CODECS = {'legacy': LEGACY, 'extended': EXTENDED}


def convert(u, source, target):
    """Map u between codecs through physical parameters (clipped to the target box)."""
    if source is target:
        return np.asarray(u, float).copy()
    return target.encode(source.decode(u))
