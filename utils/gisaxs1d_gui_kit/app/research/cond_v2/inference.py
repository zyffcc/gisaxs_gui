"""cond_v2 fast fitting: network proposals + a few LM steps in physics v2 (table engine).

    fit = FastFitterV2(model_dir)
    result = fit(q, I, sigma, composition='vc,vc', qz=0.5, gisaxs=True, steps=8)

Proposals: the 8 network hypotheses (numpy inference, scale-free inputs) are NNLS-calibrated
and ranked by chi2; `steps` Levenberg-Marquardt iterations (TabFit, number-weighted sizes,
Hosemann paracrystal, per-curve codec) refine the best `refine_heads` of them; the lowest
chi2 wins. steps=0 is the pure network. Batch use: FastFitterV2.batch(curves) with a pool.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from research.multishape_v1 import physics
from research.conditional_fit.pipeline import Candidate, network_view, _clean_inputs
from research.cond_v2.codec import CurveCodec
from research.cond_v2.model import cond_vector, features
from research.cond_v2.numpy_net import NumpyCondV2
from research.conditional_fit.fastfit import powerlaw_u

COMBOS = physics.COMBOS
FREE_NU = (1.5, 4.0)    # range of the Lorentzian low-q exponent when it is fitted (lowq_nu=None)


def parse(composition):
    from research.conditional_fit.tool import parse_composition
    return parse_composition(composition) if isinstance(composition, str) else list(composition)


def load_network(model_dir, backend=None):
    """The proposal network: 'numpy' (default, no extra dependency) or 'torch' (research.cond_v2_torch, about
    2-4x faster; needs PyTorch). backend None reads the environment variable GISAXS1D_NET."""
    backend = (backend or os.environ.get('GISAXS1D_NET') or 'numpy').lower()
    if backend == 'torch':
        from research.cond_v2_torch.model import TorchCondV2
        return TorchCondV2(model_dir), 'torch'
    if backend != 'numpy':
        raise ValueError(f"network backend must be 'numpy' or 'torch', not {backend!r}")
    return NumpyCondV2(model_dir), 'numpy'


class FastFitterV2:
    def __init__(self, model_dir, backend=None):
        self.net, self.backend = load_network(model_dir, backend)
        self.model_dir = str(model_dir)

    def propose(self, q, y, s, combos, qz=0.0, gisaxs=True, lowq='lorentz', dwba=None):
        q, y, s = _clean_inputs(q, y, s)
        codec = CurveCodec(float(q.min()), float(q.max()))
        vq, vy, vs = network_view(q, y, s)
        pts, m = features(vq[None], vy[None], vs[None], np.ones((1, len(vq)), bool), q_max=[q.max()])
        cond = cond_vector([qz], [gisaxs], [q.min()], [q.max()])
        combos = np.asarray(combos, int)
        u = self.net(np.repeat(pts, len(combos), 0), np.repeat(m, len(combos), 0), combos, np.repeat(cond, len(combos), 0))
        out = {}
        for i, combo in enumerate(combos):
            cands = []
            for h, uh in enumerate(u[i]):
                fit = self._fit(q, y, s, combo, codec, qz, gisaxs, lowq, dwba)
                if lowq == 'powerlaw':
                    uh = uh.copy()
                    uh[31] = powerlaw_u(3.0)
                coef, pred, res = fit.solve(fit.matrix_from(*fit.full(uh)))
                cands.append(Candidate(int(combo), uh, coef, pred, float(res@res), len(y), f'v2_head{h}', codec=codec,
                                       info=dict(engine='v2', lowq=lowq)))
            out[int(combo)] = sorted(cands, key=lambda c: c.chi2)
        return q, y, s, codec, out

    @staticmethod
    def _fit(q, y, s, combo, codec, qz, gisaxs, lowq='lorentz', dwba=None):
        from research.fast_inference.tabfit import TabFit
        return TabFit(q, y, s, COMBOS[combo], qz=qz if gisaxs else 0.0, sf_on=gisaxs, res_on=gisaxs, codec=codec,
                      weighting='number', sf='hosemann', lowq=lowq, dwba=dwba if gisaxs else None)

    def _nu_boxed(self, cand, box, q, y, s, codec, qz, gisaxs, lowq, dwba):
        """The candidate with its low-q exponent moved into box = (u_lo, u_hi) of coordinate 31 (amplitudes
        re-solved); unchanged when it is already inside."""
        if box[0] <= cand.u[31] <= box[1]:
            return cand
        u = cand.u.copy()
        u[31] = np.clip(u[31], *box)
        fit = self._fit(q, y, s, cand.combo, codec, qz, gisaxs, lowq, dwba)
        coef, pred, res = fit.solve(fit.matrix_from(*fit.full(u)))
        return Candidate(cand.combo, u, coef, pred, float(res@res), cand.n_points, cand.source, codec=codec,
                         info=dict(cand.info))

    def __call__(self, q, y, s, composition, qz=0.0, gisaxs=True, steps=8, refine_heads=1, lowq_nu=2.0, lowq='lorentz',
                 dwba=None):
        """lowq='lorentz': low-q "resolution" term with nu fixed at lowq_nu; lowq_nu=None fits nu inside
        FREE_NU = [1.5, 4] (network heads outside it are moved to the nearer edge, so every result is inside);
        lowq='powerlaw': C (q/q_min)^-n with n in [1, 4] free (GISAXS mode only; SAXS has no low-q term).
        dwba: research.fast_inference.tabfit.YonedaDWBA -> spheres use the four-term DWBA amplitude (GISAXS only)."""
        t0 = time.perf_counter()
        combos = parse(composition)
        q, y, s, codec, proposals = self.propose(q, y, s, combos, qz, gisaxs, lowq, dwba)
        t_net = time.perf_counter()-t0
        results = []
        nu_box = None
        if gisaxs and lowq == 'lorentz' and lowq_nu is None:
            lo, hi = codec.lo_hi
            nu_box = tuple(float(np.clip((v-lo[31])/(hi[31]-lo[31]), 0, 1)) for v in FREE_NU)
        for combo, cands in proposals.items():
            if nu_box is not None:
                cands = sorted((self._nu_boxed(c, nu_box, q, y, s, codec, qz, gisaxs, lowq, dwba) for c in cands),
                               key=lambda c: c.chi2)
            best = cands[0]
            if steps:
                for c in cands[:refine_heads]:
                    fit = self._fit(q, y, s, combo, codec, qz, gisaxs, lowq, dwba)
                    u0 = c.u.copy()
                    fixed, bounds = [], {}
                    if gisaxs and lowq == 'lorentz' and lowq_nu is not None:
                        lo, hi = codec.lo_hi
                        u0[31] = np.clip((lowq_nu-lo[31])/(hi[31]-lo[31]), 0, 1)
                        fixed = [31]
                    elif nu_box is not None:
                        bounds = dict(lower={31: nu_box[0]}, upper={31: nu_box[1]})
                    r = fit.run(u0, fixed=fixed, max_nfev=steps, **bounds)
                    cand = Candidate(combo, r['u'], r['coef'], r['prediction'], r['chi2'], len(y), f'v2_lm{steps}<{c.source}',
                                     codec=codec, info=dict(engine='v2', coords=r['coords'], cov_u=r['cov_u'],
                                                            at_bound=r['at_bound'], dof=r['dof'], lowq=lowq))
                    if cand.chi2 < best.chi2:
                        best = cand
            results.append(best)
        results.sort(key=lambda c: c.bic)
        return dict(results=results, network_seconds=t_net, total_seconds=time.perf_counter()-t0, q=q, y=y, sigma=s)
