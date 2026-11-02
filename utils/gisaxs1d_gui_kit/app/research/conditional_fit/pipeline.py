"""Composition-conditional fast fitting: network proposal -> NNLS -> optional refinement.

The composition is supplied by the user (or enumerated in auto mode); it is not
identified from the curve, because the identifiability study showed that a
single 1D cut does not determine multi-component compositions.

Stages
  propose(): r3 conditional heads give H parameter sets for each requested
             composition; amplitudes/B/C by nonnegative least squares.
  refine():  bounded variable-projection least squares from chosen proposals,
             with optional fixed coordinates and narrowed ranges.
  rank():    BIC across compositions; members within delta chi2 of the best
             are flagged as indistinguishable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from research.multishape_v1 import physics
from research.identifiability.run import COMBOS, coordinates
from research.conditional_fit.fastfit import FastFit
from research.conditional_fit.codec import LEGACY, convert

SLOT_NAMES = physics.SLOT_NAMES
GLOBAL_NAMES = physics.GLOBAL_NAMES


@dataclass
class Candidate:
    combo: int
    u: np.ndarray
    coef: np.ndarray
    prediction: np.ndarray
    chi2: float
    n_points: int
    source: str
    seconds: float = 0.0
    info: dict = field(default_factory=dict)
    codec: object = LEGACY

    @property
    def n_params(self):
        types = COMBOS[self.combo]
        return len(coordinates(types))+int(np.sum(types > 0))+2

    @property
    def bic(self):
        return self.chi2+self.n_params*np.log(self.n_points)

    def _uncertainties(self, j, typ, engine='legacy'):
        """1-sigma errors of R, sigma_R, D, sigma_D (and h, sigma_h) from the refinement covariance."""
        cov, coords = self.info.get('cov_u'), self.info.get('coords')
        if cov is None or not len(coords):
            return {}
        label = 'number' if engine == 'v2' else 'intensity'
        names = [('R_nm', 'R'), (f'sigma_R_rel_{label}_weighted', 'sigma_R'), ('D_nm', 'D'),
                 ('w_rel_hosemann' if engine == 'v2' else 'sigma_D_model', 'sigma_D')]
        if typ == 2:
            names += [('h_nm', 'h'), (f'sigma_h_rel_{label}_weighted', 'sigma_h')]
        base = self.codec.decode(self.u)
        grads = {key: np.zeros(len(coords)) for key, _ in names}
        for i, c in enumerate(coords):
            u2 = self.u.copy()
            step = 1e-6 if u2[c] < .5 else -1e-6
            u2[c] += step
            p2 = self.codec.decode(u2)
            for key, field_ in names:
                grads[key][i] = (p2[field_][j]-base[field_][j])/step
        out = {f'{key}_err': float(np.sqrt(max(g@cov@g, 0.0))) for key, g in grads.items()}
        slot_coords = {7*j+k for k in range(6)}
        out['at_bound'] = [coordinate_name(c) for c in self.info.get('at_bound', []) if c in slot_coords]
        return out

    def physical(self):
        """Physical parameters with explicit conventions (depend on the physics engine).

        legacy (multishape_v1): intensity-weighted Gaussian sizes; sigma_D_model with damping
        exp(-pi q^2 (sigma_D D)^2), Hosemann w = sqrt(2 pi) sigma_D D.
        v2: number-weighted Gaussian sizes (R_nm is the number mean); Hosemann paracrystal with
        w = w_rel * D (damping exp(-q^2 w^2 / 2)).
        """
        p = self.codec.decode(self.u)
        engine = self.info.get('engine', 'legacy')
        types = COMBOS[self.combo]
        particles = []
        for j in range(4):
            if not types[j]:
                continue
            typ = int(types[j])
            item = dict(type={1: 'sphere', 2: 'random_cylinder', 3: 'vertical_cylinder'}[typ], R_nm=p['R'][j], D_nm=p['D'][j],
                        amplitude=self.coef[j])
            if engine == 'v2':
                item.update(sigma_R_rel_number_weighted=p['sigma_R'][j], w_rel_hosemann=p['sigma_D'][j],
                            w_hosemann_nm=float(p['sigma_D'][j]*p['D'][j]))
                if typ == 2:
                    item.update(h_nm=p['h'][j], sigma_h_rel_number_weighted=p['sigma_h'][j])
            else:
                nodes, weights = physics._gaussian(p['R'][j], p['sigma_R'][j], physics.DEFAULT_ORDER)
                power = 6 if typ == 1 else 4
                number = weights/nodes**power
                # The implied number distribution weights nodes by 1/V^2; above sigma_R ~ 0.15 the 4-sigma
                # truncation dominates it (and it diverges at R->0 for sigma_R >= 0.25): not reported.
                defined = p['sigma_R'][j] <= 0.15
                item.update(sigma_R_rel_intensity_weighted=p['sigma_R'][j],
                            R_number_mean_nm=float(np.sum(number*nodes)/np.sum(number)) if defined else None,
                            sigma_D_model=p['sigma_D'][j], w_hosemann_nm=float(np.sqrt(2*np.pi)*p['sigma_D'][j]*p['D'][j]))
                if typ == 2:
                    item.update(h_nm=p['h'][j], sigma_h_rel_intensity_weighted=p['sigma_h'][j])
            item.update(self._uncertainties(j, typ, engine))
            particles.append({k: v if isinstance(v, (str, list)) or v is None else float(v) for k, v in item.items()})
        if self.info.get('lowq') == 'powerlaw':
            from research.conditional_fit.fastfit import powerlaw_exponent
            low_q = dict(low_q_form='C*(q/q_min)^-n', low_q_powerlaw_n=powerlaw_exponent(self.u))
        else:
            low_q = dict(low_q_form='C/(1+(q/sigma_res)^nu)', low_q_sigma_res_nm_inv=float(p['sigma_res']), low_q_nu=float(p['nu_res']))
        return dict(particles=particles, physics=engine, background_B=float(self.coef[4]), low_q_term_C=float(self.coef[5]),
                    **low_q,
                    chi2_per_point=self.chi2/max(self.n_points, 1))


def _clean_inputs(q, y, sigma):
    q, y, sigma = (np.asarray(v, float) for v in (q, y, sigma))
    keep = np.isfinite(q) & np.isfinite(y) & np.isfinite(sigma) & (q > 0) & (sigma > 0)
    order = np.argsort(q[keep])
    return q[keep][order], y[keep][order], sigma[keep][order]


def make_fit(q, y, sigma, combo, codec=LEGACY, engine='legacy', **mode):
    """FastFit (multishape_v1 conventions) or the table engine with physics v2 conventions
    (number-weighted Gaussian sizes, standard Hosemann paracrystal w = w_rel * D)."""
    if engine == 'v2':
        from research.fast_inference.tabfit import TabFit
        return TabFit(q, y, sigma, COMBOS[combo], codec=codec, weighting='number', sf='hosemann', **mode)
    return FastFit(q, y, sigma, COMBOS[combo], codec=codec, **mode)


def nnls_candidate(q, y, sigma, combo, u, source, codec=LEGACY, **mode):
    """NNLS-calibrated curve of one parameter vector (tabulated full-quadrature physics).

    mode: qz (Yoneda-cut out-of-plane momentum), sf_on, res_on; see FastFit."""
    fit = FastFit(q, y, sigma, COMBOS[combo], codec=codec, **mode)
    coef, prediction, residual = fit.solve(fit.matrix_from(*fit.full(u)))
    return Candidate(combo, np.asarray(u, float), coef, prediction, float(residual@residual), len(y), source,
                     codec=codec)


def network_view(q, y, sigma, max_points=192):
    """Rebin a long native curve onto a training-like grid for the network input only.

    Training grids have <=192 points: the first third geometric up to
    split=min(0.2, 0.2*q_max), the rest linear. Native points are averaged per
    bin (sigma combined as sqrt(sum sigma^2)/n); empty bins are dropped. The
    refinement always uses every native point.
    """
    if len(q) <= max_points:
        return q, y, sigma
    qmin, qmax = float(q.min()), float(q.max())
    split = min(0.2, 0.2*qmax)
    m = max_points//3
    edges = np.r_[np.geomspace(qmin, max(split, qmin*1.0001), m+1)[:-1], np.linspace(max(split, qmin), qmax, max_points-m+1)]
    edges[-1] = np.nextafter(qmax, np.inf)
    idx = np.clip(np.searchsorted(edges, q, side='right')-1, 0, len(edges)-2)
    count = np.bincount(idx, minlength=len(edges)-1)
    keep = count > 0
    mean = lambda v: (np.bincount(idx, weights=v, minlength=len(edges)-1)[keep]/count[keep])
    s = np.sqrt(np.bincount(idx, weights=sigma**2, minlength=len(edges)-1)[keep])/count[keep]
    return mean(q), mean(y), s


class Proposer:
    """Network proposals. cond_v1 runs in numpy when weights_numpy.npz exists (no TensorFlow);
    otherwise the Keras model (cond_v1 or multishape_v2, e.g. r3 point_stream) is used."""

    def __init__(self, model_dir, weights='best.weights.h5'):
        import json
        from pathlib import Path
        model_dir = Path(model_dir)
        config = json.loads((model_dir/'model_config.json').read_text())
        self.config = config
        self.backend = 'keras'
        if config.get('arch') == 'cond_v1' and (model_dir/'weights_numpy.npz').exists():
            from research.cond_v1.numpy_net import NumpyCondV1
            self.net = NumpyCondV1(model_dir)
            self.backend = 'numpy'
        elif config.get('arch') == 'cond_v1':
            from research.cond_v1 import model as cm
            self.net = cm.build_model(**{k: v for k, v in config.items() if k != 'arch'})
            self.net.load_weights(str(model_dir/weights))
        else:
            from research.multishape_v2 import new_model, weight_io
            self.net = new_model.build_model(**config)
            weight_io.load(self.net, model_dir/'curve_best.npz')

    def u(self, q, y, sigma, combos):
        q, y, sigma = network_view(q, y, sigma)
        combos = np.asarray(combos, 'int32')
        if self.config.get('arch') == 'cond_v1':
            from research.cond_v1.numpy_net import encode_points
            pts, m = encode_points(q[None], y[None], sigma[None], np.ones((1, len(q)), bool))
            if self.backend == 'numpy':
                return self.net(np.repeat(pts, len(combos), 0), np.repeat(m, len(combos), 0), combos)
            features = {'points': pts, 'point_mask': m}
        else:
            from research.multishape_v2.feature_controls import prepare_inputs
            features = prepare_inputs(q[None], y[None], sigma[None], None, None,
                                      encoder=self.config['encoder'], normalization=self.config['normalization'])
        batch = {k: np.repeat(v, len(combos), axis=0) for k, v in features.items()}
        out = self.net({**batch, 'combo': combos}, training=False)
        return out['u'].numpy().astype(float)  # [len(combos), heads, 32]


TRAIN_MEDIAN_INTENSITY = 659.0   # median over DEV curves of the per-curve median positive intensity
INPUT_SCALES = (10**-0.5, 1.0, 10**0.5)


def propose(proposer, q, y, sigma, combos, **mode):
    """All heads for each requested composition, NNLS-calibrated, best chi2 first per composition.

    The network is not intensity-scale invariant (its input is log1p(I)), so the
    curve is first rescaled to the training median intensity and the network is
    queried at INPUT_SCALES around it; amplitudes are re-solved at the original
    scale, so only the proposed nonlinear parameters depend on this choice.
    """
    q, y, sigma = _clean_inputs(q, y, sigma)
    positive = y[y > 0]
    base = TRAIN_MEDIAN_INTENSITY/np.median(positive) if len(positive) else 1.0
    heads = []
    for factor in INPUT_SCALES:
        heads.append((factor, proposer.u(q, y*base*factor, sigma*base*factor, combos)))
    result = {}
    for i, combo in enumerate(combos):
        cands = [nnls_candidate(q, y, sigma, int(combo), u, f'network_head{h}@x{factor:.2g}', **mode)
                 for factor, us in heads for h, u in enumerate(us[i])]
        result[int(combo)] = sorted(cands, key=lambda c: c.chi2)
    return result


def components(q, cand, **mode):
    """Contribution of every population, the constant background and the low-q term."""
    fit = make_fit(q, np.ones_like(q), np.ones_like(q), cand.combo, codec=cand.codec,
                   engine=cand.info.get('engine', 'legacy'), **mode)
    m = fit.matrix_from(*fit.full(cand.u))
    out = {}
    for j, typ in enumerate(COMBOS[cand.combo]):
        if typ:
            out[f'pop{j+1}_{ {1: "sphere", 2: "random_cylinder", 3: "vertical_cylinder"}[int(typ)] }'] = m[:, j]*cand.coef[j]
    out['background_B'] = m[:, 4]*cand.coef[4]
    out['low_q_term'] = m[:, 5]*cand.coef[5]
    return out


def coordinate_name(index):
    return GLOBAL_NAMES[index-28] if index >= 28 else f'pop{index//7+1}_{SLOT_NAMES[index % 7]}'


def boundary_coordinates(cand, mode=None, tol=1e-3):
    """Active nonlinear coordinates sitting at the edge of the model domain (u within tol of 0 or 1)."""
    mode = mode or {}
    fit = make_fit(np.ones(2), np.ones(2), np.ones(2), cand.combo, codec=cand.codec,
                   engine=cand.info.get('engine', 'legacy'), **mode)
    return [int(k) for k in fit.coords if cand.u[k] < tol or cand.u[k] > 1-tol]


def coordinate_index(slot, name):
    """u index for a slot parameter name (e.g. slot 0 'logR') or a global name."""
    if name in GLOBAL_NAMES:
        return 28+GLOBAL_NAMES.index(name)
    return 7*slot+SLOT_NAMES.index(name)


def refine(q, y, sigma, start: Candidate, *, fixed=(), lower=None, upper=None, max_nfev=60, codec=None,
           engine='legacy', **mode):
    """Refine the nonlinear coordinates of one candidate (FastFit variable projection).

    fixed: iterable of u indices held at start.u. lower/upper: optional dict
    {u index: bound in [0,1]} narrowing the search box (priors as ranges).
    """
    q, y, sigma = _clean_inputs(q, y, sigma)
    codec = codec or start.codec
    fit = make_fit(q, y, sigma, start.combo, codec=codec, engine=engine, **mode)
    r = fit.run(convert(start.u, start.codec, codec), fixed=fixed, lower=lower, upper=upper, max_nfev=max_nfev)
    return Candidate(start.combo, r['u'], r['coef'], r['prediction'], r['chi2'], len(y),
                     f'refined<{start.source}', seconds=r['seconds'], codec=codec,
                     info=dict(status=r['status'], coords=r['coords'], cov_u=r['cov_u'], at_bound=r['at_bound'], dof=r['dof'],
                               engine=engine))


def rank(candidates, delta=10.0):
    """Best candidate per composition ordered by BIC.

    A composition is flagged indistinguishable when its BIC is within `delta`
    of the best BIC (conventional 'very strong' evidence threshold). BIC is only
    a rough evidence proxy for mixture-type models; treat flags as a screen.
    """
    best = {}
    for c in candidates:
        if c.combo not in best or c.chi2 < best[c.combo].chi2:
            best[c.combo] = c
    ordered = sorted(best.values(), key=lambda c: c.bic)
    top = ordered[0].bic
    return [dict(candidate=c, bic=c.bic, delta_bic=c.bic-top, indistinguishable=bool(c.bic-top <= delta))
            for c in ordered]
