"""Full-mixture CAL summaries; no true-branch conditioning of predictions.

Marginal PIT maps adjust interval probabilities, not the generative mixture.
Joint sets use branch plus active standardized-chart density. They are not
physical-coordinate HPD sets, and CAL reuse for early stopping prevents an
independent validation or finite-sample conformal-coverage claim.
"""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.special import ndtr, logsumexp, expit

LEVELS = np.array([.025, .16, .5, .84, .975])


def family_fold(index):
    """Keep the generator's Ag/Au family pair in the same CAL diagnostic fold."""
    return int(hashlib.sha256(('NR_CAL_pair_v1:' + str(int(index)//2)).encode()).hexdigest()[:16], 16) % 5


def extract_mixture(expert, encoded, batch=128):
    """All branches/components, no labels; exact active marginal covariance."""
    import torch
    if not np.all(expert.active == expert.active[0]):
        raise ValueError('Common active coordinates required; do not invent inactive labels')
    active = np.flatnonzero(expert.active[0]); parts = []
    with torch.no_grad():
        for start in range(0, len(encoded), batch):
            x = torch.as_tensor(encoded[start:start+batch], device=expert.device)
            emb = expert.net.embedding(dict(vector=x)); pc, _, _ = expert.net.discrete(emb)
            weights, means, chols = [], [], []
            for b in range(expert.net.C):
                labels = torch.full((len(x),), b, device=expert.device, dtype=torch.long)
                logits, mu, factor = expert.net.conditional(emb, labels, torch.zeros_like(labels))
                fac = factor.double()[:, :, active, :]
                chol = torch.linalg.cholesky(fac @ fac.transpose(-1, -2))
                weights.append((pc[:, b, None].double()+torch.log_softmax(logits.double(), -1)).cpu().numpy())
                means.append(mu.double()[:, :, active].cpu().numpy()); chols.append(chol.cpu().numpy())
            parts.append(dict(logw=np.stack(weights, 1), mu=np.stack(means, 1), chol=np.stack(chols, 1)))
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError('Nonfinite original mixture; no CAL family may be dropped')
    if not np.allclose(np.exp(out['logw']).sum((1, 2)), 1., atol=1e-6):
        raise ValueError('Full branch-component probability does not sum to one')
    # Normalize floating-point roundoff only; every component is retained.
    out['logw'] -= logsumexp(out['logw'], axis=(1, 2))[:, None, None]
    return out, active


def marginal_cdf(mix, value):
    value = np.asarray(value, float)
    sd = np.sqrt(np.sum(mix['chol']**2, axis=-1))
    return np.sum(np.exp(mix['logw'])[..., None] * ndtr((value[:, None, None, :]-mix['mu'])/sd), axis=(1, 2))


def marginal_quantile(mix, probability):
    n, _, _, dim = mix['mu'].shape
    p = np.broadcast_to(np.asarray(probability, float), (n, dim))
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Quantile probabilities must lie in [0,1]')
    sd = np.sqrt(np.sum(mix['chol']**2, axis=-1))
    lo = np.min(mix['mu']-40*sd, axis=(1, 2)); hi = np.max(mix['mu']+40*sd, axis=(1, 2))
    for _ in range(68):
        mid = (lo+hi)/2; below = marginal_cdf(mix, mid) < p
        lo = np.where(below, mid, lo); hi = np.where(below, hi, mid)
    q = (lo+hi)/2
    # A numerically saturated PIT tail stays explicitly unbounded, not clipped.
    return np.where(p == 0, -np.inf, np.where(p == 1, np.inf, q))


def joint_nll(mix, z, branch):
    """Membership score only. Includes branch mass; no truth-selected output."""
    z = np.asarray(z, float); b = np.asarray(branch, int); i = np.arange(len(z))
    delta = z[:, None, :] - mix['mu'][i, b]
    white = np.linalg.solve(mix['chol'][i, b], delta[..., None])[..., 0]
    logn = -.5*np.sum(white**2, -1)-np.log(np.diagonal(mix['chol'][i, b], axis1=-2, axis2=-1)).sum(-1)
    logn -= .5*z.shape[1]*np.log(2*np.pi)
    return -logsumexp(mix['logw'][i, b]+logn, axis=-1)


def fit_mapping(pit, scores):
    pit = np.asarray(pit, float); scores = np.asarray(scores, float)
    if len(pit) < 100 or not np.isfinite(pit).all() or not np.isfinite(scores).all():
        raise ValueError('At least100 finite CAL families per metal required')
    return dict(n=len(pit), marginal_probabilities=np.quantile(pit, LEVELS, axis=0).tolist(),
                joint_nll_thresholds={str(level): float(np.sort(scores)[min(int(np.ceil((len(scores)+1)*level))-1, len(scores)-1)])
                                      for level in (.68, .95)})


def physical(z, names, mean, scale):
    """Undo fixed TRAIN standardization and declared monotonic chart map."""
    chart = np.asarray(z)*np.asarray(scale)+np.asarray(mean)
    out = np.empty_like(chart)
    with np.errstate(over='ignore', invalid='ignore'):
        for j, name in enumerate(names):
            out[..., j] = np.expm1(chart[..., j]) if name == 'burial_base_depth_nm' else expit(chart[..., j]) if name == 'metal_fraction' else np.exp(chart[..., j])
    return out


def safe_json(value):
    """Finite numerics or explicit reason; never an empty fabricated value."""
    if isinstance(value, dict): return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [safe_json(v) for v in value]
    if isinstance(value, (bool, np.bool_)): return bool(value)
    if isinstance(value, (int, np.integer)): return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else 'positive_unbounded' if value > 0 else 'negative_unbounded' if value < 0 else 'numerically_undefined'
    return value


def calibrated_summary(expert, encoded, metal, calibration):
    """Prediction-only entry; accepts no truth labels or true branch.

    All component means/masses are returned for downstream candidate handling.
    Marginal intervals can span modes; joint regions are defined by the saved
    joint_nll thresholds and must be tested without conditioning on true branch.
    """
    from .neural_refine_chart import NAMES
    from .neural_refine_training import sha
    if calibration['checkpoint_sha256'] != expert.checkpoint_sha256 or calibration['policy_sha256'] != expert.declaration['bank']['input_policy_sha256']:
        raise ValueError('Calibration belongs to a different selected checkpoint/policy')
    if calibration['arithmetic_sha256'] != sha(__file__):
        raise ValueError('Calibrated mixture/interval arithmetic changed')
    mix, active = extract_mixture(expert, encoded)
    names = [NAMES[j] for j in active]; mapping = calibration['by_metal'][metal]
    if names != calibration['parameters']: raise ValueError('Calibrated parameter chart differs')
    qs = np.stack([marginal_quantile(mix, p) for p in mapping['marginal_probabilities']], 1)
    mean = expert.net.chart_mean.detach().cpu().numpy()[active]; scale = expert.net.chart_scale.detach().cpu().numpy()[active]
    return safe_json(dict(parameters=names, quantile_levels=LEVELS.tolist(), marginal_values=physical(qs, names, mean, scale),
                         all_component_means=physical(mix['mu'], names, mean, scale), all_component_masses=np.exp(mix['logw']),
                         joint_nll_thresholds=mapping['joint_nll_thresholds'], branch_probabilities_calibrated=False,
                         scope='CAL-adjusted full mixture; marginal intervals are not disjoint mode intervals or post-refinement uncertainty',
                         independent_validation=False, Dcorr_truncation=False))
