"""Observable fit diagnostics; no clean signal or ground-truth labels at inference.

Sigma must be an absolute pointwise standard deviation. Bin diagnostics assume
independent errors; their thresholds require validation for each noise process.
"""
import numpy as np


def quality_metrics(predictions, observed, sigma, mask):
    p = np.atleast_2d(np.asarray(predictions, dtype=np.float64))[:, np.asarray(mask, bool)]
    y, s = [np.asarray(v, dtype=np.float64)[np.asarray(mask, bool)] for v in (observed, sigma)]
    if len(y) < 8 or not all(np.isfinite(v).all() and (v > 0).all() for v in (p, y, s)):
        raise ValueError('At least eight finite positive valid intensities and sigmas required')
    r = np.log(p)-np.log(y)
    out = {'observed_error': np.sqrt(np.mean(r*r, axis=1)),
           'standardized_rms': np.sqrt(np.mean(((p-y)/s)**2, axis=1)),
           'noise_relative_rms': np.full(len(p), np.sqrt(np.mean((s/y)**2)))}
    for count in (8, 32, 128):
        # Common deterministic weights within each bin; both curves use them.
        # Normalize before squaring to avoid underflow at very small intensities.
        biases = []; uncertainties = []
        for ids in np.array_split(np.arange(len(y)), min(count, max(2, len(y)//4))):
            w = (s[ids].min()/s[ids])**2; w /= w.sum()
            yy = np.sum(y[ids]*w); pp = np.sum(p[:, ids]*w, axis=1)
            se = np.sqrt(np.sum((s[ids]*w)**2))/yy
            biases.append(np.log(pp/yy)); uncertainties.append(se)
        b = np.stack(biases, axis=1); u = np.asarray(uncertainties)
        out[f'band{count}_rms'] = np.sqrt(np.mean(b*b, axis=1))
        out[f'band{count}_excess'] = np.sqrt(np.maximum(np.mean(b*b-u*u, axis=1), 0.))
        out[f'band{count}_max_excess'] = np.max(np.maximum(np.abs(b)-2.58*u, 0.), axis=1)
        out[f'band{count}_noise'] = np.full(len(p), np.sqrt(np.mean(u*u)))
    return out


def combo_order(combos, observed_error, classifier_nlp, beta=.01):
    """One observed-best parameter representative per original shape multiset."""
    c = np.asarray(combos); e = np.asarray(observed_error)
    reps = np.array([ix[np.argmin(e[ix])] for k in np.unique(c) for ix in [np.flatnonzero(c == k)]])
    score = e[reps]**2 + beta*np.asarray(classifier_nlp)[reps]
    return reps[np.argsort(score, kind='stable')]
