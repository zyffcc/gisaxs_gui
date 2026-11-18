"""The posterior of a cut from the proposal network: likelihood, importance weights, modes, refinement (numpy only,
with the stored random network of tests/data/ml1d: it checks the computation, not the network's quality)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.fitting.domain.cut_model import model as cut
from src.gimap.features.fitting.domain.posterior import encoding as E
from src.gimap.features.fitting.domain.posterior import infer, refine
from src.gimap.features.fitting.domain.posterior.flow import Network

DATA = Path(__file__).resolve().parent / "data" / "ml1d"
SPHERE = E.composition_index([("sphere", True)])


def _curve(seed=0, radius=4.0, distance=14.0):
    q = np.linspace(0.02, 2.0, 300)
    population = cut.Population("sphere", radius, 0.15, distance=distance, distance_spread=0.2)
    truth = 1e4 * cut.unit_terms([population], cut.Settings(), q)[0] + 2.0
    counts = np.random.default_rng(seed).poisson(truth).astype(float)
    return q, counts, np.sqrt(np.maximum(counts, 1.0))


def _values(radius=4.0, distance=14.0, tau=0.01, sigma_q=1e-4):
    v = np.full(E.DIMS, np.nan)
    v[0:2] = radius, 0.15
    v[4:6] = distance, 0.2
    v[E.GLOBAL] = 0.01, 3.0, sigma_q, tau
    return v


def test_cut_window_and_noise():
    q, counts, sigma = _curve()
    data = infer.Cut.prepare(q, counts, sigma)
    assert data.q_max == pytest.approx(2.0) and data.ratio == pytest.approx(100.0) and data.dropped == 0
    without = infer.Cut.prepare(q, counts)
    assert np.all(without.sigma > 0)
    wide = infer.Cut.prepare(np.linspace(0.001, 2.0, 2000), np.ones(2000), np.ones(2000))
    assert wide.ratio == E.RATIO_MAX and wide.dropped > 0 and wide.q.min() >= 2.0 / E.RATIO_MAX * (1 - 1e-9)


def test_likelihood_equals_the_dense_gaussian():
    """The model error integrated out by Woodbury equals the Gaussian with the full covariance."""
    q, counts, sigma = _curve()
    data = infer.Cut.prepare(q, counts, sigma)
    like = infer.Likelihood(data, SPHERE)
    values = np.stack([_values(tau=0.01), _values(radius=4.4, tau=0.05)])
    log_marginal, amplitudes, chi2 = like(values)
    x = like.columns(values)
    for b in range(2):
        model = amplitudes[b] @ x[b]
        tau = values[b, E.GLOBAL.start + 3]
        d = like.basis * model[None, :]
        cov = np.diag(sigma**2) + tau**2 * d.T @ d
        r = counts - model
        _s, log_det = np.linalg.slogdet(cov)
        dense = -0.5 * (r @ np.linalg.solve(cov, r) + log_det + q.size * math.log(2 * math.pi))
        positive = amplitudes[b] > 0
        scaled = x[b][positive] / sigma[None, :]
        _s, log_det_h = np.linalg.slogdet(scaled @ scaled.T)
        laplace = (np.sum(-np.log(amplitudes[b][positive] * infer.AMPLITUDE_DECADES * math.log(10)))
                   + 0.5 * positive.sum() * math.log(2 * math.pi) - 0.5 * log_det_h)
        assert log_marginal[b] == pytest.approx(dense + laplace, rel=1e-9)
        assert chi2[b] == pytest.approx(np.sum((r / sigma) ** 2), rel=1e-9)


def test_amplitudes_reproduce_a_noise_free_curve():
    q = np.linspace(0.02, 2.0, 300)
    data = infer.Cut.prepare(q, np.ones_like(q), np.ones_like(q))
    like = infer.Likelihood(data, SPHERE)
    x = like.columns(_values()[None])[0]
    target = 3.0 * x[0] + 0.5 * x[1] + 0.2
    data = infer.Cut.prepare(q, target, 0.01 * target)
    _log, amplitudes, chi2 = infer.Likelihood(data, SPHERE)(_values()[None])
    assert np.allclose(amplitudes[0], [3.0, 0.5, 0.2], rtol=1e-6) and chi2[0] < 1e-8


def test_the_right_values_are_more_likely():
    q, counts, sigma = _curve()
    like = infer.Likelihood(infer.Cut.prepare(q, counts, sigma), SPHERE)
    log_marginal, _a, chi2 = like(np.stack([_values(), _values(radius=5.0), _values(distance=20.0)]))
    assert log_marginal[0] > log_marginal[1] and log_marginal[0] > log_marginal[2]
    assert chi2[0] / q.size < 1.5


def test_weights_evidence_and_modes():
    network = Network.load(DATA / "flow_fixture.npz")
    q, counts, sigma = _curve()
    data = infer.Cut.prepare(q, counts, sigma)
    results = infer.infer(network, data, [SPHERE, E.composition_index([("vertical_cylinder", True)])], samples=200)
    assert sum(r.probability for r in results) == pytest.approx(1.0)
    for r in results:
        assert np.isfinite(r.log_evidence) and 1.0 <= r.ess <= 200
        assert r.weights.sum() == pytest.approx(1.0) and r.modes
        assert sum(m.weight for m in r.modes) == pytest.approx(1.0)
        active = E.active(r.index)[0]
        for m in r.modes:
            assert np.all(m.low95[active] <= m.low68[active] + 1e-12) and np.all(m.high68[active] <= m.high95[active] + 1e-12)
            assert np.all(np.isnan(m.median[~active]))


def test_predict_refines_to_the_truth():
    """Least squares from the heaviest samples finds the curve's values, whatever the (random) network proposed."""
    network = Network.load(DATA / "flow_fixture.npz")
    q, counts, sigma = _curve(seed=3)
    solutions = refine.predict(network, q, counts, sigma, compositions=[SPHERE], samples=300, seed=1)
    best = solutions[0]
    assert best.composition == SPHERE and best.method == "lm"
    assert best.chi2_reduced < 1.5
    assert best.values["R_1"].best == pytest.approx(4.0, rel=0.03)
    assert best.values["D_1"].best == pytest.approx(14.0, rel=0.03)
    assert set(best.values) == {"R_1", "sigma_R_1", "D_1", "sigma_nn_1", "tail_width", "tail_exponent", "q_resolution"}


@pytest.mark.parametrize("method", ["global", "posterior"])
def test_the_other_methods(method):
    network = Network.load(DATA / "flow_fixture.npz")
    q, counts, sigma = _curve(seed=4)
    solutions = refine.predict(network, q, counts, sigma, compositions=[SPHERE], samples=200, method=method, seed=2)
    assert solutions and all(s.method == method for s in solutions)
    if method == "global":
        assert len(solutions) == 1 and solutions[0].values["R_1"].best == pytest.approx(4.0, rel=0.03)
    with pytest.raises(ValueError):
        refine.predict(network, q, counts, sigma, compositions=[SPHERE], method="mcmc")
