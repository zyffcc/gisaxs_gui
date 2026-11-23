"""Composition integrals and shares for a normalized bimodal target."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import logsumexp, ndtr
from scipy.stats import norm

from src.gimap.features.fitting.domain.posterior import encoding as E
from src.gimap.features.fitting.domain.posterior import laplace
from src.gimap.features.fitting.domain.posterior.infer import Cut, dimensionless, physical


def test_unequal_proposal_counts_use_the_balance_denominator_at_every_draw():
    # With a normalized target equal to the actual sampling mixture, all importance weights are constant.
    # The two Gaussians overlap, so using only the generating density cannot pass this check.
    def target(z):
        x = z[:, 0]
        return np.log(7.0) + np.logaddexp(norm.logpdf(x, -1) + np.log(.25), norm.logpdf(x, 2) + np.log(.75))

    def start(mean, count):
        z = np.random.default_rng(count).normal(mean, size=(count, 1))

        def proposal(rows):
            return norm.logpdf(rows[:, 0], mean)

        weights = target(z) - proposal(z)
        trace = laplace.ProposalTrace(z, target(z), proposal, (count,))
        return laplace.Local(0, z, weights, np.zeros_like(z), np.zeros(count),
                             logsumexp(weights) - np.log(count), 0, np.array([mean]), trace)

    two = laplace.pool_composition([start(-1, 100), start(2, 300)])
    assert two.log_weights == pytest.approx(np.full(400, np.log(7.0)))
    assert two.log_evidence == pytest.approx(np.log(7.0))
    assert two.ess == pytest.approx(400)
    assert two.shares == pytest.approx([.25, .75])
    assert np.exp([s.log_evidence for s in two.locals]).sum() == pytest.approx(7.0)


def test_bimodal_evidence_and_redundant_start(monkeypatch):
    """Exercise the adaptive sampler with two peaks in the unit prior coordinate of the radius."""
    composition = E.composition_index([("sphere", False)])
    data = Cut.prepare(np.geomspace(.02, 2, 32), np.ones(32), np.ones(32))
    peaks = np.array([.34, .66])
    width = .11
    evidence = 3.5
    areas = ndtr((1 - peaks) / width) - ndtr(-peaks / width)
    fractions = np.array([.6, .4])

    class BimodalLikelihood:
        free_reflection = False

        def __init__(self, data, composition, law, coupling):
            pass

        def coordinate(self, samples):
            return E.to_unit(dimensionless(samples, data.q_max), [data.ratio], composition)[:, 0]

        def __call__(self, samples):
            x = self.coordinate(samples)
            peaks_log = norm.logpdf(x[:, None], peaks, width) + np.log(fractions / areas)
            return np.log(evidence) + logsumexp(peaks_log, axis=1), np.ones((len(x), 1)), np.zeros(len(x))

        def model_curves(self, samples):
            # The local residual supplies positive curvature around each start.
            x = self.coordinate(samples)
            nearest = peaks[np.argmin(abs(x[:, None] - peaks), axis=1)]
            curves = np.broadcast_to(data.intensity, (len(x), len(data.q))).copy()
            curves[:, 0] += (x - nearest) / width
            return None, None, curves

    monkeypatch.setattr(laplace, "Likelihood", BimodalLikelihood)
    monkeypatch.setattr(laplace, "ESS_TARGET", np.inf)

    def run(x, seed, rounds):
        unit = np.full((1, E.DIMS), .5)
        unit[0, 0] = x
        centre = physical(E.from_unit(unit, [data.ratio], composition), data.q_max)[0]
        return laplace.local_posterior(data, composition, centre, .02, np.random.default_rng(seed), 3500,
                                       rounds=rounds)

    left = run(peaks[0], 913, 2)
    right = run(peaks[1], 1009, 3)
    duplicate = run(peaks[0], 1103, 2)
    both = laplace.pool_composition([left, right])
    same_twice = laplace.pool_composition([left, duplicate])
    all_three = laplace.pool_composition([left, right, duplicate])
    for pooled in (laplace.pool_composition([left]), both, same_twice, all_three):
        assert np.exp(pooled.log_evidence) == pytest.approx(evidence, rel=.04)
        assert sum(pooled.shares) == pytest.approx(1)
    assert np.exp(same_twice.log_evidence - left.log_evidence) == pytest.approx(1, rel=.04)
    assert np.exp(all_three.log_evidence - both.log_evidence) == pytest.approx(1, rel=.04)
    assert both.round_sizes == (3500,) * 5


def test_zero_weight_draws_do_not_enter_tempered_intervals():
    samples = np.zeros((2, E.DIMS))
    samples[:, 0] = [1., 10.]
    local = laplace.Local(0, samples, np.array([-np.inf, -np.inf]), samples, np.ones(2), -np.inf, 0., samples[0],
                          laplace.ProposalTrace(samples, np.full(2, -np.inf), lambda z: np.zeros(len(z)), (2,)))
    assert not local.tempered(50).any()
    assert np.isnan(local.quantiles([.16, .84], 50)).all()
    supported = replace(local, log_weights=np.array([-np.inf, 0.]))
    assert supported.quantiles([.16, .84], 50)[:, 0] == pytest.approx([10., 10.])
