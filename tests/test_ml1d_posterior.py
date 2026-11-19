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
        top = infer.AMPLITUDE_RANGE * counts.max()
        scaled = x[b] / sigma[None, :]
        _s, log_det_h = np.linalg.slogdet(scaled[positive] @ scaled[positive].T)
        half = [0.5 * math.log(math.pi / 2) - 0.5 * math.log(scaled[k] @ scaled[k]) for k in np.flatnonzero(~positive)]
        laplace = (-len(amplitudes[b]) * math.log(top) + 0.5 * positive.sum() * math.log(2 * math.pi) - 0.5 * log_det_h
                   + sum(half))
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


def test_several_compositions_at_once():
    network = Network.load(DATA / "flow_fixture.npz")
    q, counts, sigma = _curve(seed=5)
    solutions = refine.predict(network, q, counts, sigma, allowed=E.select(("sphere", "vertical_cylinder"), 1),
                               shortlist=3, samples=100, local_samples=100)
    assert solutions and abs(sum(s.probability for s in solutions) - 1.0) < 0.05
    assert len({s.composition for s in solutions}) >= 1


def test_few_effective_samples_widen_the_intervals():
    """With one dominant weight the quantiles would sit on one sample; tempered weights give at least ESS_MIN."""
    from src.gimap.features.fitting.domain.posterior.laplace import Local

    rng = np.random.default_rng(3)
    composition = E.composition_index([("sphere", False)])
    samples = np.full((400, E.DIMS), np.nan)
    samples[:, 0] = rng.normal(5.0, 0.5, 400)
    log_w = rng.normal(0.0, 0.1, 400)
    log_w[0] = 40.0  # one sample carries all the weight
    local = Local(composition, samples, log_w, np.zeros((400, 1)), np.zeros(400), 0.0, 1.0, samples[0])
    assert 1.0 / np.sum(local.weights**2) < 1.5
    narrow = local.quantiles([0.16, 0.84])[:, 0]
    wide = local.quantiles([0.16, 0.84], 50)[:, 0]
    assert narrow[1] - narrow[0] < 0.1 and wide[1] - wide[0] > 0.5
    assert 1.0 / np.sum(local.tempered(50) ** 2) >= 50 - 1e-6
    np.testing.assert_allclose(local.tempered(1), local.weights)


def test_moved_starts_keep_the_order_and_the_paracrystal():
    """R × 0.5 and × 2 at the same D, and the paracrystals of two populations exchanged, where the order of the radii
    and D ≥ 2R still hold (posterior/moves.py)."""
    from src.gimap.features.fitting.domain.posterior import moves

    two = E.composition_index([("sphere", True), ("sphere", True)])
    v = np.full(E.DIMS, np.nan)
    v[0:6] = 2.0, 0.2, np.nan, np.nan, 30.0, 0.15
    v[6:12] = 6.0, 0.2, np.nan, np.nan, 14.0, 0.25
    starts = moves.starts(v, two)
    radii = {(round(s[0], 3), round(s[6], 3)) for s in starts}
    assert radii == {(1.0, 6.0), (4.0, 6.0), (2.0, 3.0), (2.0, 6.0)}  # the second not doubled: D = 14 < 2 × 12
    swapped = [s for s in starts if s[4] == 14.0]
    assert len(swapped) == 1 and swapped[0][10] == 30.0 and swapped[0][5] == 0.25 and swapped[0][11] == 0.15
    for s in starts:
        assert s[0] < s[6] and s[4] >= 2 * s[0] and s[10] >= 2 * s[6]
    assert moves.starts(_values(), SPHERE) == []


def test_wider_proposal_is_the_exact_mixture_density():
    """Samples from base widths (1, 2) carry log q of the equal mixture of both widths (flow.Network.sample)."""
    network = Network.load(DATA / "flow_fixture.npz")
    q, counts, sigma = _curve()
    data = infer.Cut.prepare(q, counts, sigma)
    binned = E.bin_curve(data.q, data.intensity, data.sigma, data.q_max, data.ratio)
    h = network.summary(E.features(*(a[None] for a in binned)), E.conditions([data.ratio], [0.0]))
    context = network.context(h, SPHERE)
    z1, log1 = network.sample(context, SPHERE, 50, np.random.default_rng(4))
    z2, log2 = network.sample(context, SPHERE, 50, np.random.default_rng(4), widths=(1.0, 1.0))
    assert np.allclose(z1, z2) and np.allclose(log1, log2)
    z, log_mix = network.sample(context, SPHERE, 40, np.random.default_rng(5), widths=(1.0, 2.0))
    active = np.repeat(E.active(SPHERE), z.shape[0], axis=0)
    rows = np.repeat(np.asarray(context, float)[:1], z.shape[0], axis=0)
    log_one = network.log_prob(z, rows, active)
    u, _log_det = network.forward(z, rows, active)
    d = active.sum(axis=1)
    log_two = log_one + np.sum(np.where(active, 0.375 * u * u, 0.0), axis=1) - d * np.log(2.0)
    assert np.allclose(log_mix, np.logaddexp(log_one, log_two) - np.log(2.0), atol=1e-6)


def test_shortlist_keeps_every_number_of_populations():
    """Besides the head's most probable compositions, the most probable one of every number of populations
    (``infer.PER_COUNT``), most probable first."""

    class Head:
        def composition_log_probs(self, _h):
            out = np.full((1, len(E.COMPOSITIONS)), -50.0)
            for rank, c in enumerate(allowed):
                out[0, c] = -float(rank) if len(E.COMPOSITIONS[c]) == 3 else -10.0 - rank
            return out

    allowed = E.select(("sphere", "vertical_cylinder"), 3, structure=True)
    old = infer.PER_COUNT
    try:
        infer.PER_COUNT = 0
        assert {len(E.COMPOSITIONS[c]) for c, _p in infer.candidates(Head(), None, allowed, 3)} == {3}
        infer.PER_COUNT = 1
        chosen = infer.candidates(Head(), None, allowed, 3)
        counts = [len(E.COMPOSITIONS[c]) for c, _p in chosen]
        assert counts[:3] == [3, 3, 3] and sorted(counts[3:]) == [1, 2]
        assert [p for _c, p in chosen] == sorted((p for _c, p in chosen), reverse=True)
    finally:
        infer.PER_COUNT = old


def test_window_flags_name_populations_the_curve_does_not_resolve():
    """``large`` when q_min·R ≥ 0.7 (only the decay is measured), ``small`` when q_max·R < 1 (refine.window_flags)."""
    from src.gimap.features.fitting.domain import fit_model as fm

    q = np.linspace(0.02, 2.0, 50)
    model = fm.FitModel(tuple(fm.new_component("sphere", radius=r, distance=None, structure=False)
                              for r in (40.0, 0.3, 5.0)))
    assert refine.window_flags(model, q) == ["small:R_1", "large:R_3"]
    assert refine.window_flags(fm.FitModel((fm.new_component("sphere", radius=5.0, distance=None,
                                                             structure=False),)), q) == []
    from src.gimap.features.fitting.infrastructure.adapters import posterior_job

    sentences = posterior_job.flag_sentences(["large:R_3", "small:R_1"])
    assert sentences[0].startswith("R_3: its Guinier region") and sentences[1].startswith("R_1: smaller than")
