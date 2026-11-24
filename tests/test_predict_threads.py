"""Predict parameters on several threads gives the numbers of one thread (``posterior.parallel``)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.fitting.domain.cut_model import model as cut
from src.gimap.features.fitting.domain.posterior import encoding as E
from src.gimap.features.fitting.domain.posterior import infer, parallel, refine
from src.gimap.features.fitting.domain.posterior.flow import Network

DATA = Path(__file__).resolve().parent / "data" / "ml1d"
SPHERE = E.composition_index([("sphere", True)])
TWO = E.composition_index([("sphere", True), ("vertical_cylinder", True)])


def _curve(seed=0):
    q = np.linspace(0.02, 2.0, 300)
    population = cut.Population("sphere", 4.0, 0.15, distance=14.0, distance_spread=0.2)
    truth = 1e4 * cut.unit_terms([population], cut.Settings(), q)[0] + 2.0
    counts = np.random.default_rng(seed).poisson(truth).astype(float)
    return q, counts, np.sqrt(np.maximum(counts, 1.0))


@pytest.mark.parametrize("law", ["number_gauss", "intensity_gauss"])
def test_the_chunked_likelihood_is_the_single_call(law):
    network = Network.load(DATA / "flow_fixture.npz")
    q, counts, sigma = _curve()
    data = infer.Cut.prepare(q, counts, sigma, 0.6, 0.25)  # two paths: every sphere sample its own reflection
    binned = E.bin_curve(data.q, data.intensity, data.sigma, data.q_max, data.ratio)
    h = network.summary(E.features(*(a[None] for a in binned)), infer.conditions(data))
    z, _log_q = network.sample(network.context(h, TWO), TWO, 100, np.random.default_rng(4))
    likelihood = infer.Likelihood(data, TWO, law)
    samples = infer.physical(E.from_z(z, np.full(100, data.ratio), TWO), data.q_max, disk=likelihood.free_reflection)
    alone = likelihood(samples)
    shared = parallel.evaluate(likelihood, samples, threads=4)
    for a, b in zip(alone, shared):
        assert np.array_equal(a, b, equal_nan=True)


def test_predict_on_four_threads_returns_the_solutions_of_one():
    network = Network.load(DATA / "flow_fixture.npz")
    q, counts, sigma = _curve(seed=3)
    runs = [refine.predict(network, q, counts, sigma, allowed=E.select(("sphere", "vertical_cylinder"), 2),
                           samples=200, local_samples=150, seed=1, threads=threads) for threads in (1, 4)]
    assert len(runs[0]) == len(runs[1]) > 0
    for one, four in zip(*runs):
        assert (one.composition, one.name, one.probability, one.chi2_reduced) == (
            four.composition, four.name, four.probability, four.chi2_reduced)
        assert {k: (v.best, v.low68, v.high68) for k, v in one.values.items()} == {
            k: (v.best, v.low68, v.high68) for k, v in four.values.items()}


def test_blas_runs_on_one_thread_while_predict_runs(monkeypatch):
    """A BLAS on several threads, called from several threads at once, changes last digits; Predict sets it to one."""
    from threadpoolctl import threadpool_info

    from src.gimap.features.fitting.domain import fit_engine

    seen = []
    fit = fit_engine.fit

    def recorded(*args, **kwargs):
        seen.extend(info["num_threads"] for info in threadpool_info() if info["user_api"] == "blas")
        return fit(*args, **kwargs)

    monkeypatch.setattr(fit_engine, "fit", recorded)
    network = Network.load(DATA / "flow_fixture.npz")
    q, counts, sigma = _curve(seed=3)
    refine.predict(network, q, counts, sigma, compositions=[SPHERE], samples=100, local_samples=100, threads=2)
    assert seen and set(seen) == {1}


def test_ordered_keeps_the_order_and_stops_early():
    started = []

    def slow_square(n):
        started.append(n)
        return n * n

    assert list(parallel.ordered(slow_square, range(20), threads=4)) == [n * n for n in range(20)]
    lazy = parallel.ordered(slow_square, range(5), threads=1)
    started.clear()
    assert next(lazy) == 0 and started == [0]  # one thread: each item only when it is asked for
    lazy.close()


def test_threads_from_the_environment(monkeypatch):
    monkeypatch.setenv(parallel.ENVIRONMENT, "3")
    assert parallel.default_threads() == 3
    monkeypatch.setenv(parallel.ENVIRONMENT, "")
    assert 1 <= parallel.default_threads() <= parallel.THREADS_MAX
