"""Predict's work in worker processes (``infrastructure/adapters/posterior_pool``) and the exact caches of the model:
the same numbers as without, only sooner."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.fitting.domain import fit_engine, fit_model as fm
from src.gimap.features.fitting.domain.cut_model import model as cut
from src.gimap.features.fitting.domain.cut_model import resolution
from src.gimap.features.fitting.domain.posterior import encoding as E
from src.gimap.features.fitting.domain.posterior import refine
from src.gimap.features.fitting.domain.posterior.flow import Network
from src.gimap.features.fitting.infrastructure.adapters import posterior_pool

DATA = Path(__file__).resolve().parent / "data" / "ml1d"


def _curve(seed=3):
    q = np.linspace(0.02, 2.0, 300)
    population = cut.Population("sphere", 4.0, 0.15, distance=14.0, distance_spread=0.2)
    truth = 1e4 * cut.unit_terms([population], cut.Settings(), q)[0] + 2.0
    counts = np.random.default_rng(seed).poisson(truth).astype(float)
    return q, counts, np.sqrt(np.maximum(counts, 1.0))


def _numbers(solutions):
    return [(s.composition, s.name, s.probability, s.chi2_reduced, s.flags,
             {k: (v.best, v.median, v.low68, v.high68) for k, v in s.values.items()},
             fm.model_to_dict(s.model)) for s in solutions]


def test_predict_in_worker_processes_gives_the_numbers_of_one_process():
    network = Network.load(DATA / "flow_fixture.npz")
    q, counts, sigma = _curve()
    options = dict(allowed=E.select(("sphere", "vertical_cylinder"), 2), samples=200, local_samples=150, seed=1)
    alone = refine.predict(network, q, counts, sigma, **options)
    with posterior_pool.FitPool(2).start() as pool:
        pooled = refine.predict(network, q, counts, sigma, threads=2, pool=pool, **options)
        assert not pool.broken
    assert alone and _numbers(pooled) == _numbers(alone)


def _square(n):
    return n * n


def test_a_lost_pool_computes_in_this_process():
    pool = posterior_pool.FitPool(2).start()
    try:
        assert list(pool.ordered(_square, [(n,) for n in range(5)])) == [0, 1, 4, 9, 16]
        for process in list(pool._executor._processes.values()):
            process.terminate()
            process.join(10)
        assert list(pool.ordered(_square, [(n,) for n in range(4)])) == [0, 1, 4, 9]
        assert pool.broken
        assert list(pool.ordered(_square, [(3,)])) == [9]
    finally:
        pool.close()


def test_workers_fit_the_free_memory():
    reserve, worker = posterior_pool.RESERVE, posterior_pool.WORKER_MEMORY
    assert posterior_pool.workers_for(12, free=reserve + 100 * worker) == 12
    assert posterior_pool.workers_for(12, free=reserve + 4 * worker) == 4
    assert posterior_pool.workers_for(12, free=reserve // 2) == 0
    assert posterior_pool.free_memory() is None or posterior_pool.free_memory() > 0


def _two_populations(level=1):
    """Two populations with a constant q resolution: a Predict model (its grid) or a manual one (``table_grid``)."""
    model = fm.FitModel((fm.new_component("vertical_cylinder", radius=3.0), fm.new_component("sphere", radius=6.0)))
    return fm.replace(model.with_parameter(("globals", "q_resolution"), value=0.02), predict_level=level)


@pytest.mark.parametrize("level", [1, None])
def test_the_cached_smeared_terms_are_the_computed_ones(level):
    model = _two_populations(level)
    q = np.geomspace(0.02, 2.0, 300)
    cache = {}
    first = fm.evaluate(model, q, cache=cache)
    for _ in range(2):  # from the cache, and once more
        assert np.array_equal(fm.evaluate(model, q, cache=cache), first)
    assert np.array_equal(fm.evaluate(model, q), first)  # without a cache
    moved = model.with_parameter((1, "R"), value=6.5)
    assert np.array_equal(fm.evaluate(moved, q, cache=cache), fm.evaluate(moved, q))
    scaled = model.with_parameter((0, "Int"), value=3.0)  # a scale does not enter the cached terms
    assert np.array_equal(fm.evaluate(scaled, q, cache=cache), fm.evaluate(scaled, q))
    forms = fm.particle_forms(model, q, None, cache)
    forms[0][:] = -1.0  # a caller that changes what it got changes nothing kept
    assert np.array_equal(fm.evaluate(model, q, cache=cache), first)


@pytest.mark.parametrize("level", [1, None])
def test_a_fit_with_the_caches_gives_the_fit_without(monkeypatch, level):
    model = _two_populations(level)
    q = np.geomspace(0.02, 2.0, 200)
    y = fm.evaluate(model.with_parameter((1, "R"), value=6.4), q) * np.random.default_rng(1).normal(1, 0.02, q.size)
    data = fit_engine.FitData.prepare(q, y, 0.02 * y)
    cached = fit_engine.fit(model, data, method="local")
    monkeypatch.setattr(fm.fit_resolution, "SMEARED_CACHE", 0)  # nothing is kept: every term computed again
    computed = fit_engine.fit(model, data, method="local")
    assert fm.model_to_dict(cached.model) == fm.model_to_dict(computed.model)
    assert cached.chi2_reduced == computed.chi2_reduced


def test_predict_grid_is_kept_and_handed_out_as_a_copy():
    q = np.geomspace(0.05, 2.5, 400)
    expected = resolution.table_grid(q, max(0.004, resolution.PREDICT_MAX_RELATIVE * 2.5), resolution.PREDICT_TABLE_POINTS)
    grid = resolution.predict_grid(q, 0.004)
    assert np.array_equal(grid, expected)
    grid[:] = 0.0
    assert np.array_equal(resolution.predict_grid(q, 0.004), expected)
