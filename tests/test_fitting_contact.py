"""Fitting keeps the paracrystal's distance D at or above 2R: particles of the mean radius do not overlap."""

from __future__ import annotations

import numpy as np

from src.gimap.features.fitting.domain import fit_engine as fe
from src.gimap.features.fitting.domain import fit_model as fm


def _data(model, q):
    intensity = fm.evaluate(model, q)
    return fe.FitData.prepare(q, intensity, 0.02 * intensity)


def _problem(model, q):
    return fe._Problem(model, _data(model, q), None, None, 1)


def test_with_both_free_every_searched_distance_is_at_least_two_radii_and_the_mapping_inverts() -> None:
    model = fm.FitModel(components=(fm.new_component("sphere", radius=6.0, distance=40.0),))
    problem = _problem(model, np.linspace(0.05, 2.0, 200))
    paths = [path for path, _p in problem.nonlinear]
    d_index, r_index = paths.index((0, "D")), paths.index((0, "R"))
    assert problem.pairs == [(d_index, r_index)]
    rng = np.random.default_rng(3)
    for t in rng.uniform(0.001, 0.999, size=(500, len(paths))):
        values = problem.decode(t)
        assert values[d_index] >= fe.CONTACT * values[r_index] * (1 - 1e-12) or values[d_index] >= 0.99 * problem.upper[d_index]
        assert np.allclose(problem.encode(values), t, atol=1e-7)


def test_with_one_fixed_the_range_of_the_other_is_narrowed() -> None:
    q = np.linspace(0.05, 2.0, 200)
    model = fm.FitModel(components=(fm.new_component("sphere", radius=10.0, distance=40.0),))
    fixed_r = model.with_parameter((0, "R"), free=False)
    problem = _problem(fixed_r, q)
    d_index = [path for path, _p in problem.nonlinear].index((0, "D"))
    assert problem.lower[d_index] == 20.0
    fixed_d = model.with_parameter((0, "D"), value=30.0, free=False)
    problem = _problem(fixed_d, q)
    r_index = [path for path, _p in problem.nonlinear].index((0, "R"))
    assert problem.upper[r_index] == 15.0


def test_a_fit_that_would_overlap_the_particles_stops_at_contact() -> None:
    q = np.linspace(0.05, 2.0, 300)
    truth = fm.FitModel(components=(fm.new_component("sphere", radius=5.0, structure=False),))
    data = _data(truth, q)
    start = fm.FitModel(components=(fm.new_component("sphere", radius=5.0, distance=6.0),))  # D < 2R
    result = fe.fit(start, data, method="local")
    component = result.model.components[0]
    assert component.value("D") >= fe.CONTACT * component.value("R") * (1 - 1e-9)


def test_without_a_paracrystal_nothing_is_paired() -> None:
    model = fm.FitModel(components=(fm.new_component("sphere", radius=5.0, structure=False),))
    assert _problem(model, np.linspace(0.05, 2.0, 100)).pairs == []
