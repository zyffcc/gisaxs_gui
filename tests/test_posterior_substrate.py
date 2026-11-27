"""Horizontal cuts with fixed substrate optics, using synthetic particle populations."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.fitting.application import posterior_check, single_predict
from src.gimap.features.fitting.domain import fit_engine, fit_model as fm
from src.gimap.features.fitting.domain.cut_model import model as cut_model
from src.gimap.features.fitting.domain.posterior import encoding as E
from src.gimap.features.fitting.domain.posterior import infer, refine
from src.gimap.features.fitting.domain.posterior.flow import Network
from src.gimap.features.fitting.domain.substrates import Substrate
from tests.repeatability import assert_same

GRAZING = dict(wavelength=0.105, alpha_i=0.40, alpha_f=0.145)
SUBSTRATE = Substrate("Si")
SPHERE = E.composition_index([("sphere", True)])
UPRIGHT = E.composition_index([("vertical_cylinder", True)])
Q = np.linspace(0.02, 2.0, 180)


def _values():
    values = np.full(E.DIMS, np.nan)
    values[[0, 1, 4, 5]] = 4.8, 0.12, 16.0, 0.19
    values[E.GLOBAL] = 0.008, 0.01, 0.6, 0.00001, 0.01, -0.3, 0.7
    return values


def _data(**kwargs):
    return infer.Cut.prepare(Q, np.ones_like(Q), np.ones_like(Q), substrate=SUBSTRATE, grazing=GRAZING, **kwargs)


def test_level_four_refines_noise_free_spheres_and_has_no_fitted_reflection():
    data = _data()
    geometry = data.geometry()
    assert data.level == 4 and not data.reflected and geometry.mode == "dwba"
    assert geometry.qz_born * 4.8 > 2.3
    # Integrate the four amplitudes on a fine size grid, including a structured particle term and a beam tail.
    form = cut_model.population_terms("sphere", Q, radius=4.8, radius_spread=0.12, distance=16.0,
                                      distance_spread=0.19, geometry=geometry, panels=32)[0]
    y = 2500 * form / form.max() + 1200 * cut_model.tail(Q, 0.008, 3.2) + 3.0
    proposals = Network.load(Path(__file__).parent / "data" / "ml1d" / "flow_fixture.npz")
    best = refine.predict(proposals, Q, y, np.sqrt(y), substrate=SUBSTRATE.to_dict(), grazing=GRAZING,
                          compositions=[SPHERE], reflected_beam=True, samples=150, local_samples=50, seed=1)[0]
    for name, truth in zip(("R_1", "sigma_R_1", "D_1", "sigma_nn_1"), (4.8, 0.12, 16.0, 0.19)):
        assert best.values[name].best == pytest.approx(truth, rel=0.01)
    assert fm.grazing(best.model) == geometry
    assert best.extras["diagnostics"]["substrate_level"] == 4
    assert best.extras["diagnostics"]["substrate_label"] == "Si"
    assert best.extras["reflection"] is None and not best.model.globals["refl_amp"].free
    assert not best.model.globals["refl_phase"].free
    assert not any(flag.startswith("qzR") for flag in best.flags)
    v = _values()
    w = v.copy()
    w[-2:] = 0.87, -0.44
    columns = infer.Likelihood(data, SPHERE).columns([v, w])
    assert_same(columns[1], columns[0])
    np.testing.assert_allclose(columns[0, 0], form / form.max(), rtol=1e-4)


@pytest.mark.parametrize("coupling", ["lma", "da"])
def test_known_upright_height_distribution_agrees_with_manual_fit(coupling):
    data = _data(upright_heights={"1": [14.0, 0.25]})
    v = _values()
    model = refine.fit_model(v, UPRIGHT, data, coupling=coupling)
    cylinder = model.components[0]
    assert cylinder.heights and not cylinder.params["h"].free and not cylinder.params["sigma_h"].free
    assert (cylinder.value("h"), cylinder.value("sigma_h")) == (14.0, 0.25)
    manual = fm.component_form(cylinder, Q, geometry=data.geometry())
    distributed = infer.Likelihood(data, UPRIGHT, coupling=coupling).columns(v)[0, 0]
    np.testing.assert_allclose(distributed, manual / manual.max(), rtol=1e-12)
    fixed = infer.Likelihood(_data(), UPRIGHT, coupling=coupling).columns(v)[0, 0]
    if coupling == "da":
        assert np.max(np.abs(distributed - fixed)) > 0.01
    else:
        np.testing.assert_array_equal(distributed, fixed)


@pytest.mark.parametrize("key", GRAZING)
def test_known_substrate_requires_wavelength_and_both_angles(key):
    incomplete = {k: v for k, v in GRAZING.items() if k != key}
    with pytest.raises(ValueError, match="wavelength and both"):
        infer.Cut.prepare(Q, Q, substrate=SUBSTRATE, grazing=incomplete)


@pytest.mark.parametrize("law", ["lognormal_median", "schulz"])
def test_distributed_height_uses_its_own_size_law_and_coupling(law):
    height = dict(h=14.0, sigma_h=0.33, law=law, coupling="da")
    data = _data(upright_heights={"1": height})
    v = _values()
    model = refine.fit_model(v, UPRIGHT, data, law="number_gauss", coupling="lma")
    upright = model.components[0]
    assert (upright.law, upright.coupling, upright.value("h"), upright.value("sigma_h")) == (law, "da", 14.0, 0.33)
    manual = fm.new_component("vertical_cylinder", radius=v[0], distance=v[4], heights=True, law=law)
    reference = fm.FitModel((replace(manual, coupling="da"),)).with_values(
        {(0, "sigma_R"): v[1], (0, "sigma_nn"): v[5], (0, "h"): height["h"], (0, "sigma_h"): height["sigma_h"]})
    form = fm.component_form(reference.components[0], Q, geometry=data.geometry())
    predicted = infer.Likelihood(data, UPRIGHT, "number_gauss", "lma").columns(v)[0, 0]
    np.testing.assert_allclose(predicted, form / form.max(), rtol=2e-12)


def _two_populations():
    v = _values()
    v[[6, 7, 10, 11]] = 7.5, 0.18, 27.0, 0.2
    return v


def test_two_upright_height_distributions_stay_with_their_radius_rank_in_refinement():
    composition = E.composition_index([("vertical_cylinder", True)] * 2)
    heights = {"1": dict(h=6.0, sigma_h=0.30, law="lognormal_median", coupling="da"),
               "2": dict(h=17.0, sigma_h=0.22, law="schulz", coupling="lma")}
    cut = _data(upright_heights=heights)
    model = refine.fit_model(_two_populations(), composition, cut)
    data = fit_engine.FitData.prepare(Q, np.ones_like(Q), np.ones_like(Q))
    problem = fit_engine._Problem(model, data, None, None, 1)
    where = {path: i for i, (path, _parameter) in enumerate(problem.nonlinear)}
    for first, second in [(0.96, 0.02), (1.0, 0.0)]:
        # Independent local-search steps would put the first radius above the second.
        t = problem.start()
        t[where[(0, "R")]], t[where[(1, "R")]] = first, second
        t[where[(0, "D")]] = t[where[(1, "D")]] = 0.0
        displaced = model.with_values({path: value for (path, _), value in zip(problem.nonlinear, problem.decode(t))})
        c1, c2 = displaced.components
        assert c1.value("R") <= c2.value("R")
        assert all(c.value("D") >= 2 * c.value("R") - 1e-12 for c in (c1, c2))
        assert single_predict.upright_distributions(displaced) == heights
        v = refine.model_values(displaced, composition)
        predicted = infer.Likelihood(cut, composition).columns(v)[0]
        for i, component in enumerate(displaced.components):
            form = fm.component_form(component, Q, geometry=cut.geometry())
            np.testing.assert_allclose(predicted[i], form / form.max(), rtol=2e-12)


def test_posterior_check_replays_height_conventions_and_shared_defaults_after_serialization():
    composition = E.composition_index([("vertical_cylinder", True), ("sphere", True)])
    heights = {"1": dict(h=6.0, sigma_h=0.3, law="lognormal_median", coupling="da")}
    cut = _data(upright_heights=heights)
    v = _two_populations()
    model = refine.fit_model(v, composition, cut, law="number_gauss", coupling="lma")
    restored = fm.model_from_dict(fm.model_to_dict(model))
    assert single_predict.upright_distributions(restored) == heights
    assert restored.inequalities == model.inequalities
    solution = single_predict.Predicted(restored, 1.0, "two populations", 1.0, intervals={"R_1": (v[0],) * 7},
                                        diagnostics={"substrate_level": 4, "model_error_floor": 0.0})
    data = fit_engine.FitData.prepare(Q, cut.intensity, cut.sigma)
    check = posterior_check.target_for(restored, data, solutions=[solution])
    assert (check.target.likelihood.law, check.target.likelihood.coupling) == ("number_gauss", "lma")
    np.testing.assert_array_equal(check.target.likelihood.columns(v), infer.Likelihood(cut, composition).columns(v))
