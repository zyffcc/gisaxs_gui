"""Horizontal cuts with fixed substrate optics, using synthetic particle populations."""

from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.fitting.domain import fit_model as fm
from src.gimap.features.fitting.domain.cut_model import model as cut_model
from src.gimap.features.fitting.domain.posterior import encoding as E
from src.gimap.features.fitting.domain.posterior import infer, refine
from src.gimap.features.fitting.domain.posterior.flow import Network
from src.gimap.features.fitting.domain.substrates import Substrate

GRAZING = dict(wavelength=0.105, alpha_i=0.40, alpha_f=0.145)
SUBSTRATE = Substrate("Si")
SPHERE = E.composition_index([("sphere", True)])
UPRIGHT = E.composition_index([("vertical_cylinder", True)])
Q = np.linspace(0.02, 2.0, 180)


def _values():
    values = np.full(E.DIMS, np.nan)
    values[[0, 1, 4, 5]] = 4.8, 0.12, 16.0, 0.19
    values[E.GLOBAL] = 0.008, 3.2, 0.00001, 0.01, -0.3, 0.7
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
    np.testing.assert_array_equal(columns[0], columns[1])
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
