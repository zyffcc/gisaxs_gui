"""Fitting's Gaussian q resolution σ_q: the particle terms convolved with a Gaussian, the cut taken as symmetric."""

from __future__ import annotations

import numpy as np
import pytest

from src.gimap.features.fitting.domain import fit_model as fm
from src.gimap.features.fitting.domain.fit1d_bounds import model_constraints

Q = np.linspace(0.02, 2.0, 400)


def _model(family: str = "sphere") -> fm.FitModel:
    return fm.FitModel(components=(fm.new_component(family, radius=4.0, distance=13.0),))


def _brute_force(model: fm.FitModel, sigma: float) -> np.ndarray:
    fine = np.linspace(-2.2, 2.2, 44000)
    values = fm.evaluate(model, np.abs(fine))
    weights = np.exp(-0.5 * ((fine[None, :] - Q[:, None]) / sigma) ** 2)
    return (weights @ values) / weights.sum(axis=1)


@pytest.mark.parametrize("family", ["sphere", "vertical_cylinder"])
def test_the_smeared_curve_is_the_convolution_of_the_symmetric_cut(family) -> None:
    model = _model(family)
    smeared = fm.evaluate(model.with_parameter(("globals", "q_resolution"), value=0.02), Q)
    np.testing.assert_allclose(smeared, _brute_force(model, 0.02), rtol=1e-3)


def test_no_resolution_by_default_and_the_parts_add_up() -> None:
    model = _model()
    assert model.globals["q_resolution"].value == 0 and not model.globals["q_resolution"].free
    smeared = model.with_parameter(("globals", "q_resolution"), value=0.01).with_parameter(
        ("globals", "res_amplitude"), value=5.0)
    total, parts = fm.evaluate(smeared, Q, parts=True)
    np.testing.assert_allclose(total, np.sum(list(parts.values()), axis=0), rtol=1e-12)
    assert fm.model_from_dict(fm.model_to_dict(smeared)).globals["q_resolution"].value == 0.01


def test_fit1d_does_not_pretend_to_apply_a_resolution() -> None:
    with pytest.raises(ValueError, match="q_resolution"):
        model_constraints(_model().with_parameter(("globals", "q_resolution"), value=0.01))
    model_constraints(_model())  # the default (none) is no constraint
