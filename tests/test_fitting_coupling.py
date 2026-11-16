"""LMA and DA in Fitting: P·S, or P + |<F>|²(S − 1) with the amplitude averaged by number (Physics and Methods 9.5)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.gimap.features.fitting.domain import fit_model as fm
from src.gimap.features.fitting.domain.scattering_model import _F_sphere_amp, _gaussian_grid, structure_factor_1d

Q = np.linspace(0.01, 2.0, 300)


def _component(family: str, spread: float, coupling: str = "lma"):
    component = fm.new_component(family, radius=4.0, distance=13.0)
    params = {**component.params, "sigma_R": replace(component.params["sigma_R"], value=spread)}
    return replace(component, params=params, coupling=coupling)


@pytest.mark.parametrize("family", ["sphere", "vertical_cylinder"])
def test_one_size_gives_the_same_curve_in_both(family) -> None:
    lma = fm.component_form(_component(family, 0.0), Q)
    np.testing.assert_allclose(fm.component_form(_component(family, 0.0, "da"), Q), lma, rtol=1e-12)


def test_the_sphere_da_is_the_number_average_of_the_amplitudes() -> None:
    component = _component("sphere", 0.2, "da")
    radii, weights = _gaussian_grid(4.0, 0.8, nsig=4.0, n=25)
    number = weights / radii**6  # Fitting's sphere weights are intensity weights: by number w / V²
    number /= number.sum()
    volumes = radii**3
    amplitudes = np.column_stack([volumes[j] * _F_sphere_amp(Q, radii[j]) for j in range(radii.size)])
    mean_square = (amplitudes**2 @ number) / (number @ volumes**2)
    coherent = (amplitudes @ number) ** 2 / (number @ volumes**2)
    structure = structure_factor_1d(Q, 13.0, 13.0 * component.value("sigma_D"))
    np.testing.assert_allclose(fm.component_form(component, Q), mean_square + coherent * (structure - 1), rtol=1e-10)


def test_da_adds_intensity_at_low_q_and_the_coupling_is_saved() -> None:
    lma, da = (fm.component_form(_component("sphere", 0.2, coupling), Q) for coupling in ("lma", "da"))
    assert da[0] > 1.2 * lma[0] and np.all(da > 0)
    model = fm.FitModel(components=(_component("vertical_cylinder", 0.2, "da"),))
    assert fm.model_from_dict(fm.model_to_dict(model)).components[0].coupling == "da"
    old = fm.model_to_dict(model)
    del old["components"][0]["coupling"]
    assert fm.model_from_dict(old).components[0].coupling == "lma"  # a model saved before has LMA
