"""LMA and DA in Fitting: P·S, or P + |<F>|²(S − 1) with the amplitude averaged by number (Physics and Methods 9.5)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.gimap.features.fitting.domain import fit_model as fm
from src.gimap.features.fitting.domain.cut_model import sizes
from src.gimap.features.fitting.domain.cut_model.shapes import sphere_amplitude
from src.gimap.features.fitting.domain.cut_model.structure import paracrystal

Q = np.linspace(0.01, 2.0, 300)


def _component(family: str, spread: float, coupling: str = "lma"):
    component = fm.new_component(family, radius=4.0, distance=13.0)
    params = {**component.params, "sigma_R": replace(component.params["sigma_R"], value=spread)}
    return replace(component, params=params, coupling=coupling)


@pytest.mark.parametrize("family", ["sphere", "vertical_cylinder"])
def test_one_size_gives_the_same_curve_in_both(family) -> None:
    lma = fm.component_form(_component(family, 0.0), Q)
    np.testing.assert_allclose(fm.component_form(_component(family, 0.0, "da"), Q), lma, rtol=1e-12)


@pytest.mark.parametrize("law", ["number_gauss", "intensity_gauss"])
def test_the_sphere_da_is_the_number_average_of_the_amplitudes(law) -> None:
    component = replace(_component("sphere", 0.2, "da"), law=law)
    radii, number = (a[0] for a in sizes.nodes(law, 4.0, 0.2, power=6, panels=256))  # number weights
    volumes = radii**3
    amplitudes = np.column_stack([volumes[j] * sphere_amplitude(Q * radii[j]) for j in range(radii.size)])
    mean_square = (amplitudes**2 @ number) / (number @ volumes**2)
    coherent = (amplitudes @ number) ** 2 / (number @ volumes**2)
    structure = paracrystal(Q, 13.0, component.value("sigma_nn"))[0]
    np.testing.assert_allclose(fm.component_form(component, Q), mean_square + coherent * (structure - 1), rtol=1e-6)


def test_da_adds_intensity_at_low_q_and_the_coupling_is_saved() -> None:
    lma, da = (fm.component_form(_component("sphere", 0.2, coupling), Q) for coupling in ("lma", "da"))
    assert da[0] > 1.2 * lma[0] and np.all(da > 0)
    model = fm.FitModel(components=(_component("vertical_cylinder", 0.2, "da"),))
    assert fm.model_from_dict(fm.model_to_dict(model)).components[0].coupling == "da"
    old = fm.model_to_dict(model)
    del old["components"][0]["coupling"]
    assert fm.model_from_dict(old).components[0].coupling == "lma"  # a model saved before has LMA
