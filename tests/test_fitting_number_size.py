"""Fitting's sizes by number and by intensity, and σnn/D (Physics and Methods, sections 9.3 and 9.5).

A component's size law says what R and σR/R are: the mean and relative spread of a Gaussian or a log-normal of the
particle count, or of a Gaussian of the scattered intensity (earlier Fitting), which by number is G(R)/R⁶ for
spheres (R⁴ for cylinders). The results give both."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from src.gimap.features.fitting.application.single_fit import derived_values
from src.gimap.features.fitting.domain import fit_model as fm


def _component(family: str, spread: float, structure: bool = True, law: str = "number_gauss"):
    component = fm.new_component(family, radius=5.0, distance=20.0, structure=structure, law=law)
    return replace(component, params={**component.params, "sigma_R": replace(component.params["sigma_R"], value=spread)})


def test_a_number_law_reports_its_own_mean_and_spread() -> None:
    component = _component("sphere", 0.1)
    assert fm.number_size(component) == pytest.approx((5.0, 0.1), rel=1e-3)  # the ±4σ cut narrows it by 5e-4
    by_intensity = fm.intensity_size(component)
    assert by_intensity[0] > 5.0  # the large particles dominate the intensity


def test_a_narrow_intensity_sphere_by_number_is_the_moment_of_g_over_r6() -> None:
    component = _component("sphere", 0.1, law="intensity_gauss")
    x = np.linspace(5 - 4 * 0.5, 5 + 4 * 0.5, 20001)
    w = np.exp(-0.5 * ((x - 5) / 0.5) ** 2) / x**6
    w /= w.sum()
    mean = float(np.sum(w * x))
    assert fm.number_size(component) == pytest.approx((mean, math.sqrt(np.sum(w * (x - mean) ** 2)) / mean), rel=1e-4)
    assert fm.number_size(component)[0] < 5.0  # fewer large particles than their intensity suggests
    assert fm.intensity_size(component) == pytest.approx((5.0, 0.1), rel=1e-3)


def test_broad_intensity_distributions_have_no_number_mean_and_number_laws_always_have_one() -> None:
    assert fm.number_size(_component("sphere", 0.2, law="intensity_gauss")) is None  # G/R⁶ rises towards small R
    assert fm.number_size(_component("cylinder", 0.3, law="intensity_gauss")) is None
    assert fm.number_size(_component("vertical_cylinder", 0.3)) == pytest.approx((5.0, 0.3), abs=5e-3)
    record = derived_values(_component("sphere", 0.3, law="intensity_gauss"))
    assert record["R_number_mean_nm"] is None and record["sigma_R_number_rel"] is None
    assert record["R_intensity_mean_nm"] == pytest.approx(5.0, rel=1e-3)


def test_a_log_normal_by_intensity_is_a_log_normal_with_a_larger_median() -> None:
    component = _component("sphere", 0.3, law="lognormal")
    assert fm.number_size(component) == pytest.approx((5.0, 0.3), rel=1e-3)
    assert fm.intensity_size(component) == pytest.approx((5.0 * 1.09**6, 0.3), rel=2e-3)


def test_sigma_nn_over_d_is_the_parameter() -> None:
    component = _component("sphere", 0.1)
    assert fm.sigma_nn_over_d(component) == component.value("sigma_nn")
    assert fm.sigma_nn_over_d(_component("sphere", 0.1, structure=False)) is None
