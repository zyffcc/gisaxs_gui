"""Fitting's radii by number and σ_nn/D, for comparison with fit1d and with microscopy.

Spheres and random cylinders are normalised to P(0) = 1, so their Gaussian of R is one of scattered intensity: by
number it is G(R)/R⁶ (R⁴ for cylinders). Vertical cylinders keep R⁴ in the form factor: their Gaussian counts
particles already. σ_nn/D is √(2π)·σ_D (Fitting's paracrystal against Hosemann's)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from src.gimap.features.fitting.application.single_fit import derived_values
from src.gimap.features.fitting.domain import fit_model as fm


def _component(family: str, spread: float, structure: bool = True):
    component = fm.new_component(family, radius=5.0, distance=20.0, structure=structure)
    return replace(component, params={**component.params, "sigma_R": replace(component.params["sigma_R"], value=spread)})


def test_a_narrow_sphere_by_number_is_the_moment_of_g_over_r6() -> None:
    component = _component("sphere", 0.1)
    x = np.linspace(5 - 4 * 0.5, 5 + 4 * 0.5, 25)
    w = np.exp(-0.5 * ((x - 5) / 0.5) ** 2) / x**6
    w /= w.sum()
    mean = float(np.sum(w * x))
    assert fm.number_size(component) == pytest.approx((mean, math.sqrt(np.sum(w * (x - mean) ** 2)) / mean))
    assert fm.number_size(component)[0] < 5.0  # fewer large particles than their intensity suggests


def test_broad_intensity_distributions_have_no_number_mean_and_vertical_cylinders_always_have_one() -> None:
    assert fm.number_size(_component("sphere", 0.2)) is None  # G/R⁶ rises towards small R above σ/R ≈ 0.2
    assert fm.number_size(_component("cylinder", 0.3)) is None
    assert fm.number_size(_component("vertical_cylinder", 0.3)) == pytest.approx((5.0, 0.297), abs=2e-3)
    record = derived_values(_component("sphere", 0.3))
    assert record["R_number_mean_nm"] is None and record["sigma_R_number_rel"] is None


def test_sigma_nn_over_d_is_the_hosemann_spread_of_fittings_paracrystal() -> None:
    component = _component("sphere", 0.1)
    sigma_d = component.value("sigma_D")
    assert fm.sigma_nn_over_d(component) == pytest.approx(math.sqrt(2 * math.pi) * sigma_d)
    q, distance = np.linspace(0.01, 2.0, 50), component.value("D")
    w = fm.sigma_nn_over_d(component) * distance
    np.testing.assert_allclose(np.exp(-math.pi * q**2 * (sigma_d * distance) ** 2), np.exp(-0.5 * q**2 * w**2))
    assert fm.sigma_nn_over_d(_component("sphere", 0.1, structure=False)) is None
