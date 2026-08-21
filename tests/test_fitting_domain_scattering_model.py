"""Numerical contracts for the fitting scattering-model decomposition."""

from __future__ import annotations

import numpy as np

from src.gimap.features.fitting.domain.scattering_model import (
    make_mixed_model,
    mixed_model_components,
)
from src.gimap.features.fitting.domain.signed_q import prepare_signed_q_curve


def _sphere_parameters() -> list[float]:
    # Int, R, sigma_R, D, sigma_D, BG, sigma_Res, nu_Res, int_Res, k
    return [1800.0, 18.0, 1.5, 42.0, 2.0, 7.0, 0.08, 4.0, 35.0, 1.7]


def test_total_model_is_exact_sum_of_exposed_components() -> None:
    q = np.geomspace(0.005, 5.0, 320)
    parameters = _sphere_parameters()

    evaluated = make_mixed_model(["sphere"])(q, *parameters)
    components = mixed_model_components(["sphere"], q, parameters)
    reconstructed = components["BG_total"] + components["resolution"]
    for particle in components["particles"]:
        reconstructed = reconstructed + particle["I"]

    np.testing.assert_allclose(components["total"], evaluated, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(reconstructed, evaluated, rtol=1e-12, atol=1e-12)


def test_folded_model_keeps_equal_values_for_positive_and_negative_q_branches() -> None:
    q_signed = np.array([-4.0, -2.0, -1.0, 1.0, 2.0, 4.0])
    prepared = prepare_signed_q_curve(
        q_signed,
        np.ones_like(q_signed),
        branch="both",
        combination="fold",
    )
    model = make_mixed_model(["sphere"])
    fitted = model(prepared.q, *_sphere_parameters())

    for coordinate in np.unique(prepared.q):
        branch_values = fitted[prepared.q == coordinate]
        np.testing.assert_allclose(branch_values, branch_values[0], rtol=1e-12, atol=1e-12)
