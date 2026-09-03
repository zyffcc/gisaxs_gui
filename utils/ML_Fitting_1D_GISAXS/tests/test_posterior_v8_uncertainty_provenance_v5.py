from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.uncertainty_provenance_v5 import (
    UNCERTAINTY_FEATURE_DIM,
    V5UncertaintyProvenance,
    acceptance_sigma_log,
    prepare_encoder_sigma,
)


def test_measured_and_simulated_sigma_remain_acceptance_evidence():
    intensity = np.asarray((10.0, 20.0, 40.0))
    sigma = np.asarray((1.0, 1.0, 2.0))
    for kind in ("measured_sigma", "simulated_sigma"):
        provenance = V5UncertaintyProvenance(kind)
        assert len(provenance.feature_vector) == UNCERTAINTY_FEATURE_DIM
        assert sum(provenance.feature_vector) == 1.0
        np.testing.assert_array_equal(prepare_encoder_sigma(intensity, sigma, provenance), sigma)
        np.testing.assert_allclose(
            acceptance_sigma_log(intensity, sigma, provenance), sigma / intensity
        )
        assert provenance.audit_payload()["compatibility_coverage_claim_allowed"] is True


def test_missing_sigma_proxy_is_encoder_only_and_never_acceptance_evidence():
    intensity = np.asarray((10.0, 20.0, 40.0))
    provenance = V5UncertaintyProvenance(
        "encoder_proxy_missing_sigma", encoder_relative_sigma_proxy=0.015
    )

    np.testing.assert_allclose(
        prepare_encoder_sigma(intensity, None, provenance), 0.015 * intensity
    )
    assert acceptance_sigma_log(intensity, None, provenance) is None
    assert provenance.feature_vector == (0.0, 0.0, 1.0)
    assert provenance.audit_payload()["compatibility_coverage_claim_allowed"] is False


def test_uncertainty_provenance_fails_closed_on_crossed_sources():
    intensity = np.asarray((1.0, 2.0))
    with pytest.raises(ValueError, match="requires measurement_sigma"):
        prepare_encoder_sigma(intensity, None, V5UncertaintyProvenance("measured_sigma"))
    with pytest.raises(ValueError, match="must not receive"):
        prepare_encoder_sigma(
            intensity,
            np.asarray((0.1, 0.1)),
            V5UncertaintyProvenance(
                "encoder_proxy_missing_sigma", encoder_relative_sigma_proxy=0.02
            ),
        )
    valid = V5UncertaintyProvenance("simulated_sigma")
    with pytest.raises(ValueError, match="must not define a proxy"):
        replace(valid, encoder_relative_sigma_proxy=0.1)
