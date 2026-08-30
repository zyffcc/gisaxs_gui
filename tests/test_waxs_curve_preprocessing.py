import numpy as np
import pytest

from src.gimap.features.waxs.application import (
    PreprocessWaxsFrame,
    WaxsCurve,
    WaxsPreprocessFrameRequest,
)
from src.gimap.features.waxs.domain import (
    aligned_detector_distance,
    locate_reference_peak,
    peak_normalization_factor,
)


def test_reference_peak_uses_strongest_unlogged_value_in_window():
    q = np.array([2.09, 2.12, 2.14, 2.18])
    intensity = np.array([50.0, 4.0, 9.0, 100.0])

    peak_q, peak = locate_reference_peak(q, intensity, 2.132, 0.035)

    assert peak_q == pytest.approx(2.14)
    assert peak == pytest.approx(9.0)


def test_sdd_alignment_and_linear_peak_normalization():
    assert aligned_detector_distance(1000.0, 2.2, 2.0) == pytest.approx(1100.0)
    assert peak_normalization_factor(4.0) == pytest.approx(0.25)
    assert peak_normalization_factor(4.0, 2.0) == pytest.approx(0.5)


def test_peak_preprocessing_rejects_missing_or_nonpositive_reference():
    with pytest.raises(RuntimeError, match="No finite peak data"):
        locate_reference_peak(np.array([1.0]), np.array([2.0]), 2.132, 0.035)
    with pytest.raises(RuntimeError, match="positive"):
        peak_normalization_factor(0.0)


def test_preprocess_frame_normalizes_image_and_curve_to_requested_target():
    class Integrator:
        def execute(self, _request):
            return WaxsCurve(
                np.array([2.10, 2.132, 2.16]),
                np.array([1.0, 4.0, 2.0]),
            )

    request = WaxsPreprocessFrameRequest(
        image=np.full((2, 2), 8.0),
        geometry={"distance": 200.0},
        integration={"mode": "radial", "x_axis": "q"},
        mask_min=0.0,
        mask_max=100.0,
        normalization_enabled=True,
        normalization_target_q=2.132,
        normalization_half_width=0.02,
        normalization_target_intensity=2.0,
    )

    result = PreprocessWaxsFrame(Integrator()).execute(request)

    assert result.normalization_factor == pytest.approx(0.5)
    np.testing.assert_allclose(result.image, np.full((2, 2), 4.0))
    np.testing.assert_allclose(result.curve.intensity, np.array([0.5, 2.0, 1.0]))
