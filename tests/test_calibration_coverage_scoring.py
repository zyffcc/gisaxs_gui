from contextlib import nullcontext

import numpy as np
import pytest

from src.gimap.features.calibration.application import headless
from src.gimap.features.calibration.domain.models import CalibrationCandidate, CalibrationResult, DetectorImage
from tools.benchmarks import calibration_coverage as coverage


def test_coverage_scores_the_line_checked_geometry_and_its_own_errors(monkeypatch, tmp_path):
    def candidate(x, distance, rotation, uncertainty):
        return CalibrationCandidate(
            "agbh", x, 25., distance, rot1_deg=rotation, rot2_deg=rotation / 2, tilt_refined=True,
            matched_ring_count=4, center_x_sigma_px=uncertainty, center_y_sigma_px=uncertainty,
            distance_sigma_mm=uncertainty, rot1_sigma_deg=uncertainty, rot2_sigma_deg=uncertainty,
            uncertainty_method="ring jackknife")

    first, winner = candidate(11., 1100., 1., 0.4), candidate(15., 1000., 0.1, 0.01)
    result = CalibrationResult("", 0, 0, "", 12., 1., None, 1e-4, 1e-4, first, [first, winner], "")
    image = DetectorImage(np.ones((64, 64)), None, tmp_path / "simulated.npy")
    monkeypatch.setattr(coverage.CalibrationEngine, "calibrate", lambda *args, **kwargs: result)
    monkeypatch.setattr(headless, "check_lines", lambda *a, **kw: {})
    monkeypatch.setattr(headless, "best_by_lines", lambda checks: 1)
    monkeypatch.setattr(coverage, "record_fit_points", lambda: nullcontext({id(winner): "winner points"}))

    def spread(points, fitted, wavelength):
        assert points == "winner points" and fitted is winner
        return np.zeros((4, 5)), np.eye(5)

    monkeypatch.setattr(coverage, "gimap_spread", spread)
    truth = np.array([15.5, 25.5, 1000., 0.13, 0.09])
    fit = coverage.calibrated(image, truth, 1., 1000.)
    assert fit["identified"] and fit["returned_index"] == 1
    assert list(fit["fit"].values()) == [15.5, 25.5, 1000., 0.1, 0.05]
    assert list(fit["sigma"].values()) == [0.01] * 5
    cases = [{"as_shipped": fit, "truth": dict(zip(coverage.GIMAP_KEYS, truth))}]
    row = coverage.coverage_table(cases, "as_shipped", "sigma")["4-6"]
    assert row["cases"] == row["beam_x_px"]["within_1sigma_count"] == 1
    assert row["rot1_deg"]["within_2sigma_count"] == 0
    assert coverage.tilt_accuracy(cases, "as_shipped")["p95_tilt_vector_error_deg"] == pytest.approx(0.05)
