"""Synthetic scientific contracts for SEM extraction and probability comparison."""

import math
from concurrent.futures import CancelledError

import numpy as np
import pytest

from src.gimap.features.sem.application import (
    SemComparisonSettings, SemCsvSettings, SemExtractionSettings, SemImage, SemMeasurements,
    SemParticle, SemPrediction,
)
from src.gimap.features.sem.domain import compare_distributions, extract_particles
from src.gimap.features.sem.domain.segmentation import _split
from src.gimap.features.sem.infrastructure import LocalSemCsvReader, LocalSemExporter


def _measurements(values):
    return SemMeasurements(np.asarray(values), (), {"kind": "synthetic"}, {}, {})


def test_calibrated_area_equivalent_radius_and_pixel_centroid():
    image = np.zeros((30, 40))
    image[10:14, 20:26] = 100
    result = extract_particles(SemImage(image, "synthetic"),
                               SemExtractionSettings(0.5, threshold=50, min_area_px=1))
    particle, = result.particles
    assert particle.area_px == 24
    assert particle.area_nm2 == 6
    assert particle.radius_nm == pytest.approx(math.sqrt(24 / math.pi) * 0.5)
    assert particle.diameter_nm == 2 * particle.radius_nm
    assert particle.centroid_x_px == 23
    assert particle.centroid_y_px == 12
    assert result.settings["radius_definition"] == "sqrt(area_px/pi) * nm_per_pixel"


@pytest.mark.parametrize("calibration", [None, 0, -1, float("nan")])
def test_physical_particle_size_requires_explicit_positive_calibration(calibration):
    with pytest.raises(ValueError, match="explicit calibration"):
        extract_particles(SemImage(np.ones((10, 10)), "synthetic"),
                          SemExtractionSettings(calibration))


def test_dark_particles_otsu_and_roi_exclude_scale_bar_and_border_particle():
    image = np.full((50, 60), 200.0)
    image[10:15, 20:25] = 0
    image[5:8, 5:8] = 0  # touches the selected ROI
    image[40:43, 10:50] = 0  # synthetic scale bar outside ROI
    result = extract_particles(SemImage(image, "synthetic"), SemExtractionSettings(
        1, roi_xyxy=(5, 5, 55, 35), bright_particles=False, min_area_px=1))
    assert result.summary["accepted_count"] == 1
    assert result.summary["rejected_count"] == 1
    assert result.particles[0].rejection_reason == "touches_roi_border"
    assert result.radius_nm[0] == pytest.approx(math.sqrt(25 / math.pi))
    assert not result.labels[40:].any()
    assert result.settings["threshold_method"] == "otsu_256_bins"


def test_split_touching_disks_preserves_area_and_does_not_make_tiny_fragment():
    y, x = np.mgrid[:100, :120]
    foreground = (((x - 45) ** 2 + (y - 50) ** 2 < 20 ** 2)
                  | ((x - 75) ** 2 + (y - 50) ** 2 < 20 ** 2))
    image = SemImage(foreground * 255, "synthetic")
    joined = extract_particles(image, SemExtractionSettings(1))
    separated = extract_particles(image, SemExtractionSettings(
        1, split_touching=True, peak_min_distance_px=10))
    assert joined.summary["accepted_count"] == 1
    assert separated.summary["accepted_count"] == 2
    areas = [particle.area_px for particle in separated.particles]
    assert sum(areas) == np.count_nonzero(foreground)
    assert min(areas) / max(areas) > 0.9
    assert all(particle.circularity > 0.8 for particle in separated.particles)


def test_area_and_circularity_rejections_remain_in_particle_table_and_labels():
    image = np.zeros((60, 60))
    image[5:7, 5:7] = 1
    image[20:22, 5:45] = 1
    image[35:45, 35:45] = 1
    result = extract_particles(SemImage(image, "synthetic"),
                               SemExtractionSettings(1, min_area_px=9, min_circularity=0.5))
    assert result.summary == {"detected_count": 3, "accepted_count": 1, "rejected_count": 2,
                              "invalid_pixel_count": 0, "review_required": True,
                              "method": "threshold_area_equivalent_radius"}
    assert "area_below_minimum" in result.particles[0].rejection_reason
    assert "circularity_below_minimum" in result.particles[1].rejection_reason
    assert set(np.unique(result.labels)) == {0, 1, 2, 3}


def test_comparison_default_same_bins_probability_mass_without_radius_adjustment():
    prediction = SemPrediction(np.arange(0.5, 4), np.array([1, 2, 3, 4]))
    result = compare_distributions(_measurements([0.2, 1.2, 2.2, 3.2]), prediction,
                                   SemComparisonSettings(bins=4, radius_max_nm=4))
    np.testing.assert_allclose(result.sem_probability, [0.25] * 4)
    np.testing.assert_allclose(result.prediction_probability, [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(result.display_sem_values, result.sem_probability)
    np.testing.assert_allclose(result.display_prediction_values, result.prediction_probability)
    assert result.metadata["sem_radius_scale"] == 1
    assert result.metadata["display_model_is_probability"]
    assert result.metadata["display_sem_bins"] == 4


def test_probability_range_exclusions_and_renormalization_are_recorded():
    result = compare_distributions(_measurements([0.5, 1.5, 3.5]),
                                   SemPrediction(np.arange(0.5, 4), np.ones(4)),
                                   SemComparisonSettings(bins=2, radius_max_nm=2))
    assert result.metadata["sem_count_outside_range"] == 1
    assert result.metadata["prediction_probability_outside_range"] == pytest.approx(0.5)
    assert result.sem_probability.sum() == pytest.approx(1)
    assert result.prediction_probability.sum() == pytest.approx(1)


def test_paper_display_is_explicit_and_preserves_unmodified_measurements_and_probability():
    values = np.array([0.5, 1.0, 1.5, 2.0])
    measurement = _measurements(values)
    prediction = SemPrediction(np.arange(0.5, 4), np.array([1., 2., 4., 1.]))
    before = prediction.probability.copy()
    result = compare_distributions(measurement, prediction, SemComparisonSettings(
        bins=30, radius_scale=0.95, display_mode="paper", radius_max_nm=4))
    assert len(result.display_sem_values) == 60
    assert len(result.sem_probability) == 30
    assert result.display_prediction_values.min() == 0
    assert result.display_prediction_values.max() == pytest.approx(result.sem_probability.max())
    assert result.metadata["display_model_minimum_subtracted"]
    assert not result.metadata["display_model_is_probability"]
    np.testing.assert_array_equal(measurement.radius_nm, values)
    np.testing.assert_array_equal(prediction.probability, before)


def test_standalone_comparison_does_not_require_prediction():
    result = compare_distributions(_measurements([1, 2, 3]), None, SemComparisonSettings())
    assert result.prediction_probability is None
    assert result.display_prediction_values is None
    assert result.sem_probability.sum() == pytest.approx(1)
    assert not result.metadata["prediction_included"]


def test_exported_particle_csv_roundtrip_preserves_rejected_particles_and_provenance(tmp_path):
    particles = (SemParticle(1, 2, 4, accepted=True),
                 SemParticle(2, 10, 20, accepted=False, rejection_reason="touches_roi_border"))
    measurements = SemMeasurements(np.array([2.]), particles, {"kind": "synthetic"},
                                   {"nm_per_pixel": 0.5}, {"accepted_count": 1, "rejected_count": 1})
    LocalSemExporter().export(measurements, None, tmp_path)
    reader = LocalSemCsvReader()
    assert "radius_nm" in reader.inspect(tmp_path / "sem_particles.csv")
    restored = reader.read(tmp_path / "sem_particles.csv", SemCsvSettings("radius_nm"))
    np.testing.assert_array_equal(restored.radius_nm, [2])
    assert restored.summary["rejected_count"] == 1
    assert restored.particles[1].rejection_reason == "touches_roi_border"
    assert restored.source["upstream_provenance"]["settings"]["nm_per_pixel"] == 0.5
    assert restored.settings["respected_exported_accepted_flags"]


def test_export_refuses_existing_results_without_overwriting_them(tmp_path):
    path = tmp_path / "sem_distribution.pdf"
    path.write_bytes(b"previous-scientific-result")
    with pytest.raises(FileExistsError, match="new output directory"):
        LocalSemExporter().export(_measurements([1]), None, tmp_path)
    assert path.read_bytes() == b"previous-scientific-result"
    assert set(tmp_path.iterdir()) == {path}


def test_csv_numeric_columns_need_explicit_quantity_unit_mapping(tmp_path):
    path = tmp_path / "diameters.csv"
    path.write_text("size,ignored\n0.004,88\n0.008,77\ninvalid,66\n", encoding="utf-8")
    reader = LocalSemCsvReader()
    result = reader.read(path, SemCsvSettings("size", quantity="diameter", unit="um"))
    np.testing.assert_array_equal(result.radius_nm, [2, 4])
    assert result.summary["rejected_count"] == 1
    assert result.settings["selected_value_to_radius_nm_factor"] == 500


def test_zero_accepted_particles_still_export_auditable_review_table(tmp_path):
    measurements = SemMeasurements(np.array([]), (SemParticle(
        1, 2, 4, accepted=False, rejection_reason="touches_roi_border"),),
        {"kind": "synthetic"}, {"nm_per_pixel": 1}, {"accepted_count": 0})
    paths = LocalSemExporter().export(measurements, None, tmp_path)
    assert {path.suffix for path in paths} == {".csv", ".json"}
    assert "touches_roi_border" in (tmp_path / "sem_particles.csv").read_text()


def test_extraction_honors_cancellation_before_work_without_mutating_input():
    image = SemImage(np.ones((30, 30)), "synthetic")
    with pytest.raises(CancelledError, match="cancelled"):
        extract_particles(image, SemExtractionSettings(1), cancelled=lambda: True)
    np.testing.assert_array_equal(image.image, np.ones((30, 30)))


def test_connected_components_extraction_checks_cancellation_between_particles():
    image = np.zeros((100, 100))
    image[10:20, 10:20] = 1
    image[50:60, 50:60] = 1
    checks = []

    def cancel_during_measurement():
        checks.append(True)
        # Start, threshold, foreground, labeling, first component, perimeter,
        # then cancellation before the next component is measured.
        return len(checks) >= 7

    with pytest.raises(CancelledError):
        extract_particles(SemImage(image, "synthetic"), SemExtractionSettings(1),
                          cancelled=cancel_during_measurement)
    assert len(checks) == 7


def test_watershed_checks_cancellation_during_priority_flood():
    y, x = np.mgrid[:140, :140]
    mask = (x - 70) ** 2 + (y - 70) ** 2 < 50 ** 2
    checks = []

    def cancel_after_first_flood_block():
        checks.append(True)
        return len(checks) >= 5

    with pytest.raises(CancelledError):
        _split(mask, 3, cancelled=cancel_after_first_flood_block)
    assert len(checks) == 5
