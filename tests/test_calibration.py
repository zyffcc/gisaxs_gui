from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from calibration.application import apply_calibration_result
from calibration.candidate_ranker import rank_candidates
from calibration.center_estimator import estimate_center_candidates
from calibration.engine import CalibrationEngine
from calibration.geometry_model import energy_to_wavelength, q_to_ring_radius_px
from calibration.image_loader import load_detector_image
from calibration.models import CalibrationCandidate, CalibrationResult, DetectorImage
from calibration.peak_detector import DetectedPeak
from calibration.peak_matcher import generate_distance_candidates
from calibration.preprocessing import preprocess_detector_image
from calibration.serialization import load_calibration, save_calibration
from calibration.standards import STANDARDS
from core.global_params import global_params
from ui.waxs_page import load_image_matrix


def synthetic_rings(
    shape=(512, 512), center=(247.0, 271.0), radii=(90.0, 180.0), partial=False
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float32)
    radius = np.hypot(xx - center[0], yy - center[1])
    data = np.zeros(shape, dtype=np.float32)
    for ring in radii:
        data += 300.0 * np.exp(-0.5 * ((radius - ring) / 1.5) ** 2)
    rng = np.random.default_rng(8)
    data += rng.poisson(2.0, shape).astype(np.float32)
    if partial:
        data[(xx < center[0]) & (yy < center[1])] = np.nan
        data[240:275, :] = np.nan
    return data


class CalibrationTests(unittest.TestCase):
    def test_energy_to_wavelength(self):
        self.assertAlmostEqual(energy_to_wavelength(12.398419843320026), 1.0, places=12)

    def test_q_to_ring_radius(self):
        radius = q_to_ring_radius_px(0.1, 12.398419843320026, 1000.0, 100e-6)
        expected = 1.0 * np.tan(2 * np.arcsin(0.1 / (4 * np.pi))) / 100e-6
        self.assertAlmostEqual(float(radius), float(expected), places=8)

    def test_nxs_shared_loader_preserves_giwaxs_orientation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "simple.nxs"
            source = np.arange(24, dtype=np.float32).reshape(1, 4, 6)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("/entry/instrument/detector/data", data=source)
                handle.create_dataset("/entry/instrument/detector/pixel_mask", data=np.zeros_like(source, dtype=np.uint8))
                handle.create_dataset("/entry/instrument/detector/x_pixel_size", data=[75e-6])
                handle.create_dataset("/entry/instrument/detector/y_pixel_size", data=[75e-6])
            detector = load_detector_image(path)
            legacy_api = load_image_matrix(str(path))
            expected = np.flipud(source[0].T)
            np.testing.assert_array_equal(detector.data, expected)
            np.testing.assert_array_equal(legacy_api, expected)
            self.assertAlmostEqual(detector.pixel_size_x_m, 75e-6)

    def test_cbf_loader_uses_fabio_without_reorientation(self):
        try:
            from fabio.cbfimage import CbfImage
        except ImportError:
            self.skipTest("fabio CBF writer is unavailable")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frame.cbf"
            source = np.arange(80, dtype=np.int32).reshape(8, 10)
            CbfImage(data=source).write(str(path))
            detector = load_detector_image(path)
            np.testing.assert_array_equal(detector.data, source.astype(np.float32))

    def test_cbf_loader_recovers_energy_from_companion_nxs(self):
        try:
            from fabio.cbfimage import CbfImage
        except ImportError:
            self.skipTest("fabio CBF writer is unavailable")
        with tempfile.TemporaryDirectory() as folder:
            scan_root = Path(folder) / "scan_agbh_00001"
            cbf_folder = scan_root / "embl_2m"
            nxs_folder = scan_root / "lmbd01"
            cbf_folder.mkdir(parents=True)
            nxs_folder.mkdir()
            cbf_path = cbf_folder / "scan_agbh_00001_00001.cbf"
            nxs_path = nxs_folder / "scan_agbh_00001_00001.nxs"
            CbfImage(data=np.ones((40, 48), dtype=np.int32)).write(str(cbf_path))
            with h5py.File(nxs_path, "w") as handle:
                handle.create_dataset("/entry/instrument/detector/collection/beam_energy", data=[12000.0])
            detector = load_detector_image(cbf_path)
            self.assertAlmostEqual(detector.energy_kev, 12.0)
            self.assertIn("companion NXS", detector.metadata["energy_source"])

    def test_center_recovery_with_complete_rings(self):
        center = (247.0, 271.0)
        data = synthetic_rings(center=center)
        image = DetectorImage(data, ~np.isfinite(data), Path("synthetic.cbf"))
        proposals = estimate_center_candidates(preprocess_detector_image(image), max_preview_pixels=300_000)
        best = proposals[0]
        self.assertLess(np.hypot(best.x_px - center[0], best.y_px - center[1]), 8.0)

    def test_center_recovery_with_mask_and_incomplete_rings(self):
        center = (247.0, 271.0)
        data = synthetic_rings(center=center, partial=True)
        image = DetectorImage(data, ~np.isfinite(data), Path("synthetic.cbf"))
        proposals = estimate_center_candidates(preprocess_detector_image(image), max_preview_pixels=300_000)
        self.assertTrue(any(np.hypot(item.x_px - center[0], item.y_px - center[1]) < 18.0 for item in proposals))

    def test_center_recovery_when_center_is_outside_detector(self):
        center = (-180.0, 850.0)
        data = synthetic_rings(
            shape=(600, 700), center=center, radii=(500.0, 620.0, 760.0, 900.0)
        )
        data[250:265, :] = np.nan
        data[:, 295:310] = np.nan
        image = DetectorImage(data, ~np.isfinite(data), Path("partial_waxs.cbf"))
        proposals = estimate_center_candidates(
            preprocess_detector_image(image), max_preview_pixels=300_000
        )
        self.assertTrue(any(
            item.method == "concentric arc gradients"
            and np.hypot(item.x_px - center[0], item.y_px - center[1]) < 35.0
            for item in proposals
        ))

    def test_peak_to_standard_matching(self):
        standard = STANDARDS["agbh"]
        radii = q_to_ring_radius_px(standard.q_values_inv_angstrom[:4], 12.0, 1200.0, 172e-6)
        peaks = [DetectedPeak(float(radius), 10.0, 5.0, 2.0, 0.8) for radius in radii]
        candidates = generate_distance_candidates(peaks, standard, 12.0, 172e-6, 250, 260, 0.9, (500, 2000))
        self.assertGreaterEqual(candidates[0].matched_ring_count, 4)
        self.assertAlmostEqual(candidates[0].distance_mm, 1200.0, delta=1.0)

    def test_distance_prior_affects_ranking(self):
        first = CalibrationCandidate("agbh", 10, 10, 1000, matched_ring_count=3, rms_residual_px=1, azimuthal_coverage=0.5)
        second = CalibrationCandidate("agbh", 10, 10, 2000, matched_ring_count=3, rms_residual_px=1, azimuthal_coverage=0.5)
        ranked = rank_candidates([first, second], estimated_distance_mm=1900)
        self.assertEqual(ranked[0].distance_mm, 2000)

    def test_serialization_round_trip(self):
        candidate = CalibrationCandidate("agbh", 10, 20, 1000, matched_ring_count=3, rms_residual_px=0.5)
        result = CalibrationResult("x.cbf", 10, 20, "abc", 12.0, energy_to_wavelength(12), "D", 1e-4, 1e-4, candidate, [candidate], datetime.now(timezone.utc).isoformat())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cal.json"
            save_calibration(result, path)
            loaded = load_calibration(path)
            self.assertEqual(loaded.selected_candidate.standard_key, "agbh")
            self.assertEqual(loaded.selected_candidate.matched_ring_count, 3)

    def test_apply_updates_shared_geometry(self):
        candidate = CalibrationCandidate("agbh", 123.5, 234.5, 1456.7, matched_ring_count=3)
        result = CalibrationResult("x.cbf", 10, 20, "abc", 12.0, energy_to_wavelength(12), "D", 172e-6, 172e-6, candidate, [candidate], datetime.now(timezone.utc).isoformat())
        values = apply_calibration_result(result)
        self.assertAlmostEqual(values["distance"], 1456.7)
        self.assertAlmostEqual(global_params.get_parameter("fitting", "detector.beam_center_x"), 123.5)
        self.assertAlmostEqual(global_params.get_parameter("beam", "wavelength"), result.wavelength_angstrom / 10.0)

    def test_full_engine_on_synthetic_standard(self):
        energy, distance, pixel = 12.0, 1000.0, 172e-6
        qs = STANDARDS["agbh"].q_values_inv_angstrom[:3]
        radii = q_to_ring_radius_px(qs, energy, distance, pixel)
        data = synthetic_rings(shape=(700, 700), center=(342.0, 367.0), radii=tuple(float(r) for r in radii), partial=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.npy"
            path.write_bytes(b"synthetic calibration source")
            image = DetectorImage(data, ~np.isfinite(data), path, pixel_size_x_m=pixel, pixel_size_y_m=pixel)
            result = CalibrationEngine().calibrate(image, energy_kev=energy, standard_key="agbh", estimated_distance_mm=950, distance_range_mm=(500, 1600))
            self.assertGreaterEqual(result.selected_candidate.matched_ring_count, 2)
            self.assertAlmostEqual(result.selected_candidate.center_x_px, 342.0, delta=15.0)
            self.assertAlmostEqual(result.selected_candidate.center_y_px, 367.0, delta=15.0)
            self.assertAlmostEqual(result.selected_candidate.distance_mm, distance, delta=80.0)

    def test_unknown_detector_can_be_selected_manually(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication
        from ui.geometry_calibration_dialog import GeometryCalibrationDialog

        app = QApplication.instance() or QApplication([])
        dialog = GeometryCalibrationDialog()
        image = DetectorImage(
            np.ones((128, 160), dtype=np.float32),
            np.zeros((128, 160), dtype=bool),
            Path("unknown_detector.cbf"),
        )
        dialog._image_loaded(image)
        self.assertEqual(dialog.detector_combo.currentIndex(), 0)
        self.assertIn("choose a detector model", dialog.detector_label.text())
        pilatus_index = dialog.detector_combo.findData("Pilatus 2M")
        self.assertGreater(pilatus_index, 0)
        dialog.detector_combo.setCurrentIndex(pilatus_index)
        self.assertAlmostEqual(dialog.pixel_x_spin.value(), 172.0)
        self.assertAlmostEqual(dialog.pixel_y_spin.value(), 172.0)
        self.assertEqual(dialog.detector_label.text(), "Pilatus 2M")
        dialog.close()
        app.processEvents()

    def test_calibration_dialog_supports_low_resolution_and_maximize(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication, QScrollArea
        from ui.geometry_calibration_dialog import GeometryCalibrationDialog

        app = QApplication.instance() or QApplication([])
        dialog = GeometryCalibrationDialog()
        self.assertLessEqual(dialog.minimumWidth(), 900)
        self.assertLessEqual(dialog.minimumHeight(), 560)
        self.assertTrue(dialog.windowFlags() & Qt.WindowMaximizeButtonHint)
        self.assertIsNotNone(dialog.findChild(QScrollArea, "calibrationControlsScroll"))
        self.assertEqual(dialog.calibrate_button.text(), "Auto Calibration")
        self.assertFalse(dialog.manual_panel.isVisible())
        self.assertEqual(dialog.clean_preview_button.text(), "Clean image")
        self.assertEqual(dialog.manual_refine_button.text(), "Manual refine")
        dialog.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
