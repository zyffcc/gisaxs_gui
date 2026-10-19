"""SEM GUI contracts: explicit size meaning, reviewable failures and cancellation."""

from __future__ import annotations

import threading

import numpy as np
from PIL import Image

from src.gimap.features.sem.bootstrap import create_sem_window


def test_csv_ui_requires_column_and_preserves_explicit_quantity(qt_application, tmp_path):
    source = tmp_path / "synthetic_sizes.csv"
    source.write_text("size,other\n4,100\n6,200\n8,300\n", encoding="utf-8")
    window = create_sem_window()
    try:
        window.set_prediction([1, 2, 3, 4, 5], [0.1, 0.2, 0.4, 0.2, 0.1], "Synthetic")
        window._open_source(source, "csv")
        assert window._runner.wait(5)
        assert not window.run_button.isEnabled()
        window.csv_column_combo.setCurrentText("size")
        assert window.run_button.isEnabled()
        assert window.radius_scale_spin.value() == 1.0
        assert window.display_mode_combo.currentIndex() == 0
        window.run_button.click()
        assert window._runner.wait(5)
        state = window.view_model.state
        np.testing.assert_array_equal(state.measurements.radius_nm, [2, 3, 4])
        assert state.comparison.metadata["prediction_included"]
        assert state.comparison.metadata["display_mode"] == "probability"
        assert state.comparison.metadata["sem_radius_scale"] == 1.0
        assert window.export_button.isEnabled()

        window.set_prediction(None)
        assert window._runner.wait(5)
        np.testing.assert_array_equal(state.measurements.radius_nm, [2, 3, 4])
        assert state.comparison.metadata["prediction_included"] is False
        assert window.export_button.isEnabled()
    finally:
        window.dispose()
        window.close()


def test_zero_accepted_particles_still_shows_reviewable_labels(qt_application, tmp_path):
    source = tmp_path / "synthetic_sem.png"
    image = np.zeros((40, 40), dtype=np.uint8)
    image[10:14, 10:14] = 200
    Image.fromarray(image).save(source)
    window = create_sem_window()
    try:
        window._open_source(source, "image")
        assert window._runner.wait(5)
        window.nm_per_pixel_spin.setValue(0.5)
        window.min_area_spin.setValue(1000)
        window.run_button.click()
        assert window._runner.wait(5)
        state = window.view_model.state
        assert state.measurements is not None
        assert not len(state.measurements.radius_nm)
        assert state.measurements.labels is not None
        assert any(not particle.accepted for particle in state.measurements.particles)
        assert window.particles_table.rowCount() > 0
        assert state.comparison is None
        assert window.export_button.isEnabled()
    finally:
        window.dispose()
        window.close()


def test_cancelled_extraction_cannot_restore_old_measurements(qt_application, tmp_path, monkeypatch):
    source = tmp_path / "synthetic_sem.png"
    image = np.zeros((40, 40), dtype=np.uint8)
    image[10:14, 10:14] = 200
    Image.fromarray(image).save(source)
    window = create_sem_window()
    started, release = threading.Event(), threading.Event()
    try:
        window._open_source(source, "image")
        assert window._runner.wait(5)
        window.nm_per_pixel_spin.setValue(0.5)
        extract = window.view_model.service.extract

        def slow_extract(*args):
            started.set()
            assert release.wait(5)
            return extract(*args)

        monkeypatch.setattr(window.view_model.service, "extract", slow_extract)
        window.run_button.click()
        assert started.wait(5)
        window._cancel_current()
        release.set()
        assert window._runner.wait(5)
        assert window.view_model.state.measurements is None
        assert window.view_model.state.comparison is None
        assert not window.export_button.isEnabled()
        assert window.run_button.isEnabled()
    finally:
        release.set()
        window._runner.wait(5)
        window.dispose()
        window.close()


def test_new_image_requires_its_own_explicit_calibration(qt_application, tmp_path):
    first, second = tmp_path / "first_sem.png", tmp_path / "second_sem.png"
    image = np.zeros((40, 40), dtype=np.uint8)
    image[10:14, 10:14] = 200
    for path in (first, second):
        Image.fromarray(image).save(path)
    window = create_sem_window()
    try:
        window._open_source(first, "image")
        assert window._runner.wait(5)
        window.nm_per_pixel_spin.setValue(0.5)
        window.scale_bar_nm_spin.setValue(100)
        window.scale_bar_pixels_spin.setValue(200)
        window.run_button.click()
        assert window._runner.wait(5)
        previous_radii = window.view_model.state.measurements.radius_nm.copy()

        window._open_source(second, "image")
        assert window._runner.wait(5)
        assert window.nm_per_pixel_spin.value() == 0
        assert window.scale_bar_nm_spin.value() == 0
        assert window.scale_bar_pixels_spin.value() == 0
        assert window.view_model.state.measurements is None
        assert not window.run_button.isEnabled()

        window.nm_per_pixel_spin.setValue(0.25)
        assert window.run_button.isEnabled()
        window.run_button.click()
        assert window._runner.wait(5)
        np.testing.assert_allclose(window.view_model.state.measurements.radius_nm, previous_radii / 2)
    finally:
        window.dispose()
        window.close()
