from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from matplotlib.collections import QuadMesh
from PyQt5.QtWidgets import QApplication, QPushButton

from src.gimap.features.fitting.domain import DetectorQGrid
from src.gimap.features.fitting.domain.q_space_geometry import (
    create_detector_from_image_and_params,
)
from src.gimap.features.fitting.presentation import DetectorSetupPanel
from src.gimap.features.fitting.presentation.independent_image_window import (
    IndependentMatplotlibWindow,
)
from src.gimap.features.fitting.presentation.state import DetectorDisplayState


def _app() -> QApplication:
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


class _ViewModel:
    def __init__(self) -> None:
        from src.gimap.features.fitting.application import FittingQSpaceCalculations
        from src.gimap.features.fitting.domain import DetectorSettings
        from src.gimap.features.fitting.infrastructure.adapters import QSpaceGeometryAdapter

        self.settings = DetectorSettings(show_q_axis=True, horizontal_q_axis="qy")
        self.science = SimpleNamespace(
            q_space=FittingQSpaceCalculations(QSpaceGeometryAdapter())
        )
        self.values = {
            ("fitting", "detector.show_q_axis"): True,
            ("fitting", "detector.horizontal_q_axis"): "qy",
            ("fitting", "detector.pixel_size_x"): 172.0,
            ("fitting", "detector.pixel_size_y"): 172.0,
            ("fitting", "detector.beam_center_x"): 2.0,
            ("fitting", "detector.beam_center_y"): 1.5,
            ("fitting", "detector.distance"): 2000.0,
            ("beam", "grazing_angle"): 0.4,
            ("beam", "wavelength"): 0.015,
        }

    def load_detector_settings(self):
        return self.settings

    def save_detector_settings(self, settings) -> None:
        self.settings = settings

    def get_setting(self, section, key, default=None):
        return self.values.get((section, key), default)

    def set_setting(self, section, key, value) -> None:
        self.values[(section, key)] = value

    def save_settings(self) -> None:
        return None


def test_detector_q_grid_switches_coordinate_without_changing_detector_cells() -> None:
    qy = np.array([[-1.0, 1.0], [-0.9, 1.1]])
    qr = np.array([[-1.4, 1.4], [-1.3, 1.5]])
    qz = np.array([[0.8, 0.8], [0.2, 0.2]])
    grid = DetectorQGrid(qy=qy, qz=qz, qr=qr)

    point = grid.nearest_point(-1.35, 0.75, "qr")
    assert (point.row, point.column) == (0, 0)
    assert point.horizontal_q == -1.4

    qy_display, qz_display = grid.display_meshes("qy")
    np.testing.assert_array_equal(qy_display, np.flipud(qy))
    np.testing.assert_array_equal(qz_display, np.flipud(qz))

    qy_region = grid.snap_region(-1.05, -0.85, 0.1, 0.9, "qy")
    qr_region = grid.region_from_pixels(
        qy_region.row_min,
        qy_region.row_max,
        qy_region.column_min,
        qy_region.column_max,
        "qr",
    )
    assert qr_region.column_min == qy_region.column_min
    assert qr_region.column_max == qy_region.column_max
    assert qr_region.horizontal_min == -1.4
    assert qr_region.horizontal_max == -1.3


def test_signed_qr_uses_qx_and_qy_not_out_of_plane_qz() -> None:
    detector = create_detector_from_image_and_params(
        image_shape=(5, 7),
        pixel_size_x=172.0,
        pixel_size_y=172.0,
        beam_center_x=3.0,
        beam_center_y=2.0,
        distance=2000.0,
        theta_in_deg=0.4,
        wavelength=0.015,
    )
    qx, qy, _qz, expected_qr = detector.calculate_q_vectors()
    actual_qy, _actual_qz, actual_qr = detector.get_q_coordinate_meshgrids()

    np.testing.assert_allclose(actual_qy, qy)
    np.testing.assert_allclose(actual_qr, np.copysign(np.hypot(qx, qy), qy))
    np.testing.assert_allclose(actual_qr, expected_qr)


def test_inline_detector_panel_persists_horizontal_q_choice() -> None:
    _app()
    view_model = _ViewModel()
    panel = DetectorSetupPanel(view_model, QPushButton())

    panel.horizontal_q_combo.setCurrentIndex(panel.horizontal_q_combo.findData("qr"))
    panel.apply()

    assert view_model.settings.show_q_axis is True
    assert view_model.settings.horizontal_q_axis == "qr"
    panel.close()


def test_independent_detector_uses_curvilinear_q_mesh_and_nearest_cell() -> None:
    app = _app()
    view_model = _ViewModel()
    window = IndependentMatplotlibWindow(fitting_view_model=view_model)
    state = DetectorDisplayState(show_q_axis=True, horizontal_q_axis="qr")
    window.set_detector_display_state(state)
    image = np.arange(20, dtype=float).reshape(4, 5) + 1.0

    window.update_image(image, use_log=False)

    assert isinstance(window.current_image, QuadMesh)
    assert window.ax.get_xlabel() == r"$q_r$ (nm$^{-1}$)"
    assert window._qr_mesh.shape == image.shape
    point = window._detector_q_grid().nearest_point(
        float(window._qr_mesh[1, 2]),
        float(window._qz_mesh[1, 2]),
        "qr",
    )
    pixels = window._convert_q_to_pixel_coordinates(
        point.horizontal_q,
        point.qz,
        1.0e-12,
        1.0e-12,
    )
    assert pixels["center_x"] == point.column
    assert pixels["center_y"] == image.shape[0] - 1 - point.row

    window.close()
    app.processEvents()


def test_independent_detector_reuses_seeded_q_grid_without_recalculation() -> None:
    app = _app()
    view_model = _ViewModel()
    window = IndependentMatplotlibWindow(fitting_view_model=view_model)
    image = np.arange(480_000, dtype=float).reshape(600, 800) + 1.0
    qy = np.broadcast_to(np.linspace(-1.0, 1.0, 800), image.shape)
    qz = np.broadcast_to(np.linspace(2.0, 0.0, 600)[:, None], image.shape)
    qr = np.copysign(np.hypot(qy, 0.1), qy)
    cache_key = (
        600,
        800,
        172.0,
        172.0,
        2.0,
        1.5,
        2000.0,
        0.4,
        0.015,
    )

    class _UnexpectedQCalculation:
        def create_detector(self, **_kwargs):
            raise AssertionError("seeded q grid must be reused")

    view_model.science.q_space = _UnexpectedQCalculation()
    window.seed_q_grid_cache(cache_key, qy, qz, qr)
    window.update_image(image, use_log=False)

    assert isinstance(window.current_image, QuadMesh)
    assert window._qy_mesh is qy
    assert window._qz_mesh is qz
    assert window._qr_mesh is qr
    assert window.current_image.get_array().size < image.size

    window.close()
    app.processEvents()
