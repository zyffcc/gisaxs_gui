import numpy as np

from src.gimap.features.fitting.domain.cut_math import (
    extract_pixel_profile,
    extract_q_profile,
    pixel_region_bounds,
    sample_q_mesh_line,
)
from src.gimap.features.fitting.domain.models import CutSelection
from src.gimap.features.fitting.domain.insitu_cut import compute_insitu_cut


def test_pixel_cut_preserves_origin_and_finite_mean_behavior():
    image = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, np.nan, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )
    selection = CutSelection(1.5, 1.5, 2.0, 2.0, "horizontal")

    assert pixel_region_bounds(image.shape, selection) == (0, 2, 1, 3)
    intensity, pixels = extract_pixel_profile(image, selection)

    np.testing.assert_allclose(pixels, [0, 1, 2])
    np.testing.assert_allclose(intensity, [9.0, 12.0, 11.0])


def test_q_cut_matches_direct_legacy_column_means():
    image = np.arange(1.0, 17.0).reshape(4, 4)
    qy = np.tile(np.array([-1.5, -0.5, 0.5, 1.5]), (4, 1))
    qz = np.tile(np.array([0.0, 1.0, 2.0, 3.0])[:, None], (1, 4))
    selection = CutSelection(0.0, 1.5, 2.0, 2.0, "horizontal")

    intensity, q_line, indices = extract_q_profile(image, qy, qz, selection)

    np.testing.assert_allclose(indices, [1, 2])
    np.testing.assert_allclose(q_line, [-0.5, 0.5])
    np.testing.assert_allclose(intensity, [8.0, 9.0])


def test_fractional_pixel_sampling_preserves_distinct_q_values():
    qy = np.tile(np.linspace(-2.0, 2.0, 5), (3, 1))
    actual = sample_q_mesh_line(
        qy,
        [1.25, 1.75],
        orientation="horizontal",
        image_shape=qy.shape,
    )

    np.testing.assert_allclose(actual, [-0.75, -0.25])
    assert actual[0] != actual[1]


def test_insitu_pixel_cut_preserves_legacy_origin_and_interpolation():
    result = compute_insitu_cut(
        {
            "image_data": np.arange(1.0, 17.0).reshape(4, 4),
            "vertical": 2.0,
            "parallel": 2.0,
            "center_x": 1.5,
            "center_y": 1.5,
            "cut_type": "horizontal",
            "show_q_axis": False,
            "n_points": 10,
            "method": "Linear",
        }
    )

    np.testing.assert_allclose(result["x_coords"], np.linspace(0.0, 2.0, 10))
    np.testing.assert_allclose(result["y_intensity"], np.linspace(9.0, 11.0, 10))
    assert result["source"] == "pixel"
    assert result["x_label"] == "Pixel / qy"


def test_insitu_q_cut_preserves_legacy_finite_region_means():
    image = np.arange(1.0, 17.0).reshape(4, 4)
    qy = np.tile(np.array([-1.5, -0.5, 0.5, 1.5]), (4, 1))
    qz = np.tile(np.arange(4.0)[:, None], (1, 4))

    result = compute_insitu_cut(
        {
            "image_data": image,
            "vertical": 2.0,
            "parallel": 2.0,
            "center_x": 0.0,
            "center_y": 1.5,
            "cut_type": "horizontal",
            "show_q_axis": True,
            "qy_mesh": qy,
            "qz_mesh": qz,
            "n_points": 10,
            "method": "Linear",
        }
    )

    np.testing.assert_allclose(result["x_coords"], np.linspace(-0.5, 0.5, 10))
    np.testing.assert_allclose(result["y_intensity"], np.linspace(8.0, 9.0, 10))
    assert result["source"] == "q"
    assert result["x_label"] == r"$q_y$ (nm$^{-1}$)"


def test_insitu_q_cut_uses_selected_signed_qr_coordinate_and_label():
    image = np.arange(1.0, 17.0).reshape(4, 4)
    signed_qr = np.tile(np.array([-2.0, -0.75, 0.8, 2.1]), (4, 1))
    qz = np.tile(np.arange(4.0)[:, None], (1, 4))

    result = compute_insitu_cut(
        {
            "image_data": image,
            "vertical": 2.0,
            "parallel": 2.0,
            "center_x": 0.0,
            "center_y": 1.5,
            "cut_type": "horizontal",
            "show_q_axis": True,
            "horizontal_q_axis": "qr",
            "qy_mesh": signed_qr,
            "qz_mesh": qz,
            "n_points": 10,
            "method": "Linear",
        }
    )

    np.testing.assert_allclose(result["x_coords"], np.linspace(-0.75, 0.8, 10))
    assert result["x_label"] == r"$q_r$ (nm$^{-1}$)"


def test_insitu_cut_rejects_non_detector_arrays():
    with np.testing.assert_raises_regex(RuntimeError, "2D detector image"):
        compute_insitu_cut({"image_data": np.arange(4.0)})
