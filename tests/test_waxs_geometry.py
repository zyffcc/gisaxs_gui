import numpy as np

from src.gimap.features.waxs.domain import (
    compute_q_maps,
    cut_image_by_q_range,
    q_range_mask,
)


def _geometry():
    return {
        "incidence": 0.2,
        "center_x": 2.0,
        "center_y": 2.0,
        "distance": 1000.0,
        "pixel_x": 100.0,
        "pixel_y": 100.0,
        "wavelength": 1.0,
        "qr_min": -121.0,
        "qr_max": -121.0,
        "qz_min": -121.0,
        "qz_max": -121.0,
    }


def test_q_maps_preserve_shape_center_and_orientation():
    qr, qz = compute_q_maps((3, 4), _geometry())

    assert qr.shape == (3, 4)
    assert qz.shape == (3, 4)
    assert qr[1, 0] < qr[1, 1] <= qr[1, 2] < qr[1, 3]
    assert qz[0, 1] > qz[1, 1] > qz[2, 1]
    np.testing.assert_allclose(qr[1, 1], 0.0, atol=1e-12)


def test_unset_q_sentinel_keeps_every_pixel():
    assert q_range_mask((3, 4), _geometry()).all()


def test_q_range_cut_masks_pixels_and_keeps_full_q_extent():
    geometry = _geometry()
    qr, _qz = compute_q_maps((3, 4), geometry)
    geometry["qr_min"] = float(qr[0, 2])
    image = np.arange(12, dtype=float).reshape(3, 4)

    cut, extent = cut_image_by_q_range(image, geometry)

    assert np.isnan(cut[:, :2]).all()
    np.testing.assert_array_equal(cut[:, 2:], image[:, 2:])
    assert extent is not None
    assert extent[0] <= extent[1]
    assert extent[2] <= extent[3]
