from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.waxs.application import LoadWaxsImage, LoadWaxsImageRequest
from src.gimap.features.waxs.domain import subtract_background


class _Images:
    """Repository whose frames encode the frame value per pixel."""

    def __init__(self, frame_count=3, shape=(2, 4)):
        self.count = frame_count
        self.shape = shape
        self.loaded = []

    def frame_count(self, path):
        return self.count

    def load_frame(self, path, frame_index):
        self.loaded.append((str(path), frame_index))
        return np.full(self.shape, frame_index + 1, dtype=np.float64)


def test_subtract_background_applies_coefficient():
    image = np.full((2, 3), 10.0)
    background = np.full((2, 3), 2.0)
    result = subtract_background(image, background, coefficient=1.5)
    np.testing.assert_allclose(result, 7.0)
    assert result.dtype == np.float32


def test_subtract_background_default_coefficient():
    image = np.full((2, 3), 10.0)
    background = np.full((2, 3), 2.0)
    result = subtract_background(image, background)
    np.testing.assert_allclose(result, 8.0)


def test_subtract_background_propagates_nan():
    image = np.full((2, 3), 10.0)
    image[0, 0] = np.nan
    background = np.full((2, 3), 2.0)
    result = subtract_background(image, background)
    assert np.isnan(result[0, 0])
    np.testing.assert_allclose(result[0, 1:], 8.0)


def test_subtract_background_rejects_shape_mismatch():
    image = np.full((2, 3), 10.0)
    with pytest.raises(ValueError, match="shape does not match"):
        subtract_background(image, np.full((4, 4), 1.0))


def test_load_waxs_image_without_background_is_unchanged(tmp_path):
    repository = _Images()
    path = tmp_path / "scan.nxs"
    loaded = LoadWaxsImage(repository).execute(LoadWaxsImageRequest(path, 1))
    np.testing.assert_array_equal(loaded.image, np.full((2, 4), 2.0))
    assert loaded.image.dtype == np.float32


def test_load_waxs_image_applies_background(tmp_path):
    repository = _Images()
    path = tmp_path / "scan.nxs"
    background_path = tmp_path / "bg.tif"
    loaded = LoadWaxsImage(repository).execute(
        LoadWaxsImageRequest(
            path,
            1,
            background_path=background_path,
            background_coefficient=1.0,
        )
    )
    # frame value 2 minus background frame-0 value 1 => 1
    np.testing.assert_allclose(loaded.image, np.full((2, 4), 1.0))
    assert str(background_path) in [p for p, _ in repository.loaded]


def test_load_waxs_image_applies_background_coefficient(tmp_path):
    repository = _Images()
    path = tmp_path / "scan.nxs"
    background_path = tmp_path / "bg.tif"
    loaded = LoadWaxsImage(repository).execute(
        LoadWaxsImageRequest(
            path,
            1,
            background_path=background_path,
            background_coefficient=0.5,
        )
    )
    # frame value 2 - 0.5 * background frame-0 value 1 => 1.5
    np.testing.assert_allclose(loaded.image, np.full((2, 4), 1.5))


def test_load_waxs_image_applies_background_frame_index(tmp_path):
    repository = _Images()
    path = tmp_path / "scan.nxs"
    background_path = tmp_path / "bg.nxs"
    loaded = LoadWaxsImage(repository).execute(
        LoadWaxsImageRequest(
            path,
            1,
            background_path=background_path,
            background_coefficient=1.0,
            background_frame_index=2,
        )
    )
    # main frame value 2 - background frame-2 value 3 => -1
    np.testing.assert_allclose(loaded.image, np.full((2, 4), -1.0))
    assert (str(background_path), 2) in repository.loaded


def test_load_waxs_image_clamps_background_frame_index(tmp_path):
    repository = _Images(frame_count=3)
    path = tmp_path / "scan.nxs"
    background_path = tmp_path / "bg.nxs"
    loaded = LoadWaxsImage(repository).execute(
        LoadWaxsImageRequest(
            path,
            1,
            background_path=background_path,
            background_coefficient=1.0,
            background_frame_index=999,
        )
    )
    # background frame clamped to 2 (value 3): 2 - 3 => -1
    np.testing.assert_allclose(loaded.image, np.full((2, 4), -1.0))
    assert (str(background_path), 2) in repository.loaded


def test_get_waxs_frame_count(tmp_path):
    from src.gimap.features.waxs.application import GetWaxsFrameCount

    repository = _Images(frame_count=7)
    use_case = GetWaxsFrameCount(repository)
    assert use_case.execute(tmp_path / "multi.nxs") == 7
    assert use_case.execute(tmp_path / "single.tif") == 7
