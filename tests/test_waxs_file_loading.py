from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.waxs.application import LoadWaxsImage, LoadWaxsImageRequest


class _Images:
    def __init__(self, frame_count=3, shape=(2, 4)):
        self.count = frame_count
        self.shape = shape
        self.loaded = None

    def frame_count(self, path):
        return self.count

    def load_frame(self, path, frame_index):
        self.loaded = (path, frame_index)
        return np.full(self.shape, frame_index + 1, dtype=np.float64)


def test_load_waxs_image_clamps_frame_and_preserves_float32_contract(tmp_path):
    repository = _Images(frame_count=3)
    path = tmp_path / "scan.nxs"

    loaded = LoadWaxsImage(repository).execute(LoadWaxsImageRequest(path, 99))

    assert loaded.path == path
    assert loaded.frame_index == 2
    assert loaded.frame_count == 3
    assert loaded.image.dtype == np.float32
    np.testing.assert_array_equal(loaded.image, np.full((2, 4), 3.0))
    assert repository.loaded == (path, 2)


def test_load_waxs_image_rejects_non_matrix_result(tmp_path):
    repository = _Images(shape=(1, 2, 3))

    with pytest.raises(ValueError, match="2D WAXS image"):
        LoadWaxsImage(repository).execute(
            LoadWaxsImageRequest(tmp_path / "bad.nxs")
        )


def test_load_waxs_image_propagates_structured_file_error(tmp_path):
    class Missing:
        def frame_count(self, path):
            raise FileNotFoundError(path)

        def load_frame(self, path, frame_index):
            raise AssertionError

    with pytest.raises(FileNotFoundError):
        LoadWaxsImage(Missing()).execute(
            LoadWaxsImageRequest(Path(tmp_path / "missing.nxs"))
        )
