"""NPZ validation agrees with the loader's optional global-mask default."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Training import data_loader
import tensorflow as tf


class Arrays(dict):
    @property
    def files(self):
        return list(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.mark.parametrize("mask", [None, (5,), (2,)])
def test_optional_mask_shape(mask):
    arrays = Arrays({
        key: np.zeros((1, *shape), dtype=np.float32)
        for key, shape in data_loader.EXPECTED_SAMPLE_SHAPES.items()
        if key != "global_param_mask"
    })
    if mask is not None:
        arrays["global_param_mask"] = np.ones((1, *mask), np.float32)
    with patch.object(data_loader.np, "load", return_value=arrays):
        if mask == (2,):
            with pytest.raises(ValueError, match="global_param_mask"):
                data_loader.validate_shards([Path("sample.npz")])
        else:
            data_loader.validate_shards([Path("sample.npz")])


@pytest.mark.parametrize("resolution", [0.0, 0.2])
def test_generator_matches_tensor_signature(resolution):
    arrays = Arrays({
        key: np.zeros((1, *shape), dtype=np.float32)
        for key, shape in data_loader.EXPECTED_SAMPLE_SHAPES.items()
    })
    arrays["global_param_mask"][:] = 1
    arrays["global_params_norm"][0, 3] = resolution
    with patch.object(data_loader.np, "load", return_value=arrays):
        element = next(data_loader.sample_generator(
            [Path("sample.npz")], shuffle_samples=False, max_samples=1
        ))
    signature = data_loader._signature()
    assert element[1]["resolution_present"] == float(resolution > 0)
    tf.nest.assert_same_structure(signature, element)
    for spec, value in zip(tf.nest.flatten(signature), tf.nest.flatten(element)):
        assert spec.is_compatible_with(tf.convert_to_tensor(value))
