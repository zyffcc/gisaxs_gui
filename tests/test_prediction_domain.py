import numpy as np

from src.gimap.features.prediction.domain import (
    build_complete_batches,
    coerce_array_to_shape,
    extract_cbf_index,
    normalize_input_rank,
    normalize_parameter_prediction,
    normalize_prediction_output,
    parse_index_range,
)


def test_cbf_index_and_range_rules_preserve_legacy_behavior():
    assert extract_cbf_index("sample_00042.cbf") == 42
    assert extract_cbf_index("sample42.cbf") == 42
    assert extract_cbf_index("sample.cbf") == 1
    assert extract_cbf_index("sample.tif") is None
    assert parse_index_range("7") == [7]
    assert parse_index_range("5 - 3") == [3, 4, 5]
    assert parse_index_range("bad") == []


def test_complete_batches_keep_order_and_drop_incomplete_tail():
    paths = ["one.cbf", "two.cbf", "three.cbf", "four.cbf", "five.cbf"]

    assert build_complete_batches(paths, 1) == tuple((path,) for path in paths)
    assert build_complete_batches(paths, 2) == (
        ("one.cbf", "two.cbf"),
        ("three.cbf", "four.cbf"),
    )


def test_input_rank_and_channel_coercion_are_float32_and_nhwc():
    image = np.arange(12, dtype=np.float64).reshape(3, 4)

    ranked = normalize_input_rank(image)
    two_channel = coerce_array_to_shape(ranked, (1, 3, 4, 2))

    assert ranked.shape == (1, 3, 4, 1)
    assert ranked.dtype == np.float32
    assert two_channel.shape == (1, 3, 4, 2)
    np.testing.assert_array_equal(two_channel[..., 0], two_channel[..., 1])


def test_shape_coercion_uses_injected_resize_without_owning_runtime():
    calls = []

    def resize(array, height, width):
        calls.append((array.shape, height, width))
        return np.full((array.shape[0], height, width, array.shape[-1]), 7, np.float32)

    result = coerce_array_to_shape(np.ones((2, 3)), (1, 4, 5, 1), resize)

    assert calls == [((1, 2, 3, 1), 4, 5)]
    assert result.shape == (1, 4, 5, 1)
    assert np.all(result == 7)


def test_sf_parameter_inverse_scaling_preserves_names_and_values():
    result = normalize_parameter_prediction(
        np.array([[0.0, 0.5, 1.0, 0.25]], dtype=np.float32),
        {
            "output_type": "sf_4_parameters",
            "parameter_names": ["t_Cu", "t_polymer", "D", "sigma"],
            "target_min": [0.0, 10.0, 4.0, 0.2],
            "target_max": [25.0, 50.0, 20.0, 4.0],
        },
    )

    assert result["parameter_names"] == ["t_Cu", "t_polymer", "D", "sigma"]
    np.testing.assert_allclose(result["parameters"], [0.0, 30.0, 20.0, 1.15])


def test_distribution_output_orientation_matches_existing_prediction():
    image = np.arange(6, dtype=np.float32).reshape(1, 2, 3, 1)

    result = normalize_prediction_output(image, {})

    np.testing.assert_array_equal(result["hr"], [[0, 1, 2], [3, 4, 5]])
    np.testing.assert_array_equal(result["h"], [3, 5, 7])
    np.testing.assert_array_equal(result["r"], [3, 12])


def test_scalar_and_dict_outputs_keep_legacy_shape_contracts():
    scalar = normalize_prediction_output(np.array([[0.2, 0.8]]), {})
    mapped = normalize_prediction_output({"output": np.ones((1, 2, 2, 1))}, {})

    np.testing.assert_array_equal(scalar["scalars"], [0.2, 0.8])
    assert mapped["hr"].shape == (2, 2)
