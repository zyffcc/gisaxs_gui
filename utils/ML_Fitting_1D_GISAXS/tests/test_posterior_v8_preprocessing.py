from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PosteriorV8.preprocessing import PreprocessingContract, preprocess_curve


def small_contract(**updates) -> PreprocessingContract:
    values = {
        "q_min": 0.1,
        "q_max": 1.0,
        "max_points": 8,
        "min_valid_points": 1,
    }
    values.update(updates)
    return PreprocessingContract(**values)


class PosteriorV8PreprocessingTests(unittest.TestCase):
    def test_filter_sort_preserves_q_intensity_sigma_pairing_and_source_index(self):
        result = preprocess_curve(
            q=[0.5, 0.2, 0.4, 0.3, np.nan, 0.7],
            intensity=[50.0, 20.0, 40.0, 30.0, 60.0, -1.0],
            sigma=[5.0, 2.0, 4.0, 3.0, 6.0, 7.0],
            mask=[True, True, False, True, True, True],
            q_range=(0.15, 0.55),
            contract=small_contract(),
        )

        q, intensity, sigma = result.valid_arrays()
        np.testing.assert_array_equal(q, [0.2, 0.3, 0.5])
        np.testing.assert_array_equal(intensity, [20.0, 30.0, 50.0])
        np.testing.assert_array_equal(sigma, [2.0, 3.0, 5.0])
        np.testing.assert_array_equal(result.source_indices[:3], [1, 3, 0])
        self.assertEqual(result.stats["masked_out_count"], 1)
        self.assertEqual(result.stats["nonfinite_count"], 1)
        self.assertEqual(result.stats["nonpositive_count"], 1)
        self.assertEqual(result.stats["q_unit"], "nm^-1")

    def test_p99_uses_frozen_linear_percentile_and_model_features_are_scale_invariant(self):
        result = preprocess_curve(
            q=[0.1, 0.2, 0.3, 0.4],
            intensity=[1.0, 10.0, 100.0, 1000.0],
            sigma=[0.1, 1.0, 10.0, 100.0],
            contract=small_contract(),
        )

        # Linear p99: 100 + 0.97 * (1000 - 100) = 973.
        self.assertAlmostEqual(result.stats["intensity_reference"], 973.0, places=12)
        # Linear p1: 1 + 0.03 * (10 - 1) = 1.27.  Absolute intensity scale is
        # deliberately kept only in provenance, not in model inputs.
        self.assertAlmostEqual(result.stats["intensity_low_reference"], 1.27, places=12)
        self.assertAlmostEqual(float(result.global_features[3]), np.log(973.0 / 1.27), places=6)
        self.assertAlmostEqual(float(result.global_features[4]), np.log(0.1), places=6)
        np.testing.assert_allclose(
            result.x[:4, 1], np.log([1.0, 10.0, 100.0, 1000.0]) - np.log(973.0)
        )
        np.testing.assert_allclose(
            result.x[:4, 2], np.log([0.1, 1.0, 10.0, 100.0]) - np.log(973.0)
        )

        scaled = preprocess_curve(
            q=[0.1, 0.2, 0.3, 0.4],
            intensity=np.asarray([1.0, 10.0, 100.0, 1000.0]) * 2.5e7,
            sigma=np.asarray([0.1, 1.0, 10.0, 100.0]) * 2.5e7,
            contract=small_contract(),
        )
        np.testing.assert_allclose(scaled.x, result.x, rtol=0.0, atol=2e-6)
        np.testing.assert_allclose(
            scaled.global_features, result.global_features, rtol=0.0, atol=2e-6
        )

    def test_padding_and_global_features_have_fixed_shapes(self):
        result = preprocess_curve(
            q=[0.1, 0.2, 1.0],
            intensity=[3.0, 4.0, 5.0],
            sigma=[0.3, 0.4, 0.5],
            contract=small_contract(max_points=6),
        )

        self.assertEqual(result.x.shape, (6, 3))
        self.assertEqual(result.point_mask.shape, (6,))
        self.assertEqual(result.global_features.shape, (5,))
        np.testing.assert_array_equal(result.point_mask, [True, True, True, False, False, False])
        np.testing.assert_array_equal(result.x[3:], 0.0)
        np.testing.assert_array_equal(result.q[3:], 0.0)
        np.testing.assert_array_equal(result.source_indices[3:], -1)
        np.testing.assert_allclose(result.global_features[:3], [0.0, 1.0, 0.5])

    def test_overlong_input_is_downsampled_deterministically_with_pairing(self):
        q = np.linspace(0.1, 1.0, 10)
        intensity = 1000.0 + np.arange(10)
        sigma = 2000.0 + np.arange(10)
        result = preprocess_curve(
            q,
            intensity,
            sigma,
            contract=small_contract(max_points=4),
        )

        np.testing.assert_array_equal(result.source_indices, [0, 3, 6, 9])
        np.testing.assert_array_equal(result.intensity, intensity[[0, 3, 6, 9]])
        np.testing.assert_array_equal(result.sigma, sigma[[0, 3, 6, 9]])
        self.assertEqual(result.stats["downsampled_count"], 6)

    def test_invalid_inputs_are_rejected(self):
        valid = ([0.1, 0.2], [1.0, 2.0], [0.1, 0.2])
        cases = [
            (([0.1], [1.0, 2.0], [0.1]), {}),
            (([[0.1, 0.2]], valid[1], valid[2]), {}),
            (valid, {"mask": [True]}),
            (valid, {"mask": [0, 2]}),
            (valid, {"q_range": (0.4, 0.2)}),
            (([np.nan, -1.0], [1.0, 2.0], [0.1, 0.2]), {}),
        ]
        for args, kwargs in cases:
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    preprocess_curve(*args, contract=small_contract(), **kwargs)

        for contract_kwargs in (
            {"q_min": 0.0},
            {"q_max": 0.1},
            {"max_points": 2.0},
            {"min_valid_points": True},
            {"intensity_reference_percentile": 101.0},
            {"q_unit": "angstrom^-1"},
        ):
            with self.subTest(contract_kwargs=contract_kwargs):
                with self.assertRaises(ValueError):
                    small_contract(**contract_kwargs)

    def test_same_input_is_bitwise_deterministic(self):
        kwargs = {
            "q": np.linspace(0.1, 1.0, 19)[::-1],
            "intensity": np.linspace(10.0, 100.0, 19)[::-1],
            "sigma": np.linspace(1.0, 2.0, 19)[::-1],
            "contract": small_contract(max_points=7),
        }
        first = preprocess_curve(**kwargs)
        second = preprocess_curve(**kwargs)

        for name in ("x", "point_mask", "global_features", "q", "intensity", "sigma"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        self.assertEqual(first.stats, second.stats)


if __name__ == "__main__":
    unittest.main()
