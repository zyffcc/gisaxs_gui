from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from TrainSetBuild import constraints, sampling, schema


class DConstraintTests(unittest.TestCase):
    def test_random_cylinder_uses_circumscribed_diameter(self):
        params = np.array([3.0, 0.2, 4.0, 0.3, 0.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(
            sampling.characteristic_exclusion_size(schema.TYPE_CYLINDER, params),
            np.hypot(6.0, 4.0),
        )

    def test_max_and_mean_spacing_thresholds(self):
        slot_type = np.array(
            [schema.TYPE_SPHERE, schema.TYPE_CYLINDER, schema.TYPE_VERTICAL_CYLINDER, schema.TYPE_EMPTY]
        )
        slot_exist = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        params = np.zeros((schema.MAX_SLOTS, schema.P_MAX), dtype=np.float32)
        params[0, 0] = 2.0
        params[1, [0, 2]] = [3.0, 4.0]
        params[2, 0] = 5.0
        sizes = np.array([4.0, np.hypot(6.0, 4.0), 10.0])

        self.assertAlmostEqual(
            sampling.d_spacing_threshold(slot_type, params, slot_exist, schema.D_RULE_MAX),
            float(np.max(sizes)),
        )
        self.assertAlmostEqual(
            sampling.d_spacing_threshold(slot_type, params, slot_exist, schema.D_RULE_MEAN),
            float(np.mean(sizes)),
        )

    def test_json_constraint_encodes_presence_and_rule(self):
        parsed = constraints.from_json_dict(
            {
                "mode": "free",
                "d_constraint": {
                    "presence": "optional",
                    "slot_presence": {"slot_1": "required", "slot_2": "absent"},
                    "spacing_rule": "max_diameter",
                },
            }
        )
        np.testing.assert_array_equal(parsed["d_allowed"][0], [1.0, 1.0])
        np.testing.assert_array_equal(parsed["d_allowed"][1], [0.0, 1.0])
        np.testing.assert_array_equal(parsed["d_allowed"][2], [1.0, 0.0])
        self.assertEqual(int(np.argmax(parsed["d_spacing_rule"])), schema.D_RULE_MAX)

    def test_generated_present_d_obeys_selected_rule(self):
        rng = np.random.default_rng(1234)
        for rule_id in (schema.D_RULE_MAX, schema.D_RULE_MEAN):
            sample = sampling.generate_sample(
                rng,
                max_points=128,
                k_values=np.array([4], dtype=np.int32),
                k_probs=np.array([1.0]),
                gap_drop_prob=0.0,
                d_absent_probability=0.0,
                d_rule_ids=np.array([rule_id], dtype=np.int32),
                d_rule_probs=np.array([1.0]),
            )
            threshold = sampling.d_spacing_threshold(
                sample["slot_type"],
                sample["slot_params_phys"],
                sample["slot_exist"],
                rule_id,
            )
            active = np.where(sample["slot_exist"] > 0.5)[0]
            self.assertTrue(np.all(sample["slot_params_phys"][active, 4] > threshold))


if __name__ == "__main__":
    unittest.main()
