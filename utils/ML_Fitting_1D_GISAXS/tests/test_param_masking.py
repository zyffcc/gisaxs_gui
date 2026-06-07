from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from TrainSetBuild import schema
from TrainSetBuild.physics_adapter import component_array_to_dict, evaluate_clean, global_array_to_dict
from TrainSetBuild.tfrecord_io import parse_example


def log_rmse(a, b):
    eps = 1e-30
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((np.log(np.maximum(a, eps)) - np.log(np.maximum(b, eps))) ** 2)))


class ParamMaskingTest(unittest.TestCase):
    def test_denormalize_params_with_mask_zeroes_inactive_values(self):
        for type_id in (schema.TYPE_SPHERE, schema.TYPE_CYLINDER, schema.TYPE_VERTICAL_CYLINDER):
            params_norm = np.linspace(0.15, 0.95, schema.P_MAX, dtype=np.float32)
            mask = schema.type_param_mask(type_id)
            mask[mask > 0.5] = 1.0
            mask[0] = 0.0
            masked = schema.denormalize_params_with_mask(params_norm, type_id, mask)
            self.assertTrue(np.all(masked[mask <= 0.5] == 0.0), (type_id, mask, masked))

    def test_dataset_norm_denorm_masked_oracle_reproduces_i_clean(self):
        train_dir = Path("/data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1/train")
        shards = sorted(train_dir.glob("*.tfrecord"))
        if not shards:
            self.skipTest(f"No TFRecord shards found in {train_dir}")

        sample = None
        ds = tf.data.TFRecordDataset([str(s) for s in shards]).map(parse_example)
        for candidate in ds.take(512):
            candidate_np = {k: v.numpy() for k, v in candidate.items()}
            if int(np.sum(candidate_np["slot_exist"] > 0.5)) == 1:
                sample = candidate_np
                break
        if sample is None:
            self.skipTest("No K=1 sample found in first 512 records")

        point_mask = np.asarray(sample["point_mask"]).astype(bool)
        q = np.asarray(sample["q"], dtype=np.float64)[point_mask]
        i_clean = np.asarray(sample["I_clean"], dtype=np.float64)[point_mask]

        components = []
        for slot in np.where(sample["slot_exist"] > 0.5)[0]:
            type_id = int(sample["slot_type"][slot])
            mask = np.asarray(sample["slot_param_mask"][slot], dtype=np.float32)
            if "slot_params_phys" in sample:
                mask = mask * schema.effective_param_mask(type_id, sample["slot_params_phys"][slot])
            params_phys = schema.denormalize_params_with_mask(sample["slot_params_norm"][slot], type_id, mask)
            components.append(component_array_to_dict(type_id, params_phys, float(sample["slot_weight"][slot])))

        global_phys = schema.denormalize_global_with_optional_zero(sample["global_params_norm"])
        oracle = evaluate_clean(q, components, global_array_to_dict(global_phys))
        self.assertLess(log_rmse(i_clean, oracle), 1e-5)


if __name__ == "__main__":
    unittest.main()
