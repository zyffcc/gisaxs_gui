from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from trainset.config import (
    default_project_config,
    synchronize_parameter_specs,
    trainable_parameter_names,
    validate_project_config,
)
from trainset.generator import DatasetGenerator, apply_preprocessing


class TrainsetWorkspaceTests(unittest.TestCase):
    def test_shape_and_interference_drive_flat_parameters(self) -> None:
        config = default_project_config()
        config["sample"]["particles"] = [
        {
            "plugin": "box",
            "material": "Copper",
            "enabled": True,
            "parameters": {
                "length_x_nm": {"minimum": 1.0, "maximum": 4.0},
                "length_y_nm": {"minimum": 2.0, "maximum": 5.0},
                "length_z_nm": {"minimum": 3.0, "maximum": 6.0},
            },
        }
    ]
        config["sample"]["interference"] = {"plugin": "none", "enabled": False, "parameters": {}}

        synchronized = synchronize_parameter_specs(config)

        self.assertEqual(trainable_parameter_names(synchronized), ["length_x_nm", "length_y_nm", "length_z_nm"])
        self.assertNotIn("radius_nm", synchronized["parameters"])
        self.assertNotIn("D_nm", synchronized["parameters"])


    def test_layer_fixed_value_and_range_are_distinguished(self) -> None:
        config = default_project_config()
        config["sample"]["layers"] = [
        {
            "enabled": True,
            "material": "Copper",
            "thickness_nm": {"minimum": 10.0, "maximum": 20.0},
            "roughness_nm": {"minimum": 0.5, "maximum": 0.5},
        }
    ]

        synchronized = synchronize_parameter_specs(config)

        self.assertIn("layer_0_thickness_nm", synchronized["parameters"])
        self.assertNotIn("layer_0_roughness_nm", synchronized["parameters"])


    def test_physical_constraints_are_enforced_per_sample(self) -> None:
        config = synchronize_parameter_specs(default_project_config())
        samples = DatasetGenerator(config).sample_parameters(256)

        self.assertTrue(all(sample["height_nm"] <= 2.0 * sample["radius_nm"] for sample in samples))
        self.assertTrue(all(sample["D_nm"] > 2.0 * sample["radius_nm"] for sample in samples))
        self.assertTrue(all(sample["sigma_D_ratio"] <= 0.2 for sample in samples))


    def test_impossible_constraint_is_reported(self) -> None:
        config = default_project_config()
        particle = config["sample"]["particles"][0]
        particle["parameters"]["radius_nm"] = {"minimum": 1.0, "maximum": 2.0}
        particle["parameters"]["height_nm"] = {"minimum": 5.0, "maximum": 6.0}

        valid, errors, _warnings = validate_project_config(config)

        self.assertFalse(valid)
        self.assertTrue(any("h <= 2R" in error for error in errors))


    def test_physical_background_has_an_explicit_preview_stage(self) -> None:
        config = synchronize_parameter_specs(default_project_config())
        config["roi"] = {"x": 0, "y": 0, "width": 16, "height": 16}
        config["detector"]["pixels_x"] = 16
        config["detector"]["pixels_y"] = 16
        config["preprocessing"]["steps"] = [
        {"plugin": "physical_background", "enabled": True, "fraction_min": 0.2, "fraction_max": 0.2}
    ]
        image = np.ones((16, 16), dtype=np.float32)

        stages = apply_preprocessing(image, config, None, np.random.default_rng(4))

        self.assertEqual([stage["name"] for stage in stages], ["ROI", "Physical Background"])
        self.assertGreater(float(stages[-1]["image"].max()), 1.0)


    def test_reference_demo_exercises_dataset_contract(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "reference.npy"
            np.save(image_path, np.linspace(1.0, 100.0, 32 * 32, dtype=np.float32).reshape(32, 32))
            config = copy.deepcopy(default_project_config())
            config["project"]["reference_file"] = str(image_path)
            config["roi"] = {"x": 0, "y": 0, "width": 32, "height": 32}
            config["detector"]["pixels_x"] = 32
            config["detector"]["pixels_y"] = 32

            batch = DatasetGenerator(config).generate(8, mode="demo")

            self.assertEqual(batch["images"].shape, (8, 32, 32))
            self.assertEqual(batch["masks"].shape, (8, 32, 32))
            self.assertEqual(len(batch["labels"]), 8)
            self.assertEqual(batch["mode"], "demo")


if __name__ == "__main__":
    unittest.main()
