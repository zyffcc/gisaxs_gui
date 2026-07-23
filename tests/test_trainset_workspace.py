from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from trainset.config import (
    default_project_config,
    synchronize_parameter_specs,
    trainable_parameter_names,
    validate_project_config,
)
from trainset.generator import (
    DatasetGenerator,
    apply_preprocessing,
    build_fixed_mask,
    build_random_mask,
    load_scattering_image,
    merge_threshold_mask,
)
from trainset.geometry import q_vectors, roi_to_spherical_ranges
from trainset.job_package import prepare_job_package


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

    def test_paracrystal_sigma_uses_the_configured_range_without_hidden_clipping(self) -> None:
        config = default_project_config()
        config["sample"]["interference"]["parameters"]["sigma_D_ratio"] = {"minimum": 0.25, "maximum": 0.45}
        config = synchronize_parameter_specs(config)

        samples = DatasetGenerator(config).sample_parameters(64)

        self.assertTrue(all(0.25 <= sample["sigma_D_ratio"] <= 0.45 for sample in samples))
        self.assertTrue(any(sample["sigma_D_ratio"] > 0.2 for sample in samples))


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
        {"plugin": "physical_background", "enabled": True, "target_fraction_min": 0.2, "target_fraction_max": 0.2}
    ]
        image = np.ones((16, 16), dtype=np.float32)

        stages = apply_preprocessing(image, config, None, np.random.default_rng(4))

        self.assertEqual([stage["name"] for stage in stages], ["BornAgain Raw", "Physical Background"])
        self.assertGreater(float(stages[-1]["image"].max()), 1.0)

    def test_gaussian_and_poisson_noise_are_independent(self) -> None:
        image = np.full((128, 128), 10.0, dtype=np.float32)
        config = synchronize_parameter_specs(default_project_config())
        config["preprocessing"]["steps"] = [
            {"plugin": "gaussian_noise", "enabled": False, "snr_min_db": 10.0, "snr_max_db": 10.0},
            {"plugin": "poisson_noise", "enabled": True, "count_scale_min": 1.0, "count_scale_max": 1.0},
        ]
        poisson_only = apply_preprocessing(image, config, None, np.random.default_rng(7))
        self.assertEqual([stage["name"] for stage in poisson_only], ["BornAgain Raw", "Poisson Noise"])

        config["preprocessing"]["steps"] = [
            {"plugin": "gaussian_noise", "enabled": True, "snr_min_db": 10.0, "snr_max_db": 10.0},
            {"plugin": "poisson_noise", "enabled": False, "count_scale_min": 1.0, "count_scale_max": 1.0},
        ]
        gaussian_only = apply_preprocessing(image, config, None, np.random.default_rng(7))
        self.assertEqual([stage["name"] for stage in gaussian_only], ["BornAgain Raw", "Gaussian Noise"])

    def test_poisson_count_scale_controls_relative_noise(self) -> None:
        image = np.full((256, 256), 8.0, dtype=np.float32)
        config = synchronize_parameter_specs(default_project_config())
        config["preprocessing"]["steps"] = [
            {"plugin": "poisson_noise", "enabled": True, "count_scale_min": 1.0, "count_scale_max": 1.0}
        ]
        low_scale = apply_preprocessing(image, config, None, np.random.default_rng(9))[-1]["image"]
        config["preprocessing"]["steps"][0].update({"count_scale_min": 100.0, "count_scale_max": 100.0})
        high_scale = apply_preprocessing(image, config, None, np.random.default_rng(9))[-1]["image"]

        self.assertGreater(float(np.mean((low_scale - image) ** 2)), float(np.mean((high_scale - image) ** 2)))

    def test_threshold_mask_is_combined_with_random_geometry(self) -> None:
        config = default_project_config()
        config["mask"]["mode"] = "random"
        config["mask"]["threshold"] = {"enabled": True, "minimum": 2.0, "maximum": 8.0}
        config["mask"]["random"].update(
            {"vertical_bars": 0, "horizontal_bars": 0, "circles": 0, "beamstop": False}
        )
        image = np.array([[1.0, 3.0], [7.0, 9.0]], dtype=np.float32)

        random_geometry = build_random_mask(image.shape, config, np.random.default_rng(3))
        mask = merge_threshold_mask(image, random_geometry, config)
        config["preprocessing"]["steps"] = [{"plugin": "mask", "enabled": True}]
        stages = apply_preprocessing(image, config, mask, np.random.default_rng(3))

        np.testing.assert_array_equal(mask, np.array([[True, False], [False, True]]))
        self.assertEqual(stages[-1]["name"], "Threshold + Detector Mask")

    def test_reference_threshold_locations_are_applied_to_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            reference_path = Path(folder) / "reference.npy"
            reference = np.full((5, 6), 10.0, dtype=np.float32)
            reference[1, 2] = -1.0
            reference[3, 4] = 1_000_000.0
            np.save(reference_path, reference)
            config = default_project_config()
            config["project"]["reference_file"] = str(reference_path)
            config["roi"] = {"x": 0, "y": 0, "width": 6, "height": 5}
            config["mask"]["threshold"] = {
                "enabled": True,
                "minimum": 0.0,
                "maximum": 1000.0,
                "auto_reference_upper": False,
            }

            simulated = np.full((5, 6), 5.0, dtype=np.float32)
            mask = build_fixed_mask(simulated, config)

            self.assertTrue(mask[1, 2])
            self.assertTrue(mask[3, 4])
            self.assertEqual(int(mask.sum()), 2)

    def test_detector_display_qz_decreases_from_top_to_bottom(self) -> None:
        config = default_project_config()
        config["detector"].update(
            {"pixels_x": 20, "pixels_y": 20, "beam_center_x_px": 10.0, "beam_center_y_px": 10.0}
        )
        config["roi"] = {"x": 2, "y": 3, "width": 16, "height": 14}

        qz = q_vectors(config)["qz"]
        ranges = roi_to_spherical_ranges(config)

        self.assertGreater(float(qz[3, 10]), float(qz[16, 10]))
        self.assertGreater(ranges["alpha_top_deg"], ranges["alpha_bottom_deg"])

    def test_trainset_nxs_reference_uses_shared_gui_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.nxs"
            source = np.arange(40 * 48, dtype=np.float32).reshape(1, 40, 48)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("/entry/instrument/detector/data", data=source)
                handle.create_dataset("/entry/instrument/detector/pixel_mask", data=np.zeros_like(source, dtype=np.uint8))
                handle.create_dataset("/entry/instrument/detector/x_pixel_size", data=[75e-6])
                handle.create_dataset("/entry/instrument/detector/y_pixel_size", data=[75e-6])

            loaded = load_scattering_image(path)

            np.testing.assert_array_equal(loaded, np.flipud(source[0].T))


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

    def test_exported_job_contains_reference_image_loader(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            config = default_project_config()
            config["project"]["name"] = "self_contained_job"

            package = prepare_job_package(
                config,
                Path(folder),
                Path(__file__).resolve().parents[1],
            )

            self.assertTrue((package / "src" / "calibration" / "image_loader.py").is_file())
            self.assertIn("fabio", (package / "environment.yml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
