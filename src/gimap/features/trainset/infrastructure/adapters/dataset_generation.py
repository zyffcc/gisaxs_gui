"""Focused Trainset detector-data and generation behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import h5py
import numpy as np
from scipy.stats import qmc

from ...application.ports import SimulationPort
from .configuration import synchronize_parameter_specs

from .detector_images import crop_roi, load_scattering_image
from .detector_masks import build_fixed_mask, build_random_mask, merge_threshold_mask
from .preprocessing_pipeline import apply_preprocessing


@dataclass
class PreviewResult:
    stages: List[Dict[str, Any]]
    roi_image: np.ndarray
    mask: np.ndarray
    spectrum_x: np.ndarray
    spectrum_y: np.ndarray
    stats: Dict[str, Any]


class DatasetGenerator:
    """Shared generator facade used by Preview, Dry run, local and Slurm backends."""

    def __init__(
        self,
        config: Dict[str, Any],
        simulation_port: SimulationPort | None = None,
    ):
        self.config = synchronize_parameter_specs(config)
        self.rng = np.random.default_rng(int(config.get("project", {}).get("seed", 42)))
        self._grid_cache_data: Optional[Dict[str, Any]] = None
        self.simulation_port = simulation_port

    @property
    def bornagain_available(self) -> bool:
        return bool(self.simulation_port is not None and self.simulation_port.is_available())

    def preview_reference(self, image: Optional[np.ndarray] = None) -> PreviewResult:
        if image is None:
            path = self.config.get("project", {}).get("reference_file")
            if not path:
                raise ValueError("Load a real scattering file first.")
            image = load_scattering_image(path)
        roi_image = crop_roi(image, self.config["roi"])
        if not roi_image.size:
            raise ValueError("The configured ROI is empty for the loaded detector image.")
        if self.config.get("mask", {}).get("mode") == "random":
            mask = build_random_mask(roi_image.shape, self.config, self.rng)
            mask = merge_threshold_mask(roi_image, mask, self.config)
        else:
            mask = build_fixed_mask(roi_image, self.config)
        stages = apply_preprocessing(roi_image, self.config, mask, self.rng)
        final = np.asarray(stages[-1]["image"], dtype=np.float32)
        valid = final[np.isfinite(final)]
        hist, edges = np.histogram(valid, bins=64) if valid.size else (np.zeros(64), np.arange(65))
        stats = {
            "reference_shape": list(image.shape),
            "roi_shape": list(roi_image.shape),
            "tensor_shape": [1, int(final.shape[0]), int(final.shape[1]), 1],
            "masked_fraction": float(mask.mean()),
            "minimum": float(valid.min()) if valid.size else 0.0,
            "maximum": float(valid.max()) if valid.size else 0.0,
            "dynamic_range": float(valid.max() - valid.min()) if valid.size else 0.0,
        }
        return PreviewResult(stages, roi_image, mask, (edges[:-1] + edges[1:]) / 2.0, hist, stats)

    def generate(
        self,
        n_samples: int,
        mode: str = "preview",
        progress: Optional[Callable[[int, int, str], None]] = None,
        pause: Optional[Callable[[], None]] = None,
    ) -> Any:
        if mode == "preview":
            return self.preview_reference()
        if mode == "demo":
            return self._generate_reference_demo(n_samples)
        if not self.bornagain_available:
            raise RuntimeError(
                "BornAgain is required for simulated Dry run or Full run. Install it locally or use the Maxwell backend."
            )
        from ...application.simulation import simulate_pattern

        use_grid = bool(
            mode == "full"
            and self.config.get("simulation", {}).get("grid_cache", {}).get("enabled", True)
        )
        if use_grid:
            from .grid_cache import cache_compatible

            use_grid = cache_compatible(self.config)
        if use_grid and self._grid_cache_data is None:
            from .grid_cache import load_or_build_grid

            self._grid_cache_data = load_or_build_grid(
                self.config,
                simulation_port=self.simulation_port,
                progress=progress,
                pause=pause,
            )

        samples = self.sample_parameters(n_samples)
        images: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        for index, sampled in enumerate(samples, start=1):
            if pause is not None:
                pause()
            simulation_values = self._mixture_values(sampled)
            if self._grid_cache_data is not None:
                from .grid_cache import mix_cached_grid

                raw = mix_cached_grid(
                    self.config,
                    simulation_values,
                    self._grid_cache_data,
                    self.rng,
                )
            else:
                raw = simulate_pattern(
                    self.config,
                    simulation_values,
                    simulator=self.simulation_port,
                )
            mask = (
                build_random_mask(raw.shape, self.config, self.rng)
                if self.config.get("mask", {}).get("mode") == "random"
                else build_fixed_mask(raw, self.config)
            )
            if self.config.get("mask", {}).get("mode") == "random":
                mask = merge_threshold_mask(raw, mask, self.config)
            stages = apply_preprocessing(raw, self.config, mask, self.rng)
            images.append(np.asarray(stages[-1]["image"], dtype=np.float32))
            masks.append(mask)
            if progress is not None:
                progress(index, n_samples, f"Generating dataset sample {index}/{n_samples}")
        return {
            "images": np.stack(images),
            "labels": samples,
            "masks": np.stack(masks),
            "mode": mode,
        }

    def _generate_reference_demo(self, n_samples: int) -> Dict[str, Any]:
        """Exercise the complete local I/O/training path without claiming physics.

        Images are small stochastic variants of the selected real reference;
        labels are sampled from the configured ranges. This is intentionally a
        pipeline smoke test, not a substitute for BornAgain generation.
        """
        path = str(self.config.get("project", {}).get("reference_file", ""))
        if not path:
            raise ValueError("A real reference image is required for the local demo dataset.")
        roi_image = crop_roi(load_scattering_image(path), self.config["roi"])
        if not roi_image.size:
            raise ValueError("The selected ROI is empty in the reference image.")
        samples = self.sample_parameters(n_samples)
        images: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        for _sampled in samples:
            gain = float(self.rng.uniform(0.85, 1.15))
            shifted = np.roll(
                roi_image * gain,
                (int(self.rng.integers(-2, 3)), int(self.rng.integers(-2, 3))),
                axis=(0, 1),
            )
            mask = (
                build_random_mask(shifted.shape, self.config, self.rng)
                if self.config.get("mask", {}).get("mode") == "random"
                else build_fixed_mask(shifted, self.config)
            )
            if self.config.get("mask", {}).get("mode") == "random":
                mask = merge_threshold_mask(shifted, mask, self.config)
            stages = apply_preprocessing(shifted, self.config, mask, self.rng)
            images.append(np.asarray(stages[-1]["image"], dtype=np.float32))
            masks.append(mask)
        return {
            "images": np.stack(images),
            "labels": samples,
            "masks": np.stack(masks),
            "mode": "demo",
        }

    def _mixture_values(self, sampled: Dict[str, float]) -> Dict[str, Any]:
        mixture = self.config.get("sample", {}).get("mixture", {})
        mode = str(mixture.get("mode", "single"))
        components = max(1, int(mixture.get("components", 1)))
        if mode == "single" or components == 1:
            return sampled
        particle = next(iter(self.config.get("sample", {}).get("particles", [])), {})
        particle_keys = list(particle.get("parameters", {}))
        sigma_min = max(0.0, float(mixture.get("sigma_fraction_min", 0.01)))
        sigma_max = max(sigma_min, float(mixture.get("sigma_fraction_max", 0.30)))
        values: List[Dict[str, float]] = []
        for _index in range(components):
            component = dict(sampled)
            for key in particle_keys:
                spec = self.config.get("parameters", {}).get(key, {})
                low, high = (
                    float(spec.get("minimum", sampled.get(key, 0.0))),
                    float(spec.get("maximum", sampled.get(key, 0.0))),
                )
                spread = self.rng.uniform(sigma_min, sigma_max) * max(high - low, 1e-12)
                component[key] = float(
                    np.clip(self.rng.normal(sampled.get(key, low), spread), low, high)
                )
            if "height_nm" in component and "radius_nm" in component:
                component["height_nm"] = min(
                    component["height_nm"], max(1e-6, 2.0 * component["radius_nm"] - 1e-6)
                )
            values.append(component)
        weights = (
            self.rng.dirichlet(np.ones(components)).astype(float)
            if mixture.get("random_weights", True)
            else np.full(components, 1.0 / components)
        )
        enriched: Dict[str, Any] = dict(sampled)
        enriched["__mixture_components"] = values
        enriched["__mixture_weights"] = weights.tolist()
        return enriched

    def sample_parameters(self, n_samples: int) -> List[Dict[str, float]]:
        specs = self.config.get("parameters", {})
        names = list(specs)
        sampling = self.config.get("dataset", {}).get("sampling", "latin_hypercube")
        if sampling == "latin_hypercube":
            unit = qmc.LatinHypercube(d=len(names), seed=self.rng).random(n_samples)
        elif sampling == "grid":
            side = max(2, int(np.ceil(n_samples ** (1.0 / max(len(names), 1)))))
            mesh = np.meshgrid(*([np.linspace(0.0, 1.0, side)] * len(names)), indexing="ij")
            unit = np.stack([axis.ravel() for axis in mesh], axis=1)[:n_samples]
        else:
            unit = self.rng.random((n_samples, len(names)))
        output: List[Dict[str, float]] = []
        for row in unit:
            values: Dict[str, float] = {}
            for index, name in enumerate(names):
                spec = specs[name]
                low, high = float(spec["minimum"]), float(spec["maximum"])
                if sampling == "log_uniform" or spec.get("distribution") == "log_uniform":
                    values[name] = float(
                        np.exp(np.log(low) + row[index] * (np.log(high) - np.log(low)))
                    )
                else:
                    values[name] = float(low + row[index] * (high - low))
            constraints = self.config.get("sample", {}).get("constraints", {})
            if (
                constraints.get("segment_height_le_2r", False)
                and "radius_nm" in values
                and "height_nm" in values
            ):
                height_spec = specs["height_nm"]
                feasible_high = min(float(height_spec["maximum"]), 2.0 * values["radius_nm"] - 1e-6)
                feasible_low = float(height_spec["minimum"])
                if feasible_high < feasible_low:
                    raise ValueError(
                        "Constraint h <= 2R is infeasible for a sampled radius; adjust the configured ranges."
                    )
                values["height_nm"] = float(
                    np.clip(values["height_nm"], feasible_low, feasible_high)
                )
            if (
                constraints.get("interparticle_spacing_gt_2r", False)
                and "radius_nm" in values
                and "D_nm" in values
            ):
                distance_spec = specs["D_nm"]
                feasible_low = max(
                    float(distance_spec["minimum"]), 2.0 * values["radius_nm"] + 1e-6
                )
                feasible_high = float(distance_spec["maximum"])
                if feasible_low > feasible_high:
                    raise ValueError(
                        "Constraint D > 2R is infeasible for a sampled radius; adjust the configured ranges."
                    )
                values["D_nm"] = float(np.clip(values["D_nm"], feasible_low, feasible_high))
            output.append(values)
        return output

    def write_hdf5_shards(
        self,
        output_dir: str | Path,
        n_samples: int,
        mode: str = "full",
        progress: Optional[Callable[[int, int, str], None]] = None,
        pause: Optional[Callable[[], None]] = None,
    ) -> List[Path]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        shard_size = int(self.config.get("dataset", {}).get("samples_per_shard", 2000))
        written: List[Path] = []
        generated = 0
        shard_index = 0
        while generated < n_samples:
            count = min(shard_size, n_samples - generated)

            def batch_progress(completed: int, total: int, message: str) -> None:
                if progress is None:
                    return
                if message.startswith("BornAgain form-factor grid"):
                    progress(completed, total, message)
                else:
                    progress(generated + completed, n_samples, message)

            batch = self.generate(
                count,
                mode=mode,
                progress=batch_progress,
                pause=pause,
            )
            path = destination / f"dataset_{shard_index:04d}.h5"
            label_names = list(batch["labels"][0]) if batch["labels"] else []
            labels = np.asarray(
                [[row[name] for name in label_names] for row in batch["labels"]], dtype=np.float32
            )
            with h5py.File(path, "w") as handle:
                handle.create_dataset(
                    "images", data=batch["images"], compression="gzip", chunks=True
                )
                handle.create_dataset("labels", data=labels, compression="gzip")
                handle.create_dataset("masks", data=batch["masks"], compression="gzip", chunks=True)
                handle.attrs["label_names"] = np.asarray(label_names, dtype="S")
            written.append(path)
            generated += count
            shard_index += 1
        return written
