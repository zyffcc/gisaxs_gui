from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import h5py
import numpy as np

from scipy.stats import qmc


def _largest_2d_hdf5_dataset(handle: h5py.File) -> np.ndarray:
    candidates: List[tuple[int, str]] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
            candidates.append((int(np.prod(obj.shape[-2:])), name))

    handle.visititems(visitor)
    if not candidates:
        raise ValueError("No 2D dataset was found in the HDF5/Nexus file.")
    _, name = max(candidates)
    data = np.asarray(handle[name])
    while data.ndim > 2:
        data = data[0]
    return data


def load_scattering_image(path: str | Path) -> np.ndarray:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        data = np.load(source)
    elif suffix == ".npz":
        archive = np.load(source)
        data = np.asarray(archive[archive.files[0]])
    elif suffix in {".h5", ".hdf5", ".nxs"}:
        with h5py.File(source, "r") as handle:
            data = _largest_2d_hdf5_dataset(handle)
    elif suffix in {".cbf", ".edf"}:
        import fabio

        data = np.asarray(fabio.open(str(source)).data)
    else:
        data = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if data is None:
            raise ValueError(f"Unsupported or unreadable scattering file: {source}")
        if data.ndim == 3:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    data = np.squeeze(np.asarray(data, dtype=np.float32))
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D detector image, got shape {data.shape}.")
    return data


def crop_roi(image: np.ndarray, roi: Dict[str, Any]) -> np.ndarray:
    x, y = int(roi["x"]), int(roi["y"])
    width, height = int(roi["width"]), int(roi["height"])
    return np.asarray(image[y : y + height, x : x + width])


def build_fixed_mask(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    mask_cfg = config.get("mask", {})
    threshold = mask_cfg.get("threshold", {})
    mask = np.zeros(image.shape, dtype=bool)
    if threshold.get("enabled", False):
        low = float(threshold.get("minimum", -np.inf))
        high = float(threshold.get("maximum", np.inf))
        mask |= ~np.isfinite(image) | (image < low) | (image > high)
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    for shape in mask_cfg.get("fixed_shapes", []):
        if shape.get("type") == "rectangle":
            x, y = int(shape.get("x", 0)), int(shape.get("y", 0))
            width, height = int(shape.get("width", 0)), int(shape.get("height", 0))
            mask[max(0, y) : y + max(0, height), max(0, x) : x + max(0, width)] = True
        elif shape.get("type") == "circle":
            cx, cy = float(shape.get("cx", 0)), float(shape.get("cy", 0))
            radius = max(0.0, float(shape.get("radius", 0)))
            mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    return mask


def build_random_mask(shape: tuple[int, int], config: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    height, width = shape
    spec = config.get("mask", {}).get("random", {})
    mask = np.zeros(shape, dtype=bool)
    min_width = max(1, int(spec.get("bar_width_min", 2)))
    max_width = max(min_width, int(spec.get("bar_width_max", 6)))
    for _ in range(max(0, int(spec.get("vertical_bars", 0)))):
        bar_width = int(rng.integers(min_width, max_width + 1))
        x = int(rng.integers(0, max(1, width - bar_width + 1)))
        mask[:, x : x + bar_width] = True
    for _ in range(max(0, int(spec.get("horizontal_bars", 0)))):
        bar_width = int(rng.integers(min_width, max_width + 1))
        y = int(rng.integers(0, max(1, height - bar_width + 1)))
        mask[y : y + bar_width, :] = True
    yy, xx = np.ogrid[:height, :width]
    r_min = max(1, int(spec.get("circle_radius_min", 4)))
    r_max = max(r_min, int(spec.get("circle_radius_max", 12)))
    for _ in range(max(0, int(spec.get("circles", 0)))):
        radius = int(rng.integers(r_min, r_max + 1))
        cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    if spec.get("beamstop", True):
        radius = int(rng.integers(r_min, r_max + 1))
        cx, cy = width // 2, height // 2
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        direction = -1 if rng.random() < 0.5 else 1
        x0, x1 = sorted((cx, int(np.clip(cx + direction * width, 0, width))))
        mask[max(0, cy - 2) : min(height, cy + 3), x0:x1] = True
    return mask


def apply_preprocessing(image: np.ndarray, config: Dict[str, Any], mask: Optional[np.ndarray], rng: np.random.Generator) -> List[Dict[str, Any]]:
    current_mask = None if mask is None else np.asarray(mask, dtype=bool).copy()
    stages: List[Dict[str, Any]] = [{"name": "ROI", "image": np.asarray(image, dtype=np.float32), "mask": current_mask}]
    current = np.asarray(image, dtype=np.float32).copy()
    mask_value = float(config.get("mask", {}).get("mask_value", -1.0))
    for step in config.get("preprocessing", {}).get("steps", []):
        if not step.get("enabled", False):
            continue
        plugin = step.get("plugin")
        if plugin == "noise":
            snr = float(rng.uniform(float(step.get("snr_min_db", 80.0)), float(step.get("snr_max_db", 110.0))))
            power = float(np.mean(np.square(np.nan_to_num(current))))
            sigma = np.sqrt(power / max(10 ** (snr / 10.0), 1e-12))
            current = np.maximum(current + rng.normal(0.0, sigma, current.shape), 0.0).astype(np.float32)
        elif plugin == "mask" and current_mask is not None:
            current = current.copy()
            current[current_mask] = mask_value
        elif plugin == "log":
            epsilon = float(step.get("epsilon", 1e-6))
            valid = current != mask_value if current_mask is not None else np.ones(current.shape, dtype=bool)
            transformed = np.full(current.shape, mask_value, dtype=np.float32)
            transformed[valid] = np.log(np.maximum(current[valid], 0.0) + epsilon)
            current = transformed
        elif plugin == "normalize":
            valid = current != mask_value if current_mask is not None else np.ones(current.shape, dtype=bool)
            values = current[valid]
            if values.size:
                mode = step.get("mode", "range")
                low, high = float(values.min()), float(values.max())
                if mode == "upper":
                    current[valid] = values / max(abs(high), 1e-12)
                elif mode == "lower":
                    current[valid] = values - low
                else:
                    out_low, out_high = float(step.get("lower", 0.0)), float(step.get("upper", 1.0))
                    current[valid] = out_low + (values - low) * (out_high - out_low) / max(high - low, 1e-12)
        elif plugin == "random_edge_crop":
            maximum = max(0, int(step.get("maximum_px", 0)))
            if maximum:
                output_height, output_width = current.shape[:2]
                top, bottom, left, right = [int(rng.integers(0, maximum + 1)) for _ in range(4)]
                cropped = current[top : current.shape[0] - bottom or None, left : current.shape[1] - right or None]
                if cropped.size:
                    current = cv2.resize(cropped, (output_width, output_height), interpolation=cv2.INTER_AREA).astype(np.float32)
                    if current_mask is not None:
                        cropped_mask = current_mask[top : current_mask.shape[0] - bottom or None, left : current_mask.shape[1] - right or None]
                        current_mask = cv2.resize(
                            cropped_mask.astype(np.uint8),
                            (output_width, output_height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                        current[current_mask] = mask_value
        stages.append({"name": str(plugin).replace("_", " ").title(), "image": current.copy(), "mask": current_mask})
    return stages


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

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rng = np.random.default_rng(int(config.get("project", {}).get("seed", 42)))

    @property
    def bornagain_available(self) -> bool:
        try:
            import bornagain  # type: ignore  # noqa: F401

            return True
        except Exception:
            return False

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

    def generate(self, n_samples: int, mode: str = "preview") -> Any:
        if mode == "preview":
            return self.preview_reference()
        if not self.bornagain_available:
            raise RuntimeError("BornAgain is required for simulated Dry run or Full run. Install it locally or use the Maxwell backend.")
        from .simulation import simulate_pattern

        samples = self.sample_parameters(n_samples)
        images: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        for sampled in samples:
            raw = simulate_pattern(self.config, sampled)
            mask = (
                build_random_mask(raw.shape, self.config, self.rng)
                if self.config.get("mask", {}).get("mode") == "random"
                else build_fixed_mask(raw, self.config)
            )
            stages = apply_preprocessing(raw, self.config, mask, self.rng)
            images.append(np.asarray(stages[-1]["image"], dtype=np.float32))
            masks.append(mask)
        return {
            "images": np.stack(images),
            "labels": samples,
            "masks": np.stack(masks),
            "mode": mode,
        }

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
                    values[name] = float(np.exp(np.log(low) + row[index] * (np.log(high) - np.log(low))))
                else:
                    values[name] = float(low + row[index] * (high - low))
            if "radius_nm" in values and "height_nm" in values:
                values["height_nm"] = min(values["height_nm"], max(1e-6, 2.0 * values["radius_nm"] - 1e-6))
            output.append(values)
        return output

    def write_hdf5_shards(self, output_dir: str | Path, n_samples: int) -> List[Path]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        shard_size = int(self.config.get("dataset", {}).get("samples_per_shard", 2000))
        written: List[Path] = []
        generated = 0
        shard_index = 0
        while generated < n_samples:
            count = min(shard_size, n_samples - generated)
            batch = self.generate(count, mode="full")
            path = destination / f"dataset_{shard_index:04d}.h5"
            label_names = list(batch["labels"][0]) if batch["labels"] else []
            labels = np.asarray([[row[name] for name in label_names] for row in batch["labels"]], dtype=np.float32)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("images", data=batch["images"], compression="gzip", chunks=True)
                handle.create_dataset("labels", data=labels, compression="gzip")
                handle.create_dataset("masks", data=batch["masks"], compression="gzip", chunks=True)
                handle.attrs["label_names"] = np.asarray(label_names, dtype="S")
            written.append(path)
            generated += count
            shard_index += 1
        return written
