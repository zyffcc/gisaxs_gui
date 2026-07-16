from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def default_project_config() -> Dict[str, Any]:
    """Return a complete, versioned project configuration."""
    return {
        "schema_version": 1,
        "project": {
            "name": "gisaxs_2d_project",
            "seed": 42,
            "reference_file": "",
            "workspace": "",
        },
        "dataset": {
            "dimension": "2d",
            "number_of_samples": 200000,
            "output_format": "hdf5",
            "samples_per_shard": 2000,
            "sampling": "latin_hypercube",
            "split": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "preview_samples": 16,
            "dry_run_samples": 128,
        },
        "beam": {"wavelength_nm": 0.105, "grazing_angle_deg": 0.4},
        "detector": {
            "preset": "Custom",
            "pixels_x": 1475,
            "pixels_y": 1679,
            "pixel_size_x_mm": 0.172,
            "pixel_size_y_mm": 0.172,
            "distance_mm": 3230.0,
            "beam_center_x_px": 804.0,
            "beam_center_y_px": 305.0,
        },
        "roi": {"x": 600, "y": 180, "width": 256, "height": 256},
        "mask": {
            "mode": "fixed",
            "mask_value": -1.0,
            "threshold": {"enabled": True, "minimum": 0.0, "maximum": 1000000000000.0},
            "fixed_shapes": [],
            "random": {
                "vertical_bars": 2,
                "horizontal_bars": 1,
                "bar_width_min": 2,
                "bar_width_max": 6,
                "circles": 1,
                "circle_radius_min": 4,
                "circle_radius_max": 12,
                "beamstop": True,
            },
        },
        "simulation": {
            "engine": "bornagain_yuxin",
            "image_size": [256, 256],
            "resolution_sigma_phi_deg": 0.01,
            "resolution_sigma_alpha_deg": 0.01,
        },
        "parameters": {
            "radius_nm": {"distribution": "uniform", "minimum": 1.0, "maximum": 15.0},
            "height_nm": {"distribution": "uniform", "minimum": 1.0, "maximum": 20.0},
            "spacing_nm": {"distribution": "uniform", "minimum": 3.0, "maximum": 50.0},
        },
        "sample": {
            "surface_density_per_nm2": 0.01,
            "particles": [{"plugin": "spherical_segment", "material": "Copper", "enabled": True}],
            "layers": [
                {"material": "Copper", "thickness_nm": 20.0, "roughness_nm": 0.0, "enabled": True},
                {"material": "Polymer", "thickness_nm": 50.0, "roughness_nm": 0.0, "enabled": True},
            ],
            "substrate": {"material": "Silicon", "roughness_nm": 0.0},
            "interference": {"plugin": "paracrystal", "enabled": True, "sigma_ratio": 0.1},
        },
        "preprocessing": {
            "steps": [
                {"plugin": "noise", "enabled": True, "snr_min_db": 80.0, "snr_max_db": 110.0},
                {"plugin": "mask", "enabled": True},
                {"plugin": "log", "enabled": True, "epsilon": 1e-6},
                {"plugin": "normalize", "enabled": True, "mode": "range", "lower": 0.0, "upper": 1.0},
                {"plugin": "random_edge_crop", "enabled": False, "maximum_px": 4},
            ]
        },
        "model": {
            "template": "cnn_2d",
            "channels": [32, 64, 128, 256],
            "kernel_size": 3,
            "dropout": 0.5,
            "output_mode": "regression",
        },
        "training": {
            "backend": "local",
            "local_python": "",
            "batch_size": 64,
            "epochs": 100,
            "optimizer": "adam",
            "learning_rate": 0.0001,
            "scheduler": "cosine",
        },
        "hpc": {
            "host": "maxwell.desy.de",
            "user": "",
            "remote_path": "",
            "partition": "allgpu",
            "gpus": 1,
            "cpus": 8,
            "memory": "64G",
            "time": "24:00:00",
            "python_command": "python",
            "job_array": True,
        },
        "runtime": {"last_job_id": "", "last_project_dir": ""},
    }


def merge_config(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_project_config(config: Dict[str, Any], require_reference: bool = False) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    project = config.get("project", {})
    detector = config.get("detector", {})
    roi = config.get("roi", {})
    dataset = config.get("dataset", {})

    if not str(project.get("name", "")).strip():
        errors.append("Project name is required.")
    reference = str(project.get("reference_file", "")).strip()
    if require_reference and not reference:
        errors.append("Load a real scattering file before previewing.")
    elif reference and not Path(reference).exists():
        errors.append(f"Reference file does not exist: {reference}")

    for key in ("pixels_x", "pixels_y"):
        if int(detector.get(key, 0)) <= 0:
            errors.append(f"Detector {key} must be positive.")
    for key in ("pixel_size_x_mm", "pixel_size_y_mm", "distance_mm"):
        if float(detector.get(key, 0.0)) <= 0:
            errors.append(f"Detector {key} must be positive.")

    x, y = int(roi.get("x", -1)), int(roi.get("y", -1))
    width, height = int(roi.get("width", 0)), int(roi.get("height", 0))
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        errors.append("ROI x/y must be non-negative and width/height must be positive.")
    if x + width > int(detector.get("pixels_x", 0)) or y + height > int(detector.get("pixels_y", 0)):
        errors.append("ROI extends beyond the configured detector dimensions.")

    split = dataset.get("split", {})
    split_sum = sum(float(split.get(k, 0.0)) for k in ("train", "validation", "test"))
    if abs(split_sum - 1.0) > 1e-6:
        errors.append("Dataset train/validation/test fractions must sum to 1.0.")
    if int(dataset.get("number_of_samples", 0)) <= 0:
        errors.append("Dataset sample count must be positive.")
    if int(dataset.get("samples_per_shard", 0)) <= 0:
        errors.append("Samples per shard must be positive.")

    parameters = config.get("parameters", {})
    if not parameters:
        errors.append("At least one simulation parameter is required.")
    for name, spec in parameters.items():
        try:
            if float(spec.get("minimum")) >= float(spec.get("maximum")):
                errors.append(f"Parameter {name}: minimum must be smaller than maximum.")
            if (dataset.get("sampling") == "log_uniform" or spec.get("distribution") == "log_uniform") and float(spec.get("minimum")) <= 0:
                errors.append(f"Parameter {name}: log-uniform sampling requires a positive minimum.")
        except Exception:
            errors.append(f"Parameter {name}: invalid numeric range.")

    threshold = config.get("mask", {}).get("threshold", {})
    if threshold.get("enabled") and float(threshold.get("minimum", 0.0)) >= float(threshold.get("maximum", 0.0)):
        errors.append("Mask threshold minimum must be smaller than maximum.")
    if not config.get("model", {}).get("channels"):
        errors.append("The model template requires at least one channel stage.")

    if config.get("simulation", {}).get("engine") == "bornagain_yuxin":
        try:
            import bornagain  # type: ignore  # noqa: F401
        except Exception:
            warnings.append("BornAgain is not installed locally; reference preprocessing preview works, but local simulation/full generation is unavailable.")
    if not str(config.get("hpc", {}).get("user", "")).strip():
        warnings.append("Maxwell user is not configured.")
    if not str(config.get("hpc", {}).get("remote_path", "")).strip():
        warnings.append("Maxwell remote output path is not configured.")
    return not errors, errors, warnings


def save_project_config(config: Dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".json":
        destination.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        destination.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return destination


def load_project_config(path: str | Path) -> Dict[str, Any]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    loaded = json.loads(text) if source.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("Project configuration must contain a mapping at the root.")
    return merge_config(default_project_config(), loaded)
