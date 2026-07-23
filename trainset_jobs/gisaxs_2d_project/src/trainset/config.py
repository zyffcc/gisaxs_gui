from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


PHYSICAL_BACKGROUND_PARAMETERS = (
    {
        "key": "target_fraction",
        "label": "Overall level / signal",
        "minimum": 0.05,
        "maximum": 0.30,
        "decimals": 3,
        "help": "Background 95th percentile relative to the simulated signal 75th percentile.",
    },
    {
        "key": "constant_fraction",
        "label": "Constant floor",
        "minimum": 0.00,
        "maximum": 0.03,
        "decimals": 3,
        "help": "Uniform detector background before the overall relative scaling is applied.",
    },
    {
        "key": "specular_amplitude",
        "label": "Specular ridge amplitude",
        "minimum": 0.20,
        "maximum": 1.00,
        "decimals": 3,
        "help": "Relative strength of the narrow vertical specular ridge around qy = 0.",
    },
    {
        "key": "specular_width_fraction",
        "label": "Specular qy width",
        "minimum": 0.01,
        "maximum": 0.04,
        "decimals": 4,
        "help": "Gaussian ridge width as a fraction of the ROI qy span.",
    },
    {
        "key": "specular_widening",
        "label": "Specular widening",
        "minimum": 0.00,
        "maximum": 0.12,
        "decimals": 3,
        "help": "How much the specular ridge widens toward larger qz.",
    },
    {
        "key": "specular_decay_fraction",
        "label": "Specular qz decay",
        "minimum": 0.20,
        "maximum": 0.80,
        "decimals": 3,
        "help": "Vertical decay length as a fraction of the ROI qz span.",
    },
    {
        "key": "yoneda_amplitude",
        "label": "Yoneda band amplitude",
        "minimum": 0.10,
        "maximum": 0.70,
        "decimals": 3,
        "help": "Relative strength of the horizontal Yoneda-like band.",
    },
    {
        "key": "yoneda_center_fraction",
        "label": "Yoneda qz position",
        "minimum": 0.50,
        "maximum": 0.72,
        "decimals": 3,
        "help": "Vertical band position from ROI bottom (0) to top (1) in qz coordinates.",
    },
    {
        "key": "yoneda_width_fraction",
        "label": "Yoneda qz width",
        "minimum": 0.02,
        "maximum": 0.08,
        "decimals": 3,
        "help": "Gaussian band width as a fraction of the ROI qz span.",
    },
    {
        "key": "yoneda_center_hole",
        "label": "Yoneda center hole",
        "minimum": 0.40,
        "maximum": 0.95,
        "decimals": 3,
        "help": "Suppresses the Yoneda band close to qy = 0 to avoid an unrealistically bright cross.",
    },
    {
        "key": "wedge_amplitude",
        "label": "Diffuse wedge amplitude",
        "minimum": 0.05,
        "maximum": 0.40,
        "decimals": 3,
        "help": "Relative strength of the anisotropic Guinier–Porod diffuse component.",
    },
    {
        "key": "wedge_anisotropy",
        "label": "Diffuse anisotropy",
        "minimum": 0.60,
        "maximum": 2.00,
        "decimals": 3,
        "help": "Aspect ratio of the diffuse wedge in qy versus qz.",
    },
    {
        "key": "wedge_porod_exponent",
        "label": "Porod exponent",
        "minimum": 2.00,
        "maximum": 3.80,
        "decimals": 3,
        "help": "Controls how quickly the diffuse tail decays at high q.",
    },
    {
        "key": "wedge_rg_fraction",
        "label": "Diffuse Rg scale",
        "minimum": 0.05,
        "maximum": 0.25,
        "decimals": 3,
        "help": "Guinier knee position expressed as a fraction of the normalized ROI q span.",
    },
    {
        "key": "plane_qy_slope",
        "label": "Plane slope along qy",
        "minimum": -0.08,
        "maximum": 0.08,
        "decimals": 3,
        "help": "Slow left-to-right detector background gradient.",
    },
    {
        "key": "plane_qz_slope",
        "label": "Plane slope along qz",
        "minimum": -0.08,
        "maximum": 0.08,
        "decimals": 3,
        "help": "Slow bottom-to-top detector background gradient.",
    },
    {
        "key": "low_qz_cut_fraction",
        "label": "Low-qz suppression",
        "minimum": 0.00,
        "maximum": 0.08,
        "decimals": 3,
        "help": "Smoothly suppresses background below this fraction of the ROI qz span.",
    },
    {
        "key": "blur_sigma_px",
        "label": "PSF blur σ (px)",
        "minimum": 0.00,
        "maximum": 0.60,
        "decimals": 3,
        "help": "Gaussian point-spread blur applied only to the generated physical background.",
    },
)


def default_physical_background_step() -> Dict[str, Any]:
    step: Dict[str, Any] = {"plugin": "physical_background", "enabled": False}
    for definition in PHYSICAL_BACKGROUND_PARAMETERS:
        step[f"{definition['key']}_min"] = definition["minimum"]
        step[f"{definition['key']}_max"] = definition["maximum"]
    return step


def _range_spec(value: Any, fallback: float = 0.0) -> Dict[str, float]:
    """Normalize legacy scalar and v2 min/max parameter representations."""
    if isinstance(value, dict):
        minimum = float(value.get("minimum", value.get("value", fallback)))
        maximum = float(value.get("maximum", minimum))
        return {"minimum": minimum, "maximum": maximum}
    number = float(fallback if value is None else value)
    return {"minimum": number, "maximum": number}


def synchronize_parameter_specs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronize nested physics controls with the flat generator label map.

    The nested form/structure/layer configuration is the user-facing source of
    truth. ``parameters`` remains a flat compatibility contract for dataset
    generation and model targets.
    """
    from .plugins import REGISTRY

    sample = config.setdefault("sample", {})
    particles = sample.setdefault("particles", [])
    if not particles:
        particles.append({"plugin": "spherical_segment", "material": "Copper", "enabled": True})
    particle = particles[0]
    particle_key = str(particle.get("plugin", "spherical_segment"))
    try:
        particle_plugin = REGISTRY.get("particle", particle_key)
    except KeyError:
        particle_key = "spherical_segment"
        particle["plugin"] = particle_key
        particle_plugin = REGISTRY.get("particle", particle_key)

    legacy_parameters = copy.deepcopy(config.get("parameters", {}))
    if "spacing_nm" in legacy_parameters and "D_nm" not in legacy_parameters:
        legacy_parameters["D_nm"] = legacy_parameters.pop("spacing_nm")
    particle_ranges = particle.setdefault("parameters", {})
    valid_particle_keys = {str(item["key"]) for item in particle_plugin.parameters}
    particle_ranges = {key: value for key, value in particle_ranges.items() if key in valid_particle_keys}
    for definition in particle_plugin.parameters:
        key = str(definition["key"])
        source = particle_ranges.get(key, legacy_parameters.get(key, definition))
        normalized = _range_spec(source, float(definition.get("minimum", 0.0)))
        normalized["distribution"] = str(source.get("distribution", "uniform")) if isinstance(source, dict) else "uniform"
        particle_ranges[key] = normalized
    particle["parameters"] = particle_ranges

    interference = sample.setdefault("interference", {"plugin": "none", "enabled": False})
    interference_key = str(interference.get("plugin", "none"))
    try:
        interference_plugin = REGISTRY.get("interference", interference_key)
    except KeyError:
        interference_key = "none"
        interference["plugin"] = "none"
        interference_plugin = REGISTRY.get("interference", "none")
    interference["enabled"] = interference_key != "none"
    structure_ranges = interference.setdefault("parameters", {})
    valid_structure_keys = {str(item["key"]) for item in interference_plugin.parameters}
    structure_ranges = {key: value for key, value in structure_ranges.items() if key in valid_structure_keys}
    for definition in interference_plugin.parameters:
        key = str(definition["key"])
        legacy = legacy_parameters.get(key)
        if key == "sigma_D_ratio" and legacy is None and "sigma_ratio" in interference:
            legacy = {"minimum": interference["sigma_ratio"], "maximum": interference["sigma_ratio"]}
        source = structure_ranges.get(key, legacy or definition)
        normalized = _range_spec(source, float(definition.get("minimum", 0.0)))
        normalized["distribution"] = str(source.get("distribution", "uniform")) if isinstance(source, dict) else "uniform"
        structure_ranges[key] = normalized
    interference["parameters"] = structure_ranges

    flat: Dict[str, Dict[str, Any]] = {}
    for key, spec in particle_ranges.items():
        flat[key] = {**copy.deepcopy(spec), "source": "form_factor"}
    for key, spec in structure_ranges.items():
        flat[key] = {**copy.deepcopy(spec), "source": "structure_factor"}

    normalized_layers = []
    for index, layer in enumerate(sample.get("layers", [])):
        normalized_layer = copy.deepcopy(layer)
        thickness = _range_spec(layer.get("thickness_nm", 0.0))
        roughness = _range_spec(layer.get("roughness_nm", 0.0))
        normalized_layer["thickness_nm"] = thickness
        normalized_layer["roughness_nm"] = roughness
        normalized_layers.append(normalized_layer)
        if normalized_layer.get("enabled", True) and thickness["maximum"] > thickness["minimum"]:
            flat[f"layer_{index}_thickness_nm"] = {**thickness, "distribution": "uniform", "source": "layer"}
        if normalized_layer.get("enabled", True) and roughness["maximum"] > roughness["minimum"]:
            flat[f"layer_{index}_roughness_nm"] = {**roughness, "distribution": "uniform", "source": "layer"}
    sample["layers"] = normalized_layers

    retired_physics_keys = {"spacing_nm"}
    for plugin in (*REGISTRY.list("particle"), *REGISTRY.list("interference")):
        retired_physics_keys.update(str(item["key"]) for item in plugin.parameters)
    known = set(flat) | retired_physics_keys
    for key, spec in legacy_parameters.items():
        if key not in known and not key.startswith("layer_") and isinstance(spec, dict):
            flat[key] = copy.deepcopy(spec)
            flat[key].setdefault("source", "custom")
    config["parameters"] = flat
    config.setdefault("sample", {}).setdefault("constraints", {}).pop("paracrystal_sigma_le_0_2d", None)
    config["schema_version"] = max(2, int(config.get("schema_version", 1)))
    return config


def trainable_parameter_names(config: Dict[str, Any]) -> List[str]:
    return [
        name
        for name, spec in config.get("parameters", {}).items()
        if float(spec.get("maximum", 0.0)) > float(spec.get("minimum", 0.0))
    ]


def default_project_config() -> Dict[str, Any]:
    """Return a complete, versioned project configuration."""
    return {
        "schema_version": 2,
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
            "threshold": {
                "enabled": True,
                "minimum": 0.0,
                "maximum": 1000000000000.0,
                "auto_reference_upper": True,
                "upper_quantile": 99.999,
            },
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
            "radius_nm": {"distribution": "uniform", "minimum": 1.0, "maximum": 15.0, "source": "form_factor"},
            "height_nm": {"distribution": "uniform", "minimum": 1.0, "maximum": 20.0, "source": "form_factor"},
            "D_nm": {"distribution": "uniform", "minimum": 3.0, "maximum": 50.0, "source": "structure_factor"},
            "sigma_D_ratio": {"distribution": "uniform", "minimum": 0.05, "maximum": 0.20, "source": "structure_factor"},
        },
        "sample": {
            "surface_density_per_nm2": 0.01,
            "particles": [{
                "plugin": "spherical_segment",
                "material": "Copper",
                "enabled": True,
                "parameters": {
                    "radius_nm": {"distribution": "uniform", "minimum": 1.0, "maximum": 15.0},
                    "height_nm": {"distribution": "uniform", "minimum": 1.0, "maximum": 20.0},
                },
            }],
            "mixture": {
                "mode": "gaussian_mixture",
                "components": 5,
                "sigma_fraction_min": 0.01,
                "sigma_fraction_max": 0.30,
                "random_weights": True,
            },
            "layers": [
                {"material": "Copper", "thickness_nm": {"minimum": 20.0, "maximum": 20.0}, "roughness_nm": {"minimum": 0.0, "maximum": 0.0}, "enabled": True},
                {"material": "Polymer", "thickness_nm": {"minimum": 50.0, "maximum": 50.0}, "roughness_nm": {"minimum": 0.0, "maximum": 0.0}, "enabled": True},
            ],
            "substrate": {"material": "Silicon", "roughness_nm": 0.0},
            "interference": {
                "plugin": "paracrystal",
                "enabled": True,
                "parameters": {
                    "D_nm": {"distribution": "uniform", "minimum": 3.0, "maximum": 50.0},
                    "sigma_D_ratio": {"distribution": "uniform", "minimum": 0.05, "maximum": 0.20},
                },
            },
            "constraints": {
                "segment_height_le_2r": True,
                "interparticle_spacing_gt_2r": True,
            },
        },
        "preprocessing": {
            "steps": [
                default_physical_background_step(),
                {"plugin": "gaussian_noise", "enabled": True, "snr_min_db": 80.0, "snr_max_db": 110.0},
                {"plugin": "poisson_noise", "enabled": False, "count_scale_min": 1.0, "count_scale_max": 20.0},
                {"plugin": "mask", "enabled": True},
                {"plugin": "log", "enabled": True, "epsilon": 1e-6},
                {"plugin": "normalize", "enabled": True, "mode": "range", "lower": 0.0, "upper": 1.0},
                {"plugin": "random_edge_crop", "enabled": False, "maximum_px": 4},
            ]
        },
        "model": {
            "output_mode": "regression",
            "layers": [
                {"type": "conv2d", "units": 32, "kernel": 3, "activation": "relu"},
                {"type": "maxpool2d", "pool": 2},
                {"type": "conv2d", "units": 64, "kernel": 3, "activation": "relu"},
                {"type": "maxpool2d", "pool": 2},
                {"type": "conv2d", "units": 128, "kernel": 3, "activation": "relu"},
                {"type": "global_average_pooling2d"},
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dropout", "rate": 0.30},
            ],
        },
        "training": {
            "backend": "local",
            "local_python": "",
            "batch_size": 64,
            "epochs": 100,
            "optimizer": "adam",
            "learning_rate": 0.0001,
            "scheduler": "cosine",
            "smoke_samples": 64,
            "smoke_epochs": 2,
        },
        "hpc": {
            "enabled": False,
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
        "runtime": {
            "last_job_id": "",
            "last_project_dir": "",
            "auto_remember": True,
        },
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
    config = synchronize_parameter_specs(copy.deepcopy(config))
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
            if float(spec.get("minimum")) > float(spec.get("maximum")):
                errors.append(f"Parameter {name}: minimum must not exceed maximum.")
            if (dataset.get("sampling") == "log_uniform" or spec.get("distribution") == "log_uniform") and float(spec.get("minimum")) <= 0:
                errors.append(f"Parameter {name}: log-uniform sampling requires a positive minimum.")
        except Exception:
            errors.append(f"Parameter {name}: invalid numeric range.")

    threshold = config.get("mask", {}).get("threshold", {})
    if threshold.get("enabled") and float(threshold.get("minimum", 0.0)) >= float(threshold.get("maximum", 0.0)):
        errors.append("Mask threshold minimum must be smaller than maximum.")
    mixture = config.get("sample", {}).get("mixture", {})
    if int(mixture.get("components", 1)) < 1:
        errors.append("Gaussian mixture components must be at least 1.")
    if float(mixture.get("sigma_fraction_min", 0.0)) > float(mixture.get("sigma_fraction_max", 0.0)):
        errors.append("Gaussian mixture width minimum must not exceed its maximum.")
    for index, layer in enumerate(config.get("sample", {}).get("layers", []), start=1):
        for key, label in (("thickness_nm", "thickness"), ("roughness_nm", "roughness")):
            spec = _range_spec(layer.get(key, 0.0))
            if spec["minimum"] < 0 or spec["minimum"] > spec["maximum"]:
                errors.append(f"Layer {index} {label}: use non-negative min/max with min <= max.")
    for step in config.get("preprocessing", {}).get("steps", []):
        plugin = step.get("plugin")
        if plugin == "physical_background":
            for definition in PHYSICAL_BACKGROUND_PARAMETERS:
                key = str(definition["key"])
                legacy_min = step.get("fraction_min", definition["minimum"]) if key == "target_fraction" else definition["minimum"]
                legacy_max = step.get("fraction_max", definition["maximum"]) if key == "target_fraction" else definition["maximum"]
                if float(step.get(f"{key}_min", legacy_min)) > float(step.get(f"{key}_max", legacy_max)):
                    errors.append(f"Physical background {definition['label']}: minimum must not exceed maximum.")
        if plugin in {"noise", "gaussian_noise"} and float(step.get("snr_min_db", 0.0)) > float(step.get("snr_max_db", 0.0)):
            errors.append("Gaussian noise SNR minimum must not exceed maximum.")
        if plugin == "poisson_noise":
            low = float(step.get("count_scale_min", 0.0))
            high = float(step.get("count_scale_max", 0.0))
            if low <= 0 or low > high:
                errors.append("Poisson photon-count scale must be positive with minimum <= maximum.")
    if not trainable_parameter_names(config):
        errors.append("At least one parameter must have a non-zero training range.")
    model_layers = config.get("model", {}).get("layers", [])
    if not model_layers:
        errors.append("Model Design requires at least one layer before the automatic output layer.")

    particle = next(iter(config.get("sample", {}).get("particles", [])), {})
    constraints = config.get("sample", {}).get("constraints", {})
    if particle.get("plugin") == "spherical_segment" and constraints.get("segment_height_le_2r", False):
        radius = parameters.get("radius_nm", {})
        height_spec = parameters.get("height_nm", {})
        if radius and height_spec and float(height_spec.get("minimum", 0.0)) > 2.0 * float(radius.get("maximum", 0.0)):
            errors.append("Constraint h <= 2R has no feasible values in the selected radius/height ranges.")
    interference = config.get("sample", {}).get("interference", {})
    if interference.get("plugin") == "paracrystal" and constraints.get("interparticle_spacing_gt_2r", False):
        radius = parameters.get("radius_nm", {})
        distance = parameters.get("D_nm", {})
        if radius and distance and float(distance.get("maximum", 0.0)) <= 2.0 * float(radius.get("minimum", 0.0)):
            errors.append("Constraint D > 2R has no feasible values in the selected D/radius ranges.")

    if config.get("simulation", {}).get("engine") == "bornagain_yuxin":
        try:
            import bornagain  # type: ignore  # noqa: F401
        except Exception:
            warnings.append("BornAgain is not installed locally; simulation preview and local physical generation are unavailable.")
    if config.get("hpc", {}).get("enabled", False):
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
    return synchronize_parameter_specs(merge_config(default_project_config(), loaded))
