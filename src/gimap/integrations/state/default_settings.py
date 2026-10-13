"""Built-in default values of the application settings (formerly ``core.global_params``)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_DETECTOR = {
    "preset": "Pilatus 2M",
    "distance": 2000,  # mm
    "nbins_x": 1475,
    "nbins_y": 1475,
    "pixel_size_x": 172,  # µm
    "pixel_size_y": 172,  # µm
    "beam_center_x": 737,  # px
    "beam_center_y": 737,  # px
}

DEFAULT_SETTINGS: dict[str, dict[str, Any]] = {
    "beam": {
        "wavelength": 0.1,  # nm
        "grazing_angle": 0.4,  # deg
        "beam_size_x": 0.1,  # mm
        "beam_size_y": 0.1,  # mm
        "flux": 1e12,  # photons/s
        "polarization": "horizontal",
    },
    "detector": {
        **_DETECTOR,
        "exposure_time": 1.0,  # s
        "dark_current": 0.1,  # counts/pixel/s
        "readout_noise": 5.0,  # electrons RMS
    },
    "sample": {
        "particle_shape": "Sphere",
        "particle_size": 10.0,  # nm
        "size_distribution": 0.1,
        "material": "Gold",
        "substrate": "Silicon",
        "thickness": 100.0,  # nm
        "roughness": 1.0,  # nm RMS
        "density": 0.5,
        "orientation": "random",
    },
    "preprocessing": {
        "focus_region": {"type": "q", "qr_min": 0.01, "qr_max": 3.0, "qz_min": 0.01, "qz_max": 3.0},
        "noising": {"type": "Gaussian", "snr_min": 80, "snr_max": 130},
        "others": {"crop_edge": True, "add_mask": True, "normalize": True, "logarization": True},
    },
    "trainset": {
        "file_name": "trainset",
        "save_path": "",
        "trainset_number": 1000,
        "save_every": 100,
        "batch_size": 10,
        "detector": deepcopy(_DETECTOR),
    },
    "gisaxs_predict": {
        "framework": "tensorflow 2.15.0",
        "mode": "single",
        "input_file": "",
        "input_folder": "",
        "export_path": "",
        "stack_value": "1",
        "range_value": "",
        "showing_value": "",
        "auto_scale": True,
        "vmin": None,
        "vmax": None,
        "colormap": "viridis",
    },
    "system": {
        "calculation_method": "DWBA",
        "approximation": "Born",
        "substrate_layers": 1,
        "max_iterations": 1000,
        "convergence_threshold": 1e-6,
        "parallel_processing": True,
        "num_threads": 4,
    },
}

DEFAULT_PREFERENCES: dict[str, Any] = {
    "fit.points_num": 50,
    "fit.interp_method": "Linear",
    "_auto_k_enabled": False,
}
"""Interface preferences that still have a consumer (window/scaling keys are gone:
the window uses Qt high-DPI scaling and the Appearance settings instead)."""


def default_settings() -> dict[str, dict[str, Any]]:
    return deepcopy(DEFAULT_SETTINGS)


def default_preferences() -> dict[str, Any]:
    return deepcopy(DEFAULT_PREFERENCES)


__all__ = ["DEFAULT_PREFERENCES", "DEFAULT_SETTINGS", "default_preferences", "default_settings"]
