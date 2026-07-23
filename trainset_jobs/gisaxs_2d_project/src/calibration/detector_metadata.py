from __future__ import annotations

import re
from typing import Any, Optional

import h5py
import numpy as np

from .geometry_model import energy_to_wavelength


def _scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    result = arr.reshape(-1)[0]
    if isinstance(result, bytes):
        return result.decode("utf-8", errors="replace")
    return result.item() if hasattr(result, "item") else result


def _optional(handle: h5py.File, paths: tuple[str, ...]) -> Any:
    for path in paths:
        if path in handle:
            try:
                return _scalar(handle[path][()])
            except Exception:
                continue
    return None


def _metric(handle: h5py.File, path: str) -> Optional[float]:
    if path not in handle:
        return None
    dataset = handle[path]
    value = float(_scalar(dataset[()]))
    units = str(_scalar(dataset.attrs.get("units", "")) or "").lower().replace("µ", "u")
    if units in {"mm", "millimeter", "millimetre"}:
        value *= 1e-3
    elif units in {"um", "micrometer", "micrometre"}:
        value *= 1e-6
    elif units in {"nm", "nanometer", "nanometre"}:
        value *= 1e-9
    return value


def extract_nxs_metadata(handle: h5py.File) -> dict[str, Any]:
    energy = _optional(handle, (
        "/entry/instrument/detector/collection/beam_energy",
        "/entry/instrument/beam/incident_energy",
        "/entry/instrument/monochromator/energy",
        "/entry/beam/incident_energy",
    ))
    energy_kev = float(energy) if energy is not None else None
    if energy_kev is not None and energy_kev > 100.0:
        energy_kev /= 1000.0
    wavelength = _optional(handle, (
        "/entry/instrument/beam/incident_wavelength",
        "/entry/beam/incident_wavelength",
        "/entry/instrument/monochromator/wavelength",
    ))
    wavelength_a = float(wavelength) if wavelength is not None else None
    if wavelength_a is not None and wavelength_a < 1e-6:
        wavelength_a *= 1e10
    if wavelength_a is None and energy_kev:
        wavelength_a = energy_to_wavelength(energy_kev)
    detector_name = _optional(handle, (
        "/entry/instrument/detector/description",
        "/entry/instrument/detector/local_name",
        "/entry/instrument/detector/type",
    ))
    center_x = _optional(handle, (
        "/entry/instrument/detector/beam_center_x",
        "/entry/instrument/detector/beam_center_x_pixel",
    ))
    center_y = _optional(handle, (
        "/entry/instrument/detector/beam_center_y",
        "/entry/instrument/detector/beam_center_y_pixel",
    ))
    return {
        "energy_kev": energy_kev,
        "wavelength_angstrom": wavelength_a,
        "detector_name": str(detector_name) if detector_name is not None else None,
        "pixel_size_x_m": _metric(handle, "/entry/instrument/detector/x_pixel_size"),
        "pixel_size_y_m": _metric(handle, "/entry/instrument/detector/y_pixel_size"),
        # Do not confuse P03 module-translation vectors with sample distance.
        "distance_m": _metric(handle, "/entry/instrument/detector/distance"),
        "beam_center_x_px": float(center_x) if center_x is not None else None,
        "beam_center_y_px": float(center_y) if center_y is not None else None,
    }


def extract_cbf_metadata(header: dict[str, Any], shape: tuple[int, int]) -> dict[str, Any]:
    contents = str(header.get("_array_data.header_contents", ""))
    detector_match = re.search(r"^#\s*Detector:\s*([^,\r\n]+)", contents, re.MULTILINE | re.IGNORECASE)
    pixel_match = re.search(r"Pixel_size\s+([0-9.eE+-]+)\s*m\s*x\s*([0-9.eE+-]+)\s*m", contents, re.IGNORECASE)
    energy_match = re.search(r"(?:Beam_energy|Energy)\s*[:=]?\s*([0-9.eE+-]+)\s*(eV|keV)", contents, re.IGNORECASE)
    energy = None
    if energy_match:
        energy = float(energy_match.group(1)) / (1000.0 if energy_match.group(2).lower() == "ev" else 1.0)
    px = float(pixel_match.group(1)) if pixel_match else None
    py = float(pixel_match.group(2)) if pixel_match else None
    if px is None and shape in {(1679, 1475), (1043, 981), (2527, 2463)}:
        px = py = 172e-6
    return {
        "detector_name": detector_match.group(1).strip() if detector_match else None,
        "pixel_size_x_m": px,
        "pixel_size_y_m": py,
        "energy_kev": energy,
        "wavelength_angstrom": energy_to_wavelength(energy) if energy else None,
        "distance_m": None,
        "beam_center_x_px": None,
        "beam_center_y_px": None,
        "format": "cbf",
        "header": {str(key): str(value) for key, value in header.items()},
        "transformations": [],
        "mask_semantics": "True is invalid; negative CBF sentinels and non-finite pixels",
    }
