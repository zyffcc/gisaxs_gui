"""Detector file metadata extraction adapter helpers."""

from __future__ import annotations

import re
from typing import Any, Optional

import h5py
import numpy as np

from .radiation import energy_to_wavelength


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


_LENGTH_TO_METRE = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "micrometer": 1e-6, "micrometre": 1e-6}


def _header_pixels(handle: h5py.File, paths: tuple[str, ...], pixel_size_path: str) -> Optional[float]:
    """A header position in pixels; lengths are divided by the pixel size."""
    for path in paths:
        if path not in handle:
            continue
        dataset = handle[path]
        try:
            value = float(_scalar(dataset[()]))
        except Exception:
            continue
        if not np.isfinite(value):
            return None
        units = str(_scalar(dataset.attrs.get("units", "")) or "").lower().replace("µ", "u")
        if units in {"", "pixel", "pixels", "px"}:
            return value
        scale = _LENGTH_TO_METRE.get(units)
        size = _metric(handle, pixel_size_path)
        if scale is None or not size:
            return None
        return value * scale / size
    return None


def nxs_header_beam_center_px(handle: h5py.File) -> Optional[tuple[float, float]]:
    """NeXus ``beam_center_x``/``_y`` in pixels of the stored image.

    NeXus puts the fast (x) axis last in the dataset, so the pair is
    ``(along the last axis, along the first axis)``.
    """
    fast = _header_pixels(
        handle,
        ("/entry/instrument/detector/beam_center_x", "/entry/instrument/detector/beam_center_x_pixel"),
        "/entry/instrument/detector/x_pixel_size",
    )
    slow = _header_pixels(
        handle,
        ("/entry/instrument/detector/beam_center_y", "/entry/instrument/detector/beam_center_y_pixel"),
        "/entry/instrument/detector/y_pixel_size",
    )
    return (fast, slow) if fast is not None and slow is not None else None


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
    exposure_time = _optional(handle, (
        "/entry/instrument/detector/count_time",
        "/entry/instrument/detector/frame_time",
        "/entry/instrument/detector/exposure_time",
        "/entry/instrument/detector/collection/count_time",
    ))
    timestamp = _optional(handle, (
        "/entry/start_time",
        "/entry/instrument/detector/timestamp",
        "/entry/instrument/detector/collection/date",
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
        "header_beam_center_px": nxs_header_beam_center_px(handle),
        "exposure_time_s": float(exposure_time) if exposure_time is not None else None,
        "timestamp": str(timestamp) if timestamp is not None else None,
    }


_CBF_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_WAVELENGTH_UNITS_TO_ANGSTROM = {"a": 1.0, "å": 1.0, "angstrom": 1.0, "nm": 10.0}
_LENGTH_UNITS_TO_METRE = {"m": 1.0, "mm": 1e-3}


def _cbf_header_text(contents: str, key: str) -> Optional[str]:
    """Text after ``# <key>`` on its header line (leading blanks allowed)."""
    match = re.search(
        rf"^[ \t]*#[ \t]*{key}(?![A-Za-z0-9_])[ \t:]*([^\r\n]*)",
        contents,
        re.MULTILINE | re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _cbf_quantity(contents: str, key: str, units: dict[str, float]) -> Optional[float]:
    """Positive ``<number> <unit>`` value; ``None`` if absent, unitless or in an unknown unit."""
    text = _cbf_header_text(contents, key)
    match = re.match(rf"({_CBF_NUMBER})[ \t]*([^\s,;]+)", text or "")
    if match is None:
        return None
    scale = units.get(match.group(2).casefold())
    value = float(match.group(1)) * scale if scale is not None else None
    return value if value is not None and np.isfinite(value) and value > 0 else None


def _cbf_beam_xy(contents: str) -> Optional[tuple[float, float]]:
    text = _cbf_header_text(contents, "Beam_xy")
    match = re.match(
        rf"\(\s*({_CBF_NUMBER})\s*,\s*({_CBF_NUMBER})\s*\)[ \t]*(\S*)", text or ""
    )
    if match is None or match.group(3).casefold() not in {"", "pixel", "pixels"}:
        return None
    return float(match.group(1)), float(match.group(2))


def extract_cbf_metadata(header: dict[str, Any], shape: tuple[int, int]) -> dict[str, Any]:
    """Metadata from a Pilatus/Eiger mini-CBF header.

    ``Wavelength``, ``Detector_distance`` and ``Beam_xy`` are optional header
    lines that a beamline fills in only when it passes those values to the
    detector server, so they can be stale.  They are reported under separate
    ``header_*`` keys and never replace the primary fields (energy,
    wavelength, distance, beam centre) that calibration and analysis use.
    ``header_beam_xy_px`` is the ``(x, y)`` pair exactly as written, in the
    file's pixel coordinates; its origin convention is the beamline's.
    """
    contents = str(header.get("_array_data.header_contents", ""))
    detector_match = re.search(r"^#\s*Detector:\s*([^,\r\n]+)", contents, re.MULTILINE | re.IGNORECASE)
    pixel_match = re.search(r"Pixel_size\s+([0-9.eE+-]+)\s*m\s*x\s*([0-9.eE+-]+)\s*m", contents, re.IGNORECASE)
    energy_match = re.search(r"(?:Beam_energy|Energy)\s*[:=]?\s*([0-9.eE+-]+)\s*(eV|keV)", contents, re.IGNORECASE)
    exposure_match = re.search(r"Exposure_time\s+([0-9.eE+-]+)\s*s", contents, re.IGNORECASE)
    timestamp_match = re.search(r"^#\s*(\d{4}-\d{2}-\d{2}T[^\r\n]+)", contents, re.MULTILINE)
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
        "header_wavelength_angstrom": _cbf_quantity(
            contents, "Wavelength", _WAVELENGTH_UNITS_TO_ANGSTROM
        ),
        "header_distance_m": _cbf_quantity(contents, "Detector_distance", _LENGTH_UNITS_TO_METRE),
        "header_beam_xy_px": _cbf_beam_xy(contents),
        "exposure_time_s": float(exposure_match.group(1)) if exposure_match else None,
        "timestamp": timestamp_match.group(1).strip() if timestamp_match else None,
        "format": "cbf",
        "header": {str(key): str(value) for key, value in header.items()},
        "transformations": [],
        "mask_semantics": "True is invalid; negative CBF sentinels and non-finite pixels",
    }


def _edf_float(header: dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        try:
            value = float(str(header[key]).split()[0])
        except (KeyError, ValueError, IndexError):
            continue
        if value == value and abs(value) != float("inf"):
            return value
    return None


def extract_edf_metadata(header: dict[str, Any], shape: tuple[int, int]) -> dict[str, Any]:
    """Metadata from an ESRF data format (EDF) header, in the SAXS-package keywords.

    ``PSize_1/2`` (m) are the column / row pixel sizes, ``SampleDistance`` (m) and
    ``WaveLength`` (m) the set-up and ``Center_1/2`` the beam in pixel coordinates of
    the stored image.  Like CBF headers these can be stale: the distance, wavelength
    and centre are reported under ``header_*`` keys only; the pixel size is used.
    """
    px, py = _edf_float(header, "PSize_1"), _edf_float(header, "PSize_2")
    if px is None and tuple(shape) in {(1679, 1475), (1043, 981), (2527, 2463), (619, 487)}:
        px = py = 172e-6
    wavelength_m = _edf_float(header, "WaveLength", "Wavelength", "wavelength")
    wavelength = wavelength_m * 1e10 if wavelength_m and wavelength_m < 1e-6 else wavelength_m
    center_x, center_y = _edf_float(header, "Center_1"), _edf_float(header, "Center_2")
    name = str(header.get("DetectorName") or header.get("detector") or "").strip() or None
    return {
        "detector_name": name,
        "pixel_size_x_m": px,
        "pixel_size_y_m": py if py is not None else px,
        "energy_kev": None,
        "wavelength_angstrom": None,
        "distance_m": None,
        "beam_center_x_px": None,
        "beam_center_y_px": None,
        "header_wavelength_angstrom": wavelength,
        "header_distance_m": _edf_float(header, "SampleDistance"),
        "header_beam_xy_px": None if center_x is None or center_y is None else (center_x, center_y),
        "exposure_time_s": _edf_float(header, "ExposureTime", "count_time", "acq_expo_time"),
        "timestamp": None,
        "format": "edf",
        "header": {str(key): str(value) for key, value in header.items()},
        "transformations": [],
        "mask_semantics": "True is invalid; EDF Dummy values and non-finite pixels",
    }
