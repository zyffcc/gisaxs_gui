"""Detector file metadata extraction adapter helpers."""

from __future__ import annotations

import re
from typing import Any, Optional

import h5py
import numpy as np

from .nexus_metadata import extract_nexus_metadata, nxs_header_beam_center_px  # noqa: F401  (re-exported)
from .radiation import energy_to_wavelength


def extract_nxs_metadata(handle: h5py.File) -> dict[str, Any]:
    """Header values of a NeXus file whose detector is at ``/entry/instrument/detector``.

    The reader itself uses ``extract_nexus_metadata`` with the detector group of the image dataset.
    """
    return extract_nexus_metadata(handle)


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
