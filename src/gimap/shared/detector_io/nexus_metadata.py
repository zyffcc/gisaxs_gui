"""Header values of a NeXus / HDF5 detector file, in the units GIMaP works with.

Every value is looked up relative to the detector group of the image dataset and to its NXentry
first, then at the classic ``/entry/instrument/detector`` places.  The ``units`` attribute decides
the conversion; only a value without one is read by its magnitude (energy above 100 is eV, a
wavelength below 1e-6 is metres), as GIMaP always did.  Units are read as udunits writes them: case,
spaces and the plural of a spelled-out name do not matter (``meters``, ``Electron Volt``).  A value in
a unit that is not known here is left out rather than guessed.  A dataset with one value per frame
gives the value of the frame asked.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import h5py
import numpy as np

from .nexus_datasets import _text
from .radiation import energy_to_wavelength

DEFAULT_DETECTOR_GROUP = "/entry/instrument/detector"
DEFAULT_ENTRY = "/entry"

_ENERGY_TO_KEV = {"kev": 1.0, "kiloelectronvolt": 1.0, "ev": 1e-3, "electronvolt": 1e-3,
                  "j": 1.0 / 1.602176634e-16, "joule": 1.0 / 1.602176634e-16}
_WAVELENGTH_TO_ANGSTROM = {
    "angstrom": 1.0, "ångström": 1.0, "ångstrom": 1.0, "angström": 1.0, "ang": 1.0, "a": 1.0, "å": 1.0,
    "nm": 10.0, "nanometer": 10.0, "nanometre": 10.0, "pm": 0.01, "picometer": 0.01, "picometre": 0.01,
    "m": 1e10, "meter": 1e10, "metre": 1e10,
}
_LENGTH_TO_METRE = {
    "m": 1.0, "meter": 1.0, "metre": 1.0, "cm": 1e-2, "centimeter": 1e-2, "centimetre": 1e-2,
    "mm": 1e-3, "millimeter": 1e-3, "millimetre": 1e-3,
    "um": 1e-6, "micrometer": 1e-6, "micrometre": 1e-6, "micron": 1e-6, "microns": 1e-6,
    "nm": 1e-9, "nanometer": 1e-9, "nanometre": 1e-9,
}
_TIME_TO_SECOND = {
    "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
    "ms": 1e-3, "msec": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3,
    "us": 1e-6, "usec": 1e-6, "microsecond": 1e-6, "microseconds": 1e-6,
    "ns": 1e-9, "nanosecond": 1e-9, "nanoseconds": 1e-9, "min": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hour": 3600.0,
}
_PIXEL_UNITS = {"", "pixel", "pixels", "px"}


def _scale(scales: dict[str, float], units: str) -> Optional[float]:
    """The factor of a unit as it is written; ``None`` for a unit that is not in ``scales``.

    Spaces, underscores and hyphens do not matter, and a spelled-out name may be plural (``meters``,
    ``electron volts``).  A symbol is never read as the plural of another one: ``ms`` is not metres.
    """
    name = re.sub(r"[\s_\-]+", "", units)
    if name in scales:
        return scales[name]
    return scales.get(name[:-1]) if name.endswith("s") and len(name) > 3 else None


def _plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value.item() if hasattr(value, "item") else value


def _item(dataset: h5py.Dataset, frame: Optional[int]) -> Any:
    """The value of the dataset: of ``frame`` when it holds one value per frame, else its first."""
    if dataset.shape == ():
        return _plain(dataset[()])
    if dataset.size == 0:
        return None
    if dataset.ndim == 1:
        index = frame if frame is not None and dataset.shape[0] > 1 and 0 <= frame < dataset.shape[0] else 0
        return _plain(dataset[index])
    return _plain(np.asarray(dataset[()]).reshape(-1)[0])


def _units(dataset: h5py.Dataset) -> str:
    return _text(dataset.attrs.get("units", "")).strip().lower().replace("µ", "u").replace("μ", "u")


def _find(handle: h5py.File, paths: tuple[str, ...]) -> Optional[h5py.Dataset]:
    for path in paths:
        try:
            node = handle.get(path)
        except (KeyError, OSError, ValueError):
            continue
        if isinstance(node, h5py.Dataset):
            return node
    return None


def _places(groups: tuple[str, ...], names: tuple[str, ...]) -> tuple[str, ...]:
    """Each name under each group, the groups in order, without repeating a place."""
    return tuple(dict.fromkeys(f"{group}/{name}" for group in groups for name in names))


def _number(handle: h5py.File, paths: tuple[str, ...], frame: Optional[int]) -> tuple[Optional[float], str]:
    """``(finite value, its units)`` of the first place that has one."""
    for path in paths:
        dataset = _find(handle, (path,))
        if dataset is None:
            continue
        try:
            value = float(_item(dataset, frame))
        except (TypeError, ValueError, OSError):
            continue
        if np.isfinite(value):
            return value, _units(dataset)
    return None, ""


def _quantity(handle: h5py.File, paths: tuple[str, ...], scales: dict[str, float],
              frame: Optional[int] = None) -> Optional[float]:
    """The value in the unit ``scales`` converts to; as written when the file gives no unit."""
    value, units = _number(handle, paths, frame)
    if value is None:
        return None
    if not units:
        return value
    scale = _scale(scales, units)
    return value * scale if scale is not None else None


def _header_pixels(handle: h5py.File, paths: tuple[str, ...], pixel_size: Optional[float]) -> Optional[float]:
    """A header position in pixels; lengths are divided by the pixel size."""
    value, units = _number(handle, paths, None)
    if value is None or units in _PIXEL_UNITS:
        return value
    scale = _scale(_LENGTH_TO_METRE, units)
    if scale is None or not pixel_size:
        return None
    return value * scale / pixel_size


def nxs_header_beam_center_px(handle: h5py.File, detector_group: str = DEFAULT_DETECTOR_GROUP
                              ) -> Optional[tuple[float, float]]:
    """NeXus ``beam_center_x``/``_y`` in pixels of the stored image.

    NeXus puts the fast (x) axis last in the dataset, so the pair is
    ``(along the last axis, along the first axis)``.
    """
    groups = tuple(dict.fromkeys((detector_group, DEFAULT_DETECTOR_GROUP)))
    fast = _header_pixels(handle, _places(groups, ("beam_center_x", "beam_center_x_pixel")),
                          _quantity(handle, _places(groups, ("x_pixel_size",)), _LENGTH_TO_METRE))
    slow = _header_pixels(handle, _places(groups, ("beam_center_y", "beam_center_y_pixel")),
                          _quantity(handle, _places(groups, ("y_pixel_size",)), _LENGTH_TO_METRE))
    return (fast, slow) if fast is not None and slow is not None else None


def _text_value(handle: h5py.File, paths: tuple[str, ...], frame: Optional[int] = None) -> Optional[str]:
    dataset = _find(handle, paths)
    if dataset is None:
        return None
    try:
        value = _item(dataset, frame)
    except (TypeError, ValueError, OSError):
        return None
    return None if value is None else str(value)


def _frame_time(handle: h5py.File, detector: tuple[str, ...], entries: tuple[str, ...],
                frame: Optional[int]) -> tuple[Optional[str], Optional[float]]:
    """``(time stamp of the frame as text, time of the frame in seconds)`` from a per-frame dataset."""
    if frame is None:
        return None, None
    places = _places(detector, ("timestamps", "timestamp", "collection/timestamps", "frame_time_stamps"))
    places += _places(entries, ("data/timestamps", "data/timestamp", "data/time"))
    for path in places:
        dataset = _find(handle, (path,))
        if dataset is None or dataset.ndim != 1 or not 1 < dataset.shape[0] > frame:
            continue
        try:
            value = _plain(dataset[frame])
        except (TypeError, ValueError, OSError):
            continue
        if isinstance(value, str):
            return value, None
        scale = _scale(_TIME_TO_SECOND, _units(dataset))
        return None, float(value) * (1.0 if scale is None else scale)
    return None, None


def extract_nexus_metadata(
    handle: h5py.File,
    detector_group: str = DEFAULT_DETECTOR_GROUP,
    entry: str = DEFAULT_ENTRY,
    frame: Optional[int] = None,
) -> dict[str, Any]:
    """Header values of the detector in ``detector_group`` of ``entry`` (for ``frame`` where they are per frame)."""
    detector = tuple(dict.fromkeys((detector_group.rstrip("/"), DEFAULT_DETECTOR_GROUP)))
    entries = tuple(dict.fromkeys((entry.rstrip("/"), DEFAULT_ENTRY)))
    energy, energy_units = _number(handle, _places(detector, ("collection/beam_energy",)) + _places(entries, (
        "instrument/beam/incident_energy", "instrument/monochromator/energy", "beam/incident_energy",
        "sample/beam/incident_energy")), frame)
    if energy is not None:
        if energy_units:
            scale = _scale(_ENERGY_TO_KEV, energy_units)
            energy = energy * scale if scale is not None else None
        elif energy > 100.0:  # no unit given: eV
            energy /= 1000.0
    wavelength, wavelength_units = _number(handle, _places(entries, (
        "instrument/beam/incident_wavelength", "beam/incident_wavelength", "instrument/monochromator/wavelength",
        "sample/beam/incident_wavelength")), frame)
    if wavelength is not None:
        if wavelength_units:
            scale = _scale(_WAVELENGTH_TO_ANGSTROM, wavelength_units)
            wavelength = wavelength * scale if scale is not None else None
        elif wavelength < 1e-6:  # no unit given: metres
            wavelength *= 1e10
    if wavelength is None and energy:
        wavelength = energy_to_wavelength(energy)
    center_x, _units_x = _number(handle, _places(detector, ("beam_center_x", "beam_center_x_pixel")), None)
    center_y, _units_y = _number(handle, _places(detector, ("beam_center_y", "beam_center_y_pixel")), None)
    stamp, frame_time = _frame_time(handle, detector, entries, frame)
    if stamp is None:
        stamp = _text_value(handle, _places(entries, ("start_time",)) + _places(detector, (
            "timestamp", "collection/date", "detectorSpecific/data_collection_date")))
    if stamp is None and "file_time" in handle.attrs:  # the Lambda server writes the time of the file only here
        stamp = str(_plain(np.asarray(handle.attrs["file_time"]).reshape(-1)[0]))
    return {
        "energy_kev": energy,
        "wavelength_angstrom": wavelength,
        "detector_name": _text_value(handle, _places(detector, ("description", "local_name", "type"))),
        "pixel_size_x_m": _quantity(handle, _places(detector, ("x_pixel_size",)), _LENGTH_TO_METRE),
        "pixel_size_y_m": _quantity(handle, _places(detector, ("y_pixel_size",)), _LENGTH_TO_METRE),
        # Do not confuse P03 module-translation vectors with sample distance.
        "distance_m": _quantity(handle, _places(detector, ("distance", "detector_distance")), _LENGTH_TO_METRE),
        "beam_center_x_px": center_x,
        "beam_center_y_px": center_y,
        "header_beam_center_px": nxs_header_beam_center_px(handle, detector[0]),
        "exposure_time_s": _quantity(handle, _places(detector, (
            "count_time", "frame_time", "exposure_time", "collection/count_time")), _TIME_TO_SECOND, frame),
        "timestamp": stamp,
        "frame_time_s": frame_time,
    }


__all__ = ["DEFAULT_DETECTOR_GROUP", "DEFAULT_ENTRY", "extract_nexus_metadata", "nxs_header_beam_center_px"]
