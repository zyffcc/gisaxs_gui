"""pyFAI ``.poni`` files in Analyze: Load .poni…, Export .poni…, and the ``.poni`` beside a frame.

* **Load .poni…** reads a file (``shared/geometry/poni``), checks it against the frame shown — the frame shape when
  the file states one (``Detector_config: max_shape``), the pixel size when the frame's header states one (1 %) —
  and turns it into an instrument profile of the frame's detector: the direct beam, the distance to it, the pixel
  sizes, rot1–rot3 and the wavelength, with the file named as the source. A file without a wavelength needs one
  from the user.
* **Export .poni…** writes the geometry in use as PONI version 2.1 (pixel sizes, frame shape, distance, PONI,
  rot1–rot3, wavelength). The incidence angle is no part of a ``.poni``.
* **The .poni beside a frame**: in the frame's own folder, the ``.poni`` whose name begins like the frame's (or the
  frame's like its), else the only ``.poni`` there. Several unrelated ones are no answer (none is taken). Its
  geometry is used for the frame when the file fits it (as above), ahead of an instrument profile matched only by
  detector name and frame size; the source says which file it was.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from src.gimap.shared.geometry.poni import PoniFile, format_poni, read_poni
from src.gimap.shared.reproduction import file_reference

PIXEL_TOLERANCE = 0.01
"""A ``.poni`` whose pixel size differs from the frame header's by more than this share is for another detector."""
SHAPE_MISMATCH = "The .poni file is for {poni_shape} frames; this frame is {shape}."
PIXEL_MISMATCH = "The .poni file has {poni_um} µm pixels; the frame header says {frame_um} µm."
NO_WAVELENGTH = "The .poni file states no wavelength: enter the wavelength or the energy."
SIBLING_SOURCE = ".poni beside the frame: {path}"
LOADED_SOURCE = ".poni loaded: {path}"
POSSIBLE_ENCODINGS = ("utf-8", "latin-1")


def geometry_references(resolution):
    """Source calibration when it is available. The profile's complete values survive removal of that file."""
    source = resolution.profile.source if resolution.profile else ""
    for template in (SIBLING_SOURCE, LOADED_SOURCE):
        prefix = template.partition("{path}")[0]
        if source.startswith(prefix):
            path = source[len(prefix):]
            if resolution.how == "file" or Path(path).is_file():
                return {path: file_reference(path)}
    return {}


def _shape_text(shape) -> str:
    return f"{int(shape[0])}×{int(shape[1])}"


def read_poni_file(path) -> PoniFile:
    """The ``.poni`` at ``path`` (``ValueError`` when it is no geometry file; ``OSError`` when unreadable)."""
    raw = Path(path).read_bytes()
    for encoding in POSSIBLE_ENCODINGS:
        try:
            return read_poni(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    raise ValueError("The .poni file is not a text file.")


def poni_problem(poni: PoniFile, shape, pixel_size_m: Optional[tuple[Optional[float], Optional[float]]] = None) -> Optional[str]:
    """Why ``poni`` does not fit a frame of ``shape`` (rows, columns) and header pixel size (x, y in metres; ``None``
    when the header has none), or ``None`` when nothing says it is for another detector."""
    if poni.shape is not None and shape is not None and tuple(int(v) for v in poni.shape) != tuple(int(v) for v in shape[:2]):
        return SHAPE_MISMATCH.format(poni_shape=_shape_text(poni.shape), shape=_shape_text(shape))
    if pixel_size_m:
        frame_x = pixel_size_m[0]
        if frame_x and abs(poni.tilt.pixel2_m - frame_x) > PIXEL_TOLERANCE * frame_x:
            return PIXEL_MISMATCH.format(poni_um=f"{poni.tilt.pixel2_m * 1e6:g}", frame_um=f"{frame_x * 1e6:g}")
    return None


def poni_profile(
    poni: PoniFile, path, *, name: str, detector_name: Optional[str], shape, source_template: str = LOADED_SOURCE,
    wavelength_angstrom: Optional[float] = None, incidence_deg: float = 0.0,
) -> InstrumentProfile:
    """An instrument profile with the geometry of ``poni`` for frames of ``shape`` (``ValueError`` without a
    wavelength). Its ``dropped_tilt`` is the file's own description, so what the tilt changes stays known."""
    if not poni.wavelength_angstrom and not wavelength_angstrom:
        raise ValueError(NO_WAVELENGTH)
    geometry = poni.geometry(wavelength_angstrom=wavelength_angstrom, incidence_deg=incidence_deg)
    return InstrumentProfile(
        name=name, geometry=geometry, detector_name=detector_name,
        detector_shape=tuple(int(v) for v in shape[:2]) if shape is not None else poni.shape,
        source=source_template.format(path=str(path)), dropped_tilt=poni.tilt if poni.tilted else None,
    )


def write_poni_file(geometry: DetectorGeometry, path, *, shape=None, source: str = "") -> Path:
    """Write ``geometry`` as a ``.poni`` (version 2.1) at ``path``; returns the path."""
    comments = ("Written by GIMaP (Analyze ▸ Geometry ▸ Export .poni…)",) + ((f"Geometry: {source}",) if source else ())
    target = Path(path)
    target.write_text(format_poni(geometry, shape=shape, comments=comments), encoding="utf-8")
    return target


def sibling_poni(frame_path) -> Optional[Path]:
    """The ``.poni`` beside a frame (see the module docstring), or ``None``."""
    frame = Path(frame_path)
    folder = frame.parent
    try:
        found = sorted(entry for entry in folder.iterdir() if entry.suffix.lower() == ".poni" and entry.is_file())
    except OSError:
        return None
    if not found:
        return None
    stem = frame.stem.lower()
    named = [entry for entry in found if stem.startswith(entry.stem.lower()) or entry.stem.lower().startswith(stem)]
    if named:
        return max(named, key=lambda entry: len(entry.stem))  # the most specific name
    return found[0] if len(found) == 1 else None


@dataclass
class _Cached:
    stamp: tuple
    poni: Optional[PoniFile]
    path: Optional[Path]


class SiblingPoni:
    """``(frame_path, shape, pixel_size_m, detector_name)`` → the profile of the ``.poni`` beside the frame, or
    ``None`` (no file, no wavelength, or one that does not fit the frame). The file read last is kept until its
    folder or the file changes."""

    def __init__(self) -> None:
        self._cache: dict[str, _Cached] = {}

    def _read(self, frame_path) -> tuple[Optional[Path], Optional[PoniFile]]:
        folder = Path(frame_path).parent
        try:
            stamp = (os.stat(folder).st_mtime_ns,)
        except OSError:
            return None, None
        path = sibling_poni(frame_path)
        if path is not None:
            try:
                stamp += (path.name, os.stat(path).st_mtime_ns)
            except OSError:
                return None, None
        key = f"{folder}|{Path(frame_path).stem}"
        cached = self._cache.get(key)
        if cached is not None and cached.stamp == stamp:
            return cached.path, cached.poni
        poni = None
        if path is not None:
            try:
                poni = read_poni_file(path)
            except (OSError, ValueError):
                poni = None
        self._cache[key] = _Cached(stamp, poni, path)
        return path, poni

    def __call__(self, frame_path, shape, pixel_size_m=None, detector_name: Optional[str] = None) -> Optional[InstrumentProfile]:
        if frame_path is None:
            return None
        path, poni = self._read(frame_path)
        if poni is None or not poni.wavelength_angstrom or poni_problem(poni, shape, pixel_size_m) is not None:
            return None
        named = detector_name or (poni.detector if poni.detector.lower() not in ("", "detector") else None)
        return poni_profile(poni, path, name=path.name, detector_name=named, shape=shape,
                            source_template=SIBLING_SOURCE)


__all__ = [
    "LOADED_SOURCE", "NO_WAVELENGTH", "PIXEL_MISMATCH", "SHAPE_MISMATCH", "SIBLING_SOURCE", "SiblingPoni",
    "poni_problem", "poni_profile", "read_poni_file", "sibling_poni", "write_poni_file",
]
