"""Detector frames for 2D Prediction through GIMaP's own readers, and what a module may know about them.

``DetectorPredictionImageRepository`` reads CBF, EDF and TIFF frames with ``shared/detector_io`` (the readers
of Analyze), sums a stack and keeps the facts of its first frame. Values stay as stored, as the fabio reader
gave them before: a module's preprocessing decides what to do with them. One rule makes the formats alike:
a pixel the reader knows has no data (a module gap stored as 0 on a Pilatus EDF or TIFF, an EDF ``Dummy``
value) is given as −1, the code a Pilatus CBF stores in its gaps and the one the modules already treat as
invalid. A P03 CBF comes out exactly as before (its gaps hold −1 already).

``ProfileFrameContexts`` describes a loaded frame for a module that asks for it (a preprocessing entry with
a ``context`` parameter): files, format, shape, detector, the frame's own facts (αi and where it came from,
times, monitors) and, when an instrument profile matches the detector and image size, the geometry with a
``q_map()`` that gives q for every pixel in Å⁻¹ by GIMaP's pixel conventions (pixel corners on integers,
row 0 at the top). The αi of the geometry is the frame's when its header or side file states one, otherwise
the profile's. The q maps of the last ``KEEP_Q_MAPS`` geometries and frame sizes are kept (computing them for a
Pilatus 2M frame takes longer than a prediction); their arrays are read-only.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.gimap.shared.detector_io import load_detector_image
from src.gimap.shared.geometry.q_mapping import grazing_q_map

from ...application import LoadedPredictionImage

FACT_KEYS = (
    "incidence_deg", "incidence_source", "exposure_time_s", "timestamp", "file_time", "side_monitors",
    "side_files", "module_gaps", "stored_dtype", "format", "transformations", "header_wavelength_angstrom",
    "header_distance_m", "header_beam_xy_px", "energy_source",
)
"""The frame facts handed on (when the reader found them); everything else of the header stays behind."""


def _plain(value: Any, depth: int = 0) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if depth < 2 and isinstance(value, Mapping):
        return {str(key): _plain(item, depth + 1) for key, item in value.items()}
    if depth < 2 and isinstance(value, (list, tuple)):
        return [_plain(item, depth + 1) for item in value]
    return str(value)


def _facts(image, shape: tuple[int, int]) -> dict[str, Any]:
    metadata = image.metadata or {}
    facts: dict[str, Any] = {key: _plain(metadata[key]) for key in FACT_KEYS if metadata.get(key) is not None}
    facts["shape"] = [int(shape[0]), int(shape[1])]
    facts["detector"] = image.detector_name
    header = {
        "pixel_size_x_m": image.pixel_size_x_m,
        "pixel_size_y_m": image.pixel_size_y_m,
        "wavelength_angstrom": image.wavelength_angstrom,
        "energy_kev": image.energy_kev,
        "distance_m": image.distance_m,
    }
    facts["header"] = {key: float(value) for key, value in header.items() if value is not None}
    return facts


class DetectorPredictionImageRepository:
    def load(self, paths: tuple[Path, ...]) -> LoadedPredictionImage:
        if not paths:
            raise ValueError("At least one detector image path is required")
        summed = None
        facts: dict[str, Any] = {}
        for path in paths:
            source = Path(path)
            if not source.is_file():
                raise FileNotFoundError(source)
            image = load_detector_image(source)
            values = np.array(image.data, dtype=np.float32, copy=True)
            if values.ndim != 2 or values.size == 0:
                raise ValueError(f"Detector image must be a non-empty 2D array: {source}")
            if image.mask is not None:
                no_data = np.asarray(image.mask, dtype=bool) & np.isfinite(values) & (values >= 0)
                values[no_data] = -1.0
            if summed is None:
                summed = values
                facts = _facts(image, values.shape)
            else:
                if values.shape != summed.shape:
                    raise ValueError(f"Detector stack shape mismatch: {values.shape} != {summed.shape}")
                summed += values
        return LoadedPredictionImage(summed, tuple(Path(path) for path in paths), facts)


def _frame_incidence(facts: Mapping[str, Any]) -> float | None:
    try:
        value = float(facts.get("incidence_deg"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0.0 < value < 90.0 else None


def _geometry_record(geometry) -> dict[str, float]:
    keys = (
        "pixel_size_x_m", "pixel_size_y_m", "distance_m", "beam_center_x_px", "beam_center_y_px",
        "wavelength_angstrom", "incidence_deg", "rot1_rad", "rot2_rad", "rot3_rad", "sample_roll_deg",
    )
    record = {key: float(getattr(geometry, key)) for key in keys}
    record["orientation"] = int(geometry.orientation)
    return record


KEEP_Q_MAPS = 2
"""Geometries (with their frame size) whose q maps are kept."""


class ProfileFrameContexts:
    """What a module may know about a loaded frame (see the module docstring)."""

    def __init__(self, profiles=None):
        self._profiles = profiles
        self._q_maps: OrderedDict = OrderedDict()

    def q_maps(self, shape: tuple[int, ...], geometry) -> dict[str, np.ndarray]:
        """q_x, q_y, q_z, q_∥ (Å⁻¹) and α_f (°) of every pixel, float32 and read-only; computed once per geometry."""
        key = (tuple(shape), tuple(sorted(_geometry_record(geometry).items())))
        maps = self._q_maps.get(key)
        if maps is None:
            mapped = grazing_q_map(tuple(shape), geometry)
            raw = {"qx": mapped.qx, "qy": mapped.qy, "qz": mapped.qz, "q_parallel": mapped.q_parallel}
            if mapped.sin_exit is not None:
                raw["alpha_f_deg"] = np.degrees(np.arcsin(np.clip(mapped.sin_exit, -1.0, 1.0)))
            maps = {name: np.array(value, dtype=np.float32) for name, value in raw.items()}
            for array in maps.values():
                array.flags.writeable = False
            self._q_maps[key] = maps
            while len(self._q_maps) > KEEP_Q_MAPS:
                self._q_maps.popitem(last=False)
        else:
            self._q_maps.move_to_end(key)
        return dict(maps)

    def describe(self, loaded: LoadedPredictionImage) -> dict[str, Any]:
        facts = dict(loaded.facts or {})
        shape = tuple(int(n) for n in loaded.image.shape)
        context: dict[str, Any] = {
            "files": [str(path) for path in loaded.source_paths],
            "format": Path(loaded.source_paths[0]).suffix.lower().lstrip("."),
            "shape": list(shape),
            "detector": facts.get("detector"),
            "frame": facts,
            "geometry": None,
            "geometry_source": "",
        }
        profile = None
        if self._profiles is not None:
            try:
                profile = self._profiles.match(detector_name=facts.get("detector"), shape=shape)
            except Exception:  # noqa: BLE001 - an unreadable profile store: no geometry, said in the context
                profile = None
        if profile is None:
            context["geometry_source"] = (
                f"no instrument profile for this detector and image size ({shape[1]} × {shape[0]} pixels)"
            )
            return context
        geometry = profile.geometry
        incidence = _frame_incidence(facts)
        incidence_source = f"instrument profile '{profile.name}'"
        if incidence is not None:
            geometry = replace(geometry, incidence_deg=incidence)
            incidence_source = str(facts.get("incidence_source") or "the frame's header")
        context["geometry"] = _geometry_record(geometry)
        context["geometry_source"] = f"instrument profile '{profile.name}'; αi from {incidence_source}"

        def q_map(_geometry=geometry, _shape=shape) -> dict[str, np.ndarray]:
            return self.q_maps(_shape, _geometry)

        context["q_map"] = q_map
        return context


__all__ = ["DetectorPredictionImageRepository", "FACT_KEYS", "KEEP_Q_MAPS", "ProfileFrameContexts"]
