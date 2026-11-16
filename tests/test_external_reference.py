"""GIMaP's q at fixed pixels of the public frames, against what pyFAI and pygid give there.

The reference numbers (``tests/data/external/reference_values.json``) were written by
``tools/benchmarks/make_external_reference.py`` with pyFAI 2026.9.0 and pygid 0.2.20; neither is needed here.
pyFAI: |q|, 2θ, and q∥, q_z of its fiber units (sample orientation 4). pygid places pixel i at i·p from the PONI
(pyFAI and GIMaP at (i + ½)·p), so its q_xy, q_z are GIMaP's with the direct beam moved by half a pixel along both
axes (Physics and Methods 3.7).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.assistant.domain import parse_poni
from src.gimap.shared.detector_io import load_detector_image
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile, grazing_q_map, transmission_map

EXTERNAL = Path(__file__).resolve().parent / "data" / "external"
MANIFEST = json.loads((EXTERNAL / "manifest.json").read_text(encoding="utf-8"))
REFERENCE = json.loads((EXTERNAL / "reference_values.json").read_text(encoding="utf-8"))
ABS_Q = 1e-11
"""Å⁻¹: the agreement is that of double precision (10⁻¹³ to 10⁻¹⁵ when it was written)."""


def _geometry(key: str, shape) -> DetectorGeometry:
    entry = MANIFEST[key]
    incidence = REFERENCE["frames"][key]["incidence_deg"]
    if "poni" not in entry:
        pixel = float(entry["pixel_size_m"])
        x, y = entry["beam_center_px"]
        return DetectorGeometry(pixel, pixel, float(entry["distance_m"]), float(x), float(y),
                                float(entry["wavelength_angstrom"]), incidence)
    read = parse_poni((EXTERNAL / entry["poni"]).read_text(encoding="utf-8"))
    flat = DetectorGeometry(read.pixel_size_x_m, read.pixel_size_y_m, read.distance_mm * 1e-3, read.beam_center_x_px,
                            read.beam_center_y_px, read.wavelength_angstrom or 1.0, incidence)
    return InstrumentProfile(key, flat, None, shape, dropped_tilt=read.tilt).geometry


@pytest.mark.parametrize("key", sorted(REFERENCE["frames"]))
def test_every_pixel_quantity_is_what_pyfai_gives(key) -> None:
    shape = load_detector_image(EXTERNAL / MANIFEST[key]["frame"]).data.shape[:2]
    geometry = _geometry(key, shape)
    reference = REFERENCE["frames"][key]
    index = tuple(np.array(reference["pixels"]).T)
    transmission, grazing = transmission_map(shape, geometry), grazing_q_map(shape, geometry)
    pyfai = reference["pyfai"]
    assert transmission.q[index] == pytest.approx(pyfai["q"], abs=ABS_Q)
    assert transmission.two_theta_deg[index] == pytest.approx(pyfai["two_theta_deg"], abs=1e-9)
    assert grazing.q_parallel[index] == pytest.approx(pyfai["q_parallel"], abs=ABS_Q)
    assert grazing.qz[index] == pytest.approx(pyfai["qz"], abs=ABS_Q)


@pytest.mark.parametrize("key", sorted(k for k, v in REFERENCE["frames"].items() if "pygid" in v))
def test_pygid_is_gimap_with_the_pixels_half_a_pixel_away(key) -> None:
    shape = load_detector_image(EXTERNAL / MANIFEST[key]["frame"]).data.shape[:2]
    geometry = _geometry(key, shape)
    reference = REFERENCE["frames"][key]
    index = tuple(np.array(reference["pixels"]).T)
    shifted = geometry.with_beam_center(geometry.beam_center_x_px + 0.5, geometry.beam_center_y_px + 0.5)
    moved, as_given = grazing_q_map(shape, shifted), grazing_q_map(shape, geometry)
    assert moved.q_parallel[index] == pytest.approx(reference["pygid"]["q_xy"], abs=ABS_Q)
    assert moved.qz[index] == pytest.approx(reference["pygid"]["q_z"], abs=ABS_Q)
    assert np.max(np.abs(as_given.qz[index] - np.array(reference["pygid"]["q_z"]))) > 1e-4  # the half pixel shows
