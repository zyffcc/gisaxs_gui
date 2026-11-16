"""GIMaP against pyFAI on the public frames of tests/data/external, each with its own geometry.

Per pixel: |q| and 2θ; the solid-angle factor (up to one constant: pyFAI normalises to the PONI, GIMaP to the direct
beam); the polarisation factor; the grazing-incidence q∥ and q_z against pyFAI's fiber units qip and qoop (the sample
orientation of pyFAI that matches GIMaP's axes is found by trying the eight). Then the full azimuthal I(q): GIMaP's
GIWAXS radial curve against pyFAI's integrate1d on the same pixels (valid and above the sample horizon), the same q
range and bins, on bins holding the same pixels (at least 20) in both: once with the same estimator (pyFAI integrates GIMaP's frame
divided by solid angle × polarisation, so each bin is the mean of I/(ΩP), as in GIMaP), once with pyFAI's own
normalisation (ΣI / Σ(ΩP) in each bin, which differs where ΩP and I both vary across a bin), after the one constant of
the solid-angle normalisation is divided out.

pyFAI is not a dependency of GIMaP: run with an interpreter that has it, e.g. a venv made with --system-site-packages.
usage: python tools/benchmarks/pyfai_comparison.py [--json FILE] [--polarization 0.95]
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gimap.features.analyze.domain.giwaxs import _radial_bins, giwaxs_maps, reduce_giwaxs  # noqa: E402
from src.gimap.features.analyze.domain.models import GiwaxsSettings  # noqa: E402
from src.gimap.features.analyze.domain.intensity import intensity_factor  # noqa: E402
from src.gimap.features.analyze.domain.validity import valid_pixels  # noqa: E402
from src.gimap.features.assistant.domain import parse_poni  # noqa: E402
from src.gimap.shared.detector_io import load_detector_image  # noqa: E402
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile  # noqa: E402
from src.gimap.shared.geometry.q_mapping import grazing_q_map, transmission_map  # noqa: E402

EXTERNAL = ROOT / "tests" / "data" / "external"
MIN_PIXELS = 20


def _frames():
    manifest = json.loads((EXTERNAL / "manifest.json").read_text(encoding="utf-8"))
    for key, entry in manifest.items():
        yield key, entry


def _geometries(key: str, entry: dict, shape):
    """GIMaP's geometry (the .poni tilt in it) and pyFAI's, of one frame."""
    pyfai_geometry = importlib.import_module("pyFAI.geometry")
    incidence = float(entry.get("incidence_deg", 0.2))
    if "poni" in entry:
        text = (EXTERNAL / entry["poni"]).read_text(encoding="utf-8")
        read = parse_poni(text)
        flat = DetectorGeometry(read.pixel_size_x_m, read.pixel_size_y_m, read.distance_mm * 1e-3,
                                read.beam_center_x_px, read.beam_center_y_px, read.wavelength_angstrom or 1.0,
                                incidence)
        gimap = InstrumentProfile(key, flat, None, shape, dropped_tilt=read.tilt).geometry
        reference = importlib.import_module("pyFAI").load(str(EXTERNAL / entry["poni"]))
        if reference.wavelength is None:
            reference.wavelength = 1e-10
    else:  # no .poni: an untilted detector from the manifest (beam centre in pixels, x then y)
        pixel = float(entry["pixel_size_m"])
        x, y = entry["beam_center_px"]
        gimap = DetectorGeometry(pixel, pixel, float(entry["distance_m"]), float(x), float(y),
                                 float(entry["wavelength_angstrom"]), incidence)
        reference = pyfai_geometry.Geometry(dist=float(entry["distance_m"]), poni1=float(y) * pixel,
                                            poni2=float(x) * pixel, pixel1=pixel, pixel2=pixel,
                                            wavelength=float(entry["wavelength_angstrom"]) * 1e-10)
    return gimap, reference, incidence


def _integrator(reference, shape):
    integrator = importlib.import_module("pyFAI.integrator.azimuthal").AzimuthalIntegrator()
    integrator.set_config(reference.get_config())
    if integrator.detector.shape is None or tuple(integrator.detector.shape) != tuple(shape):
        integrator.detector.shape = integrator.detector.max_shape = tuple(shape)
    return integrator


def _relative(a, b, where):
    return float(np.max(np.abs(a[where] / b[where] - 1.0)))


def _fiber(reference, shape, gimap_map, incidence, usable):
    """The pyFAI sample orientation whose qip, qoop match GIMaP's q∥, q_z, and the largest differences (Å⁻¹).

    A new integrator for each orientation: pyFAI keeps the arrays of a unit by its name."""
    units = importlib.import_module("pyFAI.units")
    best = None
    for orientation in range(1, 9):
        integrator = _integrator(reference, shape)
        arrays = []
        for name in ("qip_A^-1", "qoop_A^-1"):
            unit = units.get_unit_fiber(name, incident_angle=incidence, tilt_angle=0.0,
                                        sample_orientation=orientation, angle_unit="deg")
            arrays.append(np.asarray(integrator.array_from_unit(unit=unit, typ="center", scale=True), float))
        qip, qoop = arrays
        error = (float(np.max(np.abs(qip[usable] - gimap_map.q_parallel[usable]))),
                 float(np.max(np.abs(qoop[usable] - gimap_map.qz[usable]))))
        if best is None or max(error) < max(best[1]):
            best = (orientation, error)
    return best


def compare(key: str, entry: dict, polarization: float) -> dict:
    frame = load_detector_image(EXTERNAL / entry["frame"])
    data = np.asarray(frame.data, dtype=np.float64)
    shape = data.shape[:2]
    valid = valid_pixels(frame.data, frame.mask)
    if "mask" in entry:
        valid &= np.asarray(load_detector_image(EXTERNAL / entry["mask"]).data) == 0
    gimap, reference, incidence = _geometries(key, entry, shape)
    integrator = _integrator(reference, shape)

    # per pixel
    transmission = transmission_map(shape, gimap)
    q_pyfai = np.asarray(integrator.center_array(shape, unit="q_A^-1", scale=True), float)
    tth_pyfai = np.degrees(np.asarray(integrator.center_array(shape, unit="2th_rad", scale=False), float))
    inner = valid & (q_pyfai > 0)
    solid_gimap = intensity_factor(shape, gimap, solid_angle=True).astype(float)
    solid_pyfai = np.asarray(integrator.solidAngleArray(shape), float)
    solid_ratio = solid_gimap[valid] / solid_pyfai[valid]
    pol_gimap = intensity_factor(shape, gimap, polarization=polarization).astype(float)
    pol_pyfai = np.asarray(integrator.polarization(shape, factor=polarization), float)
    grazing = grazing_q_map(shape, gimap)
    maps = giwaxs_maps(shape, gimap)
    usable = valid & maps.above_horizon
    orientation, (ip_error, oop_error) = _fiber(reference, shape, grazing, incidence, usable)

    # the azimuthal I(q), solid angle and polarisation corrected
    factor = intensity_factor(shape, gimap, solid_angle=True, polarization=polarization)
    with np.errstate(invalid="ignore", divide="ignore"):
        corrected = (data / factor).astype(np.float32)
    usable &= np.isfinite(corrected)
    q_low, q_high = float(maps.q[usable].min()), float(maps.q[usable].max())
    bins = _radial_bins(maps, usable, q_low, q_high)
    reduction = reduce_giwaxs(corrected, usable, gimap, GiwaxsSettings(bins=bins), maps=maps, with_map=False,
                              counts=False)
    radial = next(curve for curve in reduction.curves if curve.key == "radial")
    q_gimap, i_gimap = np.asarray(radial.x, float), np.asarray(radial.intensity, float)
    step = (q_high - q_low) / bins
    pixels = np.asarray(radial.pixels)

    def against(image, **corrections):
        """GIMaP's I(q) over pyFAI's on the bins that hold the same number of pixels in both (GIMaP merges sparse bins
        at the edges of the detector; a pixel whose q lies on a bin edge can go either way)."""
        result = integrator.integrate1d(image, bins, unit="q_A^-1", radial_range=(q_low, q_high),
                                        mask=(~usable).astype(np.int8), method=("no", "histogram", "cython"),
                                        **corrections)
        q_ref, i_ref, counts = (np.asarray(v, float) for v in (result.radial, result.intensity, result.count))
        nearest = np.clip(np.searchsorted(q_ref, q_gimap), 1, q_ref.size - 1)
        nearest -= (q_gimap - q_ref[nearest - 1]) < (q_ref[nearest] - q_gimap)
        good = ((np.abs(q_ref[nearest] - q_gimap) < 0.01 * step) & (pixels >= MIN_PIXELS)
                & (counts[nearest] == pixels) & (i_ref[nearest] > 0) & (i_gimap > 0))
        ratio = i_gimap[good] / i_ref[nearest][good]
        scale = float(np.median(ratio))
        deviation = np.abs(ratio / scale - 1.0)
        return dict(bins_compared=int(good.sum()), bins_gimap=int(q_gimap.size), scale=scale, median_relative=float(np.median(deviation)),
                    p95_relative=float(np.percentile(deviation, 95)), max_relative=float(np.max(deviation)))

    # the same estimator: pyFAI integrates GIMaP's corrected frame (the mean of I/(ΩP) in every bin)
    same = against(np.where(usable, corrected, 0.0).astype(np.float64), correctSolidAngle=False)
    # pyFAI's own normalisation: ΣI / Σ(ΩP) in every bin
    own = against(data, correctSolidAngle=True, polarization_factor=polarization)
    return dict(
        frame=key, shape=list(shape), tilt_deg=round(float(gimap.tilt_deg), 3), incidence_deg=incidence,
        q_max_relative=_relative(transmission.q, q_pyfai, inner),
        two_theta_max_deg=float(np.max(np.abs(transmission.two_theta_deg[valid] - tth_pyfai[valid]))),
        solid_angle_constant=float(np.median(solid_ratio)),
        solid_angle_max_relative=float(np.max(np.abs(solid_ratio / np.median(solid_ratio) - 1.0))),
        polarization_max_relative=_relative(pol_gimap, pol_pyfai, valid),
        fiber_orientation=orientation, q_parallel_max_abs=ip_error, qz_max_abs=oop_error,
        q_range=[q_low, q_high], bins=bins, intensity_same_estimator=same, intensity_pyfai_normalisation=own,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    ap.add_argument("--polarization", type=float, default=0.95)
    a = ap.parse_args()
    pyfai = importlib.import_module("pyFAI")
    rows = [compare(key, entry, a.polarization) for key, entry in _frames()]
    for row in rows:
        print(json.dumps(row))
    if a.json:
        a.json.write_text(json.dumps(dict(pyfai=pyfai.version, polarization=a.polarization, frames=rows), indent=1),
                          encoding="utf-8")


if __name__ == "__main__":
    main()
