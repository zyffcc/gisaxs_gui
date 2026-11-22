"""Recover a(t) and c(t) from a synthetic, time-stamped GIWAXS detector stack."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gimap.features.analyze.application import AnalysisRequest, AnalyzeFrame
from src.gimap.features.analyze.application.cell_series import refine_series
from src.gimap.features.analyze.application.indexing import PhaseSpec
from src.gimap.features.analyze.application.series_map import frame_key
from src.gimap.features.analyze.infrastructure.adapters import DetectorIoFrameSource
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile


def make_stack(folder, *, blank=None):
    """Poisson counts for six rings in an expanding tetragonal film, generated from Bragg's law."""
    path = Path(folder) / "synthetic_cell_series.nxs"
    shape = (280, 280)
    centre = (140., 248.)
    pixel, distance, wavelength = .0001, .070, 1.
    profile = InstrumentProfile("Synthetic film geometry",
                                DetectorGeometry(pixel, pixel, distance, *centre, wavelength, .2), None, shape)
    row, col = np.indices(shape)
    radius = pixel * np.hypot(col + .5 - centre[0], row + .5 - centre[1])
    q = 4 * np.pi * np.sin(.5 * np.arctan2(radius, distance)) / wavelength
    times = np.arange(8) * 2.
    a, c = 10. + .009 * times, 16. - .004 * times
    hkls = [(1, 0, 0), (0, 0, 2), (1, 1, 0), (1, 1, 2), (2, 0, 0), (0, 0, 4)]
    width = .025
    rng = np.random.default_rng(2117)
    with h5py.File(path, "w") as out:
        detector = out.create_group("entry/instrument/detector")
        detector.attrs["NX_class"] = "NXdetector"
        detector.create_dataset("timestamps", data=times).attrs["units"] = "s"
        images = detector.create_dataset("data", (len(times), *shape), dtype="uint16")
        for i in range(len(times)):
            y = 18 + 5 * q
            for h, k, l in hkls:
                peak = 2 * np.pi * np.sqrt((h*h + k*k) / a[i]**2 + l*l / c[i]**2)
                y += 650 * np.exp(-4 * np.log(2.) * ((q - peak) / width)**2)
            images[i] = 0 if i == blank else rng.poisson(y)
    phase = PhaseSpec("Tetragonal film", (a[0], a[0], c[0], 90, 90, 90))
    start = 3
    seeds = [{"hkl": [h, k, l], "q": float(2*np.pi*np.sqrt((h*h+k*k)/a[start]**2+l*l/c[start]**2)), "fwhm": width}
             for h, k, l in hkls]
    indexed = [{"phase": phase.name, "system": "tetragonal", "seeds": seeds}]
    source = {"analysis": {"source_file": str(path), "frame_index": start, "measurement": "giwaxs"},
              "q_axis": {"critical_angle_deg": 0.}}
    return path, times, a, c, profile, phase, indexed, source


def benchmark(folder, *, blank=None):
    path, times, a, c, profile, phase, indexed, source = make_stack(folder, blank=blank)
    analyze = AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([profile]))
    frames = [(path, i) for i in range(len(times))]
    numbers = {frame_key(*ref): i + 1 for i, ref in enumerate(frames)}
    record = refine_series(frames, numbers, AnalysisRequest(path, mode="giwaxs"), analyze, [phase], indexed,
                           source, check_cancel=lambda: None, progress=lambda *_: None)
    return times, a, c, record


if __name__ == "__main__":
    with TemporaryDirectory() as temp:
        times, a, c, record = benchmark(temp)
    for key, truth in (("a", a), ("c", c), ("V", a**2 * c)):
        fits = [row["fits"]["Tetragonal film"] for row in record["frames"]]
        errors = np.array([f["values"][key] for f in fits]) - truth
        print(f"{key}: {len(fits)} frames, max absolute error {np.max(abs(errors)):.5g}, "
              f"RMSE {np.sqrt(np.mean(errors**2)):.5g}")
