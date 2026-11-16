"""How long Analyze takes to reduce a frame: the public frames of tests/data/external, with their own .poni.

For each frame: the first reduction (the q map of the detector is made then) and the median of the next ones (the
q map is reused, as for every further frame of a series), each the full reduction of Analyze — corrections, masks,
cuts or I(q) and I(χ) — without the automatic analysis and without writing files. One process, the BLAS threads
numpy has. The peak memory is that of the process so far (it grows with the largest frame). Prints a table; with
--json writes it too.
usage: python tools/benchmarks/analyze_speed.py [--repeat 5] [--json FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gimap.features.analyze.application import AnalysisRequest, AnalyzeFrame  # noqa: E402
from src.gimap.features.analyze.infrastructure.adapters import DetectorIoFrameSource  # noqa: E402
from src.gimap.features.assistant.domain import parse_poni  # noqa: E402
from src.gimap.integrations.state.instrument_profiles import InMemoryInstrumentProfileRepository  # noqa: E402
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile  # noqa: E402

EXTERNAL = ROOT / "tests" / "data" / "external"
FRAMES = (  # folder, frame, .poni, mode, αi
    ("saxs_agbh_pyfai", "Pilatus1M.edf", "Pilatus1M.poni", "gisaxs", 0.2),
    ("gisaxs_galaxi", "galaxi_data.tif", "galaxi_bornagain.poni", "gisaxs", 0.2),
    ("giwaxs_p08_mapi", "S121_MAI_A2_00841.tif", "LaB6_2021_12_DESY_P08.poni", "giwaxs", 0.075),
    ("giwaxs_xeuss_pbte", "PbTe_3010_r1_PT_1_10_0_00028.edf", "AgBe_250403_XENOCS.poni", "giwaxs", 0.235),
)


def _profile(folder: Path, poni: str, shape, incidence: float) -> InstrumentProfile:
    read = parse_poni((folder / poni).read_text(encoding="utf-8"))
    geometry = DetectorGeometry(read.pixel_size_x_m, read.pixel_size_y_m, read.distance_mm * 1e-3, read.beam_center_x_px,
                                read.beam_center_y_px, read.wavelength_angstrom or 1.0, incidence)
    return InstrumentProfile("bench", geometry, None, shape, dropped_tilt=read.tilt)


def _peak_mb():
    """The largest memory the process has held so far (MB), when psutil can say it (``None`` otherwise)."""
    try:
        import psutil
    except ImportError:
        return None
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows
    if peak is None:
        import resource  # POSIX: kilobytes on Linux, bytes on macOS
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    return round(peak / 2**20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    source = DetectorIoFrameSource()
    rows = []
    for folder_name, frame, poni, mode, incidence in FRAMES:
        folder = EXTERNAL / folder_name
        shape = source.load(folder / frame, 0).data.shape
        use = AnalyzeFrame(source, InMemoryInstrumentProfileRepository([_profile(folder, poni, shape, incidence)]))
        request = AnalysisRequest(path=folder / frame, frame_index=0, mode=mode, profile_name="bench")
        times = []
        for _ in range(1 + a.repeat):
            start = time.perf_counter()
            analysis = use(request)
            times.append(time.perf_counter() - start)
        assert analysis.reduction is not None
        rows.append(dict(frame=frame, mode=mode, shape=list(shape), megapixels=shape[0] * shape[1] / 1e6,
                         first_s=times[0], next_median_s=statistics.median(times[1:]), peak_memory_mb=_peak_mb()))
    machine = dict(python=platform.python_version(), processor=platform.processor(), cpus=os.cpu_count(),
                   system=platform.platform())
    print(f"{'frame':36s} {'mode':7s} {'pixels':>12s} {'first':>8s} {'next':>8s} {'peak MB':>8s}")
    for row in rows:
        print(f"{row['frame']:36s} {row['mode']:7s} {row['shape'][0]:>5d}x{row['shape'][1]:<6d} "
              f"{row['first_s']:7.2f}s {row['next_median_s']:7.2f}s {row['peak_memory_mb'] or 0:8d}")
    print(machine)
    if a.json:
        a.json.write_text(json.dumps(dict(machine=machine, frames=rows), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
