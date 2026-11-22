"""Synthetic-detector mode of insitu_online.py. Each stopwatch surrounds production code."""

from __future__ import annotations

import os
import platform
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.gimap.features.analyze.application import AnalysisRequest, AnalyzeFrame, use_cases
from src.gimap.features.analyze.domain import GisaxsCutSettings, reduce_gisaxs
from src.gimap.features.analyze.infrastructure.adapters import DetectorIoFrameSource
from src.gimap.features.fitting.application.series_chain import chained_fit
from src.gimap.features.fitting.application.series_fit import SeriesSettings, next_start
from src.gimap.features.fitting.application.single_fit import Curve, FitModel, evaluate, new_component
from src.gimap.features.fitting.application.single_predict import predict_request, predicted
from src.gimap.features.fitting.infrastructure.adapters.posterior_job import run_predict
from src.gimap.integrations.state.instrument_profiles import InMemoryInstrumentProfileRepository
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from src.gimap.shared.geometry.q_mapping import grazing_q_region


def _cpu_model() -> str:
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as cpu:
            return winreg.QueryValueEx(cpu, "ProcessorNameString")[0].strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def _model(radius: float) -> FitModel:
    return FitModel((new_component("sphere", radius=radius, distance=40.0),)).with_values({
        (0, "Int"): 2000.0, (0, "sigma_R"): 0.15, (0, "sigma_nn"): 0.25, ("globals", "background"): 4.0,
    })


def _write_frame(path: Path, geometry, width: int, row: int, index: int, radius: float) -> None:
    # Populate a horizontal band with the same intensity law as the single model in Fitting.
    q = grazing_q_region(geometry, (row, row + 1), (0, width)).qy.ravel()
    profile = evaluate(_model(radius), 10 * np.abs(q))
    rng = np.random.default_rng(index)
    counts = rng.poisson(4.0, (width, width)).astype(np.int32)
    counts[row - 4:row + 5] = rng.poisson(profile, (9, width))
    Image.fromarray(counts).save(path)


def _detector(work: Path, mpx: int, frames: int) -> dict:
    width = {1: 1000, 4: 2000}[mpx]
    geometry = DetectorGeometry(100e-6, 100e-6, 1.6, width / 2, width * 0.8, 1.0, 0.2)
    row = int(geometry.row_for_exit_angle(0.2))
    profile = InstrumentProfile("Synthetic", geometry, None, (width, width))
    analyze = AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([profile]))
    settings = SeriesSettings(start="chain", q_range=(0.05, 1.5))
    single = _model(5.2)
    previous = None
    timings = []
    network = {}
    for index in range(frames):
        path = work / f"synthetic_{mpx}M_{index}.tif"
        new_stage = index == frames // 2
        radius = 5.0 if index < frames // 2 else 7.0
        _write_frame(path, geometry, width, row, index, radius)
        request = AnalysisRequest(path=path, mode="gisaxs", profile_name=profile.name, with_map=False,
                                  gisaxs=GisaxsCutSettings(horizontal_row=row + 0.5))
        cut_s = predict_s = 0.0

        def cuts(*args, **kwargs):
            nonlocal cut_s
            start = time.perf_counter()
            result = reduce_gisaxs(*args, **kwargs)
            cut_s += time.perf_counter() - start
            return result

        def predict(data, start_model):
            nonlocal predict_s, network
            start = time.perf_counter()
            payload = predict_request(data, start_model, "model").payload
            result = run_predict(payload, lambda *_: None, lambda: False)
            solutions = predicted(result)
            network = result["network"]
            predict_s += time.perf_counter() - start
            return solutions

        with patch.object(use_cases, "reduce_gisaxs", cuts):
            start = time.perf_counter()
            analysis = analyze(request)
            cut = analysis.reduction.curve("horizontal")
            curve = Curve.from_arrays(cut.x, cut.intensity, cut.sigma, name=path.stem, unit="angstrom")
            reduction_s = time.perf_counter() - start - cut_s
        start_model = next_start(previous, single, settings)
        start = time.perf_counter()
        frame = chained_fit(index, path.name, curve, start_model, single, settings, previous,
                            new_stage=new_stage, solutions=predict)
        chained_s = time.perf_counter() - start - predict_s
        if not frame.fitted or (new_stage and predict_s == 0):
            raise RuntimeError(f"Benchmark could not fit frame {index + 1}: {frame.reason or 'no Predict restart'}")
        timings.append(dict(frame=index + 1, fit_points=frame.points, stage_start=new_stage, chain_break=frame.chain,
                            reduction_s=reduction_s, cut_s=cut_s, chained_fit_s=chained_s, predict_s=predict_s))
        previous = frame
        path.unlink()
    breaks = [f["predict_s"] for f in timings if f["predict_s"] > 0]
    fields = ("reduction_s", "cut_s", "chained_fit_s", "predict_s")
    means = {key: statistics.mean(f[key] for f in timings) for key in fields}
    return dict(shape=[width, width], frames=frames, cut_row_px=row + 0.5, mean_seconds_per_frame=means,
                predict_calls=len(breaks), mean_seconds_per_predict_call=statistics.mean(breaks),
                network=network, per_frame=timings)


def benchmark(work: Path, megapixels, frames: int) -> dict:
    return {
        "cpu": _cpu_model(), "system": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__,
        "settings": {
            "thread_environment": {key: os.getenv(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                                  "OPENBLAS_NUM_THREADS")},
            "workers": 1, "mode": "GISAXS curves only, bad-pixel detection enabled",
            "input": "synthetic local int32 TIFF, Poisson seed = zero-based frame index",
            "pixel_um": 100, "distance_m": 1.6, "wavelength_angstrom": 1.0, "incidence_deg": 0.2,
            "center_fraction_xy": [0.5, 0.8], "horizontal_half_height_px": 2.5,
            "model": "one sphere and a paracrystal, effective interface",
            "stage_radii_nm": [5, 7], "D_nm": 40, "stage_start_frame": frames // 2 + 1,
            "fit": "forward chained refinement, mean of both halves, default edge handling",
            "fit_q_range_nm^-1": [0.05, 1.5], "predict": "particles in model, seed 0, in-process job handler",
            "timing": "Wall time, including the first frame. Reduction includes TIFF read and frame preparation. "
                      "Cuts include the Yoneda search and both cuts. Chained fit excludes the timed Predict call "
                      "and includes restart refinement. Frame synthesis, polling, GUI and export are excluded.",
        },
        "detectors": [_detector(work, mpx, frames) for mpx in megapixels],
    }
