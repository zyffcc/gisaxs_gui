"""Hot and dead pixels are left out automatically; peaks and floating-point negatives are kept."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from src.gimap.features.analyze.application import AnalysisRequest, AnalyzeFrame
from src.gimap.features.analyze.domain import Corrections, find_bad_pixels
from src.gimap.features.analyze.domain.validity import valid_pixels
from src.gimap.features.analyze.infrastructure.adapters import DetectorIoFrameSource
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from tests.test_analyze_workspace import CBF, _pilatus_profile, requires_data

DATA = Path(__file__).parent / "data" / "external"


def _frame(seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).poisson(40.0, (200, 200)).astype(np.float32)


def test_isolated_outliers_are_flagged_and_anything_wider_is_kept() -> None:
    data = _frame()
    data[50, 50] = 5000.0  # a stuck pixel
    data[120, 30] = 0.0  # a dead pixel among ~40 counts
    rows, columns = np.indices(data.shape)
    data += (8000.0 * np.exp(-((rows - 100) ** 2 + (columns - 150) ** 2) / (2 * 0.8**2))).astype(np.float32)  # a sharp spot
    data[:, :20] = np.random.default_rng(2).poisson(0.2, (200, 20))  # a beam-stop shadow with a soft edge
    data[80, 20] = 0.0
    valid = np.ones(data.shape, dtype=bool)
    valid[150:152, :] = False  # a module gap
    bad = find_bad_pixels(data, valid)
    assert bad.hot[50, 50] and bad.dead[120, 30]
    assert bad.hot_count == 1 and bad.dead_count == 1  # not the spot, not the shadow edge, not the gap border
    assert not bad.mask[98:103, 148:153].any()


def test_a_floating_point_frame_keeps_its_negative_values_and_uses_a_robust_noise() -> None:
    data = np.random.default_rng(3).normal(0.0, 5.0, (200, 200)).astype(np.float32)  # dark-subtracted
    data[10, 10] = 400.0
    assert valid_pixels(data, None, negatives_valid=True).all()
    assert not valid_pixels(data, None).all()  # counting detectors: negatives are gap codes
    bad = find_bad_pixels(data, np.ones(data.shape, dtype=bool), counts=False)
    assert bad.hot[10, 10] and bad.hot_count == 1 and bad.dead_count == 0


def test_the_p08_float_frame_keeps_its_negative_pixels() -> None:
    frame = DATA / "giwaxs_p08_mapi" / "S121_MAI_A2_00841.tif"
    analyze = AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([]))
    analysis = analyze(AnalysisRequest(frame, mode="giwaxs"))
    negative = analysis.data < 0
    assert negative.sum() > 100_000 and analysis.valid[negative].mean() > 0.99
    assert analysis.bad_pixels is not None and 0 < analysis.bad_pixels.hot_count < 50


@requires_data
def test_pilatus_overflow_codes_are_left_out_and_come_back_when_turned_off() -> None:
    analyze = AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([_pilatus_profile()]))
    request = AnalysisRequest(CBF, mode="gisaxs")
    analysis = analyze(request)
    bad = analysis.bad_pixels
    assert bad is not None and bad.hot_count >= 3
    hot = np.argwhere(bad.hot)
    assert max(float(analysis.data[r, c]) for r, c in hot) > 1e6  # 2^22-like overflow values, not photons
    assert not analysis.valid[bad.mask].any() and analysis.raw_data is None and analysis.raw_valid[bad.mask].all()
    kept = analyze(replace(request, corrections=Corrections(bad_pixels=False)), loaded=analysis)
    assert kept.bad_pixels is None and kept.valid[bad.mask].all()
