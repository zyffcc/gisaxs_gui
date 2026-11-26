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


def _sparse_counting_frame(seed: int, counts: float) -> tuple[np.ndarray, np.ndarray]:
    """A short exposure (``counts`` per pixel) with a rod and a bright band, and every kind of bad pixel: hot ones also
    on the borders and corners, dead ones in the band (one beside invalid pixels), a defective row and column."""
    rng = np.random.default_rng(seed)
    data = rng.poisson(counts, (400, 360)).astype(np.float32)
    data[:, 120:122] += rng.poisson(60, (400, 2))
    data[150:154, :] += rng.poisson(60, (4, 360))
    for r, c in [(5, 5), (0, 100), (399, 50), (200, 359), (0, 0), (399, 359), (152, 40)]:
        data[r, c] = 3000.0
    for r, c in [(151, 30), (152, 200), (150, 90), (152, 77)]:  # (150, 90): at the band's edge, kept
        data[r, c] = 0.0
    columns = rng.choice(np.arange(10, 350), 24, replace=False)
    data[60, columns] = 500.0
    data[60, columns[:4] + 1] = 500.0  # pairs, which the isolated-pixel test misses
    data[rng.choice(np.arange(10, 390), 20, replace=False), 330] = 800.0
    valid = np.ones(data.shape, dtype=bool)
    valid[:, 70:72] = False  # a module gap
    valid[153:156, 74:80] = False
    data[~valid] = 0.0
    return data, valid


def test_a_sparse_counting_frame_is_searched_from_its_bright_pixels_and_flags_the_same(monkeypatch) -> None:
    from src.gimap.features.analyze.domain import bad_pixels

    for seed in range(4):
        for counts in (0.02, 0.5, 2.0):
            data, valid = _sparse_counting_frame(seed, counts)
            assert bad_pixels._sparse(data, valid) is not None  # (the search from the bright pixels)
            sparse = find_bad_pixels(data, valid)
            with monkeypatch.context() as patched:
                patched.setattr(bad_pixels, "SPARSE_SHARE", -1.0)  # the whole-frame search
                whole = find_bad_pixels(data, valid)
            for name in ("hot", "dead", "lines"):
                np.testing.assert_array_equal(getattr(sparse, name), getattr(whole, name), err_msg=f"{name} {seed} {counts}")
            assert sparse.hot[0, 0] and sparse.hot[399, 359] and sparse.dead[151, 30] and sparse.dead[152, 77]
            assert not sparse.dead[150, 90] and sparse.lines[60].any() and (sparse.hot | sparse.lines)[:, 330].sum() >= 15
    data, valid = _sparse_counting_frame(9, 0.5)
    data[40, 40] = -3.0  # a valid pixel below 0: the whole-frame search, as before
    assert bad_pixels._sparse(data, valid) is None
    assert bad_pixels._sparse(*(_sparse_counting_frame(0, 40.0))) is None  # a full frame too
