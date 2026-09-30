"""Analyze corrections (background, valid range) and the extra GIWAXS cuts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.analyze.application import (
    AnalysisRequest,
    AnalyzeFrame,
    Corrections,
    GiwaxsSettings,
    QBox,
    Sector,
    two_theta_deg,
)
from src.gimap.features.analyze.domain import (
    apply_valid_range,
    reduce_giwaxs,
    subtract_background,
)
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile

SHAPE = (240, 240)


def _geometry() -> DetectorGeometry:
    # 100 µm pixels, 60 mm away: the frame reaches wide angles (GIWAXS).
    return DetectorGeometry(100e-6, 100e-6, 0.06, 120.0, 200.0, 1.0, 0.2)


def _frame(value: float = 5.0) -> np.ndarray:
    return np.full(SHAPE, value, dtype=np.float32)


class FakeFrames:
    """In-memory frame source that counts loads."""

    def __init__(self, frames: dict[str, np.ndarray]):
        self.frames = frames
        self.loads: list[str] = []

    def load(self, path, frame_index=0):
        self.loads.append(str(path))
        data = self.frames[str(path)]
        return SimpleNamespace(data=data, mask=None, detector_name="Test", metadata={})

    def frame_count(self, path) -> int:
        return 1


def test_subtract_background_scales_and_joins_the_valid_pixels() -> None:
    frame, background = _frame(10.0), _frame(2.0)
    valid = np.ones(SHAPE, dtype=bool)
    background_valid = valid.copy()
    background_valid[0, 0] = False
    corrected, both = subtract_background(frame, valid, background, background_valid, 2.5)
    assert corrected[5, 5] == pytest.approx(5.0)
    assert not both[0, 0] and np.isnan(corrected[0, 0])
    with pytest.raises(ValueError, match="background frame"):
        subtract_background(frame, valid, background[:10], background_valid[:10])


def test_valid_range_is_inclusive_and_optional() -> None:
    data = np.array([[0.0, 1.0, 2.0, 3.0]])
    valid = np.ones_like(data, dtype=bool)
    assert apply_valid_range(data, valid, None, None) is valid
    np.testing.assert_array_equal(apply_valid_range(data, valid, 1.0, 2.0), [[False, True, True, False]])


def test_custom_sector_box_and_two_theta_curves() -> None:
    geometry = _geometry()
    image = _frame()
    valid = np.ones(SHAPE, dtype=bool)
    settings = GiwaxsSettings(
        bins=150,
        sector=Sector(-20.0, 20.0, 1.0, 3.0),
        box=QBox((0.5, 1.5), (0.5, 2.0)),
    )
    reduction = reduce_giwaxs(image, valid, geometry, settings)
    keys = {curve.key for curve in reduction.curves}
    assert {"radial", "sector", "sector_chi", "box_q", "box_qz", "box_qpar"} <= keys
    sector = reduction.curve("sector")
    assert sector.x.min() >= 1.0 - 1e-6 and sector.x.max() <= 3.0 + 1e-6
    assert len(reduction.curve("radial").x) <= 150
    chi = reduction.curve("sector_chi")
    assert chi.x.min() >= -20.0 and chi.x.max() <= 20.0
    box = reduction.curve("box_qz")
    assert box.x.min() >= 0.5 - 1e-6 and box.x.max() <= 2.0 + 1e-6
    # A flat frame stays flat whatever the cut.
    np.testing.assert_allclose(sector.intensity, 5.0)

    in_two_theta = reduce_giwaxs(image, valid, geometry, replace(settings, x_axis="two_theta"))
    radial_q = reduction.curve("radial")
    radial_2theta = in_two_theta.curve("radial")
    assert radial_2theta.x_label == "2θ (°)"
    np.testing.assert_allclose(radial_2theta.x, two_theta_deg(radial_q.x, geometry.wavelength_angstrom))
    # Box and χ profiles keep their own axes.
    assert in_two_theta.curve("box_qz").x_label == "qz (Å⁻¹)"


def test_corrections_reuse_the_loaded_frame_and_load_the_background_once(tmp_path: Path) -> None:
    frame_path, background_path = tmp_path / "frame.tif", tmp_path / "background.tif"
    for path in (frame_path, background_path):
        path.write_bytes(b"x")  # the fake source reads arrays; the stat needs a file
    frames = FakeFrames({str(frame_path): _frame(10.0), str(background_path): _frame(4.0)})
    profile = InstrumentProfile("Test", _geometry(), "Test", SHAPE)
    analyze = AnalyzeFrame(frames, InMemoryInstrumentProfileRepository([profile]))

    plain = analyze(AnalysisRequest(frame_path, mode="giwaxs"))
    assert plain.raw_data is None and plain.data[0, 0] == 10.0

    request = AnalysisRequest(
        frame_path,
        mode="giwaxs",
        corrections=Corrections(background_path=str(background_path), background_scale=0.5, maximum=100.0),
    )
    corrected = analyze(request, loaded=plain)
    assert corrected.data[0, 0] == pytest.approx(8.0)
    assert corrected.raw_data[0, 0] == 10.0
    np.testing.assert_allclose(corrected.reduction.curve("radial").intensity, 8.0, rtol=1e-6)
    again = analyze(replace(request, corrections=replace(request.corrections, background_scale=1.0)), loaded=corrected)
    assert again.data[0, 0] == pytest.approx(6.0)
    # The frame was loaded once, the background once; nothing is reloaded.
    assert frames.loads == [str(frame_path), str(background_path)]


def test_series_corrections_align_the_distance_and_normalise_to_the_peak(tmp_path: Path) -> None:
    from src.gimap.features.analyze.application import CorrectSeriesFrame, SeriesCorrection
    from src.gimap.features.analyze.domain import giwaxs_maps

    true_geometry = _geometry()
    q = giwaxs_maps(SHAPE, true_geometry).q
    q_ref = 2.0
    frame = (1.0 + 400.0 * np.exp(-0.5 * ((q - q_ref) / 0.03) ** 2)).astype(np.float32)
    frame_path = tmp_path / "frame.tif"
    frame_path.write_bytes(b"x")
    frames = FakeFrames({str(frame_path): frame})
    # The profile distance is 3 % off, as after a change of sample height.
    wrong = InstrumentProfile("Test", replace(true_geometry, distance_m=0.0618), "Test", SHAPE)
    analyze = AnalyzeFrame(frames, InMemoryInstrumentProfileRepository([wrong]))
    request = AnalysisRequest(frame_path, mode="giwaxs", giwaxs=GiwaxsSettings(bins=400))
    series = SeriesCorrection(reference_q=q_ref, half_width=0.25, align_distance=True, normalize=True)

    analysis, info = CorrectSeriesFrame(analyze)(request, series)

    assert info["aligned_distance_m"] == pytest.approx(0.06, rel=5e-3)
    radial = analysis.reduction.curve("radial")
    peak = int(np.argmax(radial.intensity))
    assert radial.x[peak] == pytest.approx(q_ref, abs=0.02)
    assert radial.intensity[peak] == pytest.approx(1.0, rel=1e-6)
    assert info["normalization_factor"] == pytest.approx(1.0 / 401.0, rel=0.05)
    # A later frame reuses the first frame's factor unless per_frame is chosen.
    again, info_again = CorrectSeriesFrame(analyze)(request, series, factor=0.5)
    assert info_again["normalization_factor"] == 0.5
