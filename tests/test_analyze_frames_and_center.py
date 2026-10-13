"""Analyze additions that replace the detector half of Cut & Fitting.

Frame summing, the gap guard around invalid pixels, the symmetry-based beam
centre and the series hand-over (every export also writes the Fitting input).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.analyze.application import (
    AnalysisRequest,
    AnalyzeFrame,
    Corrections,
    ExportAnalysis,
    SaveInstrumentProfile,
    export_stem,
)
from src.gimap.features.analyze.domain import guard_invalid, sum_frames, symmetric_center_x
from src.gimap.features.analyze.infrastructure.adapters import CsvCurveWriter
from src.gimap.features.analyze.presentation import AnalyzeViewModel
from src.gimap.integrations.state import (
    InMemoryInstrumentProfileRepository,
    InMemorySettingsRepository,
)
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile

SHAPE = (120, 400)
CENTER_INDEX = 200.3
"""Symmetry axis of the synthetic band, as a column index (canonical x = index + 0.5)."""


def _symmetric_frame() -> np.ndarray:
    columns = np.arange(SHAPE[1], dtype=np.float64)
    offset = columns - CENTER_INDEX
    profile = 1000.0 * np.exp(-((offset / 25.0) ** 2)) + 80.0 * np.cos(offset / 9.0) ** 2 + 20.0
    return np.tile(profile, (SHAPE[0], 1)).astype(np.float32)


class FakeFrames:
    """Frame source over in-memory arrays; ``counts`` gives multi-frame files."""

    def __init__(self, frames: dict[str, list[np.ndarray]]):
        self.frames = frames
        self.loads: list[tuple[str, int]] = []

    def load(self, path, frame_index=0):
        self.loads.append((Path(path).name, int(frame_index)))
        data = self.frames[Path(path).name][int(frame_index)]
        return SimpleNamespace(data=data, mask=None, detector_name="Test", metadata={})

    def frame_count(self, path) -> int:
        return len(self.frames[Path(path).name])

    def expand(self, paths):
        return [Path(path) for path in paths]

    def settled_size(self, path) -> int:
        return 1


def _geometry() -> DetectorGeometry:
    # Small-angle set-up: 100 µm pixels, 2 m, beam centre near the band's axis.
    return DetectorGeometry(100e-6, 100e-6, 2.0, CENTER_INDEX + 0.5 - 4.0, 110.0, 1.0, 0.3)


def _view_model(frames: FakeFrames, settings=None) -> AnalyzeViewModel:
    profiles = InMemoryInstrumentProfileRepository(
        [InstrumentProfile("Test", _geometry(), "Test", SHAPE)]
    )
    settings = settings if settings is not None else InMemorySettingsRepository({})
    return AnalyzeViewModel(
        analyze_frame=AnalyzeFrame(frames, profiles),
        export_analysis=ExportAnalysis(CsvCurveWriter()),
        save_profile=SaveInstrumentProfile(profiles),
        frames=frames,
        profiles=profiles,
        read_setting=settings.get,
        settings=settings,
    )


# -- domain ------------------------------------------------------------------------------


def test_gap_guard_grows_invalid_pixels_but_not_the_frame_border() -> None:
    valid = np.ones((9, 9), dtype=bool)
    valid[4, 4] = False
    guarded = guard_invalid(valid, 2)
    assert not guarded[2:7, 2:7].any()
    assert guarded[1, 4] and guarded[4, 7]
    assert guard_invalid(valid, 0) is valid
    assert guard_invalid(np.ones((5, 5), dtype=bool), 3).all()  # the border is no gap
    with pytest.raises(ValueError):
        guard_invalid(valid, 21)


def test_sum_frames_adds_pixels_valid_in_every_frame() -> None:
    first, second = np.full((3, 3), 2.0), np.full((3, 3), 5.0)
    valid_first = np.ones((3, 3), dtype=bool)
    valid_second = valid_first.copy()
    valid_second[0, 0] = False
    total, valid = sum_frames([(first, valid_first), (second, valid_second)])
    assert total[1, 1] == pytest.approx(7.0)
    assert not valid[0, 0] and np.isnan(total[0, 0])
    with pytest.raises(ValueError, match="Cannot sum"):
        sum_frames([(first, valid_first), (np.ones((2, 2)), np.ones((2, 2), dtype=bool))])


def test_symmetry_finds_the_mirror_axis_across_a_detector_gap() -> None:
    image = _symmetric_frame()
    valid = np.ones(SHAPE, dtype=bool)
    valid[:, 250:257] = False  # a module gap on one side only
    result = symmetric_center_x(image, valid, (40, 45), initial_x_px=CENTER_INDEX + 0.5 - 6.0)
    assert result.x_px == pytest.approx(CENTER_INDEX + 0.5, abs=0.05)
    assert result.loss_after < result.loss_before
    assert result.shift_px == pytest.approx(6.0, abs=0.05)
    with pytest.raises(ValueError, match="structure"):
        symmetric_center_x(np.ones(SHAPE), valid, (40, 45), 200.0)


# -- application -------------------------------------------------------------------------


def test_summed_frames_are_added_before_reducing_and_named_in_the_export(tmp_path: Path) -> None:
    frames = FakeFrames({"run.nxs": [_symmetric_frame(), 2 * _symmetric_frame(), _symmetric_frame()]})
    profiles = InMemoryInstrumentProfileRepository([InstrumentProfile("Test", _geometry(), "Test", SHAPE)])
    analyze = AnalyzeFrame(frames, profiles)
    path = tmp_path / "run.nxs"
    request = AnalysisRequest(path, 0, mode="gisaxs", summed_frames=((path, 1), (path, 2)))
    analysis = analyze(request)
    np.testing.assert_allclose(analysis.data, 4 * _symmetric_frame(), rtol=1e-6)
    assert analysis.frame_total == 3
    assert export_stem(analysis) == "run_frame0000_sum3"

    # The same frames are reused; a different sum loads again.
    loads = len(frames.loads)
    analyze(request, loaded=analysis)
    assert len(frames.loads) == loads
    analyze(AnalysisRequest(path, 0, mode="gisaxs"), loaded=analysis)
    assert len(frames.loads) == loads + 1

    written = ExportAnalysis(CsvCurveWriter())(analysis, tmp_path / "out")
    names = sorted(item.name for item in written)
    assert "run_frame0000_sum3_fit_input.dat" in names
    record = (tmp_path / "out" / "run_frame0000_sum3_analysis.json").read_text("utf-8")
    assert '"summed_frames"' in record and '"frame_index": 2' in record


def test_gap_guard_is_a_correction_applied_after_loading(tmp_path: Path) -> None:
    frame = _symmetric_frame()
    frame[:, 100] = -1.0  # a gap column coded negative, as Pilatus does
    frames = FakeFrames({"a.cbf": [frame]})
    profiles = InMemoryInstrumentProfileRepository([InstrumentProfile("Test", _geometry(), "Test", SHAPE)])
    analyze = AnalyzeFrame(frames, profiles)
    plain = analyze(AnalysisRequest(tmp_path / "a.cbf", mode="gisaxs"))
    guarded = analyze(
        AnalysisRequest(tmp_path / "a.cbf", mode="gisaxs", corrections=Corrections(gap_guard_px=3)),
        loaded=plain,
    )
    assert plain.valid[:, 97].all() and not guarded.valid[:, 97:104].any()
    assert guarded.valid[:, 96].all()
    assert len(frames.loads) == 1


# -- view model --------------------------------------------------------------------------


def test_sum_uses_following_frames_of_the_same_kind(tmp_path: Path) -> None:
    names = ["a.cbf", "b.cbf", "c.cbf", "d.tif", "e.cbf", "run.nxs"]
    frames = FakeFrames({name: [_symmetric_frame()] for name in names[:-1]})
    frames.frames["run.nxs"] = [_symmetric_frame()] * 5
    model = _view_model(frames)
    paths = [tmp_path / name for name in names]
    model.add_paths(paths)
    a, b, c, _d, _e, run = paths
    assert model.summed_frames_for(a) == ()
    model.set_sum_count(3)
    assert model.summed_frames_for(a) == ((b, 0), (c, 0))
    assert model.summed_frames_for(b) == ((c, 0),)  # the TIFF ends the CBF run
    assert model.summed_frames_for(run, 3) == ((run, 4),)
    assert model.request_for(a).summed_frames == ((b, 0), (c, 0))

    model.set_sum_count(2)
    groups = [(r.path.name, r.frame_index, [(p.name, i) for p, i in r.summed_frames]) for r in model.batch_requests()]
    assert groups == [
        ("a.cbf", 0, [("b.cbf", 0)]),
        ("c.cbf", 0, []),
        ("d.tif", 0, []),
        ("e.cbf", 0, []),
        ("run.nxs", 0, [("run.nxs", 1)]),
        ("run.nxs", 2, [("run.nxs", 3)]),
        ("run.nxs", 4, []),
    ]


def test_batch_covers_every_frame_and_watching_waits_for_complete_groups(tmp_path: Path) -> None:
    frames = FakeFrames({f"f{i}.cbf": [_symmetric_frame()] for i in range(5)})
    frames.frames["run.nxs"] = [_symmetric_frame()] * 3
    model = _view_model(frames)
    model.add_paths([tmp_path / "run.nxs"])
    assert [r.frame_index for r in model.batch_requests()] == [0, 1, 2]

    model.set_sum_count(2)
    assert model.watch_requests([tmp_path / "f0.cbf"]) == []
    ready = model.watch_requests([tmp_path / "f1.cbf", tmp_path / "f2.cbf"])
    assert [(r.path.name, [p.name for p, _ in r.summed_frames]) for r in ready] == [("f0.cbf", ["f1.cbf"])]
    ready = model.watch_requests([tmp_path / "f3.cbf"])
    assert [(r.path.name, [p.name for p, _ in r.summed_frames]) for r in ready] == [("f2.cbf", ["f3.cbf"])]


def test_gap_guard_is_remembered_and_survives_resetting_corrections() -> None:
    settings = InMemorySettingsRepository({})
    model = _view_model(FakeFrames({}), settings)
    assert model.state.corrections.gap_guard_px == 3  # as Cut & Fitting did
    model.set_gap_guard(5)
    model.set_valid_range(0.0, 10.0)
    model.reset_corrections()
    assert model.state.corrections == Corrections(gap_guard_px=5)
    assert settings.get("analyze", "gap_guard_px") == 5
    assert _view_model(FakeFrames({}), settings).state.corrections.gap_guard_px == 5


def test_refine_center_moves_the_session_centre_to_the_symmetry_axis(tmp_path: Path) -> None:
    frames = FakeFrames({"a.cbf": [_symmetric_frame()]})
    model = _view_model(frames)
    model.add_paths([tmp_path / "a.cbf"])
    model.select(0)
    model.set_mode("gisaxs")
    model.set_horizontal_band(40.0, 45.0)
    model.accept(model.analyze(model.request()))
    result = model.refine_center_x()
    assert result.x_px == pytest.approx(CENTER_INDEX + 0.5, abs=0.05)
    assert model.state.beam_center == pytest.approx((result.x_px, _geometry().beam_center_y_px))
    assert model.center_state()["overridden"]


# -- a session centre belongs to the detector it was set on ----------------------------------------------


def test_a_session_centre_set_for_another_frame_shape_is_not_used() -> None:
    from src.gimap.features.analyze.application import ResolveGeometry

    resolve = ResolveGeometry(InMemoryInstrumentProfileRepository([InstrumentProfile("Test", _geometry(), "Test", SHAPE)]))
    common = dict(detector_name="Test", shape=SHAPE, beam_center=(150.0, 60.0))
    same = resolve(**common, beam_center_shape=SHAPE)
    assert same.center_source == "session" and same.geometry.beam_center_x_px == 150.0
    assert same.ignored_center_shape is None
    anywhere = resolve(**common)  # no shape known (a settings file): every frame
    assert anywhere.center_source == "session"
    other = resolve(**common, beam_center_shape=(1043, 981))
    assert other.center_source == "profile" and other.ignored_center_shape == (1043, 981)
    profile = _geometry()
    assert (other.geometry.beam_center_x_px, other.geometry.beam_center_y_px) == (
        profile.beam_center_x_px, profile.beam_center_y_px)
    header = resolve(**common, beam_center_shape=(1043, 981), use_header_center=True, header_center=(190.0, 100.0))
    assert header.center_source == "header" and header.geometry.beam_center_x_px == 190.0


def test_the_frame_says_when_the_session_centre_was_left_out(tmp_path: Path) -> None:
    frames = FakeFrames({"a.cbf": [_symmetric_frame()]})
    analyze = AnalyzeFrame(frames, InMemoryInstrumentProfileRepository([InstrumentProfile("Test", _geometry(), "Test", SHAPE)]))
    request = AnalysisRequest(tmp_path / "a.cbf", mode="gisaxs", beam_center=(150.0, 60.0), beam_center_shape=[1043, 981])
    assert request.beam_center_shape == (1043, 981)
    analysis = analyze(request)
    assert analysis.resolution.center_source == "profile"
    assert analysis.messages[0] == (
        "The beam centre you set for 1043×981 frames is not used for this 120×400 frame (profile centre used).")
    assert analysis.detected_kind == "gisaxs"
    kept = analyze(AnalysisRequest(tmp_path / "a.cbf", mode="gisaxs", beam_center=(150.0, 60.0), beam_center_shape=SHAPE),
                   loaded=analysis)
    assert kept.resolution.center_source == "session" and not any("beam centre" in text for text in kept.messages)


def test_requests_carry_the_shape_the_session_centre_was_set_on(tmp_path: Path) -> None:
    frames = FakeFrames({"a.cbf": [_symmetric_frame()], "b.cbf": [_symmetric_frame()]})
    model = _view_model(frames)
    model.add_paths([tmp_path / "a.cbf", tmp_path / "b.cbf"])
    model.select(0)
    assert model.request().beam_center_shape is None
    model.accept(model.analyze(model.request()))
    model.set_beam_center(150.0, 60.0)
    assert model.state.beam_center_shape == SHAPE
    assert model.request().beam_center_shape == SHAPE
    assert all(request.beam_center_shape == SHAPE for request in model.batch_requests())  # Batch Export too
    # One record of the shape: the state and the requests always say the same.
    model.state.beam_center = (151.0, 61.0)  # a centre from a settings file: no shape known, every frame
    assert model.state.beam_center_shape is None and model.request().beam_center_shape is None
    model.state.beam_center = (150.0, 60.0)  # Undo brings the centre back, with its shape
    assert model.state.beam_center_shape == SHAPE and model.request().beam_center_shape == SHAPE
    assert model.session_center_shape() == SHAPE
    model.set_beam_center(152.0, 62.0, shape=[1043, 981])  # a centre loaded with the shape it was set on
    assert model.state.beam_center_shape == (1043, 981) and model.request().beam_center_shape == (1043, 981)
    model.set_beam_center(153.0, 63.0, shape=None)  # … or without one: every frame
    assert model.state.beam_center_shape is None and model.request().beam_center_shape is None
    model.clear_beam_center()
    assert model.state.beam_center_shape is None and model.request().beam_center_shape is None


@pytest.mark.skipif(
    not (Path(__file__).parent / "data" / "external" / "giwaxs_p08_mapi" / "S121_MAI_A2_00841.tif").is_file(),
    reason="public test frames (tests/data/external) are not available",
)
def test_a_centre_set_on_one_detector_is_not_applied_to_another() -> None:
    from tests.test_analyze_workspace import GALAXI, P08, _done, _public_page

    page = _public_page()
    page.add_paths([str(GALAXI)])
    _done(page)
    page.set_session_center(596.3, 719.6)
    _done(page)
    geometry = page.view_model.state.analysis.geometry
    assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == (596.3, 719.6)
    assert page.view_model.state.beam_center_shape == (1043, 981)

    page.add_paths([str(P08)])
    _done(page, 240)
    analysis = page.view_model.state.analysis
    assert analysis.path.name == P08.name
    assert (analysis.geometry.beam_center_x_px, analysis.geometry.beam_center_y_px) == pytest.approx((545.5, 1826.9), abs=0.1)
    assert analysis.resolution.center_source != "session"
    assert page.status_level() == "warning"
    assert "1043×981" in page.status_text() and "2048×2048" in page.status_text()
    assert page.center_button.property("centerSource") == "profile"

    page.file_list.setCurrentRow(0)
    _done(page)
    analysis = page.view_model.state.analysis
    assert analysis.path.name == GALAXI.name and analysis.resolution.center_source == "session"
    assert (analysis.geometry.beam_center_x_px, analysis.geometry.beam_center_y_px) == (596.3, 719.6)
    page.dispose()
    page.close()
