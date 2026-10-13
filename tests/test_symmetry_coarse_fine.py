"""Global centre discovery and subpixel refinement preserve measured data."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.gimap.features.analyze.domain.symmetry import symmetric_center_x
from tests.test_analyze_frames_and_center import CENTER_INDEX, SHAPE, _symmetric_frame, _view_model, FakeFrames


@pytest.mark.parametrize("initial", [35.5, 370.5, -150.0])
def test_coarse_search_recovers_an_axis_far_from_the_initial_centre(initial) -> None:
    image = _symmetric_frame()
    valid = np.ones(SHAPE, dtype=bool)
    valid[:, 250:257] = False
    result = symmetric_center_x(image, valid, (40, 45), initial)

    assert result.x_px == pytest.approx(CENTER_INDEX + 0.5, abs=0.01)
    assert result.coarse_x_px == pytest.approx(CENTER_INDEX + 0.5, abs=1.0)
    assert result.initial_x_px == initial
    assert result.tolerance_px == 0.0001
    assert result.to_dict()["coarse_x_px"] == result.coarse_x_px
    json.dumps(result.to_dict(), allow_nan=False)


def test_a_beamstop_and_one_sided_gap_do_not_replace_missing_counts() -> None:
    image = _symmetric_frame()
    valid = np.ones(SHAPE, dtype=bool)
    valid[:, 187:215] = False
    valid[:, 250:257] = False
    original = image.copy()
    original_valid = valid.copy()
    result = symmetric_center_x(image, valid, (40, 45), 40.0)
    damaged = image.copy()
    damaged[~valid] = 1e12
    again = symmetric_center_x(damaged, valid, (40, 45), 40.0)

    assert result.x_px == pytest.approx(CENTER_INDEX + 0.5, abs=0.02)
    assert again.x_px == pytest.approx(result.x_px, abs=1e-8)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(valid, original_valid)


def test_session_refinement_preserves_y_and_keeps_the_profile_unchanged(tmp_path) -> None:
    model = _view_model(FakeFrames({"synthetic.cbf": [_symmetric_frame()]}))
    model.add_paths([tmp_path / "synthetic.cbf"])
    model.select(0)
    model.set_mode("gisaxs")
    model.set_horizontal_band(40.0, 45.0)
    model.set_beam_center(35.5, 100.25, shape=SHAPE)
    model.accept(model.analyze(model.request()))
    profile_before = model.state.analysis.resolution.profile.to_dict()

    found = model.find_center_x()
    assert model.state.beam_center == (35.5, 100.25)  # discovery alone changes no settings
    result = model.refine_center_x()

    assert result.x_px == pytest.approx(found.x_px, abs=1e-8)
    assert model.state.beam_center == pytest.approx((CENTER_INDEX + 0.5, 100.25), abs=0.01)
    assert model.state.analysis.resolution.profile.to_dict() == profile_before


def test_cancellation_and_invalid_data_cannot_apply_a_new_centre() -> None:
    image = _symmetric_frame()
    valid = np.ones(SHAPE, dtype=bool)
    with pytest.raises(ValueError, match="cancelled"):
        symmetric_center_x(image, valid, (40, 45), 35.5, cancelled=lambda: True)
    with pytest.raises(ValueError, match="structure"):
        symmetric_center_x(np.ones(SHAPE), valid, (40, 45), 35.5)
    with pytest.raises(ValueError, match="mask"):
        symmetric_center_x(image, valid[:, :-1], (40, 45), 35.5)
    with pytest.raises(ValueError, match="finite"):
        symmetric_center_x(image, valid, (40, 45), np.nan)


def test_assistant_refinement_records_and_undoes_a_centre_without_an_initial_score(tmp_path) -> None:
    from tests.assistant_fakes import call
    from tests.test_assistant_operations import catalog

    model = _view_model(FakeFrames({"synthetic.cbf": [_symmetric_frame()]}))
    model.add_paths([tmp_path / "synthetic.cbf"])
    model.select(0)
    model.set_mode("gisaxs")
    model.set_horizontal_band(40.0, 45.0)
    model.set_beam_center(35.5, 100.25, shape=SHAPE)
    model.accept(model.analyze(model.request()))

    class Workbench:
        def status(self):
            return {"measurement": "gisaxs", "gisaxs": {"horizontal_rows": [40, 45]},
                    "geometry": {"beam_center_px": list(model.state.beam_center)}}

        def symmetry_center(self):
            return model.find_center_x().to_dict()

        def set_beam_center(self, x, y):
            model.set_beam_center(x, y)
            return self.status()

    tools = catalog(Workbench())
    outcome = tools.execute(call("refine_beam_center_symmetry"))

    assert not outcome.is_error
    assert outcome.data["loss_before"] is None
    assert outcome.data["tolerance_px"] == 0.0001
    assert "unavailable" in outcome.summary
    assert model.state.beam_center == pytest.approx((CENTER_INDEX + 0.5, 100.25), abs=0.01)
    json.dumps(outcome.data, allow_nan=False)
    assert tools.undo_all_operations() == 1
    assert model.state.beam_center == (35.5, 100.25)
