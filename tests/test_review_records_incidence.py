"""αi of the last session: used only with the instrument profile it was set with, never written into a profile,
and offered as a suggestion when the automatic analysis asks for αi."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from src.gimap.features.assistant.application import PipelineOptions, StandardPipeline
from src.gimap.features.assistant.presentation.guided_words import SENTENCES
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from tests.test_analyze_workspace import _context, _done, _page
from tests.test_assistant_calibration import CENTER, DISTANCE_M, PIXEL, SHAPE, WAVELENGTH, save_tiff
from tests.test_assistant_pipeline import SeriesWorkbench, decision, setup
from tests.test_series_map import _ring_frame


def _profiles(*names: str) -> InMemoryInstrumentProfileRepository:
    geometry = DetectorGeometry(PIXEL, PIXEL, DISTANCE_M, CENTER[0], CENTER[1], WAVELENGTH, 0.2)
    return InMemoryInstrumentProfileRepository([InstrumentProfile(name, geometry, None, SHAPE) for name in names])


def _frame(folder: Path) -> str:
    return str(save_tiff(folder / "film_0.tif", _ring_frame(200.0, seed=1)))


def _toasts(monkeypatch) -> list:
    from src.gimap.features.analyze.presentation.bindings import incidence

    notices = []
    monkeypatch.setattr(incidence, "show_toast", lambda parent, text, **options: notices.append((text, options)))
    return notices


def _close(page) -> None:
    page.dispose()
    page.close()


def test_the_last_sessions_alpha_i_is_not_used_with_another_profile(tmp_path: Path, monkeypatch) -> None:
    notices = _toasts(monkeypatch)
    profiles = _profiles("first detector")
    context = _context(profiles)
    first = _page(context)
    first.add_paths([_frame(tmp_path)])
    _done(first)
    first.incidence_spin.setValue(0.075)
    _done(first)
    assert first._incidence_profile == "first detector"
    _close(first)

    profile = profiles.load_all()[0]
    profiles.delete("first detector")
    profiles.save(InstrumentProfile("second detector", profile.geometry, None, SHAPE))
    second = _page(context)
    assert second.view_model.state.incidence_deg == pytest.approx(0.075)  # restored, not yet matched to a frame
    second.add_paths([_frame(tmp_path)])
    _done(second)
    state = second.view_model.state
    assert state.incidence_deg is None and state.analysis.geometry.incidence_deg == pytest.approx(0.2)
    assert second.incidence_spin.property("override") is False
    assert second.automation().status()["incidence_override"] is None
    assert second.automation().status()["incidence_suggestion"] == pytest.approx(0.075)
    assert len(notices) == 1 and "is not used" in notices[0][0] and "0.075" in notices[0][0]
    assert notices[0][1]["action"][0] == "Use 0.075°"
    notices[0][1]["action"][1]()  # one step back to the old value, now this session's own
    _done(second)
    assert second.view_model.state.analysis.geometry.incidence_deg == pytest.approx(0.075)
    assert second._incidence_profile == "second detector"
    assert second.automation().status()["incidence_suggestion"] is None
    second.incidence_reset_action.trigger()  # Back to Profile αi: the profile's own value
    _done(second)
    assert second.view_model.state.analysis.geometry.incidence_deg == pytest.approx(0.2)
    _close(second)


def test_the_last_sessions_alpha_i_is_kept_with_its_own_profile(tmp_path: Path, monkeypatch) -> None:
    notices = _toasts(monkeypatch)
    context = _context(_profiles("first detector"))
    first = _page(context)
    first.add_paths([_frame(tmp_path)])
    _done(first)
    first.incidence_spin.setValue(0.3)
    _done(first)
    _close(first)
    second = _page(context)
    second.add_paths([_frame(tmp_path)])
    _done(second)
    assert second.view_model.state.analysis.geometry.incidence_deg == pytest.approx(0.3)
    assert [text for text, _options in notices] == ["αi 0.3° from your last session (profile: 0.2°)"]
    _close(second)


def test_a_frame_without_a_profile_drops_the_last_sessions_alpha_i(tmp_path: Path, monkeypatch) -> None:
    notices = _toasts(monkeypatch)
    profiles = _profiles("first detector")
    context = _context(profiles)
    first = _page(context)
    first.add_paths([_frame(tmp_path)])
    _done(first)
    first.incidence_spin.setValue(0.075)
    _done(first)
    _close(first)
    profiles.delete("first detector")
    second = _page(context)
    second.add_paths([_frame(tmp_path)])
    _done(second)
    assert second.view_model.state.incidence_deg is None
    assert second.automation().status()["incidence_suggestion"] == pytest.approx(0.075)
    assert len(notices) == 1 and "no instrument profile" in notices[0][0]
    _close(second)


class _Carried(SeriesWorkbench):
    suggestion, override = None, None

    def status(self) -> dict:
        status = super().status()
        status.update(incidence_suggestion=self.suggestion, incidence_override=self.override)
        return status


def _carried_setup(tmp_path: Path, **values):
    catalog, workbench, _calibrator = setup(tmp_path, frames=1)
    workbench.__class__ = _Carried
    for name, value in values.items():
        setattr(workbench, name, value)
    return catalog, workbench


def test_the_run_asks_for_alpha_i_and_offers_the_last_sessions_value(tmp_path: Path) -> None:
    catalog, workbench = _carried_setup(tmp_path, suggestion=0.075)
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4)).run()
    used = next(item for item in workbench.calls if item[0] == "use_geometry")
    assert "incidence_deg" not in used[1]  # not written into the new detector's profile
    question = next(item for item in report["needs_attention"] if item["option"] == "incidence_deg")
    assert "Your last session used αi = 0.075° with another instrument profile" in question["hint"]
    assert any(form.startswith("Your last session used αi = {value}°") for form in SENTENCES)
    assert not any(item["what"] == "incidence angle" for item in report["decisions"])


def test_a_command_bar_alpha_i_is_used_but_not_saved_in_a_profile(tmp_path: Path) -> None:
    catalog, workbench = _carried_setup(tmp_path, override=0.3)
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4)).run()
    used = next(item for item in workbench.calls if item[0] == "use_geometry")
    assert "incidence_deg" not in used[1]
    assert decision(report, "incidence angle")["decision"] == "αi = 0.3°"
    assert not [item for item in report["needs_attention"] if item["option"] == "incidence_deg"]

    catalog, workbench = _carried_setup(tmp_path / "given")
    StandardPipeline(catalog, PipelineOptions(energy_kev=12.4, incidence_deg=0.2)).run()
    used = next(item for item in workbench.calls if item[0] == "use_geometry")
    assert used[1]["incidence_deg"] == pytest.approx(0.2)  # a value given for this frame is the profile's
