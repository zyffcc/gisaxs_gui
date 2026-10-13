"""Instrument profiles: canonical geometry + detector recognition, stored as JSON."""

from __future__ import annotations

import json

import pytest

from src.gimap.integrations.state import JsonInstrumentProfileRepository
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile, best_profile


def _geometry(**overrides) -> DetectorGeometry:
    values = dict(
        pixel_size_x_m=172e-6,
        pixel_size_y_m=172e-6,
        distance_m=4.2,
        beam_center_x_px=737.5,
        beam_center_y_px=1400.5,
        wavelength_angstrom=1.0332,
        incidence_deg=0.4,
    )
    values.update(overrides)
    return DetectorGeometry(**values)


def _profile(name: str, **overrides) -> InstrumentProfile:
    values = dict(
        name=name,
        geometry=_geometry(),
        detector_name="PILATUS 2M",
        detector_shape=(1679, 1475),
        source="calibration: AgBh",
        updated_at="2026-09-27T20:00:00+00:00",
    )
    values.update(overrides)
    return InstrumentProfile(**values)


def test_profile_round_trips_through_dict() -> None:
    profile = _profile("P03 GISAXS")
    assert InstrumentProfile.from_dict(profile.to_dict()) == profile


def test_matching_prefers_name_and_shape_and_rejects_other_shapes() -> None:
    both = _profile("both")
    shape_only = _profile("shape", detector_name="Eiger 1M")
    wrong_shape = _profile("wrong", detector_shape=(1043, 981))

    assert both.match_score(detector_name="PILATUS 2M, S/N 24-0104", shape=(1679, 1475)) == 3
    assert shape_only.match_score(detector_name="PILATUS 2M", shape=(1679, 1475)) == 1
    assert wrong_shape.match_score(detector_name="PILATUS 2M", shape=(1679, 1475)) == 0
    assert best_profile(
        [shape_only, wrong_shape, both], detector_name="pilatus  2m", shape=(1679, 1475)
    ) is both


def test_newest_profile_wins_a_tie() -> None:
    older = _profile("older", updated_at="2026-01-01T00:00:00+00:00")
    newer = _profile("newer", updated_at="2026-09-01T00:00:00+00:00")
    assert best_profile([newer, older], detector_name="PILATUS 2M", shape=(1679, 1475)) is newer


def test_repository_saves_replaces_matches_and_deletes(tmp_path) -> None:
    repository = JsonInstrumentProfileRepository(tmp_path / "config" / "instrument_profiles.json")
    assert repository.load_all() == []

    repository.save(_profile("P03 GISAXS"))
    repository.save(_profile("P03 GIWAXS", detector_name="LAMBDA 9M", detector_shape=(516, 1556)))
    moved = _profile("P03 GISAXS").updated(_geometry(distance_m=4.5), source="manual")
    repository.save(moved)

    names = [item.name for item in repository.load_all()]
    assert names == ["P03 GISAXS", "P03 GIWAXS"]
    assert repository.find("P03 GISAXS").geometry.distance_m == 4.5
    assert repository.match(detector_name="LAMBDA 9M", shape=(516, 1556)).name == "P03 GIWAXS"
    assert repository.match(detector_name="unknown", shape=(10, 10)) is None
    assert repository.delete("P03 GIWAXS") is True
    assert repository.delete("P03 GIWAXS") is False

    stored = json.loads(repository.path.read_text(encoding="utf-8"))
    assert stored["schema_version"] == 1
    assert stored["profiles"][0]["geometry"]["distance_m"] == 4.5


def test_repository_refuses_files_from_a_newer_schema(tmp_path) -> None:
    path = tmp_path / "instrument_profiles.json"
    path.write_text(json.dumps({"schema_version": 99, "profiles": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="newer GIMaP"):
        JsonInstrumentProfileRepository(path).load_all()


def test_profiles_need_a_name_and_a_valid_shape() -> None:
    with pytest.raises(ValueError, match="name"):
        _profile("  ")
    with pytest.raises(ValueError, match="shape"):
        _profile("x", detector_shape=(0, 10))
