from dataclasses import dataclass
from pathlib import Path

from src.gimap.app import AppContext, ProjectState
from src.gimap.integrations.state import (
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
    JsonSessionRepository,
    JsonSettingsRepository,
    JsonProjectParametersRepository,
)
from src.gimap.app import LoadProjectParameters, SaveProjectParameters


@dataclass
class ExampleFeatureState:
    selected_file: str = ""
    counter: int = 0

    def snapshot(self) -> dict:
        return {
            "selected_file": self.selected_file,
            "counter": self.counter,
        }

    def restore(self, state: dict) -> None:
        self.selected_file = str(state.get("selected_file", ""))
        self.counter = int(state.get("counter", 0))


def test_json_settings_preserve_legacy_user_parameter_shape(tmp_path: Path) -> None:
    path = tmp_path / "user_parameters.json"
    repository = JsonSettingsRepository(
        path,
        initial={
            "beam": {"wavelength": 0.1},
            "fitting": {"detector": {"distance": 2000.0}},
        },
    )
    repository.set("fitting", "detector.distance", 1456.7)
    repository.set("beam", "energy_kev", 12.0)
    repository.save()

    restored = JsonSettingsRepository(path)

    assert restored.get("fitting", "detector.distance") == 1456.7
    assert restored.get("beam", "energy_kev") == 12.0
    assert set(restored.snapshot()) == {"beam", "fitting"}
    assert "settings" not in restored.snapshot()


def test_in_memory_settings_reset_restores_injected_defaults() -> None:
    repository = InMemorySettingsRepository({"beam": {"wavelength": 0.015}})
    repository.set("beam", "wavelength", 0.02)
    repository.set("fitting", "detector.distance", 1600.0)

    repository.reset()

    assert repository.snapshot() == {"beam": {"wavelength": 0.015}}


def test_in_memory_user_preferences_preserve_flat_legacy_keys() -> None:
    repository = InMemoryUserPreferencesRepository(
        {"fit.points_num": 50, "ai_fitting": {"profile": "Balanced"}}
    )

    repository.set("fit.points_num", 80)
    repository.save()

    assert repository.get("fit.points_num") == 80
    assert repository.get("ai_fitting") == {"profile": "Balanced"}
    assert repository.snapshot() == {
        "fit.points_num": 80,
        "ai_fitting": {"profile": "Balanced"},
    }


def test_app_context_persists_project_and_registered_feature_state(tmp_path: Path) -> None:
    session = JsonSessionRepository(tmp_path / "session.json")
    first = AppContext(
        settings=InMemorySettingsRepository(),
        session=session,
        preferences=InMemoryUserPreferencesRepository(),
        project_state=ProjectState(project_path="project.gimap", dirty=True),
    )
    feature = first.project_state.feature_state("example", ExampleFeatureState)
    feature.selected_file = "image.nxs"
    feature.counter = 3
    first.save_session()

    second = AppContext(
        settings=InMemorySettingsRepository(),
        session=session,
        preferences=InMemoryUserPreferencesRepository(),
    )
    assert second.restore_session()
    restored = second.project_state.feature_state("example", ExampleFeatureState)

    assert second.project_state.project_path == "project.gimap"
    assert second.project_state.dirty
    assert restored.selected_file == "image.nxs"
    assert restored.counter == 3


def test_project_parameter_commands_preserve_legacy_json_shape(tmp_path: Path) -> None:
    repository = JsonProjectParametersRepository()
    save = SaveProjectParameters(repository)
    load = LoadProjectParameters(repository)
    path = tmp_path / "project-parameters.json"
    values = {
        "trainset": {"samples": 10},
        "fitting": {"points_num": 50},
        "fitting_model_parameters": {"fitting": {"BG": 0.1}},
    }

    assert save.execute(path, values) == path
    assert load.execute(path) == values
    assert path.read_text(encoding="utf-8").startswith("{\n    \"trainset\"")


def test_user_store_keeps_every_section_and_merges_defaults(tmp_path: Path) -> None:
    import json

    from src.gimap.integrations.state import StoreSettingsRepository, UserStore

    path = tmp_path / "settings.json"
    settings = StoreSettingsRepository(UserStore(path))
    # Built-in defaults are present before anything was saved.
    assert settings.get("beam", "wavelength") == 0.1
    settings.set("preprocessing", "focus_region.qr_max", 2.0)
    settings.update_section("analyze", {"mode": "giwaxs"})  # not a built-in section
    settings.save()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    restored = StoreSettingsRepository(UserStore(path))
    assert restored.get("preprocessing", "focus_region.qr_max") == 2.0
    # Nested defaults survive a partial section on disk (deep merge).
    assert restored.get("preprocessing", "focus_region.qr_min") == 0.01
    assert restored.get_section("analyze") == {"mode": "giwaxs"}

    restored.reset()
    assert restored.get("preprocessing", "focus_region.qr_max") == 3.0
    assert restored.get_section("analyze") == {}


def test_preferences_share_the_store_and_drop_scaling_keys(tmp_path: Path) -> None:
    import json

    from src.gimap.integrations.state import migrate_legacy_files
    from src.gimap.integrations.state import StorePreferencesRepository, StoreSettingsRepository
    from src.gimap.integrations.state import UserStore

    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    legacy_parameters = project / "config" / "user_parameters.json"
    legacy_preferences = project / "config" / "user_settings.json"
    legacy_parameters.write_text(
        json.dumps({"fitting": {"detector": {"distance": 1456.7}}, "classification": {"k": 3}}),
        encoding="utf-8",
    )
    legacy_preferences.write_text(
        json.dumps({"fit.points_num": 80, "visual_font_scale": 120, "window_width": 1400}),
        encoding="utf-8",
    )
    before = (legacy_parameters.read_bytes(), legacy_preferences.read_bytes())
    data_dir = tmp_path / "home"

    imported = migrate_legacy_files(data_dir, project_root=project)

    assert sorted(Path(item).name for item in imported) == ["user_parameters.json", "user_settings.json"]
    assert (legacy_parameters.read_bytes(), legacy_preferences.read_bytes()) == before
    store = UserStore(data_dir / "settings.json")
    settings = StoreSettingsRepository(store)
    preferences = StorePreferencesRepository(store)
    assert settings.get("fitting", "detector.distance") == 1456.7
    assert settings.get_section("classification") == {"k": 3}
    assert preferences.get("fit.points_num") == 80
    assert preferences.get("visual_font_scale") is None
    assert store.migrated_from["files"]
    # A second start imports nothing and keeps the user's later changes.
    settings.set("fitting", "detector.distance", 2000.0)
    settings.save()
    assert migrate_legacy_files(data_dir, project_root=project) == []
    assert StoreSettingsRepository(UserStore(data_dir / "settings.json")).get(
        "fitting", "detector.distance"
    ) == 2000.0


def test_create_app_context_uses_one_user_folder(tmp_path: Path) -> None:
    from src.gimap.app.bootstrap import create_app_context

    context = create_app_context(data_dir=tmp_path, restore_session=False)
    context.settings.set("beam", "energy_kev", 12.0)
    context.preferences.set("fit.points_num", 64)
    context.settings.save()
    context.save_session()
    context.jobs.shutdown()

    assert context.data_dir == tmp_path
    assert (tmp_path / "settings.json").is_file()
    assert (tmp_path / "session.json").is_file()
    again = create_app_context(data_dir=tmp_path, restore_session=False)
    assert again.settings.get("beam", "energy_kev") == 12.0
    assert again.preferences.get("fit.points_num") == 64
    again.jobs.shutdown()
