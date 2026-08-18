from dataclasses import dataclass
from pathlib import Path

from src.gimap.app import AppContext, ProjectState
from src.gimap.integrations.state import (
    GlobalParamsSettingsRepository,
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


class FakeGlobalParams:
    def __init__(self):
        self.values = {"beam": {"wavelength": 0.1}, "fitting": {}}
        self.saved = False

    def get_parameter(self, section, key, default=None):
        current = self.values.get(section, {})
        for segment in key.split("."):
            if not isinstance(current, dict) or segment not in current:
                return default
            current = current[segment]
        return current

    def set_parameter(self, section, key, value):
        current = self.values.setdefault(section, {})
        segments = key.split(".")
        for segment in segments[:-1]:
            current = current.setdefault(segment, {})
        current[segments[-1]] = value

    def get_module_parameters(self, section):
        return dict(self.values.get(section, {}))

    def set_module_parameters(self, section, values):
        self.values.setdefault(section, {}).update(values)

    def get_all_parameters(self):
        return {section: dict(values) for section, values in self.values.items()}

    def save_user_parameters(self):
        self.saved = True

    def load_parameters(self, _path):
        return None


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


def test_global_params_compatibility_adapter_delegates_without_new_singleton() -> None:
    manager = FakeGlobalParams()
    repository = GlobalParamsSettingsRepository(manager)

    repository.set("fitting", "detector.distance", 999.0)
    repository.update_section("beam", {"energy_kev": 10.0})
    repository.save()

    assert repository.get("fitting", "detector.distance") == 999.0
    assert repository.get_section("beam")["energy_kev"] == 10.0
    assert manager.saved


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
