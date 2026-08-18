from __future__ import annotations

from pathlib import Path

from src.gimap.app.fitting_session import FittingSessionCoordinator
from src.gimap.app.workspace_parameters import WorkspaceParameterCoordinator


class FakeSettings:
    def __init__(self, session=None):
        self.values = {"fitting": {"last_session": session or {}}}
        self.saved = False

    def get(self, section, key, default=None):
        return self.values.get(section, {}).get(key, default)

    def set(self, section, key, value):
        self.values.setdefault(section, {})[key] = value

    def save(self):
        self.saved = True


class FakeFeatureRuntime:
    def __init__(self, values=None):
        self.values = values or {}
        self.reset_count = 0

    def get_parameters(self):
        return dict(self.values)

    def set_parameters(self, values):
        self.values = dict(values)

    def validate_parameters(self):
        return True, "ok"

    def reset_to_defaults(self):
        self.reset_count += 1


class FakeModelParameters:
    def __init__(self):
        self.values = {"fitting": {"BG": 0.1}}
        self.saved = False

    def get_parameter(self, section, _key, default):
        return self.values.get(section, default)

    def replace_section(self, section, values):
        self.values[section] = dict(values)

    def save_parameters(self):
        self.saved = True


class FakeFittingRuntime(FakeFeatureRuntime):
    def __init__(self, values=None):
        super().__init__(values)
        self.model_params_manager = FakeModelParameters()
        self.reloaded = False
        self.restored = None
        self.session = {}

    def reload_particle_parameters(self):
        self.reloaded = True

    def restore_session(self, session):
        self.restored = dict(session)

    def get_session_data(self):
        return dict(self.session)


class MemoryProjectRepository:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.saved_path = None

    def load(self, _path):
        return dict(self.values)

    def save(self, path, values):
        self.values = dict(values)
        self.saved_path = Path(path)
        return self.saved_path


def _workspace_coordinator(repository=None):
    messages = []
    runtimes = {
        "trainset": FakeFeatureRuntime({"samples": 10}),
        "fitting": FakeFittingRuntime({"points": 50}),
        "classification": FakeFeatureRuntime({"classes": 3}),
        "prediction": FakeFeatureRuntime({"module": "example"}),
    }
    coordinator = WorkspaceParameterCoordinator(
        repository=repository,
        status=messages.append,
        **runtimes,
    )
    return coordinator, runtimes, messages


def test_workspace_parameter_coordinator_preserves_snapshot_shape(tmp_path):
    repository = MemoryProjectRepository()
    coordinator, runtimes, messages = _workspace_coordinator(repository)

    assert coordinator.save(tmp_path / "parameters.json")
    assert repository.values == {
        "trainset": {"samples": 10},
        "fitting": {"points": 50},
        "fitting_model_parameters": {"fitting": {"BG": 0.1}},
        "classification": {"classes": 3},
        "gisaxs_predict": {"module": "example"},
    }
    assert messages[-1].startswith("Parameters saved to")
    assert runtimes["fitting"].model_params_manager.saved is False


def test_workspace_parameter_coordinator_loads_each_public_feature_api():
    repository = MemoryProjectRepository(
        {
            "trainset": {"samples": 20},
            "fitting": {"points": 80},
            "fitting_model_parameters": {"fitting": {"BG": 0.2}},
            "classification": {"classes": 4},
            "gisaxs_predict": {"module": "new"},
        }
    )
    coordinator, runtimes, messages = _workspace_coordinator(repository)

    assert coordinator.load("parameters.json")
    assert runtimes["trainset"].values == {"samples": 20}
    assert runtimes["classification"].values == {"classes": 4}
    assert runtimes["prediction"].values == {"module": "new"}
    assert runtimes["fitting"].model_params_manager.values["fitting"] == {"BG": 0.2}
    assert runtimes["fitting"].model_params_manager.saved
    assert runtimes["fitting"].reloaded
    assert messages[-1].startswith("Parameters loaded from")


def test_workspace_parameter_validation_and_reset_are_feature_neutral():
    coordinator, runtimes, messages = _workspace_coordinator()

    assert [result[0] for result in coordinator.validate()] == [
        "Trainset parameters",
        "Fitting parameters",
        "Classification parameters",
        "GISAXS prediction parameters",
    ]
    coordinator.reset()

    assert all(runtime.reset_count == 1 for runtime in runtimes.values())
    assert messages[-1] == "All parameters have been reset to default values"


def test_fitting_session_coordinator_saves_and_restores_without_global_state():
    settings = FakeSettings()
    fitting = FakeFittingRuntime()
    fitting.session = {"last_opened_file": "image.cbf", "roi": [1, 2, 3, 4]}
    coordinator = FittingSessionCoordinator(settings, fitting)

    coordinator.save()
    coordinator.restore(settings.get("fitting", "last_session"))

    assert settings.saved
    assert fitting.restored == fitting.session
