from pathlib import Path

from src.gimap.features.waxs.application import (
    LoadWaxsConfiguration,
    SaveWaxsConfiguration,
)


def test_configuration_use_cases_delegate_portable_json_values():
    class Repository:
        def __init__(self):
            self.saved = None

        def load(self, path):
            return {"version": 1, "path": str(path)}

        def save(self, path, values):
            self.saved = (path, values)
            return path

    repository = Repository()
    target = Path("waxs-settings.json")

    assert LoadWaxsConfiguration(repository).execute(target)["version"] == 1
    assert SaveWaxsConfiguration(repository).execute(target, {"version": 1}) == target
    assert repository.saved == (target, {"version": 1})
