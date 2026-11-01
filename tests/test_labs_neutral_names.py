"""Labs: shipped identifiers describe the method, not whose dataset it came from.

The Trainset simulation engine has a neutral name; a project saved by an earlier version, which wrote the
engine's former name, still loads and is written back under the current name. The Cu prediction modules
have neutral ids (their ids are written into export records; no saved file reads them back).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.gimap.features.prediction.infrastructure import InMemoryModelPathStore, YamlModuleRepository
from src.gimap.features.trainset.infrastructure.adapters.configuration import (
    FORMER_ENGINE_NAMES,
    SIMULATION_ENGINE,
    default_project_config,
    load_project_config,
    save_project_config,
    validate_project_config,
)

MODULES = Path(__file__).resolve().parents[1] / "modules"


def test_the_default_trainset_project_uses_the_neutral_engine_name():
    assert default_project_config()["simulation"]["engine"] == SIMULATION_ENGINE == "bornagain"


@pytest.mark.parametrize("former", sorted(FORMER_ENGINE_NAMES))
def test_a_project_saved_with_the_former_engine_name_loads_and_is_saved_under_the_new_one(tmp_path, former):
    config = default_project_config()
    config["simulation"]["engine"] = former
    saved = save_project_config(config, tmp_path / "earlier_project.yaml")

    loaded = load_project_config(saved)

    assert loaded["simulation"]["engine"] == SIMULATION_ENGINE
    resaved = save_project_config(loaded, tmp_path / "resaved_project.yaml")
    assert former not in resaved.read_text(encoding="utf-8")
    # The BornAgain availability warning still applies to such a project.
    _ok, _errors, warnings = validate_project_config(config, simulation_available=False)
    assert any("BornAgain is not installed" in warning for warning in warnings)


def test_the_copper_modules_have_neutral_ids():
    modules = YamlModuleRepository(MODULES, InMemoryModelPathStore()).discover()
    ids = {module.yaml_path.parent.name: module.id for module in modules}

    assert ids["Cu_Polymer_Silicon_Sputter"] == "CuPolymer_FF_256"
    assert ids["Cu_Polymer_Silicon_Sputter_SF"] == "CuPolymer_SF_4Para"


@pytest.mark.parametrize("former", sorted(FORMER_ENGINE_NAMES))
def test_form_factor_grids_built_before_the_rename_are_still_found(tmp_path, former):
    import copy
    import hashlib
    import json

    from src.gimap.features.trainset.infrastructure.adapters.grid_cache import _payload, cache_path, particle_axes

    config = default_project_config()
    config["simulation"]["grid_cache"] = {"directory": str(tmp_path / "cache")}
    earlier = copy.deepcopy(config)
    earlier["simulation"]["engine"] = former
    assert cache_path(config) == cache_path(earlier)
    # The digest as earlier versions computed it: every simulation key but grid_cache, the engine as written then.
    payload = _payload(earlier, particle_axes(earlier))
    payload["resolution"] = {key: value for key, value in earlier["simulation"].items() if key != "grid_cache"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]
    assert cache_path(config).name == f"form_factor_grid_{digest}.npz"
