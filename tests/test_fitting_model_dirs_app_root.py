"""The 1D AI models are found in the GIMaP folder whatever folder GIMaP was started from."""

from pathlib import Path
from types import SimpleNamespace

from src.gimap.features.fitting.application import AiFittingCatalog
from src.gimap.features.fitting.infrastructure.adapters import AiFittingCatalogAdapter
from src.gimap.features.fitting.infrastructure.adapters.ai_model_registry import (
    APP_ROOT,
    resolve_model_path,
)
from src.gimap.features.fitting.presentation.bindings.ai_settings import AiSettingsMixin

REPO = Path(__file__).resolve().parents[1]


class _Prefs:
    def __init__(self, ai_fitting):
        self._data = {"ai_fitting": ai_fitting}

    def get(self, key, default=None):
        return self._data.get(key, default)


def _owner(ai_fitting):
    owner = AiSettingsMixin()
    owner.preferences = _Prefs(ai_fitting)
    owner.fitting_view_model = SimpleNamespace(
        storage=SimpleNamespace(ai_catalog=AiFittingCatalog(AiFittingCatalogAdapter()))
    )
    return owner


def test_app_root_is_the_folder_of_main_py():
    assert APP_ROOT == REPO
    assert (APP_ROOT / "main.py").is_file()


def test_default_directories_do_not_follow_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    catalog = AiFittingCatalog(AiFittingCatalogAdapter())

    assert catalog.default_model_directories() == (
        REPO / "modules/Fitting_1D_Model",
        REPO / "modules/Fitting_1D_model",
    )


def test_relative_model_paths_are_taken_inside_the_gimap_folder(tmp_path):
    assert resolve_model_path("modules/Fitting_1D_Model") == REPO / "modules/Fitting_1D_Model"
    assert resolve_model_path("models", root=tmp_path) == tmp_path / "models"
    assert resolve_model_path(tmp_path / "abs") == tmp_path / "abs"


def test_stored_settings_resolve_against_the_gimap_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    elsewhere = tmp_path / "my_models"
    owner = _owner({
        "model_base_dirs": ["modules/Fitting_1D_Model", "modules/Fitting_1D_model"],
        "extra_model_paths": ["extra/rel", str(elsewhere)],
    })

    dirs = owner._ai_fitting_base_dirs()

    assert dirs == [
        REPO / "modules/Fitting_1D_Model",
        REPO / "modules/Fitting_1D_model",
        REPO / "extra/rel",
        elsewhere,
    ]
    assert all(Path(d).is_absolute() for d in dirs)


def test_shipped_models_are_found_from_another_working_directory(tmp_path, monkeypatch):
    if not (REPO / "modules/Fitting_1D_Model").is_dir():
        return  # a checkout without the model folder: nothing to find
    monkeypatch.chdir(tmp_path)
    owner = _owner({"model_base_dirs": ["modules/Fitting_1D_Model"]})

    assert owner._scan_ai_fitting_models()
