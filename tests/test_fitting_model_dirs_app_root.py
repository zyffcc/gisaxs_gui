"""The 1D AI models are found in the GIMaP folder whatever folder GIMaP was started from."""

from pathlib import Path

from src.gimap.features.fitting.application import AiFittingCatalog
from src.gimap.features.fitting.infrastructure.adapters import AiFittingCatalogAdapter
from src.gimap.features.fitting.infrastructure.adapters.ai_model_registry import (
    APP_ROOT,
    resolve_model_path,
)

REPO = Path(__file__).resolve().parents[1]


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


def test_the_shipped_v5_model_is_found_from_another_working_directory(tmp_path, monkeypatch):
    from src.gimap.features.fitting.application.workflow_v5 import bundled_workflow

    monkeypatch.chdir(tmp_path)
    assert bundled_workflow() == REPO / "modules/Fitting_1D_Model/Workflow_v5/conditional_v2"
    assert (bundled_workflow() / "conditional_fast_manifest_v2.json").is_file()
