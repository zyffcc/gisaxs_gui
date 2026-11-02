"""Labs / 2D Prediction: the model folder a user imports lives in the user data folder, never in module.yaml.

``modules/*/module.yaml`` are program files (tracked, shared by every user of an installation). Import model...
used to write the chosen folder into them, so one user's folders ended up in everybody's files. Now the choice
is kept in the user preferences, wins over the module's own ``model_path`` at load time, and the module file
is not touched. A module without a model says so.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import pytest

from src.gimap.features.prediction.infrastructure import (
    InMemoryModelPathStore,
    PreferencesModelPathStore,
    YamlModuleRepository,
)
from src.gimap.features.prediction.infrastructure.adapters.model_paths import PREFERENCES_KEY
from src.gimap.features.prediction.presentation.bindings import module_catalog
from src.gimap.integrations.state import StorePreferencesRepository, UserStore

ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "modules"
GOLD = MODULES / "Au_Silicon_15nm" / "module.yaml"
GOLD_ID = "Au_Silicon_15nm"
GOLD_NAME = "Gold on Silicon 15nm (DESY P03)"


def _module_file(root: Path, model_path: str = "") -> Path:
    folder = root / "modules" / "synthetic_module"
    folder.mkdir(parents=True)
    path = folder / "module.yaml"
    path.write_text(
        "id: synthetic_ff\n"
        'name: "Synthetic FF"\n'
        "framework: tensorflow\n"
        "model:\n"
        "  format: keras\n"
        f"  model_path: '{model_path}'\n"
        "preprocess:\n"
        "  entry: preprocess:run\n",
        encoding="utf-8",
    )
    return path


def test_import_stores_the_model_path_in_the_user_data_folder_not_in_the_module_file(tmp_path):
    yaml_path = _module_file(tmp_path)
    before = yaml_path.read_bytes()
    settings_file = tmp_path / "user_data" / "settings.json"
    preferences = StorePreferencesRepository(UserStore(settings_file))
    repository = YamlModuleRepository(yaml_path.parents[1], PreferencesModelPathStore(preferences))
    model = tmp_path / "models" / "synthetic_saved_model"
    model.mkdir(parents=True)

    repository.update_model_path(repository.load(yaml_path), model)

    assert yaml_path.read_bytes() == before  # the module file is unchanged, byte for byte
    stored = json.loads(settings_file.read_text(encoding="utf-8"))
    assert stored["preferences"][PREFERENCES_KEY] == {"synthetic_ff": str(model)}
    # A new session (a fresh store read from disk) finds the folder again.
    again = YamlModuleRepository(
        yaml_path.parents[1], PreferencesModelPathStore(StorePreferencesRepository(UserStore(settings_file)))
    )
    assert again.load(yaml_path).model.path == str(model)


def test_a_stored_model_path_wins_over_the_module_yaml_path(tmp_path):
    yaml_path = _module_file(tmp_path, model_path="models/shipped_model")
    chosen = str(tmp_path / "elsewhere" / "chosen_model")
    repository = YamlModuleRepository(
        yaml_path.parents[1], InMemoryModelPathStore({"synthetic_ff": chosen})
    )

    assert repository.load(yaml_path).model.path == chosen
    assert [module.model.path for module in repository.discover()] == [chosen]


def test_a_model_path_already_in_a_module_yaml_still_loads(tmp_path):
    """Users whose module.yaml names a model folder (written by an earlier version) keep working."""
    model = tmp_path / "models" / "earlier_model"
    yaml_path = _module_file(tmp_path, model_path=str(model))

    module = YamlModuleRepository(yaml_path.parents[1]).load(yaml_path)

    assert module.model.path == str(model)


def test_the_shipped_module_files_name_no_machine_specific_model_folder():
    for module in YamlModuleRepository(MODULES, InMemoryModelPathStore()).discover():
        path = module.model.path
        assert not PureWindowsPath(path).is_absolute() and not PurePosixPath(path).is_absolute(), (
            module.yaml_path,
            path,
        )


def test_a_module_without_an_id_is_remembered_by_its_folder(tmp_path):
    folder = tmp_path / "modules" / "no_id_module"
    folder.mkdir(parents=True)
    yaml_path = folder / "module.yaml"
    yaml_path.write_text('name: "No id"\nmodel:\n  model_path: ""\n', encoding="utf-8")
    store = InMemoryModelPathStore()
    repository = YamlModuleRepository(folder.parent, store)

    repository.update_model_path(repository.load(yaml_path), tmp_path / "model")

    assert store.get("no_id_module") == str(tmp_path / "model")
    assert repository.load(yaml_path).model.path == str(tmp_path / "model")


# --- the page ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def window():
    from tests.test_labs_prediction import _wait, _window

    shown = _window()
    shown.show()
    shown.runtime.navigate("predict")
    _wait(lambda: False, 0.3)
    yield shown
    shown.close()


@pytest.fixture
def page(window):
    preferences = window.app_context.preferences
    preferences.set(PREFERENCES_KEY, {})
    prediction = window.runtime.prediction
    prediction._current_model = None
    yield prediction
    preferences.set(PREFERENCES_KEY, {})
    prediction._current_model = None


def _new_log_lines(window, before: int) -> list[str]:
    return window.predictStatusTextBrowser.toPlainText().splitlines()[before:]


def _select_gold(window, prediction) -> list[str]:
    before = len(window.predictStatusTextBrowser.toPlainText().splitlines())
    prediction._refresh_modules()
    window.gisaxsPredictModuleSelectCombox.setCurrentText(GOLD_NAME)  # Import model... reads the combo
    prediction._on_module_selected(GOLD_NAME)
    return _new_log_lines(window, before)


def test_a_module_without_a_model_says_no_model_is_imported(window, page):
    lines = _select_gold(window, page)

    assert page.current_parameters["module_model_path"] == ""
    assert f"[INFO] {module_catalog.NO_MODEL_IMPORTED}" in lines
    assert window.gisaxsPredictModelStatusTextLabel.isHidden()  # the Model chip under 3. Predict says it
    assert window.gisaxsPredictModelReadyLabel.text() == "Model: Not loaded"
    assert not window.gisaxsPredictPredictButton.isEnabled()


def test_a_remembered_folder_that_is_gone_is_named(window, page, tmp_path):
    gone = str(tmp_path / "deleted_model")
    window.app_context.preferences.set(PREFERENCES_KEY, {GOLD_ID: gone})

    lines = _select_gold(window, page)

    expected = module_catalog.MODEL_FOLDER_MISSING.format(path=gone)
    assert f"[WARN] {expected}" in lines
    assert page.current_parameters["module_model_path"] == ""
    assert not window.gisaxsPredictPredictButton.isEnabled()


def test_import_model_remembers_the_folder_and_leaves_module_yaml_unchanged(
    window, page, tmp_path, monkeypatch
):
    from tests.test_labs_prediction import _wait

    model = tmp_path / "gold_saved_model"
    model.mkdir()
    asked = []

    def choose(parent, caption, start="", *args):
        asked.append(start)
        return str(model) if len(asked) == 1 else ""  # the second time the user cancels

    monkeypatch.setattr(module_catalog.QFileDialog, "getExistingDirectory", staticmethod(choose))
    monkeypatch.setattr(
        page.prediction_view_model,
        "inspect_model",
        lambda path, allow_unsafe_lambda=False: SimpleNamespace(artifact_path=Path(path)),
    )
    before = GOLD.read_bytes()
    try:
        _select_gold(window, page)
        log_before = len(window.predictStatusTextBrowser.toPlainText().splitlines())
        page._on_model_import_clicked()
        _wait(lambda: not page._model_loading)
        _wait(lambda: False, 0.1)

        assert GOLD.read_bytes() == before
        assert window.app_context.preferences.get(PREFERENCES_KEY) == {GOLD_ID: str(model)}
        remembered = module_catalog.MODEL_FOLDER_REMEMBERED.format(name=GOLD_NAME)
        assert f"[INFO] {remembered}" in _new_log_lines(window, log_before)
        assert page._current_model is not None
        assert window.gisaxsPredictModelReadyLabel.text() == "Model: Loaded"

        # The stored folder wins when the modules are read again (module.yaml still names none).
        page._refresh_modules()
        assert page._modules_by_name[GOLD_NAME]["model_path"] == str(model)

        # Pressed again with this model loaded: choose another folder, starting beside this one.
        page._on_model_import_clicked()
        assert asked == [str(GOLD.parent), str(model.parent)]
        assert page._current_model is not None  # cancelled: the loaded model stays
        assert window.app_context.preferences.get(PREFERENCES_KEY) == {GOLD_ID: str(model)}
    finally:
        if GOLD.read_bytes() != before:  # never leave a changed program file behind
            GOLD.write_bytes(before)
