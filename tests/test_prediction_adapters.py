from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from src.gimap.features.prediction.domain import (
    ModelSpec,
    PredictionModule,
    PreprocessSpec,
)
from src.gimap.features.prediction.infrastructure import (
    FabioPredictionImageRepository,
    ModuleEntryPreprocessor,
    YamlModuleRepository,
    module_to_legacy_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_existing_module_yaml_files_are_discovered_with_legacy_contracts():
    modules = YamlModuleRepository(PROJECT_ROOT / "modules").discover()
    by_id = {module.id: module for module in modules}

    assert set(by_id) == {
        "A_AgIUPAC_FF_256",
        "G_AgPNO_FF_256",
        "Au_Silicon_15nm",
        "Yuxin_CuPolymer_FF_256",
        "Yuxin_CuPolymer_SF_4Para",
    }
    assert by_id["Au_Silicon_15nm"].input_shape == (1, 256, 256, 1)
    assert by_id["Au_Silicon_15nm"].outputs.names == (
        "hr distribution",
        "h distribution",
        "r distribution",
    )
    sf = by_id["Yuxin_CuPolymer_SF_4Para"]
    assert sf.input_shape == (1, 256, 256, 2)
    assert sf.outputs.parameter_names == ("t_Cu", "t_polymer", "D", "sigma")
    assert sf.model.path.startswith("C:\\Users\\")


def test_legacy_module_mapping_resolves_assets_but_preserves_windows_model_path():
    module = YamlModuleRepository(PROJECT_ROOT / "modules").load(
        PROJECT_ROOT / "modules" / "Au_Silicon_15nm" / "module.yaml"
    )

    legacy = module_to_legacy_dict(module)

    assert legacy["model_path"] == module.model.path
    assert Path(legacy["mask_path"]).is_absolute()
    assert legacy["preprocess_steps"] == list(module.preprocess.steps)


def test_model_path_update_changes_only_model_path_line(tmp_path):
    folder = tmp_path / "module"
    folder.mkdir()
    source = (
        "id: sample\nname: Sample\nframework: tensorflow\nmodel:\n"
        "  format: keras\n  model_path: ''\npreprocess:\n  entry: preprocess:run\n"
    )
    yaml_path = folder / "module.yaml"
    yaml_path.write_text(source, encoding="utf-8")
    repository = YamlModuleRepository(tmp_path)
    module = repository.load(yaml_path)

    repository.update_model_path(module, tmp_path / "new model.keras")

    updated = yaml_path.read_text(encoding="utf-8")
    assert "id: sample" in updated
    assert "entry: preprocess:run" in updated
    assert f"model_path: '{tmp_path / 'new model.keras'}'" in updated


def test_fabio_repository_sums_stack_as_float32(tmp_path, monkeypatch):
    paths = (tmp_path / "frame1.cbf", tmp_path / "frame2.cbf")
    for path in paths:
        path.write_bytes(b"test")
    values = {
        str(paths[0]): np.array([[1, 2], [3, 4]], dtype=np.uint16),
        str(paths[1]): np.array([[10, 20], [30, 40]], dtype=np.float64),
    }
    fake_fabio = SimpleNamespace(
        open=lambda path: SimpleNamespace(data=values[path])
    )
    monkeypatch.setitem(sys.modules, "fabio", fake_fabio)

    loaded = FabioPredictionImageRepository().load(paths)

    assert loaded.image.dtype == np.float32
    np.testing.assert_array_equal(loaded.image, [[11, 22], [33, 44]])
    assert loaded.source_paths == paths


def test_module_entry_preprocessor_is_lazy_and_preserves_step_order(tmp_path):
    (tmp_path / "preprocess.py").write_text(
        "import numpy as np\n"
        "def run(image, config, module_folder=None, return_steps=False):\n"
        "    result=np.asarray(image, dtype=np.float32)*2\n"
        "    steps=[{'label': name, 'image': result.copy()} for name in config['steps']]\n"
        "    return (result, steps) if return_steps else result\n",
        encoding="utf-8",
    )
    module = PredictionModule(
        id="temporary",
        name="Temporary",
        folder=tmp_path,
        model=ModelSpec("keras", "model.keras"),
        preprocess=PreprocessSpec("preprocess:run", ("first", "second"), {}),
        input_shape=(1, 2, 3, 1),
    )

    result = ModuleEntryPreprocessor().preprocess(np.ones((2, 3)), module)

    assert result.values.shape == (1, 2, 3, 1)
    assert np.all(result.values == 2)
    assert [step["label"] for step in result.steps] == ["first", "second"]
