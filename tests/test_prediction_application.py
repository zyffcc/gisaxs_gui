from pathlib import Path
import json

import numpy as np

from src.gimap.features.prediction.application import (
    DiscoverPredictionModules,
    DiscoverNumberedPredictionFiles,
    DiscoverPredictionFiles,
    FilePredictionResult,
    ExportPredictionAscii,
    ExportPredictionArray,
    ExportPredictionJsonl,
    InspectPredictionModel,
    LoadPredictionImage,
    LoadPredictionMask,
    LoadPredictionModule,
    LoadedPredictionImage,
    PredictFileBatch,
    PredictFileBatchRequest,
    PredictImage,
    PredictImageRequest,
    PredictMultipleFiles,
    PredictMultipleFilesRequest,
    PredictionExportItem,
    PredictionArrayExportRequest,
    PredictPreparedInput,
    PredictPreparedInputRequest,
    PreparePredictionInput,
    PreprocessedPredictionInput,
    RunPrediction,
    ResolvePredictionStack,
    UpdatePredictionModelPath,
)
from src.gimap.features.prediction.domain import (
    ModelRuntimeInfo,
    ModelSpec,
    OutputSpec,
    PredictionModule,
    PredictionRequest,
    PredictionResult,
)


def _module():
    return PredictionModule(
        id="module-id",
        name="Test module",
        model=ModelSpec("keras", "model.keras"),
        input_shape=(1, 2, 3, 1),
        outputs=OutputSpec(names=("hr",)),
    )


class _ModuleRepository:
    def __init__(self):
        self.module = _module()
        self.updated = None

    def discover(self):
        return (self.module,)

    def load(self, _path):
        return self.module

    def update_model_path(self, module, model_path):
        self.updated = (module, model_path)


class _ImageRepository:
    def __init__(self, failing_name=None):
        self.failing_name = failing_name
        self.calls = []

    def load(self, paths):
        self.calls.append(paths)
        if self.failing_name and any(path.name == self.failing_name for path in paths):
            raise OSError("damaged detector file")
        value = sum(int(path.stem[-1]) for path in paths)
        return LoadedPredictionImage(
            np.full((2, 3), value, dtype=np.float32), paths
        )


class _Catalog:
    def stack_paths(self, start_path, count):
        return tuple(start_path.with_name(f"frame{index}.cbf") for index in range(1, count + 1))

    def numbered_files(self, folder, suffix=".cbf"):
        from src.gimap.features.prediction.application import IndexedPredictionFile

        return (
            IndexedPredictionFile(folder / "frame_00001.cbf", 1),
            IndexedPredictionFile(folder / "frame_00003.cbf", 3),
        )

    def compatible_files(self, folder, suffixes):
        return (folder / "frame_00001.cbf", folder / "frame_00002.tif")


class _Preprocessor:
    def __init__(self):
        self.calls = []

    def preprocess(self, image, module):
        self.calls.append((np.array(image, copy=True), module))
        return PreprocessedPredictionInput(
            np.asarray(image, dtype=np.float32)[None, ..., None],
            ({"name": "identity", "shape": list(image.shape)},),
        )


class _Predictor:
    def __init__(self):
        self.requests = []

    def inspect(self, model_path, allow_unsafe_lambda=False):
        return ModelRuntimeInfo(
            artifact_path=Path(model_path),
            runtime_name="fake",
            input_shape=(1, 2, 3, 1),
        )

    def predict(self, request):
        self.requests.append(request)
        image = np.asarray(request.inputs)[0, ..., 0]
        return PredictionResult(outputs=image[None, ..., None])


class _ExportRepository:
    def __init__(self):
        self.writes = []

    def write_text(self, path, content):
        self.writes.append((Path(path), content))
        return Path(path)

    def write_array(self, path, values, **options):
        self.writes.append((Path(path), np.asarray(values), options))
        return Path(path)


def test_module_repository_use_cases_preserve_public_module_contract(tmp_path):
    repository = _ModuleRepository()
    yaml_path = tmp_path / "module.yaml"
    model_path = tmp_path / "model.keras"

    discovered = DiscoverPredictionModules(repository).execute()
    loaded = LoadPredictionModule(repository).execute(yaml_path)
    UpdatePredictionModelPath(repository).execute(loaded, model_path)

    assert discovered == (repository.module,)
    assert loaded.name == "Test module"
    assert repository.updated == (loaded, model_path)


def test_inspect_and_legacy_run_prediction_use_predictor_port(tmp_path):
    predictor = _Predictor()
    model_path = tmp_path / "model.keras"

    runtime = InspectPredictionModel(predictor).execute(model_path)
    raw = RunPrediction(predictor).execute(
        PredictionRequest(model_path, np.ones((1, 2, 3, 1)))
    )

    assert runtime.runtime_name == "fake"
    assert raw.outputs.shape == (1, 2, 3, 1)


def test_load_and_predict_image_use_cases_need_no_gui_or_tensorflow(tmp_path):
    repository = _ImageRepository()
    preprocessor = _Preprocessor()
    predictor = _Predictor()
    path = tmp_path / "frame1.cbf"

    loaded = LoadPredictionImage(repository).execute((path,))
    result = PredictImage(preprocessor, predictor).execute(
        PredictImageRequest(loaded.image, _module(), tmp_path / "model.keras")
    )

    assert loaded.image.dtype == np.float32
    assert result.model_input.shape == (1, 2, 3, 1)
    np.testing.assert_array_equal(result.outputs["hr"], loaded.image)
    np.testing.assert_array_equal(result.outputs["h"], [2, 2, 2])


def test_load_prediction_mask_uses_repository_port_without_gui(tmp_path):
    expected = np.ones((2, 3), dtype=np.uint8)

    class MaskRepository:
        def load(self, path):
            assert path == tmp_path / "mask.npy"
            return expected

    actual = LoadPredictionMask(MaskRepository()).execute(tmp_path / "mask.npy")

    assert actual is expected


def test_prepare_predict_prepared_and_stack_resolution_have_separate_ports(tmp_path):
    module = _module()
    preprocessor = _Preprocessor()
    predictor = _Predictor()
    prepared = PreparePredictionInput(preprocessor).execute(np.ones((2, 3)), module)

    result = PredictPreparedInput(predictor).execute(
        PredictPreparedInputRequest(
            prepared.values,
            module,
            tmp_path / "model.keras",
            prepared.steps,
        )
    )
    paths = ResolvePredictionStack(_Catalog()).execute(tmp_path / "start.cbf", 2)

    assert result.model_input.shape == (1, 2, 3, 1)
    assert [path.name for path in paths] == ["frame1.cbf", "frame2.cbf"]


def test_file_discovery_use_cases_depend_on_catalog_port(tmp_path):
    catalog = _Catalog()

    numbered = DiscoverNumberedPredictionFiles(catalog).execute(tmp_path)
    compatible = DiscoverPredictionFiles(catalog).execute(
        tmp_path, (".cbf", ".tif")
    )

    assert [(item.path.name, item.index) for item in numbered] == [
        ("frame_00001.cbf", 1),
        ("frame_00003.cbf", 3),
    ]
    assert [path.suffix for path in compatible] == [".cbf", ".tif"]


def test_prediction_jsonl_export_preserves_legacy_record_contract(tmp_path):
    repository = _ExportRepository()
    item = PredictionExportItem(
        filename="frame_0001.cbf",
        filepath="/data/frame_0001.cbf",
        stack_count=2,
        timestamp="2026-08-17T10:00:00",
        processing_time=1.25,
        confidence=0.9,
        prediction_data={
            "prediction_data": {
                "h": np.array([1.0, 2.0], dtype=np.float32),
                "hr": np.ones((2, 3), dtype=np.float32),
            }
        },
    )

    target = ExportPredictionJsonl(repository).execute(
        (item,), tmp_path, "20260817_100000"
    )

    assert target.name == "prediction_results_20260817_100000.jsonl"
    payload = json.loads(repository.writes[0][1])
    assert payload["index"] == 0
    assert payload["stack_count"] == 2
    assert payload["prediction_data"]["h"] == [1.0, 2.0]
    assert payload["prediction_data"]["hr"] == {
        "type": "array_2d",
        "shape": [2, 3],
        "dtype": "float32",
    }


def test_prediction_ascii_export_preserves_headers_values_and_nan_padding(tmp_path):
    repository = _ExportRepository()
    items = (
        PredictionExportItem(
            "one.cbf",
            "/data/one.cbf",
            1,
            None,
            0.1,
            None,
            {"prediction_data": {"h": np.array([1.0, 2.0]), "r": np.array([3.0])}},
        ),
        PredictionExportItem(
            "two.cbf",
            "/data/two.cbf",
            1,
            None,
            0.2,
            None,
            {"prediction_data": {"h": np.array([4.0])}},
        ),
    )

    target = ExportPredictionAscii(repository).execute(
        items, tmp_path, "20260817_100000"
    )

    assert target.name == "prediction_curves_20260817_100000.txt"
    content = repository.writes[0][1]
    assert "# Columns: one.cbf_h | one.cbf_r | two.cbf_h" in content
    assert "1\t2\tNaN\tNaN" in content


def test_prediction_array_export_preserves_savetxt_contract(tmp_path):
    repository = _ExportRepository()
    request = PredictionArrayExportRequest(
        tmp_path / "curve.txt",
        np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32),
        fmt="%.6g",
        header="x y",
        comments="",
    )

    target = ExportPredictionArray(repository).execute(request)

    assert target == request.path
    assert repository.writes[0][2] == {
        "fmt": "%.6g",
        "header": "x y",
        "comments": "",
    }


def test_file_batch_uses_same_image_prediction_and_returns_structured_error(tmp_path):
    repository = _ImageRepository(failing_name="frame2.cbf")
    predict_file = PredictFileBatch(
        LoadPredictionImage(repository),
        PredictImage(_Preprocessor(), _Predictor()),
    )

    success = predict_file.execute(
        PredictFileBatchRequest((tmp_path / "frame1.cbf",), _module(), tmp_path / "model")
    )
    failure = predict_file.execute(
        PredictFileBatchRequest((tmp_path / "frame2.cbf",), _module(), tmp_path / "model")
    )

    assert success.status == "succeeded"
    assert failure.status == "failed"
    assert failure.error_message == "damaged detector file"


def test_multi_file_prediction_continues_after_middle_error(tmp_path):
    repository = _ImageRepository(failing_name="frame2.cbf")
    predict_file = PredictFileBatch(
        LoadPredictionImage(repository),
        PredictImage(_Preprocessor(), _Predictor()),
    )
    workflow = PredictMultipleFiles(predict_file)
    progress = []
    batches = tuple((tmp_path / f"frame{index}.cbf",) for index in (1, 2, 3))

    result = workflow.execute(
        PredictMultipleFilesRequest(batches, _module(), tmp_path / "model"),
        on_progress=progress.append,
    )

    assert [item.status for item in result.items] == ["succeeded", "failed", "succeeded"]
    assert result.failed_count == 1
    assert [item.completed for item in progress] == [1, 2, 3]


def test_multi_file_prediction_can_cancel_between_batches(tmp_path):
    workflow = None

    class CancellingFilePrediction:
        def execute(self, request):
            workflow.cancel()
            return FilePredictionResult(request.paths, "succeeded")

    workflow = PredictMultipleFiles(CancellingFilePrediction())
    batches = tuple((tmp_path / f"frame{index}.cbf",) for index in (1, 2, 3))

    result = workflow.execute(
        PredictMultipleFilesRequest(batches, _module(), tmp_path / "model")
    )

    assert result.cancelled is True
    assert len(result.items) == 1
