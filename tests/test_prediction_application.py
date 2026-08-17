from pathlib import Path

import numpy as np

from src.gimap.features.prediction.application import (
    DiscoverPredictionModules,
    FilePredictionResult,
    InspectPredictionModel,
    LoadPredictionImage,
    LoadPredictionModule,
    LoadedPredictionImage,
    PredictFileBatch,
    PredictFileBatchRequest,
    PredictImage,
    PredictImageRequest,
    PredictMultipleFiles,
    PredictMultipleFilesRequest,
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
