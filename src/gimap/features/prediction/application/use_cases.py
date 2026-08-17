"""Prediction feature 的 framework-neutral use cases。"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from .models import (
    FilePredictionResult,
    ImagePredictionResult,
    LoadedPredictionImage,
    MultiplePredictionResult,
    PredictFileBatchRequest,
    PredictImageRequest,
    PredictPreparedInputRequest,
    PredictMultipleFilesRequest,
    PredictionProgress,
)
from .ports import (
    ModuleRepository,
    PredictionFileCatalog,
    PredictionImageRepository,
    Predictor,
    Preprocessor,
)
from ..domain import (
    ModelRuntimeInfo,
    PredictionRequest,
    PredictionResult,
    normalize_prediction_output,
)


def _module_output_values(module) -> dict[str, object]:
    return {
        "output_type": module.outputs.type,
        "parameter_names": list(module.outputs.parameter_names),
        "target_min": list(module.outputs.target_min),
        "target_max": list(module.outputs.target_max),
    }


class RunPrediction:
    """保留早期模型无关 Predictor API。"""

    def __init__(self, predictor: Predictor):
        self._predictor = predictor

    def execute(self, request: PredictionRequest) -> PredictionResult:
        return self._predictor.predict(request)


class DiscoverPredictionModules:
    def __init__(self, repository: ModuleRepository):
        self._repository = repository

    def execute(self):
        return self._repository.discover()


class LoadPredictionModule:
    def __init__(self, repository: ModuleRepository):
        self._repository = repository

    def execute(self, yaml_path: Path):
        return self._repository.load(yaml_path)


class UpdatePredictionModelPath:
    def __init__(self, repository: ModuleRepository):
        self._repository = repository

    def execute(self, module, model_path: Path) -> None:
        self._repository.update_model_path(module, model_path)


class InspectPredictionModel:
    def __init__(self, predictor: Predictor):
        self._predictor = predictor

    def execute(self, model_path: Path, *, allow_unsafe_lambda=False) -> ModelRuntimeInfo:
        return self._predictor.inspect(model_path, allow_unsafe_lambda)


class LoadPredictionImage:
    def __init__(self, repository: PredictionImageRepository):
        self._repository = repository

    def execute(self, paths: tuple[Path, ...]) -> LoadedPredictionImage:
        return self._repository.load(paths)


class ResolvePredictionStack:
    def __init__(self, catalog: PredictionFileCatalog):
        self._catalog = catalog

    def execute(self, start_path: Path, count: int) -> tuple[Path, ...]:
        return self._catalog.stack_paths(start_path, count)


class PreparePredictionInput:
    def __init__(self, preprocessor: Preprocessor):
        self._preprocessor = preprocessor

    def execute(self, image, module):
        return self._preprocessor.preprocess(image, module)


class PredictPreparedInput:
    def __init__(self, predictor: Predictor):
        self._predictor = predictor

    def execute(self, request: PredictPreparedInputRequest) -> ImagePredictionResult:
        raw = self._predictor.predict(
            PredictionRequest(
                model_path=request.model_path,
                inputs=request.values,
                allow_unsafe_lambda=request.allow_unsafe_lambda,
                precision_policy=request.precision_policy,
                timeout_seconds=request.timeout_seconds,
            )
        )
        outputs = normalize_prediction_output(raw.outputs, _module_output_values(request.module))
        if outputs is None:
            raise ValueError("Prediction produced no compatible output")
        return ImagePredictionResult(
            outputs=outputs,
            model_input=request.values,
            preprocess_steps=request.preprocess_steps,
            runtime=raw.runtime,
        )


class PredictImage:
    def __init__(self, preprocessor: Preprocessor, predictor: Predictor):
        self._prepare = PreparePredictionInput(preprocessor)
        self._predict_prepared = PredictPreparedInput(predictor)

    def execute(self, request: PredictImageRequest) -> ImagePredictionResult:
        prepared = self._prepare.execute(request.image, request.module)
        return self._predict_prepared.execute(
            PredictPreparedInputRequest(
                values=prepared.values,
                module=request.module,
                model_path=request.model_path,
                preprocess_steps=prepared.steps,
                allow_unsafe_lambda=request.allow_unsafe_lambda,
                precision_policy=request.precision_policy,
                timeout_seconds=request.timeout_seconds,
            )
        )


class PredictFileBatch:
    """单文件和 stack 共用的文件级 prediction use case。"""

    def __init__(self, load_image: LoadPredictionImage, predict_image: PredictImage):
        self._load_image = load_image
        self._predict_image = predict_image

    def execute(self, request: PredictFileBatchRequest) -> FilePredictionResult:
        try:
            loaded = self._load_image.execute(request.paths)
            prediction = self._predict_image.execute(
                PredictImageRequest(
                    image=loaded.image,
                    module=request.module,
                    model_path=request.model_path,
                    allow_unsafe_lambda=request.allow_unsafe_lambda,
                    precision_policy=request.precision_policy,
                    timeout_seconds=request.timeout_seconds,
                )
            )
        except Exception as exc:
            return FilePredictionResult(
                paths=request.paths, status="failed", error_message=str(exc)
            )
        return FilePredictionResult(
            paths=request.paths, status="succeeded", prediction=prediction
        )


class PredictMultipleFiles:
    def __init__(self, predict_file: PredictFileBatch):
        self._predict_file = predict_file
        self._cancelled = threading.Event()

    def execute(
        self,
        request: PredictMultipleFilesRequest,
        on_progress: Callable[[PredictionProgress], None] | None = None,
    ) -> MultiplePredictionResult:
        self._cancelled.clear()
        results = []
        total = len(request.batches)
        for batch in request.batches:
            if self._cancelled.is_set():
                break
            result = self._predict_file.execute(
                PredictFileBatchRequest(
                    paths=batch,
                    module=request.module,
                    model_path=request.model_path,
                    allow_unsafe_lambda=request.allow_unsafe_lambda,
                    precision_policy=request.precision_policy,
                    timeout_seconds=request.timeout_seconds,
                )
            )
            results.append(result)
            if on_progress is not None:
                on_progress(
                    PredictionProgress(
                        completed=len(results),
                        total=total,
                        current_paths=batch,
                        status=result.status,
                        message=result.error_message,
                    )
                )
            if result.status == "failed" and not request.continue_on_error:
                break
        return MultiplePredictionResult(
            items=tuple(results), cancelled=self._cancelled.is_set()
        )

    def cancel(self) -> None:
        self._cancelled.set()
