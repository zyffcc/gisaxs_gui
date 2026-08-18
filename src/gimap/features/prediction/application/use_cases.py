"""Prediction feature 的 framework-neutral use cases。"""

from __future__ import annotations

import threading
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

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
    PredictionExportItem,
    PredictionArrayExportRequest,
)
from .ports import (
    ModuleRepository,
    PredictionFileCatalog,
    PredictionImageRepository,
    PredictionMaskRepository,
    Predictor,
    Preprocessor,
    PredictionExportRepository,
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


def _serialized_prediction_data(
    prediction_data: dict[str, object] | None,
) -> dict[str, object] | None:
    if not prediction_data:
        return None
    try:
        serialized: dict[str, object] = {}
        inner_data = prediction_data.get("prediction_data", {})
        if not isinstance(inner_data, dict):
            return serialized
        for key, value in inner_data.items():
            if hasattr(value, "ndim") and hasattr(value, "dtype"):
                if value.ndim > 1:
                    serialized[key] = {
                        "type": "array_2d",
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                else:
                    serialized[key] = value.tolist()
            else:
                serialized[key] = value
        return serialized
    except Exception:
        return {"error": "Failed to serialize prediction data"}


class ExportPredictionJsonl:
    def __init__(self, repository: PredictionExportRepository):
        self._repository = repository

    def execute(
        self,
        items: tuple[PredictionExportItem, ...],
        export_path: Path,
        timestamp: str,
    ) -> Path:
        records = []
        for index, item in enumerate(items):
            records.append(
                json.dumps(
                    {
                        "index": index,
                        "filename": item.filename,
                        "filepath": item.filepath,
                        "stack_count": max(1, int(item.stack_count)),
                        "timestamp": item.timestamp,
                        "processing_time": item.processing_time,
                        "confidence": item.confidence,
                        "prediction_data": _serialized_prediction_data(
                            dict(item.prediction_data) if item.prediction_data else None
                        ),
                    },
                    ensure_ascii=False,
                )
            )
        content = "\n".join(records) + ("\n" if records else "")
        return self._repository.write_text(
            Path(export_path) / f"prediction_results_{timestamp}.jsonl",
            content,
        )


class ExportPredictionAscii:
    def __init__(self, repository: PredictionExportRepository):
        self._repository = repository

    def execute(
        self,
        items: tuple[PredictionExportItem, ...],
        export_path: Path,
        timestamp: str,
    ) -> Path | None:
        all_h_data = []
        all_r_data = []
        headers = []
        parameter_rows = []
        parameter_names: list[str] = []

        for item in items:
            payload = item.prediction_data or {}
            inner = payload.get("prediction_data", {})
            if not isinstance(inner, dict):
                continue
            h_data = inner.get("h")
            r_data = inner.get("r")
            p_data = inner.get("parameters")
            p_names = inner.get("parameter_names")
            if isinstance(p_data, np.ndarray):
                values = np.asarray(p_data, dtype=np.float32).reshape(-1)
                if isinstance(p_names, list) and len(p_names) >= values.size:
                    names = [str(name) for name in p_names[: values.size]]
                else:
                    names = [f"p{index + 1}" for index in range(values.size)]
                if not parameter_names:
                    parameter_names = names
                parameter_rows.append((item.filename, values))
            if isinstance(h_data, np.ndarray):
                all_h_data.append(h_data)
                headers.append(f"{item.filename}_h")
            if isinstance(r_data, np.ndarray):
                all_r_data.append(r_data)
                headers.append(f"{item.filename}_r")

        if all_h_data or all_r_data:
            lines = [
                "# Prediction 1D Curves Export",
                f"# Generated: {timestamp}",
                "# Columns: " + " | ".join(headers),
                "# Index\t" + "\t".join(headers),
            ]
            all_data = all_h_data + all_r_data
            max_len = max(len(data) for data in all_data)
            for index in range(max_len):
                row = [str(index)]
                for data in all_data:
                    row.append(f"{data[index]:.6g}" if index < len(data) else "NaN")
                lines.append("\t".join(row))
            return self._repository.write_text(
                Path(export_path) / f"prediction_curves_{timestamp}.txt",
                "\n".join(lines) + "\n",
            )

        if parameter_rows:
            lines = [
                "# Prediction Parameters Export",
                f"# Generated: {timestamp}",
                "filename\t" + "\t".join(parameter_names),
            ]
            for filename, values in parameter_rows:
                lines.append(
                    str(filename)
                    + "\t"
                    + "\t".join(f"{float(value):.8g}" for value in values)
                )
            return self._repository.write_text(
                Path(export_path) / f"prediction_parameters_{timestamp}.txt",
                "\n".join(lines) + "\n",
            )
        return None


class ExportPredictionArray:
    def __init__(self, repository: PredictionExportRepository):
        self._repository = repository

    def execute(self, request: PredictionArrayExportRequest) -> Path:
        values = np.asarray(request.values)
        if values.ndim not in (1, 2) or values.size == 0:
            raise ValueError("Prediction array export requires a non-empty 1D or 2D array")
        return self._repository.write_array(
            Path(request.path),
            values,
            fmt=request.fmt,
            header=request.header,
            comments=request.comments,
        )


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


class LoadPredictionMask:
    def __init__(self, repository: PredictionMaskRepository):
        self._repository = repository

    def execute(self, path: Path) -> np.ndarray:
        return self._repository.load(Path(path))


class ResolvePredictionStack:
    def __init__(self, catalog: PredictionFileCatalog):
        self._catalog = catalog

    def execute(self, start_path: Path, count: int) -> tuple[Path, ...]:
        return self._catalog.stack_paths(start_path, count)


class DiscoverNumberedPredictionFiles:
    def __init__(self, catalog: PredictionFileCatalog):
        self._catalog = catalog

    def execute(self, folder: Path, suffix: str = ".cbf"):
        return self._catalog.numbered_files(Path(folder), suffix)


class DiscoverPredictionFiles:
    def __init__(self, catalog: PredictionFileCatalog):
        self._catalog = catalog

    def execute(self, folder: Path, suffixes: tuple[str, ...]):
        return self._catalog.compatible_files(Path(folder), suffixes)


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
