"""动态 module preprocess entry adapter。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from ...application import PreprocessedPredictionInput
from ...domain import PredictionModule, coerce_array_to_shape, normalize_input_rank


class ModuleEntryPreprocessor:
    def preprocess(
        self, image: np.ndarray, module: PredictionModule
    ) -> PreprocessedPredictionInput:
        entry = module.preprocess.entry
        folder = module.folder
        if not entry or folder is None:
            raise ValueError(f"Prediction module {module.name!r} has no preprocessing entry")
        module_name, separator, function_name = entry.partition(":")
        if not separator or not module_name or not function_name:
            raise ValueError(f"Invalid preprocessing entry: {entry!r}")
        source = Path(folder) / f"{module_name}.py"
        if not source.is_file():
            raise FileNotFoundError(f"Preprocessing module not found: {source}")

        import_spec = importlib.util.spec_from_file_location(
            f"gimap_prediction_module_{module.id}_{module_name}", source
        )
        if import_spec is None or import_spec.loader is None:
            raise ImportError(f"Cannot load preprocessing module: {source}")
        loaded_module = importlib.util.module_from_spec(import_spec)
        import_spec.loader.exec_module(loaded_module)
        function = getattr(loaded_module, function_name, None)
        if not callable(function):
            raise AttributeError(f"Preprocessing function not found: {entry}")

        config = {
            "entry": module.preprocess.entry,
            "steps": list(module.preprocess.steps),
            "params": dict(module.preprocess.params),
        }
        input_image = np.asarray(image, dtype=np.float32).copy()
        try:
            output = function(
                input_image,
                config,
                module_folder=str(folder),
                return_steps=True,
            )
        except TypeError:
            output = function(input_image, config, module_folder=str(folder))

        steps = ()
        if isinstance(output, tuple):
            values = output[0]
            raw_steps = output[1] if len(output) > 1 else ()
        elif isinstance(output, dict):
            values = output.get("image")
            if values is None:
                values = output.get("result")
            raw_steps = output.get("steps", ())
        else:
            values = output
            raw_steps = ()
        if not isinstance(values, np.ndarray):
            raise TypeError(f"Preprocessing entry returned {type(values).__name__}, not ndarray")
        if isinstance(raw_steps, list):
            steps = tuple(dict(step) for step in raw_steps if isinstance(step, dict))

        prepared = normalize_input_rank(values)
        if module.input_shape is not None:
            prepared = coerce_array_to_shape(prepared, tuple(module.input_shape))
            expected = tuple(module.input_shape)
            if len(expected) == prepared.ndim and any(
                dimension is not None and int(dimension) != prepared.shape[index]
                for index, dimension in enumerate(expected)
            ):
                raise ValueError(
                    f"Preprocessing output shape {prepared.shape} does not match {expected}"
                )
        return PreprocessedPredictionInput(prepared, steps)
