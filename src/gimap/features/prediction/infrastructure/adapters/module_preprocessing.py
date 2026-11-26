"""动态 module preprocess entry adapter。

A module's entry is called as ``run(image, config, module_folder=…, return_steps=True)``. An entry that also
takes a ``context`` parameter (or ``**kwargs``) is given what is known about the frames
(``detector_images.ProfileFrameContexts``: files, format, facts, geometry, ``q_map()``); the plain values of it
are then written into the first step it returns (``frame_context``), so the export record says which geometry
the model input was made with.

A module's file is imported once and again only when it changes on disk (its size or time): what it keeps between
calls (a lookup table from pixels to a q grid, say) stays from frame to frame, and an edited file is used at the next
prediction. Importing it for every frame took longer than many predictions.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ...application import PreprocessedPredictionInput
from ...domain import PredictionModule, coerce_array_to_shape, normalize_input_rank


def takes_context(function) -> bool:
    """Whether a preprocessing entry accepts ``context=`` (by name, or through ``**kwargs``)."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "context" or parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )


_IMPORTED: dict[str, tuple[tuple, Any]] = {}
"""Entry modules by file: (size and time of the file when it was imported, the module)."""


def entry_module(source: Path, name: str):
    """The module of ``source``, imported once and again only when the file has changed since."""
    state = source.stat()
    stamp = (state.st_size, state.st_mtime_ns)
    key = str(source.resolve())
    kept = _IMPORTED.get(key)
    if kept is not None and kept[0] == stamp:
        return kept[1]
    import_spec = importlib.util.spec_from_file_location(name, source)
    if import_spec is None or import_spec.loader is None:
        raise ImportError(f"Cannot load preprocessing module: {source}")
    loaded_module = importlib.util.module_from_spec(import_spec)
    import_spec.loader.exec_module(loaded_module)
    _IMPORTED[key] = (stamp, loaded_module)
    return loaded_module


def context_record(context: Mapping[str, Any]) -> dict[str, Any]:
    """The plain part of a frame context (no ``q_map``), as written into the export record."""
    return {key: value for key, value in context.items() if not callable(value)}


class ModuleEntryPreprocessor:
    def preprocess(
        self, image: np.ndarray, module: PredictionModule, context: Mapping[str, Any] | None = None
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

        loaded_module = entry_module(source, f"gimap_prediction_module_{module.id}_{module_name}")
        function = getattr(loaded_module, function_name, None)
        if not callable(function):
            raise AttributeError(f"Preprocessing function not found: {entry}")

        config = {
            "entry": module.preprocess.entry,
            "steps": list(module.preprocess.steps),
            "params": dict(module.preprocess.params),
        }
        input_image = np.asarray(image, dtype=np.float32).copy()
        given = context if context is not None and takes_context(function) else None
        extra = {} if given is None else {"context": given}
        try:
            output = function(
                input_image,
                config,
                module_folder=str(folder),
                return_steps=True,
                **extra,
            )
        except TypeError:
            output = function(input_image, config, module_folder=str(folder), **extra)

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
        if given is not None and steps:
            steps = ({**steps[0], "frame_context": context_record(given)},) + steps[1:]

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
