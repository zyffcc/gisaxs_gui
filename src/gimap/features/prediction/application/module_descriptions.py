"""Framework-neutral Prediction module description mapping."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath

from ..domain import PredictionModule


def _is_windows_absolute(path: str) -> bool:
    return bool(path and PureWindowsPath(path).is_absolute())


def describe_prediction_module(module: PredictionModule) -> dict[str, object]:
    """Return the stable module values consumed by the legacy presentation."""

    params = dict(module.preprocess.params)
    result: dict[str, object] = {
        "id": module.id,
        "name": module.name,
        "framework": module.framework,
        "version": module.version,
        "folder": str(module.folder or ""),
        "yaml_path": str(module.yaml_path or ""),
        "model_format": module.model.format,
        "model_path": module.model.path,
        "preprocess_entry": module.preprocess.entry,
        "preprocess_steps": list(module.preprocess.steps),
        "preprocess_params": params,
        "preprocess_raw": {
            "entry": module.preprocess.entry,
            "steps": list(module.preprocess.steps),
            "params": params,
        },
        "io_input_shape": module.input_shape,
        "output_type": module.outputs.type,
        "parameter_names": list(module.outputs.parameter_names),
        "target_min": list(module.outputs.target_min),
        "target_max": list(module.outputs.target_max),
        "_prediction_module": module,
    }
    crop = params.get("crop")
    resize = params.get("resize")
    mask = params.get("mask")
    if isinstance(crop, dict):
        result["preprocess_crop"] = dict(crop)
    if isinstance(resize, dict):
        shape = resize.get("shape") or resize.get("size")
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            result["preprocess_resize"] = {
                "height": int(shape[0]),
                "width": int(shape[1]),
            }
    elif isinstance(resize, (list, tuple)) and len(resize) == 2:
        result["preprocess_resize"] = {
            "height": int(resize[0]),
            "width": int(resize[1]),
        }
    if isinstance(mask, dict):
        mask_path = str(mask.get("path", ""))
        if (
            mask_path
            and module.folder is not None
            and not Path(mask_path).is_absolute()
            and not _is_windows_absolute(mask_path)
        ):
            mask_path = str((module.folder / mask_path).resolve())
        result["mask_path"] = mask_path
        if isinstance(mask.get("mask_value"), (int, float)):
            result["mask_value"] = float(mask["mask_value"])
        if isinstance(mask.get("crop_mask"), dict):
            result["mask_crop"] = dict(mask["crop_mask"])
    model_path = module.model.path
    if (
        model_path
        and module.folder is not None
        and not Path(model_path).is_absolute()
        and not _is_windows_absolute(model_path)
    ):
        result["model_path"] = str((module.folder / model_path).resolve())
    return result
