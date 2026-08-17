"""现有 module.yaml 格式 repository。"""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
import re
from typing import Any

from ...domain import ModelSpec, OutputSpec, PredictionModule, PreprocessSpec


def _load_yaml_mapping(text: str) -> dict[str, Any]:
    import yaml

    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        # Double-quoted Windows paths such as C:\Users are invalid YAML escapes.
        repaired = re.sub(
            r'(?m)^(\s*model_path\s*:\s*)"([A-Za-z]:\\[^"\r\n]*)"\s*$',
            lambda match: match.group(1) + "'" + match.group(2).replace("'", "''") + "'",
            text,
        )
        value = yaml.safe_load(repaired)
    if not isinstance(value, dict):
        raise ValueError("module.yaml must contain a mapping")
    return value


def _tuple_of_floats(value) -> tuple[float, ...]:
    return tuple(float(item) for item in value) if isinstance(value, list) else ()


def _is_windows_absolute(path: str) -> bool:
    return bool(path and PureWindowsPath(path).is_absolute())


class YamlModuleRepository:
    def __init__(self, modules_root: Path):
        self.modules_root = Path(modules_root)

    def discover(self) -> tuple[PredictionModule, ...]:
        if not self.modules_root.is_dir():
            return ()
        modules = []
        for folder in sorted(self.modules_root.iterdir(), key=lambda path: path.name.casefold()):
            yaml_path = folder / "module.yaml"
            if folder.is_dir() and yaml_path.is_file():
                modules.append(self.load(yaml_path))
        return tuple(modules)

    def load(self, yaml_path: Path) -> PredictionModule:
        path = Path(yaml_path)
        data = _load_yaml_mapping(path.read_text(encoding="utf-8"))
        model = data.get("model") if isinstance(data.get("model"), dict) else {}
        preprocess = (
            data.get("preprocess") if isinstance(data.get("preprocess"), dict) else {}
        )
        params = preprocess.get("params") if isinstance(preprocess.get("params"), dict) else {}
        io = data.get("io") if isinstance(data.get("io"), dict) else {}
        outputs = data.get("outputs")
        if isinstance(outputs, list):
            output_names = tuple(
                str(item.get("name", ""))
                for item in outputs
                if isinstance(item, dict) and item.get("name")
            )
            output_spec = OutputSpec(names=output_names)
        elif isinstance(outputs, dict):
            output_spec = OutputSpec(
                type=str(outputs.get("type", "")),
                parameter_names=tuple(str(item) for item in outputs.get("parameter_names", ())),
                target_min=_tuple_of_floats(outputs.get("target_min")),
                target_max=_tuple_of_floats(outputs.get("target_max")),
            )
        else:
            output_spec = OutputSpec()
        raw_shape = io.get("input_shape")
        input_shape = (
            tuple(int(item) for item in raw_shape)
            if isinstance(raw_shape, list) and raw_shape
            else None
        )
        return PredictionModule(
            id=str(data.get("id", "")),
            name=str(data.get("name", "") or path.parent.name),
            framework=str(data.get("framework", "")),
            version=str(data.get("version", "")),
            folder=path.parent.resolve(),
            yaml_path=path.resolve(),
            model=ModelSpec(
                format=str(model.get("format", "")),
                path=str(model.get("model_path", "")),
            ),
            preprocess=PreprocessSpec(
                entry=str(preprocess.get("entry", "")),
                steps=tuple(str(item) for item in preprocess.get("steps", ())),
                params=dict(params),
            ),
            input_type=str(io.get("input_type", "cbf")),
            stack_axis=int(io.get("stack_axis", 0)),
            input_shape=input_shape,
            outputs=output_spec,
        )

    def update_model_path(self, module: PredictionModule, model_path: Path) -> None:
        if module.yaml_path is None:
            raise ValueError("Prediction module has no module.yaml path")
        path = Path(module.yaml_path)
        text = path.read_text(encoding="utf-8")
        quoted = "'" + str(model_path).replace("'", "''") + "'"
        pattern = re.compile(r"(?m)^(\s*model_path\s*:\s*).*$")
        if pattern.search(text):
            updated = pattern.sub(lambda match: match.group(1) + quoted, text, count=1)
        else:
            model_match = re.search(r"(?m)^(\s*)model\s*:\s*$", text)
            if model_match is None:
                raise ValueError("module.yaml has no model section")
            insertion = model_match.end()
            indent = model_match.group(1) + "  "
            updated = text[:insertion] + f"\n{indent}model_path: {quoted}" + text[insertion:]
        path.write_text(updated, encoding="utf-8")


def module_to_legacy_dict(module: PredictionModule) -> dict[str, object]:
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
            result["preprocess_resize"] = {"height": int(shape[0]), "width": int(shape[1])}
    elif isinstance(resize, (list, tuple)) and len(resize) == 2:
        result["preprocess_resize"] = {"height": int(resize[0]), "width": int(resize[1])}
    if isinstance(mask, dict):
        mask_path = str(mask.get("path", ""))
        if mask_path and module.folder is not None and not Path(mask_path).is_absolute() and not _is_windows_absolute(mask_path):
            mask_path = str((module.folder / mask_path).resolve())
        result["mask_path"] = mask_path
        if isinstance(mask.get("mask_value"), (int, float)):
            result["mask_value"] = float(mask["mask_value"])
        if isinstance(mask.get("crop_mask"), dict):
            result["mask_crop"] = dict(mask["crop_mask"])
    model_path = module.model.path
    if model_path and module.folder is not None and not Path(model_path).is_absolute() and not _is_windows_absolute(model_path):
        result["model_path"] = str((module.folder / model_path).resolve())
    return result
