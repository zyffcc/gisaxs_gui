"""现有 module.yaml 格式 repository。"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from ...domain import ModelSpec, OutputSpec, PredictionModule, PreprocessSpec
from ...application.module_descriptions import describe_prediction_module


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


# Backward-compatible import for callers that previously obtained this mapper
# from infrastructure. The implementation is application-owned and performs no I/O.
module_to_legacy_dict = describe_prediction_module
