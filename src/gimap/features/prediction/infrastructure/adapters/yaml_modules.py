"""现有 module.yaml 格式 repository。"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from ...domain import (
    DistributionAxisSpec,
    ModelSpec,
    OutputSpec,
    PredictionModule,
    PreprocessSpec,
)
from ...application.module_descriptions import describe_prediction_module
from .model_paths import InMemoryModelPathStore


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


def _distribution_axis(value: object) -> DistributionAxisSpec:
    if not isinstance(value, dict):
        return DistributionAxisSpec()
    minimum = value.get("min")
    maximum = value.get("max")
    return DistributionAxisSpec(
        key=str(value.get("key", "")).strip().casefold(),
        label=str(value.get("label", "")).strip(),
        unit=str(value.get("unit", "")).strip(),
        minimum=float(minimum) if isinstance(minimum, (int, float)) else None,
        maximum=float(maximum) if isinstance(maximum, (int, float)) else None,
    )


def module_model_key(module_id: str, folder: Path | None) -> str:
    """What a user's model folder is remembered by: the module's id, else its folder name."""
    return str(module_id or "").strip() or (Path(folder).name if folder is not None else "")


class YamlModuleRepository:
    """Modules from ``<root>/<module>/module.yaml``; the program files are only read.

    ``model_paths`` holds the model folder each user chose (``Import model...``); it wins over
    ``model.model_path`` of ``module.yaml``, which stays the default.
    """

    def __init__(self, modules_root: Path, model_paths=None):
        self.modules_root = Path(modules_root)
        self.model_paths = model_paths if model_paths is not None else InMemoryModelPathStore()

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
            axes = outputs.get("axes") if isinstance(outputs.get("axes"), dict) else {}
            output_spec = OutputSpec(
                type=str(outputs.get("type", "")),
                names=tuple(str(item) for item in outputs.get("names", ())),
                parameter_names=tuple(str(item) for item in outputs.get("parameter_names", ())),
                target_min=_tuple_of_floats(outputs.get("target_min")),
                target_max=_tuple_of_floats(outputs.get("target_max")),
                row_axis=_distribution_axis(axes.get("row")),
                column_axis=_distribution_axis(axes.get("column")),
            )
        else:
            output_spec = OutputSpec()
        raw_shape = io.get("input_shape")
        input_shape = (
            tuple(int(item) for item in raw_shape)
            if isinstance(raw_shape, list) and raw_shape
            else None
        )
        module_id = str(data.get("id", "") or "")
        chosen_model = self.model_paths.get(module_model_key(module_id, path.parent))
        return PredictionModule(
            id=module_id,
            name=str(data.get("name", "") or path.parent.name),
            framework=str(data.get("framework", "")),
            version=str(data.get("version", "")),
            folder=path.parent.resolve(),
            yaml_path=path.resolve(),
            model=ModelSpec(
                format=str(model.get("format", "")),
                path=chosen_model or str(model.get("model_path", "") or ""),
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
        """Remember ``model_path`` as this user's model of ``module``; ``module.yaml`` is not written."""
        key = module_model_key(module.id, module.folder)
        if not key:
            raise ValueError("Prediction module has no id or folder to remember its model by")
        self.model_paths.set(key, str(model_path))


# Backward-compatible import for callers that previously obtained this mapper
# from infrastructure. The implementation is application-owned and performs no I/O.
module_to_legacy_dict = describe_prediction_module
