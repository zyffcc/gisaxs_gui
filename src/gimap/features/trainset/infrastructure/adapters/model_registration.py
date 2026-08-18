"""Register trained Trainset artifacts as existing Prediction modules."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

from ...application.models import (
    RegisterTrainsetModelRequest,
    RegisteredTrainsetModel,
)
from ...domain import trainable_parameter_names


PREPROCESSING_MODULE = """from __future__ import annotations
import copy
import numpy as np
from trainset.generator import apply_preprocessing, build_fixed_mask, crop_roi

def preprocess(image, preprocess_config, module_folder=None, return_steps=False):
    cfg = copy.deepcopy(preprocess_config["params"]["trainset_config"])
    roi_image = crop_roi(np.asarray(image, dtype=np.float32), cfg["roi"])
    mask = build_fixed_mask(roi_image, cfg)
    stages = apply_preprocessing(
        roi_image,
        cfg,
        mask,
        np.random.default_rng(int(cfg["project"]["seed"])),
    )
    result = np.asarray(stages[-1]["image"], dtype=np.float32)
    snapshots = [{"label": stage["name"], "image": stage["image"]} for stage in stages]
    return (result, snapshots) if return_steps else result
"""


class LocalTrainsetModelRegistrationAdapter:
    MODEL_NAMES = (
        "best_model.keras",
        "best_model.h5",
        "best_model.pt",
        "best_model.pth",
    )

    def find_model(self, roots: tuple[Path, ...]) -> Path | None:
        for root in roots:
            for name in self.MODEL_NAMES:
                candidate = Path(root) / name
                if candidate.exists():
                    return candidate
        return None

    def register(
        self, request: RegisterTrainsetModelRequest
    ) -> RegisteredTrainsetModel:
        config = request.config
        model_path = Path(request.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        slug = re.sub(
            r"[^a-zA-Z0-9_-]+", "_", str(config["project"]["name"])
        ).strip("_") or "trainset_model"
        module_dir = Path(request.modules_root) / f"generated_{slug}"
        module_dir.mkdir(parents=True, exist_ok=True)
        parameter_names = trainable_parameter_names(config)
        target_min = [
            float(config["parameters"][name]["minimum"])
            for name in parameter_names
        ]
        target_max = [
            float(config["parameters"][name]["maximum"])
            for name in parameter_names
        ]
        roi = config["roi"]
        inference_config = copy.deepcopy(config)
        excluded = {
            "noise",
            "gaussian_noise",
            "poisson_noise",
            "physical_background",
            "random_edge_crop",
        }
        for step in inference_config.get("preprocessing", {}).get("steps", []):
            if step.get("plugin") in excluded:
                step["enabled"] = False
        module_name = f"{config['project']['name']} (trained)"
        module = {
            "id": f"generated_{slug}",
            "name": module_name,
            "model": {
                "format": (
                    "pytorch"
                    if model_path.suffix.lower() in {".pt", ".pth"}
                    else "tensorflow_keras"
                ),
                "model_path": str(model_path.resolve()),
            },
            "preprocess": {
                "entry": "preprocessing:preprocess",
                "steps": [
                    step["plugin"]
                    for step in inference_config["preprocessing"]["steps"]
                    if step.get("enabled")
                ],
                "params": {"trainset_config": inference_config},
            },
            "io": {
                "input_shape": [
                    1,
                    int(roi["height"]),
                    int(roi["width"]),
                    1,
                ]
            },
            "outputs": {
                "type": "parameters",
                "parameter_names": parameter_names,
                "target_min": target_min,
                "target_max": target_max,
            },
        }
        (module_dir / "module.yaml").write_text(
            yaml.safe_dump(module, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        (module_dir / "preprocessing.py").write_text(
            PREPROCESSING_MODULE,
            encoding="utf-8",
        )
        return RegisteredTrainsetModel(module_name, module_dir)
