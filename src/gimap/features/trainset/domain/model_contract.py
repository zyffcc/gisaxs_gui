"""Trainset 模型层配置与静态 tensor contract。"""

from __future__ import annotations

from typing import Any, Iterable


SUPPORTED_LAYER_TYPES = (
    "conv2d",
    "maxpool2d",
    "batch_normalization",
    "dropout",
    "global_average_pooling2d",
    "flatten",
    "dense",
)


def normalized_layers(model_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize current and schema-v1 model layer declarations."""
    layers = model_config.get("layers")
    if isinstance(layers, list) and layers:
        return [dict(layer) for layer in layers if isinstance(layer, dict)]

    output: list[dict[str, Any]] = []
    for channels in model_config.get("channels", [32, 64, 128]):
        output.append(
            {
                "type": "conv2d",
                "units": int(channels),
                "kernel": int(model_config.get("kernel_size", 3)),
                "activation": "relu",
            }
        )
        output.append({"type": "maxpool2d", "pool": 2})
    output.append({"type": "global_average_pooling2d"})
    if float(model_config.get("dropout", 0.0)) > 0:
        output.append({"type": "dropout", "rate": float(model_config["dropout"])})
    return output


def static_contract(
    input_shape: tuple[int, int, int],
    output_size: int,
    layers: Iterable[dict[str, Any]],
) -> str:
    """Describe the tensor contract without importing an ML runtime."""
    rows = [f"Input  {input_shape}"]
    spatial = True
    for index, spec in enumerate(layers, start=1):
        kind = str(spec.get("type", ""))
        rows.append(f"{index:02d}  {kind}")
        if kind in {"global_average_pooling2d", "flatten"}:
            spatial = False
        if kind == "dense" and spatial:
            rows.append("    ! Dense needs Flatten or GlobalAveragePooling2D first")
    if spatial:
        rows.append("Auto  global_average_pooling2d")
    rows.append(f"Output ({output_size},) regression parameters")
    return "\n".join(rows)
