"""Stable Trainset parameter-range rules."""

from __future__ import annotations

from typing import Any


def trainable_parameter_names(config: dict[str, Any]) -> list[str]:
    return [
        name
        for name, spec in config.get("parameters", {}).items()
        if float(spec.get("maximum", 0.0)) > float(spec.get("minimum", 0.0))
    ]
