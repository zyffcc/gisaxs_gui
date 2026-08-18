"""Nested configuration field mapping for Trainset widgets."""

from __future__ import annotations


from typing import Any, Dict


def _deep_get(mapping: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if isinstance(value, list):
            if not part.isdigit() or int(part) >= len(value):
                return default
            value = value[int(part)]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value


def _deep_set(mapping: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    target: Any = mapping
    for part in parts[:-1]:
        if isinstance(target, list):
            target = target[int(part)]
        else:
            target = target.setdefault(part, {})
    if isinstance(target, list):
        target[int(parts[-1])] = value
    else:
        target[parts[-1]] = value
