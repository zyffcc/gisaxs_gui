"""Trainset project configuration policy port."""

from __future__ import annotations

from typing import Any, Protocol


class TrainsetConfigurationPort(Protocol):
    def default(self) -> dict[str, Any]: ...

    def merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]: ...

    def synchronize(self, config: dict[str, Any]) -> dict[str, Any]: ...

    def validate(self, config: dict[str, Any], **options): ...
