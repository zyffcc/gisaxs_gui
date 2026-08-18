"""Trainset 模型 runtime contract port。"""

from __future__ import annotations

from typing import Protocol

from ..models import ModelContractRequest, ModelContractResult


class ModelContractPort(Protocol):
    def validate(self, request: ModelContractRequest) -> ModelContractResult: ...
