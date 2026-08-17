"""In-situ workflow 所需的单文件拟合边界。"""

from __future__ import annotations

from typing import Protocol

from ..insitu import InSituFileFitRequest, InSituFileFitResult


class SingleFileFitUseCase(Protocol):
    """执行一次独立拟合；序列 workflow 不关心具体拟合算法。"""

    def execute(self, request: InSituFileFitRequest) -> InSituFileFitResult:
        ...
