"""In-situ workflow 所需的单文件拟合边界。"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol, Sequence

from ..insitu import InSituFileFitRequest, InSituFileFitResult


class SingleFileFitUseCase(Protocol):
    """执行一次独立拟合；序列 workflow 不关心具体拟合算法。"""

    def execute(self, request: InSituFileFitRequest) -> InSituFileFitResult:
        ...


class InSituRecordRepository(Protocol):
    def cache_directory(self) -> Path: ...

    def session_path(self) -> Path: ...

    def ensure_directory(self) -> Path: ...

    def reset(self) -> None: ...

    def append(self, record: Mapping[str, object]) -> None: ...

    def load(self) -> list[dict[str, object]]: ...

    def export_csv(
        self,
        path: Path,
        rows: Sequence[Mapping[str, object]],
    ) -> Path: ...
