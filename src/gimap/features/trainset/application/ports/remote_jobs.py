"""Remote Trainset job and local metric artifact ports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ..models import TrainsetRemoteJobStatus


class TrainsetRemoteJobPort(Protocol):
    def connection_check(self, config: dict[str, Any]) -> str: ...

    def upload_and_submit(
        self, config: dict[str, Any], package_dir: Path
    ) -> dict[str, str]: ...

    def query(
        self, config: dict[str, Any], job_id: str
    ) -> tuple[TrainsetRemoteJobStatus, str]: ...

    def download_results(
        self, config: dict[str, Any], destination: Path
    ) -> str: ...


class TrainsetMetricsRepository(Protocol):
    def load(self, path: Path) -> tuple[dict[str, Any], ...]: ...
