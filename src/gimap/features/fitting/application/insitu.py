"""可序列化的 in-situ 文件序列 workflow。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import json
from typing import Callable, Literal, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .ports.insitu import SingleFileFitUseCase


WorkflowStatus = Literal[
    "idle", "running", "paused", "cancelled", "completed", "error"
]
FileStatus = Literal["running", "succeeded", "failed"]


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_mapping(value: Mapping[str, object] | None) -> dict[str, object]:
    result = dict(value or {})
    try:
        json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("In-situ worker output must be JSON serializable") from exc
    return result


@dataclass(frozen=True)
class InSituFileFitRequest:
    paths: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.paths or not all(str(path) for path in self.paths):
            raise ValueError("An in-situ fit request requires at least one path")
        object.__setattr__(self, "metadata", _json_mapping(self.metadata))


@dataclass(frozen=True)
class InSituFileFitResult:
    values: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _json_mapping(self.values))


@dataclass(frozen=True)
class InSituFileRecord:
    index: int
    paths: tuple[str, ...]
    status: FileStatus
    started_at: str
    finished_at: str | None = None
    values: Mapping[str, object] = field(default_factory=dict)
    error_message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _json_mapping(self.values))

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "paths": list(self.paths),
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "values": dict(self.values),
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "InSituFileRecord":
        return cls(
            index=int(value["index"]),
            paths=tuple(str(path) for path in value.get("paths", ())),
            status=str(value["status"]),  # type: ignore[arg-type]
            started_at=str(value["started_at"]),
            finished_at=(
                None if value.get("finished_at") is None else str(value["finished_at"])
            ),
            values=dict(value.get("values", {})),  # type: ignore[arg-type]
            error_message=str(value.get("error_message", "")),
        )


@dataclass(frozen=True)
class InSituWorkflowState:
    status: WorkflowStatus = "idle"
    pending_paths: tuple[str, ...] = ()
    current: InSituFileRecord | None = None
    records: tuple[InSituFileRecord, ...] = ()
    continue_on_error: bool = True

    @property
    def processed_count(self) -> int:
        return len(self.records)

    @property
    def failed_count(self) -> int:
        return sum(record.status == "failed" for record in self.records)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": self.status,
            "pending_paths": list(self.pending_paths),
            "current": None if self.current is None else self.current.to_dict(),
            "records": [record.to_dict() for record in self.records],
            "continue_on_error": self.continue_on_error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "InSituWorkflowState":
        version = int(value.get("schema_version", 1))
        if version != 1:
            raise ValueError(f"Unsupported in-situ state schema: {version}")
        current = value.get("current")
        raw_records = value.get("records", ())
        return cls(
            status=str(value.get("status", "idle")),  # type: ignore[arg-type]
            pending_paths=tuple(str(path) for path in value.get("pending_paths", ())),
            current=(
                InSituFileRecord.from_dict(current)
                if isinstance(current, Mapping)
                else None
            ),
            records=tuple(
                InSituFileRecord.from_dict(record)
                for record in raw_records  # type: ignore[union-attr]
                if isinstance(record, Mapping)
            ),
            continue_on_error=bool(value.get("continue_on_error", True)),
        )


@dataclass(frozen=True)
class InSituWorkflowRequest:
    paths: tuple[str, ...]
    continue_on_error: bool = True
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least one")


@dataclass(frozen=True)
class InSituProgress:
    processed: int
    failed: int
    total: int
    current_paths: tuple[str, ...] = ()
    message: str = ""

    @property
    def fraction(self) -> float:
        return 1.0 if self.total == 0 else min(1.0, self.processed / self.total)


class InSituWorkflowCoordinator:
    """管理序列状态；不读取文件、不执行算法，也不依赖 Qt。"""

    def __init__(self) -> None:
        self.state = InSituWorkflowState()

    def start(self, paths: Sequence[str], *, continue_on_error: bool = True) -> None:
        self.state = InSituWorkflowState(
            status="running",
            pending_paths=tuple(str(path) for path in paths),
            continue_on_error=continue_on_error,
        )

    def enqueue(self, paths: Sequence[str]) -> None:
        additions = tuple(str(path) for path in paths)
        self.state = replace(
            self.state,
            status="running" if self.state.status in ("idle", "completed") else self.state.status,
            pending_paths=self.state.pending_paths + additions,
        )

    def begin_next(self, batch_size: int = 1) -> InSituFileRecord | None:
        if self.state.status != "running" or self.state.current is not None:
            return None
        if not self.state.pending_paths:
            self.state = replace(self.state, status="completed")
            return None
        count = max(1, int(batch_size))
        paths = self.state.pending_paths[:count]
        record = InSituFileRecord(
            index=len(self.state.records) + 1,
            paths=paths,
            status="running",
            started_at=_timestamp(),
        )
        self.state = replace(
            self.state,
            pending_paths=self.state.pending_paths[count:],
            current=record,
        )
        return record

    def complete_current(
        self, values: Mapping[str, object] | None = None
    ) -> InSituFileRecord:
        return self._finish_current("succeeded", values=values)

    def fail_current(
        self, error_message: str, values: Mapping[str, object] | None = None
    ) -> InSituFileRecord:
        return self._finish_current(
            "failed", values=values, error_message=str(error_message)
        )

    def _finish_current(
        self,
        status: FileStatus,
        *,
        values: Mapping[str, object] | None,
        error_message: str = "",
    ) -> InSituFileRecord:
        if self.state.current is None:
            raise RuntimeError("No in-situ file is currently active")
        record = replace(
            self.state.current,
            status=status,
            finished_at=_timestamp(),
            values=_json_mapping(values),
            error_message=error_message,
        )
        next_status: WorkflowStatus = "running"
        if self.state.status in ("cancelled", "paused"):
            next_status = self.state.status
        elif status == "failed" and not self.state.continue_on_error:
            next_status = "error"
        elif not self.state.pending_paths:
            next_status = "completed"
        self.state = replace(
            self.state,
            status=next_status,
            current=None,
            records=self.state.records + (record,),
        )
        return record

    def pause(self) -> None:
        if self.state.status == "running":
            self.state = replace(self.state, status="paused")

    def resume(self) -> None:
        if self.state.status == "paused":
            self.state = replace(self.state, status="running")

    def cancel(self) -> None:
        if self.state.status not in ("idle", "completed", "cancelled"):
            self.state = replace(self.state, status="cancelled")

    def snapshot(self) -> dict[str, object]:
        return self.state.to_dict()

    def restore(self, snapshot: Mapping[str, object]) -> None:
        self.state = InSituWorkflowState.from_dict(snapshot)


class RunInSituWorkflow:
    """顺序调度单文件拟合，并聚合错误而不复制拟合算法。"""

    def __init__(
        self,
        single_file_fit: "SingleFileFitUseCase",
        coordinator: InSituWorkflowCoordinator | None = None,
    ):
        self._single_file_fit = single_file_fit
        self.coordinator = coordinator or InSituWorkflowCoordinator()

    def execute(
        self,
        request: InSituWorkflowRequest,
        on_progress: Callable[[InSituProgress], None] | None = None,
    ) -> InSituWorkflowState:
        self.coordinator.start(
            request.paths, continue_on_error=request.continue_on_error
        )
        total = len(request.paths)
        while self.coordinator.state.status == "running":
            current = self.coordinator.begin_next(request.batch_size)
            if current is None:
                break
            try:
                result = self._single_file_fit.execute(
                    InSituFileFitRequest(paths=current.paths)
                )
            except Exception as exc:
                self.coordinator.fail_current(str(exc))
            else:
                self.coordinator.complete_current(result.values)
            if on_progress is not None:
                state = self.coordinator.state
                on_progress(
                    InSituProgress(
                        processed=state.processed_count,
                        failed=state.failed_count,
                        total=total,
                        current_paths=current.paths,
                        message=state.records[-1].status,
                    )
                )
        return self.coordinator.state

    def cancel(self) -> None:
        self.coordinator.cancel()
