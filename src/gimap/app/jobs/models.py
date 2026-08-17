"""可跨进程传输的 Job 数据协议。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from uuid import uuid4


JobStatus = Literal["succeeded", "failed", "cancelled", "timed_out"]


def ensure_serializable(value: Any, label: str = "value") -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must contain only JSON-serializable data: {exc}") from exc


@dataclass(frozen=True)
class JobRequest:
    handler: str
    payload: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None
    job_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if ":" not in self.handler:
            raise ValueError("Job handler must use 'module:function' syntax.")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("Job timeout must be greater than zero.")
        ensure_serializable(self.payload, "JobRequest.payload")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JobRequest":
        return cls(**value)


@dataclass(frozen=True)
class JobProgress:
    job_id: str
    completed: float
    total: float
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_serializable(self.details, "JobProgress.details")

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.completed / self.total))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JobProgress":
        return cls(**value)


@dataclass(frozen=True)
class JobError:
    code: str
    message: str
    exception_type: str = ""
    traceback: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_serializable(self.details, "JobError.details")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JobError":
        return cls(**value)


@dataclass(frozen=True)
class JobResult:
    job_id: str
    status: JobStatus
    value: Any = None
    error: JobError | None = None
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        ensure_serializable(self.value, "JobResult.value")
        if self.status == "succeeded" and self.error is not None:
            raise ValueError("A successful JobResult cannot contain an error.")
        if self.status != "succeeded" and self.error is None:
            raise ValueError("A non-successful JobResult must contain an error.")

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JobResult":
        payload = dict(value)
        error = payload.get("error")
        if isinstance(error, dict):
            payload["error"] = JobError.from_dict(error)
        return cls(**payload)
