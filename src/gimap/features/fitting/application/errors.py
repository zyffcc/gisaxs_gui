"""Fitting application 的结构化错误。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


FileErrorCode = Literal[
    "not_found",
    "unsupported_format",
    "invalid_data",
    "permission_denied",
    "read_failed",
    "write_failed",
]


@dataclass(frozen=True)
class FileOperationError:
    code: FileErrorCode
    message: str
    path: str
    details: dict[str, object] = field(default_factory=dict)
