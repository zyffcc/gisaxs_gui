"""Portable Trainset job package port."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..models import PrepareTrainsetJobRequest


class TrainsetJobPackagePort(Protocol):
    def prepare(self, request: PrepareTrainsetJobRequest) -> Path: ...
