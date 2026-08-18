"""Portable job-package adapter."""

from __future__ import annotations

from ...application.models import PrepareTrainsetJobRequest
from .portable_job_package import prepare_job_package


class PortableTrainsetJobPackageAdapter:
    def prepare(self, request: PrepareTrainsetJobRequest):
        return prepare_job_package(
            request.config,
            request.workspace,
            request.project_root,
        )
