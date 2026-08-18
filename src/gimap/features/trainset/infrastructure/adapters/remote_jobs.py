"""Slurm and metrics adapters for Trainset application ports."""

from __future__ import annotations

from pathlib import Path

from .job_backends import SlurmBackend, read_metrics


class SlurmTrainsetRemoteJobAdapter:
    def connection_check(self, config):
        return SlurmBackend(config).connection_check()

    def upload_and_submit(self, config, package_dir):
        backend = SlurmBackend(config)
        backend.upload(package_dir)
        return backend.submit()

    def query(self, config, job_id):
        backend = SlurmBackend(config)
        return backend.query(job_id), backend.tail(job_id)

    def download_results(self, config, destination):
        return SlurmBackend(config).download_results(destination)


class LocalTrainsetMetricsRepository:
    def load(self, path: Path):
        return tuple(read_metrics(path))
