"""Qt-free Fitting storage and optional-runtime commands."""

from __future__ import annotations

from pathlib import Path

from ..application import LoadScatteringFileRequest


class FittingStorageViewModel:
    def __init__(
        self,
        *,
        load_scattering_file,
        inspect_scattering_sequence,
        scattering_loader_factory,
        remote_file_cache,
        insitu_records,
        parameter_files,
        ai_artifacts,
        save_fitting_log,
        check_dependency,
        model_parameters=None,
        ai_catalog=None,
    ):
        self._load_scattering_file = load_scattering_file
        self._inspect_scattering_sequence = inspect_scattering_sequence
        self._scattering_loader_factory = scattering_loader_factory
        self._remote_file_cache = remote_file_cache
        self._insitu_records = insitu_records
        self._parameter_files = parameter_files
        self._ai_artifacts = ai_artifacts
        self._save_fitting_log = save_fitting_log
        self._check_dependency = check_dependency
        self.model_parameters = model_parameters
        self.ai_catalog = ai_catalog

    def load_scattering_background(
        self,
        request: LoadScatteringFileRequest,
        *,
        prepare_path=None,
        on_progress=None,
    ):
        loader = self._load_scattering_file
        if self._scattering_loader_factory is not None:
            loader = self._scattering_loader_factory(
                prepare_path=prepare_path,
                progress=on_progress,
            )
        return loader.execute(request)

    def inspect_scattering_sequence(self, path: Path):
        return self._inspect_scattering_sequence.execute(Path(path))

    def default_remote_cache_directory(self) -> str:
        return self._remote_file_cache.default_directory()

    def display_remote_cache_directory(self, cache_dir: str) -> str:
        return self._remote_file_cache.display_directory(cache_dir)

    def resolve_remote_cache_directory(self, cache_dir: str) -> str:
        return str(self._remote_file_cache.resolve_directory(cache_dir))

    def is_remote_source(self, path: str) -> bool:
        return self._remote_file_cache.is_remote(path)

    def remote_cache_target(self, source_path: str, cache_dir: str) -> str:
        return str(self._remote_file_cache.target_path(source_path, cache_dir))

    def prepare_remote_source(
        self,
        source_path: str,
        cache_dir: str,
        max_gb: float,
        *,
        on_progress=None,
        is_cancelled=None,
    ) -> str:
        return str(
            self._remote_file_cache.prepare(
                source_path,
                cache_dir,
                max_gb,
                on_progress=on_progress,
                is_cancelled=is_cancelled,
            )
        )

    def clear_remote_cache(self, cache_dir: str) -> int:
        return self._remote_file_cache.clear(cache_dir)

    def insitu_cache_directory(self) -> Path:
        return self._insitu_records.cache_directory()

    def insitu_session_path(self) -> Path:
        return self._insitu_records.session_path()

    def ensure_insitu_cache_directory(self) -> Path:
        return self._insitu_records.ensure_directory()

    def reset_insitu_records(self) -> None:
        self._insitu_records.reset()

    def append_insitu_record(self, record) -> None:
        self._insitu_records.append(record)

    def load_insitu_records(self):
        return self._insitu_records.load()

    def export_insitu_records(self, path: Path, rows) -> Path:
        return self._insitu_records.export_csv(Path(path), rows)

    def save_parameter_snapshot(self, path: Path, values) -> Path:
        return self._parameter_files.save_snapshot(Path(path), values)

    def load_parameter_snapshot(self, path: Path):
        return self._parameter_files.load_snapshot(Path(path))

    def export_model_parameters(self, source: Path, destination: Path) -> Path:
        return self._parameter_files.export_model_parameters(source, destination)

    def import_model_parameters(self, source: Path, destination: Path) -> Path:
        return self._parameter_files.import_model_parameters(source, destination)

    def has_ai_output(self, output_dir: Path) -> bool:
        return self._ai_artifacts.has_output(Path(output_dir))

    def append_ai_log(self, output_dir: Path, text: str) -> Path:
        return self._ai_artifacts.append_log(Path(output_dir), text)

    def export_ai_output(
        self,
        output_dir: Path,
        parent_dir: Path,
        timestamp: str,
    ) -> Path:
        return self._ai_artifacts.export_output(
            Path(output_dir),
            Path(parent_dir),
            timestamp,
        )

    def save_fitting_log(self, path: Path, content: str) -> Path:
        return self._save_fitting_log.execute(Path(path), content)

    def dependency_available(self, distribution: str) -> bool:
        return self._check_dependency.execute(distribution)
