"""Qt-free Fitting storage and optional-runtime commands."""

from __future__ import annotations

from pathlib import Path


class FittingStorageViewModel:
    def __init__(
        self,
        *,
        discover_insitu_frames=None,
        insitu_records,
        parameter_files,
        ai_artifacts,
        save_fitting_log,
        check_dependency,
        model_parameters=None,
        ai_catalog=None,
        export_curve_figure=None,
    ):
        self._discover_insitu_frames = discover_insitu_frames
        self._export_curve_figure = export_curve_figure
        self._insitu_records = insitu_records
        self._parameter_files = parameter_files
        self._ai_artifacts = ai_artifacts
        self._save_fitting_log = save_fitting_log
        self._check_dependency = check_dependency
        self.model_parameters = model_parameters
        self.ai_catalog = ai_catalog

    def discover_insitu_frames(self, request):
        if self._discover_insitu_frames is None:
            raise RuntimeError("In-situ frame discovery is not configured")
        return self._discover_insitu_frames.execute(request)

    def export_curve_figure(self, request):
        """Publication figure of the plotted curve; returns an ``OperationResult``."""
        if self._export_curve_figure is None:
            raise RuntimeError("Figure export is not configured")
        return self._export_curve_figure.execute(request)

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
