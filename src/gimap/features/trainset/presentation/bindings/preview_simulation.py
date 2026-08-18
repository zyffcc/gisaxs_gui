"""Preview Simulation coordination for Trainset."""

from __future__ import annotations

import copy


from typing import Any, Dict


from PyQt5.QtCore import QTimer

from PyQt5.QtWidgets import (
    QMessageBox,
)

from src.gimap.features.trainset.application import (
    TrainsetPreviewRequest,
    TrainsetWhatIfRequest,
)


from ..background_tasks import _FunctionWorker
from ..config_fields import _deep_set


class PreviewSimulationMixin:
    """Own preview simulation presentation behavior."""

    def _new_preview_realization(self) -> None:
        if self._preview_busy:
            self.status_updated.emit("A simulated comparison is already running")
            return
        self._preview_realization += 1
        self._start_preview(force=False)

    def _generate_preview(self) -> None:
        self._start_preview(force=False)

    def _start_preview(self, force: bool = False) -> None:
        if self._preview_busy:
            self.status_updated.emit("A simulated comparison is already running")
            return
        config = self._collect_config()
        self._refresh_impact_options(config)
        valid, errors, warnings = self.trainset_view_model.validate_config(
            config,
            require_reference=False,
            simulation_available=self.simulation_port.is_available(),
        )
        if not valid:
            QMessageBox.warning(self.window, "Preview blocked", "\n".join(errors))
            return
        if not self.simulation_port.is_available():
            QMessageBox.warning(
                self.window,
                "Preview failed",
                "BornAgain is required because Local Preview displays simulated training images, not the experimental reference.",
            )
            return
        plugin, key, minimum, maximum = self._impact_range(config)
        compared_text = self.page.impact_parameter_combo.currentText()
        self._preview_busy = True
        self.page.set_preview_busy(True, 2, "Preparing the simulated comparison…")
        self.progress_updated.emit(2)
        worker = _FunctionWorker(
            self._compute_preview,
            copy.deepcopy(config),
            plugin,
            key,
            minimum,
            maximum,
            compared_text,
            self.page.preview_count.value(),
            self._preview_realization,
            warnings,
            force,
            _with_progress=True,
        )
        worker.signals.progress.connect(self._preview_progressed)
        worker.signals.finished.connect(self._preview_finished)
        worker.signals.error.connect(self._preview_failed)
        self._preview_worker = worker
        self.thread_pool.start(worker)

    def _compute_preview(
        self,
        progress,
        config: Dict[str, Any],
        plugin: str,
        key: str,
        minimum: float,
        maximum: float,
        compared_text: str,
        preview_count: int,
        realization: int,
        warnings,
        force: bool,
    ) -> Dict[str, Any]:
        result = self.trainset_view_model.generate_preview(
            TrainsetPreviewRequest(
                config=config,
                plugin=plugin,
                key=key,
                minimum=minimum,
                maximum=maximum,
                compared_text=compared_text,
                preview_count=preview_count,
                realization=realization,
                warnings=tuple(warnings),
                force=force,
            ),
            on_progress=progress,
        )
        if result is None:
            raise RuntimeError(
                self.trainset_view_model.state.error_message or "Trainset preview failed"
            )
        return result

    def _preview_progressed(self, progress: int, message: str) -> None:
        self.page.set_preview_progress(progress, message)
        self.progress_updated.emit(progress)
        self.status_updated.emit(message)

    def _preview_finished(self, result: Dict[str, Any]) -> None:
        self._preview_busy = False
        self._preview_worker = None
        self.page.set_simulation_preview(
            result["comparison_images"],
            result["comparison_labels"],
            result["stages"],
            result["stats"],
            result["spectrum_x"],
            result["spectrum_y"],
        )
        self.page.set_comparison_details(
            result["comparison_details"],
            self.config.get("parameters", {}),
            copy.deepcopy(self.config),
        )
        cache_hits = int(result["cache_hits"])
        cache_misses = int(result["cache_misses"])
        self.page.preview_cache_status.setText(
            f"BornAgain cache: {int(result['cache_size'])} image(s) · last update {cache_hits} hit / {cache_misses} rerun"
        )
        particle = next(iter(self.config.get("sample", {}).get("particles", [])), {})
        form_factor_names = list(particle.get("parameters", {}))
        self.page.set_parameter_samples(
            result["parameter_samples"],
            form_factor_names,
            self.config.get("parameters", {}),
        )
        self.page.preview_gate_table.item(0, 1).setText("Ready")
        self.page.preview_gate_table.item(1, 1).setText("Ready")
        self.page.preview_gate_table.item(2, 1).setText("Ready")
        self._storage_acceptance_changed(self.page.storage_accept_check.isChecked())
        self.page.validation_badge.setText("Preview ready")
        self.page.set_step_state(1, "Preview ready")
        self.page.set_preview_busy(
            False, 100, "Preview ready. The GUI remained responsive during simulation."
        )
        self.progress_updated.emit(100)
        self.status_updated.emit("BornAgain simulation impact preview generated")

    def _preview_failed(self, message: str) -> None:
        self._preview_busy = False
        self._preview_worker = None
        self.page.set_preview_busy(False, 0, f"Preview failed: {message}")
        QMessageBox.warning(self.window, "Preview failed", message)
        self.generation_error.emit(message)

    def _start_what_if(self, values: Dict[str, Any]) -> None:
        physics_payload = values.get("physics", values)
        overrides = values.get("overrides", {})
        numeric = {str(key): float(value) for key, value in physics_payload.items()}
        request = {"physics": numeric, "overrides": copy.deepcopy(overrides)}
        if self._what_if_busy:
            self._pending_what_if_values = request
            self.page.set_what_if_busy(
                True,
                "Current simulation is finishing · the latest edit is queued.",
            )
            return
        config = self._collect_config()
        for path, value in overrides.items():
            _deep_set(config, str(path), value)
        config = self.trainset_view_model.synchronize_config(config)
        self._what_if_busy = True
        self._pending_what_if_values = None
        self.page.set_what_if_busy(True, "Running manual simulation in the background…")
        worker = _FunctionWorker(
            self._compute_what_if,
            copy.deepcopy(config),
            numeric,
            self._preview_realization,
        )
        worker.signals.finished.connect(self._what_if_finished)
        worker.signals.error.connect(self._what_if_failed)
        self._what_if_worker = worker
        self.thread_pool.start(worker)

    def _compute_what_if(
        self,
        config: Dict[str, Any],
        sampled: Dict[str, float],
        realization: int,
    ) -> Dict[str, Any]:
        result = self.trainset_view_model.simulate_what_if(
            TrainsetWhatIfRequest(config, sampled, realization)
        )
        if result is None:
            raise RuntimeError(
                self.trainset_view_model.state.error_message or "Manual trainset simulation failed"
            )
        return result

    def _what_if_finished(self, result: Dict[str, Any]) -> None:
        self._what_if_busy = False
        self._what_if_worker = None
        values = ", ".join(f"{key}={value:.5g}" for key, value in result["values"].items())
        cache_text = "BornAgain cache reused" if result["cache_hit"] else "BornAgain recomputed"
        self.page.set_what_if_result(
            result["image"],
            f"{cache_text} · {values}\nPipeline: {result['pipeline']}",
        )
        self._run_pending_what_if()

    def _what_if_failed(self, message: str) -> None:
        self._what_if_busy = False
        self._what_if_worker = None
        self.page.set_what_if_busy(False, f"Manual simulation not completed: {message}")
        self._run_pending_what_if()

    def _run_pending_what_if(self) -> None:
        pending = self._pending_what_if_values
        self._pending_what_if_values = None
        if pending is not None:
            QTimer.singleShot(0, lambda values=pending: self._start_what_if(values))
