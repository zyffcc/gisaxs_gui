"""Verified Classification workflow progress rendering."""

from __future__ import annotations

from src.gimap.features.classification.application import ClassificationPageState


class WorkflowProgressMixin:
    """Derive step states from artifacts while navigation remains independent."""

    def _update_workflow_header(self) -> None:
        page = self.page
        if page is None:
            return
        active = list(self._active_samples())
        included = [sample for sample in active if sample.included]
        accepted = [
            sample
            for sample in included
            if sample.label and sample.label_status == "accepted"
        ]
        accepted_classes = {sample.label for sample in accepted}
        suggested = [sample for sample in included if sample.suggested_label]
        group_count = len(self.compatibility_groups)
        data_type = active[0].data_type if active else "data"

        if self.state == ClassificationPageState.IMPORTING:
            page.set_workflow_step_state("Data", "running", "Reading and checking files…")
        elif self.samples:
            groups = f"{group_count} group" if group_count == 1 else f"{group_count} groups"
            page.set_workflow_step_state(
                "Data", "complete", f"{len(self.samples)} samples · {groups}"
            )
        elif self.sources:
            page.set_workflow_step_state(
                "Data", "available", f"{len(self.sources)} sources ready to scan"
            )
        else:
            page.set_workflow_step_state("Data", "available", "Add files or folders")

        if included:
            page.set_workflow_step_state(
                "Prepare", "complete", f"{data_type} · {len(included)} included"
            )
        else:
            page.set_workflow_step_state("Prepare", "blocked", "Import data first")

        if self.state in {
            ClassificationPageState.EXPLORING,
            ClassificationPageState.CLUSTERING,
        }:
            message = (
                "Building the 2D map…"
                if self.state == ClassificationPageState.EXPLORING
                else "Building suggestions…"
            )
            page.set_workflow_step_state("Explore", "running", message)
        elif self.embedding_payload is not None:
            detail = f"{len(accepted)} accepted"
            if suggested:
                detail += f" · {len(suggested)} suggested"
            page.set_workflow_step_state("Explore", "complete", detail)
        elif len(included) >= 2:
            page.set_workflow_step_state("Explore", "available", "Build a 2D map")
        else:
            page.set_workflow_step_state("Explore", "blocked", "Need 2 compatible samples")

        if self.state == ClassificationPageState.TRAINING:
            page.set_workflow_step_state("Train", "running", "Training selected models…")
        elif self.experiment_result is not None and self._results_outdated:
            page.set_workflow_step_state("Train", "stale", "Settings or labels changed")
        elif self.experiment_result is not None:
            successful = len(self.experiment_result.successful_results)
            page.set_workflow_step_state(
                "Train", "complete", f"{successful} model results ready"
            )
        elif len(accepted_classes) >= 2:
            page.set_workflow_step_state(
                "Train", "available", f"{len(accepted_classes)} accepted classes"
            )
        else:
            page.set_workflow_step_state(
                "Train", "blocked", "Accept labels for 2+ classes"
            )

        if self.state == ClassificationPageState.PREDICTING:
            page.set_workflow_step_state("Apply", "running", "Classifying new data…")
        elif self.prediction_results:
            page.set_workflow_step_state(
                "Apply", "complete", f"{len(self.prediction_results)} predictions ready"
            )
        elif self.active_model_package is not None or self.active_result is not None:
            page.set_workflow_step_state("Apply", "available", "Active model ready")
        else:
            page.set_workflow_step_state("Apply", "blocked", "Train or load a model")

        if self.state == ClassificationPageState.ERROR:
            current = getattr(page, "_current_step", "Data")
            page.set_workflow_step_state(current, "error", "Last operation failed")


__all__ = ["WorkflowProgressMixin"]
