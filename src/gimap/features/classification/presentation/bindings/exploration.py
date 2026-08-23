"""Reduction, grouping suggestions, and human label curation."""

from __future__ import annotations

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QTableWidgetItem

from src.gimap.features.classification.application import ClassificationPageState
from src.gimap.features.classification.presentation.workers import (
    ClusteringWorker,
    EmbeddingWorker,
)


_PALETTE = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
    "#65a30d",
    "#a16207",
)


class ExplorationMixin:
    """Own the interactive explore-and-label loop."""

    def _start_embedding(self) -> None:
        if self.current_worker is not None:
            QMessageBox.information(
                self.main_window, "Explore data", "Another Classification task is running."
            )
            return
        samples = self._active_samples()
        if len(samples) < 2:
            QMessageBox.warning(
                self.main_window,
                "Explore data",
                "Import at least two compatible samples before running a reduction.",
            )
            return
        worker = EmbeddingWorker(
            samples,
            self._collect_preprocessing_config(),
            self.page.embeddingMethodCombo.currentText(),
            self.classification_view_model,
        )
        self.current_worker = worker
        self._set_state(ClassificationPageState.EXPLORING)
        self.page.explorationStatusLabel.setText("Computing reduction…")
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.finished.connect(self._on_exploration_embedding_finished)
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    def _on_exploration_embedding_finished(self, payload) -> None:
        self.current_worker = None
        if not isinstance(payload, dict) or "embedding" not in payload:
            self._on_worker_error("Reduction returned an invalid payload.")
            return
        self.embedding_payload = payload
        self._render_exploration_embedding()
        self._set_state(ClassificationPageState.READY)
        self.page.explorationStatusLabel.setText(
            f"{payload.get('method', 'Reduction')} complete · "
            f"{len(payload['matrix'].samples)} samples. Select points to inspect or label."
        )
        self.log(f"[Explore] {payload.get('method', 'Reduction')} complete.")

    def _render_exploration_embedding(self) -> None:
        if not self.embedding_payload or self.page is None:
            return
        matrix = self.embedding_payload["matrix"]
        samples = list(matrix.samples)
        keys = [self._embedding_color_key(sample) for sample in samples]
        unique = sorted(set(keys))
        colors_by_key = {
            key: _PALETTE[index % len(_PALETTE)] for index, key in enumerate(unique)
        }
        tooltips = [
            "\n".join(
                (
                    sample.file_name,
                    f"Accepted: {sample.label or 'Unlabeled'}",
                    f"Suggestion: {sample.suggested_label or '-'}",
                    f"QC: {sample.qc_status}",
                )
            )
            for sample in samples
        ]
        self.page.embeddingScatterView.set_points(
            self.embedding_payload["embedding"],
            [sample.sample_id for sample in samples],
            [colors_by_key[key] for key in keys],
            tooltips,
        )
        self._update_exploration_selection([])

    def _embedding_color_key(self, sample) -> str:
        mode = self.page.embeddingColorCombo.currentText()
        if mode == "Suggested group":
            return sample.suggested_label or "No suggestion"
        if mode == "Source":
            return sample.source_name or "Unknown source"
        if mode == "QC status":
            return sample.qc_status.title()
        if mode == "Prediction":
            return sample.predicted_label or "No prediction"
        return sample.label or "Unlabeled"

    def _start_clustering(self) -> None:
        if self.current_worker is not None:
            QMessageBox.information(
                self.main_window, "Suggest groups", "Another Classification task is running."
            )
            return
        samples = self._active_samples()
        if len(samples) < 2:
            QMessageBox.warning(
                self.main_window, "Suggest groups", "At least two compatible samples are required."
            )
            return
        worker = ClusteringWorker(
            samples,
            self._collect_preprocessing_config(),
            self.page.clusterMethodCombo.currentText(),
            self.page.clusterCountSpinBox.value(),
            self.page.minClusterSizeSpinBox.value(),
            self.classification_view_model,
        )
        self.current_worker = worker
        self._set_state(ClassificationPageState.CLUSTERING)
        self.page.explorationStatusLabel.setText("Computing grouping suggestions…")
        worker.signals.finished.connect(self._on_clustering_finished)
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    def _on_clustering_finished(self, payload) -> None:
        self.current_worker = None
        if not isinstance(payload, dict) or "labels" not in payload:
            self._on_worker_error("Grouping returned an invalid payload.")
            return
        labels = np.asarray(payload["labels"]).astype(int)
        samples = list(payload["matrix"].samples)
        method = str(payload.get("method", "Grouping"))
        for sample, cluster_id in zip(samples, labels):
            sample.suggested_label = (
                "Noise / review" if cluster_id < 0 else f"Group {cluster_id + 1}"
            )
            sample.suggestion_source = method
        self._set_state(ClassificationPageState.READY)
        self.page.embeddingColorCombo.setCurrentText("Suggested group")
        self._render_exploration_embedding()
        groups = len({int(value) for value in labels if int(value) >= 0})
        noise = int(np.sum(labels < 0))
        self.page.explorationStatusLabel.setText(
            f"{method} proposed {groups} groups; {noise} samples need individual review. "
            "Suggestions are not training labels until accepted."
        )
        self.log(f"[Explore] {method} proposed {groups} groups ({noise} noise).")

    def _update_exploration_selection(self, sample_ids) -> None:
        ids = list(sample_ids)
        self.page.selectedCountLabel.setText(f"{len(ids)} samples selected")
        samples = [sample for sample in self.samples if sample.sample_id in set(ids)]
        table = self.page.selectionTable
        table.setRowCount(len(samples))
        for row, sample in enumerate(samples):
            values = (
                sample.file_name,
                sample.label or "Unlabeled",
                sample.suggested_label or "-",
                sample.qc_status.title(),
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, sample.sample_id)
                table.setItem(row, col, item)
        if len(samples) == 1:
            self._render_exploration_preview(samples[0])

    def _selected_embedding_ids(self) -> list[str]:
        return self.page.embeddingScatterView.selected_sample_ids()

    def _assign_selected_label(self) -> None:
        sample_ids = self._selected_embedding_ids()
        label = self.page.labelEdit.text().strip()
        if not sample_ids:
            QMessageBox.information(self.main_window, "Assign label", "Select samples first.")
            return
        try:
            result = self.classification_view_model.assign_labels(
                self.samples, sample_ids, label, source="manual"
            )
        except ValueError as exc:
            QMessageBox.warning(self.main_window, "Assign label", str(exc))
            return
        self.page.labelEdit.clear()
        self._labels_changed(f"Assigned '{result.label}' to {len(result.sample_ids)} samples.")

    def _clear_selected_labels(self) -> None:
        sample_ids = self._selected_embedding_ids()
        if not sample_ids:
            QMessageBox.information(self.main_window, "Clear labels", "Select samples first.")
            return
        self.classification_view_model.clear_labels(self.samples, sample_ids)
        self._labels_changed(f"Marked {len(sample_ids)} samples as unlabeled.")

    def _accept_suggestions(self, *, all_samples: bool = False) -> None:
        sample_ids = () if all_samples else tuple(self._selected_embedding_ids())
        if not all_samples and not sample_ids:
            QMessageBox.information(
                self.main_window, "Accept suggestions", "Select suggested samples first."
            )
            return
        updated = self.classification_view_model.accept_suggestions(
            self._active_samples(), sample_ids
        )
        self._labels_changed(f"Accepted suggestions for {len(updated)} samples.")

    def _labels_changed(self, message: str) -> None:
        self._mark_results_outdated()
        self._refresh_everything()
        self._render_exploration_embedding()
        self._update_exploration_selection(self._selected_embedding_ids())
        self._persist_parameters()
        self.page.explorationStatusLabel.setText(message)
        self.log(f"[Labels] {message}")

    def _preview_activated_embedding_sample(self, sample_id: str) -> None:
        sample = self._sample_by_id(sample_id)
        if sample is not None:
            self._render_exploration_preview(sample)

    def _preview_selected_exploration_row(self) -> None:
        row = self.page.selectionTable.currentRow()
        if row < 0:
            return
        item = self.page.selectionTable.item(row, 0)
        sample = self._sample_by_id(item.data(Qt.UserRole) if item else None)
        if sample is not None:
            self._render_exploration_preview(sample)

    def _render_exploration_preview(self, sample) -> None:
        self.page.explorationSampleLabel.setText(
            f"{sample.file_name} · {sample.label or 'Unlabeled'} · {self._shape_text(sample.raw_shape)}"
        )
        if sample.raw_data is None:
            self._set_graphics_text(self.page.explorationPreviewView, "Sample is not loaded.")
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            data = np.asarray(sample.raw_data)
            fig, ax = plt.subplots(figsize=(4.2, 2.2), dpi=110)
            if sample.data_type == "1D":
                if data.ndim == 2 and data.shape[1] >= 2:
                    ax.plot(data[:, 0], data[:, 1], lw=1.0)
                else:
                    ax.plot(data.ravel(), lw=1.0)
                ax.grid(True, alpha=0.2)
            else:
                ax.imshow(np.log1p(np.maximum(data, 0)), cmap="viridis", origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            self._set_graphics_pixmap(
                self.page.explorationPreviewView, self._figure_to_pixmap(fig)
            )
            plt.close(fig)
        except Exception as exc:
            self._set_graphics_text(self.page.explorationPreviewView, str(exc))
