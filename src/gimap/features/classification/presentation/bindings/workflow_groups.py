"""Active 1D/2D compatibility-group coordination."""

from __future__ import annotations


class WorkflowGroupsMixin:
    """Keep mixed imports usable while scientific runs stay type-compatible."""

    def _refresh_group_selector(self) -> None:
        page = self.page
        if page is None:
            return
        groups = self.classification_view_model.group_samples(self.samples)
        self.compatibility_groups = groups
        combo = page.dataGroupCombo
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for group in groups:
            shape_note = (
                "one shape" if len(group.shapes) == 1 else f"{len(group.shapes)} shapes"
            )
            combo.addItem(
                f"{group.data_type} · {group.total_samples} samples · {shape_note}",
                group.key,
            )
        if groups:
            keys = [group.key for group in groups]
            combo.setCurrentIndex(keys.index(current) if current in keys else 0)
            active = groups[combo.currentIndex()]
            page.dataGroupSummaryLabel.setText(
                f"{active.included_samples}/{active.total_samples} included"
            )
        else:
            page.dataGroupSummaryLabel.setText("No compatible data")
        combo.blockSignals(False)

    def _active_group_key(self) -> str | None:
        if self.page is None:
            return None
        value = self.page.dataGroupCombo.currentData()
        return str(value) if value else None

    def _active_samples(self):
        key = self._active_group_key()
        if not key:
            return []
        return [sample for sample in self.samples if sample.data_type == key]

    def _on_active_group_changed(self) -> None:
        if self.experiment_result is not None and (
            self._experiment_group_key != self._active_group_key()
        ):
            self._results_outdated = True
        self.embedding_payload = None
        if self.page is not None:
            self.page.embeddingScatterView.show_empty(
                "Run a reduction for the selected data group."
            )
            self.page.explorationStatusLabel.setText(
                "The active group changed; reduction and grouping are group-specific."
            )
        self._refresh_everything()
