"""Dataset Table coordination for Classification."""

from __future__ import annotations


import numpy as np

from PyQt5.QtCore import Qt


from PyQt5.QtWidgets import (
    QTableWidgetItem,
)

from src.gimap.features.classification.application import (
    ClassificationSample,
    ModelEvaluationResult,
    SavedModelPackage,
)


class DatasetTableMixin:
    """Own dataset table presentation behavior."""

    def _build_saved_model_package(self, result: ModelEvaluationResult) -> SavedModelPackage:
        algorithm = next(
            (
                config
                for config in self.algorithm_configs
                if config.algorithm_id == result.algorithm_id
            ),
            None,
        )
        data_type = self.feature_matrix.data_type if self.feature_matrix is not None else "unknown"
        input_shape = self.experiment_result.input_shape if self.experiment_result else None
        return self.classification_view_model.build_model_package(
            result,
            class_names=result.labels,
            data_type=data_type,
            input_shape=input_shape,
            preprocessing=self.experiment_result.preprocessing_config,
            projection=self.experiment_result.projection_config,
            algorithm_parameters=algorithm.parameters if algorithm else {},
            validation=self.experiment_result.validation_config,
        )

    def _update_dataset_table(self) -> None:
        page = self.page
        if page is None:
            return
        search = page.datasetSearchEdit.text().strip().lower()
        class_filter = page.classFilterCombo.currentText()
        qc_filter = page.qcFilterCombo.currentText()
        rows = []
        for sample in self.samples:
            if (
                search
                and search not in sample.file_name.lower()
                and search not in sample.file_path.lower()
            ):
                continue
            if class_filter != "All classes" and sample.label != class_filter:
                continue
            if qc_filter != "All QC" and sample.qc_status.lower() != qc_filter.lower():
                continue
            rows.append(sample)
        table = page.datasetTable
        self._table_updating = True
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))
        for row, sample in enumerate(rows):
            include_item = QTableWidgetItem("")
            include_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            include_item.setCheckState(Qt.Checked if sample.included else Qt.Unchecked)
            include_item.setData(Qt.UserRole, sample.sample_id)
            table.setItem(row, 0, include_item)
            values = [
                sample.label,
                sample.file_name,
                sample.data_type,
                self._shape_text(sample.raw_shape),
                sample.load_status,
                sample.qc_status,
                sample.predicted_label or "-",
                self._optional_float(sample.confidence),
            ]
            for offset, value in enumerate(values, start=1):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, sample.sample_id)
                if offset != 1:
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                table.setItem(row, offset, item)
        table.setSortingEnabled(True)
        self._table_updating = False
        self._update_class_filter()
        self._update_run_summary()

    def _on_dataset_item_changed(self, item: QTableWidgetItem) -> None:
        if self._table_updating or item.column() != 0:
            return
        sample = self._sample_by_id(item.data(Qt.UserRole))
        if sample is None:
            return
        sample.included = item.checkState() == Qt.Checked
        self.summary = self.classification_view_model.validate_dataset(self.samples)
        self._mark_results_outdated()
        self._refresh_everything()

    def _selected_sample_ids(self) -> list[str]:
        page = self.page
        ids: list[str] = []
        for index in page.datasetTable.selectionModel().selectedRows():
            item = page.datasetTable.item(index.row(), 0)
            if item is not None:
                ids.append(str(item.data(Qt.UserRole)))
        return ids

    def _preview_current_table_sample(self) -> None:
        page = self.page
        row = page.datasetTable.currentRow()
        if row < 0:
            return
        item = page.datasetTable.item(row, 0)
        if item is None:
            return
        sample = self._sample_by_id(item.data(Qt.UserRole))
        if sample is not None:
            self.current_preview_sample_id = sample.sample_id
            self._render_sample_preview(sample)

    def _render_current_preview(self) -> None:
        sample = self._sample_by_id(self.current_preview_sample_id)
        if sample is not None:
            self._render_sample_preview(sample)

    def _render_sample_preview(self, sample: ClassificationSample) -> None:
        page = self.page
        if page is None:
            return
        data = sample.raw_data
        page.sampleFileLabel.setText(sample.file_name)
        page.sampleShapeLabel.setText(self._shape_text(sample.raw_shape))
        loaded_samples = [item for item in self.samples if item.load_status == "loaded"]
        try:
            index = loaded_samples.index(sample) + 1
        except ValueError:
            index = 0
        page.sampleIndexLabel.setText(f"{index} / {len(loaded_samples)}")
        if data is None:
            self._set_graphics_text(page.previewGraphicsView, "Sample is not loaded.")
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
            if sample.data_type == "1D":
                arr = np.asarray(data)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    x, y = arr[:, 0], arr[:, 1]
                else:
                    y = arr.ravel()
                    x = np.arange(len(y))
                if page.previewLogScaleCheckBox.isChecked():
                    ax.semilogy(x, np.maximum(y, np.finfo(float).tiny), lw=1.0)
                else:
                    ax.plot(x, y, lw=1.0)
                ax.set_xlabel("q / index")
                ax.set_ylabel("Intensity")
                ax.grid(True, alpha=0.25)
            else:
                img = np.asarray(data, dtype=float)
                if page.previewLogScaleCheckBox.isChecked():
                    img = np.log1p(np.maximum(img, 0))
                vmin = vmax = None
                if page.previewAutoScaleCheckBox.isChecked():
                    vmin = float(np.nanpercentile(img, 0.5))
                    vmax = float(np.nanpercentile(img, 99.5))
                else:
                    vmin = float(page.previewVminSpinBox.value())
                    vmax = float(page.previewVmaxSpinBox.value())
                ax.imshow(
                    img,
                    cmap=page.previewColormapCombo.currentText(),
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                )
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            self._set_graphics_pixmap(page.previewGraphicsView, self._figure_to_pixmap(fig))
            plt.close(fig)
        except Exception as exc:
            self._set_graphics_text(page.previewGraphicsView, str(exc))
