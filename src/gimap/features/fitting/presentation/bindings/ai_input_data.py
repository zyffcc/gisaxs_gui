"""Ai Input Data for fitting presentation."""

from __future__ import annotations


from PyQt5.QtWidgets import (
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QDialog,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)


class AiInputDataMixin:
    """Own ai input data behavior."""

    def _show_ai_input_data_dialog(self) -> None:
        arrays = self._current_ai_curve_arrays(apply_exclusions=False)
        if arrays is None:
            QMessageBox.warning(
                self.main_window or self.ui,
                "AI Input Data",
                "No valid AI input curve is loaded. Load or cut a 1D curve first.",
            )
            return

        existing = getattr(self, "_ai_input_data_dialog", None)
        if existing is not None and existing.isVisible():
            self._ai_input_dialog_arrays = arrays
            self._refresh_ai_input_data_dialog()
            existing.raise_()
            existing.activateWindow()
            return

        dialog = QDialog(self.main_window or self.ui)
        dialog.setWindowTitle("AI Input Data")
        dialog.resize(820, 560)
        dialog.setModal(False)
        layout = QVBoxLayout(dialog)

        summary = QLabel(dialog)
        summary.setWordWrap(True)
        layout.addWidget(summary)

        table = QTableWidget(0, 4, dialog)
        table.setHorizontalHeaderLabels(["Use", "q", "I", "sigma"])
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for col in range(1, 4):
            table.horizontalHeader().setSectionResizeMode(col, QHeaderView.Stretch)
        layout.addWidget(table, 1)

        # 函数说明：实现 selected rows 相关逻辑。
        def selected_rows() -> list[int]:
            selection = table.selectionModel()
            if selection is None:
                return []
            return sorted({index.row() for index in selection.selectedRows()})

        # 函数说明：实现 selected Q 值 相关逻辑。
        def selected_q_values() -> list[float]:
            dialog_arrays = getattr(self, "_ai_input_dialog_arrays", None)
            if dialog_arrays is None:
                return []
            q_arr, _, _ = dialog_arrays
            values = []
            for row in selected_rows():
                if 0 <= row < len(q_arr):
                    values.append(float(q_arr[row]))
            return values

        # 函数说明：删除selected。
        def delete_selected() -> None:
            values = selected_q_values()
            if not values:
                return
            self._exclude_ai_input_points(values, source="table")

        # 函数说明：恢复selected。
        def restore_selected() -> None:
            values = selected_q_values()
            if not values:
                return
            self._restore_ai_input_points(values)

        # 函数说明：恢复all。
        def restore_all() -> None:
            self._restore_all_ai_input_points()

        button_row = QHBoxLayout()
        delete_btn = QPushButton("Delete Selected", dialog)
        restore_btn = QPushButton("Restore Selected", dialog)
        restore_all_btn = QPushButton("Restore All", dialog)
        close_btn = QPushButton("Close", dialog)
        delete_btn.clicked.connect(delete_selected)
        restore_btn.clicked.connect(restore_selected)
        restore_all_btn.clicked.connect(restore_all)
        close_btn.clicked.connect(dialog.close)
        button_row.addWidget(delete_btn)
        button_row.addWidget(restore_btn)
        button_row.addWidget(restore_all_btn)
        button_row.addStretch(1)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

        # 函数说明：处理finished事件。
        def on_finished(_result):
            self._ai_input_data_dialog = None
            self._ai_input_data_table = None
            self._ai_input_data_summary = None
            self._ai_input_dialog_arrays = None

        dialog.finished.connect(on_finished)
        self._ai_input_data_dialog = dialog
        self._ai_input_data_table = table
        self._ai_input_data_summary = summary
        self._ai_input_dialog_arrays = arrays
        self._refresh_ai_input_data_dialog()
        dialog.show()

    def _refresh_ai_input_data_dialog(self) -> None:
        dialog = getattr(self, "_ai_input_data_dialog", None)
        table = getattr(self, "_ai_input_data_table", None)
        summary = getattr(self, "_ai_input_data_summary", None)
        arrays = getattr(self, "_ai_input_dialog_arrays", None)
        if dialog is None or table is None or summary is None or arrays is None:
            return
        try:
            q_arr, i_arr, sigma_arr = arrays
            excluded = getattr(self, "_ai_excluded_input_q", set()) or set()
            table.setRowCount(len(q_arr))
            for row, (q_val, i_val, sigma_val) in enumerate(zip(q_arr, i_arr, sigma_arr)):
                enabled = self._ai_q_key(q_val) not in excluded
                table.setItem(row, 0, QTableWidgetItem("Yes" if enabled else "No"))
                table.setItem(row, 1, QTableWidgetItem(f"{float(q_val):.8g}"))
                table.setItem(row, 2, QTableWidgetItem(f"{float(i_val):.8g}"))
                table.setItem(row, 3, QTableWidgetItem(f"{float(sigma_val):.8g}"))
            kept = sum(1 for q_val in q_arr if self._ai_q_key(q_val) not in excluded)
            removed = len(q_arr) - kept
            summary.setText(
                "Input points: "
                f"{len(q_arr)} | used: {kept} | excluded: {removed}. "
                "In Independent Fit Window, enable Delete Points and click a curve point to exclude it."
            )
        except Exception:
            pass

    def _exclude_ai_input_point_from_plot(self, q_value: float) -> None:
        self._exclude_ai_input_points([q_value], source="plot")

    def _exclude_ai_input_points(self, q_values, source: str = "table") -> None:
        excluded = set(getattr(self, "_ai_excluded_input_q", set()) or set())
        before = len(excluded)
        for q_value in q_values:
            excluded.add(self._ai_q_key(q_value))
            if source == "plot":
                try:
                    excluded.add(self._ai_q_key(abs(float(q_value))))
                except Exception:
                    pass
        self._ai_excluded_input_q = excluded
        added = max(0, len(excluded) - before)
        removed_from_current_cut = self._apply_deleted_point_mask_to_current_cut()
        self._refresh_ai_input_data_dialog()
        self._refresh_ai_input_outlier_views()
        self._draw_insitu_workflow_curve_preview()
        if added:
            label = "from Independent Fit Window" if source == "plot" else "from table"
            self._set_ai_workspace_status(f"Excluded {added} input point(s) {label}.", None)
        if removed_from_current_cut:
            self._add_fitting_message(
                f"Deleted-point mask applied to current cut: removed {removed_from_current_cut} point(s)",
                "INFO",
            )

    def _restore_ai_input_points(self, q_values) -> None:
        excluded = set(getattr(self, "_ai_excluded_input_q", set()) or set())
        before = len(excluded)
        for q_value in q_values:
            excluded.discard(self._ai_q_key(q_value))
            try:
                abs_q = abs(float(q_value))
                excluded.discard(self._ai_q_key(abs_q))
                excluded.discard(self._ai_q_key(-abs_q))
            except Exception:
                pass
        self._ai_excluded_input_q = excluded
        restored = max(0, before - len(excluded))
        self._refresh_ai_input_data_dialog()
        self._refresh_ai_input_outlier_views()
        self._draw_insitu_workflow_curve_preview()
        if restored:
            self._set_ai_workspace_status(f"Restored {restored} input point(s).", None)

    def _restore_all_ai_input_points(self) -> None:
        self._ai_excluded_input_q = set()
        self._refresh_ai_input_data_dialog()
        self._refresh_ai_input_outlier_views()
        self._draw_insitu_workflow_curve_preview()
        self._set_ai_workspace_status("All AI input points restored.", None)

    def _refresh_ai_input_outlier_views(self) -> None:
        try:
            mode = self.display_mode if hasattr(self, "display_mode") else "normal"
            self._update_GUI_image(mode)
            self._update_outside_window(mode)
        except Exception:
            pass
