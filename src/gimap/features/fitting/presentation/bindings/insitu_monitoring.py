"""Insitu Monitoring coordination for the fitting workspace."""

from __future__ import annotations


import time


import numpy as np

from PyQt5.QtCore import Qt, QTimer

from PyQt5.QtWidgets import (
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QDialog,
    QComboBox,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)


from ..binding_primitives import (
    GISAXS_IMAGE_COLORMAPS,
    is_matplotlib_available,
)


class InsituMonitoringMixin:
    """Own insitu monitoring presentation behavior."""

    def _create_selection_from_current_cut_controls(self):
        try:
            if not all(
                hasattr(self.ui, name)
                for name in (
                    "gisaxsInputCenterParallelValue",
                    "gisaxsInputCenterVerticalValue",
                    "gisaxsInputCutLineParallelValue",
                    "gisaxsInputCutLineVerticalValue",
                )
            ):
                return None
            return self._create_selection_from_parameters(
                float(self.ui.gisaxsInputCenterParallelValue.value()),
                float(self.ui.gisaxsInputCenterVerticalValue.value()),
                float(self.ui.gisaxsInputCutLineParallelValue.value()),
                float(self.ui.gisaxsInputCutLineVerticalValue.value()),
            )
        except Exception:
            return None

    def _draw_insitu_workflow_curve_preview(self):
        holder = getattr(self, "_insitu_workflow_canvas_curve", None)
        try:
            if holder is None or not hasattr(holder, "_insitu_figure"):
                return
            fig = holder._insitu_figure
            canvas = holder._insitu_canvas
            fig.clear()
            ax = fig.add_subplot(111)
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()
            filter_mode = self._get_independent_axis_filter_mode()

            # 函数说明：实现 prepare pair 相关逻辑。
            def prepare_pair(x_values, y_values, source="cut"):
                x = np.asarray(x_values, dtype=float).reshape(-1)
                y = np.asarray(y_values, dtype=float).reshape(-1)
                n = min(x.size, y.size)
                x, y = x[:n], y[:n]
                mask = np.isfinite(x) & np.isfinite(y)
                x, y = x[mask], y[mask]
                if filter_mode == "positive":
                    keep = x > 0
                    x, y = x[keep], y[keep]
                elif filter_mode == "negative":
                    keep = x < 0
                    x, y = np.abs(x[keep]), y[keep]
                    order = np.argsort(x)
                    x, y = x[order], y[order]
                if normalize and y.size:
                    max_y = float(np.nanmax(y))
                    if np.isfinite(max_y) and max_y > 0:
                        y = y / max_y
                x = self._convert_q_values_for_display(x, source=source)
                return x, y

            if getattr(self, "current_cut_data", None) is not None:
                x, y = prepare_pair(
                    self.current_cut_data.get("x_coords", []),
                    self.current_cut_data.get("y_intensity", []),
                    source="cut",
                )
                if x.size and y.size:
                    ax.plot(
                        x, y, marker="o", markersize=3, linewidth=1.4, color="#1f77b4", label="Cut"
                    )
            if isinstance(getattr(self, "fitting", None), dict):
                source = self.fitting.get("meta", {}).get("data_source", "cut")
                fx, fy = prepare_pair(
                    self.fitting.get("q", []), self.fitting.get("I", []), source=source
                )
                if fx.size and fy.size:
                    ax.plot(fx, fy, linewidth=2.0, color="#d62728", label="Fit")
            if log_x:
                ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")
            self._apply_fit_y_axis_limits(ax, log_y=log_y)
            self._draw_roi_guides_if_active(ax)
            ax.set_title("Cut / fitting curve")
            ax.set_xlabel(self._build_q_axis_label())
            ax.set_ylabel("Normalized Intensity" if normalize else "Intensity (a.u.)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout(pad=0.3)
            canvas.draw_idle()
        except Exception:
            pass

    def _reset_insitu_heatmap_data(self):
        self._insitu_heatmap_q = None
        self._insitu_heatmap_data = None
        self._insitu_heatmap_count = 0
        self._insitu_heatmap_capacity = 0
        self._insitu_heatmap_artist = None
        self._insitu_heatmap_colorbar = None
        self._schedule_insitu_heatmap_refresh(force=True)

    def _append_insitu_heatmap_cut(self, q_values, intensity_values):
        """Append a cut using a chunked matrix, interpolating onto the first q grid."""
        q = np.asarray(q_values, dtype=float).reshape(-1)
        values = np.asarray(intensity_values, dtype=float).reshape(-1)
        n = min(q.size, values.size)
        if n <= 0:
            return
        q, values = q[:n], values[:n]
        finite_q = np.isfinite(q)
        q, values = q[finite_q], values[finite_q]
        if q.size <= 0:
            return
        order = np.argsort(q, kind="mergesort")
        q, values = q[order], values[order]
        q, unique_indices = np.unique(q, return_index=True)
        values = values[unique_indices]
        if self._insitu_heatmap_q is None:
            self._insitu_heatmap_q = q.copy()
            self._insitu_heatmap_capacity = 256
            self._insitu_heatmap_data = np.full((256, q.size), np.nan, dtype=np.float32)
        target_q = np.asarray(self._insitu_heatmap_q, dtype=float)
        if q.size == target_q.size and np.allclose(q, target_q, rtol=1e-7, atol=1e-10):
            row = values
        else:
            valid = np.isfinite(values)
            row = (
                np.interp(target_q, q[valid], values[valid], left=np.nan, right=np.nan)
                if np.count_nonzero(valid) >= 2
                else np.full(target_q.shape, np.nan)
            )
        if self._insitu_heatmap_count >= self._insitu_heatmap_capacity:
            new_capacity = max(256, self._insitu_heatmap_capacity * 2)
            grown = np.full((new_capacity, target_q.size), np.nan, dtype=np.float32)
            grown[: self._insitu_heatmap_count] = self._insitu_heatmap_data[
                : self._insitu_heatmap_count
            ]
            self._insitu_heatmap_data = grown
            self._insitu_heatmap_capacity = new_capacity
        self._insitu_heatmap_data[self._insitu_heatmap_count] = np.asarray(row, dtype=np.float32)
        self._insitu_heatmap_count += 1
        self._schedule_insitu_heatmap_refresh()

    def _open_insitu_heatmap(self):
        existing = getattr(self, "_insitu_heatmap_dialog", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        if not is_matplotlib_available():
            QMessageBox.warning(
                self._insitu_workflow_parent_widget(), "Cut Heatmap", "Matplotlib is required."
            )
            return
        try:
            dialog = QDialog(self._insitu_workflow_parent_widget())
            dialog.setWindowTitle("In-situ Cut Heatmap")
            dialog.resize(980, 680)
            dialog.setModal(False)
            dialog.setAttribute(Qt.WA_DeleteOnClose, True)
            layout = QVBoxLayout(dialog)
            controls = QHBoxLayout()
            cmap = QComboBox(dialog)
            cmap.addItems(list(GISAXS_IMAGE_COLORMAPS))
            scale = QComboBox(dialog)
            scale.addItems(["Linear", "Log"])
            scale.setCurrentText("Log" if self._is_fit_log_y_enabled() else "Linear")
            auto_range = QCheckBox("Auto range", dialog)
            auto_range.setChecked(True)
            follow = QCheckBox("Follow latest", dialog)
            follow.setChecked(True)
            vmin, vmax = QDoubleSpinBox(dialog), QDoubleSpinBox(dialog)
            for spin in (vmin, vmax):
                spin.setDecimals(6)
                spin.setRange(-1e30, 1e30)
            vmax.setValue(1.0)
            reset_view = QPushButton("Reset view", dialog)
            for label, widget in (
                ("Colormap:", cmap),
                ("Color scale:", scale),
                ("", auto_range),
                ("vmin:", vmin),
                ("vmax:", vmax),
                ("", follow),
                ("", reset_view),
            ):
                if label:
                    controls.addWidget(QLabel(label, dialog))
                controls.addWidget(widget)
            layout.addLayout(controls)
            holder = self._make_insitu_workflow_canvas(dialog)
            layout.addWidget(holder, 1)
            status = QLabel("Waiting for auto-cut data...", dialog)
            layout.addWidget(status)
            self._insitu_heatmap_dialog = dialog
            self._insitu_heatmap_widgets = {
                "holder": holder,
                "cmap": cmap,
                "scale": scale,
                "auto": auto_range,
                "vmin": vmin,
                "vmax": vmax,
                "follow": follow,
                "status": status,
            }
            cmap.currentTextChanged.connect(lambda _text: self._refresh_insitu_heatmap())
            scale.currentTextChanged.connect(lambda _text: self._refresh_insitu_heatmap())
            auto_range.toggled.connect(self._on_insitu_heatmap_auto_range_toggled)
            vmin.valueChanged.connect(lambda _value: self._refresh_insitu_heatmap())
            vmax.valueChanged.connect(lambda _value: self._refresh_insitu_heatmap())
            reset_view.clicked.connect(lambda: self._refresh_insitu_heatmap(reset_view=True))
            follow.toggled.connect(lambda checked: self._refresh_insitu_heatmap(reset_view=checked))
            dialog.finished.connect(lambda _result: self._clear_insitu_heatmap_refs())
            self._on_insitu_heatmap_auto_range_toggled(True)
            self._refresh_insitu_heatmap(reset_view=True)
            dialog.show()
        except Exception as exc:
            self._log_insitu_workflow(f"Cut heatmap failed: {exc}", "ERROR")

    def _clear_insitu_heatmap_refs(self):
        self._insitu_heatmap_dialog = None
        self._insitu_heatmap_widgets = {}
        self._insitu_heatmap_artist = None
        self._insitu_heatmap_colorbar = None
        self._insitu_heatmap_refresh_pending = False

    def _on_insitu_heatmap_auto_range_toggled(self, checked: bool):
        widgets = getattr(self, "_insitu_heatmap_widgets", {}) or {}
        for key in ("vmin", "vmax"):
            if widgets.get(key) is not None:
                widgets[key].setEnabled(not checked)
        self._refresh_insitu_heatmap()

    def _schedule_insitu_heatmap_refresh(self, force: bool = False):
        dialog = getattr(self, "_insitu_heatmap_dialog", None)
        if dialog is None or not dialog.isVisible():
            return
        if force:
            self._insitu_heatmap_refresh_pending = False
            self._refresh_insitu_heatmap(reset_view=True)
        elif not self._insitu_heatmap_refresh_pending:
            self._insitu_heatmap_refresh_pending = True
            QTimer.singleShot(40, self._refresh_insitu_heatmap)

    def _refresh_insitu_heatmap(self, reset_view: bool = False):
        self._insitu_heatmap_refresh_pending = False
        widgets = getattr(self, "_insitu_heatmap_widgets", {}) or {}
        holder = widgets.get("holder")
        if holder is None or not hasattr(holder, "_insitu_figure"):
            return
        fig, canvas = holder._insitu_figure, holder._insitu_canvas
        count = int(getattr(self, "_insitu_heatmap_count", 0))
        q = getattr(self, "_insitu_heatmap_q", None)
        store = getattr(self, "_insitu_heatmap_data", None)
        status = widgets.get("status")
        if count <= 0 or q is None or store is None:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Waiting for auto-cut data...",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            self._insitu_heatmap_artist = None
            self._insitu_heatmap_colorbar = None
            if status is not None:
                status.setText("Waiting for auto-cut data...")
            canvas.draw_idle()
            return
        from matplotlib.colors import LogNorm, Normalize

        data = np.asarray(store[:count], dtype=float).T
        scale = widgets["scale"].currentText()
        display = np.where(data > 0, data, np.nan) if scale == "Log" else data
        finite = np.isfinite(display)
        values = display[finite]
        if values.size:
            if widgets["auto"].isChecked():
                lo, hi = np.nanpercentile(values, [1.0, 99.0])
            else:
                lo, hi = widgets["vmin"].value(), widgets["vmax"].value()
            if scale == "Log":
                lo = max(float(lo), float(np.nanmin(values)), np.finfo(float).tiny)
            if not np.isfinite(hi) or hi <= lo:
                hi = lo * 1.01 if scale == "Log" else lo + max(abs(lo) * 0.01, 1e-12)
        else:
            lo, hi = (1.0, 10.0) if scale == "Log" else (0.0, 1.0)
        norm = LogNorm(vmin=lo, vmax=hi) if scale == "Log" else Normalize(vmin=lo, vmax=hi)
        extent = (0.5, count + 0.5, float(np.nanmin(q)), float(np.nanmax(q)))
        artist = getattr(self, "_insitu_heatmap_artist", None)
        if artist is None or artist.axes not in fig.axes:
            fig.clear()
            ax = fig.add_subplot(111)
            artist = ax.imshow(
                display,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=extent,
                cmap=widgets["cmap"].currentText(),
                norm=norm,
            )
            ax.set_xlabel("Sequence number")
            ax.set_ylabel(r"$q$ (nm$^{-1}$)")
            self._insitu_heatmap_artist = artist
            self._insitu_heatmap_colorbar = fig.colorbar(artist, ax=ax, label="Intensity (a.u.)")
            fig.tight_layout()
        else:
            artist.set_data(display)
            artist.set_extent(extent)
            artist.set_cmap(widgets["cmap"].currentText())
            artist.set_norm(norm)
            if reset_view or widgets["follow"].isChecked():
                artist.axes.set_xlim(extent[0], extent[1])
                artist.axes.set_ylim(extent[2], extent[3])
            if self._insitu_heatmap_colorbar is not None:
                self._insitu_heatmap_colorbar.update_normal(artist)
        if status is not None:
            status.setText(
                f"{count} cut(s) | q points: {len(q)} | color range: {lo:.6g} to {hi:.6g}"
            )
        canvas.draw_idle()

    def _open_insitu_trend_monitor(self):
        try:
            rows = self._load_insitu_session_records()
            if not rows:
                QMessageBox.information(
                    self._insitu_workflow_parent_widget(),
                    "Trend Monitor",
                    "No in-situ workflow results yet.",
                )
                return
            existing = getattr(self, "_insitu_trend_dialog", None)
            if existing is not None and existing.isVisible():
                self._refresh_insitu_trend_monitor()
                existing.raise_()
                existing.activateWindow()
                return
            dialog = QDialog(self._insitu_workflow_parent_widget())
            dialog.setWindowTitle("In-situ Trend Monitor")
            dialog.resize(980, 620)
            layout = QVBoxLayout(dialog)
            combo = QComboBox(dialog)
            layout.addWidget(combo)
            table = QTableWidget(0, 0, dialog)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            layout.addWidget(table, 1)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            table.horizontalHeader().setStretchLastSection(True)
            plot_holder = None
            if is_matplotlib_available():
                plot_holder = self._make_insitu_workflow_canvas(dialog)
                layout.addWidget(plot_holder, 1)
            close = QPushButton("Close", dialog)
            close.clicked.connect(dialog.close)
            layout.addWidget(close)
            self._insitu_trend_dialog = dialog
            self._insitu_trend_table = table
            self._insitu_trend_combo = combo
            self._insitu_trend_plot_holder = plot_holder
            combo.currentTextChanged.connect(lambda _text: self._refresh_insitu_trend_plot())
            dialog.finished.connect(lambda _result: self._clear_insitu_trend_refs())
            dialog.show()
            self._refresh_insitu_trend_monitor()
        except Exception as exc:
            self._log_insitu_workflow(f"Trend monitor failed: {exc}", "ERROR")

    def _clear_insitu_trend_refs(self):
        self._insitu_trend_dialog = None
        self._insitu_trend_table = None
        self._insitu_trend_combo = None
        self._insitu_trend_plot_holder = None

    def _insitu_trend_parameter_keys(self, rows: list[dict]) -> list[str]:
        keys = ["chi_square"]
        skip = {
            "file_index",
            "file_name",
            "file_path",
            "timestamp",
            "run_mode",
            "load_status",
            "cut_status",
            "fit_status",
            "fitted_parameters",
            "error_message",
        }
        for row in rows:
            for key, value in row.items():
                if key in skip:
                    continue
                try:
                    float(value)
                except Exception:
                    continue
                if key not in keys:
                    keys.append(key)
        return keys

    def _refresh_insitu_trend_monitor(self):
        dialog = getattr(self, "_insitu_trend_dialog", None)
        table = getattr(self, "_insitu_trend_table", None)
        combo = getattr(self, "_insitu_trend_combo", None)
        if dialog is None or table is None or combo is None:
            return
        self._insitu_last_trend_refresh = time.perf_counter()
        self._insitu_trend_refresh_pending = False
        rows = self._load_insitu_session_records()
        parameter_keys = self._insitu_trend_parameter_keys(rows)
        current_key = combo.currentText()
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItems(parameter_keys)
            if current_key:
                idx = combo.findText(current_key)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(False)
        headers = ["file_index", "file_name", "fit_status", "error_message"] + parameter_keys
        table.setColumnCount(len(headers))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels(headers)
        for r, row in enumerate(rows):
            for c, key in enumerate(headers):
                table.setItem(r, c, QTableWidgetItem(str(row.get(key, ""))))
        self._refresh_insitu_trend_plot()

    def _schedule_insitu_trend_refresh(self, min_interval: float = 1.0):
        dialog = getattr(self, "_insitu_trend_dialog", None)
        if dialog is None:
            return
        now = time.perf_counter()
        last = float(getattr(self, "_insitu_last_trend_refresh", 0.0))
        if now - last >= min_interval:
            self._refresh_insitu_trend_monitor()
            return
        if getattr(self, "_insitu_trend_refresh_pending", False):
            return
        self._insitu_trend_refresh_pending = True
        delay_ms = max(50, int((min_interval - (now - last)) * 1000))
        QTimer.singleShot(delay_ms, self._refresh_insitu_trend_monitor)

    def _refresh_insitu_trend_plot(self):
        combo = getattr(self, "_insitu_trend_combo", None)
        plot_holder = getattr(self, "_insitu_trend_plot_holder", None)
        if combo is None or plot_holder is None or not hasattr(plot_holder, "_insitu_figure"):
            return
        rows = self._load_insitu_session_records()
        key = combo.currentText() or "chi_square"
        fig = plot_holder._insitu_figure
        canvas = plot_holder._insitu_canvas
        fig.clear()
        ax = fig.add_subplot(111)
        xs, ys = [], []
        for row in rows:
            try:
                y = float(row.get(key, ""))
                x = float(row.get("file_index", len(xs) + 1))
            except Exception:
                continue
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("file_index")
        ax.set_ylabel(key)
        ax.set_title(f"In-situ trend: {key}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(pad=0.3)
        canvas.draw_idle()
