"""Manual Refine Dialog for fitting presentation."""

from __future__ import annotations


import time


import numpy as np

from PyQt5.QtCore import Qt, QThread

from PyQt5.QtWidgets import (
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QDialog,
    QGridLayout,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)


from ..binding_primitives import (
    ManualAutoRefineWorker,
    RefineUiBridge,
)


class ManualRefineDialogMixin:
    """Own manual refine dialog behavior."""

    def _show_manual_auto_refine_dialog(self):
        """Open a local least-squares refine dialog based on current manual fitting parameters."""
        try:
            if not self.fitting_view_model.storage.dependency_available("scipy"):
                QMessageBox.warning(
                    self.main_window or self.ui,
                    "Auto Refine",
                    "SciPy is required for Auto Refine. Please install scipy first.",
                )
                return

            setup = self._build_manual_refine_setup()
            if setup is None:
                return

            dialog = QDialog(self.main_window or self.ui)
            dialog.setWindowTitle("Auto Refine Manual Fit")
            dialog.resize(980, 640)
            dialog.setModal(False)
            dialog.setAttribute(Qt.WA_DeleteOnClose, True)
            layout = QVBoxLayout(dialog)

            info = QLabel(
                "Choose parameters to refine. Current manual parameters are used as initial values; "
                "refined values will be written back to the fitting controls.",
                dialog,
            )
            info.setWordWrap(True)
            layout.addWidget(info)

            run_settings = self._ai_run_settings()
            controls = QGridLayout()
            controls.addWidget(QLabel("Max eval:", dialog), 0, 0)
            max_eval = QSpinBox(dialog)
            max_eval.setRange(1, 100000)
            max_eval.setValue(int(run_settings.get("full_refine_max_nfev", 120)))
            controls.addWidget(max_eval, 0, 1)

            controls.addWidget(QLabel("Target logRMSE:", dialog), 0, 2)
            target = QDoubleSpinBox(dialog)
            target.setDecimals(8)
            target.setRange(0.0, 10.0)
            target.setSingleStep(0.00000001)
            target.setValue(float(run_settings.get("full_refine_target_logrmse", 0.0)))
            controls.addWidget(target, 0, 3)

            controls.addWidget(QLabel("ftol:", dialog), 1, 0)
            ftol = QDoubleSpinBox(dialog)
            ftol.setDecimals(10)
            ftol.setRange(0.0, 1.0)
            ftol.setSingleStep(0.00000001)
            ftol.setValue(float(run_settings.get("full_refine_ftol", 1e-8)))
            controls.addWidget(ftol, 1, 1)

            controls.addWidget(QLabel("xtol:", dialog), 1, 2)
            xtol = QDoubleSpinBox(dialog)
            xtol.setDecimals(10)
            xtol.setRange(0.0, 1.0)
            xtol.setSingleStep(0.00000001)
            xtol.setValue(float(run_settings.get("full_refine_xtol", 1e-8)))
            controls.addWidget(xtol, 1, 3)

            controls.addWidget(QLabel("gtol:", dialog), 1, 4)
            gtol = QDoubleSpinBox(dialog)
            gtol.setDecimals(10)
            gtol.setRange(0.0, 1.0)
            gtol.setSingleStep(0.00000001)
            gtol.setValue(float(run_settings.get("full_refine_gtol", 1e-8)))
            controls.addWidget(gtol, 1, 5)

            controls.addWidget(QLabel("Progress every nfev:", dialog), 2, 0)
            progress_every = QSpinBox(dialog)
            progress_every.setRange(1, 10000)
            progress_every.setValue(
                max(1, int(run_settings.get("full_refine_progress_interval", 5) or 5))
            )
            progress_every.setToolTip(
                "Update progress every N estimated SciPy least_squares function evaluations."
            )
            controls.addWidget(progress_every, 2, 1)
            controls.addWidget(QLabel("Show every:", dialog), 2, 2)
            show_every = QSpinBox(dialog)
            show_every.setRange(0, 10000)
            show_every.setValue(10)
            show_every.setToolTip(
                "Update the Fitting Plot every N estimated SciPy least_squares function evaluations; 0 disables live plot updates."
            )
            controls.addWidget(show_every, 2, 3)
            controls.setColumnStretch(6, 1)
            layout.addLayout(controls)

            table = QTableWidget(len(setup["params"]), 5, dialog)
            table.setHorizontalHeaderLabels(["Refine", "Parameter", "Current", "Min", "Max"])
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.setSelectionMode(QAbstractItemView.SingleSelection)
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            for col in range(2, 5):
                table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
            layout.addWidget(table, 1)

            row_widgets = []
            cached_rows = self._manual_refine_dialog_state()
            initializing_rows = True
            for row, desc in enumerate(setup["params"]):
                value = float(desc["value"])
                default_selected = self._manual_refine_default_selected(desc["name"])
                lower, upper = self._default_manual_refine_bounds(desc["name"], value)
                cached = (
                    cached_rows.get(str(desc["name"]), {}) if isinstance(cached_rows, dict) else {}
                )
                if isinstance(cached, dict):
                    default_selected = bool(cached.get("checked", default_selected))
                    try:
                        lower = float(cached.get("min", lower))
                        upper = float(cached.get("max", upper))
                    except Exception:
                        pass

                check = QCheckBox(table)
                check.setChecked(default_selected)
                table.setCellWidget(row, 0, check)
                table.setItem(row, 1, QTableWidgetItem(str(desc["label"])))
                table.setItem(row, 2, QTableWidgetItem(f"{value:.10g}"))

                min_box = QDoubleSpinBox(table)
                max_box = QDoubleSpinBox(table)
                for spin in (min_box, max_box):
                    spin.setDecimals(8)
                    spin.setRange(-1e12, 1e12)
                    spin.setSingleStep(max(abs(value) * 0.01, 1e-8))
                min_box.setValue(float(lower))
                max_box.setValue(float(upper))
                table.setCellWidget(row, 3, min_box)
                table.setCellWidget(row, 4, max_box)
                row_widgets.append((desc, check, min_box, max_box))
            initializing_rows = False

            # 函数说明：实现 row 状态 相关逻辑。
            def persist_row_state():
                if initializing_rows:
                    return
                rows = {}
                for desc, check, min_box, max_box in row_widgets:
                    rows[str(desc["name"])] = {
                        "checked": bool(check.isChecked()),
                        "min": float(min_box.value()),
                        "max": float(max_box.value()),
                    }
                self._save_manual_refine_dialog_state(rows)

            for _desc, check, min_box, max_box in row_widgets:
                check.toggled.connect(lambda _checked=False: persist_row_state())
                min_box.valueChanged.connect(lambda _value: persist_row_state())
                max_box.valueChanged.connect(lambda _value: persist_row_state())

            result_label = QLabel("Ready.", dialog)
            result_label.setWordWrap(True)
            layout.addWidget(result_label)
            progress_bar = QProgressBar(dialog)
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            layout.addWidget(progress_bar)

            button_row = QHBoxLayout()
            select_all = QPushButton("Select All", dialog)
            clear = QPushButton("Clear", dialog)
            run = QPushButton("Run Refine", dialog)
            stop = QPushButton("Stop", dialog)
            stop.setEnabled(False)
            apply_current = QPushButton("Apply Current", dialog)
            apply_current.setEnabled(False)
            close = QPushButton("Close", dialog)
            button_row.addWidget(select_all)
            button_row.addWidget(clear)
            button_row.addStretch(1)
            button_row.addWidget(run)
            button_row.addWidget(stop)
            button_row.addWidget(apply_current)
            button_row.addWidget(close)
            layout.addLayout(button_row)

            # 函数说明：设置selected。
            def set_selected(predicate):
                for desc, check, _min_box, _max_box in row_widgets:
                    check.setChecked(bool(predicate(desc)))
                persist_row_state()

            select_all.clicked.connect(lambda: set_selected(lambda _desc: True))
            clear.clicked.connect(lambda: set_selected(lambda _desc: False))
            close.clicked.connect(dialog.close)
            refine_state = {
                "thread": None,
                "worker": None,
                "bridge": None,
                "latest_result": None,
                "running": False,
                "status": "Idle",
                "last_ui_update": 0.0,
                "last_preview_update": 0.0,
            }

            # 函数说明：设置running 状态。
            def set_running_state(running: bool, status: str = None):
                refine_state["running"] = bool(running)
                if status:
                    refine_state["status"] = status
                run.setEnabled(not running)
                stop.setEnabled(running)
                apply_current.setEnabled(refine_state["latest_result"] is not None and not running)
                close.setEnabled(not running)
                for widget in (
                    select_all,
                    clear,
                    table,
                    max_eval,
                    target,
                    ftol,
                    xtol,
                    gtol,
                    progress_every,
                    show_every,
                ):
                    widget.setEnabled(not running)

            # 函数说明：应用结果。
            def apply_result(result):
                if not result:
                    return
                self._apply_manual_refine_result(
                    setup, result["params"], apply_indices=result.get("selected_indices")
                )
                self._perform_manual_fitting()
                selected_for_display = result.get("selected_indices")
                selected_for_display = (
                    {int(idx) for idx in selected_for_display}
                    if selected_for_display is not None
                    else None
                )
                for row, value in enumerate(result["params"]):
                    if selected_for_display is not None and row not in selected_for_display:
                        continue
                    table.setItem(row, 2, QTableWidgetItem(f"{float(value):.10g}"))
                self._add_fitting_success(
                    f"Applied Auto Refine parameters: logRMSE={float(result.get('final_log_rmse', np.nan)):.6g}"
                )

            # 函数说明：处理progress事件。
            def on_progress(payload):
                refine_state["latest_result"] = payload
                max_nfev = max(1, int(payload.get("max_nfev", max_eval.value())))
                nfev = int(payload.get("nfev_est", payload.get("nfev", payload.get("calls", 0))))
                calls = int(payload.get("calls", 0))
                now = time.perf_counter()
                progress_bar.setValue(max(0, min(99, int(100 * nfev / max_nfev))))
                if now - float(refine_state.get("last_ui_update", 0.0)) >= 0.3 or nfev <= 1:
                    refine_state["last_ui_update"] = now
                    result_label.setText(
                        f"Running: nfev~{nfev}/{max_nfev}, residual calls={calls}, "
                        f"current logRMSE={float(payload.get('current_log_rmse', np.nan)):.6g}, "
                        f"best={float(payload.get('final_log_rmse', payload.get('best_log_rmse', np.nan))):.6g}"
                    )
                show_interval = int(payload.get("show_interval", show_every.value()) or 0)
                preview_due = (
                    show_interval > 0
                    and nfev > 0
                    and (nfev == 1 or nfev % show_interval == 0)
                    and (now - float(refine_state.get("last_preview_update", 0.0))) >= 0.75
                )
                if preview_due:
                    refine_state["last_preview_update"] = now
                    self._preview_manual_refine_curve(setup, payload.get("params"))

            # 函数说明：实现 finish worker 相关逻辑。
            def finish_worker():
                thread = refine_state.get("thread")
                worker = refine_state.get("worker")
                bridge = refine_state.get("bridge")
                if thread is not None:
                    thread.quit()
                if worker is not None:
                    worker.deleteLater()
                if bridge is not None:
                    bridge.deleteLater()
                refine_state["thread"] = None
                refine_state["worker"] = None
                refine_state["bridge"] = None
                set_running_state(False)

            # 函数说明：处理finished事件。
            def on_finished(result):
                refine_state["latest_result"] = result
                progress_bar.setValue(100 if not result.get("stopped") else progress_bar.value())
                if result.get("stopped"):
                    result_label.setText(
                        f"Stopped: best logRMSE {result['initial_log_rmse']:.6g} -> {result['final_log_rmse']:.6g}; "
                        "click Apply Current to save the current best parameters."
                    )
                    self._add_fitting_warning(
                        "Auto Refine stopped. Current best parameters are available to apply."
                    )
                else:
                    apply_result(result)
                    result_label.setText(
                        f"Done: logRMSE {result['initial_log_rmse']:.6g} -> {result['final_log_rmse']:.6g}; "
                        f"nfev={result['nfev']}; {result['message']}"
                    )
                    self._add_fitting_success(result_label.text())
                finish_worker()

            # 函数说明：处理failed事件。
            def on_failed(message):
                result_label.setText(f"Auto Refine failed: {message}")
                self._add_fitting_error(f"Auto Refine failed: {message}")
                finish_worker()

            # 函数说明：停止refine。
            def stop_refine():
                worker = refine_state.get("worker")
                if worker is not None:
                    worker.request_stop()
                    refine_state["status"] = "Stopping"
                    result_label.setText(
                        "Stopping Auto Refine after the current residual evaluation..."
                    )
                    stop.setEnabled(False)

            stop.clicked.connect(stop_refine)
            apply_current.clicked.connect(lambda: apply_result(refine_state.get("latest_result")))

            # 函数说明：处理对话框 finished事件。
            def on_dialog_finished(_result):
                worker = refine_state.get("worker")
                if worker is not None:
                    worker.request_stop()

            dialog.finished.connect(on_dialog_finished)

            # 函数说明：实现 run refine 相关逻辑。
            def run_refine():
                try:
                    options = {
                        "max_nfev": int(max_eval.value()),
                        "target_logrmse": float(target.value()),
                        "ftol": float(ftol.value()) if ftol.value() > 0 else None,
                        "xtol": float(xtol.value()) if xtol.value() > 0 else None,
                        "gtol": float(gtol.value()) if gtol.value() > 0 else None,
                        "progress_interval": int(progress_every.value()),
                        "show_interval": int(show_every.value()),
                        "min_progress_seconds": 0.5,
                    }
                    persist_row_state()
                    self._save_ai_fitting_settings(
                        full_refine_max_nfev=int(max_eval.value()),
                        full_refine_target_logrmse=float(target.value()),
                        full_refine_ftol=float(ftol.value()),
                        full_refine_xtol=float(xtol.value()),
                        full_refine_gtol=float(gtol.value()),
                        full_refine_progress_interval=int(progress_every.value()),
                    )
                    selected = []
                    for desc, check, min_box, max_box in row_widgets:
                        if not check.isChecked():
                            continue
                        lo = float(min_box.value())
                        hi = float(max_box.value())
                        if hi <= lo:
                            raise ValueError(f"{desc['label']} max must be greater than min.")
                        selected.append((desc, lo, hi))
                    if not selected:
                        QMessageBox.information(
                            dialog, "Auto Refine", "Select at least one parameter to refine."
                        )
                        return
                    refine_state["latest_result"] = None
                    progress_bar.setValue(0)
                    result_label.setText("Refining...")
                    thread = QThread(dialog)
                    worker = ManualAutoRefineWorker(self, setup, selected, options)
                    bridge = RefineUiBridge(dialog)
                    bridge.progress.connect(on_progress)
                    bridge.finished.connect(on_finished)
                    bridge.failed.connect(on_failed)
                    worker.moveToThread(thread)
                    thread.started.connect(worker.run)
                    worker.progress.connect(bridge.progress)
                    worker.finished.connect(bridge.finished)
                    worker.failed.connect(bridge.failed)
                    worker.finished.connect(thread.quit)
                    worker.failed.connect(lambda _message: thread.quit())
                    thread.finished.connect(thread.deleteLater)
                    refine_state["thread"] = thread
                    refine_state["worker"] = worker
                    refine_state["bridge"] = bridge
                    set_running_state(True, "Running")
                    thread.start()
                except Exception as exc:
                    result_label.setText(f"Auto Refine failed: {exc}")
                    self._add_fitting_error(f"Auto Refine failed: {exc}")

            run.clicked.connect(run_refine)
            dialog.finished.connect(
                lambda _result: setattr(self, "_manual_auto_refine_dialog", None)
            )
            self._manual_auto_refine_dialog = dialog
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()

        except Exception as e:
            self._add_fitting_error(f"Failed to open Auto Refine: {e}")
