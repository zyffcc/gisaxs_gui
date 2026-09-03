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

    def _show_manual_auto_refine_dialog(self, search_mode: str = "local"):
        """Open bounded global-search or local-refinement controls."""
        try:
            search_mode = "global" if str(search_mode).lower() == "global" else "local"
            is_global = search_mode == "global"
            action_name = "Global Search" if is_global else "Local Refine"
            if not self.fitting_view_model.storage.dependency_available("scipy"):
                QMessageBox.warning(
                    self.main_window or self.ui,
                    action_name,
                    f"SciPy is required for {action_name}. Please install scipy first.",
                )
                return

            setup = self._build_manual_refine_setup()
            if setup is None:
                return

            dialog = QDialog(self.main_window or self.ui)
            dialog.setObjectName(
                "manualGlobalSearchDialog" if is_global else "manualAutoRefineDialog"
            )
            dialog.setWindowTitle(
                "Global Search + Local Refine" if is_global else "Local Refine Manual Fit"
            )
            dialog.resize(1040 if is_global else 980, 680 if is_global else 640)
            dialog.setModal(False)
            dialog.setAttribute(Qt.WA_DeleteOnClose, True)
            layout = QVBoxLayout(dialog)

            source_label = (
                "current cut" if setup.get("q_source_kind") == "cut" else "imported 1D data"
            )
            info = QLabel(
                f"Input: {source_label} ({len(setup['q_raw'])} fitting points). "
                + (
                    "Differential evolution explores the broad editable ranges, profiles linear "
                    "amplitudes, then locally refines the best candidates."
                    if is_global
                    else "Polish the current values inside conservative editable local ranges. "
                    "The optimizer uses normalized coordinates to avoid false xtol stops."
                ),
                dialog,
            )
            info.setObjectName("manualAutoRefineInputSummary")
            info.setWordWrap(True)
            layout.addWidget(info)

            run_settings = self._ai_run_settings()
            stored_settings = self._manual_parameter_search_settings()
            controls = QGridLayout()
            controls.addWidget(
                QLabel("Max local eval/start:" if is_global else "Max eval:", dialog),
                0,
                0,
            )
            max_eval = QSpinBox(dialog)
            max_eval.setRange(1, 100000)
            max_eval.setValue(int(run_settings.get("full_refine_max_nfev", 120)))
            controls.addWidget(max_eval, 0, 1)

            controls.addWidget(QLabel("Target logRMSE:", dialog), 0, 2)
            target = QDoubleSpinBox(dialog)
            target.setObjectName("manualTargetLogRmseSpinBox")
            target.setDecimals(8)
            target.setRange(0.0, 10.0)
            target.setSingleStep(0.00000001)
            target.setValue(
                float(stored_settings.get("target_logrmse", 0.0))
                if is_global
                else float(run_settings.get("full_refine_target_logrmse", 0.0))
            )
            target.setToolTip(
                "0 runs the full global evaluation budget; set a positive value to stop early."
                if is_global
                else "Stop local refinement once this logRMSE is reached; 0 disables it."
            )
            controls.addWidget(target, 0, 3)

            global_samples_label = QLabel("Global evaluations:", dialog)
            global_samples = QSpinBox(dialog)
            global_samples.setObjectName("manualGlobalSamplesSpinBox")
            global_samples.setRange(8, 65536)
            global_samples.setValue(int(stored_settings.get("global_samples", 16384)))
            global_samples.setToolTip("Approximate differential-evolution curve-evaluation budget.")
            controls.addWidget(global_samples_label, 0, 4)
            controls.addWidget(global_samples, 0, 5)

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

            global_starts_label = QLabel("Local starts:", dialog)
            global_starts = QSpinBox(dialog)
            global_starts.setObjectName("manualGlobalStartsSpinBox")
            global_starts.setRange(1, 32)
            global_starts.setValue(int(stored_settings.get("global_starts", 3)))
            global_starts.setToolTip(
                "Number of best global candidates polished with local least-squares."
            )
            controls.addWidget(global_starts_label, 2, 4)
            controls.addWidget(global_starts, 2, 5)
            global_samples_label.setVisible(is_global)
            global_samples.setVisible(is_global)
            global_starts_label.setVisible(is_global)
            global_starts.setVisible(is_global)
            controls.setColumnStretch(6, 1)
            layout.addLayout(controls)

            table = QTableWidget(len(setup["params"]), 5, dialog)
            table.setObjectName("manualAutoRefineParameterTable")
            table.setHorizontalHeaderLabels(["Refine", "Parameter", "Current", "Min", "Max"])
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.setSelectionMode(QAbstractItemView.SingleSelection)
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            for col in range(2, 5):
                table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
            layout.addWidget(table, 1)

            row_widgets = []
            cached_rows = self._manual_refine_dialog_state(search_mode)
            initializing_rows = True
            for row, desc in enumerate(setup["params"]):
                value = float(desc["value"])
                default_selected = (
                    self._manual_global_default_selected(desc["name"])
                    if is_global
                    else self._manual_refine_default_selected(desc["name"])
                )
                if is_global:
                    lower, upper = self._default_manual_global_bounds(
                        desc["name"],
                        value,
                        setup.get("y"),
                        setup.get("q_model"),
                    )
                else:
                    lower, upper = self._default_manual_refine_bounds(desc["name"], value)
                cached = self._matching_manual_refine_cached_row(desc, cached_rows)
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
                parameter_item = QTableWidgetItem(str(desc["label"]))
                parameter_item.setFlags(parameter_item.flags() & ~Qt.ItemIsEditable)
                current_item = QTableWidgetItem(f"{value:.10g}")
                current_item.setFlags(current_item.flags() & ~Qt.ItemIsEditable)
                table.setItem(row, 1, parameter_item)
                table.setItem(row, 2, current_item)

                min_box = QDoubleSpinBox(table)
                max_box = QDoubleSpinBox(table)
                for spin in (min_box, max_box):
                    spin.setDecimals(12)
                    spin.setRange(-1e12, 1e12)
                    spin.setSingleStep(max(abs(value) * 0.01, 1e-12))
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
                        "current_value": float(desc["value"]),
                        "min": float(min_box.value()),
                        "max": float(max_box.value()),
                    }
                self._save_manual_refine_dialog_state(rows, search_mode)

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
            run = QPushButton("Run Global Search" if is_global else "Run Local Refine", dialog)
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
                    global_samples,
                    global_starts,
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
                    value = float(value)
                    setup["params"][row]["value"] = value
                    current_item = QTableWidgetItem(f"{value:.10g}")
                    current_item.setFlags(current_item.flags() & ~Qt.ItemIsEditable)
                    table.setItem(row, 2, current_item)
                persist_row_state()
                self._add_fitting_success(
                    f"Applied {action_name} parameters: "
                    f"logRMSE={float(result.get('final_log_rmse', np.nan)):.6g}"
                )

            # 函数说明：处理progress事件。
            def on_progress(payload):
                refine_state["latest_result"] = payload
                max_nfev = max(1, int(payload.get("max_nfev", max_eval.value())))
                nfev = int(payload.get("nfev_est", payload.get("nfev", payload.get("calls", 0))))
                calls = int(payload.get("calls", 0))
                work_total = max(1, int(payload.get("work_total", max_nfev)))
                work_done = int(payload.get("work_done", nfev))
                phase = str(payload.get("phase", "local_refine"))
                if phase == "global_search":
                    phase_label = "Differential evolution"
                elif payload.get("mode") == "global":
                    phase_label = (
                        f"Local refine {int(payload.get('local_start', 0))}/"
                        f"{int(payload.get('global_starts', 1))}"
                    )
                else:
                    phase_label = "Local refine"
                now = time.perf_counter()
                progress_bar.setValue(max(0, min(99, int(100 * work_done / work_total))))
                if now - float(refine_state.get("last_ui_update", 0.0)) >= 0.3 or nfev <= 1:
                    refine_state["last_ui_update"] = now
                    result_label.setText(
                        f"{phase_label}: work {work_done}/{work_total}, "
                        f"local nfev~{nfev}, model calls={calls}, "
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
                        f"{action_name} stopped. Current best parameters are available to apply."
                    )
                else:
                    apply_result(result)
                    initial_score = float(result["initial_log_rmse"])
                    final_score = float(result["final_log_rmse"])
                    improvement = (
                        100.0 * (initial_score - final_score) / initial_score
                        if np.isfinite(initial_score) and initial_score > 0
                        else 0.0
                    )
                    result_label.setText(
                        f"Done: logRMSE {initial_score:.6g} -> {final_score:.6g} "
                        f"({improvement:.1f}% better); local nfev={result['nfev']}; "
                        f"{result['message']}"
                    )
                    self._add_fitting_success(result_label.text())
                finish_worker()

            # 函数说明：处理failed事件。
            def on_failed(message):
                result_label.setText(f"{action_name} failed: {message}")
                self._add_fitting_error(f"{action_name} failed: {message}")
                finish_worker()

            # 函数说明：停止refine。
            def stop_refine():
                worker = refine_state.get("worker")
                if worker is not None:
                    worker.request_stop()
                    refine_state["status"] = "Stopping"
                    result_label.setText(
                        f"Stopping {action_name} after the current model evaluation..."
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
                        "mode": search_mode,
                        "max_nfev": int(max_eval.value()),
                        "global_samples": int(global_samples.value()),
                        "global_starts": int(global_starts.value()),
                        "random_seed": 1729,
                        "target_logrmse": float(target.value()),
                        "ftol": float(ftol.value()) if ftol.value() > 0 else None,
                        "xtol": float(xtol.value()) if xtol.value() > 0 else None,
                        "gtol": float(gtol.value()) if gtol.value() > 0 else None,
                        "progress_interval": int(progress_every.value()),
                        "show_interval": int(show_every.value()),
                        "min_progress_seconds": 0.5,
                    }
                    persist_row_state()
                    ai_setting_updates = {
                        "full_refine_max_nfev": int(max_eval.value()),
                        "full_refine_ftol": float(ftol.value()),
                        "full_refine_xtol": float(xtol.value()),
                        "full_refine_gtol": float(gtol.value()),
                        "full_refine_progress_interval": int(progress_every.value()),
                    }
                    if not is_global:
                        ai_setting_updates["full_refine_target_logrmse"] = float(target.value())
                    self._save_ai_fitting_settings(**ai_setting_updates)
                    if is_global:
                        self._save_manual_parameter_search_settings(
                            global_samples=int(global_samples.value()),
                            global_starts=int(global_starts.value()),
                            target_logrmse=float(target.value()),
                        )
                    selected = []
                    for desc, check, min_box, max_box in row_widgets:
                        if not check.isChecked():
                            continue
                        lo = float(min_box.value())
                        hi = float(max_box.value())
                        if hi <= lo:
                            raise ValueError(f"{desc['label']} max must be greater than min.")
                        current_value = float(desc["value"])
                        if not lo <= current_value <= hi:
                            raise ValueError(
                                f"{desc['label']} bounds must include the current value "
                                f"({current_value:.10g})."
                            )
                        selected.append((desc, lo, hi))
                    if not selected:
                        QMessageBox.information(
                            dialog, action_name, "Select at least one parameter to optimize."
                        )
                        return
                    refine_state["latest_result"] = None
                    progress_bar.setValue(0)
                    result_label.setText(
                        "Running differential evolution..." if is_global else "Refining locally..."
                    )
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
                    result_label.setText(f"{action_name} failed: {exc}")
                    self._add_fitting_error(f"{action_name} failed: {exc}")

            run.clicked.connect(run_refine)
            dialog.finished.connect(
                lambda _result: setattr(self, "_manual_auto_refine_dialog", None)
            )
            self._manual_auto_refine_dialog = dialog
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()

        except Exception as e:
            self._add_fitting_error(f"Failed to open parameter search: {e}")
