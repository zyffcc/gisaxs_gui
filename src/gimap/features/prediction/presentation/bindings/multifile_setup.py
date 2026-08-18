"""Multifile Setup coordination for Prediction."""

from __future__ import annotations

import os


from PyQt5.QtWidgets import (
    QMessageBox,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QFrame,
    QDialog,
)

from src.gimap.app.presentation.responsive_layout import (
    install_adaptive_window_profile,
    move_window_to_cursor_screen,
)


from src.gimap.features.prediction.presentation.multifile_results import (
    MultiFilePredictResultsWidget,
    MultiFilePredictManager,
)


class MultifileSetupMixin:
    """Own multifile setup presentation behavior."""

    def _setup_multifile_ui(self) -> None:
        """初始化多文件预测的外置窗口，不改动主窗口布局"""
        try:
            if getattr(self, "_multifile_window", None) is not None:
                return

            # 创建一个无模式的外置对话框，独立显示多文件结果
            win = QDialog(self.main_window)
            win.setWindowTitle("Multi-File Results")
            win.setModal(False)
            win.setMinimumSize(700, 600)
            win.resize(820, 680)
            try:
                win.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            except Exception:
                pass

            outer = QVBoxLayout(win)
            outer.setContentsMargins(10, 8, 10, 8)
            outer.setSpacing(8)
            install_adaptive_window_profile(
                win,
                lambda profile, screen, window=win: self._apply_floating_screen_profile(
                    window, profile
                ),
                apply_window_minimum=False,
            )

            # === 1. 当前文件显示区域 ===
            current_file_frame = QFrame(win)
            current_file_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
            current_file_frame.setStyleSheet(
                """
                    QFrame {
                        background-color: #ffffff;
                        border: 1px solid #ced4da;
                        border-radius: 6px;
                        margin: 2px;
                    }
                    """
            )
            current_file_layout = QVBoxLayout(current_file_frame)
            current_file_layout.setContentsMargins(8, 6, 8, 6)
            current_file_layout.setSpacing(4)

            current_file_title = QLabel("Current File", current_file_frame)
            current_file_title.setStyleSheet(
                """
                    QLabel {
                        font-weight: bold;
                        font-size: 11px;
                        color: #495057;
                        border-bottom: 1px solid #dee2e6;
                        padding-bottom: 3px;
                        margin-bottom: 3px;
                    }
                    """
            )
            current_file_layout.addWidget(current_file_title)

            self._current_file_label = QLabel("No file selected", current_file_frame)
            self._current_file_label.setStyleSheet(
                """
                    QLabel {
                        background-color: #f8f9fa;
                        border: 1px solid #e9ecef;
                        border-radius: 4px;
                        padding: 8px;
                        font-size: 10px;
                        color: #6c757d;
                        font-family: 'Consolas', 'Courier New', monospace;
                    }
                    """
            )
            self._current_file_label.setWordWrap(True)
            self._current_file_label.setMinimumHeight(50)
            current_file_layout.addWidget(self._current_file_label)
            outer.addWidget(current_file_frame)

            # === 2. 多文件结果列表区域 ===
            results_frame = QFrame(win)
            results_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
            results_frame.setStyleSheet(
                """
                    QFrame {
                        background-color: #ffffff;
                        border: 1px solid #ced4da;
                        border-radius: 6px;
                        margin: 2px;
                    }
                    """
            )
            results_layout = QVBoxLayout(results_frame)
            results_layout.setContentsMargins(8, 6, 8, 6)
            results_layout.setSpacing(4)

            results_title = QLabel("Multi-File Results", results_frame)
            results_title.setStyleSheet(
                """
                    QLabel {
                        font-weight: bold;
                        font-size: 11px;
                        color: #495057;
                        border-bottom: 1px solid #dee2e6;
                        padding-bottom: 3px;
                        margin-bottom: 3px;
                    }
                    """
            )
            results_layout.addWidget(results_title)

            self._multifile_results_widget = MultiFilePredictResultsWidget(parent=results_frame)
            self._multifile_results_widget.setStyleSheet(
                """
                    MultiFilePredictResultsWidget {
                        border: none;
                        background-color: transparent;
                    }
                    """
            )
            results_layout.addWidget(self._multifile_results_widget)
            outer.addWidget(results_frame, stretch=1)

            # === 3. 快捷操作按钮区域 ===
            actions_frame = QFrame(win)
            actions_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
            actions_frame.setStyleSheet(
                """
                    QFrame {
                        background-color: #ffffff;
                        border: 1px solid #ced4da;
                        border-radius: 6px;
                        margin: 2px;
                    }
                    """
            )
            actions_layout = QVBoxLayout(actions_frame)
            actions_layout.setContentsMargins(8, 6, 8, 6)
            actions_layout.setSpacing(6)

            actions_title = QLabel("Quick Actions", actions_frame)
            actions_title.setStyleSheet(
                """
                    QLabel {
                        font-weight: bold;
                        font-size: 11px;
                        color: #495057;
                        border-bottom: 1px solid #dee2e6;
                        padding-bottom: 3px;
                        margin-bottom: 3px;
                    }
                    """
            )
            actions_layout.addWidget(actions_title)

            buttons_layout = QHBoxLayout()
            buttons_layout.setSpacing(8)
            clear_button = QPushButton("Clear All", actions_frame)
            clear_button.setMinimumHeight(28)
            clear_button.clicked.connect(self._clear_multifile_results)
            export_all_button = QPushButton("Export All", actions_frame)
            export_all_button.setMinimumHeight(28)
            export_all_button.clicked.connect(self._export_all_results)
            buttons_layout.addWidget(clear_button)
            buttons_layout.addWidget(export_all_button)
            actions_layout.addLayout(buttons_layout)
            outer.addWidget(actions_frame)

            # 连接信号
            self._multifile_results_widget.result_selected.connect(
                self._on_multifile_result_selected
            )
            self._multifile_results_widget.result_double_clicked.connect(
                self._on_multifile_result_selected
            )
            self._multifile_results_widget.export_requested.connect(
                self._on_multifile_export_requested
            )

            # 创建多文件管理器
            if self._multifile_manager is None:
                self._multifile_manager = MultiFilePredictManager(self)
                self._multifile_manager.prediction_started.connect(
                    self._on_multifile_prediction_started
                )
                self._multifile_manager.prediction_completed.connect(
                    self._on_multifile_prediction_completed
                )
                self._multifile_manager.result_updated.connect(self._on_multifile_result_updated)
                self._multifile_manager.progress_updated.connect(
                    self._on_multifile_progress_updated
                )

            # 初始不显示，仅在切换到 multi_files 模式时显示
            self._multifile_window = win
            self._append_status_message("Multi-file external window initialized", level="INFO")

        except Exception as e:
            self._append_status_message(f"Failed to setup multi-file UI: {e}", level="ERROR")

    def _show_multifile_results_window(self) -> None:
        if getattr(self, "_multifile_window", None) is None:
            self._setup_multifile_ui()
        win = getattr(self, "_multifile_window", None)
        if win is None:
            QMessageBox.information(
                self.main_window,
                "Multi-File Results",
                "The multi-file results window is not available yet.",
            )
            return
        if self._multifile_results_widget is not None:
            self._multifile_results_widget.setVisible(True)
        if not win.isVisible():
            move_window_to_cursor_screen(win)
        win.show()
        try:
            win.raise_()
            win.activateWindow()
        except Exception:
            pass

    def _clear_multifile_results(self) -> None:
        """清空所有多文件结果"""
        if self._multifile_results_widget:
            self._multifile_results_widget.clear_all_results()

    def _export_all_results(self) -> None:
        """导出所有结果"""
        if self._multifile_results_widget:
            all_results = self._multifile_results_widget.get_all_results()
            if all_results:
                self._multifile_results_widget.onExportClicked()
            else:
                QMessageBox.information(self.main_window, "Export", "No results to export.")

    def _stop_gisaxs_predict(self) -> None:
        if not self._multifile_prediction_active:
            self._append_status_message("No active multi-file prediction to stop.", level="INFO")
            return
        if self._multifile_manager:
            self._multifile_manager.cancel_prediction()
            self._append_status_message(
                "Stopping multi-file prediction after the current file...", level="WARN"
            )
        stop_btn = getattr(self.ui, "gisaxsPredictStopButton", None)
        if stop_btn:
            stop_btn.setEnabled(False)

    def _adjust_predict_layout_for_mode(self, mode: str) -> None:
        """根据模式调整预测布局"""
        # 显示/隐藏外置的多文件窗口
        try:
            win = getattr(self, "_multifile_window", None)
            if win is not None and mode == "multi_files" and win.isVisible():
                win.raise_()
        except Exception:
            pass

        # 更新当前文件标签的可见性
        if hasattr(self, "_current_file_label"):
            if mode == "multi_files":
                self._current_file_label.setVisible(True)
                if (
                    not self._current_file_label.text()
                    or self._current_file_label.text() == "Current: No file selected"
                ):
                    self._current_file_label.setText("No file selected")
            else:
                self._current_file_label.setVisible(False)

    def _update_current_file_display(self, file_path: str, stack_count: int = 1) -> None:
        """更新当前文件显示"""
        if hasattr(self, "_current_file_label"):
            if file_path:
                file_name = os.path.basename(file_path)
                suffix = (
                    f" ({stack_count} files stacked)"
                    if stack_count and stack_count > 1
                    else " (1 file)"
                )
                self._current_file_label.setText(f"{file_name}{suffix}")
                self._current_file_label.setToolTip(file_path)
            else:
                self._current_file_label.setText("No file selected")
                self._current_file_label.setToolTip("")

    def _connect_line_edit(self, name: str, slot) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None:
            return
        widget.returnPressed.connect(slot)

    def _connect_double_spin(self, name: str, slot) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None:
            return
        widget.editingFinished.connect(slot)
