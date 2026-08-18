"""Message Log coordination for the fitting workspace."""

from __future__ import annotations


from pathlib import Path


from PyQt5.QtCore import Qt, QPoint

from PyQt5.QtWidgets import (
    QFileDialog,
    QVBoxLayout,
    QMenu,
    QAction,
    QTextBrowser,
    QSizePolicy,
    QDialog,
    QInputDialog,
)


class MessageLogMixin:
    """Own message log presentation behavior."""

    def _setup_fitting_text_browser(self):
        """No description."""
        if hasattr(self.ui, "FittingTextBrowser"):
            self.status_updated.connect(self._update_fitting_text_browser)

            self.ui.FittingTextBrowser.clear()
            self._add_fitting_message("Fitting Controller initialized", "INFO")
            self._init_fitting_textbrowser_enhancements()

    def _init_fitting_textbrowser_enhancements(self):
        """Initialize FittingTextBrowser enhancements: fixed height, context menu, detachable window."""
        tb = getattr(self.ui, "FittingTextBrowser", None)
        if tb is None:
            return
        # Keep a useful minimum while allowing the resizable Run Log card to
        # pass its additional content height through to the text browser.
        tb.setMinimumHeight(100)
        tb.setMaximumHeight(16777215)
        tb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if self._fitting_browser_original_height is None:
            self._fitting_browser_original_height = tb.height()
        tb.setContextMenuPolicy(Qt.CustomContextMenu)
        tb.customContextMenuRequested.connect(self._show_fitting_browser_menu)

    def _show_fitting_browser_menu(self, pos: QPoint):
        tb = getattr(self.ui, "FittingTextBrowser", None)
        if tb is None:
            return
        menu = QMenu(tb)
        act_copy_all = QAction("Copy All", menu)
        act_clear = QAction("Clear", menu)
        act_save = QAction("Save Log...", menu)
        act_detach = QAction("Open Detached Window", menu)
        act_set_max_lines = QAction(
            f"Set Max Lines (Current: {self._fitting_messages_max_lines})", menu
        )
        menu.addAction(act_copy_all)
        menu.addAction(act_clear)
        menu.addSeparator()
        menu.addAction(act_save)
        menu.addSeparator()
        menu.addAction(act_detach)
        menu.addSeparator()
        menu.addAction(act_set_max_lines)
        chosen = menu.exec_(tb.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == act_copy_all:
            tb.selectAll()
            tb.copy()
        elif chosen == act_clear:
            self.clear_fitting_messages()
        elif chosen == act_save:
            path, _ = QFileDialog.getSaveFileName(
                tb, "Save Fitting Log", "fitting_log.txt", "Text Files (*.txt)"
            )
            if path:
                self.save_fitting_log(path)
        elif chosen == act_detach:
            self._open_detached_fitting_browser()
        elif chosen == act_set_max_lines:
            val, ok = QInputDialog.getInt(
                tb,
                "Max Lines",
                "Set maximum display lines:",
                self._fitting_messages_max_lines,
                50,
                5000,
                50,
            )
            if ok:
                self._fitting_messages_max_lines = val
                self._trim_fitting_messages_if_needed()

    def _open_detached_fitting_browser(self):
        tb = getattr(self.ui, "FittingTextBrowser", None)
        if tb is None:
            return
        if self._detached_fitting_dialog is not None:
            try:
                self._detached_fitting_dialog.raise_()
                self._detached_fitting_dialog.activateWindow()
                return
            except Exception:
                self._detached_fitting_dialog = None
        dlg = QDialog(tb)
        dlg.setWindowTitle("Fitting Log (Detached)")
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser(dlg)
        browser.setHtml(tb.toHtml())
        layout.addWidget(browser)
        dlg.resize(640, 420)
        self._detached_fitting_dialog = dlg

        # 函数说明：实现 sync 相关逻辑。
        def sync(msg_html):
            try:
                browser.append(msg_html)
            except Exception:
                pass

        self._detached_append = sync
        # Allow reopen after close
        dlg.finished.connect(self._on_detached_closed)
        dlg.show()

    def _on_detached_closed(self, *_):
        self._detached_fitting_dialog = None
        self._detached_append = None

    def _trim_fitting_messages_if_needed(self):
        tb = getattr(self.ui, "FittingTextBrowser", None)
        if tb is None:
            return
        doc = tb.document()
        blocks = doc.blockCount()
        if blocks <= self._fitting_messages_max_lines:
            return
        remove_count = blocks - self._fitting_messages_max_lines
        cursor = tb.textCursor()
        cursor.movePosition(cursor.Start)
        for _ in range(remove_count):
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
        # Trimming notice in English (sync detached if exists)
        notice = f'<span style="color:#888;">(Log trimmed, keeping last {self._fitting_messages_max_lines} lines)</span>'
        tb.append(notice)
        if self._detached_append:
            self._detached_append(notice)

    def _update_fitting_text_browser(self, message: str):
        """No description."""
        if hasattr(self.ui, "FittingTextBrowser"):
            self._add_fitting_message(message, "STATUS")

    def _add_fitting_message(self, message: str, msg_type: str = "INFO"):
        """ittingTextBrowser"""
        if not hasattr(self.ui, "FittingTextBrowser"):
            return

        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")

        color_map = {
            "INFO": "#333333",
            "STATUS": "#2563eb",
            "WARNING": "#d97706",
            "ERROR": "#dc2626",
            "SUCCESS": "#16a34a",
            "PARTICLE": "#7c3aed",
        }

        color = color_map.get(msg_type, "#333333")

        formatted_message = (
            f'<span style="color: {color};">[{timestamp}] {msg_type}: {message}</span>'
        )

        self.ui.FittingTextBrowser.append(formatted_message)
        if self._detached_append:
            try:
                self._detached_append(formatted_message)
            except Exception:
                pass
        self._trim_fitting_messages_if_needed()

        cursor = self.ui.FittingTextBrowser.textCursor()
        cursor.movePosition(cursor.End)
        self.ui.FittingTextBrowser.setTextCursor(cursor)

    def _add_fitting_warning(self, message: str):
        """No description."""
        self._add_fitting_message(message, "WARNING")

    def _add_fitting_error(self, message: str):
        """No description."""
        self._add_fitting_message(message, "ERROR")

    def _add_fitting_success(self, message: str):
        """No description."""
        self._add_fitting_message(message, "SUCCESS")

    def _add_particle_message(self, message: str):
        """No description."""
        self._add_fitting_message(message, "PARTICLE")

    def clear_fitting_messages(self):
        """Clear fitting messages in both embedded and detached browser."""
        if hasattr(self.ui, "FittingTextBrowser"):
            self.ui.FittingTextBrowser.clear()
            self._add_fitting_message("Messages cleared", "INFO")
            # Sync detached window
            if self._detached_fitting_dialog is not None:
                for child in self._detached_fitting_dialog.children():
                    if isinstance(child, QTextBrowser):
                        child.clear()
                        child.append('<span style="color:#2E86AB;">[INFO] Messages cleared</span>')

    def get_fitting_messages(self) -> str:
        """No description."""
        if hasattr(self.ui, "FittingTextBrowser"):
            return self.ui.FittingTextBrowser.toPlainText()
        return ""

    def save_fitting_log(self, filepath: str) -> bool:
        """No description."""
        try:
            content = self.get_fitting_messages()
            if content:
                self.fitting_view_model.storage.save_fitting_log(Path(filepath), content)
                self._add_fitting_success(f"Fitting log saved to: {filepath}")
                return True
            else:
                self._add_fitting_warning("No messages to save")
                return False
        except Exception as e:
            self._add_fitting_error(f"Failed to save fitting log: {str(e)}")
            return False
