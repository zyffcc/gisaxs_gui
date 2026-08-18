"""Export Dialog for multi-file prediction."""

from __future__ import annotations


from typing import Dict, Any


from PyQt5.QtWidgets import QDialog, QButtonGroup


from .views import (
    ExportDialogView,
)


class ExportDialog(QDialog, ExportDialogView):
    """导出对话框"""

    def __init__(self, total_count: int, selected_count: int, current_count: int, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setModal(True)
        self.range_group = QButtonGroup(self)
        self.all_radio.setText(f"All Results ({total_count} items)")
        self.selected_radio.setText(f"Selected Results ({selected_count} items)")
        self.current_radio.setText(f"Current Display ({current_count} items)")
        self.range_group.addButton(self.all_radio, 0)
        self.range_group.addButton(self.selected_radio, 1)
        self.range_group.addButton(self.current_radio, 2)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # 禁用选中范围选项如果没有选中项
        if selected_count == 0:
            self.selected_radio.setEnabled(False)

    def getExportConfig(self) -> Dict[str, Any]:
        """获取导出配置"""
        return {
            "range": self.range_group.checkedId(),  # 0: all, 1: selected, 2: current
            "jsonl": self.jsonl_check.isChecked(),
            "jpg": self.jpg_check.isChecked(),
            "ascii": self.ascii_check.isChecked(),
        }
