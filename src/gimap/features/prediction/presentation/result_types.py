"""Result Types for multi-file prediction."""

from __future__ import annotations

from dataclasses import dataclass


import datetime


from enum import Enum


from typing import Dict, Optional, Any


from PyQt5.QtGui import QColor


class PredictStatus(Enum):
    """预测状态枚举"""

    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


@dataclass
class PredictResult:
    """单个预测结果数据类"""

    file_path: str
    file_name: str
    status: PredictStatus
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    processing_time: float = 0.0
    error_message: str = ""
    prediction_data: Optional[Dict[str, Any]] = None
    stack_count: int = 1

    @property
    def duration_str(self) -> str:
        """格式化的处理时间"""
        if self.processing_time > 0:
            return f"{self.processing_time:.2f}s"
        return "-"

    @property
    def status_color(self) -> QColor:
        """状态对应的颜色"""
        color_map = {
            PredictStatus.PENDING: QColor(128, 128, 128),  # 灰色
            PredictStatus.RUNNING: QColor(0, 123, 255),  # 蓝色
            PredictStatus.COMPLETED: QColor(40, 167, 69),  # 绿色
            PredictStatus.FAILED: QColor(220, 53, 69),  # 红色
            PredictStatus.CANCELLED: QColor(255, 193, 7),  # 黄色
        }
        return color_map.get(self.status, QColor(0, 0, 0))
