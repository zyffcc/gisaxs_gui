#!/usr/bin/env python3
"""
多文件预测功能测试脚本 - 新布局方式
测试MultiFilePredictResultsWidget在独立容器中的UI和功能
"""

import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controllers.multifile_predict_results import *
from datetime import datetime

class MockUI:
    """模拟UI对象"""
    def __init__(self):
        self.gisaxsPredictPage = QWidget()
        self.gisaxsPredictPage.setWindowTitle("Mock GISAXS Predict Page")
        
        # 创建主布局
        main_layout = QVBoxLayout(self.gisaxsPredictPage)
        main_layout.setObjectName("verticalLayout_16")
        
        # 模拟现有控件区域
        existing_widget = QWidget()
        existing_widget.setMinimumHeight(200)
        existing_widget.setStyleSheet("background-color: #e0e0e0; border: 1px solid #ccc;")
        existing_layout = QVBoxLayout(existing_widget)
        existing_layout.addWidget(QLabel("Existing UI Controls Area"))
        existing_layout.addWidget(QPushButton("Mock Button 1"))
        existing_layout.addWidget(QPushButton("Mock Button 2"))
        
        # 模拟图像显示区域
        image_area = QWidget()
        image_area.setMinimumHeight(300)
        image_area.setStyleSheet("background-color: #f5f5f5; border: 1px solid #bbb;")
        image_layout = QVBoxLayout(image_area)
        image_layout.addWidget(QLabel("Image Display Area (gisaxsPredictImageShowWidget)"))
        
        main_layout.addWidget(existing_widget)
        main_layout.addWidget(image_area)
        
        # 预留空间给多文件结果列表
        # (在实际应用中，这里会由_setup_multifile_ui()自动添加)

def create_test_results():
    """创建测试数据"""
    test_results = []
    
    file_names = [
        "experiment_001.cbf",
        "experiment_002.cbf",
        "experiment_003.cbf",
        "experiment_004.cbf",
        "experiment_005.cbf"
    ]
    
    statuses = [
        PredictStatus.COMPLETED,
        PredictStatus.COMPLETED,
        PredictStatus.RUNNING,
        PredictStatus.PENDING,
        PredictStatus.FAILED
    ]
    
    for i, (filename, status) in enumerate(zip(file_names, statuses)):
        result = PredictResult(
            file_path=f"/path/to/{filename}",
            file_name=filename,
            status=status,
            start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            processing_time=1.23 + i * 0.5 if status == PredictStatus.COMPLETED else 0.0,
            confidence=0.85 + i * 0.03 if status == PredictStatus.COMPLETED else 0.0,
            error_message="Network timeout" if status == PredictStatus.FAILED else "",
            prediction_data={
                "prediction_data": {
                    "hr": [[1, 2], [3, 4]],
                    "curves": {"h": [0.1, 0.2], "r": [1.1, 1.2]}
                }
            } if status == PredictStatus.COMPLETED else {}
        )
        test_results.append(result)
    
    return test_results

class TestMainWindow(QMainWindow):
    """测试主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-File Predict Results Test - New Layout")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建模拟UI
        self.ui = MockUI()
        self.setCentralWidget(self.ui.gisaxsPredictPage)
        
        # 模拟多文件预测UI初始化
        self._setup_test_multifile_ui()
        
        # 添加测试数据
        self._add_test_data()
    
    def _setup_test_multifile_ui(self):
        """模拟新的多文件UI设置 - 独立容器布局"""
        predict_page = self.ui.gisaxsPredictPage
        main_layout = predict_page.layout()
        
        # 创建多文件结果显示区域
        multifile_container = QWidget()
        multifile_container.setObjectName("multifilePredictContainer")
        multifile_container.setMaximumHeight(300)
        multifile_container.setStyleSheet("border: 2px solid #007acc; background-color: #ffffff;")
        
        container_layout = QVBoxLayout(multifile_container)
        container_layout.setContentsMargins(5, 5, 5, 5)
        
        # 添加标题
        title_label = QLabel("Multi-File Prediction Results")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-weight: bold;
                color: #333333;
                padding: 5px 0px;
            }
        """)
        container_layout.addWidget(title_label)
        
        # 创建多文件结果列表组件
        self.multifile_results_widget = MultiFilePredictResultsWidget(parent=multifile_container)
        container_layout.addWidget(self.multifile_results_widget)
        
        # 将容器添加到主布局
        main_layout.addWidget(multifile_container)
        
        # 连接信号
        self.multifile_results_widget.result_selected.connect(self._on_result_selected)
        self.multifile_results_widget.export_requested.connect(self._on_export_requested)
        
        print("Multi-file UI setup completed - container added to main layout")
    
    def _add_test_data(self):
        """添加测试数据"""
        test_results = create_test_results()
        
        for result in test_results:
            self.multifile_results_widget.add_result(result)
    
    def _on_result_selected(self, result: PredictResult):
        """结果选中处理"""
        print(f"Selected: {result.file_name} (Status: {result.status.value})")
        
    def _on_export_requested(self, results: List[PredictResult]):
        """导出请求处理"""
        print(f"Export requested for {len(results)} results")

def main():
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyleSheet("""
        QWidget {
            font-family: "Segoe UI", Arial, sans-serif;
        }
        QMainWindow {
            background-color: #f8f9fa;
        }
    """)
    
    window = TestMainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()