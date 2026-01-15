#!/usr/bin/env python3
"""
多文件预测功能测试脚本 - 新的侧边栏布局
测试重新设计后的多文件预测UI
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
        # 创建模拟的主要UI组件
        self.gisaxsPredictImageShowWidget = QWidget()
        self.gisaxsPredictImageShowWidget.setWindowTitle("Mock Image Show Widget")
        
        # 创建原有的TabWidget（模拟现有的图像显示组件）
        self.gisaxsPredictImageShowTabWidget = QTabWidget()
        
        # 添加模拟的tab
        gisaxs_tab = QWidget()
        gisaxs_tab.setStyleSheet("background-color: #f0f0f0;")
        gisaxs_layout = QVBoxLayout(gisaxs_tab)
        gisaxs_layout.addWidget(QLabel("GISAXS Image Display Area"))
        gisaxs_layout.addWidget(QLabel("(原有的图像显示区域)"))
        self.gisaxsPredictImageShowTabWidget.addTab(gisaxs_tab, "GISAXS")
        
        predict_2d_tab = QWidget()
        predict_2d_tab.setStyleSheet("background-color: #e8f4fd;")
        predict_layout = QVBoxLayout(predict_2d_tab)
        predict_layout.addWidget(QLabel("Predict-2D Results Display"))
        predict_layout.addWidget(QLabel("(预测结果显示区域)"))
        self.gisaxsPredictImageShowTabWidget.addTab(predict_2d_tab, "Predict-2D")
        
        # 设置初始布局（将在_setup_multifile_ui中被重新排列）
        initial_layout = QVBoxLayout(self.gisaxsPredictImageShowWidget)
        initial_layout.addWidget(self.gisaxsPredictImageShowTabWidget)

def create_test_results():
    """创建测试数据"""
    test_results = []
    
    file_names = [
        "experiment_001.cbf",
        "experiment_002.cbf", 
        "experiment_003.cbf",
        "experiment_004.cbf",
        "experiment_005.cbf",
        "sample_a_001.tiff",
        "sample_a_002.tiff",
        "sample_b_001.tiff",
    ]
    
    statuses = [
        PredictStatus.COMPLETED,
        PredictStatus.COMPLETED,
        PredictStatus.RUNNING,
        PredictStatus.PENDING,
        PredictStatus.FAILED,
        PredictStatus.COMPLETED,
        PredictStatus.CANCELLED,
        PredictStatus.PENDING,
    ]
    
    for i, (filename, status) in enumerate(zip(file_names, statuses)):
        result = PredictResult(
            file_path=f"/path/to/experiment/data/{filename}",
            file_name=filename,
            status=status,
            start_time=datetime.now(),
            processing_time=1.23 + i * 0.5 if status == PredictStatus.COMPLETED else 0.0,
            confidence=0.85 + i * 0.02 if status == PredictStatus.COMPLETED else 0.0,
            error_message="Memory allocation failed" if status == PredictStatus.FAILED else "",
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
        self.setWindowTitle("Multi-File Predict - Redesigned Layout Test")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建模拟UI
        self.ui = MockUI()
        self.setCentralWidget(self.ui.gisaxsPredictImageShowWidget)
        
        # 模拟多文件预测UI初始化（重新设计版本）
        self._setup_test_multifile_ui()
        
        # 添加测试数据
        self._add_test_data()
        
        # 创建菜单栏用于测试
        self._create_test_menu()
    
    def _setup_test_multifile_ui(self):
        """模拟新的多文件UI设置 - 侧边栏布局"""
        # 获取image_show_widget
        image_show_widget = self.ui.gisaxsPredictImageShowWidget
        
        # 清空原有布局并重新创建
        old_layout = image_show_widget.layout()
        if old_layout:
            # 保存原有的TabWidget
            tab_widget = self.ui.gisaxsPredictImageShowTabWidget
            if tab_widget:
                old_layout.removeWidget(tab_widget)
        
        # 创建新的整体布局 - 使用水平分割器
        main_layout = QHBoxLayout(image_show_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        
        # 左侧：图像显示区域（原有TabWidget）
        if tab_widget:
            tab_widget.setParent(image_show_widget)
            main_layout.addWidget(tab_widget, stretch=3)  # 占60%的空间
        
        # 右侧：多文件预测控制面板
        control_panel = QWidget()
        control_panel.setMaximumWidth(400)
        control_panel.setMinimumWidth(350)
        control_panel.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-left: 2px solid #dee2e6;
            }
        """)
        
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 5, 10, 5)
        control_layout.setSpacing(8)
        
        # 1. 当前文件显示
        current_file_group = QWidget()
        current_file_layout = QVBoxLayout(current_file_group)
        current_file_layout.setContentsMargins(5, 5, 5, 5)
        
        current_file_title = QLabel("Current File")
        current_file_title.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 12px;
                color: #2c3e50;
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 3px;
            }
        """)
        current_file_layout.addWidget(current_file_title)
        
        self.current_file_label = QLabel("No file selected")
        self.current_file_label.setStyleSheet("""
            QLabel {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 8px;
                font-size: 11px;
                color: #34495e;
                word-wrap: true;
            }
        """)
        self.current_file_label.setWordWrap(True)
        self.current_file_label.setMinimumHeight(40)
        current_file_layout.addWidget(self.current_file_label)
        
        control_layout.addWidget(current_file_group)
        
        # 2. 多文件结果列表
        results_group = QWidget()
        results_layout = QVBoxLayout(results_group)
        results_layout.setContentsMargins(5, 5, 5, 5)
        
        results_title = QLabel("Multi-File Results")
        results_title.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 12px;
                color: #2c3e50;
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 3px;
            }
        """)
        results_layout.addWidget(results_title)
        
        # 创建多文件结果列表组件
        self.multifile_results_widget = MultiFilePredictResultsWidget(parent=results_group)
        results_layout.addWidget(self.multifile_results_widget)
        
        control_layout.addWidget(results_group, stretch=1)  # 占大部分空间
        
        # 3. 快捷操作按钮
        actions_group = QWidget()
        actions_layout = QHBoxLayout(actions_group)
        actions_layout.setContentsMargins(5, 5, 5, 5)
        
        clear_button = QPushButton("Clear All")
        clear_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        clear_button.clicked.connect(self.clear_all_results)
        
        export_all_button = QPushButton("Export All")
        export_all_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        export_all_button.clicked.connect(self.export_all_results)
        
        actions_layout.addWidget(clear_button)
        actions_layout.addWidget(export_all_button)
        actions_layout.addStretch()
        
        control_layout.addWidget(actions_group)
        
        # 添加控制面板到主布局
        main_layout.addWidget(control_panel, stretch=2)  # 占40%的空间
        
        # 连接信号
        self.multifile_results_widget.result_selected.connect(self._on_result_selected)
        self.multifile_results_widget.export_requested.connect(self._on_export_requested)
        
        print("Multi-file UI setup completed - redesigned sidebar layout")
    
    def _add_test_data(self):
        """添加测试数据"""
        test_results = create_test_results()
        
        for result in test_results:
            self.multifile_results_widget.addResult(result)
    
    def _create_test_menu(self):
        """创建测试菜单"""
        menubar = self.menuBar()
        
        # 测试菜单
        test_menu = menubar.addMenu('Test')
        
        # 添加随机结果
        add_action = QAction('Add Random Result', self)
        add_action.triggered.connect(self._add_random_result)
        test_menu.addAction(add_action)
        
        # 切换显示/隐藏控制面板
        toggle_action = QAction('Toggle Control Panel', self)
        toggle_action.triggered.connect(self._toggle_control_panel)
        test_menu.addAction(toggle_action)
    
    def _add_random_result(self):
        """添加随机结果"""
        import random
        
        filename = f"random_file_{random.randint(1000, 9999)}.cbf"
        status = random.choice(list(PredictStatus))
        
        result = PredictResult(
            file_path=f"/random/path/{filename}",
            file_name=filename,
            status=status,
            start_time=datetime.now(),
            processing_time=random.uniform(0.5, 3.0) if status == PredictStatus.COMPLETED else 0.0,
            confidence=random.uniform(0.7, 0.95) if status == PredictStatus.COMPLETED else 0.0,
            error_message="Random error" if status == PredictStatus.FAILED else "",
        )
        
        self.multifile_results_widget.addResult(result)
    
    def _toggle_control_panel(self):
        """切换控制面板显示/隐藏"""
        # 找到控制面板
        control_panel = self.findChild(QWidget)
        if control_panel and control_panel.objectName() != "gisaxsPredictImageShowTabWidget":
            control_panel.setVisible(not control_panel.isVisible())
    
    def clear_all_results(self):
        """清空所有结果"""
        self.multifile_results_widget.clear_all_results()
        self.current_file_label.setText("No file selected")
        
    def export_all_results(self):
        """导出所有结果"""
        all_results = self.multifile_results_widget.get_all_results()
        print(f"Export requested for {len(all_results)} results")
        QMessageBox.information(self, "Export", f"Would export {len(all_results)} results")
    
    def _on_result_selected(self, result: PredictResult):
        """结果选中处理"""
        print(f"Selected: {result.file_name} (Status: {result.status.value})")
        self.current_file_label.setText(f"Viewing: {result.file_name}")
        
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
            background-color: #ffffff;
        }
    """)
    
    window = TestMainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()