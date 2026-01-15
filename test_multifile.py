#!/usr/bin/env python3
"""测试多文件预测功能的简单脚本"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QGridLayout
from controllers.multifile_predict_results import MultiFilePredictResultsWidget, PredictResult, PredictStatus
import datetime

def test_multifile_widget():
    """测试多文件预测结果Widget - 新的内嵌布局"""
    
    # 创建应用
    app = QApplication([])
    
    # 创建主窗口
    main_window = QMainWindow()
    main_window.setWindowTitle("Multi-file Predict Results Test - Embedded Layout")
    main_window.resize(1000, 700)
    
    # 创建中央Widget
    central_widget = QWidget()
    
    # 模拟predict2d tab的grid layout结构
    grid_layout = QGridLayout(central_widget)
    
    # 创建模拟的图像显示区域
    graphics_view = QWidget()
    graphics_view.setStyleSheet("background-color: lightblue; border: 1px solid black;")
    graphics_view.setMinimumSize(400, 300)
    
    # 创建模拟的参数控制Widget - 现在充满整个右侧
    param_widget = QWidget()
    param_widget.setStyleSheet("background-color: lightgreen; border: 1px solid black;")
    param_widget.setMinimumSize(350, 500)
    
    # 参数控件的内部布局
    param_layout = QVBoxLayout(param_widget)
    param_layout.addWidget(QPushButton("Auto Scale Controls"))
    param_layout.addWidget(QPushButton("Color Scale Controls"))
    param_layout.addWidget(QPushButton("Log Scale Controls"))
    param_layout.addWidget(QPushButton("Export Button"))
    
    # 添加多文件结果Widget到参数控件内部
    results_widget = MultiFilePredictResultsWidget()
    results_widget.setMinimumHeight(200)
    results_widget.setMaximumHeight(400)
    param_layout.addWidget(results_widget)  # 添加到export按钮下方
    
    param_layout.addStretch()  # 添加弹性空间
    
    # 添加到grid layout - 模拟新的布局
    grid_layout.addWidget(graphics_view, 0, 0, 1, 1)      # 图像显示区域
    grid_layout.addWidget(param_widget, 0, 1, 1, 1)       # 参数控制区域（充满右侧）
    
    # 添加测试按钮
    test_btn = QPushButton("Add Test Results")
    grid_layout.addWidget(test_btn, 1, 0, 1, 2)
    
    def add_test_results():
        # 添加一些测试结果
        test_files = [
            "test_file_001.cbf",
            "test_file_002.cbf", 
            "test_file_003.cbf",
            "test_file_004.cbf",
            "test_file_005.cbf"
        ]
        
        for i, filename in enumerate(test_files):
            idx = results_widget.addPredictResult(f"/test/path/{filename}")
            
            # 模拟不同的状态
            if i == 0:
                results_widget.updatePredictResult(idx, 
                    status=PredictStatus.COMPLETED,
                    confidence=0.95,
                    processing_time=1.2
                )
            elif i == 1:
                results_widget.updatePredictResult(idx,
                    status=PredictStatus.RUNNING
                )
            elif i == 2:
                results_widget.updatePredictResult(idx,
                    status=PredictStatus.FAILED,
                    error_message="Simulation failed for testing"
                )
            elif i == 3:
                results_widget.updatePredictResult(idx,
                    status=PredictStatus.COMPLETED,
                    confidence=0.88,
                    processing_time=2.1
                )
            # 第5个保持Pending状态
            
        # 更新进度
        results_widget.updateProgress(2, 5)  # 2个完成，共5个
    
    test_btn.clicked.connect(add_test_results)
    
    main_window.setCentralWidget(central_widget)
    main_window.show()
    
    return app.exec_()

if __name__ == "__main__":
    test_multifile_widget()