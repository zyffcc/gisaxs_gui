#!/usr/bin/env python3
"""
简化版多文件预测UI测试 - 验证侧边栏布局
"""

import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class SimpleLayoutTest(QMainWindow):
    """简化的布局测试"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-File Predict - Sidebar Layout Test")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建中央widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主要布局 - 横向布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 左侧：模拟图像显示区域
        left_widget = QTabWidget()
        
        # 添加图像显示标签页
        gisaxs_tab = QWidget()
        gisaxs_tab.setStyleSheet("background-color: #f0f0f0;")
        gisaxs_layout = QVBoxLayout(gisaxs_tab)
        gisaxs_layout.addWidget(QLabel("GISAXS Image Display Area\n(原有的图像显示区域)"))
        left_widget.addTab(gisaxs_tab, "GISAXS")
        
        predict_2d_tab = QWidget()
        predict_2d_tab.setStyleSheet("background-color: #e8f4fd;")
        predict_layout = QVBoxLayout(predict_2d_tab)
        predict_layout.addWidget(QLabel("Predict-2D Results Display\n(预测结果显示区域)"))
        left_widget.addTab(predict_2d_tab, "Predict-2D")
        
        main_layout.addWidget(left_widget, stretch=7)  # 占70%空间
        
        # 右侧：创建侧边栏控制面板
        control_panel = QWidget()
        control_panel.setMaximumWidth(380)
        control_panel.setMinimumWidth(320)
        control_panel.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                          stop: 0 #f8f9fa, stop: 1 #e9ecef);
                border-left: 2px solid #6c757d;
            }
        """)
        
        # 侧边栏主布局
        sidebar_layout = QVBoxLayout(control_panel)
        sidebar_layout.setContentsMargins(10, 8, 10, 8)
        sidebar_layout.setSpacing(8)
        
        # === 1. 当前文件显示区域 ===
        current_file_frame = QFrame()
        current_file_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        current_file_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 6px;
                margin: 2px;
            }
        """)
        
        current_file_layout = QVBoxLayout(current_file_frame)
        current_file_layout.setContentsMargins(8, 6, 8, 6)
        current_file_layout.setSpacing(4)
        
        # 当前文件标题
        current_file_title = QLabel("📁 Current File")
        current_file_title.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 11px;
                color: #495057;
                border-bottom: 1px solid #dee2e6;
                padding-bottom: 3px;
                margin-bottom: 3px;
            }
        """)
        current_file_layout.addWidget(current_file_title)
        
        # 当前文件内容
        current_file_label = QLabel("experiment_sample_001.cbf\n/path/to/experiment/data/")
        current_file_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 4px;
                padding: 8px;
                font-size: 10px;
                color: #6c757d;
                font-family: 'Consolas', 'Courier New', monospace;
            }
        """)
        current_file_label.setWordWrap(True)
        current_file_label.setFixedHeight(50)
        current_file_layout.addWidget(current_file_label)
        
        sidebar_layout.addWidget(current_file_frame)
        
        # === 2. 多文件结果列表区域 ===
        results_frame = QFrame()
        results_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        results_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 6px;
                margin: 2px;
            }
        """)
        
        results_layout = QVBoxLayout(results_frame)
        results_layout.setContentsMargins(8, 6, 8, 6)
        results_layout.setSpacing(4)
        
        # 结果列表标题
        results_title = QLabel("📊 Multi-File Results")
        results_title.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 11px;
                color: #495057;
                border-bottom: 1px solid #dee2e6;
                padding-bottom: 3px;
                margin-bottom: 3px;
            }
        """)
        results_layout.addWidget(results_title)
        
        # 模拟结果列表
        results_list = QListWidget()
        results_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #dee2e6;
                border-radius: 3px;
                background-color: #ffffff;
                font-size: 10px;
                padding: 2px;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #e9ecef;
            }
            QListWidget::item:selected {
                background-color: #007bff;
                color: white;
            }
        """)
        
        # 添加示例项目
        for i in range(8):
            status_icons = ["✅", "🔄", "⏳", "❌"]
            status_names = ["Completed", "Running", "Pending", "Failed"]
            status = i % 4
            item = QListWidgetItem(f"{status_icons[status]} exp_{i+1:03d}.cbf - {status_names[status]}")
            results_list.addItem(item)
        
        results_layout.addWidget(results_list)
        
        sidebar_layout.addWidget(results_frame, stretch=1)  # 占用大部分垂直空间
        
        # === 3. 快捷操作按钮区域 ===
        actions_frame = QFrame()
        actions_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        actions_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 6px;
                margin: 2px;
            }
        """)
        
        actions_layout = QVBoxLayout(actions_frame)
        actions_layout.setContentsMargins(8, 6, 8, 6)
        actions_layout.setSpacing(6)
        
        # 操作标题
        actions_title = QLabel("🔧 Quick Actions")
        actions_title.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 11px;
                color: #495057;
                border-bottom: 1px solid #dee2e6;
                padding-bottom: 3px;
                margin-bottom: 3px;
            }
        """)
        actions_layout.addWidget(actions_title)
        
        # 按钮布局
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)
        
        # 清空按钮
        clear_button = QPushButton("🗑️ Clear")
        clear_button.setFixedHeight(28)
        clear_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #fd7e7d, stop: 1 #e74c3c);
                color: white;
                border: 1px solid #c0392b;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: bold;
                font-size: 9px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #e74c3c, stop: 1 #c0392b);
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        buttons_layout.addWidget(clear_button)
        
        # 导出按钮  
        export_all_button = QPushButton("💾 Export")
        export_all_button.setFixedHeight(28)
        export_all_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #5dade2, stop: 1 #3498db);
                color: white;
                border: 1px solid #2980b9;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: bold;
                font-size: 9px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #3498db, stop: 1 #2980b9);
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        buttons_layout.addWidget(export_all_button)
        
        actions_layout.addLayout(buttons_layout)
        
        sidebar_layout.addWidget(actions_frame)
        
        # 将侧边栏控制面板添加到主布局
        main_layout.addWidget(control_panel, stretch=3)  # 占30%的空间
        
        # 创建菜单栏
        self._create_menu()
        
        print("Simple sidebar layout test created successfully!")
    
    def _create_menu(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 测试菜单
        test_menu = menubar.addMenu('Test')
        
        # 切换侧边栏
        toggle_action = QAction('Toggle Sidebar', self)
        toggle_action.triggered.connect(self._toggle_sidebar)
        test_menu.addAction(toggle_action)
    
    def _toggle_sidebar(self):
        """切换侧边栏显示"""
        # 找到控制面板并切换显示
        for child in self.centralWidget().children():
            if isinstance(child, QWidget) and hasattr(child, 'setMaximumWidth'):
                child.setVisible(not child.isVisible())
                break

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
    
    window = SimpleLayoutTest()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()