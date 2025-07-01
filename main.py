# main.py

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStatusBar, QProgressBar
from PyQt5.QtCore import Qt
from ui.main_window import Ui_MainWindow  # 导入转换后的 UI 类
from controllers import MainController


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("GISAXS Toolkit")
        
        # 设置状态栏
        self.setup_statusbar()
        
        # 初始化主控制器
        self.main_controller = MainController(self, self)
        
        # 连接控制器信号
        self.connect_controller_signals()
        
        # 设置窗口属性
        self.setup_window()
    
    def setup_statusbar(self):
        """设置状态栏"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.statusbar.addPermanentWidget(self.progress_bar)
        
        # 显示就绪状态
        self.statusbar.showMessage("就绪")
    
    def connect_controller_signals(self):
        """连接控制器信号"""
        # 状态更新
        self.main_controller.status_updated.connect(self.update_status)
        
        # 进度更新
        self.main_controller.progress_updated.connect(self.update_progress)
    
    def setup_window(self):
        """设置窗口属性"""
        self.setMinimumSize(1000, 800)
        self.resize(1200, 900)
        
        # 居中显示
        self.center_window()
    
    def center_window(self):
        """窗口居中显示"""
        screen = QApplication.desktop().screenGeometry()
        window = self.geometry()
        x = (screen.width() - window.width()) // 2
        y = (screen.height() - window.height()) // 2
        self.move(x, y)
    
    def update_status(self, message):
        """更新状态栏消息"""
        self.statusbar.showMessage(message)
    
    def update_progress(self, value):
        """更新进度条"""
        if value <= 0:
            self.progress_bar.setVisible(False)
        else:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(value)
            if value >= 100:
                # 进度完成后延迟隐藏
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
    
    def closeEvent(self, event):
        """关闭事件处理"""
        # 检查是否有正在进行的任务
        if hasattr(self.main_controller, 'trainset_controller'):
            if self.main_controller.trainset_controller.is_generating:
                from PyQt5.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    self,
                    "确认退出",
                    "训练集生成正在进行中，确定要退出吗？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    # 停止生成并等待
                    self.main_controller.trainset_controller._stop_generation()
                    event.accept()
                else:
                    event.ignore()
                    return
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("GISAXS Toolkit")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("GISAXS Lab")
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()