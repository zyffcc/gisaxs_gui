# main.py

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from ui.main_window import Ui_MainWindow  # 导入转换后的 UI 类
from controllers import MainController
from core.window_manager import window_manager


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("GISAXS Toolkit")
        
        # 初始化主控制器
        self.main_controller = MainController(self, self)
        
        # 连接菜单信号（如果UI中有Display菜单的话）
        self.connect_menu_signals()
        
        # 设置窗口属性
        self.setup_window()
    
    def connect_menu_signals(self):
        """连接菜单信号"""
        # 连接Display菜单项的信号（如果存在的话）
        if hasattr(self, 'actionDisplay'):
            self.actionDisplay.triggered.connect(self.show_display_settings)
    
    def show_display_settings(self):
        """显示显示设置对话框"""
        try:
            from ui.settings_dialog import SettingsDialog
            dialog = SettingsDialog(self)
            dialog.exec_()
        except ImportError:
            # 如果settings_dialog不存在，显示提示
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "提示", "显示设s置功能正在开发中...")
    
    def setup_window(self):
        """设置窗口属性"""
        # 使用窗口管理器设置自适应窗口（使用配置中的默认值）
        scale = window_manager.setup_adaptive_window(self)
        
        # 应用自适应字体
        window_manager.apply_adaptive_font(self)
        
        # 存储缩放比例供其他组件使用
        self.scale_factor = scale
        
        # 设置StackedWidget的自适应行为
        self._setup_stacked_widget()
    
    def _setup_stacked_widget(self):
        """设置StackedWidget的自适应行为"""
        try:
            from utils.layout_utils import LayoutUtils
            if hasattr(self, 'mainWindowWidget'):
                LayoutUtils.setup_adaptive_stacked_widget(self.mainWindowWidget)
        except ImportError as e:
            print(f"布局工具导入失败: {e}")
        except Exception as e:
            print(f"StackedWidget设置失败: {e}")


def main():
    """主函数"""
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
