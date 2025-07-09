# main.py

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow
from ui.main_window import Ui_MainWindow  # 导入转换后的 UI 类
from ui.menu_manager import MenuManager
from controllers import MainController
from core.window_manager import window_manager

# 导入参数访问系统
from core.global_params import global_params
from utils.parameter_access import (
    get_all_software_params, 
    get_physics_params_for_calculation,
    get_param_by_path,
    validate_params_for_physics
)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("GISAXS Toolkit")
        
        # 初始化全局参数系统
        self.initialize_parameter_system()
        
        # 初始化菜单管理器
        self.menu_manager = MenuManager(self)
        
        # 初始化主控制器
        self.main_controller = MainController(self, self)
        
        # 连接菜单信号
        self.connect_menu_signals()
        
        # 设置窗口属性
        self.setup_window()
    
    def connect_menu_signals(self):
        """连接菜单信号"""
        # 设置菜单系统
        self.menu_manager.setup_menus()
        
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
    
    def closeEvent(self, event):
        """窗口关闭事件 - 确保参数保存"""
        try:
            print("正在保存参数...")
            global_params.force_save_parameters()
            print("✓ 程序关闭前参数已保存")
            event.accept()
        except Exception as e:
            print(f"关闭时保存参数失败: {e}")
            event.accept()
    
    def initialize_parameter_system(self):
        """初始化全局参数系统"""
        print("=== 初始化GISAXS参数系统 ===")
        
        # 全局参数管理器已经通过导入自动创建
        # 参数管理器在初始化时会自动加载用户参数（如果存在）
        # 这里只需要确认系统已经正确初始化
        try:
            # 检查参数系统是否正常工作
            beam_params = global_params.get_module_parameters('beam')
            detector_params = global_params.get_module_parameters('detector')
            
            if beam_params and detector_params:
                print(f"✓ 参数系统已正确初始化")
                print(f"✓ 当前波长: {global_params.get_parameter('beam', 'wavelength')} nm")
                print(f"✓ 探测器距离: {global_params.get_parameter('detector', 'distance')} mm")
            else:
                print("⚠ 参数系统初始化不完整，使用默认参数")
                
        except Exception as e:
            print(f"参数系统初始化警告: {e}")
            print("使用内置默认参数")
    
    def get_software_parameters(self):
        """提供给外部调用的参数获取方法"""
        return get_all_software_params()
    
    def get_physics_parameters(self):
        """提供给外部调用的物理参数获取方法"""
        return get_physics_params_for_calculation()


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
