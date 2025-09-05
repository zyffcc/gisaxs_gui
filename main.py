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
        import time
        self._startup_time = time.time()
        
        self.setupUi(self)
        self.setWindowTitle("GISAXS Toolkit")
        
        # 设置初始状态栏消息
        if hasattr(self, 'statusbar'):
            self.statusbar.showMessage("界面已就绪，正在初始化组件...")
        
        # 快速初始化：仅设置基本UI
        self.setup_window()
        
        ui_ready_time = time.time() - self._startup_time
        print(f"✓ UI界面就绪用时: {ui_ready_time:.2f}秒")
        
        # 延迟初始化标志
        self._initialization_completed = False
        
        # 延迟初始化其他组件
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self._delayed_initialization)
    
    def _delayed_initialization(self):
        """延迟初始化非关键组件"""
        import time
        try:
            # 更新状态栏
            if hasattr(self, 'statusbar'):
                self.statusbar.showMessage("正在初始化参数系统...")
            
            # 初始化全局参数系统
            self.initialize_parameter_system()
            
            # 更新状态栏
            if hasattr(self, 'statusbar'):
                self.statusbar.showMessage("正在初始化菜单...")
            
            # 初始化菜单管理器
            self.menu_manager = MenuManager(self)
            
            # 更新状态栏
            if hasattr(self, 'statusbar'):
                self.statusbar.showMessage("正在初始化控制器...")
            
            # 初始化主控制器
            self.main_controller = MainController(self, self)
            
            # 连接菜单信号
            self.connect_menu_signals()
            
            # 标记初始化完成
            self._initialization_completed = True
            
            # 计算总启动时间
            total_time = time.time() - self._startup_time
            print(f"✓ 总启动用时: {total_time:.2f}秒")
            
            # 更新状态栏
            if hasattr(self, 'statusbar'):
                self.statusbar.showMessage(f"GISAXS Toolkit 就绪 (启动用时: {total_time:.1f}s)")
            
            print("✓ 延迟初始化完成")
            
        except Exception as e:
            print(f"延迟初始化失败: {e}")
            # 即使失败也要标记完成，避免界面卡死
            self._initialization_completed = True
            if hasattr(self, 'statusbar'):
                self.statusbar.showMessage("初始化完成（部分功能可能不可用）")
    
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
            # 布局工具导入失败，使用默认设置
            pass
        except Exception as e:
            # StackedWidget设置失败，使用默认设置
            pass
    
    def closeEvent(self, event):
        """窗口关闭事件 - 通过主控制器统一保存会话"""
        try:
            # 通过主控制器统一保存当前会话
            if hasattr(self, 'main_controller'):
                self.main_controller.handle_window_close()
            
            event.accept()
        except Exception as e:
            # 参数保存失败，仍然允许关闭
            print(f"关闭时保存会话失败: {e}")
            event.accept()
    
    def initialize_parameter_system(self):
        """初始化全局参数系统"""
        # 全局参数管理器已经通过导入自动创建
        # 参数管理器在初始化时会自动加载用户参数（如果存在）
        # 具体的UI同步由各个控制器负责
        try:
            # 检查参数系统是否正常工作
            beam_params = global_params.get_module_parameters('beam')
            detector_params = global_params.get_module_parameters('detector')
            
            if beam_params and detector_params:
                print("✓ 全局参数系统初始化成功")
            else:
                print("⚠ 参数系统初始化不完整，使用默认参数")
                
        except Exception as e:
            print(f"⚠ 参数系统初始化警告: {e}")
            pass
    
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
