import os
# Ensure Matplotlib uses a Qt backend compatible with PyQt5
os.environ.setdefault('MPLBACKEND', 'Qt5Agg')

# main.py

import sys
import os
import warnings
from PyQt5.QtWidgets import QApplication, QMainWindow
from ui.main_window import Ui_MainWindow  # 导入转换后的 UI 类
from ui.menu_manager import MenuManager
from controllers import MainController
from core.window_manager import window_manager

# 配置matplotlib（尽量轻量，完整预热放到窗口显示后）
try:
    import matplotlib
    # 延后重型子模块导入与字体缓存构建
except ImportError:
    pass

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
        
        # 设置初始状态栏消息（英文）
        if hasattr(self, 'statusbar'):
            self.statusbar.showMessage("UI ready. Initializing components...")
        
        # 快速初始化：仅设置基本UI
        self.setup_window()
        # 插入可左右拖动的分割器
        self._install_splitter()
        # 尝试恢复分割器比例
        self._restore_splitter_sizes()
        
        ui_ready_time = time.time() - self._startup_time
        print(f"✓ UI ready in {ui_ready_time:.2f}s")
        
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
                self.statusbar.showMessage("Initializing parameter system...")
            
            # 初始化全局参数系统
            self.initialize_parameter_system()
            
            # 更新状态栏
            if hasattr(self, 'statusbar'):
                self.statusbar.showMessage("Initializing menus...")
            
            # 初始化菜单管理器
            self.menu_manager = MenuManager(self)
            
            # 更新状态栏
            if hasattr(self, 'statusbar'):
                self.statusbar.showMessage("Initializing controllers...")
            
            # 初始化主控制器
            self.main_controller = MainController(self, self)

            
            
            # 连接菜单信号
            self.connect_menu_signals()
            
            # 标记初始化完成
            self._initialization_completed = True
            

            # 计算总启动时间
            total_time = time.time() - self._startup_time
            print(f"✓ Startup complete in {total_time:.2f}s")
            
            # 更新状态栏
            if hasattr(self, 'statusbar'):
                self.statusbar.showMessage(f"GISAXS Toolkit ready (startup: {total_time:.1f}s)")
            
            print("✓ Deferred initialization finished")
            
        except Exception as e:
            print(f"Deferred initialization failed: {e}")
            # 即使失败也要标记完成，避免界面卡死
            self._initialization_completed = True
            if hasattr(self, 'statusbar'):
                self.statusbar.showMessage("Initialization finished (some features may be unavailable)")
    
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
            # If settings_dialog is missing, show a notice
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "Information", "Display settings are under development...")
    
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

    def _install_splitter(self):
        """在中央区域为侧边栏与主内容添加可调整比例的分割器（不改 .ui）。"""
        try:
            from PyQt5.QtWidgets import QSplitter
            from PyQt5.QtCore import Qt

            # 中央部件已有水平布局，包含侧边栏与主内容
            if not hasattr(self, 'horizontalLayout') or not hasattr(self, 'sideBarScrollArea') or not hasattr(self, 'mainContentWidget'):
                return

            # 如果已经安装过分割器则跳过
            # 通过检查父子关系或类型来判断
            if isinstance(self.sideBarScrollArea.parentWidget(), QSplitter) and isinstance(self.mainContentWidget.parentWidget(), QSplitter):
                return

            splitter = QSplitter(self.centralwidget)
            splitter.setOrientation(Qt.Horizontal)
            splitter.setHandleWidth(6)

            # 解除原布局中的控件占位并加入分割器
            # 从布局中移除元素但不删除控件
            idx = self.horizontalLayout.indexOf(self.sideBarScrollArea)
            if idx != -1:
                self.horizontalLayout.takeAt(idx)
            idx = self.horizontalLayout.indexOf(self.mainContentWidget)
            if idx != -1:
                self.horizontalLayout.takeAt(idx)

            splitter.addWidget(self.sideBarScrollArea)
            splitter.addWidget(self.mainContentWidget)

            # 侧边栏当前是固定150宽度（minimum/maximum都为150），为了允许拖动，
            # 放宽最大宽度并保留最小宽度以保证最小显示。
            try:
                self.sideBarScrollArea.setMaximumWidth(16777215)
                self.mainAreaWidgetContents.setMaximumWidth(16777215)
            except Exception:
                pass

            # 设置初始比例：侧边栏150，主内容其余
            # 使用总宽度的近似值来初始化
            total = max(self.width(), 800)
            sidebar = 150
            main = max(total - sidebar, 400)
            splitter.setSizes([sidebar, main])

            # 将分割器加入到原有水平布局
            self.horizontalLayout.addWidget(splitter)
            # 记住分割器引用
            self._main_splitter = splitter

            # 允许两个面板都可拉伸
            splitter.setStretchFactor(0, 0)  # 保持侧边栏最小占比
            splitter.setStretchFactor(1, 1)  # 主内容随窗口扩展

        except Exception as e:
            # 分割器安装失败则忽略，不影响原布局
            print(f"Install splitter failed: {e}")

    def _restore_splitter_sizes(self):
        """从用户设置恢复分割器比例。"""
        try:
            from core.user_settings import user_settings
            sizes = user_settings.get('main_splitter_sizes', None)
            if sizes and hasattr(self, '_main_splitter'):
                # 验证为两个整数
                if isinstance(sizes, (list, tuple)) and len(sizes) == 2:
                    s0 = max(50, int(sizes[0]))
                    s1 = max(100, int(sizes[1]))
                    self._main_splitter.setSizes([s0, s1])
        except Exception:
            pass
    
    def closeEvent(self, event):
        """窗口关闭事件 - 通过主控制器统一保存会话"""
        try:
            # 通过主控制器统一保存当前会话
            if hasattr(self, 'main_controller'):
                self.main_controller.handle_window_close()
            # 保存分割器比例到用户设置
            try:
                if hasattr(self, '_main_splitter'):
                    sizes = self._main_splitter.sizes()
                    from core.user_settings import user_settings
                    user_settings.set('main_splitter_sizes', sizes)
                    user_settings.save_settings()
            except Exception:
                pass
            
            event.accept()
        except Exception as e:
            # Even if saving fails, still allow closing
            print(f"Failed to save session on close: {e}")
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
                print("✓ Global parameter system initialized successfully")
            else:
                print("⚠ Parameter system initialization incomplete, using default parameters")
                
        except Exception as e:
            print(f"⚠ Parameter system initialization warning: {e}")
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
    
    # 轻量预热：在窗口显示后构建字体缓存与绘图后端，避免首次绘图卡顿
    try:
        from PyQt5.QtCore import QTimer
        def _matplotlib_warmup():
            try:
                import matplotlib
                # 导入最小子模块并创建一次性Figure以触发font cache构建
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                fig = Figure(figsize=(1, 1))
                _ = FigureCanvas(fig)
                # 可选：设置常用rcParams（轻量）
                import matplotlib.pyplot as plt
                # 确保使用内置的 DejaVu 字体家族，避免缺失上标负号（superscript minus）等字形
                try:
                    fam = plt.rcParams.get('font.family', [])
                    if not fam:
                        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
                    elif isinstance(fam, str):
                        plt.rcParams['font.family'] = [fam, 'DejaVu Sans', 'Arial', 'sans-serif']
                    else:
                        # prepend DejaVu Sans if not present
                        if 'DejaVu Sans' not in fam:
                            plt.rcParams['font.family'] = ['DejaVu Sans'] + list(fam)
                except Exception:
                    pass
                # 统一坐标轴负号渲染
                plt.rcParams.setdefault('axes.unicode_minus', False)
            except Exception:
                pass
        QTimer.singleShot(200, _matplotlib_warmup)
    except Exception:
        pass
    
    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
