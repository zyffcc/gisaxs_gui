import os
# Ensure Matplotlib uses a Qt backend compatible with PyQt5
os.environ.setdefault('MPLBACKEND', 'Qt5Agg')

# main.py

import sys
import os
import warnings
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from src.gimap.app.window_view import ApplicationWindowView
from src.gimap.app.presentation.assets import app_icon
from src.gimap.app.main_window import MainWindowComponents
from src.gimap.app.menu_manager import MenuManager
from src.gimap.app.runtime import ApplicationRuntime
from src.gimap.app import AppContext
from src.gimap.app.bootstrap import create_app_context
from src.gimap.integrations.bornagain import BornAgainSimulator

def configure_application_font(app: QApplication, point_size: int = 9) -> None:
    """Install the normal UI font once, immediately after QApplication exists."""
    if sys.platform.startswith("win"):
        font = QFont("Segoe UI", point_size)
    else:
        font = QFont(app.font())
        font.setPointSize(point_size)
    app.setFont(font)


class MainWindow(QMainWindow, ApplicationWindowView):
    def __init__(self, app_context: AppContext):
        super().__init__()
        self.app_context = app_context
        import time
        self._startup_time = time.time()
        
        self.setupUi(self)
        self.setWindowTitle("GIMaP")
        self.setWindowIcon(app_icon())
        self.components = MainWindowComponents(self)
        
        # 设置初始状态栏消息（英文）
        if hasattr(self, 'statusbar'):
            self.statusbar.showMessage("UI ready. Initializing components...")
        
        # 快速初始化：仅设置基本UI
        self.setup_window()
        ui_ready_time = time.time() - self._startup_time
        print(f"UI ready in {ui_ready_time:.2f}s")
        
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
            self.menu_manager = MenuManager(self, settings=self.app_context.settings)
            
            # 更新状态栏
            if hasattr(self, 'statusbar'):
                self.statusbar.showMessage("Initializing feature workspaces...")
            
            # 初始化 application runtime
            self.runtime = ApplicationRuntime(
                self,
                self,
                simulation_port=BornAgainSimulator(runner=self.app_context.jobs),
            )

            
            
            # 连接菜单信号
            self.connect_menu_signals()
            
            # 标记初始化完成
            self._initialization_completed = True
            

            # 计算总启动时间
            total_time = time.time() - self._startup_time
            print(f"Startup complete in {total_time:.2f}s")
            
            # 更新状态栏
            if hasattr(self, 'statusbar'):
                self.statusbar.showMessage(f"GIMaP ready (startup: {total_time:.1f}s)")
            
            print("Deferred initialization finished")
            
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
            from src.gimap.app.presentation.settings_dialog import SettingsDialog
            dialog = SettingsDialog(
                self,
                preferences=self.app_context.preferences,
            )
            dialog.exec_()
        except ImportError:
            # If settings_dialog is missing, show a notice
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "Information", "Display settings are under development...")
    
    def setup_window(self):
        """Apply the injected UI scale after the app-owned shell is composed."""
        from src.gimap.app.presentation.responsive_layout import effective_ui_scale

        scale = effective_ui_scale(
            self,
            preferences=self.app_context.preferences,
        ) / 100.0
        app = QApplication.instance()
        font = QFont(app.font() if app is not None else self.font())
        font.setPointSizeF(max(4.0, 9.0 * scale))
        if app is not None:
            app.setFont(font)
        self.scale_factor = scale
    
    def closeEvent(self, event):
        """窗口关闭事件 - 通过主控制器统一保存会话"""
        try:
            # 通过主控制器统一保存当前会话
            if hasattr(self, 'runtime'):
                self.runtime.handle_window_close()
            # 保存分割器比例到用户设置
            try:
                if hasattr(self, 'components'):
                    self.components.save_state()
            except Exception:
                pass
            self.app_context.save_session()
            if self.app_context.jobs is not None:
                self.app_context.jobs.shutdown()
            
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
            beam_params = self.app_context.settings.get_section('beam')
            detector_params = self.app_context.settings.get_section('detector')
            
            if beam_params and detector_params:
                print("Global parameter system initialized successfully")
            else:
                print("⚠ Parameter system initialization incomplete, using default parameters")
                
        except Exception as e:
            print(f"⚠ Parameter system initialization warning: {e}")
            pass
    
    def get_software_parameters(self):
        """提供给外部调用的参数获取方法"""
        return self.app_context.settings.snapshot()
    
    def get_physics_parameters(self):
        """提供给外部调用的物理参数获取方法"""
        return {
            section: self.app_context.settings.get_section(section)
            for section in ("beam", "detector", "sample", "system")
        }


def main():
    """主函数"""
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app_context = create_app_context()
    configure_application_font(app, point_size=9)
    
    # 设置应用程序属性
    app.setApplicationName("GIMaP")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("GIMaP")
    app.setWindowIcon(app_icon())
    
    # 创建主窗口
    window = MainWindow(app_context)
    requested_size = os.environ.get("GIMAP_WINDOW_SIZE", "").strip().lower()
    if "x" in requested_size:
        try:
            width_text, height_text = requested_size.split("x", 1)
            window.resize(max(960, int(width_text)), max(640, int(height_text)))
        except ValueError:
            print(f"Ignoring invalid GIMAP_WINDOW_SIZE={requested_size!r}; expected WIDTHxHEIGHT")
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
