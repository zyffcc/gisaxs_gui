"""
Cut Fitting 控制器 - 处理GISAXS数据的裁剪和拟合功能
"""

import os
import json
from collections import defaultdict
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QThread, QTimer, QPoint
from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QGraphicsScene,
    QVBoxLayout,
    QWidget,
    QMainWindow,
    QMenu,
    QAction,
    QTextBrowser,
    QSizePolicy,
    QDialog,
    QInputDialog,
    QComboBox,
    QStackedWidget,
    QGridLayout,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
)

# 导入探测器参数对话框
from ui.detector_parameters_dialog import DetectorParametersDialog

# 导入模型参数管理器
from config.model_parameters_manager import ModelParametersManager

# 导入通用参数触发管理器
from utils.universal_parameter_trigger_manager import UniversalParameterTriggerManager
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence

"""Heavy libraries (matplotlib/fabio) are lazy-loaded to speed up startup."""
# Lazy availability flags (None means unchecked yet)
MATPLOTLIB_AVAILABLE = None
FABIO_AVAILABLE = None


def is_matplotlib_available():
    """Lazy-check matplotlib availability and cache the result.

    Returns True if matplotlib can be imported, else False. This avoids importing it at
    process start; the first call may pay the cost, which is acceptable when needed.
    """
    global MATPLOTLIB_AVAILABLE
    if MATPLOTLIB_AVAILABLE is None:
        try:
            # Import minimal submodules needed later; do not configure backend here
            import matplotlib  # noqa: F401
            MATPLOTLIB_AVAILABLE = True
        except Exception:
            MATPLOTLIB_AVAILABLE = False
    return MATPLOTLIB_AVAILABLE


def is_fabio_available():
    """Lazy-check fabio availability and cache the result."""
    global FABIO_AVAILABLE
    if FABIO_AVAILABLE is None:
        try:
            import fabio  # noqa: F401
            FABIO_AVAILABLE = True
        except Exception:
            FABIO_AVAILABLE = False
    return FABIO_AVAILABLE


try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    from scipy import ndimage
    from scipy.optimize import least_squares
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class IndependentMatplotlibWindow(QMainWindow):
    """独立的matplotlib窗口，支持完整的交互操作和视图保持"""
    
    # 添加信号，用于传递框选结果
    region_selected = pyqtSignal(dict)  # 传递选择区域信息
    status_updated = pyqtSignal(str)  # 传递状态更新信息
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GISAXS Image Viewer - Independent Window (Selection Mode: Right-click to activate)")
        self.setGeometry(100, 100, 900, 700)
        
        # 创建中央widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout(central_widget)

        # 创建matplotlib图形（延迟导入重库）
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.ax = None
        try:
            if is_matplotlib_available():
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
                self.figure = Figure(figsize=(10, 8), dpi=100)
                self.canvas = FigureCanvas(self.figure)
                self.toolbar = NavigationToolbar(self.canvas, self)
                layout.addWidget(self.toolbar)
                layout.addWidget(self.canvas)
                self.ax = self.figure.add_subplot(111)
        except Exception:
            # 如果导入失败，保持占位，后续逻辑会检测可用性
            pass

        self.current_image = None
        self.colorbar = None
        
        # 存储视图状态
        self.current_xlim = None
        self.current_ylim = None
        self.last_image_shape = None
        
        # Q空间缓存
        self._q_detector = None
        self._q_cache_key = None  # 用于检查参数是否改变
        self._qy_mesh = None
        self._qz_mesh = None
        
        # 框选功能相关变量
        self.selection_mode = False
        self.selection_start = None
        self.selection_rect = None
        self.current_selection = None
        self.parameter_selection = None  # 新增：参数选择矩形
        
        # 设置窗口可以接收键盘焦点
        self.setFocusPolicy(Qt.StrongFocus)
        if self.canvas is not None:
            self.canvas.setFocusPolicy(Qt.StrongFocus)
            self.canvas.setFocus()
        central_widget.setFocusPolicy(Qt.StrongFocus)
        
        # 连接视图变化事件
        if self.ax is not None:
            self.ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
            self.ax.callbacks.connect('ylim_changed', self._on_ylim_changed)
        
        # 连接鼠标事件用于框选
        if self.canvas is not None:
            self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
            self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
            self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
            self.canvas.mpl_connect('key_press_event', self._on_key_press)
    
    def _on_xlim_changed(self, ax):
        """X轴范围改变时的回调"""
        self.current_xlim = ax.get_xlim()
    
    def _on_ylim_changed(self, ax):
        """Y轴范围改变时的回调"""
        self.current_ylim = ax.get_ylim()
    
    def _on_mouse_press(self, event):
        """鼠标按下事件"""
        if event.button == 3:  # 右键
            if not self.selection_mode:
                self.selection_mode = True
                self.setWindowTitle("GISAXS Image Viewer - 框选模式激活 (拖拽选择区域, ESC退出)")
                self.canvas.setCursor(Qt.CrossCursor)
                self.status_updated.emit("Selection mode activated - Drag to select, Right-click again to exit")
            else:
                self._exit_selection_mode()
            return
            
        if event.button == 1 and self.selection_mode and event.inaxes == self.ax:  # 左键在axes内
            self.selection_start = (event.xdata, event.ydata)
            if self.selection_rect:
                self.selection_rect.remove()
                self.selection_rect = None
            self.status_updated.emit("Selection started - drag to define region")
    
    def _on_mouse_move(self, event):
        """鼠标移动事件"""
        if (self.selection_mode and self.selection_start and 
            event.inaxes == self.ax and event.xdata and event.ydata):
            
            # 移除旧的矩形
            if self.selection_rect:
                self.selection_rect.remove()
            
            # 计算矩形参数
            x0, y0 = self.selection_start
            x1, y1 = event.xdata, event.ydata
            
            width = abs(x1 - x0)
            height = abs(y1 - y0)
            x_min = min(x0, x1)
            y_min = min(y0, y1)
            
            # 创建新的矩形
            from matplotlib.patches import Rectangle
            self.selection_rect = Rectangle(
                (x_min, y_min), width, height,
                linewidth=2, edgecolor='red', facecolor='none', alpha=0.7
            )
            self.ax.add_patch(self.selection_rect)
            self.canvas.draw_idle()
    
    def _on_mouse_release(self, event):
        """鼠标释放事件"""
        if (self.selection_mode and self.selection_start and 
            event.button == 1 and event.inaxes == self.ax and 
            event.xdata and event.ydata):
            
            # 计算最终选择区域
            x0, y0 = self.selection_start
            x1, y1 = event.xdata, event.ydata
            
            # 检查当前是否为Q坐标模式
            show_q_axis = self._should_show_q_axis()
            
            # 确保有效的选择区域（最小尺寸检查）
            min_size_threshold = 0.001 if show_q_axis else 5  # Q模式下使用更小的阈值
            if abs(x1 - x0) > min_size_threshold and abs(y1 - y0) > min_size_threshold:
                # 计算区域参数
                width = abs(x1 - x0)
                height = abs(y1 - y0)
                center_x = (x0 + x1) / 2
                center_y = (y0 + y1) / 2
                
                # 获取图像尺寸用于换算
                image_shape = getattr(self, 'current_image_shape', (1, 1))
                img_height, img_width = image_shape
                
                if show_q_axis:
                    # Q坐标模式：直接使用Q坐标，同时提供像素坐标用于显示
                    selection_info = {
                        'center_x': center_x,      # Q坐标 (qy)
                        'center_y': center_y,      # Q坐标 (qz)
                        'width': width,            # Q坐标宽度
                        'height': height,          # Q坐标高度
                        'is_q_space': True,        # 标记为Q空间坐标
                        'bounds': {
                            'x_min': min(x0, x1),
                            'x_max': max(x0, x1),
                            'y_min': min(y0, y1),
                            'y_max': max(y0, y1)
                        }
                    }
                    
                    # 更新标题显示Q坐标信息
                    self.setWindowTitle(
                        f"GISAXS Image Viewer - 已选择Q区域: "
                        f"中心({center_x:.6f}, {center_y:.6f}) nm⁻¹ "
                        f"尺寸({width:.6f}×{height:.6f}) nm⁻¹"
                    )
                else:
                    # 像素坐标模式：坐标转换：图像经过flipud翻转后再用origin='lower'显示
                    # 最终matplotlib坐标与原始图像坐标是一致的
                    original_center_y = center_y
                    
                    # 构建选择信息
                    selection_info = {
                        'center_x': center_x,
                        'center_y': center_y,  # matplotlib坐标
                        'width': width,
                        'height': height,
                        'pixel_center_x': int(center_x),
                        'pixel_center_y': int(original_center_y),  # 原始图像坐标（0在左下角）
                        'pixel_width': int(width),
                        'pixel_height': int(height),
                        'is_q_space': False,       # 标记为像素空间坐标
                        'bounds': {
                            'x_min': min(x0, x1),
                            'x_max': max(x0, x1),
                            'y_min': min(y0, y1),
                            'y_max': max(y0, y1)
                        }
                    }
                    
                    # 更新标题显示选择信息（使用原始图像坐标）
                    self.setWindowTitle(
                        f"GISAXS Image Viewer - 已选择区域: "
                        f"中心({selection_info['pixel_center_x']}, {selection_info['pixel_center_y']}) "
                        f"尺寸({selection_info['pixel_width']}×{selection_info['pixel_height']})"
                    )
                
                self.current_selection = selection_info
                
                # 发送选择信号
                self.region_selected.emit(selection_info)
            
            # 重置选择状态
            self.selection_start = None
    
    def _on_key_press(self, event):
        """键盘事件"""
        if event.key == 'escape':
            self._exit_selection_mode()
        elif event.key == 'delete' or event.key == 'backspace':
            self._clear_selection()
    
    def keyPressEvent(self, event):
        """Qt键盘事件处理（作为matplotlib事件的补充）"""
        try:
            if event.key() == Qt.Key_Escape:
                self._exit_selection_mode()
            elif event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
                self._clear_selection()
            else:
                super().keyPressEvent(event)
        except Exception:
            super().keyPressEvent(event)
    
    def mousePressEvent(self, event):
        """Qt鼠标事件处理"""
        # 确保canvas有焦点以接收键盘事件
        self.canvas.setFocus()
        super().mousePressEvent(event)
    
    def _exit_selection_mode(self):
        """退出框选模式"""
        self.selection_mode = False
        self.selection_start = None
        self.canvas.unsetCursor()
        self.setWindowTitle("GISAXS Image Viewer - Independent Window (右键激活框选)")
        if self.selection_rect:
            self.selection_rect.remove()
            self.selection_rect = None
            self.canvas.draw_idle()
    
    def _clear_selection(self):
        """清除当前选择"""
        if self.selection_rect:
            self.selection_rect.remove()
            self.selection_rect = None
            self.canvas.draw_idle()
        self.current_selection = None
        if self.selection_mode:
            self.setWindowTitle("GISAXS Image Viewer - 框选模式激活 (拖拽选择区域)")
        else:
            self.setWindowTitle("GISAXS Image Viewer - Independent Window (右键激活框选)")
    
    def update_image(self, image_data, vmin=None, vmax=None, use_log=True):
        """更新独立窗口中的图像，保持用户的视图焦点，支持自定义颜色范围和线性/对数切换"""
        try:
            # 检查图像尺寸是否改变
            current_shape = image_data.shape
            shape_changed = (self.last_image_shape is None or 
                           self.last_image_shape != current_shape)
            
            # 更新当前图像尺寸信息（用于框选功能）
            self.current_image_shape = current_shape
            
            if shape_changed:
                # 图像尺寸改变了，重置视图
                self.current_xlim = None
                self.current_ylim = None
                self.last_image_shape = current_shape
            
            # 保存轴限制 - 备份当前视图状态
            saved_xlim = self.current_xlim
            saved_ylim = self.current_ylim
            preserve_view = (not shape_changed and saved_xlim is not None and saved_ylim is not None)
            
            # 暂时断开视图变化回调
            try:
                xlim_cid = None
                ylim_cid = None
                
                try:
                    for cid, func in self.ax.callbacks.callbacks['xlim_changed'].items():
                        if func.func == self._on_xlim_changed:
                            xlim_cid = cid
                            break
                    for cid, func in self.ax.callbacks.callbacks['ylim_changed'].items():
                        if func.func == self._on_ylim_changed:
                            ylim_cid = cid
                            break
                    
                    if xlim_cid is not None:
                        self.ax.callbacks.disconnect(xlim_cid)
                    if ylim_cid is not None:
                        self.ax.callbacks.disconnect(ylim_cid)
                        
                except (AttributeError, KeyError):
                    try:
                        self.ax.callbacks.disconnect('xlim_changed', self._on_xlim_changed)
                        self.ax.callbacks.disconnect('ylim_changed', self._on_ylim_changed)
                    except TypeError:
                        pass
                        
            except Exception:
                pass
            
            # 安全地移除旧的colorbar
            if self.colorbar is not None:
                try:
                    self.colorbar.remove()
                except Exception:
                    pass
                finally:
                    self.colorbar = None
            
            # 清除axes
            self.ax.clear()
            
            # 根据模式处理图像数据
            if use_log:
                safe_data = np.where(image_data > 0, image_data, 0.001)
                processed_data = np.log(safe_data, dtype=np.float32)
                scale_text = "Log Scale"
                colorbar_label = "Log Intensity"
            else:
                processed_data = image_data.astype(np.float32)
                scale_text = "Linear Scale"
                colorbar_label = "Intensity"
            
            # 如果没有提供vmin/vmax，则自动计算
            if vmin is None or vmax is None:
                auto_vmin = np.percentile(processed_data, 1)
                auto_vmax = np.percentile(processed_data, 99)
                vmin = vmin if vmin is not None else auto_vmin
                vmax = vmax if vmax is not None else auto_vmax
            
            # 垂直翻转图像数据以修正显示方向
            processed_data = np.flipud(processed_data)
            
            # 检查是否需要显示Q轴
            show_q_axis = self._should_show_q_axis()
            
            if show_q_axis:
                # 获取Q轴参数并计算extent
                extent = self._get_q_axis_extent(image_data.shape)
                
                # 获取缓存的Q空间网格来计算精确的extent
                qy_mesh, qz_mesh = self._get_cached_q_meshgrids()
                
                if qy_mesh is not None and qz_mesh is not None:
                    # 使用Q网格来计算精确的extent [left, right, bottom, top]
                    qy_min, qy_max = qy_mesh.min(), qy_mesh.max()
                    qz_min, qz_max = qz_mesh.min(), qz_mesh.max()
                    q_extent = [qy_min, qy_max, qz_min, qz_max]
                    
                    # 使用imshow显示Q轴坐标
                    self.current_image = self.ax.imshow(processed_data, cmap='viridis', aspect='equal', 
                                                      origin='lower', interpolation='nearest',
                                                      vmin=vmin, vmax=vmax, extent=q_extent)
                else:
                    # 如果Q网格不可用，回退到普通extent
                    self.current_image = self.ax.imshow(processed_data, cmap='viridis', aspect='equal', 
                                                      origin='lower', interpolation='nearest',
                                                      vmin=vmin, vmax=vmax, extent=extent)
                
                # 设置Q轴标签
                self.ax.set_xlabel(r'$q_y$ (nm$^{-1}$)')
                self.ax.set_ylabel(r'$q_z$ (nm$^{-1}$)')
            else:
                # 显示图像，使用像素坐标
                self.current_image = self.ax.imshow(processed_data, cmap='viridis', aspect='equal', 
                                                  origin='lower', interpolation='nearest',
                                                  vmin=vmin, vmax=vmax)
                # 设置像素坐标标签
                self.ax.set_xlabel('Pixels (Horizontal)')
                self.ax.set_ylabel('Pixels (Vertical)')
            
            # 设置标题
            coord_info = "Q-space" if show_q_axis else "Pixel coordinates"
            self.ax.set_title(f'GISAXS Image ({scale_text}) - {image_data.shape[1]}×{image_data.shape[0]} ({coord_info})\nVmin: {vmin:.3f}, Vmax: {vmax:.3f}')
            
            # 设置坐标轴比例
            if show_q_axis:
                self.ax.set_aspect('equal')  # Q空间也使用equal aspect，保持不拉伸
            else:
                self.ax.set_aspect('equal')  # 像素空间使用equal aspect
            
            # 创建新的颜色条
            try:
                self.colorbar = self.figure.colorbar(self.current_image, ax=self.ax)
                self.colorbar.set_label(colorbar_label)
            except Exception:
                self.colorbar = None
            
            # 先进行布局调整
            self.figure.tight_layout()
            
            # 在所有布局操作完成后，设置/恢复视图范围
            if preserve_view:
                self.ax.set_xlim(saved_xlim)
                self.ax.set_ylim(saved_ylim)
                self.current_xlim = saved_xlim
                self.current_ylim = saved_ylim
            else:
                # 根据显示模式设置默认视图范围
                if show_q_axis:
                    # Q轴模式：让matplotlib自动设置范围（由extent控制）
                    self.ax.autoscale()

                else:
                    # 像素模式：设置为像素坐标范围
                    self.ax.set_xlim(-0.5, processed_data.shape[1] - 0.5)
                    self.ax.set_ylim(-0.5, processed_data.shape[0] - 0.5)
                
                self.current_xlim = self.ax.get_xlim()
                self.current_ylim = self.ax.get_ylim()
            
            # 重新连接视图变化回调
            try:
                self.ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
                self.ax.callbacks.connect('ylim_changed', self._on_ylim_changed)
            except Exception:
                pass
            
            # 刷新画布
            self.canvas.draw()
            
            # 最终验证：确保视图没有被canvas刷新影响
            if preserve_view:
                def final_view_check():
                    current_xlim_after_draw = self.ax.get_xlim()
                    current_ylim_after_draw = self.ax.get_ylim()
                    if (abs(current_xlim_after_draw[0] - saved_xlim[0]) > 0.01 or 
                        abs(current_xlim_after_draw[1] - saved_xlim[1]) > 0.01 or
                        abs(current_ylim_after_draw[0] - saved_ylim[0]) > 0.01 or 
                        abs(current_ylim_after_draw[1] - saved_ylim[1]) > 0.01):
                        self.ax.set_xlim(saved_xlim)
                        self.ax.set_ylim(saved_ylim)
                        self.current_xlim = saved_xlim
                        self.current_ylim = saved_ylim
                        self.canvas.draw_idle()
                
                QTimer.singleShot(50, final_view_check)
            
        except Exception as e:
            pass
    
    def _convert_q_to_pixel_coordinates(self, center_qy, center_qz, width_q, height_q):
        """将Q坐标转换为像素坐标"""
        try:
            # 获取缓存的Q空间网格
            qy_mesh, qz_mesh = self._get_cached_q_meshgrids()
            
            if qy_mesh is None or qz_mesh is None:
                # 如果Q网格不可用，返回默认值
                return {'center_x': 0, 'center_y': 0, 'width': 100, 'height': 100}
            
            # 获取图像尺寸
            if hasattr(self, 'current_stack_data') and self.current_stack_data is not None:
                img_height, img_width = self.current_stack_data.shape
            else:
                img_height, img_width = qy_mesh.shape
            
            # 找到最接近目标Q坐标的像素位置
            qy_diff = np.abs(qy_mesh - center_qy)
            qz_diff = np.abs(qz_mesh - center_qz)
            combined_diff = qy_diff + qz_diff
            center_idx = np.unravel_index(np.argmin(combined_diff), qy_mesh.shape)
            center_pixel_y, center_pixel_x = center_idx
            
            # 计算Q空间到像素空间的比例因子
            qy_range = qy_mesh.max() - qy_mesh.min()
            qz_range = qz_mesh.max() - qz_mesh.min()
            pixel_x_range = img_width
            pixel_y_range = img_height
            
            qy_to_pixel_ratio = pixel_x_range / qy_range
            qz_to_pixel_ratio = pixel_y_range / qz_range
            
            # 转换宽度和高度
            width_pixel = width_q * qy_to_pixel_ratio
            height_pixel = height_q * qz_to_pixel_ratio
            
            result = {
                'center_x': int(center_pixel_x),
                'center_y': int(center_pixel_y),
                'width': int(width_pixel),
                'height': int(height_pixel)
            }
            
            return result
            
        except Exception as e:
            return {'center_x': 0, 'center_y': 0, 'width': 100, 'height': 100}

    def _update_cutline_labels_units(self):
        """根据当前显示模式更新Cut Line标签的单位"""
        try:
            show_q_axis = self._should_show_q_axis()
            
            if show_q_axis:
                # Q坐标模式：添加 (nm⁻¹) 单位
                unit_suffix = " (nm⁻¹)"
            else:
                # 像素坐标模式：添加 (pixel) 单位
                unit_suffix = " (pixel)"
            
            # 更新Center标签
            if hasattr(self.ui, 'gisaxsInputCenterVerticalLabel'):
                self.ui.gisaxsInputCenterVerticalLabel.setText(f"Vertical.{unit_suffix}")
            
            if hasattr(self.ui, 'gisaxsInputCenterParallelLabel'):
                self.ui.gisaxsInputCenterParallelLabel.setText(f"Parallel.{unit_suffix}")
            
            # 更新Cut Line标签
            if hasattr(self.ui, 'gisaxsInputCutLineVerticalLabel'):
                self.ui.gisaxsInputCutLineVerticalLabel.setText(f"Vertical.{unit_suffix}")
            
            if hasattr(self.ui, 'gisaxsInputCutLineParallelLabel'):
                self.ui.gisaxsInputCutLineParallelLabel.setText(f"Parallel.{unit_suffix}")
                
        except Exception:
            pass

    def _should_show_q_axis(self):
        """检查是否应该显示Q轴"""
        try:
            from core.global_params import GlobalParameterManager
            global_params = GlobalParameterManager()
            return global_params.get_parameter('fitting', 'detector.show_q_axis', False)
        except Exception:
            return False
    
    def _get_q_axis_extent(self, image_shape):
        """获取Q轴的extent参数和缓存Q空间网格"""
        try:
            from core.global_params import GlobalParameterManager
            global_params = GlobalParameterManager()
            
            # 获取实验参数 - 从Fitting模块专用设置中读取
            height, width = image_shape
            pixel_size_x = global_params.get_parameter('fitting', 'detector.pixel_size_x', 172.0)  # micrometers
            pixel_size_y = global_params.get_parameter('fitting', 'detector.pixel_size_y', 172.0)  # micrometers
            beam_center_x = global_params.get_parameter('fitting', 'detector.beam_center_x', width / 2.0)  # Fitting模块专用beam center
            beam_center_y = global_params.get_parameter('fitting', 'detector.beam_center_y', height / 2.0)  # Fitting模块专用beam center
            distance = global_params.get_parameter('fitting', 'detector.distance', 2565.0)  # mm
            theta_in_deg = global_params.get_parameter('beam', 'grazing_angle', 0.4)
            wavelength = global_params.get_parameter('beam', 'wavelength', 0.1045)  # nm
            
            # Q-axis calculation parameters
            
            # 创建缓存键
            cache_key = f"{width}x{height}_{pixel_size_x}_{pixel_size_y}_{beam_center_x}_{beam_center_y}_{distance}_{theta_in_deg}_{wavelength}"
            
            # 检查是否需要重新计算
            if self._q_cache_key != cache_key or self._q_detector is None:
                # 创建探测器对象并计算Q轴（延迟导入）
                from utils.q_space_calculator import create_detector_from_image_and_params
                self._q_detector = create_detector_from_image_and_params(
                    image_shape=(height, width),
                    pixel_size_x=pixel_size_x,
                    pixel_size_y=pixel_size_y,
                    beam_center_x=beam_center_x,
                    beam_center_y=beam_center_y,
                    distance=distance,
                    theta_in_deg=theta_in_deg,
                    wavelength=wavelength,
                    crop_params=None  # 不应用裁剪，因为这里使用的是处理后的图像尺寸
                )
                
                # 缓存Q空间网格
                self._qy_mesh, self._qz_mesh = self._q_detector.get_qy_qz_meshgrids()
                self._q_cache_key = cache_key
                

            
            from utils.q_space_calculator import get_q_axis_labels_and_extents
            _, _, extent = get_q_axis_labels_and_extents(self._q_detector)
            return extent
            
        except Exception:
            # 返回默认像素坐标extent
            height, width = image_shape
            return [-0.5, width - 0.5, -0.5, height - 0.5]
    
    def _get_cached_q_meshgrids(self):
        """获取缓存的Q空间网格"""
        return self._qy_mesh, self._qz_mesh
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 清理colorbar
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            finally:
                self.colorbar = None
        
        # 清理图形
        try:
            self.figure.clear()
        except Exception:
            pass
        
                        
        super().closeEvent(event)
    
    def update_parameter_selection(self, center_v, center_p, cutline_v, cutline_p):
        """根据参数更新选择框"""
        if not self.current_image or center_v == 0 and center_p == 0 and cutline_v == 0 and cutline_p == 0:
            self.clear_parameter_selection()
            return
            
        # 计算选择框的边界
        x_start = center_p - cutline_p / 2
        x_end = center_p + cutline_p / 2
        y_start = center_v - cutline_v / 2
        y_end = center_v + cutline_v / 2
        
        # 确保在图像范围内
        height, width = self.current_image.shape
        x_start = max(0, min(width - 1, x_start))
        x_end = max(0, min(width - 1, x_end))
        y_start = max(0, min(height - 1, y_start))
        y_end = max(0, min(height - 1, y_end))
        
        # 清除旧的参数选择框
        self.clear_parameter_selection()
        
        # 创建新的参数选择框
        if x_start != x_end and y_start != y_end:
            from matplotlib.patches import Rectangle
            self.parameter_selection = Rectangle((x_start, y_start), 
                                               x_end - x_start, y_end - y_start,
                                               linewidth=2, edgecolor='blue', 
                                               facecolor='none', linestyle='--',
                                               alpha=0.8)
            self.ax.add_patch(self.parameter_selection)
            self.canvas.draw()
    
    def clear_parameter_selection(self):
        """清除参数选择框"""
        if self.parameter_selection is not None:
            try:
                self.parameter_selection.remove()
            except Exception:
                pass
            finally:
                self.parameter_selection = None
                self.canvas.draw()

    def closeEvent(self, event):
        """窗口关闭事件"""
        # 清理colorbar
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            finally:
                self.colorbar = None
        
        # 清理图形
        try:
            self.figure.clear()
        except Exception:
            pass
                        
        super().closeEvent(event)


class IndependentFitWindow(QMainWindow):
    """独立的拟合结果matplotlib窗口，专门用于显示Cut分析结果"""
    
    # 状态更新信号
    status_updated = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GISAXS Cut Analysis - Independent Fit Window")
        self.setGeometry(150, 150, 800, 600)
        
        # 创建中央widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout(central_widget)

        # 创建matplotlib图形（延迟导入）
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.ax = None
        try:
            if is_matplotlib_available():
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
                self.figure = Figure(figsize=(10, 6), dpi=100)
                self.canvas = FigureCanvas(self.figure)
                self.toolbar = NavigationToolbar(self.canvas, self)
        except Exception:
            pass

        # 创建额外的控制按钮栏
        control_layout = self._create_control_buttons()

        # 添加到布局（仅当canvas可用时）
        if self.toolbar is not None:
            layout.addWidget(self.toolbar)
        layout.addLayout(control_layout)  # 额外的按钮栏
        if self.canvas is not None:
            layout.addWidget(self.canvas)

        # 创建axes
        if self.figure is not None:
            self.ax = self.figure.add_subplot(111)

        # 设置窗口可以接收键盘焦点
        self.setFocusPolicy(Qt.StrongFocus)
        if self.canvas is not None:
            self.canvas.setFocusPolicy(Qt.StrongFocus)

        # 初始化空图
        if self.figure is not None and self.canvas is not None and self.ax is not None:
            self._setup_empty_plot()
        
    def _setup_empty_plot(self):
        """设置初始空图"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Perform a cut operation to see results here\nDouble-click fitGraphicsView to open this window', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=self.ax.transAxes, fontsize=12, alpha=0.7)
        self.ax.set_xlabel('Position')
        self.ax.set_ylabel('Intensity')
        self.ax.set_title('GISAXS Cut Analysis Results')
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def _create_control_buttons(self):
        """创建额外的控制按钮"""
        from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QCheckBox, QLabel
        control_layout = QHBoxLayout()
        
        # 添加标签
        control_layout.addWidget(QLabel("Data Filter:"))
        
        # 只显示正值复选框
        self.show_positive_cb = QCheckBox("Positive Only")
        self.show_positive_cb.toggled.connect(self._on_show_positive_toggled)
        control_layout.addWidget(self.show_positive_cb)
        
        return control_layout
    
    def _on_show_positive_toggled(self, checked):
        """处理Positive Only复选框状态变化"""
        self.status_updated.emit(f"Positive Only mode: {'enabled' if checked else 'disabled'}")

    def update_plot(self, x_coords, y_intensity, x_label, y_label, title, log_x=False, log_y=False, normalize=False, y_errors=None):
        """更新拟合结果图，支持误差条"""
        try:
            # 数据预处理
            x_data = np.array(x_coords)
            y_data = np.array(y_intensity)
            
            
            # 处理误差条数据
            err_data = None
            if y_errors is not None:
                err_data = np.array(y_errors)
            
            # 应用标准化处理（按传入值）
            if normalize:
                max_intensity = np.max(y_data) if y_data.size > 0 else 0.0
                if max_intensity > 0:
                    y_data = y_data / float(max_intensity)
                    if err_data is not None:
                        err_data = err_data / float(max_intensity)
                    y_label = "Normalized Intensity"
            
            # 清除现有内容
            self.ax.clear()
            
            # 绘制数据，支持误差条
            if err_data is not None:
                # 有误差条时使用errorbar
                self.ax.errorbar(x_data, y_data, yerr=err_data, fmt='o-', 
                               markersize=4, linewidth=1.5, capsize=3, 
                               alpha=0.8, label='Data with error bars')
            else:
                # 没有误差条时使用普通的plot，或使用共享绘图函数
                try:
                    FittingController._plot_cut_data_with_log_handling(self.ax, x_data, y_data, log_x, markersize=6, linewidth=2)
                except:
                    # 如果共享函数不可用，使用基本绘图
                    self.ax.plot(x_data, y_data, 'o-', markersize=4, linewidth=1.5, alpha=0.8, label='Data')
            
            # 设置标签和标题（将 Å⁻¹ 替换为 mathtext 形式，避免字形缺失）
            try:
                # 将常见的 'Å⁻¹' 与 'A^-1' 自动替换为 mathtext 表达
                def _to_mathtext(label: str) -> str:
                    if not isinstance(label, str):
                        return label
                    # 仅替换单位部分，尽量不改变其它文本
                    return (label
                            .replace('Å⁻¹', 'Å$^{-1}$')
                            .replace('A^-1', 'Å$^{-1}$')
                            .replace('Ang^-1', 'Å$^{-1}$')
                            .replace('(Å-1)', '(Å$^{-1}$)')
                            .replace('(A^-1)', '(Å$^{-1}$)'))
                x_lbl = _to_mathtext(x_label)
                y_lbl = _to_mathtext(y_label)
            except Exception:
                x_lbl, y_lbl = x_label, y_label

            self.ax.set_xlabel(x_lbl, fontsize=13)
            self.ax.set_ylabel(y_lbl, fontsize=13)
            self.ax.set_title(title, fontsize=15)
            
            # 应用对数坐标设置
            if log_x:
                self.ax.set_xscale('log')
            else:
                self.ax.set_xscale('linear')
                
            if log_y:
                self.ax.set_yscale('log')
            else:
                self.ax.set_yscale('linear')
            
            # 网格和样式
            self.ax.grid(True, alpha=0.4, linestyle='--')
            try:
                for axis in ['top', 'bottom', 'left', 'right']:
                    self.ax.spines[axis].set_linewidth(1.8)
                self.ax.tick_params(axis='both', which='both', width=1.6, labelsize=12)
            except Exception:
                pass
            
            # 添加统计信息（位置：左下角）
            # stats_text = f'Points: {len(x_data)}\nMax: {np.max(y_data):.2e}\nMin: {np.min(y_data):.2e}'
            # self.ax.text(0.02, 0.88, stats_text, transform=self.ax.transAxes, 
            #             verticalalignment='bottom', fontsize=10, 
            #             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 调整布局
            self.figure.tight_layout()
            
            # 刷新显示
            self.canvas.draw()
            
            # 更新窗口标题
            self.setWindowTitle(f"GISAXS Cut Analysis - {title}")
            
            self.status_updated.emit(f"Independent fit window updated: {title}")
            
        except Exception as e:
            self.status_updated.emit(f"Failed to update independent fit window: {str(e)}")
            
    def closeEvent(self, event):
        """窗口关闭事件"""
        try:
            self.figure.clear()
        except Exception:
            pass
        super().closeEvent(event)


class UnifiedDisplayManager:
    """统一的显示管理器，管理所有图形显示区域"""
    
    def __init__(self, controller):
        self.controller = controller
        self.ui = controller.ui
        
    def plot_1d_data(self, q, intensity, err=None, title="", source_info="", 
                     log_x=False, log_y=False, normalize=False):
        """统一的1D数据绘制方法"""
        try:
            # 数据预处理
            plot_q = np.array(q)
            plot_I = np.array(intensity)
            plot_err = np.array(err) if err is not None else None
            
            # 标准化处理
            if normalize and len(plot_I) > 0:
                max_I = np.max(plot_I)
                if max_I > 0:
                    plot_I = plot_I / max_I
                    if plot_err is not None:
                        plot_err = plot_err / max_I
            
            # Log-Y预处理
            if log_y and len(plot_I) > 0 and not np.all(plot_I > 0):
                min_positive = np.min(plot_I[plot_I > 0]) if np.any(plot_I > 0) else 1e-10
                plot_I = np.where(plot_I <= 0, min_positive, plot_I)
                if plot_err is not None:
                    plot_err = np.where(plot_I <= min_positive, min_positive * 0.1, plot_err)
            
            # 更新GUI内显示
            self._update_gui_1d_display(plot_q, plot_I, plot_err, title, 
                                       log_x, log_y, normalize)
            
            # 更新独立窗口显示
            self._update_independent_1d_display(plot_q, plot_I, plot_err, title, 
                                               log_x, log_y, normalize)
            
            # 更新状态
            y_label = 'Intensity' + (' (normalized)' if normalize else '')
            mode_str = f"Log-X: {log_x}, Log-Y: {log_y}, Norm: {normalize}"
            self.controller.status_updated.emit(f"1D data displayed: {title} [{mode_str}]")
            
        except Exception as e:
            self.controller.status_updated.emit(f"Failed to plot 1D data: {str(e)}")
    
    def _update_gui_1d_display(self, q, intensity, err, title, log_x, log_y, normalize):
        """更新GUI内的1D数据显示"""
        try:
            if not hasattr(self.ui, 'fitGraphicsView') or not is_matplotlib_available():
                return
                
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            
            # 创建figure（4:3 比例，匹配 400x300 视图）
            figure = Figure(figsize=(8, 6))
            canvas = FigureCanvas(figure)
            ax = figure.add_subplot(111)
            
            # 使用统一的绘图逻辑
            self._unified_plot_1d_data(ax, q, intensity, err, title, log_x, log_y, normalize)
            
            figure.tight_layout()
            
            # 强制画布绘制更新
            canvas.draw()
            
            # 添加到场景
            scene = self.controller._setup_fit_graphics_scene()
            if scene is not None:
                proxy_widget = scene.addWidget(canvas)
                self.controller._fit_view_to_item(self.ui.fitGraphicsView, proxy_widget, keep_aspect=True)
                
                # 保存引用
                self.controller._current_fit_canvas = canvas
                self.controller._current_fit_figure = figure

                # 在图形绘制完成后，如果处于log-x模式，调整ROI滑块/输入框的最小值到当前显示坐标轴的下限
                try:
                    if log_x and hasattr(self.controller, '_adjust_roi_bounds_for_log_x'):
                        # 使用一个短延时，确保Qt将画布呈现后获取到最终xlim
                        QTimer.singleShot(0, self.controller._adjust_roi_bounds_for_log_x)
                except Exception:
                    pass
                
        except Exception as e:
            self.controller.status_updated.emit(f"Failed to update GUI 1D display: {str(e)}")
    
    def _unified_plot_1d_data(self, ax, q, intensity, err, title, log_x, log_y, normalize):
        """统一的1D数据绘图逻辑，同时适用于GUI内置和独立窗口"""
        try:
            # 绘制数据 - 使用与独立窗口相同的处理方式
            if err is not None:
                # 有误差条时使用errorbar
                ax.errorbar(q, intensity, yerr=err, fmt='o-', 
                           markersize=3, linewidth=1, capsize=2, 
                           alpha=0.8, label='Data with error bars')
            else:
                # 使用与独立窗口相同的Log-X处理函数
                FittingController._plot_cut_data_with_log_handling(
                    ax, q, intensity, log_x, markersize=3, linewidth=1
                )
            
            # 设置标签和标题（使用 mathtext 避免 superscript minus 字形问题；增大字号）
            ax.set_xlabel('q (Å$^{-1}$)', fontsize=13)
            ax.set_ylabel('Intensity' + (' (normalized)' if normalize else ''), fontsize=13)
            ax.set_title(title, fontsize=15)
            ax.grid(True, alpha=0.3)
            
            # 设置坐标轴 - 与独立窗口保持一致
            if log_x:
                ax.set_xscale('log')
            else:
                ax.set_xscale('linear')
                
            if log_y:
                ax.set_yscale('log')
            else:
                ax.set_yscale('linear')

            # 轴样式增强：更粗的轴线与更大的刻度字号
            try:
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(1.8)
                ax.tick_params(axis='both', which='both', width=1.6, labelsize=12)
            except Exception:
                pass
                
        except Exception as e:
            # 降级到基本绘图
            if err is not None:
                ax.errorbar(q, intensity, yerr=err, fmt='o-', markersize=3, linewidth=1, capsize=2)
            else:
                ax.plot(q, intensity, 'o-', markersize=3, linewidth=1)
            ax.set_xlabel('q (Å$^{-1}$)', fontsize=13)
            ax.set_ylabel('Intensity' + (' (normalized)' if normalize else ''), fontsize=13)
            ax.set_title(title, fontsize=15)
            ax.grid(True, alpha=0.3)
    
    def _update_independent_1d_display(self, q, intensity, err, title, log_x, log_y, normalize):
        """更新独立窗口的1D数据显示"""
        try:
            if (self.controller.independent_fit_window and 
                hasattr(self.controller.independent_fit_window, 'update_plot')):
                
                y_label = 'Intensity' + (' (normalized)' if normalize else '')
                self.controller.independent_fit_window.update_plot(
                    q, intensity, 'q (Å$^{-1}$)', y_label, title,
                    log_x, log_y, normalize, err
                )
                
        except Exception as e:
            self.controller.status_updated.emit(f"Failed to update independent 1D display: {str(e)}")
    
    def get_display_options(self):
        """获取当前显示选项"""
        return {
            'log_x': hasattr(self.ui, 'fitLogXCheckBox') and self.ui.fitLogXCheckBox.isChecked(),
            'log_y': hasattr(self.ui, 'fitLogYCheckBox') and self.ui.fitLogYCheckBox.isChecked(),
            'normalize': hasattr(self.ui, 'fitNormCheckBox') and self.ui.fitNormCheckBox.isChecked()
        }


class AsyncImageLoader(QThread):
    """异步图像加载线程"""
    
    image_loaded = pyqtSignal(np.ndarray, str)  # 图像数据, 文件路径
    progress_updated = pyqtSignal(int, str)  # 进度, 状态信息
    error_occurred = pyqtSignal(str)  # 错误信息
    
    def __init__(self):
        super().__init__()
        self.file_path = None
        self.stack_count = 1
    
    def load_image(self, file_path, stack_count=1):
        """开始加载图像"""
        self.file_path = file_path
        self.stack_count = stack_count
        self.start()
    
    def run(self):
        """在线程中运行图像加载"""
        try:
            if not is_fabio_available():
                self.error_occurred.emit("fabio library is required for CBF file processing")
                return
            
            self.progress_updated.emit(10, "Loading file...")
            
            file_ext = os.path.splitext(self.file_path)[1].lower()
            
            if file_ext != '.cbf':
                self.error_occurred.emit("Only CBF files are supported currently")
                return
            
            if self.stack_count == 1:
                # 单文件加载
                self.progress_updated.emit(50, "Loading single CBF file...")
                image_data = self._load_single_cbf_file(self.file_path)
            else:
                # 多文件叠加
                self.progress_updated.emit(30, f"Loading and stacking {self.stack_count} files...")
                image_data = self._load_multiple_cbf_files(self.file_path, self.stack_count)
            
            if image_data is not None:
                self.progress_updated.emit(90, "Processing image data...")
                self.image_loaded.emit(image_data, self.file_path)
                self.progress_updated.emit(100, "Done")
            else:
                self.error_occurred.emit("Failed to load image data")
                
        except Exception as e:
            self.error_occurred.emit(f"Error loading image: {str(e)}")
    
    def _load_single_cbf_file(self, cbf_file):
        """加载单个CBF文件"""
        try:
            import fabio
            cbf_image = fabio.open(cbf_file)
            data = cbf_image.data
            
            if data.dtype != np.float32:
                data = data.astype(np.float32, copy=False)
            
            return data
            
        except Exception as e:
            return None
    
    def _load_multiple_cbf_files(self, start_file, stack_count):
        """加载并叠加多个CBF文件"""
        try:
            file_dir = os.path.dirname(start_file)
            base_name = os.path.basename(start_file)
            
            cbf_files = [f for f in os.listdir(file_dir) if f.lower().endswith('.cbf')]
            cbf_files.sort()
            
            try:
                start_index = cbf_files.index(base_name)
            except ValueError:
                return None
            
            available_files = len(cbf_files) - start_index
            if stack_count > available_files:
                return None
            
            summed_data = None
            files_to_stack = cbf_files[start_index:start_index + stack_count]
            
            import fabio
            for i, file_name in enumerate(files_to_stack):
                file_path = os.path.join(file_dir, file_name)
                progress = 40 + int((i / len(files_to_stack)) * 40)
                self.progress_updated.emit(progress, f"Processing file {i+1}/{len(files_to_stack)}: {file_name}")
                
                try:
                    cbf_image = fabio.open(file_path)
                    data = cbf_image.data.astype(np.float32, copy=False) if cbf_image.data.dtype != np.float32 else cbf_image.data
                    
                    if summed_data is None:
                        summed_data = data.copy() if data.dtype == np.float32 else data.astype(np.float32)
                    else:
                        summed_data += data
                        
                except Exception as e:
                    continue
            
            return summed_data
            
        except Exception as e:
            return None


class FittingController(QObject):
    """Cut Fitting控制器，处理GISAXS数据的裁剪和拟合"""
    
    # 状态信号
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    parameters_changed = pyqtSignal(dict)
    fitting_completed = pyqtSignal(dict)
    
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.parent = parent  # 这是主控制器
        # 获取主窗口引用
        self.main_window = parent.parent if hasattr(parent, 'parent') else None
        
        # 统一的数据存储器
        self.q = None  # q坐标数据
        self.I = None  # I强度数据 
        self.I_fitting = None  # 拟合结果数据
        
        # 显示模式和数据源
        self.display_mode = 'normal'  # 'normal' 或 'fitting'
        self.data_source = None  # 'cut' 或 '1d'
        # ROI 与插值控制
        self._q_full_min = None
        self._q_full_max = None
        self._roi_min = None
        self._roi_max = None
        self.q_ROI = None
        self.I_ROI = None
        self._updating_roi_controls = False
        self._slider_is_source = False
        self._points_num_default = 50
        self._points_num_current = 50
        self._interp_method_default = 'Linear'

        # FittingTextBrowser增强：配置最大行数、独立窗口引用等
        self._fitting_messages_max_lines = 500
        self._detached_fitting_dialog = None
        self._detached_append = None
        self._fitting_browser_original_height = None
        self.has_fitting_data = False  # 是否有拟合数据
        
        # 当前参数
        self.current_parameters = {}
        
        # 拟合结果
        self.fitting_results = {}
        
        # 图像处理相关
        self.current_stack_data = None
        self.current_file_list = []
        # 全局数据缓存（便于导出与复用）
        self.data = None              # 当前显示（或最后加载）的单帧/合成数据
        self.summed_data = None       # 若为堆叠叠加后的数据，则在此保存
        self.cut = None               # 最近一次Cut的1D曲线 {'q':..., 'I':..., meta}
        self.fitting = None           # 最近一次Fitting的1D曲线 {'q':..., 'I':..., meta}
        # Q空间网格缓存（与当前图像及探测器参数一致时有效）
        self.qy_matrix = None
        self.qz_matrix = None
        self.qr_matrix = None
        
        # 独立matplotlib窗口
        self.independent_window = None
        
        # 独立拟合结果窗口
        self.independent_fit_window = None
        
        # 当前切割结果数据 (用于独立窗口显示)
        self.current_cut_data = None
        
        # 1D数据导入相关
        self.current_1d_data = None  # 存储导入的1D文件数据 {q, I, err}
        self.current_1d_file_path = None  # 存储当前导入的1D文件路径
        
        # 模式状态跟踪（避免重复转换）
        self._last_q_mode = None
        
        # 图像显示相关的预初始化资源
        self._graphics_scene = None
        self._figure_cache = None
        self._canvas_cache = None

        # 初始化用户设置与控件
        try:
            # 优先从 global_params 读取（与 meta 持久化保持一致），其次 user_settings
            default_points = None
            try:
                from core.global_params import global_params
                gp_val = global_params.get_parameter('fitting', 'fit.points_num', None)
                if gp_val is not None:
                    default_points = int(gp_val)
            except Exception:
                default_points = None
            try:
                from core.user_settings import user_settings
                us_val = int(user_settings.get('fit.points_num', 50))
            except Exception:
                us_val = 50
            self._points_num_default = int(default_points if default_points is not None else us_val)
            self._points_num_current = self._points_num_default
            # 插值方法从 user_settings 读取
            try:
                from core.user_settings import user_settings as _us
                self._interp_method_default = _us.get('fit.interp_method', 'Linear')
            except Exception:
                pass
        except Exception:
            pass
        try:
            self._setup_fitting_region_controls()
        except Exception:
            pass
        
        # 拟合显示相关的场景管理
        self._fit_graphics_scene = None
        self._current_fit_canvas = None
        self._current_fit_figure = None
        
        # 统一显示管理器
        self._display_manager = UnifiedDisplayManager(self)
        
        # 颜色标尺相关
        self._current_vmin = None
        self._current_vmax = None
        self._has_displayed_image = False  # 标记是否已经显示过图像
        
        # 初始化标志
        self._initialized = False
        self._initializing = True  # 标记正在初始化中
        
        # 异步图像加载线程
        self.async_image_loader = AsyncImageLoader()
        self.async_image_loader.image_loaded.connect(self._on_image_loaded)
        self.async_image_loader.progress_updated.connect(self._on_image_loading_progress)
        self.async_image_loader.error_occurred.connect(self._on_image_loading_error)
        
        # 参数选择状态
        self.current_parameter_selection = None
        
        # 显示模式管理
        self._display_mode = 'normal'  # 'normal' 或 'fitting'
        self._has_fitting_data = False  # 是否有拟合数据
        self._fitting_mode_active = False  # 是否处于拟合模式
        self._last_active_particle_ids = []  # 记录最近一次拟合时的粒子顺序
        self._particle_widgets = {}
        self._particle_parameter_meta_ids = defaultdict(list)
        self._recycled_particle_ids = []
        self._particle_widget_style_template = ''
        self._particle_widget_style_source_name = ''
        self._particle_widget_style_fallback = (
            "#fitParticleWidget_TEMPLATE {\n"
            "  border: 1px solid rgba(0,0,0,0.12);\n"
            "  border-radius: 12px;\n"
            "  padding: 8px;\n"
            "  outline: 6px solid rgba(0,0,0,0.03);\n"
            "  outline-offset: -6px;\n"
            "}"
        )
        self._particle_container_layout = None
        self._particle_add_button = None
        self._particle_show_checkboxes = {}
        self._dynamic_show_layout = None
        self._dynamic_show_container = None
        self._particle_checkbox_host_name = ''

        # 载入模式：Single/Stack/In-situ（来自组合框）
        self.load_mode = 'Single'
        self._insitu_timer = None
        self._insitu_last_file = None

        # 统一信号模式配置（默认 finished，可按控件名覆盖为 changed）
        self._default_signal_mode = 'finished'
        self._signal_mode_overrides = {
            # Fitting region 的滑块需要实时
            'fitFittingRegionSlider': 'changed',
            # Detector Para. 中的 Beam Center（若能识别到具体控件名，可在对话框创建后覆盖）
            # 示例（待与对话框控件名对齐）:
            # 'detectorBeamCenterX': 'changed',
            # 'detectorBeamCenterY': 'changed',
        }
        
        # 探测器参数对话框
        self.detector_params_dialog = None
        
        # 模型参数管理器
        self.model_params_manager = ModelParametersManager()
        
        # 通用参数触发管理器（直接使用 UniversalParameterTriggerManager，移除旧 FittingParameterTriggerManager 包装）
        self.param_trigger_manager = UniversalParameterTriggerManager(self)
        
        # 简化的状态管理（移除缓存机制）
        self._loading_parameters = False  # 标志：是否正在载入参数
        self._initializing = False  # 标志：是否正在初始化
        
        # ==========================
        # 参数去抖通用配置（统一交由 meta registry 管理）
        # ==========================
        # 去抖时间（毫秒），可根据体验调整
        self._param_debounce_ms = 280
        # 浮点比较相对+绝对容差
        self._param_abs_eps = 1e-12
        self._param_rel_eps = 1e-10
        # ROI滑块去抖时间（毫秒）：减少拖动时的全量重绘频率，缓解卡顿
        self._roi_debounce_ms = 140
        self._roi_update_timer = None
        
        # 自动K值优化状态
        self._auto_k_enabled = False  # 是否启用自动K值优化
        
        # 粒子形状连接器
        self._setup_particle_shape_connector()
        
        # 加载用户设置（包括auto-K状态）
        self._load_auto_k_enabled()
        
    def initialize(self):
        """初始化控制器"""
        if self._initialized:
            return
            
        # 先初始化UI状态（不触发信号）
        self._initialize_ui()
        # 然后设置信号连接
        self._setup_connections()
        # 会话管理已移到主控制器统一处理
        self._initialized = True
        self._initializing = False  # 初始化完成
        # 注册调试快捷键
        self._setup_meta_debug_shortcut()
        # 初始化完成后：再次根据当前组合框的实际值强制一次可见性（避免异步恢复导致的共存）
        try:
            if hasattr(self.ui, 'gisaxsInputModelCombox'):
                mode_now = self.ui.gisaxsInputModelCombox.currentText()
                self.load_mode = mode_now or getattr(self, 'load_mode', 'Single')
                # 直接调用统一的可见性更新
                self._update_stack_controls_visibility()
                # 若为 In-situ，确保数值框隐藏
                if self.load_mode == 'In-situ' and hasattr(self.ui, 'gisaxsInputStackValue'):
                    self.ui.gisaxsInputStackValue.setVisible(False)
        except Exception:
            pass

    def _setup_meta_debug_shortcut(self):
        """注册 Ctrl+Alt+M 快捷键输出 meta 注册表快照。"""
        try:
            sc = QShortcut(QKeySequence("Ctrl+Alt+M"), self.ui)
            def _dump():
                snap = self.param_trigger_manager.debug_dump_meta(verbose=False)
                self._add_fitting_message("==== META SNAPSHOT ====", "INFO")
                for wid, data in snap.items():
                    self._add_fitting_message(f"{wid}: {data}", "INFO")
            sc.activated.connect(_dump)
            self._add_fitting_message("Meta debug shortcut Ctrl+Alt+M registered", "INFO")
        except Exception as e:
            print(f"Failed to register meta debug shortcut: {e}")
    
    # ---------------- ROI helpers for plotting -----------------
    def _roi_active(self) -> bool:
        return (
            self._roi_min is not None and self._roi_max is not None and
            self._q_full_min is not None and self._q_full_max is not None and
            (abs(self._roi_min - self._q_full_min) > 1e-12 or abs(self._roi_max - self._q_full_max) > 1e-12)
        )

    def _get_roi_active_arrays(self):
        """Return (q_plot, I_plot) using ROI subset if active, else full arrays."""
        if self.q is None or self.I is None:
            return None, None
        # Helper: filter out non-finite pairs to avoid None/inf/NaN issues
        def _filter_finite(q_arr, I_arr):
            q_arr = np.asarray(q_arr)
            I_arr = np.asarray(I_arr)
            mask = np.isfinite(q_arr) & np.isfinite(I_arr)
            if not np.any(mask):
                return q_arr[:0], I_arr[:0]
            return q_arr[mask], I_arr[mask]

        if self._roi_active() and self.q_ROI is not None and self.I_ROI is not None and len(self.q_ROI) > 0:
            return _filter_finite(self.q_ROI, self.I_ROI)
        return _filter_finite(self.q, self.I)

    def _draw_roi_guides_if_active(self, ax):
        try:
            if not self._roi_active():
                return
            vmin, vmax = float(self._roi_min), float(self._roi_max)
            ax.axvline(vmin, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
            ax.axvline(vmax, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
        except Exception:
            pass

    # ---------------- ROI & Interpolation Controls -----------------
    def _setup_fitting_region_controls(self):
        """Wire up ROI slider/spinboxes and interpolation widgets.
        - Initialize defaults from user settings
        - Connect slider (live) and spinboxes (editingFinished)
        - Defer actual ROI initialization to first import/cut
        """
        # Slider
        if hasattr(self.ui, 'fitFittingRegionSlider'):
            try:
                self.ui.fitFittingRegionSlider.setDecimals(4)
            except Exception:
                pass
            try:
                self.ui.fitFittingRegionSlider.rangeChangedF.connect(self._on_roi_slider_changed)
            except Exception:
                if hasattr(self.ui.fitFittingRegionSlider, 'rangeChanged'):
                    self.ui.fitFittingRegionSlider.rangeChanged.connect(self._on_roi_slider_changed_int)

        # Min/Max spinboxes
        if hasattr(self.ui, 'fitFittingRegionMinValue'):
            try:
                self.ui.fitFittingRegionMinValue.setDecimals(4)
            except Exception:
                pass
            self.ui.fitFittingRegionMinValue.editingFinished.connect(self._on_roi_spin_finished)
        if hasattr(self.ui, 'fitFittingRegionMaxValue'):
            try:
                self.ui.fitFittingRegionMaxValue.setDecimals(4)
            except Exception:
                pass
            self.ui.fitFittingRegionMaxValue.editingFinished.connect(self._on_roi_spin_finished)

        # Points number
        if hasattr(self.ui, 'fitDataPointsNumValue'):
            try:
                self.ui.fitDataPointsNumValue.setRange(10, 5000)
                self.ui.fitDataPointsNumValue.setSingleStep(1)
                # 显示为当前稳定点数，来自 global_params/user_settings
                self.ui.fitDataPointsNumValue.setValue(int(max(10, self._points_num_current)))
            except Exception:
                pass
            # 使用 meta 注册，并根据统一模式连接信号
            try:
                def _dp_after_commit(info, value):
                    try:
                        self._points_num_current = int(value)
                    except Exception:
                        self._points_num_current = int(self._points_num_default)
                    # 记录变更前是否处于拟合模式
                    was_fitting = self._is_in_fitting_mode() if hasattr(self, '_is_in_fitting_mode') else False

                    # 根据数据源执行重切或重采样
                    if getattr(self, 'data_source', None) == 'cut':
                        self._perform_cut(points_override=int(self._points_num_current))
                    elif getattr(self, 'data_source', None) == '1d':
                        self._resample_1d(n_points=int(self._points_num_current))

                    # 如果之前处于拟合模式，则重新执行拟合（_perform_manual_fitting内会设置拟合显示模式并刷新）
                    if was_fitting:
                        self._perform_manual_fitting()

                mode = self._signal_mode_overrides.get('fitDataPointsNumValue', self._default_signal_mode)
                self.param_trigger_manager.register_parameter_widget(
                    widget=self.ui.fitDataPointsNumValue,
                    widget_id='meta_fit_points_num',
                    category='fit_controls',
                    immediate_handler=lambda v: None,
                    delayed_handler=None,
                    connect_signals=True,
                    meta={
                        'persist': 'global_params',
                        'key_path': ('fitting', 'fit.points_num'),
                        'debounce_ms': 0,
                        'epsilon_abs': 0,
                        'epsilon_rel': 0,
                        'after_commit': _dp_after_commit,
                        'trigger_fit': False,
                        'connect_mode': mode,
                    }
                )
                # 由 meta 管理器根据 connect_mode 自动连接
            except Exception:
                # 回退到旧逻辑
                self.ui.fitDataPointsNumValue.editingFinished.connect(self._on_points_num_finished)

        # Interpolation method
        if hasattr(self.ui, 'fitInterpolationMethodValue'):
            combo = self.ui.fitInterpolationMethodValue
            try:
                combo.clear(); combo.addItems(['Linear', 'Quadratic', 'Spline'])
                idx = combo.findText(self._interp_method_default)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            except Exception:
                pass
            combo.currentTextChanged.connect(self._on_interp_method_changed)

        # ROI min/max 使用 editingFinished 逻辑（建议）：已经上面绑定了 editingFinished
        # ROI slider 继续沿用 valueChanged（rangeChangedF/int）并用去抖（已在 manager 内部）

    def _initialize_roi_from_current_q(self, force_full: bool = False):
        """Initialize or refresh ROI bounds from current q/I arrays.

        - force_full=True resets ROI to full valid range regardless of previous ROI.
        - q_min/q_max are computed from pairs where both q and I are finite (exclude None/inf/NaN).
        """
        if self.q is None or self.I is None:
            return
        q_all = np.asarray(self.q)
        I_all = np.asarray(self.I)
        valid = np.isfinite(q_all) & np.isfinite(I_all)
        if not np.any(valid):
            return
        q_valid = q_all[valid]
        q_min, q_max = float(np.min(q_valid)), float(np.max(q_valid))
        # Update full range tracking to valid bounds
        prev_full = (self._q_full_min, self._q_full_max)
        self._q_full_min, self._q_full_max = q_min, q_max
        # Decide whether to reset ROI to full
        if force_full or self._roi_min is None or self._roi_max is None:
            self._roi_min, self._roi_max = q_min, q_max
        else:
            # If previous ROI is outside new bounds or full bounds changed notably, clamp or reset
            changed = (prev_full[0] is None or prev_full[1] is None or
                       abs(prev_full[0] - q_min) > 1e-12 or abs(prev_full[1] - q_max) > 1e-12)
            if changed:
                self._roi_min, self._roi_max = q_min, q_max
            else:
                # Clamp ROI into new bounds
                self._roi_min = max(q_min, min(self._roi_min, q_max))
                self._roi_max = max(self._roi_min, min(self._roi_max, q_max))
        # Update UI controls
        self._updating_roi_controls = True
        try:
            if hasattr(self.ui, 'fitFittingRegionSlider'):
                s = self.ui.fitFittingRegionSlider
                s.setRangeF(q_min, q_max)
                s.setMinValueF(self._roi_min)
                s.setMaxValueF(self._roi_max)
            if hasattr(self.ui, 'fitFittingRegionMinValue'):
                self.ui.fitFittingRegionMinValue.setRange(q_min, q_max)
                self.ui.fitFittingRegionMinValue.setValue(self._roi_min)
            if hasattr(self.ui, 'fitFittingRegionMaxValue'):
                self.ui.fitFittingRegionMaxValue.setRange(q_min, q_max)
                self.ui.fitFittingRegionMaxValue.setValue(self._roi_max)
        finally:
            self._updating_roi_controls = False

    def _on_roi_slider_changed_int(self, imin, imax):
        s = self.ui.fitFittingRegionSlider
        dec = 2
        try:
            dec = s.decimals()
        except Exception:
            pass
        scale = 10 ** dec
        self._on_roi_slider_changed(imin / scale, imax / scale)

    def _on_roi_slider_changed(self, vmin: float, vmax: float):
        if self._updating_roi_controls:
            return
        self._slider_is_source = True
        try:
            q = np.asarray(self.q) if self.q is not None else None
            if q is not None and q.size > 0:
                vmin = float(q[np.argmin(np.abs(q - vmin))])
                vmax = float(q[np.argmin(np.abs(q - vmax))])
            # Constraints
            lo = self._q_full_min if self._q_full_min is not None else vmin
            hi = self._q_full_max if self._q_full_max is not None else vmax
            vmin = max(lo, min(vmin, vmax))
            vmax = min(hi, max(vmax, vmin))
            self._roi_min, self._roi_max = vmin, vmax
            # Update spinboxes
            self._updating_roi_controls = True
            if hasattr(self.ui, 'fitFittingRegionMinValue'):
                self.ui.fitFittingRegionMinValue.setValue(vmin)
            if hasattr(self.ui, 'fitFittingRegionMaxValue'):
                self.ui.fitFittingRegionMaxValue.setValue(vmax)
        finally:
            self._updating_roi_controls = False
            self._slider_is_source = False
            # 拖动滑块时不要每次都全量刷新，使用去抖计时器合并重绘，显著降低卡顿
            self._schedule_roi_refresh()

    def _on_roi_spin_finished(self):
        if self._updating_roi_controls:
            return
        vmin = float(self.ui.fitFittingRegionMinValue.value()) if hasattr(self.ui, 'fitFittingRegionMinValue') else self._roi_min
        vmax = float(self.ui.fitFittingRegionMaxValue.value()) if hasattr(self.ui, 'fitFittingRegionMaxValue') else self._roi_max
        if self._q_full_min is not None:
            vmin = max(self._q_full_min, vmin)
        if self._q_full_max is not None:
            vmax = min(self._q_full_max, vmax)
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        self._roi_min, self._roi_max = vmin, vmax
        # Update slider
        self._updating_roi_controls = True
        try:
            if hasattr(self.ui, 'fitFittingRegionSlider'):
                s = self.ui.fitFittingRegionSlider
                if self._q_full_min is not None and self._q_full_max is not None:
                    s.setRangeF(self._q_full_min, self._q_full_max)
                s.setMinValueF(vmin)
                s.setMaxValueF(vmax)
        finally:
            self._updating_roi_controls = False
            # 输入框编辑完成：立即应用（较少频繁变化）
            self._apply_roi_to_data_and_refresh()

    def _schedule_roi_refresh(self):
        """延迟刷新ROI应用与图像更新，避免滑块拖动频繁触发重绘造成卡顿。"""
        try:
            from PyQt5.QtCore import QTimer
            if self._roi_update_timer is None:
                self._roi_update_timer = QTimer()
                self._roi_update_timer.setSingleShot(True)
                self._roi_update_timer.timeout.connect(self._apply_roi_to_data_and_refresh)
            # 重启计时器，若持续拖动则不断推迟刷新
            delay = int(getattr(self, '_roi_debounce_ms', 140))
            self._roi_update_timer.start(max(20, delay))
        except Exception:
            # 兜底：若计时器失败则直接应用
            self._apply_roi_to_data_and_refresh()

    def _apply_roi_to_data_and_refresh(self):
        if self.q is None or self.I is None:
            return
        q = np.asarray(self.q); I = np.asarray(self.I)
        # Always drop non-finite pairs before ROI masking
        valid = np.isfinite(q) & np.isfinite(I)
        q = q[valid]; I = I[valid]
        if self._roi_min is None or self._roi_max is None:
            self.q_ROI, self.I_ROI = q, I
        else:
            mask = (q >= self._roi_min) & (q <= self._roi_max)
            if not np.any(mask):
                self.q_ROI, self.I_ROI = q, I
            else:
                self.q_ROI = q[mask]
                self.I_ROI = I[mask]
        # Redraw displays
        try:
            # 仅当当前确实处于拟合模式时，才刷新拟合覆盖；避免在Normal模式下被动切换到Fitting
            current_mode = self.display_mode if hasattr(self, 'display_mode') else 'normal'
            if current_mode == 'fitting' or (hasattr(self, '_is_in_fitting_mode') and self._is_in_fitting_mode()):
                self._update_gui_fitting_display()
                current_mode = 'fitting'
            # 按当前模式刷新主视图与外部窗口
            self._update_GUI_image(current_mode)
            self._update_outside_window(current_mode)
        except Exception:
            pass

    # ---------------- ROI bounds adjustment for log-x -----------------
    def _get_current_fit_axes(self):
        """Try to get the current Matplotlib Axes used by the in-GUI 1D plot."""
        try:
            fig = getattr(self, '_current_fit_figure', None)
            if fig is None:
                return None
            axes = getattr(fig, 'axes', None)
            if not axes:
                return None
            return axes[0] if len(axes) > 0 else None
        except Exception:
            return None

    def _compute_display_xmin_for_log(self) -> float:
        """Compute a safe lower X bound to use when log-x is enabled.

        Priority:
        1) If current Axes exist, use its left xlim (must be > 0).
        2) Else, use min positive finite q from current data (full q preferred).
        3) Fallback to a tiny positive epsilon.
        """
        # 1) Try current axes xlim
        try:
            ax = self._get_current_fit_axes()
            if ax is not None:
                x0, _ = ax.get_xlim()
                if np.isfinite(x0) and x0 > 0:
                    return float(x0)
        except Exception:
            pass
        # 2) Use data-based min positive q
        try:
            q_all = None
            if self.q is not None:
                q_all = np.asarray(self.q)
            elif self.current_cut_data is not None and 'x_coords' in self.current_cut_data:
                q_all = np.asarray(self.current_cut_data['x_coords'])
            elif self.current_1d_data is not None and 'q' in self.current_1d_data:
                q_all = np.asarray(self.current_1d_data['q'])
            if q_all is not None and q_all.size > 0:
                pos = q_all[np.isfinite(q_all) & (q_all > 0)]
                if pos.size > 0:
                    return float(np.min(pos))
        except Exception:
            pass
        # 3) Fallback
        return 1e-12

    def _adjust_roi_bounds_for_log_x(self):
        """When log-x is enabled, ensure ROI slider/spin ranges start from the displayed x-axis min (>0).

        Also clamp current ROI values to the new bounds.
        """
        try:
            log_x = self._is_fit_log_x_enabled()
        except Exception:
            log_x = False

        if not hasattr(self.ui, 'fitFittingRegionSlider'):
            return

        try:
            s = self.ui.fitFittingRegionSlider
            # Determine full max from existing tracking or data
            q_max = None
            if self._q_full_max is not None:
                q_max = float(self._q_full_max)
            else:
                try:
                    q_all = np.asarray(self.q) if self.q is not None else None
                    if q_all is not None and q_all.size > 0:
                        q_max = float(np.nanmax(q_all[np.isfinite(q_all)]))
                except Exception:
                    q_max = None
            if q_max is None:
                return

            if log_x:
                xmin = self._compute_display_xmin_for_log()
                # Ensure xmin < q_max
                if not np.isfinite(xmin) or xmin <= 0 or xmin >= q_max:
                    xmin = min(max(1e-12, (q_max * 1e-6)), q_max * 0.5)  # conservative fallback
                new_min, new_max = float(xmin), float(q_max)
            else:
                # Restore to full valid data bounds if known
                q_min = float(self._q_full_min) if self._q_full_min is not None else None
                if q_min is None:
                    try:
                        q_all = np.asarray(self.q) if self.q is not None else None
                        if q_all is not None and q_all.size > 0:
                            q_min = float(np.nanmin(q_all[np.isfinite(q_all)]))
                    except Exception:
                        q_min = None
                if q_min is None:
                    return
                new_min, new_max = float(q_min), float(q_max)

            # Update control ranges and clamp current ROI values
            self._updating_roi_controls = True
            try:
                s.setRangeF(new_min, new_max)
                # Clamp current ROI values
                if self._roi_min is None or self._roi_max is None:
                    cur_min, cur_max = new_min, new_max
                else:
                    cur_min = max(new_min, min(float(self._roi_min), new_max))
                    cur_max = max(cur_min, min(float(self._roi_max), new_max))
                self._roi_min, self._roi_max = cur_min, cur_max
                s.setMinValueF(cur_min)
                s.setMaxValueF(cur_max)
                # Update spinbox ranges to match
                if hasattr(self.ui, 'fitFittingRegionMinValue'):
                    self.ui.fitFittingRegionMinValue.setRange(new_min, new_max)
                    self.ui.fitFittingRegionMinValue.setValue(cur_min)
                if hasattr(self.ui, 'fitFittingRegionMaxValue'):
                    self.ui.fitFittingRegionMaxValue.setRange(new_min, new_max)
                    self.ui.fitFittingRegionMaxValue.setValue(cur_max)
            finally:
                self._updating_roi_controls = False
        except Exception:
            pass

    # ---------------- Interpolation -----------------
    def _on_points_num_finished(self):
        n = None
        try:
            if hasattr(self.ui, 'fitDataPointsNumValue'):
                n = int(self.ui.fitDataPointsNumValue.value())
        except Exception:
            n = None
        if n is None:
            return
        if n < 10:
            n = 10
        # Keep a stable in-controller cache for repeated cuts
        try:
            self._points_num_current = int(n)
        except Exception:
            self._points_num_current = int(self._points_num_default)
        # Persist
        try:
            from core.user_settings import user_settings
            user_settings.set('fit.points_num', int(n)); user_settings.save_settings()
        except Exception:
            pass
        # 记录变更前是否处于拟合模式
        was_fitting = self._is_in_fitting_mode() if hasattr(self, '_is_in_fitting_mode') else False

        # Apply 重新切割/重采样
        if getattr(self, 'data_source', None) == 'cut':
            self._perform_cut(points_override=n)
        elif getattr(self, 'data_source', None) == '1d':
            self._resample_1d(n_points=n)

        # 如果之前在拟合模式，则重新进行拟合（_perform_manual_fitting会设置拟合显示并刷新）
        if was_fitting:
            self._perform_manual_fitting()

    def _on_interp_method_changed(self, method: str):
        meth = method or 'Linear'
        try:
            from core.user_settings import user_settings
            user_settings.set('fit.interp_method', meth); user_settings.save_settings()
        except Exception:
            pass
        if self.data_source == '1d' and self.q is not None:
            self._resample_1d(n_points=len(self.q), method=meth, keep_same_count=True)
        elif self.data_source == 'cut':
            self._perform_cut()

    def _resample_1d(self, n_points: int, method: str = None, keep_same_count: bool = False):
        if self.current_1d_data is None or self.q is None or self.I is None:
            return
        q0 = np.asarray(self.current_1d_data.get('q', self.q))
        I0 = np.asarray(self.current_1d_data.get('I', self.I))
        if q0.size < 2:
            return
        method = method or (self.ui.fitInterpolationMethodValue.currentText() if hasattr(self.ui, 'fitInterpolationMethodValue') else 'Linear')
        if keep_same_count:
            n_points = len(self.q)
        # Fallback to stable current points if not valid
        try:
            n_points = int(max(10, n_points))
        except Exception:
            n_points = int(max(10, getattr(self, '_points_num_current', self._points_num_default)))
        q_new = np.linspace(q0.min(), q0.max(), int(max(10, n_points)))
        I_new = self._interpolate_series(q0, I0, q_new, method)
        self.q, self.I = q_new, I_new
        if self._q_full_min is None or self._q_full_max is None:
            self._initialize_roi_from_current_q()
        self._apply_roi_to_data_and_refresh()

    def _interpolate_series(self, x, y, x_new, method: str):
        m = (method or 'Linear').lower()
        if m == 'linear':
            return np.interp(x_new, x, y)
        if m == 'quadratic':
            try:
                from scipy.interpolate import interp1d
                f = interp1d(x, y, kind='quadratic', bounds_error=False, fill_value='extrapolate')
                return f(x_new)
            except Exception:
                coeff = np.polyfit(x, y, deg=min(2, max(1, len(x)-1)))
                return np.polyval(coeff, x_new)
        if m == 'spline':
            try:
                from scipy.interpolate import CubicSpline
                return CubicSpline(x, y, extrapolate=True)(x_new)
            except Exception:
                return np.interp(x_new, x, y)
        return np.interp(x_new, x, y)
        
    def _setup_connections(self):
        """设置信号连接"""
        # 连接GISAXS导入相关按钮
        if hasattr(self.ui, 'gisaxsInputImportButton'):
            self.ui.gisaxsInputImportButton.clicked.connect(self._import_gisaxs_file)
            
        # 连接导入文件输入框的回车事件
        if hasattr(self.ui, 'gisaxsInputImportButtonValue'):
            self.ui.gisaxsInputImportButtonValue.returnPressed.connect(self._on_import_value_changed)
            
        # 连接Stack值变化（改为回车触发）
        if hasattr(self.ui, 'gisaxsInputStackValue'):
            self.ui.gisaxsInputStackValue.returnPressed.connect(self._on_stack_value_changed)
            
        # 连接Show按钮
        if hasattr(self.ui, 'gisaxsInputShowButton'):
            self.ui.gisaxsInputShowButton.clicked.connect(self._show_image)
            
        # 连接AutoShow复选框
        if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox'):
            self.ui.gisaxsInputAutoShowCheckBox.toggled.connect(self._on_auto_show_changed)
        
        # 连接载入模式组合框
        if hasattr(self.ui, 'gisaxsInputModelCombox'):
            try:
                self.ui.gisaxsInputModelCombox.currentTextChanged.connect(self._on_load_mode_changed)
            except Exception:
                pass
            
        # 连接Log复选框
        if hasattr(self.ui, 'gisaxsInputIntLogCheckBox'):
            self.ui.gisaxsInputIntLogCheckBox.toggled.connect(self._on_log_changed)
            
        # 连接AutoScale复选框
        if hasattr(self.ui, 'gisaxsInputAutoScaleCheckBox'):
            self.ui.gisaxsInputAutoScaleCheckBox.toggled.connect(self._on_auto_scale_changed)
            
        # 连接Q模式切换按钮
        if hasattr(self.ui, 'gisaxsInputDisplayModeQ'):
            self.ui.gisaxsInputDisplayModeQ.toggled.connect(self._on_q_mode_changed)
        if hasattr(self.ui, 'gisaxsInputDisplayModePixel'):
            self.ui.gisaxsInputDisplayModePixel.toggled.connect(self._on_q_mode_changed)
            
        # Vmin/Vmax值变化现在通过触发管理器处理
        # 见 _connect_cutline_parameter_signals() 方法
            
        # 连接AutoFinding按钮
        if hasattr(self.ui, 'gisaxsInputCenterAutoFindingButton'):
            self.ui.gisaxsInputCenterAutoFindingButton.clicked.connect(self._auto_find_center)
            
        # 连接Cut按钮
        if hasattr(self.ui, 'gisaxsInputCutButton'):
            self.ui.gisaxsInputCutButton.clicked.connect(self._perform_cut)
            
        # 连接Detector Parameters按钮
        if hasattr(self.ui, 'gisaxsInputDetectorParaButton'):
            self.ui.gisaxsInputDetectorParaButton.clicked.connect(self._show_detector_parameters)
            
        # 设置GraphicsView双击事件
        if hasattr(self.ui, 'gisaxsInputGraphicsView'):
            self.ui.gisaxsInputGraphicsView.mouseDoubleClickEvent = self._on_graphics_view_double_click
            
        # 设置fitGraphicsView双击事件
        if hasattr(self.ui, 'fitGraphicsView'):
            self.ui.fitGraphicsView.mouseDoubleClickEvent = self._on_fit_graphics_view_double_click
            
        # 连接拟合相关按钮（如果UI中存在的话）
        if hasattr(self.ui, 'fitStartButton'):
            self.ui.fitStartButton.clicked.connect(self._start_fitting)
            
        # 连接Clear Fitting按钮
        if hasattr(self.ui, 'FittingClearFittingButton_2'):
            self.ui.FittingClearFittingButton_2.clicked.connect(self._clear_fitting_data)
            
        # 连接拟合图的log相关复选框
        if hasattr(self.ui, 'fitLogXCheckBox'):
            self.ui.fitLogXCheckBox.toggled.connect(self._on_fit_log_changed)
        if hasattr(self.ui, 'fitLogYCheckBox'):
            self.ui.fitLogYCheckBox.toggled.connect(self._on_fit_log_changed)
        
        # 连接组件叠加显示复选框（若存在则即刻刷新拟合图）
        for _name in ['fitBGShowCheckBox', 'fitResShowCheckBox']:
            if hasattr(self.ui, _name):
                try:
                    getattr(self.ui, _name).toggled.connect(self._on_component_checkbox_changed)
                except Exception:
                    pass
            
        # 连接Normalize复选框
        if hasattr(self.ui, 'OthersNormalizeCheckBox'):
            self.ui.OthersNormalizeCheckBox.toggled.connect(self._on_normalize_changed)
        if hasattr(self.ui, 'fitNormCheckBox'):
            self.ui.fitNormCheckBox.toggled.connect(self._on_normalize_changed)
            
        if hasattr(self.ui, 'fitResetButton'):
            self.ui.fitResetButton.clicked.connect(self._reset_fitting)
            
        # 连接fitImport1dFileButton按钮
        if hasattr(self.ui, 'fitImport1dFileButton'):
            self.ui.fitImport1dFileButton.clicked.connect(self._import_1d_file)
            
        # 连接fitImport1dFileValue文本框的回车事件
        if hasattr(self.ui, 'fitImport1dFileValue'):
            self.ui.fitImport1dFileValue.returnPressed.connect(self._on_1d_file_value_changed)
            
        # 连接FittingExportButton按钮
        if hasattr(self.ui, 'FittingExportButton'):
            self.ui.FittingExportButton.clicked.connect(self._export_fitting_data)
            
        # 连接FittingManualFittingButton按钮
        if hasattr(self.ui, 'FittingManualFittingButton'):
            self.ui.FittingManualFittingButton.clicked.connect(self._perform_manual_fitting)
            
        # 连接FittingAutoKButton按钮
        if hasattr(self.ui, 'FittingAutoKButton'):
            self.ui.FittingAutoKButton.clicked.connect(self._on_auto_k_button_clicked)
            
        # 连接拟合显示选项复选框
        if hasattr(self.ui, 'fitCurrentDataCheckBox'):
            self.ui.fitCurrentDataCheckBox.toggled.connect(self._on_current_data_checkbox_changed)
            
        # 连接Cut Line和Center参数控件的信号（统一模式）
        self._connect_cutline_parameter_signals(
            mode=self._default_signal_mode,
            overrides=self._signal_mode_overrides,
        )
            
        # 连接参数输入框的信号（如果存在的话）
        self._connect_parameter_widgets()
        
        # 连接FittingTextBrowser和状态信息
        self._setup_fitting_text_browser()
        
    def _connect_cutline_parameter_signals(self, mode: str = 'changed', overrides: dict = None):
        """连接Cut Line/Center/Vmin/Vmax：使用 meta 去抖 & 持久化到 global_params
        参数:
            mode: 'changed' 或 'finished'。
                  - 'changed': 实时 valueChanged/textChanged/currentTextChanged 触发 meta 去抖更新
                  - 'finished': editingFinished/returnPressed 等提交时再持久化
            overrides: 可选dict，键为控件对象名，值为该控件的模式('changed'|'finished')，用于覆盖全局 mode。
        默认保持向后兼容：mode='changed'。
        """
        from functools import partial
        mapping = [
            ('gisaxsInputCenterVerticalValue', 'center_vertical'),
            ('gisaxsInputCenterParallelValue', 'center_parallel'),
            ('gisaxsInputCutLineVerticalValue', 'cutline_vertical'),
            ('gisaxsInputCutLineParallelValue', 'cutline_parallel'),
            ('gisaxsInputVminValue', 'vmin'),
            ('gisaxsInputVmaxValue', 'vmax'),
        ]
        overrides = overrides or {}
        for widget_name, param_key in mapping:
            if not hasattr(self.ui, widget_name):
                continue
            w = getattr(self.ui, widget_name)
            def _after_commit(info, value, p=param_key):
                try:
                    self._on_parameter_display_changed()
                    self._add_fitting_message(f"Meta commit GISAXS {p} = {value}", "INFO")
                except Exception:
                    pass
            widget_mode = overrides.get(widget_name, mode)
            meta = {
                'persist': 'global_params',
                'key_path': ('fitting', f'gisaxs_input.{param_key}'),
                'trigger_fit': False,
                'debounce_ms': self._param_debounce_ms,
                'epsilon_abs': self._param_abs_eps,
                'epsilon_rel': self._param_rel_eps,
                'after_commit': _after_commit,
                'connect_mode': widget_mode,
            }
            self.param_trigger_manager.register_parameter_widget(
                widget=w,
                widget_id=f'meta_gisaxs_{param_key}',
                category='gisaxs_input',
                immediate_handler=lambda v: None,
                delayed_handler=None,
                connect_signals=True,
                meta=meta
            )
            # 由 meta 管理器根据 connect_mode 自动连接，无需手动连接

    def _connect_parameter_widgets(self):
        """连接参数输入控件的信号"""
        # 这里可以根据UI文件中的具体控件来连接
        # 例如：裁剪区域参数、拟合参数等
        
        # 重新设置粒子形状连接（确保在所有UI连接建立后）
        self._setup_particle_connections()
        
    def _initialize_ui(self):
        """初始化界面状态"""
        # 清空导入文件输入框
        if hasattr(self.ui, 'gisaxsInputImportButtonValue'):
            self.ui.gisaxsInputImportButtonValue.clear()
            
        # 设置Stack默认值
        if hasattr(self.ui, 'gisaxsInputStackValue'):
            self.ui.gisaxsInputStackValue.setText("1")
            
        # 清空stack显示标签
        if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
            self.ui.gisaxsInputStackDisplayLabel.setText("")
        
        # 初始化载入模式（从会话恢复）
        if hasattr(self.ui, 'gisaxsInputModelCombox'):
            try:
                from core.global_params import GlobalParameterManager
                gp = GlobalParameterManager()
                last_mode = gp.get_parameter('fitting', 'gisaxs_input.load_mode', 'Single')
                idx = self.ui.gisaxsInputModelCombox.findText(last_mode)
                self.ui.gisaxsInputModelCombox.setCurrentIndex(idx if idx >= 0 else 0)
                self.load_mode = self.ui.gisaxsInputModelCombox.currentText()
            except Exception:
                self.load_mode = 'Single'
            # 关键修正：在UI完全初始化前，若为 In-situ，立即隐藏数值型 Stack 输入框，避免与专用 LineEdit 共存闪现
            try:
                if self.load_mode == 'In-situ' and hasattr(self.ui, 'gisaxsInputStackValue'):
                    self.ui.gisaxsInputStackValue.setVisible(False)
            except Exception:
                pass
            # 根据模式更新Stack输入框可见性
            self._update_stack_controls_visibility()
            
        # 设置Log复选框默认选中
        if hasattr(self.ui, 'gisaxsInputIntLogCheckBox'):
            self.ui.gisaxsInputIntLogCheckBox.setChecked(True)
            
        # 设置AutoScale复选框默认选中
        if hasattr(self.ui, 'gisaxsInputAutoScaleCheckBox'):
            self.ui.gisaxsInputAutoScaleCheckBox.setChecked(True)
            
        # 初始化拟合相关复选框状态（阻塞信号避免触发方法调用）
        self._initialize_fit_checkboxes()
            
        # 初始化Vmin/Vmax值为0（稍后会在显示图像时自动计算）
        if hasattr(self.ui, 'gisaxsInputVminValue'):
            self.ui.gisaxsInputVminValue.setValue(0.0)
            # 设置智能精度：支持小数但默认显示简洁
            self.ui.gisaxsInputVminValue.setDecimals(6)
            self.ui.gisaxsInputVminValue.setRange(-99999.999999, 99999.999999)
            self.ui.gisaxsInputVminValue.setSingleStep(0.1)
            self.ui.gisaxsInputVminValue.setKeyboardTracking(True)
            self._setup_smart_display(self.ui.gisaxsInputVminValue)
            
        if hasattr(self.ui, 'gisaxsInputVmaxValue'):
            self.ui.gisaxsInputVmaxValue.setValue(0.0)
            # 设置智能精度：支持小数但默认显示简洁
            self.ui.gisaxsInputVmaxValue.setDecimals(6)
            self.ui.gisaxsInputVmaxValue.setRange(-99999.999999, 99999.999999)
            self.ui.gisaxsInputVmaxValue.setSingleStep(0.1)
            self.ui.gisaxsInputVmaxValue.setKeyboardTracking(True)
            self._setup_smart_display(self.ui.gisaxsInputVmaxValue)
        
        # 设置Cut Line Center控件的范围（支持实数域，包括负数）
        if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
            self.ui.gisaxsInputCenterVerticalValue.setRange(-99999.99, 99999.99)
            self.ui.gisaxsInputCenterVerticalValue.setDecimals(2)
            self.ui.gisaxsInputCenterVerticalValue.setValue(0.0)
            self.ui.gisaxsInputCenterVerticalValue.setKeyboardTracking(True)
            
        if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
            self.ui.gisaxsInputCenterParallelValue.setRange(-99999.99, 99999.99)
            self.ui.gisaxsInputCenterParallelValue.setDecimals(2)
            self.ui.gisaxsInputCenterParallelValue.setValue(0.0)
            self.ui.gisaxsInputCenterParallelValue.setKeyboardTracking(True)
            
        # 设置Cut Line Vertical/Parallel控件的范围（支持实数域，包括负数）
        if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
            self.ui.gisaxsInputCutLineVerticalValue.setRange(-99999.99, 99999.99)
            self.ui.gisaxsInputCutLineVerticalValue.setDecimals(2)
            self.ui.gisaxsInputCutLineVerticalValue.setValue(10.0)
            self.ui.gisaxsInputCutLineVerticalValue.setKeyboardTracking(True)
            
        if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
            self.ui.gisaxsInputCutLineParallelValue.setRange(-99999.99, 99999.99)
            self.ui.gisaxsInputCutLineParallelValue.setDecimals(2)
            self.ui.gisaxsInputCutLineParallelValue.setValue(10.0)
            self.ui.gisaxsInputCutLineParallelValue.setKeyboardTracking(True)
        
        # 设置Cut Line参数的步长（根据模式动态调整）
        self._update_cutline_step_sizes()
        
        # 强制更新一次步长以确保正确性
        if hasattr(self, '_on_q_mode_changed'):
            # 延迟执行以确保UI完全初始化
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(100, self._update_cutline_step_sizes)
            
        # 设置默认参数
        self._set_default_parameters()
        
        # 初始化Cut Line标签的单位
        self._update_cutline_labels_units()
        
        # 初始化Q模式状态（避免第一次调用时误触发转换）
        self._initialize_q_mode_state()
        
        # 检查依赖库
        self._check_dependencies()

        # 若 In-situ 且自动显示，启动轮询
        try:
            if getattr(self, 'load_mode', 'Single') == 'In-situ' and self._is_auto_show_enabled():
                self._start_insitu_timer()
            # 初始化结束后直接执行一次严格的可见性修正（简化方案，避免双输入框共存）
            self._enforce_insitu_visibility_once()
        except Exception:
            pass
    
    def _initialize_fit_checkboxes(self):
        """初始化拟合相关复选框状态（阻塞信号避免触发方法调用）"""
        try:
            # 初始化fitCurrentDataCheckBox（默认不勾选）
            if hasattr(self.ui, 'fitCurrentDataCheckBox'):
                self.ui.fitCurrentDataCheckBox.blockSignals(True)
                self.ui.fitCurrentDataCheckBox.setChecked(False)
                self.ui.fitCurrentDataCheckBox.blockSignals(False)
            
            # 初始化fitLogXCheckBox（默认不勾选）
            if hasattr(self.ui, 'fitLogXCheckBox'):
                self.ui.fitLogXCheckBox.blockSignals(True)
                self.ui.fitLogXCheckBox.setChecked(False)
                self.ui.fitLogXCheckBox.blockSignals(False)
            
            # 初始化fitLogYCheckBox（默认不勾选）
            if hasattr(self.ui, 'fitLogYCheckBox'):
                self.ui.fitLogYCheckBox.blockSignals(True)
                self.ui.fitLogYCheckBox.setChecked(False)
                self.ui.fitLogYCheckBox.blockSignals(False)
            
            # 初始化fitNormCheckBox（默认不勾选）
            if hasattr(self.ui, 'fitNormCheckBox'):
                self.ui.fitNormCheckBox.blockSignals(True)
                self.ui.fitNormCheckBox.setChecked(False)
                self.ui.fitNormCheckBox.blockSignals(False)
                
        except Exception as e:
            pass
    
    def _restore_fit_checkboxes(self, session_data):
        """恢复拟合复选框状态（阻塞信号避免触发方法调用）"""
        try:
            # 恢复fitCurrentDataCheckBox
            if hasattr(self.ui, 'fitCurrentDataCheckBox'):
                self.ui.fitCurrentDataCheckBox.blockSignals(True)
                self.ui.fitCurrentDataCheckBox.setChecked(session_data.get('fit_current_data', False))
                self.ui.fitCurrentDataCheckBox.blockSignals(False)
            
            # 恢复fitLogXCheckBox
            if hasattr(self.ui, 'fitLogXCheckBox'):
                self.ui.fitLogXCheckBox.blockSignals(True)
                self.ui.fitLogXCheckBox.setChecked(session_data.get('fit_log_x', False))
                self.ui.fitLogXCheckBox.blockSignals(False)
            
            # 恢复fitLogYCheckBox
            if hasattr(self.ui, 'fitLogYCheckBox'):
                self.ui.fitLogYCheckBox.blockSignals(True)
                self.ui.fitLogYCheckBox.setChecked(session_data.get('fit_log_y', False))
                self.ui.fitLogYCheckBox.blockSignals(False)
            
            # 恢复fitNormCheckBox
            if hasattr(self.ui, 'fitNormCheckBox'):
                self.ui.fitNormCheckBox.blockSignals(True)
                self.ui.fitNormCheckBox.setChecked(session_data.get('fit_norm', False))
                self.ui.fitNormCheckBox.blockSignals(False)
                
        except Exception as e:
            pass
    
    def _initialize_q_mode_state(self):
        """初始化Q模式状态，避免第一次调用时误触发转换"""
        try:
            # 获取当前Q轴显示状态并设置为初始状态
            current_q_mode = self._should_show_q_axis()
            self._last_q_mode = current_q_mode
        except Exception as e:
            # 如果获取状态失败，默认设置为像素模式
            self._last_q_mode = False
        
    def _setup_smart_display(self, spinbox):
        """设置SpinBox的智能显示格式"""
        try:
            # 连接值变化信号到格式更新方法
            spinbox.valueChanged.connect(lambda value: self._update_spinbox_format(spinbox, value))
            spinbox.editingFinished.connect(lambda: self._update_spinbox_format(spinbox, spinbox.value()))
            self._update_spinbox_format(spinbox, spinbox.value())
        except Exception:
            spinbox.setDecimals(2)
        
    def _update_spinbox_format(self, spinbox, value):
        """根据当前模式和值更新SpinBox的显示格式"""
        try:
            is_log_mode = False
            if hasattr(self.ui, 'gisaxsInputIntLogCheckBox'):
                is_log_mode = self.ui.gisaxsInputIntLogCheckBox.isChecked()
            
            if is_log_mode:
                spinbox.setDecimals(2)
            else:
                if abs(value - round(value)) < 1e-9:
                    spinbox.setDecimals(0)
                else:
                    value_str = f"{value:.6f}".rstrip('0').rstrip('.')
                    if '.' in value_str:
                        decimal_places = len(value_str.split('.')[1])
                        decimal_places = min(decimal_places, 6)
                        decimal_places = max(decimal_places, 1)
                        spinbox.setDecimals(decimal_places)
                    else:
                        spinbox.setDecimals(0)
        except Exception:
            try:
                if hasattr(self.ui, 'gisaxsInputIntLogCheckBox') and self.ui.gisaxsInputIntLogCheckBox.isChecked():
                    spinbox.setDecimals(2)
                else:
                    spinbox.setDecimals(0)
            except:
                spinbox.setDecimals(2)
    
    def _refresh_vmin_vmax_display(self):
        """刷新Vmin/Vmax控件的显示格式"""
        try:
            if hasattr(self.ui, 'gisaxsInputVminValue'):
                self._update_spinbox_format(self.ui.gisaxsInputVminValue, self.ui.gisaxsInputVminValue.value())
            if hasattr(self.ui, 'gisaxsInputVmaxValue'):
                self._update_spinbox_format(self.ui.gisaxsInputVmaxValue, self.ui.gisaxsInputVmaxValue.value())
        except Exception:
            pass

    def _check_dependencies(self):
        """检查所需的依赖库"""
        if not is_fabio_available():
            self.status_updated.emit("Warning: fabio library not available. CBF processing will be disabled.")
        if not is_matplotlib_available():
            self.status_updated.emit("Warning: matplotlib not available. Image display will be disabled.")
    
    def _is_q_space_mode(self):
        """检查当前是否为Q-space模式"""
        try:
            # 使用和_should_show_q_axis()相同的逻辑
            return self._should_show_q_axis()
        except Exception:
            return False
    
    def _delayed_cut_update(self):
        """延迟执行Cut更新，避免频繁操作"""
        try:
            # 仅重新执行Cut并更新图像显示
            if hasattr(self, '_cut_data') and self._cut_data is not None:
                # 重新执行Cut操作以更新图像
                self._execute_cut()
        except Exception as e:
            pass
    
    def _on_parameter_display_changed(self):
        """参数显示立即更新（不触发图像重绘）"""
        try:
            # 初始化期间不更新
            if getattr(self, '_initializing', False):
                return
            # 仅更新与堆栈/模式相关的显示标签
            if hasattr(self, '_update_stack_display'):
                self._update_stack_display()
        except Exception:
            pass
                
            # 立即更新选择框显示，但不执行Cut操作
            center_x = 0
            center_y = 0
            width = 0
            height = 0
            
            # 获取当前参数值
            if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                center_x = self.ui.gisaxsInputCenterParallelValue.value()
            if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                center_y = self.ui.gisaxsInputCenterVerticalValue.value()
            if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                width = self.ui.gisaxsInputCutLineParallelValue.value()
            if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                height = self.ui.gisaxsInputCutLineVerticalValue.value()
            
            # 仅更新选择框显示（快速操作）
            if width > 0 and height > 0:
                selection_info = self._create_selection_from_parameters(center_x, center_y, width, height)
                self._update_parameter_selection_display(selection_info)
                
            # 启动延迟更新定时器用于图像更新
            self._trigger_delayed_cut_update()
            
        except Exception as e:
            pass
    
    def _trigger_delayed_cut_update(self):
        """触发延迟的Cut更新"""
        try:
            if not hasattr(self, '_cut_update_timer'):
                from PyQt5.QtCore import QTimer
                self._cut_update_timer = QTimer()
                self._cut_update_timer.setSingleShot(True)
                self._cut_update_timer.timeout.connect(self._delayed_cut_image_update)
            
            # 重置定时器（防抖动）
            self._cut_update_timer.stop()
            self._cut_update_timer.start(300)  # 300ms延迟更新图像
            
        except Exception as e:
            pass
    
    def _delayed_cut_image_update(self):
        """延迟执行Cut图像更新"""
        try:
            # 检查是否已经有Cut结果数据，如果有则重新执行Cut操作
            if (self.current_cut_data is not None and 
                hasattr(self, 'current_stack_data') and self.current_stack_data is not None):
                
                # 获取当前参数
                center_x = 0
                center_y = 0
                width = 0
                height = 0
                
                if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                    center_x = self.ui.gisaxsInputCenterParallelValue.value()
                if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                    center_y = self.ui.gisaxsInputCenterVerticalValue.value()
                if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                    width = self.ui.gisaxsInputCutLineParallelValue.value()
                if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                    height = self.ui.gisaxsInputCutLineVerticalValue.value()
                
                # 重新执行Cut操作
                self._perform_cut()
                self.status_updated.emit(f"Auto-updated cut with new parameters: Center({center_x}, {center_y}), Size({width}×{height})")
                
        except Exception as e:
            pass
    
    def _on_cutline_parameters_immediate_update(self):
        """编辑完成时立即更新（用于回车键或失去焦点）"""
        try:
            # 停止延迟定时器，立即执行更新
            if hasattr(self, '_cut_update_timer'):
                self._cut_update_timer.stop()
            
            # 立即执行图像更新
            self._delayed_cut_image_update()
            
        except Exception as e:
            pass
    
    def _update_cutline_step_sizes(self):
        """根据当前模式更新Cut Line参数的步长"""
        try:
            # 检查当前模式
            is_q_mode = self._is_q_space_mode()
            
            # 根据模式设置步长
            if is_q_mode:
                # Q-space模式：使用0.01步长
                step_size = 0.01
            else:
                # Pixel模式：使用1.0步长
                step_size = 1.0
            
            # 应用步长到所有Cut Line参数控件
            cutline_controls = [
                'gisaxsInputCenterVerticalValue',
                'gisaxsInputCenterParallelValue', 
                'gisaxsInputCutLineVerticalValue',
                'gisaxsInputCutLineParallelValue'
            ]
            
            for control_name in cutline_controls:
                if hasattr(self.ui, control_name):
                    control = getattr(self.ui, control_name)
                    control.setSingleStep(step_size)
            
            self.status_updated.emit(f"Cut Line step size updated to {step_size} ({'Q-space' if is_q_mode else 'Pixel'} mode)")
            
        except Exception as e:
            self.status_updated.emit(f"Error updating cut line step sizes: {str(e)}")
    
    def _update_cutline_labels_units(self):
        """根据当前显示模式更新Cut Line标签的单位"""
        try:
            show_q_axis = self._should_show_q_axis()
            
            if show_q_axis:
                # Q坐标模式：添加 (nm⁻¹) 单位
                unit_suffix = " (nm⁻¹)"
            else:
                # 像素坐标模式：添加 (pixel) 单位
                unit_suffix = " (pixel)"
            
            # 更新Center标签
            if hasattr(self.ui, 'gisaxsInputCenterVerticalLabel'):
                self.ui.gisaxsInputCenterVerticalLabel.setText(f"Vertical.{unit_suffix}")
            
            if hasattr(self.ui, 'gisaxsInputCenterParallelLabel'):
                self.ui.gisaxsInputCenterParallelLabel.setText(f"Parallel.{unit_suffix}")
            
            # 更新Cut Line标签
            if hasattr(self.ui, 'gisaxsInputCutLineVerticalLabel'):
                self.ui.gisaxsInputCutLineVerticalLabel.setText(f"Vertical.{unit_suffix}")
            
            if hasattr(self.ui, 'gisaxsInputCutLineParallelLabel'):
                self.ui.gisaxsInputCutLineParallelLabel.setText(f"Parallel.{unit_suffix}")
                
        except Exception as e:
            pass
    
    def _should_show_q_axis(self):
        """检查是否应该显示Q轴"""
        try:
            from core.global_params import GlobalParameterManager
            global_params = GlobalParameterManager()
            return global_params.get_parameter('fitting', 'detector.show_q_axis', False)
        except Exception:
            return False
    
    def _get_cached_q_meshgrids(self):
        """获取缓存的Q空间网格 - FittingController版本"""
        try:
            # 如果独立窗口存在且有缓存的Q网格，使用它
            if (self.independent_window is not None and 
                hasattr(self.independent_window, '_qy_mesh') and 
                self.independent_window._qy_mesh is not None):
                return self.independent_window._qy_mesh, self.independent_window._qz_mesh
            
            # 否则直接计算Q网格
            if hasattr(self, 'current_stack_data') and self.current_stack_data is not None:
                from core.global_params import GlobalParameterManager
                from utils.q_space_calculator import create_detector_from_image_and_params
                
                global_params = GlobalParameterManager()
                height, width = self.current_stack_data.shape
                
                pixel_size_x = global_params.get_parameter('fitting', 'detector.pixel_size_x', 172.0)
                pixel_size_y = global_params.get_parameter('fitting', 'detector.pixel_size_y', 172.0)
                beam_center_x = global_params.get_parameter('fitting', 'detector.beam_center_x', width / 2.0)
                beam_center_y = global_params.get_parameter('fitting', 'detector.beam_center_y', height / 2.0)
                distance = global_params.get_parameter('fitting', 'detector.distance', 2565.0)
                theta_in_deg = global_params.get_parameter('beam', 'grazing_angle', 0.4)
                wavelength = global_params.get_parameter('beam', 'wavelength', 0.1045)
                
                detector = create_detector_from_image_and_params(
                    image_shape=(height, width),
                    pixel_size_x=pixel_size_x,
                    pixel_size_y=pixel_size_y,
                    beam_center_x=beam_center_x,
                    beam_center_y=beam_center_y,
                    distance=distance,
                    theta_in_deg=theta_in_deg,
                    wavelength=wavelength,
                    crop_params=None
                )
                
                return detector.get_qy_qz_meshgrids()
            
            return None, None
            
        except Exception as e:
            return None, None
        
    def _set_default_parameters(self):
        """设置默认参数"""
        self.current_parameters = {
            'imported_gisaxs_file': '',  # 导入的GISAXS文件
            'stack_count': 1,  # 叠加数量
            'cut_region': {'x': 0, 'y': 0, 'width': 100, 'height': 100},
            'fitting_params': {}
        }
    
    def get_parameters(self):
        """获取当前参数"""
        return self.current_parameters.copy()
        
    def set_parameters(self, parameters):
        """设置参数"""
        self.current_parameters.update(parameters)
        self.parameters_changed.emit(self.current_parameters)
    
    def get_imported_file(self):
        """获取导入的GISAXS文件路径"""
        return self.current_parameters.get('imported_gisaxs_file', '')
    
    # ========== 图像导入和处理方法 ==========
    
    def _import_gisaxs_file(self):
        """导入GISAXS文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "导入GISAXS文件",
            "",
            "GISAXS Files (*.tif *.tiff *.dat *.txt *.h5 *.hdf5 *.jpg *.png *.bmp *.cbf);;TIF Files (*.tif *.tiff);;Data Files (*.dat *.txt);;HDF5 Files (*.h5 *.hdf5 *cbf);;Image Files (*.jpg *.png *.bmp);;All Files (*)"
        )
        
        if file_path:
            # 更新参数
            self.current_parameters['imported_gisaxs_file'] = file_path
            
            # 更新UI中的文本框显示
            if hasattr(self.ui, 'gisaxsInputImportButtonValue'):
                file_name = os.path.basename(file_path)
                self.ui.gisaxsInputImportButtonValue.setText(file_name)
                
            # 发送状态更新信号
            self.status_updated.emit(f"Imported GISAXS file: {os.path.basename(file_path)}")
            self.parameters_changed.emit(self.current_parameters)
            
            # 触发主控制器保存会话
            if hasattr(self.parent, 'save_current_session'):
                self.parent.save_current_session()
            
            # 验证文件
            self._validate_imported_file(file_path)
            
            # 更新显示信息
            self._update_stack_display()
            
            # 如果AutoShow被选中，则自动显示
            if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked():
                self._show_image()
    
    def _validate_imported_file(self, file_path):
        """验证导入的GISAXS文件"""
        try:
            if not os.path.exists(file_path):
                QMessageBox.warning(self.main_window, "File Error", f"File does not exist: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                QMessageBox.warning(self.main_window, "File Error", "File is empty")
                return False
            
            file_ext = os.path.splitext(file_path)[1].lower()
            supported_extensions = ['.tif', '.tiff', '.dat', '.txt', '.h5', '.hdf5', '.jpg', '.png', '.bmp', '.cbf']
            
            if file_ext not in supported_extensions:
                reply = QMessageBox.question(
                    self.main_window,
                    "File Format Warning",
                    f"The file extension '{file_ext}' may not be supported.\nSupported formats: {', '.join(supported_extensions)}\n\nContinue import?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return False
            
            self.status_updated.emit(f"File validation passed - {os.path.basename(file_path)}")
            return True
            
        except Exception as e:
            QMessageBox.critical(self.main_window, "File Validation Error", f"Error validating file:\n{str(e)}")
            return False
    
    def _on_import_value_changed(self):
        """当Import Value输入框内容改变且按回车时的处理"""
        try:
            if not hasattr(self.ui, 'gisaxsInputImportButtonValue'):
                return
                
            file_path_input = self.ui.gisaxsInputImportButtonValue.text().strip()
            
            if not file_path_input:
                self.status_updated.emit("Please enter a valid file path")
                return
            
            # 如果输入的只是文件名，尝试使用当前目录或之前的目录
            if not os.path.isabs(file_path_input):
                current_file = self.current_parameters.get('imported_gisaxs_file', '')
                if current_file and os.path.exists(current_file):
                    current_dir = os.path.dirname(current_file)
                    file_path_input = os.path.join(current_dir, file_path_input)
                else:
                    file_path_input = os.path.abspath(file_path_input)
            
            # 验证文件是否存在
            if not os.path.exists(file_path_input):
                self.status_updated.emit(f"File does not exist: {os.path.basename(file_path_input)}")
                QMessageBox.warning(self.main_window, "File Error", f"File does not exist:\n{file_path_input}")
                return
            
            # 更新参数
            self.current_parameters['imported_gisaxs_file'] = file_path_input
            
            # 更新UI显示为文件名
            file_name = os.path.basename(file_path_input)
            self.ui.gisaxsInputImportButtonValue.setText(file_name)
            
            # 验证文件
            if self._validate_imported_file(file_path_input):
                self.status_updated.emit(f"Updated GISAXS file: {file_name}")
                self.parameters_changed.emit(self.current_parameters)
                
                # 触发主控制器保存会话
                if hasattr(self.parent, 'save_current_session'):
                    self.parent.save_current_session()
                
                self._update_stack_display()
                self._refresh_vmin_vmax_display()
                
                if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked():
                    self._show_image()
                else:
                    self.status_updated.emit("File updated. Click 'Show' to display the image")
            
        except Exception as e:
            self.status_updated.emit(f"Import value processing error: {str(e)}")
            QMessageBox.critical(self.main_window, "Processing Error", f"Error handling the imported file path:\n{str(e)}")
    
    # ========== Stack 处理方法 ==========
    
    def _on_stack_value_changed(self):
        """当Stack值改变时的处理（回车触发）"""
        try:
            stack_text = self.ui.gisaxsInputStackValue.text() if hasattr(self.ui, 'gisaxsInputStackValue') else "1"
            
            try:
                stack_count = int(stack_text)
            except ValueError:
                if hasattr(self.ui, 'gisaxsInputStackValue'):
                    self.ui.gisaxsInputStackValue.setText("1")
                stack_count = 1
            
            if stack_count < 1:
                if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                    self.ui.gisaxsInputStackDisplayLabel.setText("At least 1")
                return
            
            self.current_parameters['stack_count'] = stack_count
            self._update_stack_display()
            self._refresh_vmin_vmax_display()
            
            should_reload_image = False
            
            if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked():
                should_reload_image = True
            elif self.current_stack_data is not None:
                imported_file = self.current_parameters.get('imported_gisaxs_file', '')
                if imported_file and os.path.splitext(imported_file)[1].lower() == '.cbf':
                    should_reload_image = True
            
            if should_reload_image:
                self._show_image()
            else:
                self.status_updated.emit(f"Stack count updated to {stack_count}")
            
        except Exception as e:
            self.status_updated.emit(f"Stack value processing error: {str(e)}")
    
    def _update_stack_display(self):
        """更新stack显示信息（根据 Single/Stack/In-situ 模式）"""
        try:
            imported_file = self.current_parameters.get('imported_gisaxs_file', '')
            if not imported_file:
                return

            file_ext = os.path.splitext(imported_file)[1].lower()
            mode = getattr(self, 'load_mode', 'Single')
            stack_count = self.current_parameters.get('stack_count', 1)

            if file_ext != '.cbf':
                if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                    self.ui.gisaxsInputStackDisplayLabel.setText(f"File: {os.path.basename(imported_file)}")
                return

            if mode == 'Single' or stack_count == 1:
                if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                    self.ui.gisaxsInputStackDisplayLabel.setText(f"Single: {os.path.basename(imported_file)}")
                return

            if mode == 'Stack':
                file_dir = os.path.dirname(imported_file)
                base_name = os.path.basename(imported_file)
                cbf_files = [f for f in os.listdir(file_dir) if f.lower().endswith('.cbf')]
                cbf_files.sort()
                try:
                    start_index = cbf_files.index(base_name)
                    available_files = len(cbf_files) - start_index
                    if stack_count > available_files:
                        if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                            self.ui.gisaxsInputStackDisplayLabel.setText(f"Maximum available: {available_files}")
                    else:
                        start_file = cbf_files[start_index]
                        end_file = cbf_files[start_index + stack_count - 1]
                        start_name = os.path.splitext(start_file)[0]
                        end_name = os.path.splitext(end_file)[0]
                        if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                            self.ui.gisaxsInputStackDisplayLabel.setText(f"Stack: {start_name} - {end_name}")
                except ValueError:
                    if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                        self.ui.gisaxsInputStackDisplayLabel.setText("File not found in directory")
                return

            if mode == 'In-situ':
                dir_path = os.path.dirname(imported_file)
                sv = ''
                try:
                    if hasattr(self.ui, 'gisaxsInputStackValue'):
                        sv = self.ui.gisaxsInputStackValue.text().strip()
                except Exception:
                    sv = ''
                latest = self._find_latest_cbf(dir_path)
                if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                    if sv == '' or sv.endswith('-'):
                        self.ui.gisaxsInputStackDisplayLabel.setText(
                            f"In-situ: latest -> {os.path.splitext(os.path.basename(latest or ''))[0]}"
                        )
                    elif '-' in sv:
                        self.ui.gisaxsInputStackDisplayLabel.setText(f"In-situ range: {sv}")
                    else:
                        self.ui.gisaxsInputStackDisplayLabel.setText(f"In-situ index: {sv}")
                return

        except Exception as e:
            self.status_updated.emit(f"Display update error: {str(e)}")
    
    # ========== 图像显示和处理方法 ==========
    
    def _sync_ui_to_parameters(self):
        """同步UI控件的当前值到参数系统"""
        try:
            # 同步文件路径 - 需要智能处理
            if hasattr(self.ui, 'gisaxsInputImportButtonValue'):
                file_input = self.ui.gisaxsInputImportButtonValue.text().strip()
                if file_input:
                    # 如果输入的是完整路径，直接使用
                    if os.path.isabs(file_input) and os.path.exists(file_input):
                        self.current_parameters['imported_gisaxs_file'] = file_input
                    # 如果只是文件名，尝试与当前目录或上次的目录组合
                    elif not os.path.isabs(file_input):
                        file_found = False
                        
                        # 先尝试使用当前已有的文件目录
                        current_file = self.current_parameters.get('imported_gisaxs_file', '')
                        if current_file and os.path.dirname(current_file):
                            # 使用上次文件的目录
                            new_path = os.path.join(os.path.dirname(current_file), file_input)
                            if os.path.exists(new_path):
                                self.current_parameters['imported_gisaxs_file'] = new_path
                                file_found = True
                        
                        # 如果上面没找到，尝试在Experiment_data目录中查找
                        if not file_found:
                            experiment_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Experiment_data')
                            if os.path.exists(experiment_dir):
                                new_path = os.path.join(experiment_dir, file_input)
                                if os.path.exists(new_path):
                                    self.current_parameters['imported_gisaxs_file'] = new_path
                                    file_found = True
                        
                        # 如果都没找到，不更新参数，保持原有设置
                        if not file_found:
                            self.status_updated.emit(f"Error: File '{file_input}' not found in any expected location")
                            # 不更新 imported_gisaxs_file 参数，这样后续检查会发现没有有效文件
                            return  # 直接返回，不继续处理
                    
                    # 如果是绝对路径但文件不存在
                    elif os.path.isabs(file_input) and not os.path.exists(file_input):
                        self.status_updated.emit(f"Error: File '{file_input}' does not exist")
                        # 不更新参数，保持原有设置
                        return  # 直接返回，不继续处理
            
            # 同步Stack/范围值按载入模式
            if hasattr(self.ui, 'gisaxsInputStackValue'):
                sv = self.ui.gisaxsInputStackValue.text().strip()
                if getattr(self, 'load_mode', 'Single') == 'Single':
                    self.current_parameters['stack_count'] = 1
                elif self.load_mode == 'Stack':
                    try:
                        self.current_parameters['stack_count'] = max(1, int(sv or '1'))
                    except Exception:
                        self.current_parameters['stack_count'] = 1
                elif self.load_mode == 'In-situ':
                    self.current_parameters['insitu_range'] = sv
                    
        except Exception as e:
            self.status_updated.emit(f"Failed to sync UI parameters: {str(e)}")
    
    def _show_image(self):
        """显示图像"""
        try:
            # 先同步UI控件的最新值到参数系统
            self._sync_ui_to_parameters()
            
            imported_file = self.current_parameters.get('imported_gisaxs_file', '')
            if not imported_file:
                self.status_updated.emit("No file imported to show")
                return
            
            # 检查依赖库
            if not is_fabio_available():
                QMessageBox.warning(self.main_window, "Missing Library", 
                                  "fabio library is required for CBF file processing.\nPlease install it using: pip install fabio")
                return
                
            if not is_matplotlib_available():
                QMessageBox.warning(self.main_window, "Missing Library", 
                                  "matplotlib library is required for image display.\nPlease install it using: pip install matplotlib")
                return
            
            # 处理文件并显示（按模式）
            file_ext = os.path.splitext(imported_file)[1].lower()
            if file_ext != '.cbf':
                self.status_updated.emit("Image display only supports CBF files currently")
                return

            mode = getattr(self, 'load_mode', 'Single')
            if mode == 'Single':
                self.status_updated.emit("Please wait while the image starts loading (Single)...")
                self.async_image_loader.load_image(imported_file, 1)
            elif mode == 'Stack':
                stack_count = self.current_parameters.get('stack_count', 1)
                self.status_updated.emit(f"Please wait while stacking {stack_count} files...")
                self.async_image_loader.load_image(imported_file, stack_count)
            else:
                # In-situ 模式：根据范围选择目标文件
                sv = ''
                try:
                    if hasattr(self.ui, 'gisaxsInputStackValue'):
                        sv = self.ui.gisaxsInputStackValue.text().strip()
                except Exception:
                    sv = ''
                dir_path = os.path.dirname(imported_file)
                target = self._resolve_insitu_target(dir_path, imported_file, sv)
                if not target:
                    self.status_updated.emit("No CBF file found for In-situ mode")
                    return
                self._insitu_last_file = target
                self._show_image_insitu(target)
            
        except Exception as e:
            self.status_updated.emit(f"Show image error: {str(e)}")

    def _on_load_mode_changed(self, text: str):
        """载入模式组合框改变回调"""
        try:
            self.load_mode = text or 'Single'
            # 持久化
            try:
                from core.global_params import GlobalParameterManager
                gp = GlobalParameterManager()
                gp.set_parameter('fitting', 'gisaxs_input.load_mode', self.load_mode)
            except Exception:
                pass
            # 控件可见性
            self._update_stack_controls_visibility()
            # 定时器管理
            if self.load_mode == 'In-situ':
                if self._is_auto_show_enabled():
                    self._start_insitu_timer()
            else:
                self._stop_insitu_timer()
            # 更新显示标签
            self._update_stack_display()
        except Exception:
            pass

    def _update_stack_controls_visibility(self):
        """根据模式控制Stack输入框显示与否"""
        try:
            base_widget = getattr(self.ui, 'gisaxsInputStackValue', None)
            try:
                from PyQt5.QtWidgets import QLineEdit
                from PyQt5.QtGui import QIntValidator, QRegularExpressionValidator
                from PyQt5.QtCore import QRegularExpression
                if self.load_mode == 'In-situ':
                    # 隐藏原始控件（可能是数值型，不支持 1-）
                    if base_widget is not None:
                        base_widget.setVisible(False)
                    # 创建或显示专用 In-situ LineEdit
                    if not hasattr(self, '_insitu_lineedit') or self._insitu_lineedit is None:
                        # 尝试使用与原控件同一父级
                        parent = None
                        try:
                            parent = base_widget.parent() if base_widget is not None else self.ui.gisaxsInputStackDisplayLabel.parent()
                        except Exception:
                            parent = None
                        from PyQt5.QtWidgets import QLineEdit as _QLE
                        self._insitu_lineedit = _QLE(parent)
                        # 放入父布局（如果存在）：插入到状态显示标签之前
                        try:
                            layout = parent.layout() if parent is not None else None
                            if layout is not None:
                                # 如果能定位到显示标签，则插入到它前面；否则追加
                                try:
                                    disp_label = getattr(self.ui, 'gisaxsInputStackDisplayLabel', None)
                                    if disp_label is not None:
                                        idx = layout.indexOf(disp_label)
                                        if idx >= 0:
                                            layout.insertWidget(idx, self._insitu_lineedit)
                                        else:
                                            layout.addWidget(self._insitu_lineedit)
                                    else:
                                        layout.addWidget(self._insitu_lineedit)
                                except Exception:
                                    layout.addWidget(self._insitu_lineedit)
                        except Exception:
                            pass
                        # 验证器：允许 N | A-B | A-
                        regex = QRegularExpression(r"^\s*(?:\d+|\d+\s*-\s*\d+|\d+\s*-)\s*$")
                        self._insitu_lineedit.setValidator(QRegularExpressionValidator(regex, self._insitu_lineedit))
                        # 默认值
                        self._insitu_lineedit.setText('1-')
                        self._insitu_lineedit.setPlaceholderText('e.g. 1-, 1-10, 5')
                        # 信号连接：回车/编辑完成
                        try:
                            self._insitu_lineedit.returnPressed.connect(self._on_stack_value_changed)
                            self._insitu_lineedit.editingFinished.connect(self._on_stack_value_changed)
                        except Exception:
                            pass
                    self._insitu_lineedit.setVisible(True)
                else:
                    # 非 In-situ 模式：隐藏 LineEdit，恢复原控件
                    if hasattr(self, '_insitu_lineedit') and self._insitu_lineedit is not None:
                        self._insitu_lineedit.setVisible(False)
                    if base_widget is not None:
                        base_widget.setVisible(self.load_mode != 'Single')
                        if self.load_mode == 'Stack':
                            base_widget.setValidator(QIntValidator(1, 9999, base_widget))
                            base_widget.setPlaceholderText('e.g. 5')
                        else:
                            base_widget.setValidator(None)
                            base_widget.setPlaceholderText('')
            except Exception:
                # 缺少 PyQt5 环境时的容错
                if base_widget is not None:
                    base_widget.setVisible(self.load_mode != 'Single')
            if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                self.ui.gisaxsInputStackDisplayLabel.setVisible(True)
        except Exception:
            pass

    def _enforce_insitu_visibility_once(self):
        """初始化结束后再执行一次严格的可见性修正，确保 In-situ 下不出现双输入框。"""
        try:
            mode = getattr(self, 'load_mode', 'Single')
            base_widget = getattr(self.ui, 'gisaxsInputStackValue', None)
            insitu_edit = getattr(self, '_insitu_lineedit', None)
            if mode == 'In-situ':
                if base_widget is not None:
                    base_widget.setVisible(False)
                if insitu_edit is not None:
                    insitu_edit.setVisible(True)
            elif mode == 'Stack':
                if insitu_edit is not None:
                    insitu_edit.setVisible(False)
                if base_widget is not None:
                    base_widget.setVisible(True)
            else:
                # Single
                if insitu_edit is not None:
                    insitu_edit.setVisible(False)
                if base_widget is not None:
                    base_widget.setVisible(False)
        except Exception:
            pass

    # 移除复杂的守护定时器逻辑，保持简单直接的可见性修正

    def _is_auto_show_enabled(self) -> bool:
        return hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked()

    def _start_insitu_timer(self):
        try:
            if self._insitu_timer is None:
                from PyQt5.QtCore import QTimer
                self._insitu_timer = QTimer()
                self._insitu_timer.setSingleShot(False)
                self._insitu_timer.timeout.connect(self._insitu_poll_latest)
            self._insitu_timer.start(2000)
        except Exception:
            pass

    def _stop_insitu_timer(self):
        try:
            if self._insitu_timer is not None:
                self._insitu_timer.stop()
        except Exception:
            pass

    def _insitu_poll_latest(self):
        try:
            if self.load_mode != 'In-situ':
                return
            imported_file = self.current_parameters.get('imported_gisaxs_file', '')
            if not imported_file:
                return
            dir_path = os.path.dirname(imported_file)
            latest = self._find_latest_cbf(dir_path)
            if latest and latest != self._insitu_last_file:
                sv = ''
                try:
                    if hasattr(self.ui, 'gisaxsInputStackValue'):
                        sv = self.ui.gisaxsInputStackValue.text().strip()
                except Exception:
                    sv = ''
                if (sv.endswith('-') or sv.strip() == '' or sv.strip().endswith('-')) and self._is_auto_show_enabled():
                    self._insitu_last_file = latest
                    self._show_image_insitu(latest)
        except Exception:
            pass

    def _get_stack_value_text(self) -> str:
        try:
            if getattr(self, 'load_mode', 'Single') == 'In-situ' and hasattr(self, '_insitu_lineedit') and self._insitu_lineedit is not None:
                return self._insitu_lineedit.text().strip()
            if hasattr(self.ui, 'gisaxsInputStackValue'):
                return self.ui.gisaxsInputStackValue.text().strip()
        except Exception:
            pass
        return ''

    def _resolve_insitu_target(self, dir_path: str, imported_file: str, sv_text: str) -> str:
        try:
            files = [f for f in os.listdir(dir_path) if f.lower().endswith('.cbf')]
            if not files:
                return None
            files.sort()
            def _index_from_name(name: str):
                base = os.path.splitext(name)[0]
                import re
                m = re.search(r'(\d+)$', base)
                return int(m.group(1)) if m else None
            indexed = [(fn, _index_from_name(fn)) for fn in files]
            latest_file = files[-1]
            t = (sv_text or '').strip()
            if t == '':
                return os.path.join(dir_path, latest_file)
            if '-' in t:
                parts = t.split('-')
                try:
                    a = int(parts[0]) if parts[0] != '' else None
                except Exception:
                    a = None
                b = None
                try:
                    if len(parts) > 1 and parts[1] != '':
                        b = int(parts[1])
                except Exception:
                    b = None
                if b is None:
                    return os.path.join(dir_path, latest_file)
                for fn, idx in indexed:
                    if idx == b:
                        return os.path.join(dir_path, fn)
                return os.path.join(dir_path, latest_file)
            else:
                try:
                    n = int(t)
                    for fn, idx in indexed:
                        if idx == n:
                            return os.path.join(dir_path, fn)
                except Exception:
                    pass
                return os.path.join(dir_path, latest_file)
        except Exception:
            return None

    def _find_latest_cbf(self, dir_path: str) -> str:
        try:
            files = [f for f in os.listdir(dir_path) if f.lower().endswith('.cbf')]
            if not files:
                return None
            files.sort()
            return os.path.join(dir_path, files[-1])
        except Exception:
            return None

    def _show_image_insitu(self, target_path: str):
        try:
            if not target_path:
                return
            self.async_image_loader.load_image(target_path, 1)
            if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                base = os.path.basename(target_path)
                self.ui.gisaxsInputStackDisplayLabel.setText(f"In-situ: {os.path.splitext(base)[0]}")
        except Exception:
            pass
    
    def _on_image_loaded(self, image_data, file_path):
        """图像加载完成"""
        try:
            self.status_updated.emit(f"Image loading complete: {os.path.basename(file_path)}")
            self._display_image(image_data)
        except Exception as e:
            self.status_updated.emit(f"Error while displaying image: {str(e)}")
    
    def _on_image_loading_progress(self, progress, status):
        """图像加载进度更新"""
        try:
            self.status_updated.emit(f"Image loading... {progress}% - {status}")
            self.progress_updated.emit(progress)
        except Exception as e:
            self.status_updated.emit(f"Progress update error: {str(e)}")
    
    def _on_image_loading_error(self, error_message):
        """图像加载错误处理"""
        QMessageBox.critical(self.main_window, "Image loading error", error_message)
    
    def _display_image(self, image_data):
        """显示图像数据"""
        try:
            # 存储当前数据
            self.current_stack_data = image_data
            # 同步全局缓存，便于导出/后续计算
            self.data = image_data
            try:
                sc = int(self.current_parameters.get('stack_count', 1))
            except Exception:
                sc = 1
            self.summed_data = image_data if sc and sc > 1 else None
            # 重新计算并缓存Q空间网格（qy, qz, qr）
            self._compute_q_meshgrids_and_store()
            
            # 处理颜色标尺逻辑
            self._handle_color_scale(image_data)
            
            # 更新Cut Line标签的单位
            self._update_cutline_labels_units()
            
            # 更新内嵌的GraphicsView
            if hasattr(self.ui, 'gisaxsInputGraphicsView'):
                self._update_graphics_view(image_data)
            
            # 如果独立窗口已打开，同时更新独立窗口
            if self.independent_window is not None and self.independent_window.isVisible():
                is_log = self._is_log_mode_enabled()
                self.independent_window.update_image(image_data, self._current_vmin, self._current_vmax, use_log=is_log)
            
            # 状态信息
            window_status = " (+ Independent window)" if (self.independent_window and self.independent_window.isVisible()) else ""
            vmin_vmax_info = f" [Vmin: {self._current_vmin:.3f}, Vmax: {self._current_vmax:.3f}]" if self._current_vmin is not None and self._current_vmax is not None else ""
            mode_text = "Log" if self._is_log_mode_enabled() else "Linear"
            self.status_updated.emit(f"{mode_text} image displayed: {image_data.shape}{vmin_vmax_info}{window_status}")
            
        except Exception as e:
            self.status_updated.emit(f"Display error: {str(e)}")

    def _compute_q_meshgrids_and_store(self):
        """根据当前图像及探测器参数计算并缓存 qy/qz/qr 网格。"""
        try:
            if self.current_stack_data is None:
                return
            from core.global_params import GlobalParameterManager
            from utils.q_space_calculator import create_detector_from_image_and_params
            global_params = GlobalParameterManager()
            height, width = self.current_stack_data.shape
            pixel_size_x = global_params.get_parameter('fitting', 'detector.pixel_size_x', 172.0)
            pixel_size_y = global_params.get_parameter('fitting', 'detector.pixel_size_y', 172.0)
            beam_center_x = global_params.get_parameter('fitting', 'detector.beam_center_x', width / 2.0)
            beam_center_y = global_params.get_parameter('fitting', 'detector.beam_center_y', height / 2.0)
            distance = global_params.get_parameter('fitting', 'detector.distance', 2565.0)
            theta_in_deg = global_params.get_parameter('beam', 'grazing_angle', 0.4)
            wavelength = global_params.get_parameter('beam', 'wavelength', 0.1045)
            detector = create_detector_from_image_and_params(
                image_shape=(height, width),
                pixel_size_x=pixel_size_x,
                pixel_size_y=pixel_size_y,
                beam_center_x=beam_center_x,
                beam_center_y=beam_center_y,
                distance=distance,
                theta_in_deg=theta_in_deg,
                wavelength=wavelength,
                crop_params=None
            )
            qy_mesh, qz_mesh = detector.get_qy_qz_meshgrids()
            self.qy_matrix = qy_mesh
            self.qz_matrix = qz_mesh
            try:
                self.qr_matrix = np.sqrt(np.square(qy_mesh) + np.square(qz_mesh))
            except Exception:
                self.qr_matrix = None
        except Exception:
            # 失败则清空缓存，不中断主流程
            self.qy_matrix = None
            self.qz_matrix = None
            self.qr_matrix = None
    
    def _update_graphics_view(self, image_data):
        """更新GraphicsView中的图像显示"""
        self._update_graphics_view_with_selection(image_data, None)
    
    def _prepare_image_data_for_display(self, image_data):
        """根据当前显示模式准备图像数据"""
        try:
            is_log = self._is_log_mode_enabled()
            
            if is_log:
                # Log模式：先过滤掉负值和零值，然后取对数
                safe_data = np.where(image_data > 0, image_data, 0.001)
                processed_data = np.log(safe_data, dtype=np.float32)
            else:
                # 线性模式：直接使用原始数据
                processed_data = image_data.astype(np.float32)
            
            return processed_data, is_log
            
        except Exception:
            # 出错时返回原始数据和Log模式
            return image_data.astype(np.float32), True
    
    def _refresh_image_display(self):
        """刷新当前图像显示（保持当前数据，更新颜色标尺或显示模式）"""
        try:
            if self.current_stack_data is not None:
                # 更新内嵌的GraphicsView
                if hasattr(self.ui, 'gisaxsInputGraphicsView'):
                    self._update_graphics_view(self.current_stack_data)
                
                # 如果独立窗口已打开，同时更新独立窗口
                if self.independent_window is not None and self.independent_window.isVisible():
                    is_log = self._is_log_mode_enabled()
                    self.independent_window.update_image(self.current_stack_data, 
                                                       self._current_vmin, self._current_vmax, 
                                                       use_log=is_log)
        except Exception as e:
            self.status_updated.emit(f"Refresh display error: {str(e)}")
    
    def _on_graphics_view_double_click(self, event):
        """GraphicsView双击事件处理"""
        try:
            if not is_matplotlib_available():
                QMessageBox.warning(self.main_window, "Missing Library",
                                  "matplotlib library is required for independent window.\nPlease install it using: pip install matplotlib")
                return
            
            # 如果没有当前图像数据，提示用户
            if self.current_stack_data is None:
                QMessageBox.information(self.main_window, "No Image", "Please import and display an image first.")
                return
            
            # 打开或显示独立窗口
            self._show_independent_window()
            
        except Exception as e:
            self.status_updated.emit(f"Double-click error: {str(e)}")
    
    def _show_independent_window(self):
        """显示独立的matplotlib窗口"""
        try:
            # 如果窗口不存在或已关闭，创建新窗口
            if self.independent_window is None or not self.independent_window.isVisible():
                self.independent_window = IndependentMatplotlibWindow(self.main_window)
                # 连接区域选择信号
                self.independent_window.region_selected.connect(self._on_region_selected)
                # 连接状态更新信号
                self.independent_window.status_updated.connect(self.status_updated.emit)
            
            # 更新窗口中的图像（使用当前的vmin/vmax和log模式）
            if self.current_stack_data is not None:
                is_log = self._is_log_mode_enabled()
                # 传递图像尺寸信息
                self.independent_window.current_image_shape = self.current_stack_data.shape
                self.independent_window.update_image(self.current_stack_data, 
                                                   self._current_vmin, self._current_vmax, 
                                                   use_log=is_log)
            
            # 显示窗口并置于前台
            self.independent_window.show()
            self.independent_window.raise_()
            self.independent_window.activateWindow()
            
            # 设置焦点到canvas以确保键盘事件工作
            self.independent_window.canvas.setFocus()
            
            self.status_updated.emit("Independent window opened - Right-click to activate selection, ESC to exit selection mode")
            
        except Exception as e:
            self.status_updated.emit(f"Independent window error: {str(e)}")
    
    def _on_region_selected(self, selection_info):
        """处理从独立窗口传来的区域选择信息，更新Cut Line参数"""
        try:
            # 检查选择信息中是否包含坐标空间标记
            is_q_space = selection_info.get('is_q_space', False)
            
            # 更新Cut Line的Center参数
            updated_controls = []
            
            if is_q_space:
                # Q坐标模式：框选得到的就是Q坐标，直接使用
                center_qy = selection_info.get('center_x', 0)  # Q坐标中的x对应qy  
                center_qz = selection_info.get('center_y', 0)  # Q坐标中的y对应qz
                width_q = selection_info.get('width', 0)
                height_q = selection_info.get('height', 0)
                
                self.status_updated.emit(
                    f"Q-space region selected: Center({center_qy:.6f}, {center_qz:.6f}) nm⁻¹, "
                    f"Size({width_q:.6f}×{height_q:.6f}) nm⁻¹"
                )
                
                try:
                    # 直接更新显示数值为Q空间坐标
                    if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                        self.ui.gisaxsInputCenterVerticalValue.setValue(center_qz)  # Vertical对应qz（Y方向）
                        updated_controls.append('gisaxsInputCenterVerticalValue')
                        
                    if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                        self.ui.gisaxsInputCenterParallelValue.setValue(center_qy)  # Parallel对应qy（X方向）
                        updated_controls.append('gisaxsInputCenterParallelValue')
                    
                    if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                        self.ui.gisaxsInputCutLineVerticalValue.setValue(height_q)  # Vertical尺寸对应height_q
                        updated_controls.append('gisaxsInputCutLineVerticalValue')
                        
                    if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                        self.ui.gisaxsInputCutLineParallelValue.setValue(width_q)  # Parallel尺寸对应width_q
                        updated_controls.append('gisaxsInputCutLineParallelValue')
                        
                    # 为了在主窗口中显示选择框，需要将Q坐标转换为像素坐标
                    pixel_coords = self._convert_q_to_pixel_coordinates(center_qy, center_qz, width_q, height_q)
                    center_x = pixel_coords['center_x']
                    center_y = pixel_coords['center_y']
                    width = pixel_coords['width']
                    height = pixel_coords['height']
                        
                except Exception as e:
                    self.status_updated.emit(f"Q-space parameter update failure: {str(e)}")
                    return
            else:
                # 像素坐标模式：直接使用像素坐标
                center_x = selection_info.get('pixel_center_x', 0)
                center_y = selection_info.get('pixel_center_y', 0)
                width = selection_info.get('pixel_width', 0)
                height = selection_info.get('pixel_height', 0)
                
                self.status_updated.emit(
                    f"Pixel region selected: Center({center_x}, {center_y}), Size({width}×{height})"
                )
                
                # 像素模式：直接使用像素坐标
                # Vertical对应Y轴方向（竖直方向），Parallel对应X轴方向（水平方向）
                # 更新Vertical中心点 -> Y坐标
                if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                    self.ui.gisaxsInputCenterVerticalValue.setValue(center_y)
                    updated_controls.append('gisaxsInputCenterVerticalValue')
                    
                # 更新Parallel中心点 -> X坐标
                if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                    self.ui.gisaxsInputCenterParallelValue.setValue(center_x)
                    updated_controls.append('gisaxsInputCenterParallelValue')
                
                # 更新Cut Line的Vertical和Parallel值（修正映射关系）
                # Vertical对应高度（Y方向），Parallel对应宽度（X方向）
                if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                    self.ui.gisaxsInputCutLineVerticalValue.setValue(height)
                    updated_controls.append('gisaxsInputCutLineVerticalValue')
                    
                if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                    self.ui.gisaxsInputCutLineParallelValue.setValue(width)
                    updated_controls.append('gisaxsInputCutLineParallelValue')
            
            # 在主窗口GraphicsView中也显示选择区域
            if is_q_space:
                # Q坐标模式：直接传递Q坐标信息给主窗口
                main_view_selection_info = {
                    'bounds': {
                        'x_min': center_qy - width_q / 2,
                        'x_max': center_qy + width_q / 2,
                        'y_min': center_qz - height_q / 2,
                        'y_max': center_qz + height_q / 2
                    },
                    'center_x': center_qy,
                    'center_y': center_qz,
                    'width': width_q,
                    'height': height_q,
                    'is_q_space': True
                }
            else:
                # 像素坐标模式：使用像素坐标
                main_view_selection_info = {
                    'bounds': {
                        'x_min': center_x - width / 2,
                        'x_max': center_x + width / 2,
                        'y_min': center_y - height / 2,
                        'y_max': center_y + height / 2
                    },
                    'pixel_center_x': center_x,
                    'pixel_center_y': center_y,
                    'pixel_width': width,
                    'pixel_height': height,
                    'is_q_space': False
                }
            
            self._draw_selection_on_main_view(main_view_selection_info)
            
            # 显示成功更新的信息
            if updated_controls:
                coord_mode = "Q坐标" if is_q_space else "像素坐标"
                self.status_updated.emit(f"Updated Cut Line parameters ({coord_mode}): {', '.join(updated_controls)}")
                # 在独立窗口中也显示成功消息
                if self.independent_window and self.independent_window.isVisible():
                    if is_q_space:
                        self.independent_window.setWindowTitle(
                            f"GISAXS Image Viewer - Q参数已更新: "
                            f"Center({center_qy:.6f}, {center_qz:.6f}) nm⁻¹, Size({width_q:.6f}×{height_q:.6f}) nm⁻¹"
                        )
                    else:
                        self.independent_window.setWindowTitle(
                            f"GISAXS Image Viewer - 参数已更新: "
                            f"Center({center_x}, {center_y}), Size({width}×{height})"
                        )
            else:
                self.status_updated.emit("No matching Cut Line controls found for parameter update")
            
        except Exception as e:
            self.status_updated.emit(f"Error updating Cut Line parameters: {str(e)}")
    
    @staticmethod
    def _plot_cut_data_with_log_handling(ax, x_coords, y_intensity, is_log_x, markersize=4, linewidth=1.5):
        """共享的绘图函数，处理log-x模式下的负值坐标"""
        try:
            x_array = np.array(x_coords)
            y_array = np.array(y_intensity)
            
            if is_log_x:
                # 分离正值和负值数据
                positive_mask = x_array > 0
                x_positive = x_array[positive_mask]
                y_positive = y_array[positive_mask]
                
                # 负值数据（取绝对值）
                negative_mask = x_array < 0
                x_negative_abs = np.abs(x_array[negative_mask])
                y_negative = y_array[negative_mask]
                
                # 零值数据（特殊处理）
                zero_mask = x_array == 0
                x_zero = x_array[zero_mask]
                y_zero = y_array[zero_mask]
                
                # 绘制正值数据（实线）
                if len(x_positive) > 0:
                    ax.plot(x_positive, y_positive, 'bo-', markersize=markersize, linewidth=linewidth, 
                           markerfacecolor='lightblue', alpha=0.8, label='Positive coordinates')
                
                # 绘制负值数据（虚线，坐标取绝对值）
                if len(x_negative_abs) > 0:
                    ax.plot(x_negative_abs, y_negative, 'ro--', markersize=markersize, linewidth=linewidth, 
                           markerfacecolor='lightcoral', alpha=0.8, label='Negative coordinates (|x|)')
                
                # 绘制零值数据（如果存在）
                if len(x_zero) > 0:
                    # 零值在对数坐标中无法显示，用最小正值代替
                    min_positive = min(np.min(x_positive) if len(x_positive) > 0 else 1e-6,
                                     np.min(x_negative_abs) if len(x_negative_abs) > 0 else 1e-6)
                    x_zero_replacement = np.full_like(x_zero, min_positive * 0.1)
                    ax.plot(x_zero_replacement, y_zero, 'go^', markersize=markersize+2, 
                           markerfacecolor='lightgreen', alpha=0.8, label='Zero coordinates (approximated)')
                
                # 添加图例
                ax.legend(loc='best', fontsize=max(8, markersize*2))
                
            else:
                # 非Log-X模式，正常绘制
                ax.plot(x_array, y_array, 'bo-', markersize=markersize, linewidth=linewidth, 
                       markerfacecolor='lightblue', alpha=0.8)
                
        except Exception as e:
            raise Exception(f"Plot data error: {str(e)}")
    
    def _on_fit_graphics_view_double_click(self, event):
        """fitGraphicsView双击事件处理 - 直接更新并显示独立窗口"""
        try:
            if not is_matplotlib_available():
                QMessageBox.warning(self.main_window, "Missing Library",
                                  "matplotlib library is required for independent window.\nPlease install it using: pip install matplotlib")
                return
            
            # 检查数据可用性
            if self.q is None or self.I is None:
                QMessageBox.information(self.main_window, "No Data", "No data available for display.")
                return
            
            # 创建或显示独立拟合窗口
            if self.independent_fit_window is None or not self.independent_fit_window.isVisible():
                self.independent_fit_window = IndependentFitWindow(self.main_window)
                self.independent_fit_window.status_updated.connect(self.status_updated.emit)
                # 连接Positive Only复选框到更新函数
                self.independent_fit_window.show_positive_cb.toggled.connect(self._on_positive_only_changed)
                # 与主界面的“Positive Only”状态保持一致
                try:
                    if hasattr(self.ui, 'PositiveOnlyCheckBox'):
                        self.independent_fit_window.show_positive_cb.setChecked(self.ui.PositiveOnlyCheckBox.isChecked())
                except Exception:
                    pass
                
                # 显示窗口
                self.independent_fit_window.show()
                self.independent_fit_window.raise_()
                self.independent_fit_window.activateWindow()
            
            # 立即更新窗口内容（确保双击后直接显示完整内容）
            # 修正策略：以当前显示模式为主；仅当“确实处于拟合模式”时才强制使用fitting
            mode = (self.display_mode if hasattr(self, 'display_mode') else 'normal')
            try:
                if hasattr(self, '_is_in_fitting_mode') and callable(self._is_in_fitting_mode) and self._is_in_fitting_mode():
                    mode = 'fitting'
            except Exception:
                pass
            # 若没有可用的拟合数据，则强制回退到 normal 模式，避免以“空拟合”模式打开外置窗口
            try:
                has_fit = bool(getattr(self, 'has_fitting_data', False) and getattr(self, 'I_fitting', None) is not None)
                if mode == 'fitting' and not has_fit:
                    mode = 'normal'
            except Exception:
                pass

            # 关键修正：若以拟合模式打开，先确保GUI中的拟合图刷新（同步内部状态与ROI/Normalize等），再刷新外部窗口
            if mode == 'fitting':
                try:
                    self._update_gui_fitting_display()
                except Exception:
                    pass
                self._update_outside_window('fitting')
            else:
                self._update_outside_window(mode)
            
            # 确保窗口获得焦点
            if hasattr(self.independent_fit_window, 'canvas'):
                self.independent_fit_window.canvas.setFocus()
                # 强制刷新画布
                self.independent_fit_window.canvas.draw()
            
            self.status_updated.emit(f"{mode.capitalize()} mode independent window updated")
            
        except Exception as e:
            self.status_updated.emit(f"Fit double-click error: {str(e)}")
    

    
    def _on_cutline_parameters_changed(self):
        """当Cut Line参数改变时，更新图像中的选择框显示，并自动重新执行Cut操作（如果之前已有Cut结果）"""
        try:
            # 如果正在初始化中，跳过处理
            if getattr(self, '_initializing', True):
                return
                
            # 使用防抖动机制，避免频繁更新导致卡顿
            if not hasattr(self, '_cutline_update_timer'):
                from PyQt5.QtCore import QTimer
                self._cutline_update_timer = QTimer()
                self._cutline_update_timer.setSingleShot(True)
                self._cutline_update_timer.timeout.connect(self._delayed_cutline_update)
            
            # 停止之前的定时器，重新开始计时
            self._cutline_update_timer.stop()
            self._cutline_update_timer.start(150)  # 150ms延迟，适中的响应时间
            
        except Exception as e:
            pass
    
    def _delayed_cutline_update(self):
        """延迟执行Cut Line参数更新，减少卡顿"""
        try:
            # 获取当前的Cut Line参数值
            center_x = 0
            center_y = 0
            width = 0
            height = 0
            
            # 获取Center参数
            if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                center_x = self.ui.gisaxsInputCenterParallelValue.value()
            if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                center_y = self.ui.gisaxsInputCenterVerticalValue.value()
                
            # 获取Cut Line尺寸参数
            if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                width = self.ui.gisaxsInputCutLineParallelValue.value()
            if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                height = self.ui.gisaxsInputCutLineVerticalValue.value()
            
            # 如果所有参数都为0，清除选择框显示
            if center_x == 0 and center_y == 0 and width == 0 and height == 0:
                self._clear_parameter_selection()
                return
                
            # 如果任何尺寸参数为0，也清除选择框
            if width <= 0 or height <= 0:
                self._clear_parameter_selection()
                return
            
            # 构建选择区域信息
            selection_info = self._create_selection_from_parameters(center_x, center_y, width, height)
            
            # 更新图像显示
            self._update_parameter_selection_display(selection_info)
            
            # 检查是否已经有Cut结果数据，如果有则自动重新执行Cut操作
            if (self.current_cut_data is not None and 
                hasattr(self, 'current_stack_data') and self.current_stack_data is not None):
                
                # 自动重新执行Cut操作
                self._perform_cut()
                self.status_updated.emit(f"Auto-updated cut with new parameters: Center({center_x}, {center_y}), Size({width}×{height})")
            else:
                # 更新状态信息
                self.status_updated.emit(f"Parameter selection: Center({center_x}, {center_y}), Size({width}×{height}) - Perform Cut to see results")
            
        except Exception as e:
            self.status_updated.emit(f"Error updating parameter selection: {str(e)}")
    
    def _create_selection_from_parameters(self, center_x, center_y, width, height):
        """根据参数创建选择区域信息"""
        # 计算选择区域的边界
        half_width = width / 2
        half_height = height / 2
        
        x_min = center_x - half_width
        x_max = center_x + half_width
        y_min = center_y - half_height
        y_max = center_y + half_height
        
        # 构建选择信息，格式与鼠标选择一致
        selection_info = {
            'center_x': center_x,
            'center_y': center_y,
            'width': width,
            'height': height,
            'pixel_center_x': int(center_x),
            'pixel_center_y': int(center_y),
            'pixel_width': int(width),
            'pixel_height': int(height),
            'bounds': {
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max
            },
            'is_parameter_based': True  # 标记这是基于参数的选择
        }
        
        return selection_info
    
    def _update_parameter_selection_display(self, selection_info):
        """更新基于参数的选择框显示"""
        try:
            # 存储当前的参数选择信息
            self.current_parameter_selection = selection_info
            
            # 在主窗口中显示选择框
            if self.current_stack_data is not None:
                self._update_graphics_view_with_selection(self.current_stack_data, selection_info)
            
            # 在独立窗口中显示选择框
            if self.independent_window is not None and self.independent_window.isVisible():
                # 从selection_info中提取参数
                center_v = selection_info.get('pixel_center_y', 0)
                center_p = selection_info.get('pixel_center_x', 0) 
                cutline_v = selection_info.get('pixel_height', 0)
                cutline_p = selection_info.get('pixel_width', 0)
                self.independent_window.update_parameter_selection(center_v, center_p, cutline_v, cutline_p)
            
        except Exception as e:
            self.status_updated.emit(f"Error updating parameter selection display: {str(e)}")
    
    def _clear_parameter_selection(self):
        """清除参数选择框显示"""
        try:
            # 清除存储的参数选择信息
            self.current_parameter_selection = None
            
            # 刷新主窗口显示（不显示选择框）
            if self.current_stack_data is not None:
                self._update_graphics_view_with_selection(self.current_stack_data, None)
            
            # 清除独立窗口中的选择框
            if self.independent_window is not None and self.independent_window.isVisible():
                self.independent_window.clear_parameter_selection()
            
            self.status_updated.emit("Parameter selection cleared")
            
        except Exception as e:
            self.status_updated.emit(f"Error clearing parameter selection: {str(e)}")
    
    def _draw_selection_on_main_view(self, selection_info):
        """在主窗口的GraphicsView中绘制选择区域"""
        try:
            if not hasattr(self.ui, 'gisaxsInputGraphicsView') or self.current_stack_data is None:
                return
            
            # 重新刷新主窗口的图像显示，并添加选择矩形
            self._update_graphics_view_with_selection(self.current_stack_data, selection_info)
            
        except Exception as e:
            self.status_updated.emit(f"Error drawing selection on main view: {str(e)}")
    
    def _update_graphics_view_with_selection(self, image_data, selection_info=None):
        """更新GraphicsView中的图像显示，可选择性地添加选择矩形"""
        try:
            if not is_matplotlib_available():
                self.status_updated.emit("matplotlib not available for image display")
                return
            
            graphics_view = self.ui.gisaxsInputGraphicsView
            
            # 使用预创建的scene，只需要清除内容
            if self._graphics_scene is None:
                self._graphics_scene = QGraphicsScene()
                graphics_view.setScene(self._graphics_scene)
            else:
                self._graphics_scene.clear()
            
            # 计算图像比例和尺寸
            img_height, img_width = image_data.shape
            aspect_ratio = img_width / img_height
            
            # 优化figure尺寸计算
            base_size = 6
            if aspect_ratio > 1:
                fig_width = base_size
                fig_height = base_size / aspect_ratio
            else:
                fig_height = base_size
                fig_width = base_size * aspect_ratio
            
            # 限制最小尺寸
            fig_width = max(fig_width, 3)
            fig_height = max(fig_height, 2.5)
            
            # 创建figure，降低DPI以提高性能（延迟导入matplotlib）
            try:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            except Exception:
                # 无法导入matplotlib，直接返回
                return
            fig = Figure(figsize=(fig_width, fig_height), dpi=72)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            
            # 根据当前显示模式准备图像数据
            processed_data, is_log = self._prepare_image_data_for_display(image_data)
            
            # 垂直翻转图像数据以修正显示方向
            processed_data = np.flipud(processed_data)
            
            # 检查是否需要显示Q轴
            show_q_axis = self._should_show_q_axis()
            
            # 显示图像，使用计算出的vmin/vmax
            vmin = self._current_vmin if self._current_vmin is not None else np.min(processed_data)
            vmax = self._current_vmax if self._current_vmax is not None else np.max(processed_data)
            
            if show_q_axis:
                # Q坐标模式：获取Q轴extent并使用
                try:
                    # 获取缓存的Q空间网格来计算精确的extent
                    qy_mesh, qz_mesh = self._get_cached_q_meshgrids()
                    
                    if qy_mesh is not None and qz_mesh is not None:
                        # 使用Q网格来计算精确的extent [left, right, bottom, top]
                        qy_min, qy_max = qy_mesh.min(), qy_mesh.max()
                        qz_min, qz_max = qz_mesh.min(), qz_mesh.max()
                        q_extent = [qy_min, qy_max, qz_min, qz_max]
                        
                        im = ax.imshow(processed_data, cmap='viridis', aspect='equal', origin='lower', 
                                      interpolation='nearest', vmin=vmin, vmax=vmax, extent=q_extent)
                        
                        # 设置Q轴标签
                        ax.set_xlabel(r'$q_y$ (nm$^{-1}$)')
                        ax.set_ylabel(r'$q_z$ (nm$^{-1}$)')
                    else:
                        # 如果Q网格不可用，回退到像素模式
                        show_q_axis = False
                except Exception as e:
                    pass
                    show_q_axis = False
            
            if not show_q_axis:
                # 像素坐标模式
                im = ax.imshow(processed_data, cmap='viridis', aspect='equal', origin='lower', 
                              interpolation='nearest', vmin=vmin, vmax=vmax)
                # 设置像素坐标标签
                ax.set_xlabel('Pixels (Horizontal)')
                ax.set_ylabel('Pixels (Vertical)')
            
            # 如果提供了选择信息，绘制选择矩形
            if selection_info:
                bounds = selection_info.get('bounds', {})
                x_min = bounds.get('x_min', 0)
                x_max = bounds.get('x_max', 0)
                y_min = bounds.get('y_min', 0)
                y_max = bounds.get('y_max', 0)
                
                # 绘制选择矩形
                from matplotlib.patches import Rectangle
                selection_rect = Rectangle(
                    (x_min, y_min), 
                    x_max - x_min, 
                    y_max - y_min,
                    linewidth=2, edgecolor='red', facecolor='none', alpha=0.8
                )
                ax.add_patch(selection_rect)
                
                # 添加中心点标记
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                ax.plot(center_x, center_y, 'r+', markersize=10, markeredgewidth=2)
            
            if not show_q_axis:
                ax.axis('off')
            
            # 简化布局调整
            fig.tight_layout(pad=0.05)
            
            # 设置显示范围
            if show_q_axis:
                # Q模式：让matplotlib自动设置范围（由extent控制）
                ax.autoscale()
            else:
                # 像素模式：设置为像素坐标范围
                ax.set_xlim(-0.5, processed_data.shape[1] - 0.5)
                ax.set_ylim(-0.5, processed_data.shape[0] - 0.5)
            
            # 绘制canvas
            canvas.draw()
            
            # 添加到场景
            proxy_widget = self._graphics_scene.addWidget(canvas)
            
            # 调整视图以适配固定画布（无滚动条）
            self._fit_view_to_item(graphics_view, proxy_widget, keep_aspect=True)
            
            mode_text = "Log" if self._is_log_mode_enabled() else "Linear"
            coord_mode = "Q-space" if show_q_axis else "Pixel coordinates"
            selection_text = " with selection" if selection_info else ""
            self.status_updated.emit(f"{mode_text} image displayed ({coord_mode}){selection_text} (Double-click to open independent window)")
            
        except Exception as e:
            self.status_updated.emit(f"Display error: {str(e)}")
    
    # ========== 颜色标尺和显示模式控制 ==========
    
    def _calculate_vmin_vmax(self, image_data, use_log=True):
        """计算图像的Vmin和Vmax值（1%和99%分位数）"""
        try:
            if use_log:
                safe_data = np.where(image_data > 0, image_data, 0.001)
                log_data = np.log(safe_data)
                vmin = np.percentile(log_data, 1)
                vmax = np.percentile(log_data, 99)
            else:
                vmin = np.percentile(image_data, 1)
                vmax = np.percentile(image_data, 99)
            
            return vmin, vmax
        except Exception:
            return None, None
    
    def _update_vmin_vmax_ui(self, vmin, vmax):
        """更新UI中的Vmin和Vmax值"""
        try:
            if vmin is not None and vmax is not None:
                if hasattr(self.ui, 'gisaxsInputVminValue'):
                    self.ui.gisaxsInputVminValue.setValue(float(vmin))
                if hasattr(self.ui, 'gisaxsInputVmaxValue'):
                    self.ui.gisaxsInputVmaxValue.setValue(float(vmax))
                
                self._current_vmin = vmin
                self._current_vmax = vmax
                self._refresh_vmin_vmax_display()
        except Exception:
            pass
    
    def _get_vmin_vmax_from_ui(self):
        """从UI获取当前的Vmin和Vmax值"""
        try:
            vmin = None
            vmax = None
            
            if hasattr(self.ui, 'gisaxsInputVminValue'):
                vmin = self.ui.gisaxsInputVminValue.value()
            if hasattr(self.ui, 'gisaxsInputVmaxValue'):
                vmax = self.ui.gisaxsInputVmaxValue.value()
                
            return vmin, vmax
        except Exception:
            return None, None
    
    def _handle_color_scale(self, image_data):
        """处理颜色标尺逻辑"""
        try:
            is_auto_scale = self._is_auto_scale_enabled()
            is_first_image = not self._has_displayed_image
            is_log = self._is_log_mode_enabled()
            
            if is_first_image:
                # 第一次显示图像，无论AutoScale状态如何都自动计算
                vmin, vmax = self._calculate_vmin_vmax(image_data, use_log=is_log)
                if vmin is not None and vmax is not None:
                    self._update_vmin_vmax_ui(vmin, vmax)
                self._has_displayed_image = True
                
            elif is_auto_scale:
                # 不是第一次显示且AutoScale启用，重新计算
                vmin, vmax = self._calculate_vmin_vmax(image_data, use_log=is_log)
                if vmin is not None and vmax is not None:
                    self._update_vmin_vmax_ui(vmin, vmax)
                    
            else:
                # 不是第一次显示且AutoScale未启用，使用UI中现有的值
                vmin, vmax = self._get_vmin_vmax_from_ui()
                self._current_vmin = vmin
                self._current_vmax = vmax
                
        except Exception:
            # 如果出错，回退到自动计算
            try:
                is_log = self._is_log_mode_enabled()
                vmin, vmax = self._calculate_vmin_vmax(image_data, use_log=is_log)
                if vmin is not None and vmax is not None:
                    self._update_vmin_vmax_ui(vmin, vmax)
            except Exception:
                pass
    
    def _is_auto_scale_enabled(self):
        """检查AutoScale是否被启用"""
        return self._get_checkbox_state('gisaxsInputAutoScaleCheckBox', True)
    
    def _is_log_mode_enabled(self):
        """检查是否启用Log模式"""
        return self._get_checkbox_state('gisaxsInputIntLogCheckBox', True)
    
    def _on_auto_scale_changed(self):
        """AutoScale复选框状态改变时的处理"""
        try:
            is_enabled = self._is_auto_scale_enabled()
            self.status_updated.emit(f"AutoScale {'enabled' if is_enabled else 'disabled'}")
            
            if is_enabled and self.current_stack_data is not None:
                is_log = self._is_log_mode_enabled()
                vmin, vmax = self._calculate_vmin_vmax(self.current_stack_data, use_log=is_log)
                if vmin is not None and vmax is not None:
                    self._update_vmin_vmax_ui(vmin, vmax)
                    self._refresh_image_display()
        except Exception as e:
            self.status_updated.emit(f"AutoScale change error: {str(e)}")
    
    def _on_vmin_value_changed(self):
        """Vmin值改变时的处理"""
        try:
            vmin, vmax = self._get_vmin_vmax_from_ui()
            if vmin is not None:
                self._current_vmin = vmin
                if self.current_stack_data is not None:
                    self._refresh_image_display()
        except Exception:
            pass
    
    def _on_vmax_value_changed(self):
        """Vmax值改变时的处理"""
        try:
            vmin, vmax = self._get_vmin_vmax_from_ui()
            if vmax is not None:
                self._current_vmax = vmax
                if self.current_stack_data is not None:
                    self._refresh_image_display()
        except Exception:
            pass
    
    def _on_auto_show_changed(self):
        """AutoShow选项改变时的处理"""
        auto_show = hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked()
        self.status_updated.emit(f"AutoShow {'enabled' if auto_show else 'disabled'}")
        # In-situ 定时器控制
        try:
            if getattr(self, 'load_mode', 'Single') == 'In-situ':
                if auto_show:
                    self._start_insitu_timer()
                else:
                    self._stop_insitu_timer()
        except Exception:
            pass
    
    def _on_log_changed(self):
        """Log选项改变时的处理 - 支持线性/对数切换"""
        try:
            is_log = self._is_log_mode_enabled()
            
            # 刷新Vmin/Vmax控件的显示格式
            self._refresh_vmin_vmax_display()
            
            # 如果当前有图像数据，重新计算vmin/vmax并更新显示
            if self.current_stack_data is not None:
                if self._is_auto_scale_enabled():
                    vmin, vmax = self._calculate_vmin_vmax(self.current_stack_data, use_log=is_log)
                    if vmin is not None and vmax is not None:
                        self._update_vmin_vmax_ui(vmin, vmax)
                
                # 重新显示图像
                self._refresh_image_display()
                
            self.status_updated.emit(f"*** DISPLAY MODE CHANGED TO: {'LOG' if is_log else 'LINEAR'} ***")
                
        except Exception as e:
            self.status_updated.emit(f"Log mode change error: {str(e)}")
    
    def _on_fit_display_option_changed(self):
        """拟合显示选项改变时的处理"""
        try:
            # 如果正在初始化中，跳过处理
            if getattr(self, '_initializing', True):
                return
                
            # 检查fitCurrentDataCheckBox的状态
            if hasattr(self.ui, 'fitCurrentDataCheckBox') and self.ui.fitCurrentDataCheckBox.isChecked():
                # fitCurrentDataCheckBox被勾选时，重新执行cut操作
                self._perform_cut()
                self.status_updated.emit("Fit display options changed - Cut results updated")
            else:
                # fitCurrentDataCheckBox未勾选时，使用新的更新函数显示1D数据
                if self.current_1d_data is not None and hasattr(self, 'q') and self.q is not None:
                    mode = self.display_mode if hasattr(self, 'display_mode') else 'normal'
                    self._update_GUI_image(mode)
                    self._update_outside_window(mode)
                    self.status_updated.emit("Fit display options changed - 1D data display updated")
                else:
                    self.status_updated.emit("Fit display options changed - no data to update")
                
        except Exception as e:
            self.status_updated.emit(f"Fit display option change error: {str(e)}")
    
    def _on_current_data_checkbox_changed(self, checked):
        """当fitCurrentDataCheckBox状态改变时的处理"""
        try:
            # 如果正在初始化中，跳过处理
            if getattr(self, '_initializing', True):
                return
                
            if checked:
                # 勾选状态：显示Cut数据（如果有GISAXS数据的话）
                if self.current_stack_data is not None:
                    # 有GISAXS数据，执行Cut操作
                    self._perform_cut()
                    self.status_updated.emit("Current Data enabled - Cut operation performed")
                else:
                    # 没有GISAXS数据，显示提示
                    self.status_updated.emit("Current Data enabled - No GISAXS data available for cut operation")
            else:
                # 不勾选状态：恢复到1D数据显示（如果有的话）
                if self.current_1d_data is not None:
                    # 恢复1D数据到q,I存储器
                    self.q = self.current_1d_data['q']
                    self.I = self.current_1d_data['I']
                    self.data_source = '1d'
                    self.display_mode = 'normal'
                    
                    # 强制更新显示
                    self._update_GUI_image('normal')
                    self._update_outside_window('normal')
                    self.status_updated.emit("Current Data disabled - 1D data restored")
                else:
                    # 没有1D数据，清空显示区域
                    self._clear_fit_graphics_view()
                    # 同时清空外部窗口（如果存在）
                    if hasattr(self, 'independent_fit_window') and self.independent_fit_window is not None and self.independent_fit_window.isVisible():
                        self.independent_fit_window.ax.clear()
                        self.independent_fit_window.canvas.draw()
                    self.status_updated.emit("Current Data disabled - No 1D data available")
                    
        except Exception as e:
            self.status_updated.emit(f"Current Data checkbox change error: {str(e)}")
    
    # ========== 拟合相关方法 ==========
    
    def _import_1d_file(self):
        """导入1D数据文件（.dat或.txt格式）"""
        try:
            # 从会话数据获取上次使用的目录
            from core.global_params import global_params
            fitting_session = global_params.get_parameter('fitting', 'last_session', {})
            last_1d_directory = fitting_session.get('last_1d_directory')
            
            # 确定起始目录
            if last_1d_directory and os.path.exists(last_1d_directory):
                start_directory = last_1d_directory
            else:
                start_directory = os.getcwd()  # 当前工作目录
            
            # 打开文件选择对话框
            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window,
                "Select 1D SAXS Data File",
                start_directory,
                "Data Files (*.dat *.txt);;All Files (*)"
            )
            
            # 如果用户取消了选择
            if not file_path:
                return
                
            # 保存文件路径用于下次打开
            self.current_1d_file_path = file_path
            
            # 保存目录到会话数据
            current_directory = os.path.dirname(file_path)
            fitting_session['last_1d_directory'] = current_directory
            global_params.set_parameter('fitting', 'last_session', fitting_session)
            
            # 加载数据
            self._load_1d_data(file_path)
            
        except Exception as e:
            self.status_updated.emit(f"Failed to import 1D file: {str(e)}")
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"Failed to import 1D file:\n{str(e)}"
            )
    
    def _on_1d_file_value_changed(self):
        """当fitImport1dFileValue文本框回车时的处理"""
        try:
            if not hasattr(self.ui, 'fitImport1dFileValue'):
                return
                
            file_path_input = self.ui.fitImport1dFileValue.text().strip()
            
            if not file_path_input:
                self.status_updated.emit("No file path entered")
                return
            
            # 导入global_params
            from core.global_params import global_params
            
            # 如果输入的不是绝对路径，尝试使用当前目录或之前的目录
            if not os.path.isabs(file_path_input):
                # 从会话数据获取上次使用的目录
                fitting_session = global_params.get_parameter('fitting', 'last_session', {})
                last_1d_directory = fitting_session.get('last_1d_directory')
                
                if last_1d_directory and os.path.exists(last_1d_directory):
                    file_path_input = os.path.join(last_1d_directory, file_path_input)
                else:
                    file_path_input = os.path.join(os.getcwd(), file_path_input)
            
            # 验证文件是否存在
            if not os.path.exists(file_path_input):
                QMessageBox.warning(
                    self.main_window,
                    "File Not Found",
                    f"File does not exist:\n{file_path_input}"
                )
                return
            
            # 验证文件扩展名
            file_ext = os.path.splitext(file_path_input)[1].lower()
            if file_ext not in ['.dat', '.txt']:
                QMessageBox.warning(
                    self.main_window,
                    "Invalid File Type",
                    f"Only .dat and .txt files are supported.\nSelected: {file_ext}"
                )
                return
            
            # 更新文本框显示完整路径
            self.ui.fitImport1dFileValue.setText(file_path_input)
            
            # 保存文件路径
            self.current_1d_file_path = file_path_input
            
            # 保存目录到会话数据
            current_directory = os.path.dirname(file_path_input)
            fitting_session = global_params.get_parameter('fitting', 'last_session', {})
            fitting_session['last_1d_directory'] = current_directory
            global_params.set_parameter('fitting', 'last_session', fitting_session)
            
            # 加载数据
            self._load_1d_data(file_path_input)
            
        except Exception as e:
            self.status_updated.emit(f"Failed to process 1D file path: {str(e)}")
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"Failed to process 1D file path:\n{str(e)}"
            )
    
    def _load_1d_data(self, file_path):
        """导入1D文件 - 数据来源于1D图（1D模式）"""
        try:
            # 导入加载函数
            from utils.load_SAXS_data import load_xy_any
            
            # 加载数据
            self.status_updated.emit(f"Loading 1D data from {os.path.basename(file_path)}...")
            data = load_xy_any(file_path)
            
            # 导入数据到 q,I 存储器
            self.q = data.q
            self.I = data.I
            
            # 存储原始数据用于兼容性
            self.current_1d_data = {
                'q': data.q,
                'I': data.I,
                'err': getattr(data, 'err', None) if hasattr(data, 'err') else None,
                'file_path': file_path
            }
            
            # 设置为1D模式，切换显示模式为normal
            self.data_source = '1d'
            self.display_mode = 'normal'
            if hasattr(self.ui, 'fitCurrentDataCheckBox'):
                self.ui.fitCurrentDataCheckBox.blockSignals(True)
                self.ui.fitCurrentDataCheckBox.setChecked(False)
                self.ui.fitCurrentDataCheckBox.blockSignals(False)
            
            # 更新fitImport1dFileValue显示
            if hasattr(self.ui, 'fitImport1dFileValue'):
                self.ui.fitImport1dFileValue.setText(file_path)
            
            # 初始化并应用ROI（重置为全范围）
            try:
                self._initialize_roi_from_current_q(force_full=True)
            except Exception:
                pass
            self._apply_roi_to_data_and_refresh()
            # 更新显示
            self._update_GUI_image('normal')
            self._update_outside_window('normal')
            
            self.status_updated.emit(f"Successfully loaded 1D data: {os.path.basename(file_path)} ({len(self.q)} points)")
            
        except Exception as e:
            self.status_updated.emit(f"Failed to load 1D data: {str(e)}")
            QMessageBox.critical(self.main_window, "Error", f"Failed to load 1D data from {os.path.basename(file_path)}:\n{str(e)}")
    

    
    def _setup_fit_graphics_scene(self):
        """统一的fitGraphicsView场景设置方法"""
        try:
            if not hasattr(self.ui, 'fitGraphicsView'):
                return None
                
            # 创建或复用场景
            if not hasattr(self, '_fit_graphics_scene') or self._fit_graphics_scene is None:
                self._fit_graphics_scene = QGraphicsScene()
                self.ui.fitGraphicsView.setScene(self._fit_graphics_scene)
                # Configure the view for a fixed-size, scroll-less canvas
                try:
                    from PyQt5.QtWidgets import QGraphicsView, QFrame
                    view = self.ui.fitGraphicsView
                    view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                    view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                    view.setDragMode(QGraphicsView.NoDrag)
                    view.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
                    view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
                    view.setInteractive(False)
                    view.setFrameShape(QFrame.NoFrame)
                    from PyQt5.QtGui import QPainter
                    view.setRenderHint(QPainter.Antialiasing, False)
                    view.setRenderHint(QPainter.SmoothPixmapTransform, True)
                    view.setRenderHint(QPainter.TextAntialiasing, True)
                except Exception:
                    pass
            else:
                # 清空现有内容但保持场景
                self._fit_graphics_scene.clear()
                
            return self._fit_graphics_scene
            
        except Exception as e:
            self.status_updated.emit(f"Failed to setup fit graphics scene: {str(e)}")
            return None

    def _fit_view_to_item(self, graphics_view, item, keep_aspect=True):
        """Fit the view to the given item bounds; disable scrollbars by sizing the scene to the item."""
        try:
            scene = graphics_view.scene()
            if scene is None or item is None:
                return
            scene.setSceneRect(item.sceneBoundingRect())
            if keep_aspect:
                graphics_view.fitInView(item, Qt.KeepAspectRatio)
            else:
                graphics_view.fitInView(item)
            graphics_view.update()
        except Exception:
            pass
    
    def _clear_fit_graphics_view(self):
        """清空fitGraphicsView显示区域"""
        try:
            if not hasattr(self.ui, 'fitGraphicsView'):
                return
                
            scene = self._setup_fit_graphics_scene()
            if scene is not None:
                scene.clear()
                
            self.status_updated.emit("Fit graphics view cleared")
            
        except Exception as e:
            self.status_updated.emit(f"Failed to clear fit graphics view: {str(e)}")
        
    def _start_fitting(self):
        """开始拟合"""
        if not self.current_parameters.get('imported_gisaxs_file'):
            QMessageBox.warning(
                self.parent,
                "警告", 
                "请先选择GISAXS文件！"
            )
            return
            
        try:
            self.status_updated.emit("Start Cut Fitting Processing...")
            self.progress_updated.emit(0)
            
            # TODO: 实现实际的拟合逻辑
            self._run_fitting_process()
            
            self.progress_updated.emit(100)
            self.status_updated.emit("Cut Fitting processing complete!")
            
        except Exception as e:
            QMessageBox.critical(
                self.parent,
                "错误",
                f"Cut Fitting处理时出错:\n{str(e)}"
            )
            
    def _run_fitting_process(self):
        """运行拟合过程"""
        # TODO: 实现具体的拟合算法
        # 这里应该包含：
        # 1. 数据裁剪
        # 2. 预处理
        # 3. 拟合计算
        # 4. 结果输出
        pass
        
    def _reset_fitting(self):
        """重置拟合参数"""
        self._set_default_parameters()
        
    def _auto_find_center(self):
        """自动寻找GISAXS图像的中心点"""
        # 先同步UI控件的最新值到参数系统
        self._sync_ui_to_parameters()
        
        # 检查是否已经导入了图像数据
        if self.current_stack_data is None:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "Please import an image first."
            )
            return
        
        # 检查是否已经显示了图像
        if not self._has_displayed_image:
            QMessageBox.warning(
                self.main_window,
                "Warning", 
                "Please display the image first by clicking the Show button."
            )
            return
            
        try:
            self.status_updated.emit("Searching for the center point automatically...")
            
            # 使用对数强度进行计算
            data = np.log10(np.maximum(self.current_stack_data, 1))
            
            # 1. 纵向中心(center_y): 沿横向累加，找光强最强的地方
            vertical_profile = np.sum(data, axis=1)  # 沿横向累加
            raw_center_y = np.argmax(vertical_profile)  # 找最强位置
            # 由于显示时图像被flipud，需要转换坐标
            height = data.shape[0]
            pixel_center_y = float(height - 1 - raw_center_y)
            
            # 2. 横向中心(center_x): 沿纵向累加，找光强重心（对称位置）
            horizontal_profile = np.sum(data, axis=0)  # 沿纵向累加
            pixel_center_x = float(np.sum(np.arange(len(horizontal_profile)) * horizontal_profile) / np.sum(horizontal_profile))
            
            # 3. 高度固定为20pixel
            pixel_cutline_height = 20.0
            
            # 4. 宽度: 找到可以覆盖横向强度95%的区域宽度
            pixel_cutline_width = self._calculate_95_percent_width(horizontal_profile)
            
            # 检查是否需要转换为Q坐标
            show_q_axis = self._should_show_q_axis()
            
            if show_q_axis:
                # Q模式：需要将像素坐标转换为Q坐标
                try:
                    # 使用pixel_to_q_space方法进行坐标转换
                    from utils.q_space_calculator import create_detector_from_image_and_params
                    from core.global_params import GlobalParameterManager
                    
                    global_params = GlobalParameterManager()
                    height, width = self.current_stack_data.shape
                    
                    # 获取探测器参数
                    pixel_size_x = global_params.get_parameter('fitting', 'detector.pixel_size_x', 172.0)
                    pixel_size_y = global_params.get_parameter('fitting', 'detector.pixel_size_y', 172.0)
                    beam_center_x = global_params.get_parameter('fitting', 'detector.beam_center_x', width / 2.0)
                    beam_center_y = global_params.get_parameter('fitting', 'detector.beam_center_y', height / 2.0)
                    distance = global_params.get_parameter('fitting', 'detector.distance', 2565.0)
                    theta_in_deg = global_params.get_parameter('beam', 'grazing_angle', 0.4)
                    wavelength = global_params.get_parameter('beam', 'wavelength', 0.1045)
                    
                    # 创建探测器对象
                    detector = create_detector_from_image_and_params(
                        image_shape=(height, width),
                        pixel_size_x=pixel_size_x,
                        pixel_size_y=pixel_size_y,
                        beam_center_x=beam_center_x,
                        beam_center_y=beam_center_y,
                        distance=distance,
                        theta_in_deg=theta_in_deg,
                        wavelength=wavelength,
                        crop_params=None
                    )
                    
                    # 转换中心点坐标 - pixel_to_q_space返回(qx, qy, qz)三个值
                    center_qx, center_qy, center_qz = detector.pixel_to_q_space(pixel_center_x, pixel_center_y)
                    
                    # 转换尺寸：计算边界点的Q坐标差值
                    left_qx, left_qy, _ = detector.pixel_to_q_space(pixel_center_x - pixel_cutline_width/2, pixel_center_y)
                    right_qx, right_qy, _ = detector.pixel_to_q_space(pixel_center_x + pixel_cutline_width/2, pixel_center_y)
                    _, _, bottom_qz = detector.pixel_to_q_space(pixel_center_x, pixel_center_y - pixel_cutline_height/2)
                    _, _, top_qz = detector.pixel_to_q_space(pixel_center_x, pixel_center_y + pixel_cutline_height/2)
                    
                    cutline_width_q = abs(right_qy - left_qy)
                    cutline_height_q = abs(top_qz - bottom_qz)
                    
                    # 更新UI控件为Q坐标
                    if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                        self.ui.gisaxsInputCenterVerticalValue.setValue(center_qz)  # Vertical对应qz
                    if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                        self.ui.gisaxsInputCenterParallelValue.setValue(center_qy)  # Parallel对应qy
                    if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                        self.ui.gisaxsInputCutLineVerticalValue.setValue(cutline_height_q)
                    if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                        self.ui.gisaxsInputCutLineParallelValue.setValue(cutline_width_q)
                    
                    self.status_updated.emit(f"Auto-search completion (Q-coordinate): Center({center_qy:.6f}, {center_qz:.6f}) nm⁻¹, CutLine({cutline_width_q:.6f}, {cutline_height_q:.6f}) nm⁻¹")
                    
                except Exception as e:
                    self.status_updated.emit(f"Q-coordinate conversion failed, using pixel coordinates: {str(e)}")
                    # 回退到像素坐标模式
                    show_q_axis = False
                    
            if not show_q_axis:
                # 像素模式：直接使用像素坐标
                if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                    self.ui.gisaxsInputCenterVerticalValue.setValue(pixel_center_y)
                if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                    self.ui.gisaxsInputCenterParallelValue.setValue(pixel_center_x)
                if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                    self.ui.gisaxsInputCutLineVerticalValue.setValue(pixel_cutline_height)
                if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                    self.ui.gisaxsInputCutLineParallelValue.setValue(pixel_cutline_width)
                
                self.status_updated.emit(f"Auto Find Complete (pixel coordinates): Center({pixel_center_x:.1f}, {pixel_center_y:.1f}), CutLine({pixel_cutline_width:.1f}, {pixel_cutline_height:.1f})")
            
            # 强制刷新UI显示
            if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                self.ui.gisaxsInputCenterVerticalValue.update()
            if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                self.ui.gisaxsInputCenterParallelValue.update()
            if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                self.ui.gisaxsInputCutLineVerticalValue.update()
            if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                self.ui.gisaxsInputCutLineParallelValue.update()
                
            # 手动触发参数选择显示更新
            self._on_cutline_parameters_changed()
                
        except Exception as e:
            self.status_updated.emit(f"Auto Find Center Failure: {str(e)}")
    
    def _calculate_95_percent_width(self, profile):
        """计算包含95%强度的宽度"""
        if len(profile) == 0:
            return 50.0
            
        # 计算总强度
        total_intensity = np.sum(profile)
        if total_intensity == 0:
            return 50.0
        
        # 找到峰值位置
        center_idx = np.argmax(profile)
        
        # 计算95%强度的目标值
        target_intensity = total_intensity * 0.95
        
        # 从中心点开始向两边扩展，直到累积强度达到95%
        left_idx = center_idx
        right_idx = center_idx
        current_intensity = profile[center_idx]
        
        # 交替向左右扩展
        while current_intensity < target_intensity and (left_idx > 0 or right_idx < len(profile) - 1):
            # 决定向哪边扩展（选择强度更高的一边）
            left_val = profile[left_idx - 1] if left_idx > 0 else 0
            right_val = profile[right_idx + 1] if right_idx < len(profile) - 1 else 0
            
            if left_val >= right_val and left_idx > 0:
                left_idx -= 1
                current_intensity += profile[left_idx]
            elif right_idx < len(profile) - 1:
                right_idx += 1
                current_intensity += profile[right_idx]
            else:
                break
        
        # 计算宽度
        width = right_idx - left_idx + 1
        
        # 确保有合理的范围
        min_width = 20.0
        max_width = len(profile) * 0.8
        
        return float(max(min_width, min(width, max_width)))
            
    def _show_detector_parameters(self):
        """显示探测器参数对话框"""
        try:
            # 如果对话框已经存在且可见，则将其置于前台
            if hasattr(self, 'detector_params_dialog') and self.detector_params_dialog is not None:
                if self.detector_params_dialog.isVisible():
                    self.detector_params_dialog.raise_()
                    self.detector_params_dialog.activateWindow()
                    return
            
            # 创建非模态对话框
            self.detector_params_dialog = DetectorParametersDialog(self.main_window)
            
            # 连接参数改变信号
            self.detector_params_dialog.parameters_changed.connect(self._on_detector_parameters_changed)
            
            # 连接对话框关闭信号，用于清理资源
            self.detector_params_dialog.finished.connect(self._on_detector_dialog_finished)
            
            # 显示非模态对话框
            self.detector_params_dialog.show()
            self.detector_params_dialog.raise_()
            self.detector_params_dialog.activateWindow()
            
            self.status_updated.emit("Detector Parameters dialog box is open - main interface parameters can be modified at the same time")
            
        except Exception as e:
            self.status_updated.emit(f"Failed to display Detector Parameters dialog box: {str(e)}")
            QMessageBox.critical(
                self.main_window,
                "error",
                f"Detector Parameters dialog box cannot be displayed: {str(e)}"
            )
            
    def _on_detector_dialog_finished(self):
        """探测器参数对话框关闭时的清理"""
        try:
            self.detector_params_dialog = None
            self.status_updated.emit("The Detector Parameters dialog box is closed")
        except Exception as e:
            self.status_updated.emit(f"Failed to clear detector dialog: {str(e)}")
            
    def _on_detector_parameters_changed(self, parameters):
        """处理探测器参数改变"""
        try:
            # 更新Cut Line标签的单位（因为show_q_axis可能改变了）
            self._update_cutline_labels_units()
            
            # 更新Cut Line参数的步长（因为显示模式可能改变了）
            self._update_cutline_step_sizes()
            
            # 如果Q轴显示状态改变，需要转换现有的数值
            self._update_parameter_values_for_q_axis()
            # 探测器相关参数修改后，重计算Q空间网格缓存
            try:
                self._compute_q_meshgrids_and_store()
            except Exception:
                pass
            
            # 检查是否已经有Cut结果数据，如果有则自动重新执行Cut操作
            if (self.current_cut_data is not None and 
                hasattr(self, 'current_stack_data') and self.current_stack_data is not None):
                
                # 自动重新执行Cut操作
                self._perform_cut()
                self.status_updated.emit("Detector parameters have been updated, Cut results have been automatically recalculated")
            else:
                self.status_updated.emit("Detector parameters updated and saved")
            
        except Exception as e:
            self.status_updated.emit(f"Failure to process detector parameter change: {str(e)}")
    
    def _update_parameter_values_for_q_axis(self):
        """根据Q轴显示状态切换时转换参数数值并更新显示"""
        try:
            show_q_axis = self._should_show_q_axis()
            
            # 如果是第一次调用，直接设置当前模式但不进行转换
            if not hasattr(self, '_last_q_mode') or self._last_q_mode is None:
                self._last_q_mode = show_q_axis
                self.status_updated.emit(f"Q-axis display mode is set: {'Q coordinate' if show_q_axis else 'Pixel coordinate'}")
                return
            
            # 获取当前的数值
            if not hasattr(self.ui, 'gisaxsInputCenterVerticalValue') or not hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                return
                
            center_vertical = self.ui.gisaxsInputCenterVerticalValue.value()
            center_parallel = self.ui.gisaxsInputCenterParallelValue.value()
            
            cutline_vertical = 0
            cutline_parallel = 0
            if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                cutline_vertical = self.ui.gisaxsInputCutLineVerticalValue.value()
            if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                cutline_parallel = self.ui.gisaxsInputCutLineParallelValue.value()
            

            
            # 检查是否有图像数据用于转换
            if self.current_stack_data is None:
                self.status_updated.emit("No image data, no coordinate conversion possible")
                return
            
            # 创建detector用于坐标转换
            detector = self._get_detector_for_conversion()
            if detector is None:
                self.status_updated.emit("Unable to create detector, coordinate transformation failed")
                return
            
            # 检查当前参数是否已经是目标模式（避免重复转换）
            if self._last_q_mode == show_q_axis:
                return
            
            # 1. 转换Cut line和Center的四个数值
            if show_q_axis:
                # Pixel -> Q-space转换

                new_center_parallel, new_center_vertical, new_cutline_parallel, new_cutline_vertical = \
                    self._convert_pixel_to_q_parameters(center_parallel, center_vertical, cutline_parallel, cutline_vertical, detector)
                conversion_msg = "Parameters have been converted from pixel coordinates to Q-space coordinates"
            else:
                # Q-space -> Pixel转换

                new_center_parallel, new_center_vertical, new_cutline_parallel, new_cutline_vertical = \
                    self._convert_q_to_pixel_parameters(center_parallel, center_vertical, cutline_parallel, cutline_vertical, detector)
                conversion_msg = "Parameters have been converted from Q-space coordinates to pixel coordinates"
            
            # 更新UI控件的值（暂时断开信号连接避免循环触发）
            self._temporarily_disconnect_parameter_signals()
            
            self.ui.gisaxsInputCenterVerticalValue.setValue(new_center_vertical)
            self.ui.gisaxsInputCenterParallelValue.setValue(new_center_parallel)
            if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                self.ui.gisaxsInputCutLineVerticalValue.setValue(new_cutline_vertical)
            if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                self.ui.gisaxsInputCutLineParallelValue.setValue(new_cutline_parallel)
            
            self._reconnect_parameter_signals()
            
            # 2. 如果有二维图显示，更新二维图和外置窗口
            if hasattr(self, 'current_stack_data') and self.current_stack_data is not None:
                self._refresh_display_for_mode_change()
            
            # 3. 如果有cut data，更新一维Cut图
            if self.current_cut_data is not None:
                self._perform_cut()
                
            # 记录当前模式，避免重复转换
            self._last_q_mode = show_q_axis
            
            self.status_updated.emit(conversion_msg)
                
        except Exception as e:
            self.status_updated.emit(f"Mode switching and parameter conversion failure: {str(e)}")
    
    def _get_detector_for_conversion(self):
        """获取用于坐标转换的detector对象"""
        try:
            from utils.q_space_calculator import create_detector_from_image_and_params
            from core.global_params import GlobalParameterManager
            
            global_params = GlobalParameterManager()
            height, width = self.current_stack_data.shape
            
            # 获取探测器参数
            pixel_size_x = global_params.get_parameter('fitting', 'detector.pixel_size_x', 172.0)
            pixel_size_y = global_params.get_parameter('fitting', 'detector.pixel_size_y', 172.0)
            beam_center_x = global_params.get_parameter('fitting', 'detector.beam_center_x', width / 2.0)
            beam_center_y = global_params.get_parameter('fitting', 'detector.beam_center_y', height / 2.0)
            distance = global_params.get_parameter('fitting', 'detector.distance', 2565.0)
            theta_in_deg = global_params.get_parameter('beam', 'grazing_angle', 0.4)
            wavelength = global_params.get_parameter('beam', 'wavelength', 0.1045)
            
            detector = create_detector_from_image_and_params(
                image_shape=(height, width),
                pixel_size_x=pixel_size_x,
                pixel_size_y=pixel_size_y,
                beam_center_x=beam_center_x,
                beam_center_y=beam_center_y,
                distance=distance,
                theta_in_deg=theta_in_deg,
                wavelength=wavelength
            )
            
            return detector
            
        except Exception as e:
            self.status_updated.emit(f"Failed to create detector: {str(e)}")
            return None
    
    def _convert_pixel_to_q_parameters(self, pixel_parallel, pixel_vertical, pixel_cutline_parallel, pixel_cutline_vertical, detector):
        """将像素参数转换为Q空间参数"""
        try:
            # 转换中心点坐标
            center_qx, center_qy, center_qz = detector.pixel_to_q_space(pixel_parallel, pixel_vertical)
            
            # 转换Cut Line尺寸
            # Parallel方向（X方向）的尺寸
            left_qx, left_qy, _ = detector.pixel_to_q_space(pixel_parallel - pixel_cutline_parallel/2, pixel_vertical)
            right_qx, right_qy, _ = detector.pixel_to_q_space(pixel_parallel + pixel_cutline_parallel/2, pixel_vertical)
            q_cutline_parallel = abs(right_qy - left_qy)
            
            # Vertical方向（Y方向）的尺寸
            _, _, bottom_qz = detector.pixel_to_q_space(pixel_parallel, pixel_vertical - pixel_cutline_vertical/2)
            _, _, top_qz = detector.pixel_to_q_space(pixel_parallel, pixel_vertical + pixel_cutline_vertical/2)
            q_cutline_vertical = abs(top_qz - bottom_qz)
            
            return center_qy, center_qz, q_cutline_parallel, q_cutline_vertical
            
        except Exception as e:
            self.status_updated.emit(f"Pixel to Q-space conversion failed: {str(e)}")
            return pixel_parallel, pixel_vertical, pixel_cutline_parallel, pixel_cutline_vertical
    
    def _convert_q_to_pixel_parameters(self, q_parallel, q_vertical, q_cutline_parallel, q_cutline_vertical, detector):
        """将Q空间参数转换为像素参数"""
        try:
            # 转换中心点坐标 - 假设qx=0（在平面内）
            center_x, center_y = detector.q_to_pixel_space(0, q_parallel, q_vertical)
            
            # 转换Cut Line尺寸
            # Parallel方向：qy方向的变化
            left_x, _ = detector.q_to_pixel_space(0, q_parallel - q_cutline_parallel/2, q_vertical)
            right_x, _ = detector.q_to_pixel_space(0, q_parallel + q_cutline_parallel/2, q_vertical)
            pixel_cutline_parallel = abs(right_x - left_x)
            
            # Vertical方向：qz方向的变化
            _, bottom_y = detector.q_to_pixel_space(0, q_parallel, q_vertical - q_cutline_vertical/2)
            _, top_y = detector.q_to_pixel_space(0, q_parallel, q_vertical + q_cutline_vertical/2)
            pixel_cutline_vertical = abs(top_y - bottom_y)
            
            return center_x, center_y, pixel_cutline_parallel, pixel_cutline_vertical
            
        except Exception as e:
            self.status_updated.emit(f"Q-space to pixel conversion failed: {str(e)}")
            return q_parallel, q_vertical, q_cutline_parallel, q_cutline_vertical
    
    def _temporarily_disconnect_parameter_signals(self):
        """暂时断开参数显示变化信号连接（坐标转换期间避免触发显示更新）"""
        try:
            if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                self.ui.gisaxsInputCenterVerticalValue.valueChanged.disconnect(self._on_parameter_display_changed)
            if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                self.ui.gisaxsInputCenterParallelValue.valueChanged.disconnect(self._on_parameter_display_changed)
            if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                self.ui.gisaxsInputCutLineVerticalValue.valueChanged.disconnect(self._on_parameter_display_changed)
            if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                self.ui.gisaxsInputCutLineParallelValue.valueChanged.disconnect(self._on_parameter_display_changed)
        except Exception:
            pass
    
    def _reconnect_parameter_signals(self):
        """重新连接参数显示变化信号"""
        try:
            if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                self.ui.gisaxsInputCenterVerticalValue.valueChanged.connect(self._on_parameter_display_changed)
            if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                self.ui.gisaxsInputCenterParallelValue.valueChanged.connect(self._on_parameter_display_changed)
            if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                self.ui.gisaxsInputCutLineVerticalValue.valueChanged.connect(self._on_parameter_display_changed)
            if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                self.ui.gisaxsInputCutLineParallelValue.valueChanged.connect(self._on_parameter_display_changed)
        except Exception:
            pass
    
    def _refresh_display_for_mode_change(self):
        """为模式切换刷新显示"""
        try:
            # 刷新主窗口显示
            if hasattr(self, '_show_image'):
                self._show_image()
            
            # 刷新独立窗口（如果存在）
            if self.independent_window is not None and self.independent_window.isVisible():
                self._show_independent_window()
                
        except Exception as e:
            self.status_updated.emit(f"Refresh display failure: {str(e)}")

    def convert_parameters_to_q_space(self, pixel_center_x, pixel_center_y, pixel_width, pixel_height):
        """将像素参数转换为Q空间参数
        
        Args:
            pixel_center_x: 像素中心X坐标 
            pixel_center_y: 像素中心Y坐标
            pixel_width: 像素宽度
            pixel_height: 像素高度
            
        Returns:
            tuple: (q_center_parallel, q_center_vertical, q_size_parallel, q_size_vertical)
        """
        try:
            from utils.q_space_calculator import Detector
            detector = Detector()
            
            # 转换中心点到Q空间
            qx_center, qy_center, qz_center = detector.pixel_to_q_space(pixel_center_x, pixel_center_y)
            
            # 计算Cut Line的Q空间尺寸
            # Vertical尺寸: Y方向的变化
            qx1, qy1, qz1 = detector.pixel_to_q_space(pixel_center_x, pixel_center_y + pixel_height/2)
            qx2, qy2, qz2 = detector.pixel_to_q_space(pixel_center_x, pixel_center_y - pixel_height/2)
            q_size_vertical = qz1 - qz2
            
            # Parallel尺寸: X方向的变化  
            qx3, qy3, qz3 = detector.pixel_to_q_space(pixel_center_x + pixel_width/2, pixel_center_y)
            qx4, qy4, qz4 = detector.pixel_to_q_space(pixel_center_x - pixel_width/2, pixel_center_y)
            q_size_parallel = qy3 - qy4
            
            return qy_center, qz_center, q_size_parallel, q_size_vertical
            
        except Exception as e:
            self.status_updated.emit(f"Q-space conversion failed: {str(e)}")
            return pixel_center_x, pixel_center_y, pixel_width, pixel_height
            
    def _calculate_image_center(self):
        """计算图像中心（简单重心方法）"""
        if self.current_stack_data is None:
            return 500.0, 500.0  # 默认值
            
        try:
            # 使用对数强度进行重心计算
            data = np.log10(np.maximum(self.current_stack_data, 1))
            
            # 计算垂直重心（Y方向） - 纵向找峰值
            vertical_profile = np.sum(data, axis=1)  # 沿横向累加
            vertical_center = np.argmax(vertical_profile)  # 找最强位置
            
            # 计算平行重心（X方向） - 横向找对称位置
            horizontal_profile = np.sum(data, axis=0)  # 沿纵向累加
            parallel_center = np.sum(np.arange(len(horizontal_profile)) * horizontal_profile) / np.sum(horizontal_profile)
            
            return float(vertical_center), float(parallel_center)
            
        except Exception as e:
            return 500.0, 500.0

    # ========== 会话管理方法 ==========
    
    def get_session_data(self):
        """获取当前会话数据（供主控制器调用）"""
        try:
            session_data = {
                'last_opened_file': self.current_parameters.get('imported_gisaxs_file', ''),
                'last_directory': os.path.dirname(self.current_parameters.get('imported_gisaxs_file', '')) if self.current_parameters.get('imported_gisaxs_file') else '',
                'last_1d_directory': os.path.dirname(self.current_1d_file_path) if self.current_1d_file_path else None,
                'last_1d_file': self.current_1d_file_path,
                'stack_count': self.current_parameters.get('stack_count', 1),
                'load_mode': getattr(self, 'load_mode', 'Single'),
                'insitu_range': self.current_parameters.get('insitu_range', ''),
                'center_vertical': self.ui.gisaxsInputCenterVerticalValue.value() if hasattr(self.ui, 'gisaxsInputCenterVerticalValue') else 0.0,
                'center_parallel': self.ui.gisaxsInputCenterParallelValue.value() if hasattr(self.ui, 'gisaxsInputCenterParallelValue') else 0.0,
                'cutline_vertical': self.ui.gisaxsInputCutLineVerticalValue.value() if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue') else 10.0,
                'cutline_parallel': self.ui.gisaxsInputCutLineParallelValue.value() if hasattr(self.ui, 'gisaxsInputCutLineParallelValue') else 10.0,
                'auto_show': self.ui.gisaxsInputAutoShowCheckBox.isChecked() if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') else False,
                'log_mode': self.ui.gisaxsInputIntLogCheckBox.isChecked() if hasattr(self.ui, 'gisaxsInputIntLogCheckBox') else True,
                'auto_scale': self.ui.gisaxsInputAutoScaleCheckBox.isChecked() if hasattr(self.ui, 'gisaxsInputAutoScaleCheckBox') else True,
                'vmin': self._current_vmin if self._current_vmin is not None else 0.0,
                'vmax': self._current_vmax if self._current_vmax is not None else 0.0,
                # 添加拟合选项状态保存
                'fit_current_data': self.ui.fitCurrentDataCheckBox.isChecked() if hasattr(self.ui, 'fitCurrentDataCheckBox') else False,
                'fit_log_x': self.ui.fitLogXCheckBox.isChecked() if hasattr(self.ui, 'fitLogXCheckBox') else False,
                'fit_log_y': self.ui.fitLogYCheckBox.isChecked() if hasattr(self.ui, 'fitLogYCheckBox') else False,
                'fit_norm': self.ui.fitNormCheckBox.isChecked() if hasattr(self.ui, 'fitNormCheckBox') else False
            }
            return session_data
        except Exception as e:
            return {}

    def restore_session(self, session_data):
        """恢复会话状态（供主控制器调用）"""
        try:
            # 恢复文件路径
            last_file = session_data.get('last_opened_file', '')
            if last_file and os.path.exists(last_file):
                self.current_parameters['imported_gisaxs_file'] = last_file
                if hasattr(self.ui, 'gisaxsInputImportButtonValue'):
                    self.ui.gisaxsInputImportButtonValue.setText(os.path.basename(last_file))
                self.status_updated.emit(f"Restored last file: {os.path.basename(last_file)}")
            
            # 恢复加载模式
            mode = session_data.get('load_mode', getattr(self, 'load_mode', 'Single'))
            self.load_mode = mode
            if hasattr(self.ui, 'gisaxsInputModelCombox'):
                try:
                    # 更新下拉框选择（可能触发信号）
                    index = self.ui.gisaxsInputModelCombox.findText(mode)
                    if index >= 0:
                        self.ui.gisaxsInputModelCombox.setCurrentIndex(index)
                except Exception:
                    pass
            
            # 恢复Stack设置
            stack_count = session_data.get('stack_count', 1)
            if hasattr(self.ui, 'gisaxsInputStackValue'):
                # In-situ 模式优先恢复范围文本
                if mode == 'In-situ':
                    insitu_range = session_data.get('insitu_range', '')
                    if insitu_range:
                        self.ui.gisaxsInputStackValue.setText(str(insitu_range))
                    else:
                        self.ui.gisaxsInputStackValue.setText(str(stack_count))
                else:
                    self.ui.gisaxsInputStackValue.setText(str(stack_count))
            self.current_parameters['stack_count'] = stack_count
            if mode == 'In-situ':
                # 保存恢复的范围到当前参数，便于显示和后续逻辑
                self.current_parameters['insitu_range'] = session_data.get('insitu_range', '')
            
            # 恢复Center参数
            if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                self.ui.gisaxsInputCenterVerticalValue.setValue(session_data.get('center_vertical', 0.0))
            if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                self.ui.gisaxsInputCenterParallelValue.setValue(session_data.get('center_parallel', 0.0))
            
            # 恢复Cut Line参数
            if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                self.ui.gisaxsInputCutLineVerticalValue.setValue(session_data.get('cutline_vertical', 10.0))
            if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                self.ui.gisaxsInputCutLineParallelValue.setValue(session_data.get('cutline_parallel', 10.0))
            
            # 恢复显示设置
            if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox'):
                self.ui.gisaxsInputAutoShowCheckBox.setChecked(session_data.get('auto_show', False))
            if hasattr(self.ui, 'gisaxsInputIntLogCheckBox'):
                self.ui.gisaxsInputIntLogCheckBox.setChecked(session_data.get('log_mode', True))
            if hasattr(self.ui, 'gisaxsInputAutoScaleCheckBox'):
                self.ui.gisaxsInputAutoScaleCheckBox.setChecked(session_data.get('auto_scale', True))
            
            # 恢复Vmin/Vmax
            vmin = session_data.get('vmin', 0.0)
            vmax = session_data.get('vmax', 0.0)
            if hasattr(self.ui, 'gisaxsInputVminValue'):
                self.ui.gisaxsInputVminValue.setValue(vmin)
            if hasattr(self.ui, 'gisaxsInputVmaxValue'):
                self.ui.gisaxsInputVmaxValue.setValue(vmax)
            
            # 恢复拟合选项状态（阻塞信号避免触发方法调用）
            self._restore_fit_checkboxes(session_data)
            
            # 恢复1D文件导入路径（用于文件浏览器默认目录）
            last_1d_directory = session_data.get('last_1d_directory')
            if last_1d_directory and os.path.exists(last_1d_directory):
                # 这里只是保存路径，不需要实际加载文件
                pass
            
            # 恢复1D文件路径显示（如果存在的话）
            last_1d_file = session_data.get('last_1d_file')
            if last_1d_file and os.path.exists(last_1d_file):
                self.current_1d_file_path = last_1d_file
                if hasattr(self.ui, 'fitImport1dFileValue'):
                    self.ui.fitImport1dFileValue.setText(last_1d_file)
                
                # 如果需要，可以选择重新加载1D数据（但这里我们只恢复路径显示）
                # 实际的数据加载由用户手动触发
            
            # 更新显示
            self._update_stack_display()
            # 根据模式调整输入框可见性
            if hasattr(self.ui, 'gisaxsInputStackValue'):
                try:
                    self.ui.gisaxsInputStackValue.setVisible(self.load_mode != 'Single')
                except Exception:
                    pass
            
            # 重新初始化Q模式状态（避免恢复后误触发转换）
            self._initialize_q_mode_state()
            
            # 如果AutoShow启用且有文件，自动显示
            if (session_data.get('auto_show', False) and 
                last_file and os.path.exists(last_file)):
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(500, self._show_image)  # 延迟500ms显示图像
            
            self.status_updated.emit("Previous session restored successfully")
            
        except Exception as e:
            self.status_updated.emit(f"Failed to restore session: {str(e)}")
    
    # ========== Cut功能相关方法 ==========
    
    def _perform_cut(self, points_override: int = None):
        """执行Cut操作 - 数据来自二维图（Cut模式）"""
        try:
            # 1. 检查是否导入了图像数据
            if self.current_stack_data is None:
                QMessageBox.warning(self.main_window, "Warning", "Please import an image first.")
                return
            
            # 2. 获取Cut Line参数
            vertical_value = 0.0
            parallel_value = 0.0
            if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                vertical_value = self.ui.gisaxsInputCutLineVerticalValue.value()
            if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                parallel_value = self.ui.gisaxsInputCutLineParallelValue.value()
            
            # 3. 检查参数是否为正数
            if vertical_value <= 0 or parallel_value <= 0:
                QMessageBox.warning(self.main_window, "Warning", "Please select a valid region.")
                return
            
            # 4. 决定此次Cut的目标点数，并执行切割导入到 q,I 存储器
            # 在解析点数之前，强制提交 DataPoints 文本框的当前编辑值（无需按回车）
            try:
                if hasattr(self.ui, 'fitDataPointsNumValue'):
                    # QAbstractSpinBox 方法：将编辑中的文本解析到值
                    self.ui.fitDataPointsNumValue.interpretText()
            except Exception:
                pass
            # （调试日志已移除）
            def _resolve_target_points():
                # 统一解析优先级：显式覆盖 > UI 当前值 > 稳定缓存 > 配置/默认
                if points_override is not None:
                    try:
                        _n = max(10, int(points_override))
                        return _n
                    except Exception:
                        pass
                # UI value
                try:
                    if hasattr(self.ui, 'fitDataPointsNumValue'):
                        w = self.ui.fitDataPointsNumValue
                        n_ui = None
                        # 优先尝试 SpinBox.value()
                        try:
                            n_ui = int(w.value())
                        except Exception:
                            # 若不是 SpinBox，则回退解析 text()
                            try:
                                if hasattr(w, 'text'):
                                    txt = w.text()
                                    if txt is not None and str(txt).strip() != '':
                                        # 支持整数/浮点文本
                                        n_ui = int(float(str(txt).strip()))
                            except Exception:
                                n_ui = None
                        if isinstance(n_ui, int) and n_ui >= 10:
                            return n_ui
                except Exception:
                    pass
                # stable cache
                n = getattr(self, '_points_num_current', None)
                if isinstance(n, (int, float)):
                    try:
                        n = int(n)
                        if n >= 10:
                            return n
                    except Exception:
                        pass
                # global params (if available)
                try:
                    from core.global_params import global_params as _gp
                    _gp_val = _gp.get_parameter('fitting', 'fit.points_num', None)
                    if _gp_val is not None:
                        _gv = int(_gp_val)
                        if _gv >= 10:
                            return _gv
                except Exception:
                    pass
                # user settings / default（统一 >=10）
                try:
                    from core.user_settings import user_settings
                    n_cfg = int(user_settings.get('fit.points_num', self._points_num_default))
                    if n_cfg >= 10:
                        return n_cfg
                except Exception:
                    pass
                _def_n = max(10, int(self._points_num_default))
                return _def_n

            n_points_cut = _resolve_target_points()
            # 最终保护：若UI显示一个有效(>=10)且与决议值不同的点数，则以UI为准（静默覆盖）
            try:
                if hasattr(self.ui, 'fitDataPointsNumValue'):
                    try:
                        _ui_now = int(self.ui.fitDataPointsNumValue.value())
                    except Exception:
                        _ui_now = None
                    if isinstance(_ui_now, int) and _ui_now >= 10 and _ui_now != int(n_points_cut):
                        n_points_cut = int(_ui_now)
            except Exception:
                pass
            # 每次实际Cut都同步稳定缓存，防止后续再次退化
            try:
                self._points_num_current = int(n_points_cut)
            except Exception:
                pass
            try:
                self.status_updated.emit(f"DataPoints resolved for Cut: {n_points_cut} (UI={getattr(self.ui.fitDataPointsNumValue,'value',lambda:None)() if hasattr(self.ui,'fitDataPointsNumValue') else 'NA'}, current={getattr(self,'_points_num_current', 'NA')})")
            except Exception:
                pass

            if vertical_value <= parallel_value:
                self._perform_horizontal_cut(vertical_value, parallel_value, points_override=n_points_cut)
                self.status_updated.emit(f"Horizontal cut performed: Vertical={vertical_value:.2f}, Parallel={parallel_value:.2f}, Points={n_points_cut}")
            else:
                self._perform_vertical_cut(vertical_value, parallel_value, points_override=n_points_cut)
                self.status_updated.emit(f"Vertical cut performed: Vertical={vertical_value:.2f}, Parallel={parallel_value:.2f}, Points={n_points_cut}")
            
            # 5. 设置为Cut模式，切换显示模式为 normal，并显式退出拟合模式
            self.data_source = 'cut'
            # 统一切换到 Normal 模式，确保内部拟合状态标志被重置
            try:
                if hasattr(self, '_switch_to_normal_display_mode') and callable(self._switch_to_normal_display_mode):
                    self._switch_to_normal_display_mode()
                else:
                    # 兜底：直接重置相关标志位
                    self.display_mode = 'normal'
                    if hasattr(self, '_display_mode'):
                        self._display_mode = 'normal'
                    if hasattr(self, '_fitting_mode_active'):
                        self._fitting_mode_active = False
                # 无论采用何种路径，保持对外的 display_mode 一致为 normal
                self.display_mode = 'normal'
            except Exception:
                # 再次兜底，至少保证对外 display_mode 为 normal
                self.display_mode = 'normal'
            if hasattr(self.ui, 'fitCurrentDataCheckBox'):
                try:
                    self.ui.fitCurrentDataCheckBox.blockSignals(True)
                    self.ui.fitCurrentDataCheckBox.setChecked(True)
                finally:
                    try:
                        self.ui.fitCurrentDataCheckBox.blockSignals(False)
                    except Exception:
                        pass
            
            # 初始化并应用ROI后更新显示（重置为全范围）
            try:
                self._initialize_roi_from_current_q(force_full=True)
            except Exception:
                pass
            self._apply_roi_to_data_and_refresh()
            self._update_GUI_image('normal')
            self._update_outside_window('normal')
                
        except Exception as e:
            self.status_updated.emit(f"Cut operation failed: {str(e)}")
            QMessageBox.critical(self.main_window, "Error", f"Cut operation failed:\n{str(e)}")

    def _update_GUI_image(self, mode):
        """统一的GUI图像更新函数"""
        try:
            if not self._has_valid_data():
                return
                
            # 获取显示选项
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()
            
            # 处理数据（使用ROI子集如果激活）
            q_data, I_data = self._get_roi_active_arrays()
            if q_data is None or I_data is None:
                return
            
            # 归一化（若启用）：使用同一归一化因子应用于实验数据与拟合数据，确保同一尺度
            norm_factor = 1.0
            if normalize:
                max_I = np.max(I_data) if I_data.size > 0 else 0.0
                if max_I > 0:
                    norm_factor = float(max_I)
                    I_data = I_data / norm_factor
            
            # 创建图形
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            
            scene = self._setup_fit_graphics_scene()
            if scene is None:
                return
            
            # 使用 4:3 比例减少上下留白
            fig = Figure(figsize=(9.6, 7.2), dpi=80)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            
            # Normal模式下的特殊处理：处理负数q值
            if mode == 'normal' and log_x:
                # 分离正数和负数q值
                positive_mask = q_data > 0
                negative_mask = q_data < 0
                zero_mask = q_data == 0
                
                # 绘制正数部分（蓝色点线图）
                if np.any(positive_mask):
                    ax.plot(q_data[positive_mask], I_data[positive_mask], 'o-', 
                           color='blue', markersize=4, linewidth=1, alpha=0.8,
                           label=f'{self.data_source.upper()} Data (q>0)' if self.data_source else 'Data (q>0)', zorder=2)
                
                # 绘制负数部分（红色点线图，使用|q|）
                if np.any(negative_mask):
                    ax.plot(np.abs(q_data[negative_mask]), I_data[negative_mask], 'o-', 
                           color='red', markersize=4, linewidth=1, alpha=0.8,
                           label=f'{self.data_source.upper()} Data (q<0, |q|)' if self.data_source else 'Data (q<0, |q|)', zorder=2)
                
                # 处理q=0的点（如果有）
                if np.any(zero_mask):
                    ax.plot(q_data[zero_mask], I_data[zero_mask], 'o', 
                           color='green', markersize=6, alpha=0.8,
                           label=f'{self.data_source.upper()} Data (q=0)' if self.data_source else 'Data (q=0)', zorder=3)
            else:
                # 其他情况：使用散点图
                ax.scatter(q_data, I_data, s=30, alpha=0.7, color='blue', 
                          label=f'{self.data_source.upper()} Data' if self.data_source else 'Data', zorder=2)
            # ROI 辅助线
            self._draw_roi_guides_if_active(ax)

            # 叠加组件曲线（BG、Res、Particles），使用虚线并基于最近一次拟合参数（仅在fitting模式显示）
            try:
                show_bg = self._get_checkbox_state('fitBGShowCheckBox', False)
                show_res = self._get_checkbox_state('fitResShowCheckBox', False)
                particle_flags = self._get_particle_sequence_flags()
                show_any = show_bg or show_res or any(particle_flags.values())
            except Exception:
                particle_flags = {}
                show_any = False

            # 组件曲线归一化与数据一致
            norm_divisor = norm_factor if normalize and norm_factor > 0 else None
            if mode == 'fitting' and show_any:
                shapes, params_list = self._get_last_fitting_spec_and_params()
                if shapes and params_list:
                    try:
                        from utils.fitting import mixed_model_components
                        comp = mixed_model_components(shapes, q_data, params_list)
                        # BG
                        if show_bg and comp.get('BG_total') is not None:
                            y_bg = comp['BG_total'] / norm_divisor if norm_divisor else comp['BG_total']
                            ax.plot(q_data, y_bg, linestyle='--', color='#666666', linewidth=1.5, label='bg', zorder=2)
                        # Resolution function
                        if show_res and comp.get('resolution') is not None:
                            y_res = comp['resolution'] / norm_divisor if norm_divisor else comp['resolution']
                            ax.plot(q_data, y_res, linestyle='--', color='#8E44AD', linewidth=1.5, label='Res.', zorder=2)
                        # Particles
                        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
                        for item in comp.get('particles', []):
                            idx = int(item.get('index', 0))
                            if particle_flags.get(idx, False):
                                yv = item.get('I')
                                if yv is not None:
                                    shape_name = str(item.get('shape', 'Particle')).capitalize()
                                    widget_id = self._sequence_index_to_widget_id(idx)
                                    color_key = widget_id if widget_id is not None else idx
                                    color = colors[(color_key - 1) % len(colors)] if color_key else colors[(idx-1) % len(colors)]
                                    yv_plot = yv / norm_divisor if norm_divisor else yv
                                    label_id = f"{shape_name} {widget_id}" if widget_id is not None else f"{shape_name} {idx}"
                                    ax.plot(q_data, yv_plot, linestyle='--', color=color, linewidth=1.5, label=label_id, zorder=2)
                        # 刷新图例以包含新线
                        ax.legend()
                    except Exception:
                        pass

            #（组件叠加显示仅在 _plot_fitting_result 中实现，避免在普通更新路径重复绘制）
            
            # 如果是fitting模式且有拟合数据，绘制拟合曲线（按ROI对齐）
            if mode == 'fitting' and self.has_fitting_data and self.I_fitting is not None:
                # Align fitting curve with current q_data (which may be ROI subset)
                I_fitting_arr = np.asarray(self.I_fitting)
                q_full = np.asarray(self.q)
                # Build mapping: select indices whose q are within ROI boundaries
                if self._roi_active():
                    roi_mask_full = (q_full >= self._roi_min) & (q_full <= self._roi_max)
                    if np.any(roi_mask_full) and roi_mask_full.sum() == len(q_data):
                        I_fitting_data = I_fitting_arr[roi_mask_full]
                    else:
                        # Fallback: if sizes match already, use as-is
                        I_fitting_data = I_fitting_arr[:len(q_data)]
                else:
                    I_fitting_data = I_fitting_arr[:len(q_data)]
                if normalize and norm_factor > 0:
                    I_fitting_data = I_fitting_data / norm_factor
                ax.plot(q_data, I_fitting_data, color='red', linewidth=2, 
                       label='Fitting', zorder=3)
            
            # 设置标签和样式
            ax.set_xlabel('q (Å$^{-1}$)' if not (mode == 'normal' and log_x and np.any(q_data < 0)) else '|q| (Å$^{-1}$)')
            ax.set_ylabel('Normalized Intensity' if normalize else 'Intensity (a.u.)')
            ax.set_title(f'{mode.capitalize()} Mode - {self.data_source.upper() if self.data_source else "Data"}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 应用对数坐标
            self._apply_log_scales(ax, log_x, log_y)
            
            fig.tight_layout()
            
            # 添加到场景
            proxy_widget = scene.addWidget(canvas)
            self._fit_view_to_item(self.ui.fitGraphicsView, proxy_widget, keep_aspect=True)
            
            # 保存引用
            self._current_fit_figure = fig
            self._current_fit_canvas = canvas
            
        except Exception:
            pass

    def _update_outside_window(self, mode):
        """统一的外部窗口更新函数"""
        try:
            if not hasattr(self, 'independent_fit_window') or self.independent_fit_window is None or not self.independent_fit_window.isVisible():
                return
                
            # 获取显示选项
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()
            
            # 检查"Positive Only"复选框状态
            positive_only = self._is_positive_only_enabled()
            
            # 选择数据源，并应用ROI（如果激活）与有效性过滤
            q_data = None
            I_data = None
            try:
                if mode == 'normal':
                    # Normal 模式：严格与内嵌视图一致，直接使用 self._get_roi_active_arrays()
                    q_data, I_data = self._get_roi_active_arrays()
                else:
                    # Fitting 模式：遵循数据源选择（Current Cut 优先，其次 1D，再回退到全局）
                    use_current_cut = hasattr(self.ui, 'fitCurrentDataCheckBox') and self.ui.fitCurrentDataCheckBox.isChecked()
                    if use_current_cut and getattr(self, 'current_cut_data', None) is not None:
                        q_src = np.asarray(self.current_cut_data.get('x_coords', []))
                        I_src = np.asarray(self.current_cut_data.get('y_intensity', []))
                        if self._roi_active() and q_src.size > 0:
                            mask = (q_src >= self._roi_min) & (q_src <= self._roi_max)
                            if np.any(mask):
                                q_src = q_src[mask]; I_src = I_src[mask]
                        q_data, I_data = q_src, I_src
                    elif getattr(self, 'current_1d_data', None) is not None:
                        q_src = np.asarray(self.current_1d_data.get('q', []))
                        I_src = np.asarray(self.current_1d_data.get('I', []))
                        if self._roi_active() and q_src.size > 0:
                            mask = (q_src >= self._roi_min) & (q_src <= self._roi_max)
                            if np.any(mask):
                                q_src = q_src[mask]; I_src = I_src[mask]
                        q_data, I_data = q_src, I_src
                    else:
                        q_data, I_data = self._get_roi_active_arrays()
            except Exception:
                q_data, I_data = self._get_roi_active_arrays()

            if q_data is None or I_data is None or len(q_data) == 0 or len(I_data) == 0:
                return
            finite_mask = np.isfinite(q_data) & np.isfinite(I_data)
            q_data = q_data[finite_mask]
            I_data = I_data[finite_mask]
            
            # 如果启用"Positive Only"，过滤掉负数q值
            if positive_only:
                positive_mask = q_data > 0
                q_data = q_data[positive_mask]
                I_data = I_data[positive_mask]
            
            # 归一化（若启用）：使用同一归一化因子应用于实验数据与拟合数据
            norm_factor = 1.0
            if normalize:
                max_I = np.max(I_data) if I_data.size > 0 else 0.0
                if max_I > 0:
                    norm_factor = float(max_I)
                    I_data = I_data / norm_factor
            
            ax = self.independent_fit_window.ax
            ax.clear()
            
            # Normal模式下的特殊处理：处理负数q值（仅当未启用positive_only时）
            if mode == 'normal' and log_x and not positive_only:
                # 分离正数和负数q值
                positive_mask = q_data > 0
                negative_mask = q_data < 0
                zero_mask = q_data == 0
                
                # 绘制正数部分（蓝色点线图）
                if np.any(positive_mask):
                    ax.plot(q_data[positive_mask], I_data[positive_mask], 'o-', 
                           color='blue', markersize=4, linewidth=1, alpha=0.8,
                           label=f'{self.data_source.upper()} Data (q>0)' if self.data_source else 'Data (q>0)', zorder=2)
                
                # 绘制负数部分（红色点线图，使用|q|）
                if np.any(negative_mask):
                    ax.plot(np.abs(q_data[negative_mask]), I_data[negative_mask], 'o-', 
                           color='red', markersize=4, linewidth=1, alpha=0.8,
                           label=f'{self.data_source.upper()} Data (q<0, |q|)' if self.data_source else 'Data (q<0, |q|)', zorder=2)
                
                # 处理q=0的点（如果有）
                if np.any(zero_mask):
                    ax.plot(q_data[zero_mask], I_data[zero_mask], 'o', 
                           color='green', markersize=6, alpha=0.8,
                           label=f'{self.data_source.upper()} Data (q=0)' if self.data_source else 'Data (q=0)', zorder=3)
            else:
                # 其他情况：使用散点图
                ax.scatter(q_data, I_data, s=30, alpha=0.7, color='blue', 
                          label=f'{self.data_source.upper()} Data' if self.data_source else 'Data', zorder=2)
            # ROI 辅助线
            self._draw_roi_guides_if_active(ax)

            # 叠加组件曲线（BG、Res、Particles），仅在fitting模式显示
            try:
                show_bg = self._get_checkbox_state('fitBGShowCheckBox', False)
                show_res = self._get_checkbox_state('fitResShowCheckBox', False)
                particle_flags = self._get_particle_sequence_flags()
                show_any = show_bg or show_res or any(particle_flags.values())
            except Exception:
                particle_flags = {}
                show_any = False

            if mode == 'fitting' and show_any:
                shapes, params_list = self._get_last_fitting_spec_and_params()
                if shapes and params_list:
                    try:
                        from utils.fitting import mixed_model_components
                        comp = mixed_model_components(shapes, q_data, params_list)
                        norm_divisor = norm_factor if normalize and norm_factor > 0 else None
                        # BG
                        if show_bg and comp.get('BG_total') is not None:
                            y_bg = comp['BG_total'] / norm_divisor if norm_divisor else comp['BG_total']
                            ax.plot(q_data, y_bg, linestyle='--', color='#666666', linewidth=1.5, label='bg', zorder=2)
                        # Resolution function
                        if show_res and comp.get('resolution') is not None:
                            y_res = comp['resolution'] / norm_divisor if norm_divisor else comp['resolution']
                            ax.plot(q_data, y_res, linestyle='--', color='#8E44AD', linewidth=1.5, label='Res.', zorder=2)
                        # Particles
                        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
                        for item in comp.get('particles', []):
                            idx = int(item.get('index', 0))
                            if particle_flags.get(idx, False):
                                yv = item.get('I')
                                if yv is not None:
                                    shape_name = str(item.get('shape', 'Particle')).capitalize()
                                    widget_id = self._sequence_index_to_widget_id(idx)
                                    color_key = widget_id if widget_id is not None else idx
                                    color = colors[(color_key - 1) % len(colors)] if color_key else colors[(idx-1) % len(colors)]
                                    yv_plot = yv / norm_divisor if norm_divisor else yv
                                    label_id = f"{shape_name} {widget_id}" if widget_id is not None else f"{shape_name} {idx}"
                                    ax.plot(q_data, yv_plot, linestyle='--', color=color, linewidth=1.5, label=label_id, zorder=2)
                        ax.legend()
                    except Exception:
                        pass
            
            # 如果是fitting模式且有拟合数据，绘制拟合曲线（与数据对齐，并考虑Positive Only）
            if mode == 'fitting' and self.has_fitting_data and self.I_fitting is not None:
                I_fitting_arr = np.asarray(self.I_fitting)
                q_full = np.asarray(self.q)
                # Build mask to align with displayed q_data (ROI + positive_only)
                mask_full = np.isfinite(q_full)
                if self._roi_active():
                    mask_full &= (q_full >= self._roi_min) & (q_full <= self._roi_max)
                if positive_only:
                    mask_full &= (q_full > 0)
                I_fitting_data = I_fitting_arr[mask_full]
                # Trim/pad safety: align length with q_data
                if len(I_fitting_data) != len(q_data):
                    I_fitting_data = I_fitting_data[:len(q_data)]
                # 与实验数据使用相同归一化因子
                if normalize and norm_factor > 0:
                    I_fitting_data = I_fitting_data / norm_factor
                ax.plot(q_data, I_fitting_data, color='red', linewidth=2.5, 
                       label='Fitting', zorder=3)
            
            # 设置标签和样式
            x_label = 'q (Å$^{-1}$)'
            if mode == 'normal' and log_x and not positive_only and np.any(np.array(self.q) < 0):
                x_label = '|q| (Å$^{-1}$)'
            elif positive_only:
                x_label = 'q (Å$^{-1}$) [Positive Only]'
                
            ax.set_xlabel(x_label)
            ax.set_ylabel('Normalized Intensity' if normalize else 'Intensity (a.u.)')
            ax.set_title(f'{mode.capitalize()} Mode - {self.data_source.upper() if self.data_source else "Data"}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 设置坐标轴样式
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.8)
            
            # 应用对数坐标
            self._apply_log_scales(ax, log_x, log_y)
            
            # 刷新显示
            if hasattr(self.independent_fit_window, 'canvas'):
                self.independent_fit_window.canvas.draw()
                
        except Exception:
            pass
    
    def _get_cut_center_coordinates(self):
        """获取切割区域的中心点坐标"""
        center_x = 0.0
        center_y = 0.0
        
        if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
            center_x = self.ui.gisaxsInputCenterParallelValue.value()
        if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
            center_y = self.ui.gisaxsInputCenterVerticalValue.value()
            
        return center_x, center_y
    
    def _perform_cut_operation(self, vertical_value, parallel_value, cut_type: str, points_override: int = None):
        """通用切割操作方法
        
        Args:
            vertical_value: 垂直方向的切割参数
            parallel_value: 平行方向的切割参数
            cut_type: 切割类型，'horizontal' 或 'vertical'
        """
        try:
            # 获取切割区域的中心点
            center_x, center_y = self._get_cut_center_coordinates()
            
            # 检查是否为Q坐标模式
            show_q_axis = self._should_show_q_axis()
            
            # 根据切割类型设置参数
            if cut_type == 'horizontal':
                q_mode_method = self._extract_horizontal_cut_q_mode
                pixel_mode_method = self._extract_horizontal_cut_pixel_mode
                pixel_to_q_method = self._convert_pixel_to_qy
                x_label = r'$q_y$ (nm$^{-1}$)'
                title = "Horizontal Cut"
            elif cut_type == 'vertical':
                q_mode_method = self._extract_vertical_cut_q_mode
                pixel_mode_method = self._extract_vertical_cut_pixel_mode
                pixel_to_q_method = self._convert_pixel_to_qz
                x_label = r'$q_z$ (nm$^{-1}$)'
                title = "Vertical Cut"
            else:
                raise Exception(f"Unknown cut type: {cut_type}")
            
            if show_q_axis:
                # Q坐标模式：直接使用Q坐标
                cut_data, q_coords = q_mode_method(
                    center_x, center_y, vertical_value, parallel_value, points_override=points_override
                )
                x_coordinates = q_coords
            else:
                # 像素坐标模式：提取像素数据后转换为Q坐标
                cut_data, pixel_coords = pixel_mode_method(
                    center_x, center_y, vertical_value, parallel_value, points_override=points_override
                )
                # 转换像素坐标到Q坐标
                x_coordinates = pixel_to_q_method(pixel_coords)
            
            # 绘制结果
            self._plot_cut_result(x_coordinates, cut_data, x_label, "Intensity (a.u.)", title)
            
        except Exception as e:
            raise Exception(f"{cut_type.capitalize()} cut failed: {str(e)}")
    
    def _perform_horizontal_cut(self, vertical_value, parallel_value, points_override: int = None):
        """执行横切操作"""
        self._perform_cut_operation(vertical_value, parallel_value, 'horizontal', points_override=points_override)
    
    def _perform_vertical_cut(self, vertical_value, parallel_value, points_override: int = None):
        """执行纵切操作"""
        self._perform_cut_operation(vertical_value, parallel_value, 'vertical', points_override=points_override)
    
    def _extract_cut_q_mode(self, center_qy, center_qz, height_q, width_q, cut_type: str, points_override: int = None):
        """Q坐标模式下的通用数据提取方法
        
        Args:
            center_qy, center_qz: Q空间中心坐标
            height_q, width_q: Q空间区域尺寸
            cut_type: 切割类型，'horizontal' 或 'vertical'
        """
        try:
            # 获取Q网格
            qy_mesh, qz_mesh = self._get_cached_q_meshgrids()
            if qy_mesh is None or qz_mesh is None:
                raise Exception("Q-space meshgrids not available")
            
            # 定义切割区域边界
            qy_min = center_qy - width_q / 2
            qy_max = center_qy + width_q / 2
            qz_min = center_qz - height_q / 2
            qz_max = center_qz + height_q / 2
            
            # 创建区域掩码
            mask = ((qy_mesh >= qy_min) & (qy_mesh <= qy_max) & 
                    (qz_mesh >= qz_min) & (qz_mesh <= qz_max))
            
            # 在区域内进行求和
            region_data = np.where(mask, self.current_stack_data, 0)
            
            # 根据切割类型进行求和和坐标提取
            if cut_type == 'horizontal':
                # 沿纵向求和得到横向分布
                intensity_sum = np.sum(region_data, axis=0)
                q_line = qy_mesh[0, :]  # 取第一行的qy值
            elif cut_type == 'vertical':
                # 沿横向求和得到纵向分布
                intensity_sum = np.sum(region_data, axis=1)
                q_line = qz_mesh[:, 0]  # 取第一列的qz值
            else:
                raise Exception(f"Unknown cut type: {cut_type}")
            
            # 过滤有效数据点
            valid_indices = np.isfinite(intensity_sum) & (intensity_sum > 0)
            if not np.any(valid_indices):
                raise Exception("No valid data in the selected region")
            
            valid_q = q_line[valid_indices]
            valid_intensity = intensity_sum[valid_indices]
            
            # 插值到指定点数（仅使用有限值）
            # Determine target points count
            if points_override is not None:
                n_points = int(points_override)
            else:
                # Prefer stable current value, fallback to UI value, then default
                n_points = int(getattr(self, '_points_num_current', self._points_num_default))
                try:
                    if hasattr(self.ui, 'fitDataPointsNumValue'):
                        _val = int(self.ui.fitDataPointsNumValue.value())
                        if _val >= 10:
                            n_points = _val
                except Exception:
                    pass
            n_points = max(10, n_points)
            q_interp = np.linspace(valid_q.min(), valid_q.max(), n_points)
            method = None
            try:
                method = self.ui.fitInterpolationMethodValue.currentText() if hasattr(self.ui, 'fitInterpolationMethodValue') else self._interp_method_default
            except Exception:
                method = self._interp_method_default
            intensity_interp = self._interpolate_series(valid_q, valid_intensity, q_interp, method)
            try:
                self.status_updated.emit(f"Cut(Q) extracted points: {len(q_interp)} (method={method})")
            except Exception:
                pass
            return intensity_interp, q_interp
            
        except Exception as e:
            raise Exception(f"Q-mode {cut_type} cut extraction failed: {str(e)}")
    
    def _extract_horizontal_cut_q_mode(self, center_qy, center_qz, height_q, width_q, points_override: int = None):
        """Q坐标模式下的横切数据提取"""
        return self._extract_cut_q_mode(center_qy, center_qz, height_q, width_q, 'horizontal', points_override=points_override)
    
    def _extract_vertical_cut_q_mode(self, center_qy, center_qz, height_q, width_q, points_override: int = None):
        """Q坐标模式下的纵切数据提取"""
        return self._extract_cut_q_mode(center_qy, center_qz, height_q, width_q, 'vertical', points_override=points_override)
    
    def _extract_cut_pixel_mode(self, center_x, center_y, height, width, cut_type: str, points_override: int = None):
        """像素坐标模式下的通用数据提取方法
        
        Args:
            center_x, center_y: 中心坐标
            height, width: 区域尺寸
            cut_type: 切割类型，'horizontal' 或 'vertical'
        """
        try:
            img_height, img_width = self.current_stack_data.shape
            
            # 计算像素边界
            x_min = max(0, int(center_x - width / 2))
            x_max = min(img_width, int(center_x + width / 2))
            y_min = max(0, int(center_y - height / 2))
            y_max = min(img_height, int(center_y + height / 2))
            
            # 由于显示时图像被flipud，需要调整y坐标
            y_min_adj = img_height - 1 - y_max
            y_max_adj = img_height - 1 - y_min
            y_min_adj, y_max_adj = max(0, y_min_adj), min(img_height, y_max_adj)
            
            # 提取区域数据
            region_data = self.current_stack_data[y_min_adj:y_max_adj+1, x_min:x_max+1]
            
            if region_data.size == 0:
                raise Exception("Empty region selected")
            
            # 根据切割类型进行求和
            if cut_type == 'horizontal':
                # 沿纵向求和得到横向分布
                intensity_sum = np.sum(region_data, axis=0)
                pixel_coords = np.arange(x_min, x_min + len(intensity_sum))
            elif cut_type == 'vertical':
                # 沿横向求和得到纵向分布
                intensity_sum = np.sum(region_data, axis=1)
                pixel_coords = np.arange(y_min, y_min + len(intensity_sum))
            else:
                raise Exception(f"Unknown cut type: {cut_type}")
            
            # 插值到指定点数（仅使用有限值）
            if len(pixel_coords) > 1:
                # Determine target points count
                if points_override is not None:
                    n_points = int(points_override)
                else:
                    n_points = int(getattr(self, '_points_num_current', self._points_num_default))
                    try:
                        if hasattr(self.ui, 'fitDataPointsNumValue'):
                            _val = int(self.ui.fitDataPointsNumValue.value())
                            if _val >= 10:
                                n_points = _val
                    except Exception:
                        pass
                n_points = max(10, n_points)
                # 过滤有限
                finite_mask = np.isfinite(pixel_coords) & np.isfinite(intensity_sum)
                pixel_coords_f = pixel_coords[finite_mask]
                intensity_sum_f = intensity_sum[finite_mask]
                if len(pixel_coords_f) < 2:
                    pixel_coords_f = pixel_coords
                    intensity_sum_f = intensity_sum
                pixel_interp = np.linspace(pixel_coords_f.min(), pixel_coords_f.max(), n_points)
                method = None
                try:
                    method = self.ui.fitInterpolationMethodValue.currentText() if hasattr(self.ui, 'fitInterpolationMethodValue') else self._interp_method_default
                except Exception:
                    method = self._interp_method_default
                intensity_interp = self._interpolate_series(pixel_coords_f, intensity_sum_f, pixel_interp, method)
                try:
                    self.status_updated.emit(f"Cut(Pixel) extracted points: {len(pixel_interp)} (method={method})")
                except Exception:
                    pass
            else:
                pixel_interp = pixel_coords
                intensity_interp = intensity_sum
            
            return intensity_interp, pixel_interp
            
        except Exception as e:
            raise Exception(f"Pixel-mode {cut_type} cut extraction failed: {str(e)}")
    
    def _extract_horizontal_cut_pixel_mode(self, center_x, center_y, height, width, points_override: int = None):
        """像素坐标模式下的横切数据提取"""
        return self._extract_cut_pixel_mode(center_x, center_y, height, width, 'horizontal', points_override=points_override)
    
    def _extract_vertical_cut_pixel_mode(self, center_x, center_y, height, width, points_override: int = None):
        """像素坐标模式下的纵切数据提取"""
        return self._extract_cut_pixel_mode(center_x, center_y, height, width, 'vertical', points_override=points_override)
    
    def _get_detector_for_pixel_conversion(self):
        """获取用于像素坐标转换的detector对象（统一方法）"""
        try:
            from utils.q_space_calculator import create_detector_from_image_and_params
            from core.global_params import GlobalParameterManager
            
            global_params = GlobalParameterManager()
            height, width = self.current_stack_data.shape
            
            # 获取探测器参数
            pixel_size_x = global_params.get_parameter('fitting', 'detector.pixel_size_x', 172.0)
            pixel_size_y = global_params.get_parameter('fitting', 'detector.pixel_size_y', 172.0)
            beam_center_x = global_params.get_parameter('fitting', 'detector.beam_center_x', width / 2.0)
            beam_center_y = global_params.get_parameter('fitting', 'detector.beam_center_y', height / 2.0)
            distance = global_params.get_parameter('fitting', 'detector.distance', 2565.0)
            theta_in_deg = global_params.get_parameter('beam', 'grazing_angle', 0.4)
            wavelength = global_params.get_parameter('beam', 'wavelength', 0.1045)
            
            return create_detector_from_image_and_params(
                image_shape=(height, width),
                pixel_size_x=pixel_size_x,
                pixel_size_y=pixel_size_y,
                beam_center_x=beam_center_x,
                beam_center_y=beam_center_y,
                distance=distance,
                theta_in_deg=theta_in_deg,
                wavelength=wavelength,
                crop_params=None
            )
        except Exception:
            return None
    
    def _convert_pixel_coords_to_q(self, pixel_coords, conversion_type: str):
        """通用像素坐标到Q坐标转换方法
        
        Args:
            pixel_coords: 像素坐标数组
            conversion_type: 转换类型，'qy' 或 'qz'
        """
        try:
            detector = self._get_detector_for_pixel_conversion()
            if detector is None:
                raise Exception("Failed to create detector")
            
            height, width = self.current_stack_data.shape
            q_coords = []
            
            if conversion_type == 'qy':
                # 转换到qy坐标（使用图像中心的y坐标）
                center_y = height / 2.0
                for px in pixel_coords:
                    _, qy, _ = detector.pixel_to_q_space(px, center_y)
                    q_coords.append(qy)
            elif conversion_type == 'qz':
                # 转换到qz坐标（使用图像中心的x坐标）
                center_x = width / 2.0
                for py in pixel_coords:
                    _, _, qz = detector.pixel_to_q_space(center_x, py)
                    q_coords.append(qz)
            else:
                raise Exception(f"Unknown conversion type: {conversion_type}")
            
            return np.array(q_coords)
            
        except Exception as e:
            # 如果转换失败，返回归一化的像素坐标
            self.status_updated.emit(f"Pixel to {conversion_type} conversion failed: {str(e)}")
            return (pixel_coords - pixel_coords.mean()) / pixel_coords.std()
    
    def _convert_pixel_to_qy(self, pixel_coords):
        """将像素坐标转换为qy坐标"""
        return self._convert_pixel_coords_to_q(pixel_coords, 'qy')
    
    def _convert_pixel_to_qz(self, pixel_coords):
        """将像素坐标转换为qz坐标"""
        return self._convert_pixel_coords_to_q(pixel_coords, 'qz')
    
    def _plot_cut_result(self, x_coords, y_intensity, x_label, y_label, title):
        """在fitGraphicsView中绘制切割结果"""
        try:
            # 导入数据到统一的 q,I 存储器（先过滤非有限值）
            x_arr = np.asarray(x_coords)
            y_arr = np.asarray(y_intensity)
            finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            x_arr = x_arr[finite_mask]
            y_arr = y_arr[finite_mask]
            self.q = x_arr
            self.I = y_arr
            
            # 存储当前切割数据，用于兼容性
            self.current_cut_data = {
                'x_coords': x_arr.copy() if hasattr(x_arr, 'copy') else list(x_arr),
                'y_intensity': y_arr.copy() if hasattr(y_arr, 'copy') else list(y_arr),
                'x_label': x_label,
                'y_label': y_label,
                'title': title
            }
            # 统一的全局Cut缓存（q轴数据命名为q，便于导出与复用）
            try:
                import time
                self.cut = {
                    'q': x_arr.copy() if hasattr(x_arr, 'copy') else np.array(x_arr),
                    'I': y_arr.copy() if hasattr(y_arr, 'copy') else np.array(y_arr),
                    'meta': {
                        'x_label': x_label,
                        'y_label': y_label,
                        'title': title,
                        'timestamp': time.time(),
                        'source': 'cut'
                    }
                }
            except Exception:
                # 若失败，至少保证基本结构
                self.cut = {'q': x_arr, 'I': y_arr, 'meta': {'source': 'cut'}}
            
            if not is_matplotlib_available():
                QMessageBox.warning(self.main_window, "Missing Library",
                                  "matplotlib library is required for plotting.\nPlease install it using: pip install matplotlib")
                return
            
            # 获取显示选项
            options = self._display_manager.get_display_options()
            
            # 使用统一显示管理器
            # 注意：对于Cut数据，x_coords可能不是q，需要适配
            if "q" in x_label.lower():
                # 如果x轴是q，直接使用1D数据显示方法
                self._display_manager.plot_1d_data(
                    x_coords, y_intensity, None, title, "cut_data",
                    options['log_x'], options['log_y'], options['normalize']
                )
            else:
                # 如果x轴不是q，使用特殊处理（保持原有逻辑）
                self._plot_cut_data_legacy(x_coords, y_intensity, x_label, y_label, title, options)
            
            self.status_updated.emit(f"Cut result plotted: {title}")
            
        except Exception as e:
            self.status_updated.emit(f"Plot failed: {str(e)}")
            QMessageBox.critical(
                self.main_window,
                "Plot Error",
                f"Failed to plot cut result:\n{str(e)}"
            )
    
    def _plot_cut_data_legacy(self, x_coords, y_intensity, x_label, y_label, title, options):
        """处理非q数据的Cut结果显示（保持原有逻辑）"""
        try:
            # 应用标准化处理
            y_data = np.array(y_intensity)
            if options['normalize']:
                max_intensity = np.max(y_data)
                if max_intensity > 0:
                    y_data = y_data / max_intensity
                    y_label = "Normalized Intensity"
            
            # 创建图形
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            
            # 使用统一的场景管理方法
            scene = self._setup_fit_graphics_scene()
            if scene is None:
                return
            
            # 创建matplotlib图形
            fig = Figure(figsize=(8, 6), dpi=80)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            
            # 使用共享函数绘制数据
            self._plot_cut_data_with_log_handling(ax, x_coords, y_data, options['log_x'], markersize=4, linewidth=1.5)
            
            # 设置坐标轴
            ax.set_xlabel(x_label, fontsize=13)
            ax.set_ylabel(y_label, fontsize=13)
            ax.set_title(title, fontsize=15)
            ax.grid(True, alpha=0.3)

            # 设置坐标轴样式
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.8)
            ax.tick_params(axis='both', which='both', width=1.6, labelsize=12)
            
            # 应用对数坐标设置
            if options['log_x']:
                ax.set_xscale('log')
            if options['log_y']:
                ax.set_yscale('log')
            
            # 调整布局
            fig.tight_layout()
            
            # 添加到场景
            proxy_widget = scene.addWidget(canvas)
            self._fit_view_to_item(self.ui.fitGraphicsView, proxy_widget, keep_aspect=True)
            
            # 如果独立拟合窗口存在且可见，也更新其显示
            if self.independent_fit_window is not None and self.independent_fit_window.isVisible():
                self.independent_fit_window.update_plot(
                    x_coords, y_intensity, x_label, y_label, title,  
                    log_x=options['log_x'],
                    log_y=options['log_y'],
                    normalize=options['normalize']
                )
            
            self.status_updated.emit(f"Cut result plotted: {title}")
            
        except Exception as e:
            self.status_updated.emit(f"Legacy plot cut data error: {str(e)}")
    
    def _get_checkbox_state(self, checkbox_name: str, default_value: bool = False) -> bool:
        """通用复选框状态检查方法"""
        try:
            if hasattr(self.ui, checkbox_name):
                checkbox = getattr(self.ui, checkbox_name)
                return checkbox.isChecked()
            return default_value
        except Exception:
            return default_value
    
    def _is_fit_log_x_enabled(self):
        """检查是否启用X轴对数显示"""
        return self._get_checkbox_state('fitLogXCheckBox', False)
    
    def _is_fit_log_y_enabled(self):
        """检查是否启用Y轴对数显示"""
        return self._get_checkbox_state('fitLogYCheckBox', False)
    
    def _is_fit_norm_enabled(self):
        """检查是否启用标准化"""
        # 只检查fitNormCheckBox
        return self._get_checkbox_state('fitNormCheckBox', False)
    
    def _is_positive_only_enabled(self):
        """检查是否启用Positive Only模式（仅显示正数q值）"""
        # 首先检查独立窗口中的复选框
        if (hasattr(self, 'independent_fit_window') and 
            self.independent_fit_window is not None and 
            hasattr(self.independent_fit_window, 'show_positive_cb')):
            return self.independent_fit_window.show_positive_cb.isChecked()
        
        # 如果独立窗口不存在，检查主窗口中的复选框
        return self._get_checkbox_state('PositiveOnlyCheckBox', False)
    
    def _normalize_intensity_data(self, I_data):
        """统一的强度数据归一化方法"""
        if len(I_data) == 0:
            return I_data
        max_I = np.max(I_data)
        if max_I > 0:
            return I_data / max_I
        return I_data
    
    def _apply_log_scales(self, ax, log_x=False, log_y=False):
        """统一的对数坐标设置方法"""
        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
    
    def _has_valid_data(self):
        """检查是否有有效的q,I数据"""
        return (hasattr(self, 'q') and hasattr(self, 'I') and 
                self.q is not None and self.I is not None and 
                len(self.q) > 0 and len(self.I) > 0)
    
    # =========================================================================
    # FittingTextBrowser 状态信息显示
    # =========================================================================
    
    def _setup_fitting_text_browser(self):
        """设置FittingTextBrowser来显示状态信息"""
        if hasattr(self.ui, 'FittingTextBrowser'):
            # 连接status_updated信号到FittingTextBrowser
            self.status_updated.connect(self._update_fitting_text_browser)
            
            # 初始化FittingTextBrowser
            self.ui.FittingTextBrowser.clear()
            self._add_fitting_message("Fitting Controller initialized", "INFO")
            self._init_fitting_textbrowser_enhancements()

    def _init_fitting_textbrowser_enhancements(self):
        """Initialize FittingTextBrowser enhancements: fixed height, context menu, detachable window."""
        tb = getattr(self.ui, 'FittingTextBrowser', None)
        if tb is None:
            return
        # Fixed height at 100px per new requirement
        tb.setMinimumHeight(100)
        tb.setMaximumHeight(100)
        if self._fitting_browser_original_height is None:
            self._fitting_browser_original_height = tb.height()
        # 设置自定义右键菜单
        tb.setContextMenuPolicy(Qt.CustomContextMenu)
        tb.customContextMenuRequested.connect(self._show_fitting_browser_menu)

    def _show_fitting_browser_menu(self, pos: QPoint):
        tb = getattr(self.ui, 'FittingTextBrowser', None)
        if tb is None:
            return
        menu = QMenu(tb)
        act_copy_all = QAction("Copy All", menu)
        act_clear = QAction("Clear", menu)
        act_save = QAction("Save Log...", menu)
        act_detach = QAction("Open Detached Window", menu)
        act_set_max_lines = QAction(f"Set Max Lines (Current: {self._fitting_messages_max_lines})", menu)
        menu.addAction(act_copy_all)
        menu.addAction(act_clear)
        menu.addSeparator()
        menu.addAction(act_save)
        menu.addSeparator()
        menu.addAction(act_detach)
        menu.addSeparator()
        menu.addAction(act_set_max_lines)
        chosen = menu.exec_(tb.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == act_copy_all:
            tb.selectAll(); tb.copy()
        elif chosen == act_clear:
            self.clear_fitting_messages()
        elif chosen == act_save:
            path, _ = QFileDialog.getSaveFileName(tb, "Save Fitting Log", "fitting_log.txt", "Text Files (*.txt)")
            if path:
                self.save_fitting_log(path)
        elif chosen == act_detach:
            self._open_detached_fitting_browser()
        elif chosen == act_set_max_lines:
            val, ok = QInputDialog.getInt(tb, "Max Lines", "Set maximum display lines:", self._fitting_messages_max_lines, 50, 5000, 50)
            if ok:
                self._fitting_messages_max_lines = val
                self._trim_fitting_messages_if_needed()

    def _open_detached_fitting_browser(self):
        tb = getattr(self.ui, 'FittingTextBrowser', None)
        if tb is None:
            return
        if self._detached_fitting_dialog is not None:
            try:
                self._detached_fitting_dialog.raise_(); self._detached_fitting_dialog.activateWindow(); return
            except Exception:
                self._detached_fitting_dialog = None
        dlg = QDialog(tb)
        dlg.setWindowTitle("Fitting Log (Detached)")
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser(dlg)
        browser.setHtml(tb.toHtml())
        layout.addWidget(browser)
        dlg.resize(640, 420)
        self._detached_fitting_dialog = dlg
        def sync(msg_html):
            try: browser.append(msg_html)
            except Exception: pass
        self._detached_append = sync
        # Allow reopen after close
        dlg.finished.connect(self._on_detached_closed)
        dlg.show()

    def _on_detached_closed(self, *_):
        self._detached_fitting_dialog = None
        self._detached_append = None

    def _trim_fitting_messages_if_needed(self):
        tb = getattr(self.ui, 'FittingTextBrowser', None)
        if tb is None:
            return
        doc = tb.document()
        blocks = doc.blockCount()
        if blocks <= self._fitting_messages_max_lines:
            return
        remove_count = blocks - self._fitting_messages_max_lines
        cursor = tb.textCursor()
        cursor.movePosition(cursor.Start)
        for _ in range(remove_count):
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
        # Trimming notice in English (sync detached if exists)
        notice = f'<span style="color:#888;">(Log trimmed, keeping last {self._fitting_messages_max_lines} lines)</span>'
        tb.append(notice)
        if self._detached_append:
            self._detached_append(notice)

    
    def _update_fitting_text_browser(self, message: str):
        """更新FittingTextBrowser显示状态信息"""
        if hasattr(self.ui, 'FittingTextBrowser'):
            self._add_fitting_message(message, "STATUS")
    
    def _add_fitting_message(self, message: str, msg_type: str = "INFO"):
        """添加消息到FittingTextBrowser"""
        if not hasattr(self.ui, 'FittingTextBrowser'):
            return
            
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 根据消息类型设置颜色
        color_map = {
            "INFO": "#2E86AB",      # 蓝色
            "STATUS": "#28A745",    # 绿色  
            "WARNING": "#FD7E14",   # 橙色
            "ERROR": "#DC3545",     # 红色
            "SUCCESS": "#198754",   # 深绿色
            "PARTICLE": "#6F42C1"   # 紫色（粒子相关）
        }
        
        color = color_map.get(msg_type, "#333333")
        
        # 格式化消息
        formatted_message = f'<span style="color: {color};">[{timestamp}] {msg_type}: {message}</span>'
        
        # 添加到FittingTextBrowser
        self.ui.FittingTextBrowser.append(formatted_message)
        # 同步到独立窗口
        if self._detached_append:
            try:
                self._detached_append(formatted_message)
            except Exception:
                pass
        # 行数裁剪
        self._trim_fitting_messages_if_needed()
        
        # 自动滚动到底部
        cursor = self.ui.FittingTextBrowser.textCursor()
        cursor.movePosition(cursor.End)
        self.ui.FittingTextBrowser.setTextCursor(cursor)
    
    def _add_fitting_warning(self, message: str):
        """添加警告消息"""
        self._add_fitting_message(message, "WARNING")
    
    def _add_fitting_error(self, message: str):
        """添加错误消息"""
        self._add_fitting_message(message, "ERROR")
    
    def _add_fitting_success(self, message: str):
        """添加成功消息"""
        self._add_fitting_message(message, "SUCCESS")
    
    def _add_particle_message(self, message: str):
        """添加粒子相关消息"""
        self._add_fitting_message(message, "PARTICLE")
    
    def clear_fitting_messages(self):
        """Clear fitting messages in both embedded and detached browser."""
        if hasattr(self.ui, 'FittingTextBrowser'):
            self.ui.FittingTextBrowser.clear()
            self._add_fitting_message("Messages cleared", "INFO")
            # Sync detached window
            if self._detached_fitting_dialog is not None:
                for child in self._detached_fitting_dialog.children():
                    if isinstance(child, QTextBrowser):
                        child.clear()
                        child.append('<span style="color:#2E86AB;">[INFO] Messages cleared</span>')
    
    def get_fitting_messages(self) -> str:
        """获取FittingTextBrowser的所有内容"""
        if hasattr(self.ui, 'FittingTextBrowser'):
            return self.ui.FittingTextBrowser.toPlainText()
        return ""
    
    def save_fitting_log(self, filepath: str) -> bool:
        """保存拟合日志到文件"""
        try:
            content = self.get_fitting_messages()
            if content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                self._add_fitting_success(f"Fitting log saved to: {filepath}")
                return True
            else:
                self._add_fitting_warning("No messages to save")
                return False
        except Exception as e:
            self._add_fitting_error(f"Failed to save fitting log: {str(e)}")
            return False
    
    # =========================================================================
    # 粒子形状连接器 - 管理现有的粒子形状页面
    # =========================================================================
    
    def _setup_particle_shape_connector(self):
        """设置粒子形状连接器"""
        self._initialize_particle_ui_registry()
        if not getattr(self, 'particle_shape_configs', None):
            self.particle_shape_configs = {}
        
        # 控件类型定义 - 方便添加新的参数控件
        self.particle_control_types = {
            'Sphere': ['Int', 'R', 'SigmaR', 'D', 'SigmaD', 'BG'],
            'Cylinder': ['Int', 'R', 'SigmaR', 'h', 'Sigmah', 'D', 'SigmaD', 'BG']
        }
        
        # 设置连接
        self._setup_particle_connections()
        
        # 设置参数控件连接
        self._setup_particle_parameter_connections()
        
        # 设置全局拟合参数连接
        self._setup_global_parameter_connections()
        
        # 设置所有参数控件的数値范围
        self._setup_parameter_ranges()
        
        # 初始化粒子状态（根据模型参数设置）
        self._initialize_particle_states()
        
        # 初始化全局参数状态
        self._initialize_global_parameters()
        
        self._add_fitting_success("Particle Shape Connector initialized")
    
    def _iter_particle_widget_ids(self):
        """返回当前粒子widget的编号列表（按升序）。"""
        return sorted(self.particle_shape_configs.keys()) if getattr(self, 'particle_shape_configs', None) else []

    def _collect_active_particles(self):
        """收集当前激活的粒子形状及其widget编号。"""
        active_shapes = []
        widget_order = []
        for widget_id in self._iter_particle_widget_ids():
            combo_name = f'fitParticleShapeCombox_{widget_id}'
            if not hasattr(self.ui, combo_name):
                continue
            combobox = getattr(self.ui, combo_name)
            current_text = combobox.currentText().strip() if combobox.currentText() else ''
            if current_text and current_text.lower() != 'none':
                active_shapes.append(current_text.lower())
                widget_order.append(widget_id)
        return active_shapes, widget_order

    def _get_particle_sequence_flags(self):
        """返回最近一次拟合顺序中，各组件是否需要显示。"""
        flags = {}
        sequence = getattr(self, '_last_active_particle_ids', []) or []
        for idx, widget_id in enumerate(sequence, 1):
            checkbox_name = f'fitParticle{widget_id}ShowCheckBox'
            flags[idx] = self._get_checkbox_state(checkbox_name, False)
        return flags

    def _sequence_index_to_widget_id(self, seq_index: int):
        sequence = getattr(self, '_last_active_particle_ids', []) or []
        if 1 <= seq_index <= len(sequence):
            return sequence[seq_index - 1]
        return None

    def _initialize_particle_ui_registry(self):
        """解析UI中的粒子控件容器、现有widget以及新增按钮。"""
        try:
            if hasattr(self.ui, 'scrollAreaWidgetContents'):
                self._particle_scroll_container = self.ui.scrollAreaWidgetContents
                self._particle_container_layout = self._particle_scroll_container.layout()
        except Exception:
            self._particle_scroll_container = None
            self._particle_container_layout = None

        add_button = None
        for name in ('addModelButton', 'fitAddModelButton', 'pushButton'):
            if hasattr(self.ui, name):
                add_button = getattr(self.ui, name)
                break
        if add_button:
            self._particle_add_button = add_button
            if not add_button.toolTip():
                add_button.setToolTip("Add particle model")
            if not getattr(add_button, '_particle_handler_connected', False):
                add_button.clicked.connect(self._on_add_particle_clicked)
                add_button._particle_handler_connected = True

        self._prepare_dynamic_show_checkbox_area()

        self.particle_shape_configs = {}
        idx = 1
        while hasattr(self.ui, f'fitParticleWidget_{idx}') and hasattr(self.ui, f'fitParticleShapeCombox_{idx}'):
            self._register_existing_particle_widget(idx)
            idx += 1
        self._next_particle_candidate = idx

    def _prepare_dynamic_show_checkbox_area(self):
        if self._dynamic_show_layout is not None:
            return

        preferred_names = ('ParticlesNumWidget', 'fitParticlesNumWidget')
        for name in preferred_names:
            host = getattr(self.ui, name, None)
            if isinstance(host, QWidget):
                layout = host.layout()
                if layout is None:
                    layout = QVBoxLayout(host)
                    layout.setContentsMargins(0, 0, 0, 0)
                    layout.setSpacing(4)
                self._dynamic_show_container = host
                self._dynamic_show_layout = layout
                self._particle_checkbox_host_name = name
                return

        host_widget = getattr(self.ui, 'fitFittingShowWidget', None)
        if host_widget is None:
            return
        base_layout = host_widget.layout()
        if base_layout is None:
            from PyQt5.QtWidgets import QGridLayout
            base_layout = QGridLayout(host_widget)

        self._dynamic_show_container = QWidget(host_widget)
        layout = QVBoxLayout(self._dynamic_show_container)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)
        self._dynamic_show_layout = layout
        self._particle_checkbox_host_name = 'fitFittingShowWidget'

        # 将额外区域插入到最下方（若为网格则追加一行）
        if hasattr(base_layout, 'addWidget'):
            if hasattr(base_layout, 'rowCount'):
                row_index = base_layout.rowCount()
                try:
                    base_layout.addWidget(self._dynamic_show_container, row_index, 0, 1, 2)
                    return
                except Exception:
                    pass
            base_layout.addWidget(self._dynamic_show_container)

    def _register_existing_particle_widget(self, widget_id: int):
        widget = getattr(self.ui, f'fitParticleWidget_{widget_id}', None)
        if widget is None:
            return
        self._particle_widgets[widget_id] = widget
        if not self._particle_widget_style_template:
            self._particle_widget_style_template = widget.styleSheet()
            self._particle_widget_style_source_name = widget.objectName() or ''
        self._apply_particle_widget_style(widget, widget_id)
        self.particle_shape_configs[widget_id] = self._build_particle_config(widget_id)
        self._install_particle_context_menu(widget, widget_id)

        checkbox = getattr(self.ui, f'fitParticle{widget_id}ShowCheckBox', None)
        if checkbox is not None:
            self._register_particle_show_checkbox(widget_id, checkbox)

    def _build_particle_config(self, widget_id: int) -> dict:
        return {
            'combobox': f'fitParticleShapeCombox_{widget_id}',
            'stack_widget': f'fitParticleStackWidget_{widget_id}',
            'pages': {
                0: {'name': 'Sphere', 'page_index': 0},
                1: {'name': 'Cylinder', 'page_index': 1},
                2: {'name': 'None', 'page_index': 0},
            }
        }

    def _register_particle_show_checkbox(self, widget_id: int, checkbox: QCheckBox = None):
        if checkbox is None:
            checkbox = self._create_particle_show_checkbox(widget_id)
        if checkbox is None:
            return None
        checkbox_name = checkbox.objectName() or f'fitParticle{widget_id}ShowCheckBox'
        checkbox.setObjectName(checkbox_name)
        checkbox.setProperty('particleCheckboxId', widget_id)
        if not hasattr(self.ui, checkbox_name):
            setattr(self.ui, checkbox_name, checkbox)
        if widget_id not in self._particle_show_checkboxes:
            checkbox.toggled.connect(self._on_component_checkbox_changed)
        checkbox.setText(checkbox.text() or f'Particle {widget_id}')
        self._particle_show_checkboxes[widget_id] = checkbox
        return checkbox

    def _create_particle_show_checkbox(self, widget_id: int):
        self._prepare_dynamic_show_checkbox_area()
        parent = self._dynamic_show_container or getattr(self.ui, 'fitFittingShowWidget', None)
        if parent is None:
            return None
        checkbox = QCheckBox(f'Particle {widget_id}', parent)
        checkbox.setObjectName(f'fitParticle{widget_id}ShowCheckBox')
        checkbox.setProperty('particleCheckboxId', widget_id)
        if self._dynamic_show_layout is not None:
            self._insert_particle_checkbox_widget(checkbox, widget_id)
        elif hasattr(parent, 'layout') and parent.layout() is not None:
            parent.layout().addWidget(checkbox)
        return checkbox

    def _insert_particle_checkbox_widget(self, checkbox: QCheckBox, widget_id: int):
        layout = self._dynamic_show_layout
        if layout is None:
            return
        can_insert = hasattr(layout, 'insertWidget')
        inserted = False
        for pos in range(layout.count()):
            item = layout.itemAt(pos)
            if item is None:
                continue
            existing = item.widget()
            if existing is None:
                continue
            existing_id = existing.property('particleCheckboxId')
            if existing_id is None:
                continue
            if widget_id < existing_id and can_insert:
                layout.insertWidget(pos, checkbox)
                inserted = True
                break
        if not inserted:
            layout.addWidget(checkbox)

    def _create_particle_widget(self, widget_id: int) -> QWidget:
        parent = getattr(self, '_particle_scroll_container', getattr(self.ui, 'scrollAreaWidgetContents', self.ui))
        container = QWidget(parent)
        container.setObjectName(f'fitParticleWidget_{widget_id}')
        container.setMinimumSize(180, 300)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        combo = QComboBox(container)
        combo.setObjectName(f'fitParticleShapeCombox_{widget_id}')
        combo.addItems(['Sphere', 'Cylinder', 'None'])
        layout.addWidget(combo)

        stack = QStackedWidget(container)
        stack.setObjectName(f'fitParticleStackWidget_{widget_id}')
        layout.addWidget(stack)

        sphere_page = QWidget(stack)
        sphere_grid = QGridLayout(sphere_page)
        sphere_grid.setContentsMargins(0, 0, 0, 0)
        sphere_fields = [
            ('Int.', f'fitParticleSphereIntValue_{widget_id}'),
            ('R [nm]', f'fitParticleSphereRValue_{widget_id}'),
            ('σ [R]', f'fitParticleSphereSigmaRValue_{widget_id}'),
            ('D [nm]', f'fitParticleSphereDValue_{widget_id}'),
            ('σ [D]', f'fitParticleSphereSigmaDValue_{widget_id}'),
            ('BG', f'fitParticleSphereBGValue_{widget_id}'),
        ]
        for row, (label_text, name) in enumerate(sphere_fields):
            label = QLabel(label_text, sphere_page)
            value = QDoubleSpinBox(sphere_page)
            value.setObjectName(name)
            value.setDecimals(6)
            sphere_grid.addWidget(label, row, 0)
            sphere_grid.addWidget(value, row, 1)
        stack.addWidget(sphere_page)

        cylinder_page = QWidget(stack)
        cyl_grid = QGridLayout(cylinder_page)
        cyl_grid.setContentsMargins(0, 0, 0, 0)
        cylinder_fields = [
            ('Int.', f'fitParticleCylinderIntValue_{widget_id}'),
            ('R [nm]', f'fitParticleCylinderRValue_{widget_id}'),
            ('σ [R]', f'fitParticleCylinderSigmaRValue_{widget_id}'),
            ('h [nm]', f'fitParticleCylinderhValue_{widget_id}'),
            ('σ [h]', f'fitParticleCylinderSigmahValue_{widget_id}'),
            ('D [nm]', f'fitParticleCylinderDValue_{widget_id}'),
            ('σ [D]', f'fitParticleCylinderSigmaDValue_{widget_id}'),
            ('BG', f'fitParticleCylinderBGValue_{widget_id}'),
        ]
        for row, (label_text, name) in enumerate(cylinder_fields):
            label = QLabel(label_text, cylinder_page)
            value = QDoubleSpinBox(cylinder_page)
            value.setObjectName(name)
            value.setDecimals(6)
            cyl_grid.addWidget(label, row, 0)
            cyl_grid.addWidget(value, row, 1)
        stack.addWidget(cylinder_page)
        stack.setCurrentIndex(0)

        self._register_ui_children(container)
        self._apply_particle_widget_style(container, widget_id)
        return container

    def _apply_particle_widget_style(self, widget: QWidget, widget_id: int):
        if widget is None:
            return
        template = self._particle_widget_style_template or self._particle_widget_style_fallback
        if not template:
            return
        target_selector = f'#fitParticleWidget_{widget_id}'
        source_name = self._particle_widget_style_source_name
        style = template
        if source_name and f'#{source_name}' in template:
            style = template.replace(f'#{source_name}', target_selector)
        elif 'fitParticleWidget_TEMPLATE' in template:
            style = template.replace('fitParticleWidget_TEMPLATE', f'fitParticleWidget_{widget_id}')
        widget.setStyleSheet(style)

    def _register_ui_children(self, widget: QWidget):
        if widget is None:
            return
        for child in widget.findChildren(QWidget):
            name = child.objectName()
            if name:
                setattr(self.ui, name, child)
        name = widget.objectName()
        if name:
            setattr(self.ui, name, widget)

    def _on_add_particle_clicked(self):
        try:
            widget_id = self._allocate_particle_id()
            particle_key = f'particle_{widget_id}'
            self.model_params_manager.ensure_particle_entry('fitting', particle_key, shape='Sphere')
            new_widget = self._create_particle_widget(widget_id)
            self._attach_particle_widget(new_widget, widget_id)
            self._add_fitting_success(f"Particle {widget_id} added")
        except Exception as e:
            self._add_fitting_error(f"Failed to add particle widget: {e}")

    def _allocate_particle_id(self) -> int:
        if self._recycled_particle_ids:
            self._recycled_particle_ids.sort()
            return self._recycled_particle_ids.pop(0)
        candidate = getattr(self, '_next_particle_candidate', 1)
        while candidate in self.particle_shape_configs:
            candidate += 1
        self._next_particle_candidate = candidate + 1
        return candidate

    def _attach_particle_widget(self, widget: QWidget, widget_id: int):
        if widget is None:
            return
        if self._particle_container_layout is not None and self._particle_add_button is not None:
            index = self._particle_container_layout.indexOf(self._particle_add_button)
            if index == -1:
                self._particle_container_layout.addWidget(widget)
            else:
                self._particle_container_layout.insertWidget(index, widget)
        elif self._particle_container_layout is not None:
            self._particle_container_layout.addWidget(widget)

        self._particle_widgets[widget_id] = widget
        self.particle_shape_configs[widget_id] = self._build_particle_config(widget_id)
        self._install_particle_context_menu(widget, widget_id)
        self._register_particle_show_checkbox(widget_id)

        self._setup_particle_connections([widget_id])
        self._setup_particle_parameter_connections([widget_id])
        self._setup_parameter_ranges([widget_id])
        self._initialize_particle_states([widget_id])

    def _install_particle_context_menu(self, widget: QWidget, widget_id: int):
        if widget is None:
            return
        widget.setProperty('particle_id', widget_id)
        widget.setContextMenuPolicy(Qt.CustomContextMenu)
        try:
            widget.customContextMenuRequested.connect(self._handle_particle_context_menu_request)
        except Exception:
            pass

    def _handle_particle_context_menu_request(self, pos: QPoint):
        widget = self.sender()
        if widget is None:
            return
        widget_id = widget.property('particle_id')
        if widget_id is None:
            return
        global_pos = widget.mapToGlobal(pos)
        self._show_particle_context_menu(int(widget_id), global_pos)

    def _show_particle_context_menu(self, widget_id: int, global_pos: QPoint):
        menu = QMenu(self.ui)
        remove_action = menu.addAction('Remove Particle')
        if len(self._iter_particle_widget_ids()) <= 1:
            remove_action.setEnabled(False)
        action = menu.exec_(global_pos)
        if action == remove_action:
            self._remove_particle_widget(widget_id)

    def _remove_particle_widget(self, widget_id: int):
        if widget_id not in self.particle_shape_configs:
            return
        if len(self._iter_particle_widget_ids()) <= 1:
            self._add_fitting_warning('At least one particle widget must remain')
            return

        widget = self._particle_widgets.pop(widget_id, None)
        if widget is not None:
            try:
                widget.customContextMenuRequested.disconnect(self._handle_particle_context_menu_request)
            except Exception:
                pass
            if self._particle_container_layout is not None:
                self._particle_container_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()

        checkbox = self._particle_show_checkboxes.pop(widget_id, None)
        if checkbox is not None:
            checkbox.setParent(None)
            checkbox.deleteLater()
            cb_name = checkbox.objectName()
            if cb_name and hasattr(self.ui, cb_name):
                delattr(self.ui, cb_name)

        meta_ids = self._particle_parameter_meta_ids.pop(widget_id, [])
        for meta_id in meta_ids:
            try:
                self.param_trigger_manager.unregister_widget(meta_id)
            except Exception:
                pass

        self._cleanup_particle_ui_attributes(widget_id)
        self.particle_shape_configs.pop(widget_id, None)
        self._recycled_particle_ids.append(widget_id)
        self.model_params_manager.remove_particle('fitting', f'particle_{widget_id}')
        self.model_params_manager.save_parameters()
        self._last_active_particle_ids = [wid for wid in self._last_active_particle_ids if wid != widget_id]

        try:
            self._update_GUI_image('fitting' if self._is_in_fitting_mode() else 'normal')
        except Exception:
            pass
        self._add_fitting_success(f"Particle {widget_id} removed")

    def _cleanup_particle_ui_attributes(self, widget_id: int):
        names = [
            f'fitParticleWidget_{widget_id}',
            f'fitParticleShapeCombox_{widget_id}',
            f'fitParticleStackWidget_{widget_id}',
        ]
        for shape in ('Sphere', 'Cylinder'):
            mapping = self._get_parameter_widget_mapping(widget_id, shape)
            names.extend(mapping.values())
        for name in names:
            if hasattr(self.ui, name):
                try:
                    attr = getattr(self.ui, name)
                    if hasattr(attr, 'deleteLater'):
                        attr.deleteLater()
                except Exception:
                    pass
                try:
                    delattr(self.ui, name)
                except Exception:
                    pass

    def _setup_particle_connections(self, widget_ids=None):
        """设置粒子形状连接，可指定widget列表"""
        widget_ids = widget_ids or self._iter_particle_widget_ids()
        for widget_id in widget_ids:
            config = self.particle_shape_configs[widget_id]
            if hasattr(self.ui, config['combobox']):
                combobox = getattr(self.ui, config['combobox'])
                
                # 连接ComboBox到页面切换函数
                combobox.currentIndexChanged.connect(
                    lambda index, wid=widget_id: self._on_particle_shape_changed(wid, index)
                )
                
                self._add_fitting_message(f"Connected Particle Widget {widget_id}: {config['combobox']} -> {config['stack_widget']}", "INFO")
    
    def _setup_parameter_ranges(self, widget_ids=None):
        """设置参数控件的数值范围为实数域，可限定粒子"""
        # 设置一个很大的范围来覆盖实数域（Python float的范围）
        min_value = -1e10  # 负一百亿
        max_value = 1e10   # 正一百亿
        decimals = 2       # 2位小数精度
        
        # 参数控件计数器
        widgets_set = 0
        
        # 设置粒子参数控件范围
        widget_ids = widget_ids or self._iter_particle_widget_ids()
        for widget_id in widget_ids:
            # 球形参数控件
            sphere_mapping = self._get_parameter_widget_mapping(widget_id, 'Sphere')
            for param_key, widget_name in sphere_mapping.items():
                if hasattr(self.ui, widget_name):
                    widget = getattr(self.ui, widget_name)
                    widget.setRange(min_value, max_value)
                    # 为BG和Int参数设置6位精度
                    if 'BG' in widget_name or 'Int' in widget_name:
                        widget.setDecimals(6)
                        widget.setSingleStep(0.1)
                    else:
                        widget.setDecimals(decimals)
                        widget.setSingleStep(0.1)  # 设置单步大小
                    widgets_set += 1
                    
            # 圆柱形参数控件
            cylinder_mapping = self._get_parameter_widget_mapping(widget_id, 'Cylinder')
            for param_key, widget_name in cylinder_mapping.items():
                if hasattr(self.ui, widget_name):
                    widget = getattr(self.ui, widget_name)
                    widget.setRange(min_value, max_value)
                    # 为BG和Int参数设置6位精度
                    if 'BG' in widget_name or 'Int' in widget_name:
                        widget.setDecimals(6)
                        widget.setSingleStep(0.1)
                    else:
                        widget.setDecimals(decimals)
                        widget.setSingleStep(0.1)  # 设置单步大小
                    widgets_set += 1
        
        # 设置全局拟合参数控件范围
        if hasattr(self.ui, 'fitSigmaResValue'):
            self.ui.fitSigmaResValue.setRange(min_value, max_value)
            self.ui.fitSigmaResValue.setDecimals(6)  # Br 可能非常小
            self.ui.fitSigmaResValue.setSingleStep(0.1)
            widgets_set += 1

        if hasattr(self.ui, 'fitNuResValue'):
            self.ui.fitNuResValue.setRange(min_value, max_value)
            self.ui.fitNuResValue.setDecimals(4)
            self.ui.fitNuResValue.setSingleStep(0.1)
            widgets_set += 1

        if hasattr(self.ui, 'fitIntResValue'):
            self.ui.fitIntResValue.setRange(min_value, max_value)
            self.ui.fitIntResValue.setDecimals(6)
            self.ui.fitIntResValue.setSingleStep(0.01)
            widgets_set += 1
            
        if hasattr(self.ui, 'fitKValue'):
            self.ui.fitKValue.setRange(min_value, max_value)
            self.ui.fitKValue.setDecimals(4)  # k值设置4位精度
            self.ui.fitKValue.setSingleStep(0.1)  # k值通常接近1
            widgets_set += 1
        
        self._add_fitting_success(f"Set ranges for {widgets_set} parameter widgets: [{min_value}, {max_value}] with {decimals} decimals")
    
    def _setup_particle_parameter_connections(self, widget_ids=None):
        """设置粒子参数控件的信号连接（迁移到 meta 去抖 + 持久化 + 拟合触发）"""
        from functools import partial
        widget_ids = widget_ids or self._iter_particle_widget_ids()
        for widget_id in widget_ids:
            for shape_name in ('Sphere', 'Cylinder'):
                mapping = self._get_parameter_widget_mapping(widget_id, shape_name)
                shape_lower = shape_name.lower()
                for param_key, widget_name in mapping.items():
                    if not hasattr(self.ui, widget_name):
                        continue
                    w = getattr(self.ui, widget_name)
                    def _after_commit(info, value, wid=widget_id, shp=shape_lower, p=param_key):
                        try:
                            self._add_particle_message(f"Meta commit {wid}.{shp}.{p} = {value}")
                            # 逻辑：参数变化后触发一维拟合曲线更新（等同手动拟合按钮）
                            # 条件：若当前有可用数据 (cut 或 1d)；如果没有数据则忽略
                            has_data = (hasattr(self, 'current_cut_data') and self.current_cut_data is not None) or \
                                       (hasattr(self, 'current_1d_data') and self.current_1d_data is not None)
                            if has_data:
                                # 修改拟合参数意味着用户意图进行拟合显示：先进入拟合模式
                                try:
                                    self.display_mode = 'fitting'
                                    self._display_mode = 'fitting'
                                    self._fitting_mode_active = True
                                except Exception:
                                    pass
                                # 进入拟合模式或保持当前模式下刷新 1D 曲线
                                self._perform_manual_fitting()
                        except Exception:
                            pass
                    widget_mode = self._signal_mode_overrides.get(widget_name, self._default_signal_mode)
                    meta = {
                        'persist': 'model_particle',
                        'particle_id': f'particle_{widget_id}',
                        'shape': shape_lower,
                        'param': param_key,
                        'trigger_fit': True,
                        'debounce_ms': self._param_debounce_ms,
                        'epsilon_abs': self._param_abs_eps,
                        'epsilon_rel': self._param_rel_eps,
                        'after_commit': _after_commit,
                        'connect_mode': widget_mode,
                    }
                    meta_id = f'meta_particle_{widget_id}_{shape_lower}_{param_key}'
                    self.param_trigger_manager.register_parameter_widget(
                        widget=w,
                        widget_id=meta_id,
                        category='fitting_particles',
                        immediate_handler=lambda v: None,
                        delayed_handler=None,
                        connect_signals=True,
                        meta=meta
                    )
                    self._particle_parameter_meta_ids[widget_id].append(meta_id)
                    # 由 meta 管理器根据 connect_mode 自动连接
    
    def _setup_global_parameter_connections(self):
        """设置全局拟合参数控件的信号连接（meta 去抖）"""
        from functools import partial
        mapping = [
            ('fitSigmaResValue', 'sigma_res'),
            ('fitNuResValue', 'nu_res'),
            ('fitIntResValue', 'int_res'),
            ('fitKValue', 'k_value'),
        ]
        for widget_name, param_key in mapping:
            if not hasattr(self.ui, widget_name):
                continue
            w = getattr(self.ui, widget_name)
            def _after_commit(info, value, p=param_key):
                try:
                    self._add_particle_message(f"Meta commit global {p} = {value}")
                    has_data = (hasattr(self, 'current_cut_data') and self.current_cut_data is not None) or \
                               (hasattr(self, 'current_1d_data') and self.current_1d_data is not None)
                    if has_data:
                        # 修改全局拟合参数同样视作拟合模式
                        try:
                            self.display_mode = 'fitting'
                            self._display_mode = 'fitting'
                            self._fitting_mode_active = True
                        except Exception:
                            pass
                        self._perform_manual_fitting()
                except Exception:
                    pass
            widget_mode = self._signal_mode_overrides.get(widget_name, self._default_signal_mode)
            meta = {
                'persist': 'model_global',
                'param': param_key,
                'trigger_fit': True,
                'debounce_ms': self._param_debounce_ms,
                'epsilon_abs': self._param_abs_eps,
                'epsilon_rel': self._param_rel_eps,
                'after_commit': _after_commit,
                'connect_mode': widget_mode,
            }
            self.param_trigger_manager.register_parameter_widget(
                widget=w,
                widget_id=f'meta_global_{param_key}',
                category='fitting_global',
                immediate_handler=lambda v: None,
                delayed_handler=None,
                connect_signals=True,
                meta=meta
            )
            # 由 meta 管理器根据 connect_mode 自动连接
            self._add_fitting_message(f"Connected (meta, mode={self._signal_mode_overrides.get(widget_name, self._default_signal_mode)}) {widget_name}", "INFO")

    
    def _initialize_global_parameters(self):
        """根据模型参数初始化全局拟合参数"""
        try:
            # 初始化 sigma_res
            if hasattr(self.ui, 'fitSigmaResValue'):
                saved_value = self.model_params_manager.get_global_parameter('fitting', 'sigma_res', 0.1)
                self.ui.fitSigmaResValue.blockSignals(True)
                self.ui.fitSigmaResValue.setValue(saved_value)
                self.ui.fitSigmaResValue.blockSignals(False)
                self._add_fitting_message(f"Initialized fitSigmaResValue to {saved_value}", "INFO")

            # 初始化 nu_res
            if hasattr(self.ui, 'fitNuResValue'):
                saved_value = self.model_params_manager.get_global_parameter('fitting', 'nu_res', 5.0)
                self.ui.fitNuResValue.blockSignals(True)
                self.ui.fitNuResValue.setValue(saved_value)
                self.ui.fitNuResValue.blockSignals(False)
                self._add_fitting_message(f"Initialized fitNuResValue to {saved_value}", "INFO")

            # 初始化 int_res
            if hasattr(self.ui, 'fitIntResValue'):
                saved_value = self.model_params_manager.get_global_parameter('fitting', 'int_res', 0.0)
                self.ui.fitIntResValue.blockSignals(True)
                self.ui.fitIntResValue.setValue(saved_value)
                self.ui.fitIntResValue.blockSignals(False)
                self._add_fitting_message(f"Initialized fitIntResValue to {saved_value}", "INFO")
            
            # 初始化 k_value
            if hasattr(self.ui, 'fitKValue'):
                saved_value = self.model_params_manager.get_global_parameter('fitting', 'k_value', 1.0)
                self.ui.fitKValue.blockSignals(True)
                self.ui.fitKValue.setValue(saved_value)
                self.ui.fitKValue.blockSignals(False)
                self._add_fitting_message(f"Initialized fitKValue to {saved_value}", "INFO")
                
        except Exception as e:
            self._add_fitting_error(f"Failed to initialize global parameters: {e}")
    
    def _initialize_particle_states(self, widget_ids=None):
        """根据模型参数初始化粒子状态：直接从JSON读取，可限定粒子"""
        try:
            # 设置初始化标志，避免触发保存
            self._initializing = True
            
            target_ids = widget_ids or self._iter_particle_widget_ids()
            for widget_id in target_ids:
                particle_id = f"particle_{widget_id}"
                
                # 从JSON获取保存的形状和启用状态
                saved_shape = self.model_params_manager.get_particle_shape('fitting', particle_id)
                is_enabled = self.model_params_manager.get_particle_enabled('fitting', particle_id)
                
                self._add_fitting_message(f"Initializing {particle_id}: shape={saved_shape}, enabled={is_enabled}", "INFO")
                
                # 设置ComboBox选择（不触发信号处理函数）
                config = self.particle_shape_configs[widget_id]
                if hasattr(self.ui, config['combobox']):
                    combobox = getattr(self.ui, config['combobox'])
                    
                    # 找到对应的combo index
                    combo_index = None
                    for index, page_config in config['pages'].items():
                        if page_config['name'] == saved_shape:
                            combo_index = index
                            break
                    
                    if combo_index is not None:
                        # 暂时断开信号连接
                        combobox.blockSignals(True)
                        combobox.setCurrentIndex(combo_index)
                        combobox.blockSignals(False)
                        
                        # 设置页面
                        page_config = config['pages'][combo_index]
                        self._switch_particle_page(widget_id, page_config, saved_shape)
                        
                        # 根据启用状态设置控件
                        if not is_enabled or saved_shape == 'None':
                            # 冻结控件
                            self._freeze_particle_controls(widget_id)
                            self._add_fitting_message(f"{particle_id} controls frozen (disabled/None)", "INFO")
                        else:
                            # 解冻控件并加载参数
                            self._unfreeze_particle_controls(widget_id, saved_shape)
                            self._load_particle_parameters_from_json(widget_id, saved_shape)
                            self._add_fitting_message(f"{particle_id} controls active with {saved_shape} parameters", "INFO")
                
        except Exception as e:
            self._add_fitting_error(f"Failed to initialize particle states: {e}")
            import traceback
            self._add_fitting_error(f"Traceback: {traceback.format_exc()}")
        finally:
            # 重置初始化标志
            self._initializing = False
    
    def _set_particle_page_and_state(self, widget_id: int, combo_index: int, shape_name: str):
        """设置粒子页面和状态（初始化时使用，不保存到模型参数）"""
        config = self.particle_shape_configs[widget_id]
        page_config = config['pages'][combo_index]
        
        if hasattr(self.ui, config['stack_widget']):
            stack_widget = getattr(self.ui, config['stack_widget'])
            
            # 切换页面
            stack_widget.setCurrentIndex(page_config['page_index'])
            
            # 处理控件状态
            if shape_name == 'None':
                self._set_particle_none_state(widget_id)
            else:
                self._set_particle_active_state(widget_id, shape_name)
    
    def _load_particle_parameters(self, widget_id: int, shape_name: str):
        """从模型参数加载粒子参数到UI控件"""
        if shape_name == 'None':
            return
            
        try:
            particle_id = f"particle_{widget_id}"
            shape_lower = shape_name.lower()
            
            # 获取参数映射
            param_mapping = self._get_parameter_widget_mapping(widget_id, shape_name)
            
            # 从模型参数获取值并设置到控件
            for param_key, widget_name in param_mapping.items():
                if hasattr(self.ui, widget_name):
                    widget = getattr(self.ui, widget_name)
                    
                    # 从模型参数获取值
                    value = self.model_params_manager.get_particle_parameter(
                        'fitting', particle_id, shape_lower, param_key
                    )
                    
                    if value is not None:
                        # 暂时断开信号以避免触发保存
                        widget.blockSignals(True)
                        widget.setValue(value)
                        widget.blockSignals(False)
                        
                        self._add_particle_message(f"Loaded {param_key}={value} for particle {widget_id} ({shape_name})")
                    else:
                        self._add_particle_message(f"No value found for {param_key} in particle {widget_id} ({shape_name})")
                        
        except Exception as e:
            self._add_fitting_error(f"Failed to load parameters for particle {widget_id}: {e}")
    
    def _get_parameter_widget_mapping(self, widget_id: int, shape_name: str) -> dict:
        """获取参数到控件的映射关系"""
        mapping = {}
        
        if shape_name == 'Sphere':
            mapping = {
                'intensity': f'fitParticleSphereIntValue_{widget_id}',
                'radius': f'fitParticleSphereRValue_{widget_id}',
                'sigma_radius': f'fitParticleSphereSigmaRValue_{widget_id}',
                'diameter': f'fitParticleSphereDValue_{widget_id}',
                'sigma_diameter': f'fitParticleSphereSigmaDValue_{widget_id}',
                'background': f'fitParticleSphereBGValue_{widget_id}'
            }
        elif shape_name == 'Cylinder':
            mapping = {
                'intensity': f'fitParticleCylinderIntValue_{widget_id}',
                'radius': f'fitParticleCylinderRValue_{widget_id}',
                'sigma_radius': f'fitParticleCylinderSigmaRValue_{widget_id}',
                'height': f'fitParticleCylinderhValue_{widget_id}',
                'sigma_height': f'fitParticleCylinderSigmahValue_{widget_id}',
                'diameter': f'fitParticleCylinderDValue_{widget_id}',
                'sigma_diameter': f'fitParticleCylinderSigmaDValue_{widget_id}',
                'background': f'fitParticleCylinderBGValue_{widget_id}'
            }
        
        return mapping
    
    def _on_particle_shape_changed(self, widget_id: int, combo_index: int):
        """处理粒子形状改变事件 - 重构版本：直接操作JSON"""
        config = self.particle_shape_configs[widget_id]
        page_config = config['pages'][combo_index]
        
        if hasattr(self.ui, config['stack_widget']):
            stack_widget = getattr(self.ui, config['stack_widget'])
            shape_name = page_config['name']
            
            # 获取当前形状状态
            particle_id = f"particle_{widget_id}"
            current_shape = self.model_params_manager.get_particle_shape('fitting', particle_id)
            
            # 检查是否真的需要切换
            if current_shape == shape_name:
                self._add_particle_message(f"⚠️ Particle {widget_id} already in {shape_name} state, skipping")
                return
            
            self._add_particle_message(f"🔄 Particle {widget_id} shape changing: {current_shape} -> {shape_name}")
            
            # 1. 更新JSON中的形状和启用状态
            if shape_name == 'None':
                # 切换到None：设置为disabled状态
                self.model_params_manager.set_particle_shape('fitting', particle_id, 'None')
                self.model_params_manager.set_particle_enabled('fitting', particle_id, False)
                self._add_particle_message(f"💾 Saved {particle_id} as None (disabled)")
            else:
                # 切换到具体形状：设置为enabled状态
                self.model_params_manager.set_particle_shape('fitting', particle_id, shape_name)
                self.model_params_manager.set_particle_enabled('fitting', particle_id, True)
                self._add_particle_message(f"� Saved {particle_id} as {shape_name} (enabled)")
            
            # 保存JSON文件
            self.model_params_manager.save_parameters()
            
            # 2. 切换UI页面
            self._switch_particle_page(widget_id, page_config, shape_name)
            
            # 3. 处理控件状态和参数加载
            if shape_name == 'None':
                # None状态：冻结所有控件
                self._freeze_particle_controls(widget_id)
                self._add_particle_message(f"❄️ Particle {widget_id} controls frozen (None state)")
            else:
                # 具体形状：解冻控件并加载参数
                self._unfreeze_particle_controls(widget_id, shape_name)
                self._load_particle_parameters_from_json(widget_id, shape_name)
                self._add_particle_message(f"🔓 Particle {widget_id} controls unfrozen ({shape_name} state)")
    
    def _switch_particle_page(self, widget_id: int, page_config: dict, shape_name: str):
        """切换粒子页面"""
        config = self.particle_shape_configs[widget_id]
        if hasattr(self.ui, config['stack_widget']):
            stack_widget = getattr(self.ui, config['stack_widget'])
            target_page_index = page_config['page_index']
            current_page_index = stack_widget.currentIndex()
            
            self._add_particle_message(f"🔄 Switching page: {current_page_index} -> {target_page_index} for {shape_name}")
            
            # 强制页面切换（处理相同索引的情况）
            if target_page_index == current_page_index:
                # 临时切换到不同页面再切回来，确保UI刷新
                temp_page_index = 1 if target_page_index == 0 else 0
                stack_widget.setCurrentIndex(temp_page_index)
                from PyQt5.QtWidgets import QApplication
                QApplication.processEvents()
                stack_widget.setCurrentIndex(target_page_index)
                self._add_particle_message(f"🔄 Forced refresh: temp({temp_page_index}) -> {target_page_index}")
            else:
                stack_widget.setCurrentIndex(target_page_index)
            
            # 验证页面切换成功
            final_page_index = stack_widget.currentIndex()
            if final_page_index == target_page_index:
                self._add_particle_message(f"✅ Page switch confirmed: {final_page_index}")
            else:
                self._add_particle_message(f"❌ Page switch failed! Expected {target_page_index}, got {final_page_index}")
    
    def _freeze_particle_controls(self, widget_id: int):
        """冻结粒子的所有控件（None状态）"""
        for shape_name in ['Sphere', 'Cylinder']:
            param_mapping = self._get_parameter_widget_mapping(widget_id, shape_name)
            for param_key, widget_name in param_mapping.items():
                if hasattr(self.ui, widget_name):
                    widget = getattr(self.ui, widget_name)
                    widget.setEnabled(False)
    
    def _unfreeze_particle_controls(self, widget_id: int, active_shape: str):
        """解冻粒子控件（激活特定形状，冻结其他形状）"""
        for shape_name in ['Sphere', 'Cylinder']:
            param_mapping = self._get_parameter_widget_mapping(widget_id, shape_name)
            is_active = (shape_name == active_shape)
            
            for param_key, widget_name in param_mapping.items():
                if hasattr(self.ui, widget_name):
                    widget = getattr(self.ui, widget_name)
                    widget.setEnabled(is_active)
    
    def _load_particle_parameters_from_json(self, widget_id: int, shape_name: str):
        """从JSON直接加载粒子参数到UI"""
        try:
            # 设置载入标志，防止触发保存
            self._loading_parameters = True
            
            particle_id = f"particle_{widget_id}"
            shape_params = self.model_params_manager.get_particle_parameter('fitting', particle_id, shape_name.lower())
            
            if not shape_params:
                self._add_particle_message(f"⚠️ No parameters found in JSON for {particle_id}.{shape_name}")
                return
            
            # 获取参数映射
            param_mapping = self._get_parameter_widget_mapping(widget_id, shape_name)
            loaded_count = 0
            
            # 直接设置参数到UI控件
            for param_key, widget_name in param_mapping.items():
                if param_key in shape_params and hasattr(self.ui, widget_name):
                    widget = getattr(self.ui, widget_name)
                    value = shape_params[param_key]
                    widget.setValue(value)
                    loaded_count += 1
                    self._add_particle_message(f"📥 Loaded {param_key}={value} for {particle_id}.{shape_name}")
            
            self._add_particle_message(f"✅ Loaded {loaded_count} parameters from JSON for {particle_id}.{shape_name}")
            
        except Exception as e:
            self._add_fitting_error(f"Failed to load parameters from JSON: {e}")
        finally:
            # 重置载入标志
            self._loading_parameters = False
    

    

    

    

    
    def set_particle_shape(self, widget_id: int, shape_name: str):
        """
        程序化设置粒子形状
        
        Args:
            widget_id: 粒子编号 (1, 2, 3)
            shape_name: 形状名称 ('Sphere', 'Cylinder', 'None')
        """
        config = self.particle_shape_configs.get(widget_id)
        if not config:
            self._add_fitting_warning(f"Particle Widget {widget_id} not found")
            return False
        
        # 找到对应的combo index
        combo_index = None
        for index, page_config in config['pages'].items():
            if page_config['name'] == shape_name:
                combo_index = index
                break
        
        if combo_index is None:
            self._add_fitting_warning(f"Shape {shape_name} not found for particle widget {widget_id}")
            return False
        
        # 设置ComboBox
        if hasattr(self.ui, config['combobox']):
            combobox = getattr(self.ui, config['combobox'])
            combobox.setCurrentIndex(combo_index)
            return True
        
        return False
    
    def get_particle_shape(self, widget_id: int) -> str:
        """
        获取粒子当前形状
        
        Args:
            widget_id: 粒子编号 (1, 2, 3)
            
        Returns:
            当前形状名称
        """
        config = self.particle_shape_configs.get(widget_id)
        if not config:
            return 'None'
        
        if hasattr(self.ui, config['combobox']):
            combobox = getattr(self.ui, config['combobox'])
            current_index = combobox.currentIndex()
            
            page_config = config['pages'].get(current_index)
            if page_config:
                return page_config['name']
        
        return 'None'
    
    def get_particles_status(self) -> dict:
        """获取所有粒子的当前状态"""
        status = {}
        for widget_id in self._iter_particle_widget_ids():
            status[widget_id] = self.get_particle_shape(widget_id)
        return status
    
    def reset_all_particles(self):
        """重置所有粒子为None状态"""
        for widget_id in self._iter_particle_widget_ids():
            self.set_particle_shape(widget_id, 'None')
        self._add_fitting_success("All particles reset to None state")
    
    def add_new_particle_shape(self, shape_name: str, control_types: list):
        """
        添加新的粒子形状配置 - 方便扩展
        
        Args:
            shape_name: 形状名称，如 'Ellipsoid'
            control_types: 控件类型列表，如 ['Int', 'Ra', 'Rb', 'Rc', 'D', 'BG']
        """
        self.particle_control_types[shape_name] = control_types
        
        # 为每个widget添加新的页面配置
        for widget_id in self._iter_particle_widget_ids():
            pages = self.particle_shape_configs[widget_id]['pages']
            new_index = len(pages) - 1  # None总是最后一个，所以新形状插在倒数第二
            
            # 重新排列索引，把None移到最后
            none_config = pages.pop(len(pages) - 1)  # 移除None
            
            # 添加新形状
            pages[new_index] = {
                'name': shape_name,
                'page_index': new_index,  # 假设新页面索引等于配置索引
            }
            
            # 重新添加None
            pages[len(pages)] = none_config
        
        self._add_fitting_success(f"Added new particle shape: {shape_name} with {len(control_types)} controls")
        self._add_fitting_warning("Note: You need to add corresponding UI pages and ComboBox items manually")
    
    def get_all_particle_parameters(self) -> dict:
        """获取所有粒子的完整参数信息"""
        return self.model_params_manager.get_all_particles('fitting')
    
    def save_particle_parameters(self) -> bool:
        """手动保存粒子参数到文件"""
        return self.model_params_manager.save_parameters()
    
    def reload_particle_parameters(self) -> bool:
        """重新加载粒子参数并更新UI"""
        success = self.model_params_manager.load_parameters()
        if success:
            self._initialize_particle_states()
            self._initialize_global_parameters()
            self._add_fitting_success("Particle and global parameters reloaded from file")
        else:
            self._add_fitting_error("Failed to reload particle parameters")
        return success
    
    def export_particle_parameters(self, filepath: str) -> bool:
        """导出粒子参数到指定文件"""
        try:
            import shutil
            shutil.copy2(self.model_params_manager.config_file, filepath)
            self._add_fitting_success(f"Particle parameters exported to: {filepath}")
            return True
        except Exception as e:
            self._add_fitting_error(f"Failed to export parameters: {e}")
            return False
    
    def import_particle_parameters(self, filepath: str) -> bool:
        """从指定文件导入粒子参数"""
        try:
            import shutil
            shutil.copy2(filepath, self.model_params_manager.config_file)
            success = self.reload_particle_parameters()
            if success:
                self._add_fitting_success(f"Particle parameters imported from: {filepath}")
            return success
        except Exception as e:
            self._add_fitting_error(f"Failed to import parameters: {e}")
            return False
    
    def get_global_parameter(self, param: str) -> float:
        """获取全局拟合参数值"""
        return self.model_params_manager.get_global_parameter('fitting', param, 0.0)
    
    def set_global_parameter(self, param: str, value: float) -> bool:
        """设置全局拟合参数值"""
        success = self.model_params_manager.set_global_parameter('fitting', param, value)
        if success:
            # 更新UI控件（如果存在）
            if param == 'sigma_res' and hasattr(self.ui, 'fitSigmaResValue'):
                self.ui.fitSigmaResValue.blockSignals(True)
                self.ui.fitSigmaResValue.setValue(value)
                self.ui.fitSigmaResValue.blockSignals(False)
            elif param == 'k_value' and hasattr(self.ui, 'fitKValue'):
                self.ui.fitKValue.blockSignals(True)
                self.ui.fitKValue.setValue(value)
                self.ui.fitKValue.blockSignals(False)
            
            # 保存参数
            self.model_params_manager.save_parameters()
        return success
    
    def get_all_global_parameters(self) -> dict:
        """获取所有全局拟合参数"""
        return self.model_params_manager.get_all_global_parameters('fitting')
    
    def reset_global_parameters(self):
        """重置全局参数为默认值"""
        self.set_global_parameter('sigma_res', 0.1)
        self.set_global_parameter('k_value', 1.0)
        self._add_fitting_success("Global parameters reset to default values")
    
    # ================================
    # 用户设置管理
    # ================================
    
    def _save_auto_k_enabled(self):
        """保存auto-K按钮状态到用户设置"""
        try:
            from core.user_settings import user_settings
            user_settings.set('_auto_k_enabled', self._auto_k_enabled)
            user_settings.save_settings()
        except Exception as e:
            print(f"Failed to save auto-K setting: {e}")
    
    def _load_auto_k_enabled(self):
        """从用户设置加载auto-K按钮状态"""
        try:
            from core.user_settings import user_settings
            self._auto_k_enabled = user_settings.get('_auto_k_enabled', False)
            self._update_auto_k_button_style()
        except Exception as e:
            print(f"Failed to load auto-K setting: {e}")
            self._auto_k_enabled = False
    
    def _update_auto_k_button_style(self):
        """更新auto-K按钮的视觉状态"""
        if hasattr(self.ui, 'FittingAutoKButton'):
            if self._auto_k_enabled:
                self.ui.FittingAutoKButton.setStyleSheet(
                    "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
                )
                self.ui.FittingAutoKButton.setText("<-Auto-K: ON")
            else:
                self.ui.FittingAutoKButton.setStyleSheet("")
                self.ui.FittingAutoKButton.setText("<-Auto-K: OFF")
    
    def _on_auto_k_button_clicked(self):
        """处理auto-K按钮点击事件"""
        # 切换状态
        self._auto_k_enabled = not self._auto_k_enabled
        
        # 保存状态
        self._save_auto_k_enabled()
        
        # 更新按钮样式
        self._update_auto_k_button_style()
        
        # 添加状态信息
        status = "enabled" if self._auto_k_enabled else "disabled"
        self._add_fitting_message(f"Auto K-value optimization {status}")
        
        # 如果启用了auto-K且当前有拟合数据，立即执行一次优化
        if self._auto_k_enabled and hasattr(self, 'I') and hasattr(self, 'I_fitting'):
            if self.I is not None and self.I_fitting is not None:
                self._optimize_k_value()
    
    def _optimize_k_value(self):
        """简单稳定的K值优化：基于原始数据的解析解"""
        try:
            if not hasattr(self, 'I') or not hasattr(self, 'I_fitting') or \
               self.I is None or self.I_fitting is None:
                self._add_fitting_error("No fitting data available for K-value optimization")
                return
            
            # 验证数据形状和有效性
            if self.I.size == 0 or self.I_fitting.size == 0:
                self._add_fitting_error("Empty data arrays for K-value optimization")
                return
                
            if self.I.shape != self.I_fitting.shape:
                self._add_fitting_error(f"Data shape mismatch: I{self.I.shape} vs I_fitting{self.I_fitting.shape}")
                return
            
            # 检查数据中是否有NaN或无穷值
            if np.any(~np.isfinite(self.I)) or np.any(~np.isfinite(self.I_fitting)):
                self._add_fitting_error("Data contains NaN or infinite values")
                return
            
            # 获取当前K值
            current_k = self.get_global_parameter('k_value')
            self._add_fitting_message(f"Starting K-value optimization from {current_k:.6f}...")
            
            # 若启用了ROI，仅在ROI范围内进行K优化
            try:
                q_arr = np.asarray(self.q) if hasattr(self, 'q') and self.q is not None else None
            except Exception:
                q_arr = None
            roi_min = getattr(self, '_roi_min', None)
            roi_max = getattr(self, '_roi_max', None)

            I_exp_full = self.I
            I_fit_full = self.I_fitting

            if (
                q_arr is not None and q_arr.size == I_exp_full.size == I_fit_full.size and
                roi_min is not None and roi_max is not None and
                np.isfinite(roi_min) and np.isfinite(roi_max) and roi_min < roi_max
            ):
                mask = (
                    np.isfinite(q_arr) & np.isfinite(I_exp_full) & np.isfinite(I_fit_full) &
                    (q_arr >= float(roi_min)) & (q_arr <= float(roi_max))
                )
                if np.any(mask):
                    I_exp_used = I_exp_full[mask]
                    I_fit_used = I_fit_full[mask]
                else:
                    I_exp_used = I_exp_full
                    I_fit_used = I_fit_full
            else:
                I_exp_used = I_exp_full
                I_fit_used = I_fit_full

            # 步骤1：从拟合数据中提取基函数（注意后续均基于ROI子集 I_exp_used/I_fit_used）
            k_safe = max(abs(current_k), 1e-12)  # 避免除零
            I_base = I_fit_used / k_safe
            
            # 步骤2：计算最优K值的解析解
            # 最小化 ||k * I_base - I_exp||^2 对k求导 = 0  
            # 解得: k_opt = (I_base · I_exp) / (I_base · I_base)
            I_exp_flat = I_exp_used.flatten()
            I_base_flat = I_base.flatten()
            
            # 过滤掉无效值
            valid_mask = np.isfinite(I_exp_flat) & np.isfinite(I_base_flat) & (I_base_flat != 0)
            
            if not np.any(valid_mask):
                self._add_fitting_error("No valid data points for K optimization")
                return
            
            I_exp_valid = I_exp_flat[valid_mask]
            I_base_valid = I_base_flat[valid_mask]
            
            # 计算解析解
            numerator = np.dot(I_base_valid, I_exp_valid)
            denominator = np.dot(I_base_valid, I_base_valid)
            
            if denominator <= 1e-12:
                self._add_fitting_error("Base function has zero norm, cannot optimize K")
                return
            
            k_opt = numerator / denominator
            
            # 确保K值为正
            if k_opt <= 0:
                # 使用非负最小二乘
                try:
                    from scipy.optimize import nnls
                    # 将问题重新表述为 A*k = b，其中 A = I_base_valid.reshape(-1,1), b = I_exp_valid
                    A = I_base_valid.reshape(-1, 1)
                    b = I_exp_valid
                    k_nnls, residual = nnls(A, b)
                    k_opt = k_nnls[0] if len(k_nnls) > 0 else 1.0
                    optimization_method = "NNLS"
                except ImportError:
                    self._add_fitting_warning("scipy.optimize.nnls not available, using absolute value of analytical solution")
                    k_opt = abs(k_opt)
                    optimization_method = "Analytical (abs)"
            else:
                optimization_method = "Analytical"
            
            # 验证优化结果
            if not np.isfinite(k_opt) or k_opt <= 0:
                self._add_fitting_error(f"Invalid optimized K-value: {k_opt}")
                return
            
            # 计算优化前后的残差
            residual_before = np.sum((current_k * I_base_valid - I_exp_valid) ** 2)
            residual_after = np.sum((k_opt * I_base_valid - I_exp_valid) ** 2)
            
            # 步骤3：更新拟合数据
            if I_base.size == self.I_fitting.size:
                # 未使用子集，直接整体更新
                self.I_fitting = k_opt * (self.I_fitting / k_safe)
            else:
                # 使用了ROI子集：仅在ROI范围内更新拟合数据，其他区段按比例同步
                try:
                    if 'mask' in locals() and mask is not None and mask.size == self.I_fitting.size:
                        I_base_full = self.I_fitting / k_safe
                        I_base_full = np.asarray(I_base_full)
                        I_base_full[mask] = I_base  # ROI处替换为子集对应的基函数
                        self.I_fitting = k_opt * I_base_full
                    else:
                        # 找不到一致的mask，退化为整体更新
                        self.I_fitting = k_opt * (self.I_fitting / k_safe)
                except Exception:
                    self.I_fitting = k_opt * (self.I_fitting / k_safe)
            
            # 步骤4：保存K值到参数系统
            success = self.set_global_parameter('k_value', k_opt)
            if not success:
                self._add_fitting_error("Failed to set optimized K-value")
                return
            
            # 步骤5：同步更新UI控件中的K值显示
            if hasattr(self.ui, 'fitKValue'):
                # 临时阻止信号，避免触发递归优化
                self.ui.fitKValue.blockSignals(True)
                self.ui.fitKValue.setValue(k_opt)
                self.ui.fitKValue.blockSignals(False)
                self._add_fitting_message(f"UI K-value updated to {k_opt:.6f}")

            # 同步更新当前拟合元数据中的 k 值，确保叠加组件使用最新的 k
            try:
                if isinstance(getattr(self, 'fitting', None), dict):
                    meta = self.fitting.get('meta') or {}
                    params_meta = meta.get('params') or {}
                    # params_template 末尾使用键名 'k'
                    params_meta['k'] = float(k_opt)
                    meta['params'] = params_meta
                    self.fitting['meta'] = meta
            except Exception:
                pass
            
            # 步骤6：更新显示
            self._update_GUI_image('fitting')
            self._update_outside_window('fitting')
            
            # 记录优化结果
            improvement = ((residual_before - residual_after) / max(residual_before, 1e-12)) * 100
            
            self._add_fitting_success(
                f"K-value optimized ({optimization_method}): {current_k:.6f} → {k_opt:.6f}"
            )
            self._add_fitting_success(
                f"Residual improvement: {improvement:.2f}% "
                f"({residual_before:.6e} → {residual_after:.6e})"
            )
            
            # 添加数据状态信息
            data_info = f"Data range - I_exp: [{np.min(I_exp_valid):.3e}, {np.max(I_exp_valid):.3e}], "
            data_info += f"I_base: [{np.min(I_base_valid):.3e}, {np.max(I_base_valid):.3e}]"
            self._add_fitting_message(data_info)
            
        except ImportError:
            self._add_fitting_error("scipy.optimize.nnls not available, using analytical solution only")
            # 回退到简单解析解，即使K可能为负
        except Exception as e:
            self._add_fitting_error(f"Error during K-value optimization: {e}")
            # 确保K值被恢复
            if 'current_k' in locals():
                self.set_global_parameter('k_value', current_k)
    
    # ================================
    # 参数缓存管理
    # ================================
    

    

    def _on_parameter_editing_finished(self, widget_id: int, shape: str, param: str):
        """当参数编辑完成时的回调方法 - 保存参数并根据模式决定是否触发拟合更新"""
        try:
            # 如果正在载入参数或初始化，跳过处理
            if self._loading_parameters or self._initializing:
                return
            
            # 获取当前参数值
            param_mapping = self._get_parameter_widget_mapping(widget_id, shape)
            widget_name = param_mapping.get(param)
            
            if widget_name and hasattr(self.ui, widget_name):
                widget = getattr(self.ui, widget_name)
                current_value = widget.value()
                
                # 保存参数到JSON
                particle_id = f"particle_{widget_id}"
                success = self.model_params_manager.set_particle_parameter('fitting', particle_id, shape.lower(), param, current_value)
                
                if success:
                    self.model_params_manager.save_parameters()
                    self._add_particle_message(f"💾 Saved to JSON: {particle_id}.{shape.lower()}.{param} = {current_value}")
                else:
                    self._add_fitting_error(f"Failed to save parameter: {particle_id}.{shape.lower()}.{param} = {current_value}")
            
            # 检查当前是否处于拟合模式
            is_fitting_mode = self._is_in_fitting_mode()
            
            if is_fitting_mode:
                self._add_particle_message(f"🔄 Fitting mode: Auto-updating after {shape}.{param} edit finished")
                # 触发手动拟合更新
                self._perform_manual_fitting()
            else:
                self._add_particle_message(f"📝 Normal mode: Parameter {shape}.{param} edit finished (saved only)")
                
        except Exception as e:
            self._add_fitting_error(f"Failed to handle parameter editing finished: {e}")
    
    def _on_global_parameter_editing_finished(self, param_name: str):
        """当全局参数编辑完成时的回调方法 - 保存参数并根据模式决定是否触发拟合更新"""
        try:
            # 如果正在载入参数或初始化，跳过处理
            if self._loading_parameters or self._initializing:
                return
            
            # 获取当前参数值并保存
            current_value = None
            if param_name == 'sigma_res' and hasattr(self.ui, 'fitSigmaResValue'):
                current_value = self.ui.fitSigmaResValue.value()
            elif param_name == 'nu_res' and hasattr(self.ui, 'fitNuResValue'):
                current_value = self.ui.fitNuResValue.value()
            elif param_name == 'int_res' and hasattr(self.ui, 'fitIntResValue'):
                current_value = self.ui.fitIntResValue.value()
            elif param_name == 'k_value' and hasattr(self.ui, 'fitKValue'):
                current_value = self.ui.fitKValue.value()
            
            if current_value is not None:
                # 保存全局参数到JSON
                success = self.model_params_manager.set_global_parameter('fitting', param_name, current_value)
                
                if success:
                    self.model_params_manager.save_parameters()
                    self._add_particle_message(f"💾 Saved global parameter to JSON: {param_name} = {current_value}")
                else:
                    self._add_fitting_error(f"Failed to save global parameter: {param_name} = {current_value}")
            
            # 检查当前是否处于拟合模式
            is_fitting_mode = self._is_in_fitting_mode()
            
            if is_fitting_mode:
                self._add_particle_message(f"🔄 Fitting mode: Auto-updating after global {param_name} edit finished")
                # 触发手动拟合更新
                self._perform_manual_fitting()
            else:
                self._add_particle_message(f"📝 Normal mode: Global parameter {param_name} edit finished (saved only)")
                
        except Exception as e:
            self._add_fitting_error(f"Failed to handle global parameter editing finished: {e}")
    
    def _is_in_fitting_mode(self) -> bool:
        """检查当前是否处于拟合模式"""
        return hasattr(self, '_fitting_mode_active') and self._fitting_mode_active
    

    

    
    # ================================
    # 数据导出功能
    # ================================
    
    def _export_fitting_data(self):
        """导出Fitting界面图形数据"""
        try:
            import numpy as np
            
            # 检查fitGraphicsView是否为空
            if not hasattr(self.ui, 'fitGraphicsView') or self.ui.fitGraphicsView is None:
                self._add_fitting_error("fitGraphicsView is not available")
                return
            
            # 构造可导出数据源选项
            options = []
            if getattr(self, 'cut', None) is not None:
                options.append('Cut Data')
            if getattr(self, 'fitting', None) is not None:
                options.append('Fitting Data')
            if getattr(self, 'current_1d_data', None) is not None:
                options.append('1D File Data')
            if not options:
                self._add_fitting_error("No available data to export (need Cut, Fitting, or 1D data)")
                return

            # 默认选项：优先Cut，其次Fitting，再次1D
            default_index = 0
            # 让用户选择要导出的数据类型
            choice, ok = QInputDialog.getItem(
                None,
                "Select Data to Export",
                "Data source:",
                options,
                default_index,
                False
            )
            if not ok:
                return

            # 根据选择填充x/y
            x_data = None
            y_data = None
            data_name = ""
            if choice == 'Cut Data' and self.cut is not None:
                x_data = np.array(self.cut.get('q', []))
                y_data = np.array(self.cut.get('I', []))
                data_name = 'Cut_Data'
            elif choice == 'Fitting Data' and self.fitting is not None:
                x_data = np.array(self.fitting.get('q', []))
                y_data = np.array(self.fitting.get('I', []))
                data_name = 'Fitting_Data'
            elif choice == '1D File Data' and self.current_1d_data is not None:
                x_data = np.array(self.current_1d_data.get('q', []))
                y_data = np.array(self.current_1d_data.get('I', []))
                data_name = '1D_File_Data'
            else:
                self._add_fitting_error("Selected data is not available to export")
                return
            
            # 弹出文件保存对话框
            filename, _ = QFileDialog.getSaveFileName(
                None,
                f"Export {data_name}",
                f"{data_name}.txt",
                "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
            )
            
            if not filename:
                return  # 用户取消了保存
            
            # 确保数据长度一致
            min_length = min(len(x_data), len(y_data))
            x_data = x_data[:min_length]
            y_data = y_data[:min_length]
            
            # 按列排列数据并保存
            combined_data = np.column_stack([x_data, y_data])
            
            # 根据文件扩展名选择保存格式
            if filename.lower().endswith('.csv'):
                np.savetxt(filename, combined_data, delimiter=',', 
                          header='X,Y', comments='', fmt='%.6e')
            else:
                np.savetxt(filename, combined_data, delimiter='\t', 
                          header='X\tY', comments='', fmt='%.6e')
            
            self._add_fitting_success(f"{data_name} exported successfully to: {filename}")
            
        except Exception as e:
            self._add_fitting_error(f"Export failed: {str(e)}")

    
    def _perform_manual_fitting(self):
        """执行手动拟合计算"""
        try:
            from utils.fitting import make_mixed_model, params_template, mixed_model_components
            
            # 1. 判断粒子形状ComboBox的状态
            active_shapes, shape_configs = self._collect_active_particles()
            
            if not active_shapes:
                self._add_fitting_error("No active particle shapes selected for fitting")
                return
            
            self._add_fitting_success(f"Active shapes: {active_shapes}")
            self._last_active_particle_ids = shape_configs.copy()
            
            # 2. 获取q值数组
            q_data = None
            if hasattr(self.ui, 'fitCurrentDataCheckBox') and self.ui.fitCurrentDataCheckBox.isChecked():
                # 使用Cut数据的q值
                if hasattr(self, 'current_cut_data') and self.current_cut_data is not None:
                    q_data = np.array(self.current_cut_data['x_coords'])
                else:
                    self._add_fitting_error("No Cut data available for fitting")
                    return
            else:
                # 使用1D文件数据的q值
                if hasattr(self, 'current_1d_data') and self.current_1d_data is not None:
                    q_data = np.array(self.current_1d_data['q'])
                else:
                    self._add_fitting_error("No 1D file data available for fitting")
                    return
            
            # 3. 创建混合模型
            model_func = make_mixed_model(active_shapes)
            param_names = params_template(active_shapes)
            
            self._add_fitting_success(f"Created model with parameters: {param_names}")
            
            # 4. 获取当前参数值
            params = []
            for i, shape in enumerate(active_shapes, 1):
                shape_idx = shape_configs[i-1]  # 对应的shape配置索引
                
                if shape == 'sphere':
                    # Sphere参数：Int, R, sigma_R, D, sigma_D, BG
                    int_param = self._get_particle_parameter(shape_idx, 'Int', 1.0)
                    r_param = self._get_particle_parameter(shape_idx, 'R', 10.0)
                    sigma_r_param = self._get_particle_parameter(shape_idx, 'sigma_R', 0.1)
                    d_param = self._get_particle_parameter(shape_idx, 'D', 100.0)
                    sigma_d_param = self._get_particle_parameter(shape_idx, 'sigma_D', 0.1)
                    bg_param = self._get_particle_parameter(shape_idx, 'BG', 0.0)
                    
                    # 检查结构因子条件
                    if d_param == 0 or sigma_d_param == 0:
                        self._add_fitting_success(f"Shape {i} ({shape}): Structure factor disabled (D={d_param}, sigma_D={sigma_d_param})")
                    else:
                        self._add_fitting_success(f"Shape {i} ({shape}): Structure factor enabled (D={d_param}, sigma_D={sigma_d_param})")
                    
                    self._add_fitting_success(f"Shape {i} ({shape}): BG={bg_param}")
                    params.extend([int_param, r_param, sigma_r_param, d_param, sigma_d_param, bg_param])
                    
                elif shape == 'cylinder':
                    # Cylinder参数：Int, R, sigma_R, h, sigma_h, D, sigma_D, BG
                    int_param = self._get_particle_parameter(shape_idx, 'Int', 1.0)
                    r_param = self._get_particle_parameter(shape_idx, 'R', 10.0)
                    sigma_r_param = self._get_particle_parameter(shape_idx, 'sigma_R', 0.1)
                    h_param = self._get_particle_parameter(shape_idx, 'h', 20.0)
                    sigma_h_param = self._get_particle_parameter(shape_idx, 'sigma_h', 0.1)
                    d_param = self._get_particle_parameter(shape_idx, 'D', 100.0)
                    sigma_d_param = self._get_particle_parameter(shape_idx, 'sigma_D', 0.1)
                    bg_param = self._get_particle_parameter(shape_idx, 'BG', 0.0)
                    
                    # 检查结构因子条件
                    if d_param == 0 or sigma_d_param == 0:
                        self._add_fitting_success(f"Shape {i} ({shape}): Structure factor disabled (D={d_param}, sigma_D={sigma_d_param})")
                    else:
                        self._add_fitting_success(f"Shape {i} ({shape}): Structure factor enabled (D={d_param}, sigma_D={sigma_d_param})")
                    
                    self._add_fitting_success(f"Shape {i} ({shape}): BG={bg_param}")
                    params.extend([int_param, r_param, sigma_r_param, h_param, sigma_h_param, d_param, sigma_d_param, bg_param])
            
            # 5. 添加全局分辨率与缩放参数：sigma_Res、nu_Res、int_Res、k
            # 优先从UI控件获取最新值（BG参数为粒子级别，已在上面添加）

            # 获取 sigma_Res (Br)
            if hasattr(self.ui, 'fitSigmaResValue'):
                sigma_res_param = self.ui.fitSigmaResValue.value()
            else:
                sigma_res_param = self.get_global_parameter('sigma_res') if hasattr(self, 'get_global_parameter') else 0.1

            # 获取 nu_Res (Lorentzian阶数)
            if hasattr(self.ui, 'fitNuResValue'):
                nu_res_param = self.ui.fitNuResValue.value()
            else:
                nu_res_param = self.get_global_parameter('nu_res') if hasattr(self, 'get_global_parameter') else 5.0

            # 获取 int_Res (Lorentzian幅度)
            if hasattr(self.ui, 'fitIntResValue'):
                int_res_param = self.ui.fitIntResValue.value()
            else:
                int_res_param = self.get_global_parameter('int_res') if hasattr(self, 'get_global_parameter') else 0.0

            # 获取 k (全局缩放)
            if hasattr(self.ui, 'fitKValue'):
                k_param = self.ui.fitKValue.value()
            else:
                k_param = self.get_global_parameter('k_value') if hasattr(self, 'get_global_parameter') else 1.0

            # 输出诊断信息
            if sigma_res_param == 0 or int_res_param == 0:
                self._add_fitting_success(
                    f"Lorentzian resolution component disabled (sigma_res={sigma_res_param}, int_res={int_res_param})"
                )
            else:
                self._add_fitting_success(
                    f"Lorentzian resolution active: sigma_res={sigma_res_param}, nu_res={nu_res_param}, int_res={int_res_param}"
                )

            params.extend([sigma_res_param, nu_res_param, int_res_param, k_param])
            
            # 显示详细的参数信息
            param_dict = dict(zip(param_names, params))
            self._add_fitting_success(f"Using parameters: {param_dict}")
            
            # 验证参数获取
            self._validate_parameter_retrieval(active_shapes, shape_configs)
            
            # 6. 执行拟合计算
            try:
                # 计算拟合结果
                fitting_result = model_func(q_data, *params)
                self._add_fitting_success(f"Fitting calculation completed successfully")
                
                # 显示拟合结果的基本统计信息
                result_stats = {
                    'min': float(np.min(fitting_result)),
                    'max': float(np.max(fitting_result)),
                    'mean': float(np.mean(fitting_result)),
                    'sum': float(np.sum(fitting_result))
                }
                self._add_fitting_success(f"Result stats: {result_stats}")
                
                # 存储拟合结果到 I_fitting
                self.I_fitting = fitting_result
                self.has_fitting_data = True
                # 保持与另外一套状态标志同步
                try:
                    self._has_fitting_data = True
                except Exception:
                    pass
                # 全局缓存：fitting 曲线（包含参数与形状信息）
                try:
                    import time
                    self.fitting = {
                        'q': np.array(q_data, copy=True),
                        'I': np.array(fitting_result, copy=True),
                        'meta': {
                            'shapes': active_shapes,
                            'params': param_dict,
                            'timestamp': time.time(),
                            'source': 'fitting'
                        }
                    }
                except Exception:
                    self.fitting = {'q': q_data, 'I': fitting_result, 'meta': {'source': 'fitting'}}
                
                # 切换显示模式为 Fitting with data
                self.display_mode = 'fitting'
                self._fitting_mode_active = True  # 设置拟合模式标志
                
                # 更新显示
                self._update_GUI_image('fitting')
                self._update_outside_window('fitting')
                
                # 如果启用了auto-K优化且有实验数据，执行K值优化
                if self._auto_k_enabled and hasattr(self, 'I') and self.I is not None:
                    try:
                        self._optimize_k_value()
                    except Exception as opt_error:
                        self._add_fitting_error(f"Auto K-value optimization failed: {opt_error}")
                
            except Exception as calc_error:
                self._add_fitting_error(f"Fitting calculation failed: {str(calc_error)}")
                
        except Exception as e:
            self._add_fitting_error(f"Manual fitting failed: {str(e)}")
    
    def _store_fitting_data(self, q_data, intensity_data, active_shapes):
        """存储拟合数据，不改变显示模式"""
        try:
            self._fitting_q_data = np.array(q_data)
            self._fitting_intensity_data = np.array(intensity_data)
            self._fitting_shapes = active_shapes.copy() if active_shapes else []
            self._has_fitting_data = True
            
            # 更新GUI显示（内嵌图表）
            self._update_gui_fitting_display()
            
        except Exception as e:
            pass
    
    def _switch_to_fitting_display_mode(self):
        """切换到拟合显示模式（仅在外部窗口成功打开时调用）"""
        try:
            # 同步更新显示模式（内部+对外）
            self._display_mode = 'fitting'
            self.display_mode = 'fitting'
            self._fitting_mode_active = True
            
            # 更新GUI和外部窗口的显示
            self._refresh_all_displays_for_fitting_mode()
            
        except Exception as e:
            pass
    
    def _switch_to_normal_display_mode(self):
        """切换到普通显示模式（Normal mode）"""
        try:
            # 同步更新显示模式（内部+对外）
            self._display_mode = 'normal'
            self.display_mode = 'normal'
            self._fitting_mode_active = False
            
            # 清除拟合相关数据（如果有的话）
            self._fitting_q_data = None
            self._fitting_intensity_data = None
            self._fitting_shapes = []
            self._has_fitting_data = False
            # 统一清理通用拟合状态，避免残留导致绘图逻辑误判
            try:
                self.has_fitting_data = False
                self.I_fitting = None
            except Exception:
                pass
            
        except Exception as e:
            pass
    
    def _update_gui_fitting_display(self):
        """更新GUI中的拟合显示（fitGraphicsView），保持当前显示模式"""
        try:
            if not hasattr(self, '_fitting_q_data') or self._fitting_q_data is None:
                return
            # 一旦更新拟合图像，则视为进入拟合模式（满足“更新了fitting图像就是fitting模式”的期望）
            try:
                self.display_mode = 'fitting'
                self._display_mode = 'fitting'
                self._fitting_mode_active = True
            except Exception:
                pass
                
            # 在GUI中显示拟合结果，但不改变显示模式
            self._plot_fitting_result(self._fitting_q_data, self._fitting_intensity_data, self._fitting_shapes)
            
        except Exception as e:
            pass
    
    def _refresh_all_displays_for_fitting_mode(self):
        """在拟合模式下刷新所有显示（GUI + 外部窗口）"""
        try:
            if not self._has_fitting_data:
                return
                
            # 1. 刷新GUI显示
            self._update_gui_fitting_display()
            
            # 2. 刷新外部窗口显示（如果存在且可见）
            if (hasattr(self, 'independent_fit_window') and 
                self.independent_fit_window is not None and 
                self.independent_fit_window.isVisible()):
                
                self._refresh_external_window_fitting_display()
                
        except Exception as e:
            pass
    
    def _refresh_external_window_fitting_display(self):
        """刷新外部窗口的拟合显示"""
        try:
            if not self._has_fitting_data:
                return
                
            # 获取当前显示选项
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()
            
            # 获取原始数据
            original_x_data = None
            original_y_data = None
            data_label = ""
            
            if hasattr(self.ui, 'fitCurrentDataCheckBox') and self.ui.fitCurrentDataCheckBox.isChecked():
                if hasattr(self, 'current_cut_data') and self.current_cut_data is not None:
                    original_x_data = np.array(self.current_cut_data['x_coords'])
                    original_y_data = np.array(self.current_cut_data['y_intensity'])
                    data_label = "Cut Data"
            else:
                if hasattr(self, 'current_1d_data') and self.current_1d_data is not None:
                    original_x_data = np.array(self.current_1d_data['q'])
                    original_y_data = np.array(self.current_1d_data['I'])
                    data_label = "1D File Data"
            
            # 更新外部窗口显示
            x_label = "q (Å$^{-1}$)"
            y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
            title = f'Manual Fitting Result - {", ".join(self._fitting_shapes)}'
            
            self._update_independent_window_with_fitting(
                original_x_data, original_y_data, data_label,
                self._fitting_q_data, self._fitting_intensity_data, self._fitting_shapes,
                x_label, y_label, title,
                log_x, log_y, normalize
            )
            
        except Exception as e:
            pass
    
    def _get_particle_parameter(self, shape_idx, param_name, default_value):
        """获取指定粒子形状的参数值"""
        try:
            # 获取当前粒子的形状
            current_shape = self.get_particle_shape(shape_idx)
            if current_shape == 'None':
                return default_value
            
            # 尝试从UI控件获取（优先从UI获取最新值）
            control_name = self._get_ui_control_name(shape_idx, current_shape, param_name)
            if control_name and hasattr(self.ui, control_name):
                control = getattr(self.ui, control_name)
                if hasattr(control, 'value'):
                    value = control.value()
                    return value
                elif hasattr(control, 'text'):
                    try:
                        value = float(control.text())
                        return value
                    except ValueError:
                        pass
            
            # 如果UI控件获取失败，尝试从model_params_manager获取
            if hasattr(self, 'model_params_manager'):
                particle_data = self.model_params_manager.get_particle('fitting', shape_idx)
                if particle_data and param_name in particle_data:
                    value = particle_data[param_name]
                    return value
            
            # 返回默认值
            return default_value
            
        except Exception as e:
            return default_value
    
    def _get_ui_control_name(self, shape_idx, shape_name, param_name):
        """根据形状和参数名获取UI控件名称"""
        try:
            # 参数名映射表
            param_mapping = {
                'Int': 'Int',
                'R': 'R', 
                'sigma_R': 'SigmaR',
                'D': 'D',
                'sigma_D': 'SigmaD',
                'h': 'h',
                'sigma_h': 'Sigmah',
                'BG': 'BG'
            }
            
            ui_param = param_mapping.get(param_name)
            if ui_param:
                return f'fitParticle{shape_name}{ui_param}Value_{shape_idx}'
            else:
                return None
                
        except Exception as e:
            return None
    
    def _plot_fitting_result(self, q_data, intensity_data, active_shapes):
        """在fitGraphicsView中绘制拟合结果"""
        try:
            if not hasattr(self.ui, 'fitGraphicsView'):
                return
            
            if not is_matplotlib_available():
                self._add_fitting_error("Matplotlib not available for plotting")
                return
            
            # 获取显示选项
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()
            
            # 获取原始数据点用于scatter显示
            original_x_data = None
            original_y_data = None
            data_label = ""
            
            if hasattr(self.ui, 'fitCurrentDataCheckBox') and self.ui.fitCurrentDataCheckBox.isChecked():
                # 使用Cut数据
                if hasattr(self, 'current_cut_data') and self.current_cut_data is not None:
                    original_x_data = np.array(self.current_cut_data['x_coords'])
                    original_y_data = np.array(self.current_cut_data['y_intensity'])
                    data_label = "Cut Data"
            else:
                # 使用1D文件数据
                if hasattr(self, 'current_1d_data') and self.current_1d_data is not None:
                    original_x_data = np.array(self.current_1d_data['q'])
                    original_y_data = np.array(self.current_1d_data['I'])
                    data_label = "1D File Data"
            
            # 创建图形
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            
            # 使用统一的场景管理方法
            scene = self._setup_fit_graphics_scene()
            if scene is None:
                return
            
            # 创建matplotlib图形
            fig = Figure(figsize=(9.6, 7.2), dpi=80)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            
            # 应用标准化处理
            fitting_y_data = np.array(intensity_data)  # 拟合数据不进行归一化
            plot_original_y = original_y_data.copy() if original_y_data is not None else None
            norm_divisor = None
            
            if normalize and original_y_data is not None:
                # 只对原始数据进行标准化，拟合数据保持原始
                max_original = np.max(original_y_data)
                if max_original > 0:
                    norm_divisor = max_original
                    plot_original_y = original_y_data / max_original
                    # 拟合数据按照同样的比例缩放以保持相对关系
                    fitting_y_data = fitting_y_data / max_original
            
            # 绘制原始数据点（scatter）
            if original_x_data is not None and plot_original_y is not None:
                ax.scatter(original_x_data, plot_original_y, 
                          s=20, alpha=0.7, color='blue', 
                          label=data_label, zorder=2)
            
            # 绘制拟合线
            ax.plot(q_data, fitting_y_data, 
                   color='red', linewidth=2, 
                   label=f'Fitting ({", ".join(active_shapes)})', 
                   zorder=3)
            
            # 设置标签和标题
            x_label = "q (Å$^{-1}$)" if "q" in str(original_x_data).lower() or len(q_data) > 0 else "Position"
            y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
            title = f'Manual Fitting Result - {", ".join(active_shapes)}'
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 设置坐标轴样式
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.8)
            ax.tick_params(axis='both', which='both', width=1.6, labelsize=12)
            
            # 应用对数坐标设置
            if log_x:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')

            # ROI 辅助线
            self._draw_roi_guides_if_active(ax)
            
            # 调整布局
            fig.tight_layout()
            
            # 添加到场景
            proxy_widget = scene.addWidget(canvas)
            self._fit_view_to_item(self.ui.fitGraphicsView, proxy_widget, keep_aspect=True)
            
            # 保存当前的 figure 和 canvas 以便后续操作（如清除拟合线）
            self._current_fit_figure = fig
            self._current_fit_canvas = canvas
            
            # 存储拟合结果数据
            if not hasattr(self, 'current_fitting_data'):
                self.current_fitting_data = {}
            
            self.current_fitting_data = {
                'q': q_data.copy(),
                'I_fitted': intensity_data.copy(),
                'shapes': active_shapes.copy(),
                'title': title,
                'original_x': original_x_data.copy() if original_x_data is not None else None,
                'original_y': original_y_data.copy() if original_y_data is not None else None,
                'data_label': data_label
            }
            
            self._add_fitting_success(f"Fitting result plotted for shapes: {active_shapes}")
            
        except Exception as e:
            self._add_fitting_error(f"Failed to plot fitting result: {str(e)}")

    def _get_last_fitting_spec_and_params(self, fallback_shapes=None):
        """返回 (shapes:list[str], params_in_order:list[float])。
        优先读取 self.fitting['meta'] 中的 shapes 与 params（与当前显示拟合结果一致）。
        若缺失则根据当前 UI 重新抓取参数作为后备。
        """
        try:
            shapes = None; param_dict = None
            if isinstance(getattr(self, 'fitting', None), dict):
                meta = self.fitting.get('meta', {})
                shapes = meta.get('shapes')
                param_dict = meta.get('params')
            if shapes and param_dict:
                from utils.fitting import params_template
                tmpl = params_template(shapes)
                params_list = []
                ok = True
                for name in tmpl:
                    if name in param_dict:
                        params_list.append(float(param_dict[name]))
                    else:
                        ok = False; break
                if ok:
                    return shapes, params_list
            # 回退到根据 UI 取参数
            act_shapes, act_idx = self._collect_active_particles()
            if not act_shapes:
                return (fallback_shapes, None) if fallback_shapes else (None, None)
            self._last_active_particle_ids = act_idx.copy()
            # 逐形状按模板顺序取值
            params_list = []
            for j, s in enumerate(act_shapes, 1):
                if s == 'sphere':
                    params_list.extend([
                        self._get_particle_parameter(act_idx[j-1], 'Int', 1.0),
                        self._get_particle_parameter(act_idx[j-1], 'R', 10.0),
                        self._get_particle_parameter(act_idx[j-1], 'sigma_R', 0.1),
                        self._get_particle_parameter(act_idx[j-1], 'D', 100.0),
                        self._get_particle_parameter(act_idx[j-1], 'sigma_D', 0.1),
                        self._get_particle_parameter(act_idx[j-1], 'BG', 0.0),
                    ])
                elif s == 'cylinder':
                    params_list.extend([
                        self._get_particle_parameter(act_idx[j-1], 'Int', 1.0),
                        self._get_particle_parameter(act_idx[j-1], 'R', 10.0),
                        self._get_particle_parameter(act_idx[j-1], 'sigma_R', 0.1),
                        self._get_particle_parameter(act_idx[j-1], 'h', 20.0),
                        self._get_particle_parameter(act_idx[j-1], 'sigma_h', 0.1),
                        self._get_particle_parameter(act_idx[j-1], 'D', 100.0),
                        self._get_particle_parameter(act_idx[j-1], 'sigma_D', 0.1),
                        self._get_particle_parameter(act_idx[j-1], 'BG', 0.0),
                    ])
            # 追加 sigma_Res, k
            if hasattr(self.ui, 'fitSigmaResValue'):
                sigma_res_val = float(self.ui.fitSigmaResValue.value())
            else:
                sigma_res_val = float(self.get_global_parameter('sigma_res')) if hasattr(self, 'get_global_parameter') else 0.0
            if hasattr(self.ui, 'fitKValue'):
                k_val = float(self.ui.fitKValue.value())
            else:
                k_val = float(self.get_global_parameter('k_value')) if hasattr(self, 'get_global_parameter') else 1.0
            params_list.extend([sigma_res_val, k_val])
            return act_shapes, [float(x) for x in params_list]
        except Exception:
            return (fallback_shapes, None) if fallback_shapes else (None, None)

    def _on_component_checkbox_changed(self, *_):
        """组件叠加显示选项变更时，根据模式决定是否刷新。

        需求：Normal 模式下不触发任何重绘；仅在 Fitting 模式下刷新叠加组件显示。
        """
        try:
            # 仅在拟合模式下响应这些复选框变化
            if not self._is_in_fitting_mode():
                return
            # 在拟合模式下刷新显示；组件曲线会从最近拟合或UI参数获取
            self._update_GUI_image('fitting')
            self._update_outside_window('fitting')
        except Exception:
            pass
    
    def _show_fitting_in_external_window(self, q_data, intensity_data, active_shapes):
        """在外置窗口中显示拟合结果，返回是否成功打开"""
        try:
            # 获取显示选项
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()
            
            # 获取原始数据点
            original_x_data = None
            original_y_data = None
            data_label = ""
            
            if hasattr(self.ui, 'fitCurrentDataCheckBox') and self.ui.fitCurrentDataCheckBox.isChecked():
                # 使用Cut数据
                if hasattr(self, 'current_cut_data') and self.current_cut_data is not None:
                    original_x_data = np.array(self.current_cut_data['x_coords'])
                    original_y_data = np.array(self.current_cut_data['y_intensity'])
                    data_label = "Cut Data"
            else:
                # 使用1D文件数据
                if hasattr(self, 'current_1d_data') and self.current_1d_data is not None:
                    original_x_data = np.array(self.current_1d_data['q'])
                    original_y_data = np.array(self.current_1d_data['I'])
                    data_label = "1D File Data"
            
            # 创建或显示独立拟合窗口
            if self.independent_fit_window is None or not self.independent_fit_window.isVisible():
                self.independent_fit_window = IndependentFitWindow(self.main_window)
                
                # 连接信号
                self.independent_fit_window.status_updated.connect(self.status_updated.emit)
            
            # 准备数据标题和标签
            x_label = "q (Å$^{-1}$)" if "q" in str(original_x_data).lower() or len(q_data) > 0 else "Position"
            y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
            title = f'Manual Fitting Result - {", ".join(active_shapes)}'
            
            # 更新独立窗口的显示 - 使用自定义的拟合显示方法
            self._update_independent_window_with_fitting(
                original_x_data, original_y_data, data_label,
                q_data, intensity_data, active_shapes,
                x_label, y_label, title,
                log_x, log_y, normalize
            )
            
            # 显示窗口
            self.independent_fit_window.show()
            self.independent_fit_window.raise_()
            self.independent_fit_window.activateWindow()
            
            # 设置焦点
            if hasattr(self.independent_fit_window, 'canvas'):
                self.independent_fit_window.canvas.setFocus()
            
            self._add_fitting_success(f"Fitting result displayed in external window")
            return True  # 成功打开外部窗口
            
        except Exception as e:
            self._add_fitting_error(f"Failed to show fitting in external window: {str(e)}")
            return False  # 打开外部窗口失败
    
    def _update_independent_window_with_fitting(self, original_x, original_y, data_label,
                                               fitting_x, fitting_y, shapes,
                                               x_label, y_label, title,
                                               log_x, log_y, normalize):
        """更新独立窗口以显示拟合结果（原始数据点+拟合线）"""
        try:
            if not hasattr(self.independent_fit_window, 'ax'):
                return
            
            ax = self.independent_fit_window.ax
            ax.clear()
            
            # 应用标准化处理
            plot_fitting_y = np.array(fitting_y)  # 拟合数据不进行归一化
            plot_original_y = original_y.copy() if original_y is not None else None
            
            if normalize and original_y is not None:
                # 只对原始数据进行标准化，拟合数据按同样比例缩放
                max_original = np.max(original_y)
                if max_original > 0:
                    plot_original_y = original_y / max_original
                    # 拟合数据按照同样的比例缩放以保持相对关系
                    plot_fitting_y = fitting_y / max_original
            
            # 绘制原始数据点（scatter）
            if original_x is not None and plot_original_y is not None:
                ax.scatter(original_x, plot_original_y, 
                          s=30, alpha=0.7, color='blue', 
                          label=data_label, zorder=2)
            
            # 绘制拟合线
            ax.plot(fitting_x, plot_fitting_y, 
                   color='red', linewidth=2.5, 
                   label=f'Fitting ({", ".join(shapes)})', 
                   zorder=3)
            
            # 设置标签和样式
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 设置坐标轴样式
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.8)
            
            # 应用对数坐标设置
            if log_x:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')
            
            # 刷新显示
            if hasattr(self.independent_fit_window, 'canvas'):
                self.independent_fit_window.canvas.draw()
            
        except Exception as e:
            self._add_fitting_error(f"Failed to update independent window with fitting: {str(e)}")
    
    def _validate_parameter_retrieval(self, active_shapes, shape_configs):
        """验证参数获取是否正常工作"""
        try:
            self._add_fitting_success("=== Parameter Retrieval Validation ===")
            
            for i, shape in enumerate(active_shapes, 1):
                shape_idx = shape_configs[i-1]
                current_shape = self.get_particle_shape(shape_idx)
                
                self._add_fitting_success(f"Shape {i}: {shape} (widget {shape_idx}, actual: {current_shape})")
                
                if shape == 'sphere':
                    params_to_check = [
                        ('Int', 'fitParticleSphereIntValue'),
                        ('R', 'fitParticleSphereRValue'),
                        ('sigma_R', 'fitParticleSphereSigmaRValue'),
                        ('D', 'fitParticleSphereDValue'),
                        ('sigma_D', 'fitParticleSphereSigmaDValue'),
                        ('BG', 'fitParticleSphereBGValue')
                    ]
                elif shape == 'cylinder':
                    params_to_check = [
                        ('Int', 'fitParticleCylinderIntValue'),
                        ('R', 'fitParticleCylinderRValue'),
                        ('sigma_R', 'fitParticleCylinderSigmaRValue'),
                        ('h', 'fitParticleCylinderhValue'),
                        ('sigma_h', 'fitParticleCylinderSigmahValue'),
                        ('D', 'fitParticleCylinderDValue'),
                        ('sigma_D', 'fitParticleCylinderSigmaDValue'),
                        ('BG', 'fitParticleCylinderBGValue')
                    ]
                else:
                    continue
                
                for param_name, base_control_name in params_to_check:
                    control_name = f"{base_control_name}_{shape_idx}"
                    
                    if hasattr(self.ui, control_name):
                        control = getattr(self.ui, control_name)
                        if hasattr(control, 'value'):
                            value = control.value()
                            self._add_fitting_success(f"  ✓ {param_name}: {control_name} = {value}")
                        else:
                            self._add_fitting_error(f"  ✗ {param_name}: {control_name} has no 'value' method")
                    else:
                        self._add_fitting_error(f"  ✗ {param_name}: {control_name} not found in UI")
            
            # 检查全局参数
            self._add_fitting_success("Global Parameters:")
            if hasattr(self.ui, 'fitSigmaResValue'):
                sigma_res = self.ui.fitSigmaResValue.value()
                self._add_fitting_success(f"  ✓ sigma_res: fitSigmaResValue = {sigma_res}")
            else:
                self._add_fitting_error(f"  ✗ fitSigmaResValue not found")
                
            if hasattr(self.ui, 'fitKValue'):
                k_value = self.ui.fitKValue.value()
                self._add_fitting_success(f"  ✓ k_value: fitKValue = {k_value}")
            else:
                self._add_fitting_error(f"  ✗ fitKValue not found")
            
            self._add_fitting_success("=== Validation Complete ===")
            
        except Exception as e:
            self._add_fitting_error(f"Parameter validation failed: {str(e)}")
    
    def _clear_fitting_data(self):
        """清空fitting数据"""
        try:
            # 检查是否有I_fitting数据
            if not hasattr(self, 'I_fitting') or self.I_fitting is None:
                self.status_updated.emit("No fitting data to clear")
                return
            
            # 删除I_fitting数据
            self.I_fitting = None
            self.has_fitting_data = False
            
            # 切换到正常模式
            self.display_mode = 'normal'
            self._fitting_mode_active = False  # 重置拟合模式标志
            
            # 更新显示（切换回正常模式）
            self._update_GUI_image('normal')
            self._update_outside_window('normal')
            
            self.status_updated.emit("Fitting data cleared")
            
        except Exception as e:
            self.status_updated.emit(f"Error clearing fitting data: {str(e)}")
    
    def _force_update_gui_points_only(self):
        """强制更新GUI显示，只显示数据点，清除所有拟合线"""
        try:
            if not hasattr(self.ui, 'fitGraphicsView'):
                return
                
            # 检查是否有figure和canvas
            if not hasattr(self, '_current_fit_figure') or self._current_fit_figure is None:
                return
                
            if not hasattr(self, '_current_fit_canvas') or self._current_fit_canvas is None:
                return
            
            # 获取当前数据
            x_data, y_data, data_label = self._get_current_data_for_display()
            if x_data is None or y_data is None:
                return
            
            # 清除现有图形并重新绘制
            self._current_fit_figure.clear()
            ax = self._current_fit_figure.add_subplot(111)
            
            # 获取显示选项
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()
            
            # 处理数据
            plot_y = y_data.copy()
            if normalize:
                max_val = np.max(y_data)
                if max_val > 0:
                    plot_y = y_data / max_val
            
            # 绘制数据点
            ax.scatter(x_data, plot_y, s=30, alpha=0.7, color='blue', 
                      label=data_label, zorder=2)
            
            # 设置标签和样式
            x_label = "q (Å$^{-1}$)"
            y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
            title = f'Data Points Only - {data_label}'
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 设置对数坐标
            if log_x:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')
                
            # 强制刷新画布
            self._current_fit_canvas.draw()
            
        except Exception as e:
            # 静默处理错误，避免在没有拟合数据时产生噪音
            pass
    

    def _update_fitting_plot_points_only(self):
        """只显示数据点，不显示拟合曲线"""
        try:
            if not hasattr(self, 'current_cut_data') or self.current_cut_data is None:
                return
                
            # 清除现有的图形
            if hasattr(self.ui, 'fitGraphicsView') and hasattr(self, '_current_fit_figure') and self._current_fit_figure is not None:
                self._current_fit_figure.clear()
                ax = self._current_fit_figure.add_subplot(111)
                
                # 获取当前的log设置
                log_x = self._get_checkbox_state('fitLogXCheckBox', False)
                log_y = self._get_checkbox_state('fitLogYCheckBox', False)
                
                # 绘制数据点
                cut_data = self.current_cut_data
                # 支持两种字段名格式
                x_data = None
                y_data = None
                if 'x_coords' in cut_data and 'y_intensity' in cut_data:
                    x_data = cut_data['x_coords']
                    y_data = cut_data['y_intensity']
                elif 'x' in cut_data and 'y' in cut_data:
                    x_data = cut_data['x']
                    y_data = cut_data['y']
                
                if x_data is not None and y_data is not None:
                    ax.scatter(x_data, y_data, c='blue', s=20, alpha=0.7, label='Data')
                    
                    # 设置坐标轴
                    if log_x:
                        ax.set_xscale('log')
                    if log_y:
                        ax.set_yscale('log')
                    
                    ax.set_xlabel('Q (Å$^{-1}$)')
                    ax.set_ylabel('Intensity')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # 刷新显示
                if hasattr(self, '_current_fit_canvas') and self._current_fit_canvas is not None:
                    self._current_fit_canvas.draw()
                
        except Exception as e:
            pass
    
    def _on_fit_log_changed(self):
        """处理Log-x/Log-y复选框变化"""
        try:
            mode = self.display_mode if hasattr(self, 'display_mode') else 'normal'
            self._update_GUI_image(mode)
            self._update_outside_window(mode)
            self.status_updated.emit("Display log scale updated")
            # 同时调整ROI的范围（log-x下将最小值设置为当前显示坐标轴的下限）
            try:
                QTimer.singleShot(0, self._adjust_roi_bounds_for_log_x)
            except Exception:
                self._adjust_roi_bounds_for_log_x()
        except Exception as e:
            self.status_updated.emit(f"Error updating log scale: {str(e)}")

    def _on_normalize_changed(self):
        """处理Normalize复选框变化"""
        try:
            mode = self.display_mode if hasattr(self, 'display_mode') else 'normal'
            self._update_GUI_image(mode)
            self._update_outside_window(mode)
            self.status_updated.emit("Normalize setting updated")
        except Exception as e:
            self.status_updated.emit(f"Error updating normalize setting: {str(e)}")

    def _on_positive_only_changed(self):
        """处理Positive Only复选框变化"""
        try:
            mode = self.display_mode if hasattr(self, 'display_mode') else 'normal'
            self._update_outside_window(mode)  # 只更新外部窗口，主窗口不受影响
            self.status_updated.emit("Positive Only setting updated")
        except Exception as e:
            self.status_updated.emit(f"Error updating Positive Only setting: {str(e)}")


    

    
    def _update_fitting_plot(self):
        """更新拟合图显示（包含拟合曲线）"""
        try:
            if not hasattr(self, 'fitting_data') or self.fitting_data is None:
                return

            if hasattr(self.ui, 'fitGraphicsView') and hasattr(self, '_current_fit_figure') and self._current_fit_figure is not None:
                self._current_fit_figure.clear()
                ax = self._current_fit_figure.add_subplot(111)

                # 获取当前的log设置
                log_x = self._get_checkbox_state('fitLogXCheckBox', False)
                log_y = self._get_checkbox_state('fitLogYCheckBox', False)

                # 绘制数据点（兼容两种字段名）
                if hasattr(self, 'current_cut_data') and self.current_cut_data is not None:
                    cut_data = self.current_cut_data
                    if 'x_coords' in cut_data and 'y_intensity' in cut_data:
                        ax.scatter(cut_data['x_coords'], cut_data['y_intensity'], c='blue', s=20, alpha=0.7, label='Data')
                    elif 'x' in cut_data and 'y' in cut_data:
                        ax.scatter(cut_data['x'], cut_data['y'], c='blue', s=20, alpha=0.7, label='Data')

                # 绘制拟合曲线（从 self.fitting_data）
                fitting_data = self.fitting_data
                if isinstance(fitting_data, dict) and 'x' in fitting_data and 'y' in fitting_data:
                    ax.plot(fitting_data['x'], fitting_data['y'], 'r-', linewidth=2, label='Fit')

                # 设置坐标轴
                if log_x:
                    ax.set_xscale('log')
                if log_y:
                    ax.set_yscale('log')

                ax.set_xlabel('Q (Å$^{-1}$)')
                ax.set_ylabel('Intensity')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # 刷新显示
                if hasattr(self, '_current_fit_canvas') and self._current_fit_canvas is not None:
                    self._current_fit_canvas.draw()

        except Exception:
            pass
    
    def _update_fitting_mode_displays_without_line(self):
        """在拟合模式下更新显示，只显示数据点，无拟合线"""
        try:
            # 1. 更新GUI显示 - 只显示数据点
            self._update_gui_points_only()
            
            # 2. 更新外部窗口显示 - 只显示数据点
            if (hasattr(self, 'independent_fit_window') and 
                self.independent_fit_window is not None and 
                self.independent_fit_window.isVisible()):
                
                self._update_external_window_points_only()
                
        except Exception as e:
            pass
    
    def _update_gui_points_only(self):
        """更新GUI显示，只显示数据点"""
        try:
            if not hasattr(self.ui, 'fitGraphicsView'):
                return
                
            # 获取当前数据
            x_data, y_data, data_label = self._get_current_data_for_display()
            if x_data is None or y_data is None:
                return
                
            # 获取显示选项
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()
            
            # 在GUI中显示数据点
            self._plot_data_points_only(x_data, y_data, data_label, log_x, log_y, normalize)
            
        except Exception as e:
            pass
    
    def _update_external_window_points_only(self):
        """更新外部窗口显示，只显示数据点"""
        try:
            if not hasattr(self.independent_fit_window, 'ax'):
                return
                
            # 获取当前数据
            x_data, y_data, data_label = self._get_current_data_for_display()
            if x_data is None or y_data is None:
                return
                
            # 获取显示选项
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()
            
            ax = self.independent_fit_window.ax
            ax.clear()
            
            # 处理数据
            plot_y = y_data.copy()
            if normalize:
                max_val = np.max(y_data)
                if max_val > 0:
                    plot_y = y_data / max_val
            
            # 只绘制数据点
            ax.scatter(x_data, plot_y, s=30, alpha=0.7, color='blue', 
                      label=data_label, zorder=2)
            
            # 设置标签和样式（拟合模式风格）
            x_label = "q (Å$^{-1}$)"
            y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
            title = f'Fitting Display Mode - {data_label}'
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 设置坐标轴样式（与拟合模式一致）
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.8)
            
            # 设置对数坐标
            if log_x:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')
                
            # 刷新画布
            if hasattr(self.independent_fit_window, 'canvas'):
                self.independent_fit_window.canvas.draw()
                
        except Exception as e:
            pass
    
    def _get_current_data_for_display(self):
        """获取当前要显示的数据"""
        try:
            if hasattr(self.ui, 'fitCurrentDataCheckBox') and self.ui.fitCurrentDataCheckBox.isChecked():
                # 使用Cut数据
                if hasattr(self, 'current_cut_data') and self.current_cut_data is not None:
                    return (np.array(self.current_cut_data['x_coords']),
                           np.array(self.current_cut_data['y_intensity']),
                           "Cut Data")
            else:
                # 使用1D文件数据
                if hasattr(self, 'current_1d_data') and self.current_1d_data is not None:
                    return (np.array(self.current_1d_data['q']),
                           np.array(self.current_1d_data['I']),
                           "1D File Data")
            
            return None, None, ""
            
        except Exception as e:
            return None, None, ""
    
    def _plot_data_points_only(self, x_data, y_data, data_label, log_x, log_y, normalize):
        """在GUI中绘制仅数据点的图形"""
        try:
            if not hasattr(self.ui, 'fitGraphicsView'):
                return
                
            # Use the existing fitting GUI figure and canvas
            if hasattr(self, '_current_fit_figure') and self._current_fit_figure is not None:
                self._current_fit_figure.clear()
                ax = self._current_fit_figure.add_subplot(111)
                
                # Processing data
                plot_y = y_data.copy()
                if normalize:
                    max_val = np.max(y_data)
                    if max_val > 0:
                        plot_y = y_data / max_val
                
                # Plotting data points
                ax.scatter(x_data, plot_y, s=30, alpha=0.7, color='blue', 
                          label=data_label, zorder=2)
                
                # Setting up labels and styles
                x_label = "q (Å$^{-1}$)"
                y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
                title = f'Fitting Display Mode - {data_label}'
                
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Setting logarithmic coordinates
                if log_x:
                    ax.set_xscale('log')
                if log_y:
                    ax.set_yscale('log')
                    
                # Refresh Canvas
                if hasattr(self, '_current_fit_canvas') and self._current_fit_canvas is not None:
                    self._current_fit_canvas.draw()
                    
        except Exception as e:
            pass
