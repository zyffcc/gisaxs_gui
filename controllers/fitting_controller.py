"""
Cut Fitting 控制器 - 处理GISAXS数据的裁剪和拟合功能
"""

import os
import json
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QThread, QTimer
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGraphicsScene, QVBoxLayout, QWidget, QMainWindow

# 导入探测器参数对话框
from ui.detector_parameters_dialog import DetectorParametersDialog

# 尝试导入所需的库
try:
    import fabio
    FABIO_AVAILABLE = True
except ImportError:
    FABIO_AVAILABLE = False

try:
    import matplotlib
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    matplotlib.use('Qt5Agg')  # 确保使用Qt5后端
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import Q-space calculator
from utils.q_space_calculator import create_detector_from_image_and_params, get_q_axis_labels_and_extents

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    from scipy import ndimage
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
        self.setWindowTitle("GISAXS Image Viewer - Independent Window (框选模式: 右键激活)")
        self.setGeometry(100, 100, 900, 700)
        
        # 创建中央widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout(central_widget)
        
        # 创建matplotlib图形
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # 添加到布局
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # 创建axes
        self.ax = self.figure.add_subplot(111)
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
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        central_widget.setFocusPolicy(Qt.StrongFocus)
        
        # 连接视图变化事件
        self.ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
        self.ax.callbacks.connect('ylim_changed', self._on_ylim_changed)
        
        # 连接鼠标事件用于框选
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # 确保窗口可以接收键盘事件
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()
    
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
                    
                    print(f"Using Q-space display:")
                    print(f"  Data shape: {processed_data.shape}")
                    print(f"  Q extent: [{qy_min:.3f}, {qy_max:.3f}, {qz_min:.3f}, {qz_max:.3f}]")
                    
                    # 使用imshow显示Q轴坐标
                    self.current_image = self.ax.imshow(processed_data, cmap='viridis', aspect='equal', 
                                                      origin='lower', interpolation='nearest',
                                                      vmin=vmin, vmax=vmax, extent=q_extent)
                    print(f"  Successfully created Q-space display")
                else:
                    # 如果Q网格不可用，回退到普通extent
                    print(f"  Warning: Q-mesh not available - using fallback extent")
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
                    print(f"  Q-axis view auto-scaled")
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
            print(f"Independent window update error: {e}")
    
    def _convert_q_to_pixel_coordinates(self, center_qy, center_qz, width_q, height_q):
        """将Q坐标转换为像素坐标"""
        try:
            print(f"Converting Q to pixel coordinates:")
            print(f"  Input Q coords: center_qy={center_qy:.6f}, center_qz={center_qz:.6f}")
            print(f"  Input Q sizes: width_q={width_q:.6f}, height_q={height_q:.6f}")
            
            # 获取缓存的Q空间网格
            qy_mesh, qz_mesh = self._get_cached_q_meshgrids()
            
            if qy_mesh is None or qz_mesh is None:
                print("  Warning: Q meshgrids not available, using defaults")
                # 如果Q网格不可用，返回默认值
                return {'center_x': 0, 'center_y': 0, 'width': 100, 'height': 100}
            
            print(f"  Q meshgrid shapes: qy={qy_mesh.shape}, qz={qz_mesh.shape}")
            print(f"  Q ranges: qy=[{qy_mesh.min():.6f}, {qy_mesh.max():.6f}], qz=[{qz_mesh.min():.6f}, {qz_mesh.max():.6f}]")
            
            # 获取图像尺寸
            if hasattr(self, 'current_stack_data') and self.current_stack_data is not None:
                img_height, img_width = self.current_stack_data.shape
            else:
                img_height, img_width = qy_mesh.shape
            
            print(f"  Image size: {img_width}x{img_height}")
            
            # 创建像素坐标网格
            pixel_x = np.arange(img_width)
            pixel_y = np.arange(img_height)
            
            # 找到最接近目标Q坐标的像素位置
            # 对于中心点
            qy_diff = np.abs(qy_mesh - center_qy)
            qz_diff = np.abs(qz_mesh - center_qz)
            combined_diff = qy_diff + qz_diff
            center_idx = np.unravel_index(np.argmin(combined_diff), qy_mesh.shape)
            center_pixel_y, center_pixel_x = center_idx
            
            print(f"  Found center pixel: ({center_pixel_x}, {center_pixel_y})")
            
            # 计算Q空间到像素空间的比例因子
            qy_range = qy_mesh.max() - qy_mesh.min()
            qz_range = qz_mesh.max() - qz_mesh.min()
            pixel_x_range = img_width
            pixel_y_range = img_height
            
            qy_to_pixel_ratio = pixel_x_range / qy_range
            qz_to_pixel_ratio = pixel_y_range / qz_range
            
            print(f"  Q to pixel ratios: qy={qy_to_pixel_ratio:.3f}, qz={qz_to_pixel_ratio:.3f}")
            
            # 转换宽度和高度
            width_pixel = width_q * qy_to_pixel_ratio
            height_pixel = height_q * qz_to_pixel_ratio
            
            result = {
                'center_x': int(center_pixel_x),
                'center_y': int(center_pixel_y),
                'width': int(width_pixel),
                'height': int(height_pixel)
            }
            
            print(f"  Final pixel coords: {result}")
            return result
            
        except Exception as e:
            print(f"Error converting Q to pixel coordinates: {e}")
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
                
        except Exception as e:
            print(f"Error updating cutline labels: {e}")

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
            
            # 调试信息
            print(f"Q-axis calculation parameters:")
            print(f"  Image shape: {width}x{height}")
            print(f"  Pixel size: {pixel_size_x}x{pixel_size_y} μm")
            print(f"  Beam center: ({beam_center_x}, {beam_center_y}) pixels")
            print(f"  Distance: {distance} mm")
            print(f"  Incident angle: {theta_in_deg}°")
            print(f"  Wavelength: {wavelength} nm")
            
            # 创建缓存键
            cache_key = f"{width}x{height}_{pixel_size_x}_{pixel_size_y}_{beam_center_x}_{beam_center_y}_{distance}_{theta_in_deg}_{wavelength}"
            
            # 检查是否需要重新计算
            if self._q_cache_key != cache_key or self._q_detector is None:
                # 创建探测器对象并计算Q轴
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
                
                print(f"Q-space cache updated for image shape {width}x{height}")
                print(f"  Q-mesh shapes: qy={self._qy_mesh.shape}, qz={self._qz_mesh.shape}")
                print(f"  Qy range: [{self._qy_mesh.min():.6f}, {self._qy_mesh.max():.6f}]")
                print(f"  Qz range: [{self._qz_mesh.min():.6f}, {self._qz_mesh.max():.6f}]")
            
            _, _, extent = get_q_axis_labels_and_extents(self._q_detector)
            return extent
            
        except Exception as e:
            print(f"Error calculating Q-axis extent: {e}")
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
        
        # 创建matplotlib图形
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # 添加到布局
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # 创建axes
        self.ax = self.figure.add_subplot(111)
        
        # 设置窗口可以接收键盘焦点
        self.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        
        # 初始化空图
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
        
    def update_plot(self, x_coords, y_intensity, x_label, y_label, title, log_x=False, log_y=False, normalize=False):
        """更新拟合结果图"""
        try:
            # 数据预处理
            x_data = np.array(x_coords)
            y_data = np.array(y_intensity)
            
            # 应用标准化处理
            if normalize:
                max_intensity = np.max(y_data)
                if max_intensity > 0:
                    y_data = y_data / max_intensity
                    y_label = "Normalized Intensity"
            
            # 清除现有内容
            self.ax.clear()
            
            # 使用共享绘图函数
            FittingController._plot_cut_data_with_log_handling(self.ax, x_data, y_data, log_x, markersize=6, linewidth=2)
            
            # 设置标签和标题
            self.ax.set_xlabel(x_label, fontsize=12)
            self.ax.set_ylabel(y_label, fontsize=12)
            self.ax.set_title(title, fontsize=14, fontweight='bold')
            
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
            
            # 添加统计信息（位置：左下角）
            stats_text = f'Points: {len(x_data)}\nMax: {np.max(y_data):.2e}\nMin: {np.min(y_data):.2e}'
            self.ax.text(0.02, 0.02, stats_text, transform=self.ax.transAxes, 
                        verticalalignment='bottom', fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
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
            if not FABIO_AVAILABLE:
                self.error_occurred.emit("fabio library is required for CBF file processing")
                return
            
            self.progress_updated.emit(10, "开始加载文件...")
            
            file_ext = os.path.splitext(self.file_path)[1].lower()
            
            if file_ext != '.cbf':
                self.error_occurred.emit("目前只支持CBF文件格式")
                return
            
            if self.stack_count == 1:
                # 单文件加载
                self.progress_updated.emit(50, "加载单个CBF文件...")
                image_data = self._load_single_cbf_file(self.file_path)
            else:
                # 多文件叠加
                self.progress_updated.emit(30, f"加载并叠加 {self.stack_count} 个文件...")
                image_data = self._load_multiple_cbf_files(self.file_path, self.stack_count)
            
            if image_data is not None:
                self.progress_updated.emit(90, "处理图像数据...")
                self.image_loaded.emit(image_data, self.file_path)
                self.progress_updated.emit(100, "加载完成")
            else:
                self.error_occurred.emit("加载图像数据失败")
                
        except Exception as e:
            self.error_occurred.emit(f"加载图像时出错: {str(e)}")
    
    def _load_single_cbf_file(self, cbf_file):
        """加载单个CBF文件"""
        try:
            cbf_image = fabio.open(cbf_file)
            data = cbf_image.data
            
            if data.dtype != np.float32:
                data = data.astype(np.float32, copy=False)
            
            return data
            
        except Exception as e:
            print(f"Error loading single CBF file: {e}")
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
                print(f"Start file not found: {base_name}")
                return None
            
            available_files = len(cbf_files) - start_index
            if stack_count > available_files:
                print(f"Requested {stack_count} files, only {available_files} available")
                return None
            
            summed_data = None
            files_to_stack = cbf_files[start_index:start_index + stack_count]
            
            for i, file_name in enumerate(files_to_stack):
                file_path = os.path.join(file_dir, file_name)
                progress = 40 + int((i / len(files_to_stack)) * 40)
                self.progress_updated.emit(progress, f"处理文件 {i+1}/{len(files_to_stack)}: {file_name}")
                
                try:
                    cbf_image = fabio.open(file_path)
                    data = cbf_image.data.astype(np.float32, copy=False) if cbf_image.data.dtype != np.float32 else cbf_image.data
                    
                    if summed_data is None:
                        summed_data = data.copy() if data.dtype == np.float32 else data.astype(np.float32)
                    else:
                        summed_data += data
                        
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")
                    continue
            
            return summed_data
            
        except Exception as e:
            print(f"Error loading multiple CBF files: {e}")
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
        
        # 当前参数
        self.current_parameters = {}
        
        # 拟合结果
        self.fitting_results = {}
        
        # 图像处理相关
        self.current_stack_data = None
        self.current_file_list = []
        
        # 独立matplotlib窗口
        self.independent_window = None
        
        # 独立拟合结果窗口
        self.independent_fit_window = None
        
        # 当前切割结果数据 (用于独立窗口显示)
        self.current_cut_data = None
        
        # 模式状态跟踪（避免重复转换）
        self._last_q_mode = None
        
        # 图像显示相关的预初始化资源
        self._graphics_scene = None
        self._figure_cache = None
        self._canvas_cache = None
        
        # 颜色标尺相关
        self._current_vmin = None
        self._current_vmax = None
        self._has_displayed_image = False  # 标记是否已经显示过图像
        
        # 初始化标志
        self._initialized = False
        
        # 异步图像加载线程
        self.async_image_loader = AsyncImageLoader()
        self.async_image_loader.image_loaded.connect(self._on_image_loaded)
        self.async_image_loader.progress_updated.connect(self._on_image_loading_progress)
        self.async_image_loader.error_occurred.connect(self._on_image_loading_error)
        
        # 参数选择状态
        self.current_parameter_selection = None
        
    def initialize(self):
        """初始化控制器"""
        if self._initialized:
            return
            
        self._setup_connections()
        self._initialize_ui()
        # 会话管理已移到主控制器统一处理
        self._initialized = True
        
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
            
        # 连接Vmin/Vmax值变化
        if hasattr(self.ui, 'gisaxsInputVminValue'):
            self.ui.gisaxsInputVminValue.valueChanged.connect(self._on_vmin_value_changed)
            
        if hasattr(self.ui, 'gisaxsInputVmaxValue'):
            self.ui.gisaxsInputVmaxValue.valueChanged.connect(self._on_vmax_value_changed)
            
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
            
        if hasattr(self.ui, 'fitResetButton'):
            self.ui.fitResetButton.clicked.connect(self._reset_fitting)
            
        # 连接Cut Line和Center参数控件的信号（逆向选择功能）
        self._connect_cutline_parameter_signals()
            
        # 连接参数输入框的信号（如果存在的话）
        self._connect_parameter_widgets()
        
    def _connect_cutline_parameter_signals(self):
        """连接Cut Line和Center参数控件的信号，实现逆向选择功能"""
        # 连接Center参数控件 - 使用多重信号实现立即显示+延迟更新
        if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
            # 立即显示参数变化
            self.ui.gisaxsInputCenterVerticalValue.valueChanged.connect(self._on_parameter_display_changed)
            # 延迟图像更新（仅在编辑完成或延迟后触发）
            self.ui.gisaxsInputCenterVerticalValue.editingFinished.connect(self._on_cutline_parameters_immediate_update)
            
        if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
            # 立即显示参数变化
            self.ui.gisaxsInputCenterParallelValue.valueChanged.connect(self._on_parameter_display_changed)
            # 延迟图像更新
            self.ui.gisaxsInputCenterParallelValue.editingFinished.connect(self._on_cutline_parameters_immediate_update)
            
        # 连接Cut Line参数控件 - 同样的机制
        if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
            # 立即显示参数变化
            self.ui.gisaxsInputCutLineVerticalValue.valueChanged.connect(self._on_parameter_display_changed)
            # 延迟图像更新
            self.ui.gisaxsInputCutLineVerticalValue.editingFinished.connect(self._on_cutline_parameters_immediate_update)
            
        if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
            # 立即显示参数变化
            self.ui.gisaxsInputCutLineParallelValue.valueChanged.connect(self._on_parameter_display_changed)
            # 延迟图像更新
            self.ui.gisaxsInputCutLineParallelValue.editingFinished.connect(self._on_cutline_parameters_immediate_update)
    
    def _connect_parameter_widgets(self):
        """连接参数输入控件的信号"""
        # 这里可以根据UI文件中的具体控件来连接
        # 例如：裁剪区域参数、拟合参数等
        pass
        
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
            
        # 设置Log复选框默认选中
        if hasattr(self.ui, 'gisaxsInputIntLogCheckBox'):
            self.ui.gisaxsInputIntLogCheckBox.setChecked(True)
            
        # 设置AutoScale复选框默认选中
        if hasattr(self.ui, 'gisaxsInputAutoScaleCheckBox'):
            self.ui.gisaxsInputAutoScaleCheckBox.setChecked(True)
            
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
        
        # 检查依赖库
        self._check_dependencies()
        
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
        if not FABIO_AVAILABLE:
            self.status_updated.emit("Warning: fabio library not available. CBF processing will be disabled.")
        if not MATPLOTLIB_AVAILABLE:
            self.status_updated.emit("Warning: matplotlib not available. Image display will be disabled.")
    
    def _is_q_space_mode(self):
        """检查当前是否为Q-space模式"""
        try:
            # 使用和_should_show_q_axis()相同的逻辑
            return self._should_show_q_axis()
        except Exception:
            return False
    
    def _on_q_mode_changed(self):
        """Q模式切换时的处理"""
        try:
            # 更新步长
            self._update_cutline_step_sizes()
            
            # 立即更新参数显示（无论是否有Cut数据）
            self._update_parameter_values_for_q_axis()
            
            # 如果有Cut数据，还需要更新图像显示（但添加防抖动机制）
            if hasattr(self, '_cut_data') and self._cut_data is not None:
                # 使用QTimer延迟执行，避免频繁更新
                if not hasattr(self, '_q_mode_timer'):
                    from PyQt5.QtCore import QTimer
                    self._q_mode_timer = QTimer()
                    self._q_mode_timer.setSingleShot(True)
                    self._q_mode_timer.timeout.connect(self._delayed_cut_update)
                
                self._q_mode_timer.stop()
                self._q_mode_timer.start(100)  # 100ms延迟
                
        except Exception as e:
            print(f"Q模式切换处理失败: {e}")
    
    def _delayed_cut_update(self):
        """延迟执行Cut更新，避免频繁操作"""
        try:
            # 仅重新执行Cut并更新图像显示
            if hasattr(self, '_cut_data') and self._cut_data is not None:
                # 重新执行Cut操作以更新图像
                self._execute_cut()
        except Exception as e:
            print(f"延迟Cut更新失败: {e}")
    
    def _on_parameter_display_changed(self):
        """参数显示立即更新（不更新图像）"""
        try:
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
            print(f"参数显示更新失败: {e}")
    
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
            print(f"延迟更新触发失败: {e}")
    
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
            print(f"延迟Cut图像更新失败: {e}")
    
    def _on_cutline_parameters_immediate_update(self):
        """编辑完成时立即更新（用于回车键或失去焦点）"""
        try:
            # 停止延迟定时器，立即执行更新
            if hasattr(self, '_cut_update_timer'):
                self._cut_update_timer.stop()
            
            # 立即执行图像更新
            self._delayed_cut_image_update()
            
        except Exception as e:
            print(f"立即更新失败: {e}")
    
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
            print(f"Error updating cutline labels: {e}")
    
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
            print(f"Error getting Q meshgrids: {e}")
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
            self.status_updated.emit(f"已导入GISAXS文件: {os.path.basename(file_path)}")
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
                QMessageBox.warning(self.main_window, "文件错误", f"文件不存在: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                QMessageBox.warning(self.main_window, "文件错误", "文件为空")
                return False
            
            file_ext = os.path.splitext(file_path)[1].lower()
            supported_extensions = ['.tif', '.tiff', '.dat', '.txt', '.h5', '.hdf5', '.jpg', '.png', '.bmp', '.cbf']
            
            if file_ext not in supported_extensions:
                reply = QMessageBox.question(
                    self.main_window,
                    "文件格式警告",
                    f"文件格式 '{file_ext}' 可能不被支持。\n支持的格式: {', '.join(supported_extensions)}\n\n是否继续导入？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return False
            
            self.status_updated.emit(f"文件验证通过 - {os.path.basename(file_path)}")
            return True
            
        except Exception as e:
            QMessageBox.critical(self.main_window, "文件验证错误", f"验证文件时出错:\n{str(e)}")
            return False
    
    def _on_import_value_changed(self):
        """当Import Value输入框内容改变且按回车时的处理"""
        try:
            if not hasattr(self.ui, 'gisaxsInputImportButtonValue'):
                return
                
            file_path_input = self.ui.gisaxsInputImportButtonValue.text().strip()
            
            if not file_path_input:
                self.status_updated.emit("请输入有效的文件路径")
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
                self.status_updated.emit(f"文件不存在: {os.path.basename(file_path_input)}")
                QMessageBox.warning(self.main_window, "文件错误", f"文件不存在:\n{file_path_input}")
                return
            
            # 更新参数
            self.current_parameters['imported_gisaxs_file'] = file_path_input
            
            # 更新UI显示为文件名
            file_name = os.path.basename(file_path_input)
            self.ui.gisaxsInputImportButtonValue.setText(file_name)
            
            # 验证文件
            if self._validate_imported_file(file_path_input):
                self.status_updated.emit(f"已更新GISAXS文件: {file_name}")
                self.parameters_changed.emit(self.current_parameters)
                
                # 触发主控制器保存会话
                if hasattr(self.parent, 'save_current_session'):
                    self.parent.save_current_session()
                
                self._update_stack_display()
                self._refresh_vmin_vmax_display()
                
                if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked():
                    self._show_image()
                else:
                    self.status_updated.emit(f"文件已更新，点击Show按钮显示图像")
            
        except Exception as e:
            self.status_updated.emit(f"Import value processing error: {str(e)}")
            QMessageBox.critical(self.main_window, "处理错误", f"处理导入文件路径时出错:\n{str(e)}")
    
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
        """更新stack显示信息"""
        try:
            imported_file = self.current_parameters.get('imported_gisaxs_file', '')
            if not imported_file:
                return
            
            file_ext = os.path.splitext(imported_file)[1].lower()
            stack_count = self.current_parameters.get('stack_count', 1)
            
            if file_ext != '.cbf':
                if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                    self.ui.gisaxsInputStackDisplayLabel.setText(f"Single File: {os.path.basename(imported_file)}")
                return
            
            if stack_count == 1:
                if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                    self.ui.gisaxsInputStackDisplayLabel.setText(f"Single File: {os.path.basename(imported_file)}")
            else:
                file_dir = os.path.dirname(imported_file)
                base_name = os.path.basename(imported_file)
                
                cbf_files = []
                for file in os.listdir(file_dir):
                    if file.lower().endswith('.cbf'):
                        cbf_files.append(file)
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
                            self.ui.gisaxsInputStackDisplayLabel.setText(f"{start_name} - {end_name}")
                            
                except ValueError:
                    if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                        self.ui.gisaxsInputStackDisplayLabel.setText("File not found in directory")
                        
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
            
            # 同步Stack数值
            if hasattr(self.ui, 'gisaxsInputStackValue'):
                try:
                    stack_text = self.ui.gisaxsInputStackValue.text().strip()
                    if stack_text:
                        stack_count = int(stack_text)
                        self.current_parameters['stack_count'] = max(1, stack_count)
                except ValueError:
                    # 如果转换失败，使用默认值
                    self.current_parameters['stack_count'] = 1
                    
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
            if not FABIO_AVAILABLE:
                QMessageBox.warning(self.main_window, "Missing Library", 
                                  "fabio library is required for CBF file processing.\nPlease install it using: pip install fabio")
                return
                
            if not MATPLOTLIB_AVAILABLE:
                QMessageBox.warning(self.main_window, "Missing Library", 
                                  "matplotlib library is required for image display.\nPlease install it using: pip install matplotlib")
                return
            
            # 处理文件并显示
            file_ext = os.path.splitext(imported_file)[1].lower()
            stack_count = self.current_parameters.get('stack_count', 1)
            
            if file_ext != '.cbf':
                self.status_updated.emit("Image display only supports CBF files currently")
                return
            
            # 使用异步加载
            self.status_updated.emit("开始加载图像，请稍候...")
            self.async_image_loader.load_image(imported_file, stack_count)
            
        except Exception as e:
            self.status_updated.emit(f"Show image error: {str(e)}")
    
    def _on_image_loaded(self, image_data, file_path):
        """图像加载完成"""
        try:
            self.status_updated.emit(f"图像加载完成: {os.path.basename(file_path)}")
            self._display_image(image_data)
        except Exception as e:
            self.status_updated.emit(f"显示图像时出错: {str(e)}")
    
    def _on_image_loading_progress(self, progress, status):
        """图像加载进度更新"""
        try:
            self.status_updated.emit(f"图像加载中... {progress}% - {status}")
            self.progress_updated.emit(progress)
        except Exception as e:
            self.status_updated.emit(f"进度更新错误: {str(e)}")
    
    def _on_image_loading_error(self, error_message):
        """图像加载错误处理"""
        QMessageBox.critical(self.main_window, "图像加载错误", error_message)
    
    def _display_image(self, image_data):
        """显示图像数据"""
        try:
            # 存储当前数据
            self.current_stack_data = image_data
            
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
            if not MATPLOTLIB_AVAILABLE:
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
                    self.status_updated.emit(f"Q空间参数更新失败: {str(e)}")
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
        """fitGraphicsView双击事件处理 - 打开独立拟合结果窗口"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                QMessageBox.warning(self.main_window, "Missing Library",
                                  "matplotlib library is required for independent window.\nPlease install it using: pip install matplotlib")
                return
            
            # 如果没有切割数据，提示用户
            if self.current_cut_data is None:
                QMessageBox.information(self.main_window, "No Cut Data", "Please perform a cut operation first.")
                return
            
            # 打开或显示独立拟合窗口
            self._show_independent_fit_window()
            
        except Exception as e:
            self.status_updated.emit(f"Fit double-click error: {str(e)}")
            
    def _show_independent_fit_window(self):
        """显示独立的拟合结果matplotlib窗口"""
        try:
            # 如果窗口不存在或已关闭，创建新窗口
            if self.independent_fit_window is None or not self.independent_fit_window.isVisible():
                self.independent_fit_window = IndependentFitWindow(self.main_window)
                # 连接状态更新信号
                self.independent_fit_window.status_updated.connect(self.status_updated.emit)
            
            # 更新窗口中的拟合结果
            if self.current_cut_data is not None:
                self.independent_fit_window.update_plot(
                    self.current_cut_data['x_coords'],
                    self.current_cut_data['y_intensity'],
                    self.current_cut_data['x_label'],
                    self.current_cut_data['y_label'],
                    self.current_cut_data['title'],
                    log_x=self._is_fit_log_x_enabled(),
                    log_y=self._is_fit_log_y_enabled(),
                    normalize=self._is_fit_norm_enabled()
                )
            
            # 显示窗口并置于前台
            self.independent_fit_window.show()
            self.independent_fit_window.raise_()
            self.independent_fit_window.activateWindow()
            
            # 设置焦点
            self.independent_fit_window.canvas.setFocus()
            
            self.status_updated.emit("Independent fit window opened - Enhanced visualization of cut results")
            
        except Exception as e:
            self.status_updated.emit(f"Independent fit window error: {str(e)}")
    
    def _on_cutline_parameters_changed(self):
        """当Cut Line参数改变时，更新图像中的选择框显示，并自动重新执行Cut操作（如果之前已有Cut结果）"""
        try:
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
            print(f"Cut Line参数变化处理失败: {e}")
    
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
            if not MATPLOTLIB_AVAILABLE:
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
            
            # 创建figure，降低DPI以提高性能
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
                    print(f"Q-axis display error: {e}")
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
            
            # 设置canvas尺寸
            min_width = int(fig_width * 72)
            min_height = int(fig_height * 72)
            canvas.setMinimumSize(min_width, min_height)
            
            # 调整视图
            graphics_view.fitInView(proxy_widget, Qt.KeepAspectRatio)
            graphics_view.update()
            
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
        try:
            if hasattr(self.ui, 'gisaxsInputAutoScaleCheckBox'):
                return self.ui.gisaxsInputAutoScaleCheckBox.isChecked()
            return True
        except Exception:
            return True
    
    def _is_log_mode_enabled(self):
        """检查是否启用Log模式"""
        try:
            if hasattr(self.ui, 'gisaxsInputIntLogCheckBox'):
                return self.ui.gisaxsInputIntLogCheckBox.isChecked()
            return True
        except Exception:
            return True
    
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
    
    # ========== 拟合相关方法 ==========
        
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
            self.status_updated.emit("开始Cut Fitting处理...")
            self.progress_updated.emit(0)
            
            # TODO: 实现实际的拟合逻辑
            self._run_fitting_process()
            
            self.progress_updated.emit(100)
            self.status_updated.emit("Cut Fitting处理完成！")
            
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
            self.status_updated.emit("正在自动寻找中心点...")
            
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
                    
                    self.status_updated.emit(f"自动寻找完成 (Q坐标): Center({center_qy:.6f}, {center_qz:.6f}) nm⁻¹, CutLine({cutline_width_q:.6f}, {cutline_height_q:.6f}) nm⁻¹")
                    
                except Exception as e:
                    self.status_updated.emit(f"Q坐标转换失败，使用像素坐标: {str(e)}")
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
                
                self.status_updated.emit(f"自动寻找完成 (像素坐标): Center({pixel_center_x:.1f}, {pixel_center_y:.1f}), CutLine({pixel_cutline_width:.1f}, {pixel_cutline_height:.1f})")
            
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
            self.status_updated.emit(f"自动寻找中心失败: {str(e)}")
    
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
            # 创建对话框
            dialog = DetectorParametersDialog(self.main_window)
            
            # 连接参数改变信号
            dialog.parameters_changed.connect(self._on_detector_parameters_changed)
            
            # 显示对话框
            result = dialog.exec_()
            
            if result == dialog.Accepted:
                self.status_updated.emit("探测器参数已更新")
            
        except Exception as e:
            self.status_updated.emit(f"显示探测器参数对话框失败: {str(e)}")
            QMessageBox.critical(
                self.main_window,
                "错误",
                f"无法显示探测器参数对话框：{str(e)}"
            )
            
    def _on_detector_parameters_changed(self, parameters):
        """处理探测器参数改变"""
        try:
            # 更新Cut Line标签的单位（因为show_q_axis可能改变了）
            self._update_cutline_labels_units()
            
            # 更新Cut Line参数的步长（因为显示模式可能改变了）
            self._update_cutline_step_sizes()
            
            # 如果Q轴显示状态改变，需要转换现有的数值
            self._update_parameter_values_for_q_axis()
            
            # 检查是否已经有Cut结果数据，如果有则自动重新执行Cut操作
            if (self.current_cut_data is not None and 
                hasattr(self, 'current_stack_data') and self.current_stack_data is not None):
                
                # 自动重新执行Cut操作
                self._perform_cut()
                self.status_updated.emit("探测器参数已更新，Cut结果已自动重新计算")
            else:
                self.status_updated.emit("探测器参数已更新并保存")
            
        except Exception as e:
            self.status_updated.emit(f"处理探测器参数改变失败: {str(e)}")
    
    def _update_parameter_values_for_q_axis(self):
        """根据Q轴显示状态切换时转换参数数值并更新显示"""
        try:
            show_q_axis = self._should_show_q_axis()
            print(f"DEBUG: Q模式状态 = {show_q_axis}, 上次状态 = {getattr(self, '_last_q_mode', None)}")
            
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
            
            print(f"DEBUG: 当前参数值 - CV:{center_vertical}, CP:{center_parallel}, CLV:{cutline_vertical}, CLP:{cutline_parallel}")
            
            # 检查是否有图像数据用于转换
            if self.current_stack_data is None:
                print("DEBUG: 无图像数据，无法进行坐标转换")
                self.status_updated.emit("无图像数据，无法进行坐标转换")
                return
            
            # 创建detector用于坐标转换
            detector = self._get_detector_for_conversion()
            if detector is None:
                print("DEBUG: 无法创建detector，坐标转换失败")
                self.status_updated.emit("无法创建detector，坐标转换失败")
                return
            
            # 检查当前参数是否已经是目标模式（避免重复转换）
            if hasattr(self, '_last_q_mode') and self._last_q_mode == show_q_axis:
                print("DEBUG: 模式未变化，跳过转换")
                return
            
            # 1. 转换Cut line和Center的四个数值
            if show_q_axis:
                # Pixel -> Q-space转换
                print("DEBUG: 执行 Pixel -> Q-space 转换")
                new_center_parallel, new_center_vertical, new_cutline_parallel, new_cutline_vertical = \
                    self._convert_pixel_to_q_parameters(center_parallel, center_vertical, cutline_parallel, cutline_vertical, detector)
                conversion_msg = "参数已从像素坐标转换为Q空间坐标"
            else:
                # Q-space -> Pixel转换
                print("DEBUG: 执行 Q-space -> Pixel 转换")
                new_center_parallel, new_center_vertical, new_cutline_parallel, new_cutline_vertical = \
                    self._convert_q_to_pixel_parameters(center_parallel, center_vertical, cutline_parallel, cutline_vertical, detector)
                conversion_msg = "参数已从Q空间坐标转换为像素坐标"
            
            print(f"DEBUG: 转换后参数值 - CV:{new_center_vertical}, CP:{new_center_parallel}, CLV:{new_cutline_vertical}, CLP:{new_cutline_parallel}")
            
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
            
            print(f"DEBUG: 转换完成，新模式状态 = {self._last_q_mode}")
            self.status_updated.emit(conversion_msg)
                
        except Exception as e:
            print(f"DEBUG: 模式切换和参数转换失败: {str(e)}")
            self.status_updated.emit(f"模式切换和参数转换失败: {str(e)}")
    
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
            self.status_updated.emit(f"创建detector失败: {str(e)}")
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
            self.status_updated.emit(f"像素到Q空间转换失败: {str(e)}")
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
            self.status_updated.emit(f"Q空间到像素转换失败: {str(e)}")
            return q_parallel, q_vertical, q_cutline_parallel, q_cutline_vertical
    
    def _temporarily_disconnect_parameter_signals(self):
        """暂时断开参数变化信号连接"""
        try:
            if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                self.ui.gisaxsInputCenterVerticalValue.valueChanged.disconnect()
            if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                self.ui.gisaxsInputCenterParallelValue.valueChanged.disconnect()
            if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                self.ui.gisaxsInputCutLineVerticalValue.valueChanged.disconnect()
            if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                self.ui.gisaxsInputCutLineParallelValue.valueChanged.disconnect()
        except Exception:
            pass
    
    def _reconnect_parameter_signals(self):
        """重新连接参数变化信号"""
        try:
            if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                self.ui.gisaxsInputCenterVerticalValue.valueChanged.connect(self._on_cutline_parameters_changed)
            if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                self.ui.gisaxsInputCenterParallelValue.valueChanged.connect(self._on_cutline_parameters_changed)
            if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                self.ui.gisaxsInputCutLineVerticalValue.valueChanged.connect(self._on_cutline_parameters_changed)
            if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                self.ui.gisaxsInputCutLineParallelValue.valueChanged.connect(self._on_cutline_parameters_changed)
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
            self.status_updated.emit(f"刷新显示失败: {str(e)}")

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
            self.status_updated.emit(f"Q空间转换失败: {str(e)}")
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
            print(f"计算图像中心失败: {e}")
            return 500.0, 500.0

    # ========== 会话管理方法 ==========
    
    def get_session_data(self):
        """获取当前会话数据（供主控制器调用）"""
        try:
            session_data = {
                'last_opened_file': self.current_parameters.get('imported_gisaxs_file', ''),
                'last_directory': os.path.dirname(self.current_parameters.get('imported_gisaxs_file', '')) if self.current_parameters.get('imported_gisaxs_file') else '',
                'stack_count': self.current_parameters.get('stack_count', 1),
                'center_vertical': self.ui.gisaxsInputCenterVerticalValue.value() if hasattr(self.ui, 'gisaxsInputCenterVerticalValue') else 0.0,
                'center_parallel': self.ui.gisaxsInputCenterParallelValue.value() if hasattr(self.ui, 'gisaxsInputCenterParallelValue') else 0.0,
                'cutline_vertical': self.ui.gisaxsInputCutLineVerticalValue.value() if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue') else 10.0,
                'cutline_parallel': self.ui.gisaxsInputCutLineParallelValue.value() if hasattr(self.ui, 'gisaxsInputCutLineParallelValue') else 10.0,
                'auto_show': self.ui.gisaxsInputAutoShowCheckBox.isChecked() if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') else False,
                'log_mode': self.ui.gisaxsInputIntLogCheckBox.isChecked() if hasattr(self.ui, 'gisaxsInputIntLogCheckBox') else True,
                'auto_scale': self.ui.gisaxsInputAutoScaleCheckBox.isChecked() if hasattr(self.ui, 'gisaxsInputAutoScaleCheckBox') else True,
                'vmin': self._current_vmin if self._current_vmin is not None else 0.0,
                'vmax': self._current_vmax if self._current_vmax is not None else 0.0
            }
            return session_data
        except Exception as e:
            print(f"Failed to get session data: {str(e)}")
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
            
            # 恢复Stack设置
            stack_count = session_data.get('stack_count', 1)
            if hasattr(self.ui, 'gisaxsInputStackValue'):
                self.ui.gisaxsInputStackValue.setText(str(stack_count))
            self.current_parameters['stack_count'] = stack_count
            
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
            
            # 更新显示
            self._update_stack_display()
            
            # 如果AutoShow启用且有文件，自动显示
            if (session_data.get('auto_show', False) and 
                last_file and os.path.exists(last_file)):
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(500, self._show_image)  # 延迟500ms显示图像
            
            self.status_updated.emit("Previous session restored successfully")
            
        except Exception as e:
            self.status_updated.emit(f"Failed to restore session: {str(e)}")
    
    # ========== Cut功能相关方法 ==========
    
    def _perform_cut(self):
        """执行Cut操作 - gisaxsInputCutButton按钮的处理逻辑"""
        try:
            # 1. 检查是否导入了图像数据
            if self.current_stack_data is None:
                QMessageBox.warning(
                    self.main_window,
                    "Warning",
                    "Please import an image first."
                )
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
                QMessageBox.warning(
                    self.main_window,
                    "Warning",
                    "Please select a valid region."
                )
                return
            
            # 4. 确定切割模式
            if vertical_value <= parallel_value:
                # 横切模式
                self._perform_horizontal_cut(vertical_value, parallel_value)
                self.status_updated.emit(f"Horizontal cut performed: Vertical={vertical_value:.2f}, Parallel={parallel_value:.2f}")
            else:
                # 纵切模式
                self._perform_vertical_cut(vertical_value, parallel_value)
                self.status_updated.emit(f"Vertical cut performed: Vertical={vertical_value:.2f}, Parallel={parallel_value:.2f}")
                
        except Exception as e:
            self.status_updated.emit(f"Cut operation failed: {str(e)}")
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"Cut operation failed:\n{str(e)}"
            )
    
    def _perform_horizontal_cut(self, vertical_value, parallel_value):
        """执行横切操作"""
        try:
            # 获取切割区域的中心点
            center_x = 0.0
            center_y = 0.0
            
            if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                center_x = self.ui.gisaxsInputCenterParallelValue.value()
            if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                center_y = self.ui.gisaxsInputCenterVerticalValue.value()
            
            # 检查是否为Q坐标模式
            show_q_axis = self._should_show_q_axis()
            
            if show_q_axis:
                # Q坐标模式：直接使用Q坐标
                cut_data, q_coords = self._extract_horizontal_cut_q_mode(
                    center_x, center_y, vertical_value, parallel_value
                )
                x_label = r'$q_y$ (nm$^{-1}$)'
                x_coordinates = q_coords
            else:
                # 像素坐标模式：提取像素数据后转换为Q坐标
                cut_data, pixel_coords = self._extract_horizontal_cut_pixel_mode(
                    center_x, center_y, vertical_value, parallel_value
                )
                # 转换像素坐标到qy
                x_coordinates = self._convert_pixel_to_qy(pixel_coords)
                x_label = r'$q_y$ (nm$^{-1}$)'
            
            # 绘制结果
            self._plot_cut_result(x_coordinates, cut_data, x_label, "Intensity (a.u.)", "Horizontal Cut")
            
        except Exception as e:
            raise Exception(f"Horizontal cut failed: {str(e)}")
    
    def _perform_vertical_cut(self, vertical_value, parallel_value):
        """执行纵切操作"""
        try:
            # 获取切割区域的中心点
            center_x = 0.0
            center_y = 0.0
            
            if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                center_x = self.ui.gisaxsInputCenterParallelValue.value()
            if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                center_y = self.ui.gisaxsInputCenterVerticalValue.value()
            
            # 检查是否为Q坐标模式
            show_q_axis = self._should_show_q_axis()
            
            if show_q_axis:
                # Q坐标模式：直接使用Q坐标
                cut_data, q_coords = self._extract_vertical_cut_q_mode(
                    center_x, center_y, vertical_value, parallel_value
                )
                x_label = r'$q_z$ (nm$^{-1}$)'
                x_coordinates = q_coords
            else:
                # 像素坐标模式：提取像素数据后转换为Q坐标
                cut_data, pixel_coords = self._extract_vertical_cut_pixel_mode(
                    center_x, center_y, vertical_value, parallel_value
                )
                # 转换像素坐标到qz
                x_coordinates = self._convert_pixel_to_qz(pixel_coords)
                x_label = r'$q_z$ (nm$^{-1}$)'
            
            # 绘制结果
            self._plot_cut_result(x_coordinates, cut_data, x_label, "Intensity (a.u.)", "Vertical Cut")
            
        except Exception as e:
            raise Exception(f"Vertical cut failed: {str(e)}")
    
    def _extract_horizontal_cut_q_mode(self, center_qy, center_qz, height_q, width_q):
        """Q坐标模式下的横切数据提取"""
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
            
            # 在区域内沿纵向求和
            region_data = np.where(mask, self.current_stack_data, 0)
            
            # 沿纵向求和得到横向分布
            horizontal_sum = np.sum(region_data, axis=0)
            
            # 获取对应的qy坐标
            qy_line = qy_mesh[0, :]  # 取第一行的qy值
            
            # 过滤有效数据点
            valid_indices = horizontal_sum > 0
            if not np.any(valid_indices):
                raise Exception("No valid data in the selected region")
            
            valid_qy = qy_line[valid_indices]
            valid_intensity = horizontal_sum[valid_indices]
            
            # 插值到50个点
            qy_interp = np.linspace(valid_qy.min(), valid_qy.max(), 50)
            intensity_interp = np.interp(qy_interp, valid_qy, valid_intensity)
            
            return intensity_interp, qy_interp
            
        except Exception as e:
            raise Exception(f"Q-mode horizontal cut extraction failed: {str(e)}")
    
    def _extract_vertical_cut_q_mode(self, center_qy, center_qz, height_q, width_q):
        """Q坐标模式下的纵切数据提取"""
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
            
            # 在区域内沿横向求和
            region_data = np.where(mask, self.current_stack_data, 0)
            
            # 沿横向求和得到纵向分布
            vertical_sum = np.sum(region_data, axis=1)
            
            # 获取对应的qz坐标
            qz_line = qz_mesh[:, 0]  # 取第一列的qz值
            
            # 过滤有效数据点
            valid_indices = vertical_sum > 0
            if not np.any(valid_indices):
                raise Exception("No valid data in the selected region")
            
            valid_qz = qz_line[valid_indices]
            valid_intensity = vertical_sum[valid_indices]
            
            # 插值到50个点
            qz_interp = np.linspace(valid_qz.min(), valid_qz.max(), 50)
            intensity_interp = np.interp(qz_interp, valid_qz, valid_intensity)
            
            return intensity_interp, qz_interp
            
        except Exception as e:
            raise Exception(f"Q-mode vertical cut extraction failed: {str(e)}")
    
    def _extract_horizontal_cut_pixel_mode(self, center_x, center_y, height, width):
        """像素坐标模式下的横切数据提取"""
        try:
            img_height, img_width = self.current_stack_data.shape
            
            # 计算像素边界（注意图像坐标系统）
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
            
            # 沿纵向求和得到横向分布
            horizontal_sum = np.sum(region_data, axis=0)
            
            # 生成像素坐标
            pixel_coords = np.arange(x_min, x_min + len(horizontal_sum))
            
            # 插值到50个点
            if len(pixel_coords) > 1:
                pixel_interp = np.linspace(pixel_coords.min(), pixel_coords.max(), 50)
                intensity_interp = np.interp(pixel_interp, pixel_coords, horizontal_sum)
            else:
                pixel_interp = pixel_coords
                intensity_interp = horizontal_sum
            
            return intensity_interp, pixel_interp
            
        except Exception as e:
            raise Exception(f"Pixel-mode horizontal cut extraction failed: {str(e)}")
    
    def _extract_vertical_cut_pixel_mode(self, center_x, center_y, height, width):
        """像素坐标模式下的纵切数据提取"""
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
            
            # 沿横向求和得到纵向分布
            vertical_sum = np.sum(region_data, axis=1)
            
            # 生成像素坐标（原始坐标，未翻转）
            pixel_coords = np.arange(y_min, y_min + len(vertical_sum))
            
            # 插值到50个点
            if len(pixel_coords) > 1:
                pixel_interp = np.linspace(pixel_coords.min(), pixel_coords.max(), 50)
                intensity_interp = np.interp(pixel_interp, pixel_coords, vertical_sum)
            else:
                pixel_interp = pixel_coords
                intensity_interp = vertical_sum
            
            return intensity_interp, pixel_interp
            
        except Exception as e:
            raise Exception(f"Pixel-mode vertical cut extraction failed: {str(e)}")
    
    def _convert_pixel_to_qy(self, pixel_coords):
        """将像素坐标转换为qy坐标"""
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
                wavelength=wavelength,
                crop_params=None
            )
            
            # 转换坐标（使用图像中心的y坐标）
            center_y = height / 2.0
            qy_coords = []
            for px in pixel_coords:
                _, qy, _ = detector.pixel_to_q_space(px, center_y)
                qy_coords.append(qy)
            
            return np.array(qy_coords)
            
        except Exception as e:
            # 如果转换失败，返回归一化的像素坐标
            self.status_updated.emit(f"Pixel to qy conversion failed: {str(e)}")
            return (pixel_coords - pixel_coords.mean()) / pixel_coords.std()
    
    def _convert_pixel_to_qz(self, pixel_coords):
        """将像素坐标转换为qz坐标"""
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
                wavelength=wavelength,
                crop_params=None
            )
            
            # 转换坐标（使用图像中心的x坐标）
            center_x = width / 2.0
            qz_coords = []
            for py in pixel_coords:
                _, _, qz = detector.pixel_to_q_space(center_x, py)
                qz_coords.append(qz)
            
            return np.array(qz_coords)
            
        except Exception as e:
            # 如果转换失败，返回归一化的像素坐标
            self.status_updated.emit(f"Pixel to qz conversion failed: {str(e)}")
            return (pixel_coords - pixel_coords.mean()) / pixel_coords.std()
    
    def _plot_cut_result(self, x_coords, y_intensity, x_label, y_label, title):
        """在fitGraphicsView中绘制切割结果"""
        try:
            # 存储当前切割数据，用于独立窗口显示
            self.current_cut_data = {
                'x_coords': x_coords.copy() if hasattr(x_coords, 'copy') else list(x_coords),
                'y_intensity': y_intensity.copy() if hasattr(y_intensity, 'copy') else list(y_intensity),
                'x_label': x_label,
                'y_label': y_label,
                'title': title
            }
            
            if not MATPLOTLIB_AVAILABLE:
                QMessageBox.warning(self.main_window, "Missing Library",
                                  "matplotlib library is required for plotting.\nPlease install it using: pip install matplotlib")
                return
            
            # 检查fitGraphicsView是否存在
            if not hasattr(self.ui, 'fitGraphicsView'):
                self.status_updated.emit("fitGraphicsView not found in UI")
                return
            
            # 应用标准化处理
            y_data = np.array(y_intensity)
            if self._is_fit_norm_enabled():
                max_intensity = np.max(y_data)
                if max_intensity > 0:
                    y_data = y_data / max_intensity
                    y_label = "Normalized Intensity"
            
            # 创建图形
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from PyQt5.QtWidgets import QGraphicsScene
            
            # 清除现有内容
            if not hasattr(self, '_fit_graphics_scene') or self._fit_graphics_scene is None:
                self._fit_graphics_scene = QGraphicsScene()
                self.ui.fitGraphicsView.setScene(self._fit_graphics_scene)
            else:
                self._fit_graphics_scene.clear()
            
            # 创建matplotlib图形
            fig = Figure(figsize=(8, 6), dpi=80)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            
            # 使用共享函数绘制数据
            is_log_x = self._is_fit_log_x_enabled()
            self._plot_cut_data_with_log_handling(ax, x_coords, y_data, is_log_x, markersize=4, linewidth=1.5)
            
            # 设置坐标轴
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # 应用对数坐标设置
            if is_log_x:
                ax.set_xscale('log')
            if self._is_fit_log_y_enabled():
                ax.set_yscale('log')
            
            # 调整布局
            fig.tight_layout()
            
            # 添加到场景
            proxy_widget = self._fit_graphics_scene.addWidget(canvas)
            
            # 调整视图
            self.ui.fitGraphicsView.fitInView(proxy_widget)
            
            # 同步更新独立拟合窗口（如果打开的话）
            if self.independent_fit_window is not None and self.independent_fit_window.isVisible():
                self.independent_fit_window.update_plot(
                    self.current_cut_data['x_coords'],
                    self.current_cut_data['y_intensity'],
                    self.current_cut_data['x_label'],
                    self.current_cut_data['y_label'],
                    self.current_cut_data['title'],
                    log_x=self._is_fit_log_x_enabled(),
                    log_y=self._is_fit_log_y_enabled(),
                    normalize=self._is_fit_norm_enabled()
                )
            
            self.status_updated.emit(f"Cut result plotted: {title}")
            
        except Exception as e:
            self.status_updated.emit(f"Plot failed: {str(e)}")
            QMessageBox.critical(
                self.main_window,
                "Plot Error",
                f"Failed to plot cut result:\n{str(e)}"
            )
    
    def _is_fit_log_x_enabled(self):
        """检查是否启用X轴对数显示"""
        try:
            if hasattr(self.ui, 'fitLogXCheckBox'):
                return self.ui.fitLogXCheckBox.isChecked()
            return False
        except Exception:
            return False
    
    def _is_fit_log_y_enabled(self):
        """检查是否启用Y轴对数显示"""
        try:
            if hasattr(self.ui, 'fitLogYCheckBox'):
                return self.ui.fitLogYCheckBox.isChecked()
            return False
        except Exception:
            return False
    
    def _is_fit_norm_enabled(self):
        """检查是否启用标准化"""
        try:
            if hasattr(self.ui, 'fitNormCheckBox'):
                return self.ui.fitNormCheckBox.isChecked()
            return False
        except Exception:
            return False
            
        except Exception as e:
            self.status_updated.emit(f"Failed to restore session: {str(e)}")
