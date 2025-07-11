"""
Cut Fitting 控制器 - 处理GISAXS数据的裁剪和拟合功能
"""

import os
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QThread, QTimer
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGraphicsScene, QVBoxLayout, QWidget, QMainWindow

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

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


class IndependentMatplotlibWindow(QMainWindow):
    """独立的matplotlib窗口，支持完整的交互操作和视图保持"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GISAXS Image Viewer - Independent Window")
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
        
        # 连接视图变化事件
        self.ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
        self.ax.callbacks.connect('ylim_changed', self._on_ylim_changed)
    
    def _on_xlim_changed(self, ax):
        """X轴范围改变时的回调"""
        self.current_xlim = ax.get_xlim()
    
    def _on_ylim_changed(self, ax):
        """Y轴范围改变时的回调"""
        self.current_ylim = ax.get_ylim()
    
    def update_image(self, image_data, vmin=None, vmax=None, use_log=True):
        """更新独立窗口中的图像，保持用户的视图焦点，支持自定义颜色范围和线性/对数切换"""
        try:
            # 检查图像尺寸是否改变
            current_shape = image_data.shape
            shape_changed = (self.last_image_shape is None or 
                           self.last_image_shape != current_shape)
            
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
            
            # 显示图像，使用指定的颜色范围
            self.current_image = self.ax.imshow(processed_data, cmap='viridis', aspect='equal', 
                                              origin='lower', interpolation='nearest',
                                              vmin=vmin, vmax=vmax)
            
            # 设置标题
            self.ax.set_title(f'GISAXS Image ({scale_text}) - {image_data.shape[1]}×{image_data.shape[0]}\nVmin: {vmin:.3f}, Vmax: {vmax:.3f}')
            
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
                # 设置为默认的全图显示
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
        self.parent = parent
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
        
    def initialize(self):
        """初始化控制器"""
        if self._initialized:
            return
            
        self._setup_connections()
        self._initialize_ui()
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
            
        # 连接Vmin/Vmax值变化
        if hasattr(self.ui, 'gisaxsInputVminValue'):
            self.ui.gisaxsInputVminValue.valueChanged.connect(self._on_vmin_value_changed)
            
        if hasattr(self.ui, 'gisaxsInputVmaxValue'):
            self.ui.gisaxsInputVmaxValue.valueChanged.connect(self._on_vmax_value_changed)
            
        # 设置GraphicsView双击事件
        if hasattr(self.ui, 'gisaxsInputGraphicsView'):
            self.ui.gisaxsInputGraphicsView.mouseDoubleClickEvent = self._on_graphics_view_double_click
            
        # 连接拟合相关按钮（如果UI中存在的话）
        if hasattr(self.ui, 'fitStartButton'):
            self.ui.fitStartButton.clicked.connect(self._start_fitting)
            
        if hasattr(self.ui, 'fitResetButton'):
            self.ui.fitResetButton.clicked.connect(self._reset_fitting)
            
        # 连接参数输入框的信号（如果存在的话）
        self._connect_parameter_widgets()
        
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
            
        # 设置默认参数
        self._set_default_parameters()
        
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
    
    def _show_image(self):
        """显示图像"""
        try:
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
            
            # 显示图像，使用计算出的vmin/vmax
            vmin = self._current_vmin if self._current_vmin is not None else np.min(processed_data)
            vmax = self._current_vmax if self._current_vmax is not None else np.max(processed_data)
            
            im = ax.imshow(processed_data, cmap='viridis', aspect='equal', origin='lower', 
                          interpolation='nearest', vmin=vmin, vmax=vmax)
            ax.axis('off')
            
            # 简化布局调整
            fig.tight_layout(pad=0.05)
            
            # 设置显示范围
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
            self.status_updated.emit(f"{mode_text} image displayed with color scale (Double-click to open independent window)")
            
        except Exception as e:
            self.status_updated.emit(f"Display error: {str(e)}")
    
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
            
            # 更新窗口中的图像（使用当前的vmin/vmax和log模式）
            if self.current_stack_data is not None:
                is_log = self._is_log_mode_enabled()
                self.independent_window.update_image(self.current_stack_data, 
                                                   self._current_vmin, self._current_vmax, 
                                                   use_log=is_log)
            
            # 显示窗口并置于前台
            self.independent_window.show()
            self.independent_window.raise_()
            self.independent_window.activateWindow()
            
            self.status_updated.emit("Independent matplotlib window opened")
            
        except Exception as e:
            self.status_updated.emit(f"Independent window error: {str(e)}")
    
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
        self._initialize_ui()
        self.status_updated.emit("已重置Cut Fitting参数")
        self.parameters_changed.emit(self.current_parameters)
