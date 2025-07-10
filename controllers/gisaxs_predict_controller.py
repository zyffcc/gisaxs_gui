"""
GISAXS Predict 控制器 - 处理GISAXS数据的预测功能
"""

from PyQt5.QtCore import QObject, pyqtSignal, Qt, QThread, QTimer
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGraphicsScene, QVBoxLayout, QWidget, QMainWindow
import os
import numpy as np
import hashlib
import time
import json
import traceback

# 尝试导入所需的库
try:
    import fabio
    FABIO_AVAILABLE = True
except ImportError:
    FABIO_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    import matplotlib
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
            else:
                # 图像形状未改变，保持当前视图
                pass
            
            # 保存轴限制 - 备份当前视图状态
            saved_xlim = self.current_xlim
            saved_ylim = self.current_ylim
            preserve_view = (not shape_changed and saved_xlim is not None and saved_ylim is not None)
            
            # 暂时断开视图变化回调，避免在更新过程中触发
            try:
                # 获取当前连接的回调ID
                xlim_cid = None
                ylim_cid = None
                
                # 先尝试新的方式（matplotlib 3.3+）
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
                    # 如果新方式失败，使用旧方式（matplotlib < 3.3）
                    try:
                        self.ax.callbacks.disconnect('xlim_changed', self._on_xlim_changed)
                        self.ax.callbacks.disconnect('ylim_changed', self._on_ylim_changed)
                    except TypeError:
                        # 如果都失败，则不断开连接，继续执行
                        pass
                        
            except Exception as e:
                # 回调断开失败，继续执行
                pass
            
            # 安全地移除旧的colorbar
            if self.colorbar is not None:
                try:
                    self.colorbar.remove()
                except Exception as e:
                    pass
                finally:
                    self.colorbar = None
            
            # 清除axes
            self.ax.clear()
            
            # 根据模式处理图像数据
            
            if use_log:
                # 使用更快的log计算方法，避免警告
                safe_data = np.where(image_data > 0, image_data, 0.001)
                processed_data = np.log(safe_data, dtype=np.float32)
                scale_text = "Log Scale"
                colorbar_label = "Log Intensity"
            else:
                # 线性模式
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
            except Exception as e:
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
            except Exception as e:
                # 回调连接失败，继续执行
                pass
            
            # 刷新画布
            self.canvas.draw()
            
            # 最终验证：确保视图没有被canvas刷新影响（某些matplotlib版本可能会有这个问题）
            if preserve_view:
                # 使用QTimer延迟执行，确保canvas完全绘制完成后再最终检查
                from PyQt5.QtCore import QTimer
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
                        self.canvas.draw_idle()  # 使用draw_idle避免递归
                    else:
                        # 视图正确保持，无需调整
                        pass
                
                QTimer.singleShot(50, final_view_check)  # 50ms后检查
            
            
        except Exception as e:
            import traceback
            # 错误记录已简化
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        
        # 清理colorbar
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception as e:
                pass
            finally:
                self.colorbar = None
        
        # 清理图形
        try:
            self.figure.clear()
        except Exception as e:
            pass
        
        super().closeEvent(event)


class GisaxsPredictController(QObject):
    """GISAXS预测控制器，处理GISAXS数据的预测"""
    
    # 状态信号
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    parameters_changed = pyqtSignal(dict)
    prediction_completed = pyqtSignal(dict)
    
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.parent = parent
        # 获取主窗口引用
        self.main_window = parent.parent if hasattr(parent, 'parent') else None
        
        # 当前参数
        self.current_parameters = {}
        
        # 预测结果
        self.prediction_results = {}
        
        # Stack处理相关
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
        self._pre_initialize_display_resources()  # 预初始化显示资源
        self._initialized = True
        
    def _setup_connections(self):
        """设置信号连接"""
        # 连接选择文件夹按钮
        if hasattr(self.ui, 'gisaxsPredictChooseFolderButton'):
            self.ui.gisaxsPredictChooseFolderButton.clicked.connect(self._choose_gisaxs_folder)
            
        # 连接选择GISAXS文件按钮
        if hasattr(self.ui, 'gisaxsPredictChooseGisaxsFileButton'):
            self.ui.gisaxsPredictChooseGisaxsFileButton.clicked.connect(self._choose_gisaxs_file)
            
        # 连接导入GISAXS文件按钮
        if hasattr(self.ui, 'gisaxsInputImportButton'):
            self.ui.gisaxsInputImportButton.clicked.connect(self._import_gisaxs_file)
            
        # 连接导入文件输入框的回车事件
        if hasattr(self.ui, 'gisaxsInputImportButtonValue'):
            self.ui.gisaxsInputImportButtonValue.returnPressed.connect(self._on_import_value_changed)
            
        # 连接导出文件夹按钮
        if hasattr(self.ui, 'gisaxsPredictExportFolderButton'):
            self.ui.gisaxsPredictExportFolderButton.clicked.connect(self._choose_export_folder)
            
        # 连接预测按钮
        if hasattr(self.ui, 'gisaxsPredictPredictButton'):
            self.ui.gisaxsPredictPredictButton.clicked.connect(self._run_gisaxs_predict)
            
        # 连接模式切换
        if hasattr(self.ui, 'gisaxsPredictSingleFileRadioButton'):
            self.ui.gisaxsPredictSingleFileRadioButton.toggled.connect(self._on_mode_changed)
            
        if hasattr(self.ui, 'gisaxsPredictMultiFilesRadioButton'):
            self.ui.gisaxsPredictMultiFilesRadioButton.toggled.connect(self._on_mode_changed)
            
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
            
    def _initialize_ui(self):
        """初始化界面状态"""
        # 设置默认框架
        if hasattr(self.ui, 'gisaxsPredictFrameworkCombox'):
            self.ui.gisaxsPredictFrameworkCombox.setCurrentIndex(0)
        
        # 设置默认为单文件模式
        if hasattr(self.ui, 'gisaxsPredictSingleFileRadioButton'):
            self.ui.gisaxsPredictSingleFileRadioButton.setChecked(True)
        
        # 清空输入框
        if hasattr(self.ui, 'gisaxsPredictExportFolderValue'):
            self.ui.gisaxsPredictExportFolderValue.clear()
            
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
            self.ui.gisaxsInputVminValue.setDecimals(6)  # 内部支持6位小数精度
            self.ui.gisaxsInputVminValue.setRange(-99999.999999, 99999.999999)
            self.ui.gisaxsInputVminValue.setSingleStep(0.1)  # 合理的步长
            self.ui.gisaxsInputVminValue.setKeyboardTracking(True)
            # 设置显示格式：自动隐藏尾随零
            self._setup_smart_display(self.ui.gisaxsInputVminValue)
            
        if hasattr(self.ui, 'gisaxsInputVmaxValue'):
            self.ui.gisaxsInputVmaxValue.setValue(0.0)
            # 设置智能精度：支持小数但默认显示简洁
            self.ui.gisaxsInputVmaxValue.setDecimals(6)  # 内部支持6位小数精度
            self.ui.gisaxsInputVmaxValue.setRange(-99999.999999, 99999.999999)
            self.ui.gisaxsInputVmaxValue.setSingleStep(0.1)  # 合理的步长
            self.ui.gisaxsInputVmaxValue.setKeyboardTracking(True)
            # 设置显示格式：自动隐藏尾随零
            self._setup_smart_display(self.ui.gisaxsInputVmaxValue)
            
        # 设置默认参数
        self._set_default_parameters()
        
        # 检查依赖库
        self._check_dependencies()
        
    def _setup_smart_display(self, spinbox):
        """设置SpinBox的智能显示格式
        
        Args:
            spinbox: QDoubleSpinBox控件
        """
        try:
            # 连接值变化信号到格式更新方法
            spinbox.valueChanged.connect(lambda value: self._update_spinbox_format(spinbox, value))
            
            # 连接编辑完成信号，确保在用户输入后更新格式
            spinbox.editingFinished.connect(lambda: self._update_spinbox_format(spinbox, spinbox.value()))
            
            # 初始化显示格式
            self._update_spinbox_format(spinbox, spinbox.value())
            
            
        except Exception as e:
            # 出错时使用默认格式
            spinbox.setDecimals(2)
        
    def _update_spinbox_format(self, spinbox, value):
        """根据当前模式和值更新SpinBox的显示格式
        
        Args:
            spinbox: QDoubleSpinBox控件
            value: 当前值
        """
        try:
            # 获取当前是否为Log模式
            is_log_mode = False
            if hasattr(self.ui, 'gisaxsInputIntLogCheckBox'):
                is_log_mode = self.ui.gisaxsInputIntLogCheckBox.isChecked()
            
            if is_log_mode:
                # Log模式：固定显示2位小数
                spinbox.setDecimals(2)
            else:
                # 线性模式：智能显示格式
                if abs(value - round(value)) < 1e-9:
                    # 值为整数时，显示为整数
                    spinbox.setDecimals(0)
                else:
                    # 值为小数时，根据实际精度显示（最多6位）
                    value_str = f"{value:.6f}".rstrip('0').rstrip('.')
                    if '.' in value_str:
                        decimal_places = len(value_str.split('.')[1])
                        decimal_places = min(decimal_places, 6)  # 最多6位
                        decimal_places = max(decimal_places, 1)  # 至少1位
                        spinbox.setDecimals(decimal_places)
                    else:
                        spinbox.setDecimals(0)
                        
        except Exception as e:
            # 出错时使用默认格式
            try:
                if hasattr(self.ui, 'gisaxsInputIntLogCheckBox') and self.ui.gisaxsInputIntLogCheckBox.isChecked():
                    spinbox.setDecimals(2)
                else:
                    spinbox.setDecimals(0)
            except:
                spinbox.setDecimals(2)  # 最后的回退方案
    
    def _refresh_vmin_vmax_display(self):
        """刷新Vmin/Vmax控件的显示格式"""
        try:
            if hasattr(self.ui, 'gisaxsInputVminValue'):
                self._update_spinbox_format(self.ui.gisaxsInputVminValue, self.ui.gisaxsInputVminValue.value())
            if hasattr(self.ui, 'gisaxsInputVmaxValue'):
                self._update_spinbox_format(self.ui.gisaxsInputVmaxValue, self.ui.gisaxsInputVmaxValue.value())
        except Exception as e:
            # 刷新显示格式失败，使用默认格式
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
            'framework': 'tensorflow 2.15.0',
            'mode': 'single_file',
            'input_file': '',
            'input_folder': '',
            'imported_gisaxs_file': '',  # 新增：导入的GISAXS文件
            'stack_count': 1,  # 新增：叠加数量
            'export_path': '',
            'batch_size': 32,
            'model_path': ''
        }
        
    def _choose_gisaxs_folder(self):
        """选择GISAXS文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self.main_window,
            "选择GISAXS文件夹",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder_path:
            self.current_parameters['input_folder'] = folder_path
            self.status_updated.emit(f"已选择GISAXS文件夹: {folder_path}")
            self.parameters_changed.emit(self.current_parameters)
    
    def _choose_gisaxs_file(self):
        """选择GISAXS文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "选择GISAXS文件",
            "",
            "GISAXS Files (*.tif *.tiff *.dat *.txt *.h5 *.hdf5);;All Files (*)"
        )
        
        if file_path:
            self.current_parameters['input_file'] = file_path
            self.status_updated.emit(f"已选择GISAXS文件: {file_path}")
            self.parameters_changed.emit(self.current_parameters)

    def _import_gisaxs_file(self):
        """导入GISAXS文件 - 为gisaxsInputImportButton按钮提供功能"""
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
                # 只显示文件名，而不是完整路径
                file_name = os.path.basename(file_path)
                self.ui.gisaxsInputImportButtonValue.setText(file_name)
                
            # 发送状态更新信号
            self.status_updated.emit(f"已导入GISAXS文件: {os.path.basename(file_path)}")
            self.parameters_changed.emit(self.current_parameters)
            
            # 可以在这里添加文件验证逻辑
            self._validate_imported_file(file_path)
            
            # 更新显示信息
            self._update_stack_display()
            
            # 如果AutoShow被选中，则自动显示
            if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked():
                self._show_image()
    
    def _validate_imported_file(self, file_path):
        """验证导入的GISAXS文件"""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                QMessageBox.warning(
                    self.main_window,
                    "文件错误",
                    f"文件不存在: {file_path}"
                )
                return False
            
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                QMessageBox.warning(
                    self.main_window,
                    "文件错误",
                    "文件为空"
                )
                return False
            
            # 检查文件扩展名
            file_ext = os.path.splitext(file_path)[1].lower()
            supported_extensions = ['.tif', '.tiff', '.dat', '.txt', '.h5', '.hdf5', '.jpg', '.png', '.bmp' , '.cbf']
            
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
            
            # 显示文件信息
            file_info = f"文件: {os.path.basename(file_path)}\n"
            file_info += f"大小: {file_size / 1024:.1f} KB\n"
            file_info += f"格式: {file_ext}\n"
            file_info += f"路径: {file_path}"
            
            self.status_updated.emit(f"文件验证通过 - {os.path.basename(file_path)}")
            
            # 可以在这里添加更多的文件内容验证
            # 例如：检查图像尺寸、数据格式等
            
            return True
            
        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "文件验证错误",
                f"验证文件时出错:\n{str(e)}"
            )
            return False
    
    def get_imported_file(self):
        """获取导入的GISAXS文件路径"""
        return self.current_parameters.get('imported_gisaxs_file', '')
    
    def _on_import_value_changed(self):
        """当Import Value输入框内容改变且按回车时的处理"""
        try:
            # 获取输入框中的文件路径
            if not hasattr(self.ui, 'gisaxsInputImportButtonValue'):
                return
                
            file_path_input = self.ui.gisaxsInputImportButtonValue.text().strip()
            
            if not file_path_input:
                self.status_updated.emit("请输入有效的文件路径")
                return
            
            # 如果输入的只是文件名，尝试使用当前目录或之前的目录
            if not os.path.isabs(file_path_input):
                # 尝试使用当前导入文件的目录
                current_file = self.current_parameters.get('imported_gisaxs_file', '')
                if current_file and os.path.exists(current_file):
                    current_dir = os.path.dirname(current_file)
                    file_path_input = os.path.join(current_dir, file_path_input)
                else:
                    # 使用当前工作目录
                    file_path_input = os.path.abspath(file_path_input)
            
            # 验证文件是否存在
            if not os.path.exists(file_path_input):
                self.status_updated.emit(f"文件不存在: {os.path.basename(file_path_input)}")
                QMessageBox.warning(
                    self.main_window,
                    "文件错误",
                    f"文件不存在:\n{file_path_input}"
                )
                return
            
            # 更新参数
            self.current_parameters['imported_gisaxs_file'] = file_path_input
            
            # 更新UI显示为文件名
            file_name = os.path.basename(file_path_input)
            self.ui.gisaxsInputImportButtonValue.setText(file_name)
            
            # 验证文件
            if self._validate_imported_file(file_path_input):
                # 发送状态更新信号
                self.status_updated.emit(f"已更新GISAXS文件: {file_name}")
                self.parameters_changed.emit(self.current_parameters)
                
                # 更新stack显示
                self._update_stack_display()
                
                # 刷新Vmin/Vmax显示格式
                self._refresh_vmin_vmax_display()
                
                # 如果AutoShow被选中，则自动显示图像
                if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked():
                    self._show_image()
                else:
                    self.status_updated.emit(f"文件已更新，点击Show按钮显示图像")
            
        except Exception as e:
            self.status_updated.emit(f"Import value processing error: {str(e)}")
            QMessageBox.critical(
                self.main_window,
                "处理错误",
                f"处理导入文件路径时出错:\n{str(e)}"
            )
    
    def _on_stack_value_changed(self):
        """当Stack值改变时的处理（回车触发）"""
        try:
            # 获取stack值
            stack_text = self.ui.gisaxsInputStackValue.text() if hasattr(self.ui, 'gisaxsInputStackValue') else "1"
            
            # 验证stack值
            try:
                stack_count = int(stack_text)
            except ValueError:
                # 如果不是有效数字，重置为1
                if hasattr(self.ui, 'gisaxsInputStackValue'):
                    self.ui.gisaxsInputStackValue.setText("1")
                stack_count = 1
            
            # 验证stack值范围
            if stack_count < 1:
                if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                    self.ui.gisaxsInputStackDisplayLabel.setText("At least 1")
                return
            
            # 更新参数
            self.current_parameters['stack_count'] = stack_count
            
            # 更新显示信息（不处理文件）
            self._update_stack_display()
            
            # 刷新Vmin/Vmax显示格式（确保格式正确）
            self._refresh_vmin_vmax_display()
            
            # 判断是否需要重新加载图像
            should_reload_image = False
            
            # 如果AutoShow被选中，总是重新加载
            if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked():
                should_reload_image = True
            
            # 如果当前有图像数据且是CBF文件，也考虑重新加载（因为Stack可能影响叠加结果）
            elif self.current_stack_data is not None:
                imported_file = self.current_parameters.get('imported_gisaxs_file', '')
                if imported_file and os.path.splitext(imported_file)[1].lower() == '.cbf':
                    should_reload_image = True
            
            # 执行图像重新加载
            if should_reload_image:
                self._show_image()
            else:
                self.status_updated.emit(f"Stack count updated to {stack_count}")
            
        except Exception as e:
            self.status_updated.emit(f"Stack value processing error: {str(e)}")
    
    def _update_stack_display(self):
        """更新stack显示信息，不处理文件"""
        try:
            imported_file = self.current_parameters.get('imported_gisaxs_file', '')
            if not imported_file:
                return
            
            file_ext = os.path.splitext(imported_file)[1].lower()
            stack_count = self.current_parameters.get('stack_count', 1)
            
            if file_ext != '.cbf':
                # 非CBF文件
                if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                    self.ui.gisaxsInputStackDisplayLabel.setText(f"Single File: {os.path.basename(imported_file)}")
                return
            
            if stack_count == 1:
                # 单文件模式
                if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                    self.ui.gisaxsInputStackDisplayLabel.setText(f"Single File: {os.path.basename(imported_file)}")
            else:
                # 检查可用文件数量
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
                        # 显示将要叠加的文件范围
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
    
    def _show_image(self):
        """显示图像"""
        try:
            imported_file = self.current_parameters.get('imported_gisaxs_file', '')
            if not imported_file:
                self.status_updated.emit("No file imported to show")
                return
            
            # 检查依赖库
            if not FABIO_AVAILABLE:
                QMessageBox.warning(
                    self.main_window,
                    "Missing Library",
                    "fabio library is required for CBF file processing.\nPlease install it using: pip install fabio"
                )
                return
                
            if not MATPLOTLIB_AVAILABLE:
                QMessageBox.warning(
                    self.main_window,
                    "Missing Library", 
                    "matplotlib library is required for image display.\nPlease install it using: pip install matplotlib"
                )
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
            
            # 显示图像
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
        QMessageBox.critical(
            self.main_window,
            "图像加载错误",
            error_message
        )
    
    def _calculate_vmin_vmax(self, image_data, use_log=True):
        """计算图像的Vmin和Vmax值（1%和99%分位数）"""
        try:
            if use_log:
                # 对于log显示，先计算log值再取分位数
                safe_data = np.where(image_data > 0, image_data, 0.001)
                log_data = np.log(safe_data)
                vmin = np.percentile(log_data, 1)
                vmax = np.percentile(log_data, 99)
            else:
                # 对于线性显示，直接取原始数据的分位数
                vmin = np.percentile(image_data, 1)
                vmax = np.percentile(image_data, 99)
            
            return vmin, vmax
            
        except Exception as e:
            return None, None
    
    def _update_vmin_vmax_ui(self, vmin, vmax):
        """更新UI中的Vmin和Vmax值"""
        try:
            if vmin is not None and vmax is not None:
                if hasattr(self.ui, 'gisaxsInputVminValue'):
                    self.ui.gisaxsInputVminValue.setValue(float(vmin))
                    
                if hasattr(self.ui, 'gisaxsInputVmaxValue'):
                    self.ui.gisaxsInputVmaxValue.setValue(float(vmax))
                    
                # 更新内部状态
                self._current_vmin = vmin
                self._current_vmax = vmax
                
                # 刷新显示格式（确保新值按照当前模式正确显示）
                self._refresh_vmin_vmax_display()
                
        except Exception as e:
            # 更新UI值失败，使用默认处理
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
            
        except Exception as e:
            return None, None
    
    def _is_auto_scale_enabled(self):
        """检查AutoScale是否被启用"""
        try:
            if hasattr(self.ui, 'gisaxsInputAutoScaleCheckBox'):
                return self.ui.gisaxsInputAutoScaleCheckBox.isChecked()
            return True  # 默认启用
        except Exception as e:
            return True
    
    def _on_auto_scale_changed(self):
        """AutoScale复选框状态改变时的处理"""
        try:
            is_enabled = self._is_auto_scale_enabled()
            self.status_updated.emit(f"AutoScale {'enabled' if is_enabled else 'disabled'}")
            
            # 如果当前有图像数据且AutoScale被启用，重新计算并更新Vmin/Vmax
            if is_enabled and self.current_stack_data is not None:
                is_log = self._is_log_mode_enabled()
                vmin, vmax = self._calculate_vmin_vmax(self.current_stack_data, use_log=is_log)
                if vmin is not None and vmax is not None:
                    self._update_vmin_vmax_ui(vmin, vmax)
                    # 刷新显示图像以应用新的颜色范围
                    self._refresh_image_display()
                    
        except Exception as e:
            self.status_updated.emit(f"AutoScale change error: {str(e)}")
    
    def _on_vmin_value_changed(self):
        """Vmin值改变时的处理"""
        try:
            vmin, vmax = self._get_vmin_vmax_from_ui()
            if vmin is not None:
                self._current_vmin = vmin
                
                # 实时更新图像显示
                if self.current_stack_data is not None:
                    self._refresh_image_display()
                
        except Exception as e:
            # Vmin值变化处理失败
            pass
    
    def _on_vmax_value_changed(self):
        """Vmax值改变时的处理"""
        try:
            vmin, vmax = self._get_vmin_vmax_from_ui()
            if vmax is not None:
                self._current_vmax = vmax
                
                # 实时更新图像显示
                if self.current_stack_data is not None:
                    self._refresh_image_display()
                
        except Exception as e:
            # Vmax值变化处理失败
            pass
        
    def _load_single_cbf_file(self, cbf_file):
        """加载单个CBF文件 - 优化版本"""
        try:
            
            # 优化：直接以float32读取，减少转换开销
            cbf_image = fabio.open(cbf_file)
            data = cbf_image.data
            
            # 如果数据不是float32，进行高效转换
            if data.dtype != np.float32:
                # 使用copy=False可能的话，减少内存分配
                data = data.astype(np.float32, copy=False)
            
            return data
            
        except Exception as e:
            self.status_updated.emit(f"Error loading single CBF file: {str(e)}")
            return None
    
    def _load_multiple_cbf_files(self, start_file, stack_count):
        """加载并叠加多个CBF文件 - 优化版本"""
        try:
            file_dir = os.path.dirname(start_file)
            base_name = os.path.basename(start_file)
            
            # 获取文件列表（优化：只获取一次）
            cbf_files = [f for f in os.listdir(file_dir) if f.lower().endswith('.cbf')]
            cbf_files.sort()
            
            # 找到起始文件索引
            try:
                start_index = cbf_files.index(base_name)
            except ValueError:
                self.status_updated.emit(f"Start file not found: {base_name}")
                return None
            
            # 检查可用文件数量
            available_files = len(cbf_files) - start_index
            if stack_count > available_files:
                self.status_updated.emit(f"Requested {stack_count} files, only {available_files} available")
                return None
            
            # 叠加文件 - 优化内存使用和性能
            summed_data = None
            files_to_stack = cbf_files[start_index:start_index + stack_count]
            
            for i, file_name in enumerate(files_to_stack):
                file_path = os.path.join(file_dir, file_name)
                try:
                    cbf_image = fabio.open(file_path)
                    
                    # 优化：直接转换为float32，减少中间步骤
                    data = cbf_image.data.astype(np.float32, copy=False) if cbf_image.data.dtype != np.float32 else cbf_image.data
                    
                    if summed_data is None:
                        # 第一个文件，直接复制
                        summed_data = data.copy() if data.dtype == np.float32 else data.astype(np.float32)
                    else:
                        # 后续文件，直接累加（numpy会自动优化）
                        summed_data += data
                        
                except Exception as e:
                    self.status_updated.emit(f"Error processing file {file_name}: {str(e)}")
                    continue
            
            return summed_data
            
        except Exception as e:
            self.status_updated.emit(f"Error loading multiple CBF files: {str(e)}")
            return None
    
    def _display_image(self, image_data):
        """显示图像数据 - 同时更新内嵌和独立窗口，并处理颜色标尺"""
        try:
            
            # 存储当前数据
            self.current_stack_data = image_data
            
            # 处理颜色标尺逻辑
            self._handle_color_scale(image_data)
            
            # 更新内嵌的GraphicsView
            if hasattr(self.ui, 'gisaxsInputGraphicsView'):
                self._update_graphics_view(image_data)
            
            # 如果独立窗口已打开，同时更新独立窗口并使用当前的vmin/vmax
            if self.independent_window is not None and self.independent_window.isVisible():
                is_log = self._is_log_mode_enabled()
                self.independent_window.update_image(image_data, self._current_vmin, self._current_vmax, use_log=is_log)
            
            # 简单状态信息
            window_status = " (+ Independent window)" if (self.independent_window and self.independent_window.isVisible()) else ""
            vmin_vmax_info = f" [Vmin: {self._current_vmin:.3f}, Vmax: {self._current_vmax:.3f}]" if self._current_vmin is not None and self._current_vmax is not None else ""
            mode_text = "Log" if self._is_log_mode_enabled() else "Linear"
            self.status_updated.emit(f"{mode_text} image displayed: {image_data.shape}{vmin_vmax_info}{window_status}")
            
        except Exception as e:
            self.status_updated.emit(f"Display error: {str(e)}")
    
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
                
        except Exception as e:
            # 如果出错，回退到自动计算
            try:
                is_log = self._is_log_mode_enabled()
                vmin, vmax = self._calculate_vmin_vmax(image_data, use_log=is_log)
                if vmin is not None and vmax is not None:
                    self._update_vmin_vmax_ui(vmin, vmax)
            except Exception as e2:
                # 回退计算也失败
                pass
    
    def _update_graphics_view(self, image_data):
        """更新GraphicsView中的图像显示 - 优化版本，支持颜色标尺"""
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
            base_size = 6  # 减小基础尺寸以提高性能
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
            import traceback
            # 错误记录已简化
    
    def _on_auto_show_changed(self):
        """AutoShow选项改变时的处理"""
        auto_show = hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked()
        self.status_updated.emit(f"AutoShow {'enabled' if auto_show else 'disabled'}")
    
    def _on_log_changed(self):
        """Log选项改变时的处理 - 支持线性/对数切换"""
        try:
            is_log = hasattr(self.ui, 'gisaxsInputIntLogCheckBox') and self.ui.gisaxsInputIntLogCheckBox.isChecked()
            
            # 刷新Vmin/Vmax控件的显示格式（Log模式下显示2位小数，线性模式下智能显示）
            self._refresh_vmin_vmax_display()
            
            # 如果当前有图像数据，重新计算vmin/vmax并更新显示
            if self.current_stack_data is not None:
                
                # 重新计算颜色范围（根据当前模式）
                if self._is_auto_scale_enabled():
                    vmin, vmax = self._calculate_vmin_vmax(self.current_stack_data, use_log=is_log)
                    if vmin is not None and vmax is not None:
                        self._update_vmin_vmax_ui(vmin, vmax)
                
                # 重新显示图像
                self._refresh_image_display()
                
            self.status_updated.emit(f"*** DISPLAY MODE CHANGED TO: {'LOG' if is_log else 'LINEAR'} ***")
                
        except Exception as e:
            self.status_updated.emit(f"Log mode change error: {str(e)}")
    

    
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
    
    def _is_log_mode_enabled(self):
        """检查是否启用Log模式"""
        try:
            if hasattr(self.ui, 'gisaxsInputIntLogCheckBox'):
                return self.ui.gisaxsInputIntLogCheckBox.isChecked()
            return True  # 默认启用Log模式
        except Exception as e:
            return True
    
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
            
            # 验证数据是否确实不同
            if is_log:
                linear_test = image_data.astype(np.float32)
                are_different = not np.allclose(processed_data, linear_test, rtol=1e-3)
            else:
                log_test = np.log(np.where(image_data > 0, image_data, 0.001), dtype=np.float32)
                are_different = not np.allclose(processed_data, log_test, rtol=1e-3)
            
            return processed_data, is_log
            
        except Exception as e:
            import traceback
            # 出错时返回原始数据和Log模式
            return image_data.astype(np.float32), True

    def _on_graphics_view_double_click(self, event):
        """GraphicsView双击事件处理"""
        try:
            
            if not MATPLOTLIB_AVAILABLE:
                QMessageBox.warning(
                    self.main_window,
                    "Missing Library",
                    "matplotlib library is required for independent window.\nPlease install it using: pip install matplotlib"
                )
                return
            
            # 如果没有当前图像数据，提示用户
            if self.current_stack_data is None:
                QMessageBox.information(
                    self.main_window,
                    "No Image",
                    "Please import and display an image first."
                )
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
                self.independent_window.update_image(self.current_stack_data, self._current_vmin, self._current_vmax, use_log=is_log)
            
            # 显示窗口并置于前台
            self.independent_window.show()
            self.independent_window.raise_()
            self.independent_window.activateWindow()
            
            self.status_updated.emit("Independent matplotlib window opened")
            
        except Exception as e:
            self.status_updated.emit(f"Independent window error: {str(e)}")
            import traceback
            # 错误记录已简化
    
    def _choose_export_folder(self):
        """选择导出文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self.main_window,
            "选择导出文件夹",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder_path:
            self.current_parameters['export_path'] = folder_path
            
            # 更新UI显示
            if hasattr(self.ui, 'gisaxsPredictExportFolderValue'):
                self.ui.gisaxsPredictExportFolderValue.setText(folder_path)
                
            self.status_updated.emit(f"已选择导出文件夹: {folder_path}")
            self.parameters_changed.emit(self.current_parameters)
            
    def _on_mode_changed(self):
        """模式切换时的处理"""
        if hasattr(self.ui, 'gisaxsPredictSingleFileRadioButton') and self.ui.gisaxsPredictSingleFileRadioButton.isChecked():
            self.current_parameters['mode'] = 'single_file'
            self.status_updated.emit("切换到单文件预测模式")
        elif hasattr(self.ui, 'gisaxsPredictMultiFilesRadioButton') and self.ui.gisaxsPredictMultiFilesRadioButton.isChecked():
            self.current_parameters['mode'] = 'multi_files'
            self.status_updated.emit("切换到多文件预测模式")
            
        self.parameters_changed.emit(self.current_parameters)
        
    def _run_gisaxs_predict(self):
        """运行GISAXS预测"""
        # 获取当前参数
        self._update_parameters_from_ui()
        
        # 验证参数
        if not self._validate_parameters():
            return
            
        try:
            self.status_updated.emit("开始GISAXS预测...")
            self.progress_updated.emit(0)
            
            # 更新状态显示
            self._update_status_display()
            
            # 执行预测
            if self.current_parameters['mode'] == 'single_file':
                results = self._predict_single_file()
            else:
                results = self._predict_multi_files()
                
            # 保存结果
            if self.current_parameters.get('export_path'):
                self._save_results(results)
                
            self.prediction_results = results
            self.progress_updated.emit(100)
            self.status_updated.emit("GISAXS预测完成！")
            self.prediction_completed.emit(results)
            
        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "错误",
                f"GISAXS预测过程中出错:\n{str(e)}"
            )
            
    def _update_parameters_from_ui(self):
        """从UI更新参数"""
        # 获取框架选择
        if hasattr(self.ui, 'gisaxsPredictFrameworkCombox'):
            self.current_parameters['framework'] = self.ui.gisaxsPredictFrameworkCombox.currentText()
        
        # 获取导出路径
        if hasattr(self.ui, 'gisaxsPredictExportFolderValue'):
            self.current_parameters['export_path'] = self.ui.gisaxsPredictExportFolderValue.text()
            
    def _validate_parameters(self):
        """验证参数"""
        if not self.current_parameters.get('export_path'):
            QMessageBox.warning(
                self.main_window,
                "参数错误",
                "请选择导出文件夹!"
            )
            return False
        
        # 检查导出路径是否存在
        if not os.path.exists(self.current_parameters['export_path']):
            QMessageBox.warning(
                self.main_window,
                "路径错误",
                f"导出文件夹不存在: {self.current_parameters['export_path']}"
            )
            return False
            
        # 检查输入文件/文件夹
        if self.current_parameters['mode'] == 'single_file':
            if not self.current_parameters.get('input_file'):
                QMessageBox.warning(self.main_window, "参数错误", "请选择输入文件!")
                return False
        else:
            if not self.current_parameters.get('input_folder'):
                QMessageBox.warning(self.main_window, "参数错误", "请选择输入文件夹!")
                return False
        
        return True
        
    def _update_status_display(self):
        """更新状态显示"""
        if hasattr(self.ui, 'predictStatusTextBrowser'):
            self.ui.predictStatusTextBrowser.append("=== GISAXS预测开始 ===")
            self.ui.predictStatusTextBrowser.append(f"框架: {self.current_parameters['framework']}")
            self.ui.predictStatusTextBrowser.append(f"模式: {self.current_parameters['mode']}")
            self.ui.predictStatusTextBrowser.append(f"导出路径: {self.current_parameters['export_path']}")
            self.ui.predictStatusTextBrowser.append("注意: 物理处理功能尚未实现，这里仅做UI演示")
            
    def _predict_single_file(self):
        """单文件预测"""
        # TODO: 实现实际的单文件预测逻辑
        self.progress_updated.emit(50)
        
        result = {
            'mode': 'single_file',
            'input_file': self.current_parameters['input_file'],
            'predicted_parameters': {
                'particle_size': 25.3,  # 示例结果
                'particle_shape': 'sphere',
                'lattice_constant': 5.64
            },
            'confidence': 0.92,
            'processing_time': 1.5
        }
        
        return result
        
    def _predict_multi_files(self):
        """多文件预测"""
        # TODO: 实现实际的多文件预测逻辑
        self.progress_updated.emit(50)
        
        result = {
            'mode': 'multi_files',
            'input_folder': self.current_parameters['input_folder'],
            'processed_files': 5,  # 示例结果
            'average_confidence': 0.87,
            'processing_time': 8.2
        }
        
        return result
        
    def _save_results(self, results):
        """保存预测结果"""
        try:
            import json
            output_path = os.path.join(
                self.current_parameters['export_path'],
                'gisaxs_prediction_results.json'
            )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            self.status_updated.emit(f"预测结果已保存到: {output_path}")
            
        except Exception as e:
            self.status_updated.emit(f"保存结果时出错: {str(e)}")
            
    def get_parameters(self):
        """获取当前参数"""
        return self.current_parameters.copy()
        
    def set_parameters(self, parameters):
        """设置参数"""
        self.current_parameters.update(parameters)
        self.parameters_changed.emit(self.current_parameters)
        
    def get_results(self):
        """获取预测结果"""
        return self.prediction_results.copy()
        
    def reset_to_defaults(self):
        """重置参数到默认值"""
        self.current_parameters = {}
        self.prediction_results = {}
        
        # 重置UI到默认状态
        self._initialize_ui()
        
        self.status_updated.emit("GISAXS预测参数已重置为默认值")
        self.parameters_changed.emit(self.current_parameters)
    
    def _pre_initialize_display_resources(self):
        """预初始化显示资源，提高后续图像显示的流畅性"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                return
                
            
            # 预创建GraphicsScene
            if hasattr(self.ui, 'gisaxsInputGraphicsView'):
                self._graphics_scene = QGraphicsScene()
                self.ui.gisaxsInputGraphicsView.setScene(self._graphics_scene)
            
            # 预创建一个基础的Figure和Canvas模板（减少首次创建开销）
            base_fig = Figure(figsize=(6, 6), dpi=72)
            base_canvas = FigureCanvas(base_fig)
            base_ax = base_fig.add_subplot(111)
            
            # 创建一个简单的测试图像来"热身"matplotlib
            test_data = np.ones((10, 10), dtype=np.float32)
            base_ax.imshow(test_data, cmap='viridis', aspect='equal', origin='lower')
            base_fig.tight_layout()
            base_canvas.draw()
            
            # 清理测试资源
            base_fig.clear()
            del base_fig, base_canvas, base_ax, test_data
            
        except Exception as e:
            # 预初始化失败不影响主要功能
            pass


class AsyncImageLoader(QThread):
    """异步图像加载线程，避免UI阻塞"""
    
    # 信号
    image_loaded = pyqtSignal(np.ndarray, str)  # 图像数据, 文件路径
    progress_updated = pyqtSignal(int, str)  # 进度, 状态信息
    error_occurred = pyqtSignal(str)  # 错误信息
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.file_path = None
        self.stack_count = 1
        self.should_cancel = False
    
    def load_image(self, file_path, stack_count=1):
        """设置加载参数并启动线程"""
        self.file_path = file_path
        self.stack_count = stack_count
        self.should_cancel = False
        self.start()
    
    def cancel_loading(self):
        """取消加载"""
        self.should_cancel = True
    
    def run(self):
        """线程主函数"""
        try:
            if not self.file_path:
                return
                
            start_time = time.time()
            self.progress_updated.emit(10, "Starting image loading...")
            
            if self.should_cancel:
                return
                
            file_ext = os.path.splitext(self.file_path)[1].lower()
            if file_ext != '.cbf':
                self.error_occurred.emit("Only CBF files are supported")
                return
            
            if self.stack_count == 1:
                self.progress_updated.emit(30, "Loading single CBF file...")
                image_data = self._load_single_cbf_optimized(self.file_path)
            else:
                self.progress_updated.emit(30, f"Loading {self.stack_count} CBF files for stacking...")
                image_data = self._load_multiple_cbf_optimized(self.file_path, self.stack_count)
            
            if self.should_cancel:
                return
                
            if image_data is not None:
                load_time = time.time() - start_time
                self.progress_updated.emit(90, f"Image loaded in {load_time:.2f}s")
                self.image_loaded.emit(image_data, self.file_path)
                self.progress_updated.emit(100, "Loading completed")
            else:
                self.error_occurred.emit("Failed to load image data")
                
        except Exception as e:
            self.error_occurred.emit(f"Loading error: {str(e)}")
    
    def _load_single_cbf_optimized(self, cbf_file):
        """优化的单个CBF文件加载 - 性能增强版"""
        try:
            
            # 使用更高效的读取方式
            import fabio
            
            # 尝试直接读取数据，避免不必要的元数据处理
            try:
                # 方式1：使用fabio的快速读取
                cbf_image = fabio.open(cbf_file)
                data = cbf_image.data
                
                # 检查是否需要取消
                if self.should_cancel:
                    return None
                
                # 优化数据类型转换 - 使用视图而非复制（如果可能）
                if data.dtype == np.float32:
                    # 已经是float32，直接使用
                    result_data = data
                elif data.dtype in [np.int32, np.int16, np.uint32, np.uint16]:
                    # 整数类型，高效转换
                    result_data = data.astype(np.float32, copy=False)
                else:
                    # 其他类型，需要复制转换
                    result_data = data.astype(np.float32)
                
                return result_data
                
            except Exception as e1:
                
                # 方式2：如果标准方式失败，尝试更底层的读取
                try:
                    # 使用h5py或直接文件读取（如果是HDF5格式的CBF）
                    import h5py
                    with h5py.File(cbf_file, 'r') as f:
                        # 尝试找到数据集
                        for key in f.keys():
                            if isinstance(f[key], h5py.Dataset):
                                data = f[key][:]
                                if data.dtype != np.float32:
                                    data = data.astype(np.float32, copy=False)
                                return data
                except:
                    pass
                
                # 方式3：最后尝试标准方式
                cbf_image = fabio.open(cbf_file)
                data = cbf_image.data.astype(np.float32)
                return data
                
        except Exception as e:
            return None
    
    def _load_multiple_cbf_optimized(self, start_file, stack_count):
        """优化的多个CBF文件加载和叠加 - 性能增强版"""
        try:
            file_dir = os.path.dirname(start_file)
            base_name = os.path.basename(start_file)
            
            # 快速获取文件列表 - 优化：使用os.scandir更快
            try:
                cbf_files = []
                with os.scandir(file_dir) as entries:
                    for entry in entries:
                        if entry.is_file() and entry.name.lower().endswith('.cbf'):
                            cbf_files.append(entry.name)
                cbf_files.sort()
            except:
                # 回退到标准方式
                cbf_files = [f for f in os.listdir(file_dir) if f.lower().endswith('.cbf')]
                cbf_files.sort()
            
            try:
                start_index = cbf_files.index(base_name)
            except ValueError:
                return None
            
            available_files = len(cbf_files) - start_index
            if stack_count > available_files:
                return None
            
            files_to_stack = cbf_files[start_index:start_index + stack_count]
            summed_data = None
            
            import fabio
            
            # 预分配内存策略：先读取第一个文件确定尺寸
            first_file_path = os.path.join(file_dir, files_to_stack[0])
            try:
                first_data = self._load_single_cbf_optimized(first_file_path)
                if first_data is None or self.should_cancel:
                    return None
                
                # 预分配结果数组，避免多次内存分配
                summed_data = first_data.copy()
                self.progress_updated.emit(35, f"Processing file 1/{len(files_to_stack)}")
                
                # 处理剩余文件
                for i, file_name in enumerate(files_to_stack[1:], 1):
                    if self.should_cancel:
                        return None
                        
                    # 更新进度
                    progress = 35 + int((i / len(files_to_stack)) * 45)
                    self.progress_updated.emit(progress, f"Processing file {i+1}/{len(files_to_stack)}")
                    
                    file_path = os.path.join(file_dir, file_name)
                    try:
                        # 直接读取并累加，避免中间变量
                        cbf_image = fabio.open(file_path)
                        data = cbf_image.data
                        
                        # 高效累加：直接在目标数组上操作
                        if data.dtype == summed_data.dtype:
                            summed_data += data
                        else:
                            # 需要类型转换
                            if data.dtype != np.float32:
                                data = data.astype(np.float32, copy=False)
                            summed_data += data
                            
                    except Exception as e:
                        continue
                
            except Exception as e:
                # 回退到原始方法
                summed_data = None
                for i, file_name in enumerate(files_to_stack):
                    if self.should_cancel:
                        return None
                        
                    progress = 30 + int((i / len(files_to_stack)) * 50)
                    self.progress_updated.emit(progress, f"Processing file {i+1}/{len(files_to_stack)}")
                    
                    file_path = os.path.join(file_dir, file_name)
                    file_data = self._load_single_cbf_optimized(file_path)
                    
                    if file_data is not None:
                        if summed_data is None:
                            summed_data = file_data.copy()
                        else:
                            summed_data += file_data
            
            return summed_data
            
        except Exception as e:
            return None
