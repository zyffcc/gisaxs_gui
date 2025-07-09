"""
GISAXS Predict 控制器 - 处理GISAXS数据的预测功能
"""

from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGraphicsScene, QVBoxLayout, QWidget
import os
import numpy as np

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
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


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
        
        # 初始化标志
        self._initialized = False
        
    def initialize(self):
        """初始化控制器"""
        if self._initialized:
            return
            
        self._setup_connections()
        self._initialize_ui()
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
        if hasattr(self.ui, 'LogCheckBox'):
            self.ui.LogCheckBox.toggled.connect(self._on_log_changed)
            
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
        if hasattr(self.ui, 'LogCheckBox'):
            self.ui.LogCheckBox.setChecked(True)
            
        # 设置默认参数
        self._set_default_parameters()
        
        # 检查依赖库
        self._check_dependencies()
        
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
            
            # 如果AutoShow被选中，则自动显示
            if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked():
                self._show_image()
            
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
            
            if stack_count == 1:
                image_data = self._load_single_cbf_file(imported_file)
            else:
                image_data = self._load_multiple_cbf_files(imported_file, stack_count)
            
            if image_data is not None:
                self._display_image(image_data)
                
        except Exception as e:
            self.status_updated.emit(f"Show image error: {str(e)}")
    
    def _load_single_cbf_file(self, cbf_file):
        """加载单个CBF文件"""
        try:
            cbf_image = fabio.open(cbf_file)
            return cbf_image.data
        except Exception as e:
            self.status_updated.emit(f"Error loading single CBF file: {str(e)}")
            return None
    
    def _load_multiple_cbf_files(self, start_file, stack_count):
        """加载并叠加多个CBF文件"""
        try:
            file_dir = os.path.dirname(start_file)
            base_name = os.path.basename(start_file)
            
            # 获取文件列表
            cbf_files = []
            for file in os.listdir(file_dir):
                if file.lower().endswith('.cbf'):
                    cbf_files.append(file)
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
            
            # 叠加文件
            summed_data = None
            files_to_stack = cbf_files[start_index:start_index + stack_count]
            
            for file_name in files_to_stack:
                file_path = os.path.join(file_dir, file_name)
                try:
                    cbf_image = fabio.open(file_path)
                    data = cbf_image.data
                    
                    if summed_data is None:
                        summed_data = data.astype(np.float64)
                    else:
                        summed_data += data.astype(np.float64)
                        
                except Exception as e:
                    self.status_updated.emit(f"Error processing file {file_name}: {str(e)}")
                    continue
            
            return summed_data
            
        except Exception as e:
            self.status_updated.emit(f"Error loading multiple CBF files: {str(e)}")
            return None
    
    def _display_image(self, image_data):
        """显示图像数据"""
        try:
            # 显示图像到GraphicsView - 传递原始数据，让显示方法处理对数转换
            if hasattr(self.ui, 'gisaxsInputGraphicsView'):
                self._update_graphics_view(image_data)
            
            # 更新状态信息
            use_log = hasattr(self.ui, 'LogCheckBox') and self.ui.LogCheckBox.isChecked()
            title_suffix = " (Log Scale)" if use_log else ""
            
            if use_log:
                # 使用简单的log方法计算状态显示
                log_data = np.log(image_data + 0.001)
                finite_data = log_data[np.isfinite(log_data)]
                
                if len(finite_data) > 0:
                    log_min = np.min(finite_data)
                    log_max = np.max(finite_data)
                    nan_count = np.sum(np.isnan(log_data))
                    
                    self.status_updated.emit(f"Image displayed: shape {image_data.shape}, "
                                           f"Original range: {np.min(image_data):.2f} - {np.max(image_data):.2f}, "
                                           f"Log range: {log_min:.2f} - {log_max:.2f}, "
                                           f"NaN: {nan_count}{title_suffix}")
                else:
                    self.status_updated.emit(f"Image displayed: shape {image_data.shape}, "
                                           f"Original range: {np.min(image_data):.2f} - {np.max(image_data):.2f}, "
                                           f"Log: All NaN values!{title_suffix}")
            else:
                self.status_updated.emit(f"Image displayed: shape {image_data.shape}, "
                                       f"Range: {np.min(image_data):.2f} - {np.max(image_data):.2f}{title_suffix}")
            
            # 存储当前显示的数据
            self.current_stack_data = image_data
            
        except Exception as e:
            self.status_updated.emit(f"Display error: {str(e)}")
    
    def _update_graphics_view(self, image_data):
        """更新GraphicsView中的图像显示 - 支持交互式缩放和平移"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                self.status_updated.emit("matplotlib not available for image display")
                return
            
            graphics_view = self.ui.gisaxsInputGraphicsView
            
            # 清除之前的内容
            scene = graphics_view.scene()
            if scene:
                scene.clear()
            else:
                scene = QGraphicsScene()
                graphics_view.setScene(scene)
            
            # 创建matplotlib图形和canvas - 调整尺寸以适应纯图像显示
            fig = Figure(figsize=(10, 8), dpi=100, facecolor='white')
            canvas = FigureCanvas(fig)
            
            # 创建导航工具栏
            toolbar = NavigationToolbar(canvas, None)
            
            # 创建包含canvas和toolbar的widget
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            layout.setContentsMargins(0, 0, 0, 0)
            
            # 创建axes并显示图像
            ax = fig.add_subplot(111)
            
            # 检查是否启用对数显示
            use_log = hasattr(self.ui, 'LogCheckBox') and self.ui.LogCheckBox.isChecked()
            
            # 打印调试信息
            print(f"Debug: Image data shape: {image_data.shape}")
            print(f"Debug: Data range: {np.min(image_data):.6f} - {np.max(image_data):.6f}")
            print(f"Debug: Data type: {image_data.dtype}")
            print(f"Debug: Use log: {use_log}")
            
            if use_log:
                # 使用用户喜欢的简单直接方法：np.log(data + 0.001)
                # 让matplotlib自动处理NaN值，效果更好
                display_data = np.log(image_data + 0.001)
                
                # 不设置vmin/vmax，让matplotlib自动缩放
                vmin = None
                vmax = None
                
                # 计算统计信息用于调试
                finite_data = display_data[np.isfinite(display_data)]
                if len(finite_data) > 0:
                    actual_min = np.min(finite_data)
                    actual_max = np.max(finite_data)
                    print(f"Debug: Log display with simple method np.log(data + 0.001)")
                    print(f"Debug: Original range: {np.min(image_data):.3f} - {np.max(image_data):.3f}")
                    print(f"Debug: Log range (finite): {actual_min:.3f} - {actual_max:.3f}")
                    print(f"Debug: NaN count: {np.sum(np.isnan(display_data))}")
                    print(f"Debug: Finite count: {len(finite_data)}")
                else:
                    print("Debug: No finite values in log data!")
                    # 如果没有有限值，回退到原始数据
                    display_data = image_data
                    vmin = None
                    vmax = None
                
            else:
                display_data = image_data
                
                # 获取数据统计信息
                data_min = np.min(image_data)
                data_max = np.max(image_data)
                data_mean = np.mean(image_data)
                data_std = np.std(image_data)
                
                # 计算不同的百分位数
                p1 = np.percentile(image_data, 1)
                p5 = np.percentile(image_data, 5)
                p95 = np.percentile(image_data, 95)
                p99 = np.percentile(image_data, 99)
                
                print(f"Debug: Data statistics - min: {data_min:.3f}, max: {data_max:.3f}, mean: {data_mean:.3f}, std: {data_std:.3f}")
                print(f"Debug: Percentiles - 1%: {p1:.3f}, 5%: {p5:.3f}, 95%: {p95:.3f}, 99%: {p99:.3f}")
                
                # 智能选择显示范围
                dynamic_range = data_max - data_min
                
                # 如果动态范围很大，使用更严格的百分位数
                if dynamic_range > 1000:
                    # 极端动态范围，使用5%-95%百分位数
                    vmin = p5
                    vmax = p95
                    print(f"Debug: Using 5%-95% percentiles due to extreme dynamic range")
                elif dynamic_range > 100:
                    # 中等动态范围，使用1%-99%百分位数
                    vmin = p1
                    vmax = p99
                    print(f"Debug: Using 1%-99% percentiles due to large dynamic range")
                else:
                    # 小动态范围，使用实际最小最大值
                    vmin = data_min
                    vmax = data_max
                    print(f"Debug: Using actual min/max for small dynamic range")
                
                # 处理负值情况
                if data_min < 0:
                    # 如果有负值，可能需要调整策略
                    if abs(data_min) > data_max * 0.1:  # 负值占比较大
                        # 使用均值±3倍标准差作为范围
                        vmin = max(data_min, data_mean - 3 * data_std)
                        vmax = min(data_max, data_mean + 3 * data_std)
                        print(f"Debug: Adjusted range for significant negative values: {vmin:.3f} - {vmax:.3f}")
                    else:
                        # 负值很小，可能是背景噪声，设置vmin为0
                        vmin = 0
                        print(f"Debug: Set vmin=0 to handle small negative values")
                
                # 确保范围合理
                if vmax - vmin < dynamic_range * 0.001:  # 范围太小
                    vmin = data_min
                    vmax = data_max
                    print(f"Debug: Range too small, using actual min/max")
                
                print(f"Debug: Final linear range: {vmin:.3f} - {vmax:.3f}")
                print(f"Debug: Range compression ratio: {(vmax - vmin) / dynamic_range:.3f}")
            
            # 确保vmin != vmax
            if abs(vmax - vmin) < 1e-10:
                vmax = vmin + 1
                print(f"Debug: Adjusted range to avoid vmin=vmax: {vmin:.3f} - {vmax:.3f}")
            
            # 显示图像 - 简洁模式，无颜色条和坐标轴
            if use_log and vmin is None and vmax is None:
                # log模式且无指定范围，让matplotlib自动处理
                print(f"Debug: Using log mode with auto range")
                im = ax.imshow(display_data, cmap='viridis', aspect='auto', origin='lower')
            else:
                # 线性模式或指定了范围
                print(f"Debug: Using linear mode with vmin={vmin}, vmax={vmax}")
                im = ax.imshow(display_data, cmap='viridis', aspect='auto', origin='lower', 
                              vmin=vmin, vmax=vmax)
            
            print(f"Debug: imshow completed, image object: {im}")
            print(f"Debug: display_data shape: {display_data.shape}")
            print(f"Debug: display_data finite values: {np.sum(np.isfinite(display_data)) if hasattr(display_data, 'shape') else 'N/A'}")
            
            # 隐藏坐标轴
            ax.axis('off')
            
            # 调整布局，去除边距
            fig.tight_layout(pad=0)
            
            # 强制刷新canvas
            canvas.draw()
            
            # 将widget添加到scene
            proxy_widget = scene.addWidget(widget)
            
            # 设置widget的最小尺寸，确保可见
            widget.setMinimumSize(800, 600)
            
            # 调整视图以适应内容
            graphics_view.fitInView(proxy_widget, Qt.KeepAspectRatio)
            
            # 强制更新视图
            graphics_view.update()
            
            # 启用交互功能的提示
            self.status_updated.emit("Image displayed with zoom/pan controls. Use toolbar or mouse wheel to interact.")
            
        except Exception as e:
            self.status_updated.emit(f"Graphics view update error: {str(e)}")
            import traceback
            print(f"Error details: {traceback.format_exc()}")
    
    def _on_auto_show_changed(self):
        """AutoShow选项改变时的处理"""
        auto_show = hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked()
        self.status_updated.emit(f"AutoShow {'enabled' if auto_show else 'disabled'}")
    
    def _on_log_changed(self):
        """Log选项改变时的处理"""
        # 如果当前有显示的数据，重新显示
        if self.current_stack_data is not None:
            self._display_image(self.current_stack_data)
    
    def _process_stack_files(self):
        """处理stack文件叠加 - 已弃用，功能已分离到_show_image"""
        # 这个方法保留用于向后兼容，但功能已迁移到_show_image
        self._update_stack_display()
    
    def _process_single_cbf_file(self, cbf_file):
        """处理单个CBF文件 - 已弃用，使用_load_single_cbf_file"""
        # 保留用于向后兼容
        pass
    
    def _process_multiple_cbf_files(self, start_file, stack_count):
        """处理多个CBF文件叠加 - 已弃用，使用_load_multiple_cbf_files"""
        # 保留用于向后兼容
        pass
    
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
