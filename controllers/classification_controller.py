"""
Classification 控制器 - 处理GISAXS数据的分类功能
"""

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import os


class ClassificationController(QObject):
    """Classification控制器，处理GISAXS数据的分类"""
    
    # 状态信号
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    parameters_changed = pyqtSignal(dict)
    classification_completed = pyqtSignal(dict)
    
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.parent = parent
        # 获取主窗口引用
        self.main_window = parent.parent if hasattr(parent, 'parent') else None
        
        # 当前参数
        self.current_parameters = {}
        
        # 分类结果
        self.classification_results = {}
        
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
        # 连接文件/文件夹选择按钮
        if hasattr(self.ui, 'classificationChooseFileButton'):
            self.ui.classificationChooseFileButton.clicked.connect(self._choose_input_file)
            
        if hasattr(self.ui, 'classificationChooseFolderButton'):
            self.ui.classificationChooseFolderButton.clicked.connect(self._choose_input_folder)
            
        if hasattr(self.ui, 'classificationOutputButton'):
            self.ui.classificationOutputButton.clicked.connect(self._choose_output_folder)
            
        # 连接分类相关按钮
        if hasattr(self.ui, 'classificationStartButton'):
            self.ui.classificationStartButton.clicked.connect(self._start_classification)
            
        if hasattr(self.ui, 'classificationResetButton'):
            self.ui.classificationResetButton.clicked.connect(self._reset_classification)
            
        # 连接模型选择相关控件
        if hasattr(self.ui, 'classificationModelComboBox'):
            self.ui.classificationModelComboBox.currentTextChanged.connect(self._on_model_changed)
            
        # 连接输入模式切换
        self._connect_input_mode_widgets()
        
    def _connect_input_mode_widgets(self):
        """连接输入模式切换控件"""
        # 单文件/批量处理模式切换
        if hasattr(self.ui, 'classificationSingleFileRadio'):
            self.ui.classificationSingleFileRadio.toggled.connect(self._on_input_mode_changed)
            
        if hasattr(self.ui, 'classificationBatchModeRadio'):
            self.ui.classificationBatchModeRadio.toggled.connect(self._on_input_mode_changed)
            
    def _initialize_ui(self):
        """初始化界面状态"""
        # 设置默认参数
        self._set_default_parameters()
        
        # 初始化模型列表
        self._initialize_model_list()
        
        # 清空输入显示
        self._clear_input_displays()
        
    def _set_default_parameters(self):
        """设置默认参数"""
        self.current_parameters = {
            'input_mode': 'single',  # 'single' or 'batch'
            'input_file': '',
            'input_folder': '',
            'output_folder': '',
            'model_name': 'default',
            'confidence_threshold': 0.8,
            'batch_size': 32
        }
        
    def _initialize_model_list(self):
        """初始化分类模型列表"""
        if hasattr(self.ui, 'classificationModelComboBox'):
            # 添加可用的分类模型
            models = [
                "CNN基础模型",
                "ResNet模型", 
                "VGG模型",
                "自定义模型"
            ]
            
            self.ui.classificationModelComboBox.clear()
            self.ui.classificationModelComboBox.addItems(models)
            self.ui.classificationModelComboBox.setCurrentIndex(0)
            
    def _clear_input_displays(self):
        """清空输入显示"""
        # 清空文件路径显示
        if hasattr(self.ui, 'classificationFilePathLabel'):
            self.ui.classificationFilePathLabel.clear()
            
        if hasattr(self.ui, 'classificationFolderPathLabel'):
            self.ui.classificationFolderPathLabel.clear()
            
        if hasattr(self.ui, 'classificationOutputPathLabel'):
            self.ui.classificationOutputPathLabel.clear()
            
    def _choose_input_file(self):
        """选择输入文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "选择GISAXS文件",
            "",
            "GISAXS Files (*.tif *.tiff *.dat *.txt *.h5 *.hdf5);;Image Files (*.png *.jpg *.jpeg);;All Files (*)"
        )
        
        if file_path:
            self.current_parameters['input_file'] = file_path
            self.current_parameters['input_mode'] = 'single'
            
            # 更新UI显示
            if hasattr(self.ui, 'classificationFilePathLabel'):
                self.ui.classificationFilePathLabel.setText(os.path.basename(file_path))
                
            # 设置单文件模式
            if hasattr(self.ui, 'classificationSingleFileRadio'):
                self.ui.classificationSingleFileRadio.setChecked(True)
                
            self.status_updated.emit(f"已选择输入文件: {file_path}")
            self.parameters_changed.emit(self.current_parameters)
            
    def _choose_input_folder(self):
        """选择输入文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self.main_window,
            "选择包含GISAXS文件的文件夹",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder_path:
            self.current_parameters['input_folder'] = folder_path
            self.current_parameters['input_mode'] = 'batch'
            
            # 更新UI显示
            if hasattr(self.ui, 'classificationFolderPathLabel'):
                self.ui.classificationFolderPathLabel.setText(os.path.basename(folder_path))
                
            # 设置批量模式
            if hasattr(self.ui, 'classificationBatchModeRadio'):
                self.ui.classificationBatchModeRadio.setChecked(True)
                
            self.status_updated.emit(f"已选择输入文件夹: {folder_path}")
            self.parameters_changed.emit(self.current_parameters)
            
    def _choose_output_folder(self):
        """选择输出文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self.main_window,
            "选择分类结果输出文件夹",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder_path:
            self.current_parameters['output_folder'] = folder_path
            
            # 更新UI显示
            if hasattr(self.ui, 'classificationOutputPathLabel'):
                self.ui.classificationOutputPathLabel.setText(folder_path)
                
            self.status_updated.emit(f"已选择输出文件夹: {folder_path}")
            self.parameters_changed.emit(self.current_parameters)
            
    def _on_model_changed(self, model_name):
        """模型选择改变时的处理"""
        self.current_parameters['model_name'] = model_name
        self.status_updated.emit(f"已选择分类模型: {model_name}")
        self.parameters_changed.emit(self.current_parameters)
        
    def _on_input_mode_changed(self):
        """输入模式改变时的处理"""
        if hasattr(self.ui, 'classificationSingleFileRadio') and self.ui.classificationSingleFileRadio.isChecked():
            self.current_parameters['input_mode'] = 'single'
            self.status_updated.emit("切换到单文件分类模式")
        elif hasattr(self.ui, 'classificationBatchModeRadio') and self.ui.classificationBatchModeRadio.isChecked():
            self.current_parameters['input_mode'] = 'batch'
            self.status_updated.emit("切换到批量分类模式")
            
        self.parameters_changed.emit(self.current_parameters)
        
    def _start_classification(self):
        """开始分类"""
        # 验证输入参数
        if not self._validate_parameters():
            return
            
        try:
            self.status_updated.emit("开始GISAXS数据分类...")
            self.progress_updated.emit(0)
            
            # 执行分类
            if self.current_parameters['input_mode'] == 'single':
                results = self._classify_single_file()
            else:
                results = self._classify_batch()
                
            # 保存结果
            if self.current_parameters.get('output_folder'):
                self._save_results(results)
                
            self.classification_results = results
            self.progress_updated.emit(100)
            self.status_updated.emit("分类完成！")
            self.classification_completed.emit(results)
            
        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "错误",
                f"分类过程中出错:\n{str(e)}"
            )
            
    def _validate_parameters(self):
        """验证参数"""
        if self.current_parameters['input_mode'] == 'single':
            if not self.current_parameters.get('input_file'):
                QMessageBox.warning(self.main_window, "警告", "请选择输入文件！")
                return False
                
            if not os.path.exists(self.current_parameters['input_file']):
                QMessageBox.warning(self.main_window, "警告", "输入文件不存在！")
                return False
                
        elif self.current_parameters['input_mode'] == 'batch':
            if not self.current_parameters.get('input_folder'):
                QMessageBox.warning(self.main_window, "警告", "请选择输入文件夹！")
                return False
                
            if not os.path.exists(self.current_parameters['input_folder']):
                QMessageBox.warning(self.main_window, "警告", "输入文件夹不存在！")
                return False
                
        return True
        
    def _classify_single_file(self):
        """分类单个文件"""
        # TODO: 实现单文件分类逻辑
        file_path = self.current_parameters['input_file']
        
        # 模拟分类过程
        self.progress_updated.emit(50)
        
        # 这里应该调用实际的分类模型
        result = {
            'file': file_path,
            'predicted_class': 'Class_A',  # 示例结果
            'confidence': 0.95,
            'processing_time': 0.5
        }
        
        return {'single_file': result}
        
    def _classify_batch(self):
        """批量分类"""
        # TODO: 实现批量分类逻辑
        folder_path = self.current_parameters['input_folder']
        
        # 获取文件列表
        file_list = self._get_gisaxs_files(folder_path)
        
        results = {}
        total_files = len(file_list)
        
        for i, file_path in enumerate(file_list):
            # 更新进度
            progress = int((i + 1) / total_files * 100)
            self.progress_updated.emit(progress)
            
            # 分类单个文件
            # 这里应该调用实际的分类模型
            result = {
                'predicted_class': f'Class_{i % 3 + 1}',  # 示例结果
                'confidence': 0.85 + (i % 10) * 0.01,
                'processing_time': 0.3
            }
            
            results[os.path.basename(file_path)] = result
            
        return {'batch_results': results}
        
    def _get_gisaxs_files(self, folder_path):
        """获取文件夹中的GISAXS文件"""
        extensions = ['.tif', '.tiff', '.dat', '.txt', '.h5', '.hdf5', '.png', '.jpg', '.jpeg']
        file_list = []
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    file_list.append(os.path.join(root, file))
                    
        return file_list
        
    def _save_results(self, results):
        """保存分类结果"""
        # TODO: 实现结果保存逻辑
        output_path = os.path.join(
            self.current_parameters['output_folder'],
            'classification_results.json'
        )
        
        try:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            self.status_updated.emit(f"分类结果已保存到: {output_path}")
            
        except Exception as e:
            self.status_updated.emit(f"保存结果时出错: {str(e)}")
            
    def _reset_classification(self):
        """重置分类参数"""
        self._set_default_parameters()
        self._initialize_ui()
        self.classification_results = {}
        
        self.status_updated.emit("已重置分类参数")
        self.parameters_changed.emit(self.current_parameters)
        
    def get_parameters(self):
        """获取当前参数"""
        return self.current_parameters.copy()
        
    def set_parameters(self, parameters):
        """设置参数"""
        self.current_parameters.update(parameters)
        self.parameters_changed.emit(self.current_parameters)
        
    def get_results(self):
        """获取分类结果"""
        return self.classification_results.copy()
