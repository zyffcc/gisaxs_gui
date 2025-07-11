"""
GISAXS Predict 控制器 - 处理GISAXS数据的预测功能
"""

import os
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox

# 尝试导入所需的库
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
            
        # 设置默认参数
        self._set_default_parameters()
        
        # 检查依赖库
        self._check_dependencies()
        
    def _check_dependencies(self):
        """检查所需的依赖库"""
        if not TENSORFLOW_AVAILABLE:
            self.status_updated.emit("Warning: TensorFlow not available. TensorFlow predictions will be disabled.")
            
        if not TORCH_AVAILABLE:
            self.status_updated.emit("Warning: PyTorch not available. PyTorch predictions will be disabled.")
        
    def _set_default_parameters(self):
        """设置默认参数"""
        self.current_parameters = {
            'framework': 'tensorflow 2.15.0',
            'mode': 'single_file',
            'input_file': '',
            'input_folder': '',
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
            
            if self.current_parameters['mode'] == 'single_file':
                results = self._predict_single_file()
            else:
                results = self._predict_multi_files()
            
            if results:
                self._save_results(results)
                self.prediction_completed.emit(results)
                self.status_updated.emit("GISAXS预测完成！")
                self.progress_updated.emit(100)
            else:
                self.status_updated.emit("GISAXS预测失败")
                
        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "预测错误",
                f"运行GISAXS预测时出错:\n{str(e)}"
            )
            self.status_updated.emit(f"GISAXS预测错误: {str(e)}")
            
    def _update_parameters_from_ui(self):
        """从UI更新参数"""
        # 获取框架选择
        if hasattr(self.ui, 'gisaxsPredictFrameworkCombox'):
            framework_index = self.ui.gisaxsPredictFrameworkCombox.currentIndex()
            framework_text = self.ui.gisaxsPredictFrameworkCombox.currentText()
            self.current_parameters['framework'] = framework_text
        
        # 获取导出路径
        if hasattr(self.ui, 'gisaxsPredictExportFolderValue'):
            export_path = self.ui.gisaxsPredictExportFolderValue.text().strip()
            if export_path:
                self.current_parameters['export_path'] = export_path
            
    def _validate_parameters(self):
        """验证参数"""
        if not self.current_parameters.get('export_path'):
            QMessageBox.warning(
                self.main_window,
                "参数错误",
                "请选择导出文件夹"
            )
            return False
        
        # 检查导出路径是否存在
        if not os.path.exists(self.current_parameters['export_path']):
            QMessageBox.warning(
                self.main_window,
                "路径错误",
                f"导出路径不存在: {self.current_parameters['export_path']}"
            )
            return False
            
        # 检查输入文件/文件夹
        mode = self.current_parameters.get('mode', 'single_file')
        if mode == 'single_file':
            if not self.current_parameters.get('input_file'):
                QMessageBox.warning(
                    self.main_window,
                    "参数错误",
                    "请选择输入文件"
                )
                return False
                
            if not os.path.exists(self.current_parameters['input_file']):
                QMessageBox.warning(
                    self.main_window,
                    "文件错误",
                    f"输入文件不存在: {self.current_parameters['input_file']}"
                )
                return False
        else:
            if not self.current_parameters.get('input_folder'):
                QMessageBox.warning(
                    self.main_window,
                    "参数错误",
                    "请选择输入文件夹"
                )
                return False
                
            if not os.path.exists(self.current_parameters['input_folder']):
                QMessageBox.warning(
                    self.main_window,
                    "文件夹错误",
                    f"输入文件夹不存在: {self.current_parameters['input_folder']}"
                )
                return False
        
        return True
        
    def _predict_single_file(self):
        """预测单个文件"""
        try:
            input_file = self.current_parameters['input_file']
            self.status_updated.emit(f"处理文件: {os.path.basename(input_file)}")
            self.progress_updated.emit(25)
            
            # TODO: 实现实际的预测逻辑
            # 这里应该加载模型并进行预测
            self.progress_updated.emit(50)
            
            # 模拟预测结果
            results = {
                'file': input_file,
                'predictions': [],
                'confidence': 0.95,
                'processing_time': 1.5
            }
            
            self.progress_updated.emit(75)
            return results
            
        except Exception as e:
            self.status_updated.emit(f"单文件预测错误: {str(e)}")
            return None
        
    def _predict_multi_files(self):
        """预测多个文件"""
        try:
            input_folder = self.current_parameters['input_folder']
            self.status_updated.emit(f"处理文件夹: {input_folder}")
            self.progress_updated.emit(10)
            
            # 获取文件列表
            supported_extensions = ['.tif', '.tiff', '.dat', '.txt', '.h5', '.hdf5']
            files = []
            for file in os.listdir(input_folder):
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    files.append(os.path.join(input_folder, file))
            
            if not files:
                self.status_updated.emit("没有找到支持的文件")
                return None
            
            self.progress_updated.emit(20)
            
            # TODO: 实现实际的批量预测逻辑
            results = {
                'folder': input_folder,
                'total_files': len(files),
                'processed_files': len(files),
                'predictions': [],
                'average_confidence': 0.92,
                'total_processing_time': len(files) * 1.2
            }
            
            # 模拟处理进度
            for i, file in enumerate(files):
                progress = 20 + int((i / len(files)) * 60)
                self.progress_updated.emit(progress)
                self.status_updated.emit(f"处理文件 {i+1}/{len(files)}: {os.path.basename(file)}")
                
                # 这里应该调用实际的预测函数
                # prediction = self._predict_file(file)
                # results['predictions'].append(prediction)
            
            self.progress_updated.emit(80)
            return results
            
        except Exception as e:
            self.status_updated.emit(f"多文件预测错误: {str(e)}")
            return None
        
    def _save_results(self, results):
        """保存预测结果"""
        try:
            export_path = self.current_parameters['export_path']
            
            # 生成结果文件名
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(export_path, f"gisaxs_prediction_results_{timestamp}.json")
            
            # 保存为JSON格式
            import json
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            self.status_updated.emit(f"结果已保存到: {result_file}")
            
        except Exception as e:
            self.status_updated.emit(f"保存结果错误: {str(e)}")
    
    def get_parameters(self):
        """获取当前参数"""
        return self.current_parameters.copy()
        
    def set_parameters(self, parameters):
        """设置参数"""
        self.current_parameters.update(parameters)
        self.parameters_changed.emit(self.current_parameters)
