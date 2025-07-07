"""
GISAXS Predict 控制器 - 处理GISAXS数据的预测功能
"""

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import os


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
        
    def reset_to_defaults(self):
        """重置参数到默认值"""
        self.current_parameters = {}
        self.prediction_results = {}
        
        # 重置UI到默认状态
        self._initialize_ui()
        
        self.status_updated.emit("GISAXS预测参数已重置为默认值")
        self.parameters_changed.emit(self.current_parameters)
