"""
主控制器 - 协调所有子控制器，管理应用程序的主要逻辑
"""

import json
import os
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from .beam_controller import BeamController
from .detector_controller import DetectorController
from .sample_controller import SampleController
from .preprocessing_controller import PreprocessingController
from .trainset_controller import TrainsetController


class MainController(QObject):
    """主控制器，协调所有功能模块"""
    
    # 状态更新信号
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.parent = parent
        
        # 初始化子控制器
        self.beam_controller = BeamController(ui, self)
        self.detector_controller = DetectorController(ui, self)
        self.sample_controller = SampleController(ui, self)
        self.preprocessing_controller = PreprocessingController(ui, self)
        self.trainset_controller = TrainsetController(ui, self)
        
        # 当前项目参数
        self.current_parameters = {}
        
        # 设置界面连接
        self._setup_connections()
        
        # 初始化界面
        self._initialize_ui()
    
    def _setup_connections(self):
        """设置信号连接"""
        # 主要按钮连接
        self.ui.trainsetBuildButton.clicked.connect(self._switch_to_trainset_build)
        self.ui.gisaxsPredictButton.clicked.connect(self._switch_to_gisaxs_predict)
        
        # 连接子控制器信号
        self.beam_controller.parameters_changed.connect(self._on_parameters_changed)
        self.detector_controller.parameters_changed.connect(self._on_parameters_changed)
        self.sample_controller.parameters_changed.connect(self._on_parameters_changed)
        self.preprocessing_controller.parameters_changed.connect(self._on_parameters_changed)
        
        # 连接训练集生成信号
        self.trainset_controller.generation_started.connect(
            lambda: self.status_updated.emit("训练集生成开始...")
        )
        self.trainset_controller.generation_finished.connect(
            lambda: self.status_updated.emit("训练集生成完成！")
        )
        self.trainset_controller.progress_updated.connect(self.progress_updated)
    
    def _initialize_ui(self):
        """初始化界面状态"""
        # 设置默认页面
        self.ui.mainWindowWidget.setCurrentIndex(0)  # 训练集构建页面
        
        # 初始化各个控制器
        self.beam_controller.initialize()
        self.detector_controller.initialize()
        self.sample_controller.initialize()
        self.preprocessing_controller.initialize()
        self.trainset_controller.initialize()
        
        # 更新状态
        self.status_updated.emit("GISAXS Toolkit 就绪")
    
    def _switch_to_trainset_build(self):
        """切换到训练集构建页面"""
        self.ui.mainWindowWidget.setCurrentIndex(0)
        self.status_updated.emit("切换到训练集构建模式")
    
    def _switch_to_gisaxs_predict(self):
        """切换到GISAXS预测页面"""
        self.ui.mainWindowWidget.setCurrentIndex(1)
        self.status_updated.emit("切换到GISAXS预测模式")
        
        # 初始化预测页面的基本状态
        self._initialize_gisaxs_predict_page()
    
    def _initialize_gisaxs_predict_page(self):
        """初始化GISAXS预测页面的状态"""
        # 设置默认框架
        if hasattr(self.ui, 'gisaxsPredictFrameworkCombox'):
            self.ui.gisaxsPredictFrameworkCombox.setCurrentIndex(0)
        
        # 设置默认为单文件模式
        if hasattr(self.ui, 'gisaxsPredictSingleFileRadioButton'):
            self.ui.gisaxsPredictSingleFileRadioButton.setChecked(True)
        
        # 清空输入框
        if hasattr(self.ui, 'gisaxsPredictExportFolderValue'):
            self.ui.gisaxsPredictExportFolderValue.clear()
        
        # 连接预测页面的信号
        self._connect_gisaxs_predict_signals()
        
        # 强制压缩预测页面布局
        self._force_compact_predict_layout()
    
    def _force_compact_predict_layout(self):
        """强制压缩预测页面布局 - 已集成到布局工具中"""
        # 新的布局工具已经自动处理页面压缩，无需手动调用
        pass
    
    def _connect_gisaxs_predict_signals(self):
        """连接GISAXS预测页面的信号"""
        # 连接选择文件夹按钮
        if hasattr(self.ui, 'gisaxsPredictChooseFolderButton'):
            if not hasattr(self, '_gisaxs_folder_connected'):
                self.ui.gisaxsPredictChooseFolderButton.clicked.connect(self._choose_gisaxs_folder)
                self._gisaxs_folder_connected = True
        
        # 连接选择GISAXS文件按钮
        if hasattr(self.ui, 'gisaxsPredictChooseGisaxsFileButton'):
            if not hasattr(self, '_gisaxs_file_connected'):
                self.ui.gisaxsPredictChooseGisaxsFileButton.clicked.connect(self._choose_gisaxs_file)
                self._gisaxs_file_connected = True
        
        # 连接导出文件夹按钮
        if hasattr(self.ui, 'gisaxsPredictExportFolderButton'):
            if not hasattr(self, '_gisaxs_export_connected'):
                self.ui.gisaxsPredictExportFolderButton.clicked.connect(self._choose_export_folder)
                self._gisaxs_export_connected = True
        
        # 连接预测按钮
        if hasattr(self.ui, 'gisaxsPredictPredictButton'):
            if not hasattr(self, '_gisaxs_predict_connected'):
                self.ui.gisaxsPredictPredictButton.clicked.connect(self._run_gisaxs_predict)
                self._gisaxs_predict_connected = True
    
    def _choose_gisaxs_folder(self):
        """选择GISAXS文件夹"""
        from PyQt5.QtWidgets import QFileDialog
        
        folder_path = QFileDialog.getExistingDirectory(
            self.parent,
            "选择GISAXS文件夹",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder_path:
            self.status_updated.emit(f"已选择GISAXS文件夹: {folder_path}")
            # 这里可以添加文件夹路径的存储逻辑
    
    def _choose_gisaxs_file(self):
        """选择GISAXS文件"""
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent,
            "选择GISAXS文件",
            "",
            "GISAXS Files (*.tif *.tiff *.dat *.txt);;All Files (*)"
        )
        
        if file_path:
            self.status_updated.emit(f"已选择GISAXS文件: {file_path}")
            # 这里可以添加文件路径的存储逻辑
    
    def _choose_export_folder(self):
        """选择导出文件夹"""
        from PyQt5.QtWidgets import QFileDialog
        
        folder_path = QFileDialog.getExistingDirectory(
            self.parent,
            "选择导出文件夹",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder_path and hasattr(self.ui, 'gisaxsPredictExportFolderValue'):
            self.ui.gisaxsPredictExportFolderValue.setText(folder_path)
            self.status_updated.emit(f"已选择导出文件夹: {folder_path}")
    
    def _run_gisaxs_predict(self):
        """运行GISAXS预测"""
        # 获取预测参数
        predict_params = self._get_gisaxs_predict_parameters()
        
        # 验证参数
        if not self._validate_gisaxs_predict_parameters(predict_params):
            return
        
        # 显示预测开始信息
        self.status_updated.emit("开始GISAXS预测...")
        
        # 更新预测状态显示
        if hasattr(self.ui, 'predictStatusTextBrowser'):
            self.ui.predictStatusTextBrowser.append("=== GISAXS预测开始 ===")
            self.ui.predictStatusTextBrowser.append(f"框架: {predict_params['framework']}")
            self.ui.predictStatusTextBrowser.append(f"模式: {predict_params['mode']}")
            self.ui.predictStatusTextBrowser.append(f"导出路径: {predict_params['export_path']}")
            self.ui.predictStatusTextBrowser.append("注意: 物理处理功能尚未实现，这里仅做UI演示")
        
        # TODO: 这里将来会调用实际的预测算法
        # 目前只是演示UI功能
        self.status_updated.emit("GISAXS预测完成（演示模式）")
    
    def _get_gisaxs_predict_parameters(self):
        """获取GISAXS预测参数"""
        params = {}
        
        # 获取框架选择
        if hasattr(self.ui, 'gisaxsPredictFrameworkCombox'):
            params['framework'] = self.ui.gisaxsPredictFrameworkCombox.currentText()
        else:
            params['framework'] = 'tensorflow 2.15.0'
        
        # 获取模式选择
        if hasattr(self.ui, 'gisaxsPredictSingleFileRadioButton') and self.ui.gisaxsPredictSingleFileRadioButton.isChecked():
            params['mode'] = 'single_file'
        elif hasattr(self.ui, 'gisaxsPredictMultiFilesRadioButton') and self.ui.gisaxsPredictMultiFilesRadioButton.isChecked():
            params['mode'] = 'multi_files'
        else:
            params['mode'] = 'single_file'
        
        # 获取导出路径
        if hasattr(self.ui, 'gisaxsPredictExportFolderValue'):
            params['export_path'] = self.ui.gisaxsPredictExportFolderValue.text()
        else:
            params['export_path'] = ''
        
        return params
    
    def _validate_gisaxs_predict_parameters(self, params):
        """验证GISAXS预测参数"""
        if not params.get('export_path'):
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                self.parent,
                "参数错误",
                "请选择导出文件夹!"
            )
            return False
        
        # 检查导出路径是否存在
        import os
        if not os.path.exists(params['export_path']):
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                self.parent,
                "路径错误",
                f"导出文件夹不存在: {params['export_path']}"
            )
            return False
        
        return True
    
    def _on_parameters_changed(self, module_name, parameters):
        """当参数发生改变时的处理"""
        self.current_parameters[module_name] = parameters
        self.status_updated.emit(f"{module_name}参数已更新")
    
    def get_all_parameters(self):
        """获取所有模块的参数"""
        return {
            'beam': self.beam_controller.get_parameters(),
            'detector': self.detector_controller.get_parameters(),
            'sample': self.sample_controller.get_parameters(),
            'preprocessing': self.preprocessing_controller.get_parameters(),
            'trainset': self.trainset_controller.get_parameters()
        }
    
    def load_parameters_from_file(self, file_path):
        """从文件加载参数"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                parameters = json.load(f)
            
            # 分发参数到各个控制器
            if 'beam' in parameters:
                self.beam_controller.set_parameters(parameters['beam'])
            if 'detector' in parameters:
                self.detector_controller.set_parameters(parameters['detector'])
            if 'sample' in parameters:
                self.sample_controller.set_parameters(parameters['sample'])
            if 'preprocessing' in parameters:
                self.preprocessing_controller.set_parameters(parameters['preprocessing'])
            if 'trainset' in parameters:
                self.trainset_controller.set_parameters(parameters['trainset'])
            
            self.status_updated.emit(f"参数从 {file_path} 加载成功")
            return True
            
        except Exception as e:
            self.status_updated.emit(f"参数加载失败: {str(e)}")
            return False
    
    def save_parameters_to_file(self, file_path):
        """保存参数到文件"""
        try:
            parameters = self.get_all_parameters()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(parameters, f, indent=4, ensure_ascii=False)
            
            self.status_updated.emit(f"参数保存到 {file_path} 成功")
            return True
            
        except Exception as e:
            self.status_updated.emit(f"参数保存失败: {str(e)}")
            return False
    
    def validate_all_parameters(self):
        """验证所有参数的有效性"""
        validation_results = []
        
        # 验证各个模块
        modules = [
            ('光束参数', self.beam_controller),
            ('探测器参数', self.detector_controller),
            ('样品参数', self.sample_controller),
            ('预处理参数', self.preprocessing_controller),
            ('训练集参数', self.trainset_controller)
        ]
        
        for name, controller in modules:
            is_valid, message = controller.validate_parameters()
            validation_results.append((name, is_valid, message))
        
        return validation_results
    
    def show_validation_results(self):
        """显示参数验证结果"""
        results = self.validate_all_parameters()
        
        valid_count = sum(1 for _, is_valid, _ in results if is_valid)
        total_count = len(results)
        
        if valid_count == total_count:
            QMessageBox.information(
                self.parent, 
                "参数验证", 
                "所有参数验证通过！"
            )
        else:
            error_messages = []
            for name, is_valid, message in results:
                if not is_valid:
                    error_messages.append(f"{name}: {message}")
            
            QMessageBox.warning(
                self.parent,
                "参数验证失败",
                "以下参数存在问题:\n\n" + "\n".join(error_messages)
            )
    
    def reset_all_parameters(self):
        """重置所有参数到默认值"""
        self.beam_controller.reset_to_defaults()
        self.detector_controller.reset_to_defaults()
        self.sample_controller.reset_to_defaults()
        self.preprocessing_controller.reset_to_defaults()
        self.trainset_controller.reset_to_defaults()
        
        self.status_updated.emit("所有参数已重置为默认值")
