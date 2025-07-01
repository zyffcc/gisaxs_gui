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
