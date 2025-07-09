"""
主控制器 - 协调所有子控制器，管理应用程序的主要逻辑
"""

import json
import os
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from .trainset_controller import TrainsetController
from .fitting_controller import FittingController
from .classification_controller import ClassificationController
from .gisaxs_predict_controller import GisaxsPredictController
from core.global_params import global_params


class MainController(QObject):
    """主控制器，协调所有功能模块"""
    
    # 状态更新信号
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.parent = parent
        
        # 初始化四个主页面控制器
        self.trainset_controller = TrainsetController(ui, self)
        self.fitting_controller = FittingController(ui, self)
        self.classification_controller = ClassificationController(ui, self)
        self.gisaxs_predict_controller = GisaxsPredictController(ui, self)
        
        # 注册控制器到全局参数管理器
        self._register_controllers()
        
        # 当前项目参数
        self.current_parameters = {}
        
        # 设置界面连接
        self._setup_connections()
        
        # 设置按钮样式
        self._setup_button_styles()
        
        # 初始化界面
        self._initialize_ui()
        
        # 注册UI控件到全局参数系统
        self._register_ui_controls()
    
    def _register_controllers(self):
        """注册控制器到全局参数管理器"""
        try:
            # 注册主要的控制器到全局参数管理器
            global_params.register_controller('trainset', self.trainset_controller)
            global_params.register_controller('fitting', self.fitting_controller)
            global_params.register_controller('classification', self.classification_controller)
            global_params.register_controller('gisaxs_predict', self.gisaxs_predict_controller)
            print("✓ 主控制器：子控制器注册完成")
        except Exception as e:
            print(f"主控制器：子控制器注册失败: {e}")
    
    def _setup_connections(self):
        """设置信号连接"""
        # 主要按钮连接
        self.ui.trainsetBuildButton.clicked.connect(self._switch_to_trainset_build)
        self.ui.gisaxsPredictButton.clicked.connect(self._switch_to_gisaxs_predict)
        self.ui.cutAndFittingButton.clicked.connect(self._switch_to_cut_fitting)
        self.ui.ClassficationButton.clicked.connect(self._switch_to_classification)
        
        # 连接主页面控制器信号
        self.trainset_controller.parameters_changed.connect(self._on_parameters_changed)
        self.trainset_controller.generation_started.connect(
            lambda: self.status_updated.emit("训练集生成开始...")
        )
        self.trainset_controller.generation_finished.connect(
            lambda: self.status_updated.emit("训练集生成完成！")
        )
        self.trainset_controller.progress_updated.connect(self.progress_updated)
        
        self.fitting_controller.parameters_changed.connect(
            lambda params: self._on_parameters_changed("拟合参数", params)
        )
        self.fitting_controller.status_updated.connect(self.status_updated)
        self.fitting_controller.progress_updated.connect(self.progress_updated)
        
        self.classification_controller.parameters_changed.connect(
            lambda params: self._on_parameters_changed("分类参数", params)
        )
        self.classification_controller.status_updated.connect(self.status_updated)
        self.classification_controller.progress_updated.connect(self.progress_updated)
        self.classification_controller.classification_completed.connect(self._on_classification_completed)
        
        self.gisaxs_predict_controller.parameters_changed.connect(
            lambda params: self._on_parameters_changed("GISAXS预测参数", params)
        )
        self.gisaxs_predict_controller.status_updated.connect(self.status_updated)
        self.gisaxs_predict_controller.progress_updated.connect(self.progress_updated)
        self.trainset_controller.progress_updated.connect(self.progress_updated)
    
    def _initialize_ui(self):
        """初始化界面状态"""
        # 设置默认页面 - 根据实际UI结构修正页面索引
        # 实际页面顺序：trainsetBuild(0), gisaxsPredict(1), gisaxsFitting(2), classification(3)
        # 按钮顺序：cutAndFitting, gisaxsPredict, trainsetBuild, classification
        self.ui.mainWindowWidget.setCurrentIndex(2)  # 默认显示Cut Fitting页面
        
        # 设置初始按钮状态
        self._update_button_states(0)  # 默认选中Cut Fitting按钮
        
        # 初始化四个主页面控制器
        self.trainset_controller.initialize()
        self.fitting_controller.initialize()
        self.classification_controller.initialize()
        self.gisaxs_predict_controller.initialize()
        
        # 更新状态
        self.status_updated.emit("GISAXS Toolkit 就绪")
    
    def _register_ui_controls(self):
        """自动注册UI控件到全局参数系统"""
        print("=== 开始注册UI控件到全局参数系统 ===")
        
        try:
            from PyQt5.QtWidgets import QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QSlider
            
            # 获取所有支持的控件类型
            supported_types = (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QSlider)
            registered_count = 0
            
            # 递归搜索UI中的所有控件
            def register_widget_recursively(widget, prefix=""):
                nonlocal registered_count
                
                # 检查当前控件
                if isinstance(widget, supported_types):
                    # 生成控件ID
                    object_name = getattr(widget, 'objectName', lambda: '')()
                    if object_name:
                        control_id = f"{prefix}{object_name}" if prefix else object_name
                        
                        # 注册控件
                        global_params.register_control(control_id, widget)
                        registered_count += 1
                        print(f"  ✓ 注册控件: {control_id} ({type(widget).__name__})")
                
                # 递归检查子控件
                if hasattr(widget, 'children'):
                    for child in widget.children():
                        if hasattr(child, 'metaObject'):  # 确保是QWidget
                            register_widget_recursively(child, prefix)
            
            # 从主窗口开始注册
            register_widget_recursively(self.ui)
            
            print(f"✓ UI控件注册完成，共注册 {registered_count} 个控件")
            
            # 连接控件值变化信号到参数同步
            global_params.ui.control_value_changed.connect(self._on_control_value_changed)
            
        except Exception as e:
            print(f"UI控件注册失败: {e}")
    
    def _on_control_value_changed(self, control_id: str, value):
        """当UI控件值发生变化时的回调"""
        # 可以在这里添加控件值变化的处理逻辑
        # 例如：自动保存到参数系统、验证输入等
        print(f"控件 '{control_id}' 值已更新为: {value}")
    
    def _switch_to_cut_fitting(self):
        """切换到Cut Fitting页面"""
        self.ui.mainWindowWidget.setCurrentIndex(2)  # Cut Fitting是第2页(gisaxsFittingPage)
        self.status_updated.emit("切换到Cut Fitting模式")
        
        # 更新按钮状态
        self._update_button_states(0)
        
        # 确保控制器已初始化
        if not self.fitting_controller._initialized:
            self.fitting_controller.initialize()
    
    def _switch_to_gisaxs_predict(self):
        """切换到GISAXS预测页面"""
        self.ui.mainWindowWidget.setCurrentIndex(1)  # GISAXS Predict是第1页
        self.status_updated.emit("切换到GISAXS预测模式")
        
        # 更新按钮状态
        self._update_button_states(1)
        
        # 确保控制器已初始化
        if not self.gisaxs_predict_controller._initialized:
            self.gisaxs_predict_controller.initialize()
    
    def _switch_to_trainset_build(self):
        """切换到训练集构建页面"""
        self.ui.mainWindowWidget.setCurrentIndex(0)  # Trainset Build是第0页(trainsetBuildPage)
        self.status_updated.emit("切换到训练集构建模式")
        
        # 更新按钮状态
        self._update_button_states(2)
        
    def _switch_to_classification(self):
        """切换到Classification页面"""
        self.ui.mainWindowWidget.setCurrentIndex(3)  # Classification是第3页
        self.status_updated.emit("切换到Classification模式")
        
        # 更新按钮状态
        self._update_button_states(3)
        
        # 确保控制器已初始化
        if not self.classification_controller._initialized:
            self.classification_controller.initialize()
    
    def _on_parameters_changed(self, module_name, parameters):
        """当参数发生改变时的处理"""
        self.current_parameters[module_name] = parameters
        self.status_updated.emit(f"{module_name}参数已更新")
    
    def get_all_parameters(self):
        """获取所有模块的参数"""
        return {
            'trainset': self.trainset_controller.get_parameters(),
            'fitting': self.fitting_controller.get_parameters(),
            'classification': self.classification_controller.get_parameters(),
            'gisaxs_predict': self.gisaxs_predict_controller.get_parameters()
        }
    
    def load_parameters_from_file(self, file_path):
        """从文件加载参数"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                parameters = json.load(f)
            
            # 分发参数到各个主页面控制器
            if 'trainset' in parameters:
                self.trainset_controller.set_parameters(parameters['trainset'])
            if 'fitting' in parameters:
                self.fitting_controller.set_parameters(parameters['fitting'])
            if 'classification' in parameters:
                self.classification_controller.set_parameters(parameters['classification'])
            if 'gisaxs_predict' in parameters:
                self.gisaxs_predict_controller.set_parameters(parameters['gisaxs_predict'])
            
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
        
        # 验证四个主页面模块
        modules = [
            ('训练集参数', self.trainset_controller),
            ('拟合参数', self.fitting_controller),
            ('分类参数', self.classification_controller),
            ('GISAXS预测参数', self.gisaxs_predict_controller)
        ]
        
        for name, controller in modules:
            if hasattr(controller, 'validate_parameters'):
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
        self.trainset_controller.reset_to_defaults()
        self.fitting_controller.reset_to_defaults()
        self.classification_controller.reset_to_defaults()
        self.gisaxs_predict_controller.reset_to_defaults()
        
        self.status_updated.emit("所有参数已重置为默认值")
    
    def _on_classification_completed(self, results):
        """分类完成时的处理"""
        self.status_updated.emit(f"分类完成，处理了 {len(results)} 个项目")
        
        # 可以在这里添加结果显示逻辑
        # 例如：更新UI显示分类结果统计信息
    
    def _setup_button_styles(self):
        """设置导航按钮样式"""
        button_style = """
        QPushButton {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 12px 16px;
            color: #495057;
            margin: 2px;
        }
        
        QPushButton:hover {
            background-color: #e9ecef;
            border-color: #adb5bd;
            color: #212529;
        }
        
        QPushButton:pressed {
            background-color: #dee2e6;
            border-color: #6c757d;
            border-style: inset;
            color: #212529;
        }
        
        QPushButton:checked {
            background-color: #007bff;
            color: white;
            border-color: #0056b3;
        }
        
        QPushButton:disabled {
            background-color: #e9ecef;
            color: #6c757d;
            border-color: #dee2e6;
        }
        """
        
        # 应用样式到所有导航按钮
        navigation_buttons = [
            self.ui.cutAndFittingButton,
            self.ui.gisaxsPredictButton,
            self.ui.trainsetBuildButton,
            self.ui.ClassficationButton
        ]
        
        for button in navigation_buttons:
            if hasattr(button, 'setStyleSheet'):
                button.setStyleSheet(button_style)
                button.setCheckable(True)  # 使按钮可以保持选中状态
                
        # 设置按钮组管理
        self._setup_button_group()
    
    def _setup_button_group(self):
        """设置按钮组管理，确保只有一个按钮被选中"""
        from PyQt5.QtWidgets import QButtonGroup
        
        self.navigation_button_group = QButtonGroup()
        
        # 添加所有导航按钮到按钮组
        if hasattr(self.ui, 'cutAndFittingButton'):
            self.navigation_button_group.addButton(self.ui.cutAndFittingButton, 0)
        if hasattr(self.ui, 'gisaxsPredictButton'):
            self.navigation_button_group.addButton(self.ui.gisaxsPredictButton, 1)
        if hasattr(self.ui, 'trainsetBuildButton'):
            self.navigation_button_group.addButton(self.ui.trainsetBuildButton, 2)
        if hasattr(self.ui, 'ClassficationButton'):
            self.navigation_button_group.addButton(self.ui.ClassficationButton, 3)
        
        # 设置互斥选择
        self.navigation_button_group.setExclusive(True)
        
        # 连接按钮组信号
        self.navigation_button_group.buttonClicked.connect(self._on_navigation_button_clicked)
    
    def _on_navigation_button_clicked(self, button):
        """处理导航按钮点击"""
        button_id = self.navigation_button_group.id(button)
        
        # 根据按钮ID切换页面
        if button_id == 0:
            self._switch_to_cut_fitting()
        elif button_id == 1:
            self._switch_to_gisaxs_predict()
        elif button_id == 2:
            self._switch_to_trainset_build()
        elif button_id == 3:
            self._switch_to_classification()
    
    def _update_button_states(self, active_index):
        """更新按钮状态，确保只有当前页面的按钮被选中"""
        navigation_buttons = [
            self.ui.cutAndFittingButton,
            self.ui.gisaxsPredictButton,
            self.ui.trainsetBuildButton,
            self.ui.ClassficationButton
        ]
        
        for i, button in enumerate(navigation_buttons):
            if hasattr(button, 'setChecked'):
                button.setChecked(i == active_index)
