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
        
        # 快速初始化基础组件
        self.current_parameters = {}
        
        # 创建控制器（但不立即初始化）
        self.trainset_controller = TrainsetController(ui, self)
        self.fitting_controller = FittingController(ui, self)
        self.classification_controller = ClassificationController(ui, self)
        self.gisaxs_predict_controller = GisaxsPredictController(ui, self)
        
        # 注册控制器到全局参数管理器
        self._register_controllers()
        
        # 设置界面连接
        self._setup_connections()
        
        # 设置按钮样式
        self._setup_button_styles()
        
        # 延迟初始化其他组件
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(200, self._delayed_controller_initialization)

    def _delayed_controller_initialization(self):
        """延迟初始化控制器和其他耗时组件"""
        try:
            print("Master Controller: Starting delayed initialization of controllers...")
            
            # 初始化界面
            self._initialize_ui()
            
            # 注册UI控件到全局参数系统（可能耗时）
            self._register_ui_controls()
            
            # 初始化控制器（可能耗时）
            self.trainset_controller.initialize()
            self.fitting_controller.initialize()
            self.classification_controller.initialize()
            self.gisaxs_predict_controller.initialize()
            
            # 统一精度配置已禁用，由各个controller单独管理
            # from PyQt5.QtCore import QTimer
            # QTimer.singleShot(500, self._configure_control_precision)
            # QTimer.singleShot(1500, self._force_precision_override)  # 再次强制覆盖
            
            # 延迟加载上次会话状态（避免启动时立即加载文件）
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(1000, self._load_last_session)
            
            print("Master Controller: Controllers delayed initialization complete")
            
        except Exception as e:
            print(f"Master Controller: Controllers delayed initialization failed: {e}")
    
    def _load_last_session(self):
        """加载上次会话状态（异步，不阻塞启动）"""
        try:
            # 获取fitting模块的上次会话数据
            fitting_session = global_params.get_parameter('fitting', 'last_session', {})
            
            if fitting_session:
                # 如果有上次打开的文件，通知fitting控制器恢复
                last_file = fitting_session.get('last_opened_file')
                if last_file and os.path.exists(last_file):
                    # Restore session asynchronously to ensure UI is fully initialized
                    print(f"Preparing to restore last session: {os.path.basename(last_file)}")
                    
                    # 延迟恢复，让用户先看到界面
                    from PyQt5.QtCore import QTimer
                    QTimer.singleShot(2000, lambda: self._restore_session_async(fitting_session))
                else:
                    print("Master Controller: Last session file does not exist, skipping restore")
            else:
                print("Master Controller: No last session data found")
        except Exception as e:
            print(f"Master Controller: Failed to load last session: {e}")
    
    def _restore_session_async(self, fitting_session):
        """异步恢复会话"""
        try:
            self.fitting_controller.restore_session(fitting_session)
            print("Master Controller: Last session restored")
        except Exception as e:
            print(f"Master Controller: Failed to restore session: {e}")

    def save_current_session(self):
        """保存当前会话状态"""
        try:
            # 保存fitting模块的会话状态
            fitting_session = self.fitting_controller.get_session_data()
            if fitting_session:
                global_params.set_parameter('fitting', 'last_session', fitting_session)
                global_params.save_user_parameters()
                print("Master Controller: Current session saved")
        except Exception as e:
            print(f"Master Controller: Failed to save current session: {e}")

    def save_session_on_close(self):
        """程序关闭时保存会话"""
        self.save_current_session()
        print("Master Controller: Session saved on close")
    
    def _register_controllers(self):
        """注册控制器到全局参数管理器"""
        try:
            # 注册主要的控制器到全局参数管理器
            global_params.register_controller('trainset', self.trainset_controller)
            global_params.register_controller('fitting', self.fitting_controller)
            global_params.register_controller('classification', self.classification_controller)
            global_params.register_controller('gisaxs_predict', self.gisaxs_predict_controller)
            print("Master Controller: Sub-controller registration complete")
        except Exception as e:
            print(f"Master Controller: Sub-controller registration failed: {e}")
    
    def _setup_connections(self):
        """设置信号连接"""
        # 主要按钮连接
        self.ui.trainsetBuildButton.clicked.connect(self._switch_to_trainset_build)
        self.ui.gisaxsPredictButton.clicked.connect(self._switch_to_gisaxs_predict)
        self.ui.cutAndFittingButton.clicked.connect(self._switch_to_cut_fitting)
        self.ui.ClassficationButton.clicked.connect(self._switch_to_classification)
        # 直接打开独立的WAXS窗口
        try:
            self.ui.WAXSButton.clicked.connect(self._open_waxs_standalone)
        except Exception:
            pass
        
        # 连接主页面控制器信号
        self.trainset_controller.parameters_changed.connect(self._on_parameters_changed)
        self.trainset_controller.generation_started.connect(
            lambda: self.status_updated.emit("Trainset generation started...")
        )
        self.trainset_controller.generation_finished.connect(
            lambda: self.status_updated.emit("Trainset generation completed!")
        )
        self.trainset_controller.progress_updated.connect(self.progress_updated)
        
        self.fitting_controller.parameters_changed.connect(
            lambda params: self._on_parameters_changed("Fitting parameters", params)
        )
        self.fitting_controller.status_updated.connect(self.status_updated)
        self.fitting_controller.progress_updated.connect(self.progress_updated)
        
        self.classification_controller.parameters_changed.connect(
            lambda params: self._on_parameters_changed("Classification parameters", params)
        )
        self.classification_controller.status_updated.connect(self.status_updated)
        self.classification_controller.progress_updated.connect(self.progress_updated)
        self.classification_controller.classification_completed.connect(self._on_classification_completed)
        
        self.gisaxs_predict_controller.parameters_changed.connect(
            lambda params: self._on_parameters_changed("GISAXS prediction parameters", params)
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
        self.status_updated.emit("GISAXS Toolkit ready")

    def _open_waxs_standalone(self):
        """直接打开独立的 WAXS/WAXS.py 窗口。"""
        try:
            from WAXS.WAXS import MainWindow as WAXSMainWindow
            self._waxs_window = WAXSMainWindow()
            self._waxs_window.show()
            self.status_updated.emit("WAXS standalone window opened")
        except Exception as e:
            QMessageBox.warning(self.parent if isinstance(self.parent, QMainWindow) else None,
                                "WAXS", f"Failed to open WAXS window: {e}")
            print(f"Failed to open WAXS standalone window: {e}")
    
    def _register_ui_controls(self):
        """自动注册UI控件到全局参数系统（优化版）"""
        print("=== Starting registration of UI controls to global parameter system ===")
        
        try:
            from PyQt5.QtWidgets import QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QSlider
            
            # 获取所有支持的控件类型
            supported_types = (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QSlider)
            registered_count = 0
            
            # 预定义关键控件名称，优先注册
            priority_controls = [
                'gisaxsInputImportButtonValue', 'gisaxsInputStackValue',
                'gisaxsInputCenterVerticalValue', 'gisaxsInputCenterParallelValue',
                'gisaxsInputCutLineVerticalValue', 'gisaxsInputCutLineParallelValue',
                'gisaxsInputVminValue', 'gisaxsInputVmaxValue'
            ]
            
            # 递归搜索UI中的所有控件（优化版）
            def register_widget_recursively(widget, prefix="", max_depth=6):
                nonlocal registered_count
                
                if max_depth <= 0:  # 限制递归深度
                    return
                
                # 检查当前控件
                if isinstance(widget, supported_types):
                    # 生成控件ID
                    object_name = getattr(widget, 'objectName', lambda: '')()
                    if object_name:
                        control_id = f"{prefix}{object_name}" if prefix else object_name
                        
                        # 注册控件
                        global_params.register_control(control_id, widget)
                        registered_count += 1
                        
                        # 只对优先控件输出日志
                        if object_name in priority_controls:
                            print(f"Registered key control: {control_id} ({type(widget).__name__})")
                
                # 递归检查子控件（优化：检查所有子控件但限制深度）
                if hasattr(widget, 'children'):
                    for child in widget.children():
                        if hasattr(child, 'metaObject'):  # 确保是QWidget
                            register_widget_recursively(child, prefix, max_depth - 1)
            
            # 从主窗口开始注册（增加深度）
            register_widget_recursively(self.ui, "", 6)
            
            print(f"UI controls registration completed, total {registered_count} controls registered")
            
            # 连接控件值变化信号到参数同步
            global_params.ui.control_value_changed.connect(self._on_control_value_changed)
            
        except Exception as e:
            print(f"Failed to register UI controls: {e}")
    
    def _on_control_value_changed(self, control_id: str, value):
        """当UI控件值发生变化时的回调"""
        # 可以在这里添加控件值变化的处理逻辑
        # 例如：自动保存到参数系统、验证输入等
        print(f"Control '{control_id}' value updated to: {value}")
    
    def _switch_to_cut_fitting(self):
        """切换到Cut Fitting页面"""
        self.ui.mainWindowWidget.setCurrentIndex(2)  # Cut Fitting是第2页(gisaxsFittingPage)
        self.status_updated.emit("Switched to Cut Fitting mode")
        
        # 更新按钮状态
        self._update_button_states(0)
        
        # 确保控制器已初始化
        if not self.fitting_controller._initialized:
            self.fitting_controller.initialize()
    
    def handle_window_close(self):
        """处理窗口关闭事件"""
        self.save_session_on_close()
    
    def _switch_to_gisaxs_predict(self):
        """切换到GISAXS预测页面"""
        self.ui.mainWindowWidget.setCurrentIndex(1)  # GISAXS Predict是第1页
        self.status_updated.emit("Switched to GISAXS prediction mode")
        
        # 更新按钮状态
        self._update_button_states(1)
        
        # 确保控制器已初始化
        if not self.gisaxs_predict_controller._initialized:
            self.gisaxs_predict_controller.initialize()
    
    def _switch_to_trainset_build(self):
        """切换到训练集构建页面"""
        self.ui.mainWindowWidget.setCurrentIndex(0)  # Trainset Build是第0页(trainsetBuildPage)
        self.status_updated.emit("Switched to Trainset Build mode")
        
        # 更新按钮状态
        self._update_button_states(2)
        
    def _switch_to_classification(self):
        """切换到Classification页面"""
        self.ui.mainWindowWidget.setCurrentIndex(3)  # Classification是第3页
        self.status_updated.emit("Switched to Classification mode")
        
        # 更新按钮状态
        self._update_button_states(3)
        
        # 确保控制器已初始化
        if not self.classification_controller._initialized:
            self.classification_controller.initialize()
    
    def _on_parameters_changed(self, module_name, parameters):
        """当参数发生改变时的处理"""
        self.current_parameters[module_name] = parameters
        self.status_updated.emit(f"{module_name} parameters updated")
    
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
            
            self.status_updated.emit(f"Parameters loaded from {file_path} successfully")
            return True
            
        except Exception as e:
            self.status_updated.emit(f"Failed to load parameters: {str(e)}")
            return False
    
    def save_parameters_to_file(self, file_path):
        """保存参数到文件"""
        try:
            parameters = self.get_all_parameters()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(parameters, f, indent=4, ensure_ascii=False)
            
            self.status_updated.emit(f"Parameters saved to {file_path} successfully")
            return True
            
        except Exception as e:
            self.status_updated.emit(f"Failed to save parameters: {str(e)}")
            return False
    
    def validate_all_parameters(self):
        """验证所有参数的有效性"""
        validation_results = []
        
        # 验证四个主页面模块
        modules = [
            ('Trainset parameters', self.trainset_controller),
            ('Fitting parameters', self.fitting_controller),
            ('Classification parameters', self.classification_controller),
            ('GISAXS prediction parameters', self.gisaxs_predict_controller)
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
                "Parameter Validation", 
                "All parameters are valid!"
            )
        else:
            error_messages = []
            for name, is_valid, message in results:
                if not is_valid:
                    error_messages.append(f"{name}: {message}")
            
            QMessageBox.warning(
                self.parent,
                "Parameter Validation Failed",
                "The following parameters have issues:\n\n" + "\n".join(error_messages)
            )
    
    def reset_all_parameters(self):
        """Reset all parameters to default values"""
        self.trainset_controller.reset_to_defaults()
        self.fitting_controller.reset_to_defaults()
        self.classification_controller.reset_to_defaults()
        self.gisaxs_predict_controller.reset_to_defaults()
        
        self.status_updated.emit("All parameters have been reset to default values")
    
    def _on_classification_completed(self, results):
        """Handle classification completion"""
        self.status_updated.emit(f"Classification completed, processed {len(results)} items")
        
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
            button.setChecked(i == active_index)
    
    def _configure_control_precision(self):
        """统一配置所有数值控件：无限范围，9位有效数字，去除尾随零"""
        print("=== UNIFIED PRECISION CONFIGURATION (FORCE OVERRIDE) ===")
        
        # 获取所有QDoubleSpinBox控件
        all_controls = []
        for attr_name in dir(self.ui):
            if attr_name.endswith('Value') and not attr_name.startswith('_'):
                try:
                    attr = getattr(self.ui, attr_name)
                    if hasattr(attr, 'setDecimals'):  # 是QDoubleSpinBox
                        all_controls.append(attr_name)
                except AttributeError:
                    pass
        
        print(f"Found {len(all_controls)} QDoubleSpinBox controls to configure")
        
        for control_name in all_controls:
            control = getattr(self.ui, control_name)
            
            # 强制统一设置：无限范围，9位精度，智能显示
            old_decimals = control.decimals()
            old_range = (control.minimum(), control.maximum())
            
            control.setRange(float('-inf'), float('inf'))  # 无限范围
            control.setDecimals(9)  # 最多9位有效数字
            control.setSingleStep(0.001)  # 统一步长
            
            # 特殊要求：某些控件需要非负值
            if ('Int' in control_name or 'R' in control_name or 'Sigma' in control_name or 
                'Distance' in control_name or 'Wavelength' in control_name):
                control.setMinimum(0.0)  # 强度、半径、标准差等不能为负
            
            # 强制设置智能显示格式（去除尾随零）
            self._setup_smart_decimal_display(control)
            
            # 强制刷新当前值的显示
            current_value = control.value()
            control.setValue(current_value + 0.000000001)  # 微小变化触发更新
            control.setValue(current_value)  # 恢复原值
            
            range_info = f"(-inf,inf)" if control.minimum() == float('-inf') else f"({control.minimum()},inf)"
            print(f"FORCED {control_name}: old_decimals={old_decimals} -> 9, old_range={old_range} -> {range_info}")
        
        print(f"=== SUCCESSFULLY FORCED {len(all_controls)} CONTROLS WITH UNIFIED SETTINGS ===")
    
    def _force_precision_override(self):
        """最终强制覆盖任何可能被其他代码修改的精度设置"""
        print("=== FINAL FORCE OVERRIDE FOR PRECISION SETTINGS ===")
        
        # 再次确保所有控件都有正确的设置
        for attr_name in dir(self.ui):
            if attr_name.endswith('Value') and not attr_name.startswith('_'):
                try:
                    control = getattr(self.ui, attr_name)
                    if hasattr(control, 'setDecimals'):
                        current_decimals = control.decimals()
                        if current_decimals != 9:
                            print(f"FORCE FIXING {attr_name}: {current_decimals} -> 9")
                            control.setDecimals(9)
                            self._setup_smart_decimal_display(control)
                            # 触发重新显示
                            value = control.value()
                            control.setValue(value + 0.000000001)
                            control.setValue(value)
                except Exception as e:
                    print(f"Error in force override for {attr_name}: {e}")
        
        print("=== FINAL FORCE OVERRIDE COMPLETED ===")
    
    def _setup_smart_decimal_display(self, control):
        """强制设置智能小数显示：9位有效数字，去除尾随零"""
        from PyQt5.QtWidgets import QDoubleSpinBox
        from PyQt5.QtCore import QLocale  
        import types
        
        if not isinstance(control, QDoubleSpinBox):
            return
            
        try:
            # 强制确保9位小数设置
            control.setDecimals(9)
            
            # 保存原始方法
            original_method = control.textFromValue
            
            def smart_text_from_value(value):
                """智能格式化：最多9位有效数字，去除尾随零"""
                try:
                    # 零值特殊处理
                    if abs(value) < 1e-15:  # 更严格的零值判断
                        return "0"
                    
                    # 整数特殊处理
                    if abs(value - round(value)) < 1e-10:
                        return str(int(round(value)))
                    
                    # 使用9位精度格式化并去除尾随零
                    formatted = f"{value:.9f}".rstrip('0').rstrip('.')
                    
                    # 确保不返回空字符串或只有小数点
                    if not formatted or formatted == '.':
                        return "0"
                        
                    return formatted
                    
                except Exception:
                    return f"{value:.9f}".rstrip('0').rstrip('.') or "0"
            
            # 强制替换textFromValue方法
            control.textFromValue = types.MethodType(lambda self, value: smart_text_from_value(value), control)
            
            # 强制设置英文区域（确保小数点格式）
            locale = QLocale(QLocale.English, QLocale.UnitedStates)
            control.setLocale(locale)
            
            # 强制禁用键盘跟踪以避免输入时格式混乱
            control.setKeyboardTracking(False)
            
        except Exception as e:
            print(f"Error setting up smart display: {e}")
