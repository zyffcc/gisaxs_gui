"""
全局参数管理器 - 提供整个软件的参数访问接口
单例模式设计，确保全局唯一的参数管理实例
"""

import json
import os
import atexit
from typing import Dict, Any, Optional
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtWidgets import QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QSlider
import threading


class UIControlManager(QObject):
    """UI控件管理器，负责注册、同步和访问UI控件的值"""
    control_value_changed = pyqtSignal(str, object)  # 控件值变化信号 (控件ID, 新值)

    def __init__(self):
        super().__init__()
        self._controls = {}  # 存储控件引用 {control_id: widget}
        self._control_values = {}  # 缓存控件的值 {control_id: value}
        self._signal_mapper = {
            QLineEdit: 'textChanged',
            QSpinBox: 'valueChanged',
            QDoubleSpinBox: 'valueChanged',
            QComboBox: 'currentTextChanged',
            QCheckBox: 'stateChanged',
            QSlider: 'valueChanged'
        }

    def register_control(self, control_id: str, widget: QObject):
        """注册一个UI控件，并连接其信号以自动同步值"""
        if control_id in self._controls:
            print(f"Warning: Control ID '{control_id}' is already registered. Overwriting.")
        
        self._controls[control_id] = widget
        
        # 连接信号
        widget_class = type(widget)
        signal_name = self._signal_mapper.get(widget_class)
        
        if signal_name:
            signal = getattr(widget, signal_name)
            signal.connect(lambda value, cid=control_id: self._on_control_value_changed(cid, value))
        else:
            print(f"Warning: No signal mapping found for widget type {widget_class.__name__}. Value will not be auto-synced.")

        # 初始化当前值
        self._update_cached_value(control_id)

    def _on_control_value_changed(self, control_id: str, value: Any):
        """当连接的控件信号触发时，更新缓存的值并发出通知信号"""
        self._control_values[control_id] = value
        self.control_value_changed.emit(control_id, value)
        # print(f"Control '{control_id}' value changed to: {value}") # for debugging

    def _update_cached_value(self, control_id: str):
        """根据控件类型获取并更新其当前值"""
        widget = self._controls.get(control_id)
        if not widget:
            return

        value = None
        if isinstance(widget, QLineEdit):
            value = widget.text()
        elif isinstance(widget, QComboBox):
            value = widget.currentText()
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider)):
            value = widget.value()
        elif isinstance(widget, QCheckBox):
            value = widget.isChecked()
        
        if value is not None:
            self._control_values[control_id] = value

    def get_control_value(self, control_id: str) -> Optional[Any]:
        """获取指定ID控件的当前值"""
        if control_id not in self._controls:
            print(f"Error: Control ID '{control_id}' not found.")
            return None
        
        # 确保获取的是最新值
        self._update_cached_value(control_id)
        return self._control_values.get(control_id)

    def set_control_value(self, control_id: str, value: Any):
        """设置指定ID控件的值"""
        widget = self._controls.get(control_id)
        if not widget:
            print(f"Error: Control ID '{control_id}' not found.")
            return

        # 断开信号避免循环触发
        signal_name = self._signal_mapper.get(type(widget))
        if signal_name:
            getattr(widget, signal_name).disconnect()

        if isinstance(widget, QLineEdit):
            widget.setText(str(value))
        elif isinstance(widget, QComboBox):
            widget.setCurrentText(str(value))
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider)):
            widget.setValue(value)
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        
        # 更新缓存并重新连接信号
        self._control_values[control_id] = value
        if signal_name:
            getattr(widget, signal_name).connect(lambda val, cid=control_id: self._on_control_value_changed(cid, val))

    def get_all_control_values(self) -> Dict[str, Any]:
        """获取所有已注册控件的当前值"""
        result = {}
        for control_id in self._controls:
            self._update_cached_value(control_id)
            result[control_id] = self._control_values.get(control_id)
        return result

    def get_registered_controls(self) -> Dict[str, str]:
        """获取所有已注册控件的信息"""
        return {control_id: type(widget).__name__ for control_id, widget in self._controls.items()}


class GlobalParameterManager(QObject):
    """全局参数管理器，使用单例模式"""
    
    _instance = None
    _lock = threading.Lock()
    
    # 信号定义
    parameters_updated = pyqtSignal(str, dict)  # 参数更新信号 (模块名, 参数字典)
    parameter_changed = pyqtSignal(str, str, object)  # 单个参数变化 (模块名, 参数名, 新值)
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # 防止重复初始化
        if self._initialized:
            return
            
        super().__init__()
        self._initialized = True
        
        # UI控件管理器
        self.ui = UIControlManager()
        
        # 参数存储
        self._parameters = {
            'beam': {},
            'detector': {},
            'sample': {},
            'preprocessing': {},
            'trainset': {},
            'fitting': {},
            'gisaxs_predict': {},
            'classification': {},
            'system': {}
        }
        
        # 控制器引用
        self._controllers = {}
        
        # 用户参数文件路径
        self.user_params_file = os.path.join('config', 'user_parameters.json')
        self.default_params_file = os.path.join('config', 'default_parameters.json')
        
        # 确保配置目录存在
        os.makedirs('config', exist_ok=True)
        
        # 初始化默认参数
        self._init_default_parameters()
        
        # 保存默认参数到文件（首次运行时）
        self._save_default_parameters()
        
        # 加载用户上次使用的参数
        self._load_user_parameters()
        
        # 设置自动保存
        self._setup_auto_save()
        
        # 注册退出时保存参数
        atexit.register(self._save_user_parameters)
    
    def _init_default_parameters(self):
        """初始化默认参数"""
        self._parameters.update({
            'beam': {
                'wavelength': 0.1,  # nm
                'grazing_angle': 0.4,  # degrees
                'beam_size_x': 0.1,  # mm
                'beam_size_y': 0.1,  # mm
                'flux': 1e12,  # photons/s
                'polarization': 'horizontal'
            },
            'detector': {
                'preset': 'Pilatus 2M',
                'distance': 2000,  # mm
                'nbins_x': 1475,
                'nbins_y': 1475,
                'pixel_size_x': 172,  # μm
                'pixel_size_y': 172,  # μm
                'beam_center_x': 737,  # bin
                'beam_center_y': 737,  # bin
                'exposure_time': 1.0,  # seconds
                'dark_current': 0.1,  # counts/pixel/s
                'readout_noise': 5.0  # electrons RMS
            },
            'sample': {
                'particle_shape': 'Sphere',
                'particle_size': 10.0,  # nm
                'size_distribution': 0.1,  # relative std
                'material': 'Gold',
                'substrate': 'Silicon',
                'thickness': 100.0,  # nm
                'roughness': 1.0,  # nm RMS
                'density': 0.5,  # particle density
                'orientation': 'random'
            },
            'preprocessing': {
                'focus_region': {
                    'type': 'q',
                    'qr_min': 0.01,
                    'qr_max': 3.0,
                    'qz_min': 0.01,
                    'qz_max': 3.0,
                },
                'noising': {
                    'type': 'Gaussian',
                    'snr_min': 80,
                    'snr_max': 130,
                },
                'others': {
                    'crop_edge': True,
                    'add_mask': True,
                    'normalize': True,
                    'logarization': True,
                }
            },
            'trainset': {
                'file_name': 'trainset',
                'save_path': '',
                'trainset_number': 1000,
                'save_every': 100,
                'batch_size': 10
            },
            'system': {
                'calculation_method': 'DWBA',
                'approximation': 'Born',
                'substrate_layers': 1,
                'max_iterations': 1000,
                'convergence_threshold': 1e-6,
                'parallel_processing': True,
                'num_threads': 4
            }
        })
    
    def register_controller(self, name: str, controller):
        """注册控制器"""
        self._controllers[name] = controller
        print(f"✓ 已注册控制器: {name}")
    
    def get_parameter(self, module: str, param_name: str, default=None):
        """获取单个参数
        
        Args:
            module: 模块名 ('beam', 'detector', 'sample', etc.)
            param_name: 参数名，支持嵌套访问 (如 'focus_region.type')
            default: 默认值
            
        Returns:
            参数值
        """
        try:
            if module not in self._parameters:
                return default
                
            # 支持嵌套参数访问
            keys = param_name.split('.')
            value = self._parameters[module]
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
                    
            return value
        except Exception:
            return default
    
    def set_parameter(self, module: str, param_name: str, value):
        """设置单个参数
        
        Args:
            module: 模块名
            param_name: 参数名，支持嵌套设置
            value: 参数值
        """
        try:
            if module not in self._parameters:
                self._parameters[module] = {}
            
            # 支持嵌套参数设置
            keys = param_name.split('.')
            target = self._parameters[module]
            
            # 导航到父级字典
            for key in keys[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]
            
            # 设置最终值
            old_value = target.get(keys[-1])
            target[keys[-1]] = value
            
            # 发出信号
            if old_value != value:
                self.parameter_changed.emit(module, param_name, value)
                
        except Exception as e:
            print(f"设置参数失败: {module}.{param_name} = {value}, 错误: {e}")
    
    def get_module_parameters(self, module: str) -> Dict[str, Any]:
        """获取整个模块的参数
        
        Args:
            module: 模块名
            
        Returns:
            模块参数字典的深拷贝
        """
        if module in self._parameters:
            return self._deep_copy_dict(self._parameters[module])
        return {}
    
    def set_module_parameters(self, module: str, parameters: Dict[str, Any]):
        """设置整个模块的参数
        
        Args:
            module: 模块名
            parameters: 参数字典
        """
        if module not in self._parameters:
            self._parameters[module] = {}
            
        self._parameters[module].update(parameters)
        self.parameters_updated.emit(module, self._deep_copy_dict(parameters))
    
    def get_all_parameters(self) -> Dict[str, Dict[str, Any]]:
        """获取所有参数的深拷贝"""
        return self._deep_copy_dict(self._parameters)
    
    def update_from_controller(self, module: str):
        """从对应的控制器更新参数
        
        Args:
            module: 模块名
        """
        if module in self._controllers:
            controller = self._controllers[module]
            if hasattr(controller, 'get_parameters'):
                try:
                    params = controller.get_parameters()
                    if isinstance(params, dict):
                        self.set_module_parameters(module, params)
                        print(f"✓ 已从 {module} 控制器更新参数")
                except Exception as e:
                    print(f"从 {module} 控制器获取参数失败: {e}")
    
    def sync_to_controller(self, module: str):
        """同步参数到对应的控制器
        
        Args:
            module: 模块名
        """
        if module in self._controllers:
            controller = self._controllers[module]
            if hasattr(controller, 'set_parameters'):
                try:
                    params = self.get_module_parameters(module)
                    controller.set_parameters(params)
                    print(f"✓ 已同步参数到 {module} 控制器")
                except Exception as e:
                    print(f"同步参数到 {module} 控制器失败: {e}")
    
    def sync_all_from_controllers(self):
        """从所有已注册的控制器同步参数"""
        for module in self._controllers:
            self.update_from_controller(module)
    
    def sync_all_to_controllers(self):
        """同步所有参数到控制器"""
        for module in self._controllers:
            self.sync_to_controller(module)
    
    def save_parameters(self, file_path: str):
        """保存所有参数到文件
        
        Args:
            file_path: 保存路径
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self._parameters, f, indent=4, ensure_ascii=False)
            print(f"✓ 参数已保存到: {file_path}")
        except Exception as e:
            print(f"保存参数失败: {e}")
    
    def load_parameters(self, file_path: str):
        """从文件加载参数
        
        Args:
            file_path: 文件路径
        """
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_params = json.load(f)
                
                # 更新参数，保留现有结构
                for module, params in loaded_params.items():
                    if module in self._parameters:
                        self._parameters[module].update(params)
                    else:
                        self._parameters[module] = params
                
                print(f"✓ 参数已从文件加载: {file_path}")
                
                # 同步到所有控制器
                self.sync_all_to_controllers()
            else:
                print(f"参数文件不存在: {file_path}")
        except Exception as e:
            print(f"加载参数失败: {e}")
    
    def get_physics_parameters(self) -> Dict[str, Any]:
        """获取物理计算所需的所有参数
        
        Returns:
            物理计算参数字典
        """
        return {
            'beam': self.get_module_parameters('beam'),
            'detector': self.get_module_parameters('detector'),
            'sample': self.get_module_parameters('sample'),
            'system': self.get_module_parameters('system')
        }
    
    def _deep_copy_dict(self, d):
        """深拷贝字典"""
        if isinstance(d, dict):
            return {k: self._deep_copy_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._deep_copy_dict(item) for item in d]
        else:
            return d
    
    def reset_to_defaults(self, module: Optional[str] = None):
        """重置参数到默认值
        
        Args:
            module: 模块名，如果为None则重置所有模块
        """
        if module:
            self._init_default_parameters()
            if module in self._parameters:
                # 重新初始化以获取默认值
                temp_manager = GlobalParameterManager.__new__(GlobalParameterManager)
                temp_manager._init_default_parameters()
                self._parameters[module] = temp_manager._parameters[module]
                self.parameters_updated.emit(module, self.get_module_parameters(module))
                print(f"✓ {module} 模块参数已重置为默认值")
        else:
            self._init_default_parameters()
            self.sync_all_to_controllers()
            print("✓ 所有参数已重置为默认值")
    
    def print_all_parameters(self):
        """打印所有参数（用于调试）"""
        print("\n=== 全局参数状态 ===")
        for module, params in self._parameters.items():
            print(f"\n[{module.upper()}]")
            self._print_dict(params, indent=2)
    
    def _print_dict(self, d, indent=0):
        """递归打印字典"""
        for key, value in d.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                self._print_dict(value, indent + 2)
            else:
                print(" " * indent + f"{key}: {value}")
    
    def _save_default_parameters(self):
        """保存默认参数到文件"""
        try:
            if not os.path.exists(self.default_params_file):
                with open(self.default_params_file, 'w', encoding='utf-8') as f:
                    json.dump(self._parameters, f, indent=4, ensure_ascii=False)
                print(f"✓ 默认参数已保存到: {self.default_params_file}")
        except Exception as e:
            print(f"保存默认参数失败: {e}")
    
    def _load_user_parameters(self):
        """加载用户上次使用的参数"""
        try:
            if os.path.exists(self.user_params_file):
                with open(self.user_params_file, 'r', encoding='utf-8') as f:
                    user_params = json.load(f)
                
                # 更新参数，保持结构完整性
                for module, params in user_params.items():
                    if module in self._parameters and isinstance(params, dict):
                        self._parameters[module].update(params)
                
                print(f"✓ 已加载用户参数: {self.user_params_file}")
            else:
                print("首次启动，使用默认参数")
        except Exception as e:
            print(f"加载用户参数失败，使用默认参数: {e}")
    
    def _save_user_parameters(self):
        """保存用户当前参数"""
        try:
            with open(self.user_params_file, 'w', encoding='utf-8') as f:
                json.dump(self._parameters, f, indent=4, ensure_ascii=False)
            print(f"✓ 用户参数已保存到: {self.user_params_file}")
        except Exception as e:
            print(f"保存用户参数失败: {e}")
    
    def _setup_auto_save(self):
        """设置自动保存定时器"""
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self._save_user_parameters)
        self.auto_save_timer.start(30000)  # 每30秒自动保存一次
        print("✓ 自动保存已启用（每30秒）")
    
    def reset_to_initial_parameters(self):
        """重置到初始默认参数"""
        try:
            if os.path.exists(self.default_params_file):
                with open(self.default_params_file, 'r', encoding='utf-8') as f:
                    default_params = json.load(f)
                
                self._parameters = default_params
                print("✓ 参数已重置为初始默认值")
            else:
                # 如果默认参数文件不存在，重新初始化
                self._init_default_parameters()
                print("✓ 参数已重置为内置默认值")
            
            # 同步到所有控制器
            self.sync_all_to_controllers()
            
            # 发出参数更新信号
            for module in self._parameters:
                self.parameters_updated.emit(module, self.get_module_parameters(module))
            
            # 立即保存重置后的参数
            self._save_user_parameters()
            
        except Exception as e:
            print(f"重置参数失败: {e}")
    
    def force_save_parameters(self):
        """强制保存当前参数"""
        self._save_user_parameters()
    
    # === UI控件管理便捷方法 ===
    
    def get_control_value(self, control_id: str) -> Optional[Any]:
        """获取UI控件的值 (便捷访问)"""
        return self.ui.get_control_value(control_id)

    def set_control_value(self, control_id: str, value: Any):
        """设置UI控件的值 (便捷访问)"""
        self.ui.set_control_value(control_id, value)

    def register_control(self, control_id: str, widget: QObject):
        """注册一个UI控件 (便捷访问)"""
        self.ui.register_control(control_id, widget)
    
    def get_all_control_values(self) -> Dict[str, Any]:
        """获取所有已注册控件的当前值"""
        return self.ui.get_all_control_values()
    
    def get_registered_controls(self) -> Dict[str, str]:
        """获取所有已注册控件的信息"""
        return self.ui.get_registered_controls()


# 创建全局实例
global_params = GlobalParameterManager()


# 便捷函数
def get_param(module: str, param_name: str, default=None):
    """便捷函数：获取参数"""
    return global_params.get_parameter(module, param_name, default)


def set_param(module: str, param_name: str, value):
    """便捷函数：设置参数"""
    global_params.set_parameter(module, param_name, value)


def get_all_params() -> Dict[str, Dict[str, Any]]:
    """便捷函数：获取所有参数"""
    return global_params.get_all_parameters()


def get_physics_params() -> Dict[str, Any]:
    """便捷函数：获取物理计算参数"""
    return global_params.get_physics_parameters()


def save_params(file_path: str):
    """便捷函数：保存参数"""
    global_params.save_parameters(file_path)


def load_params(file_path: str):
    """便捷函数：加载参数"""
    global_params.load_parameters(file_path)
