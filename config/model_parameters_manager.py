"""
模型参数管理器
负责管理所有模型相关参数的加载、保存和访问
"""
import copy
import json
import os
from typing import Dict, Any, Optional
from PyQt5.QtCore import QObject, pyqtSignal


class ModelParametersManager(QObject):
    """模型参数管理器"""
    
    # 参数变更信号
    parameters_changed = pyqtSignal(str, dict)  # section, parameters
    
    def __init__(self, config_file: str = None):
        super().__init__()
        
        # 默认配置文件路径
        self.config_file = config_file or os.path.join(
            os.path.dirname(__file__), 'model_parameters.json'
        )
        
        # 参数存储
        self._parameters = {}
        
        # 加载参数
        self.load_parameters()
    
    def load_parameters(self) -> bool:
        """加载模型参数"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self._parameters = json.load(f)
                return True
            else:
                # 创建默认参数
                self._create_default_parameters()
                return self.save_parameters()
        except Exception as e:
            print(f"Failed to load model parameters: {e}")
            self._create_default_parameters()
            return False
    
    def save_parameters(self) -> bool:
        """保存模型参数"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._parameters, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Failed to save model parameters: {e}")
            return False
    
    def _create_default_parameters(self):
        """创建默认参数"""
        particles = {
            f"particle_{idx}": self._build_default_particle_entry(
                shape="Sphere" if idx == 1 else "None"
            )
            for idx in (1, 2, 3)
        }
        # particle_1 默认启用，其余保持禁用
        particles["particle_1"]["enabled"] = True

        self._parameters = {
            "fitting": {
                "global_parameters": {
                    "sigma_res": 0.1,
                    "k_value": 1.0
                },
                "particles": particles
            },
            "gisaxs_predict": {
                "particles": {}
            },
            "classification": {
                "particles": {}
            },
            "metadata": {
                "version": "1.0",
                "created": "2025-10-06",
                "description": "Model parameters configuration for GISAXS GUI application"
            }
        }
    
    def _build_default_shape_parameters(self) -> Dict[str, Dict[str, float]]:
        """生成形状默认参数，供粒子初始化与扩展使用"""
        return {
            "sphere": {
                "intensity": 1.0,
                "radius": 10.0,
                "sigma_radius": 0.1,
                "diameter": 20.0,
                "sigma_diameter": 0.1,
                "background": 0.0
            },
            "cylinder": {
                "intensity": 1.0,
                "radius": 10.0,
                "sigma_radius": 0.1,
                "height": 20.0,
                "sigma_height": 0.1,
                "diameter": 20.0,
                "sigma_diameter": 0.1,
                "background": 0.0
            }
        }

    def _build_default_particle_entry(self, shape: str = "None", parameters: Dict[str, Dict[str, float]] = None) -> Dict[str, Any]:
        """构造一个粒子的默认配置。"""
        return {
            "shape": shape,
            "enabled": shape != "None",
            "parameters": copy.deepcopy(parameters) if parameters else self._build_default_shape_parameters()
        }

    def get_parameter(self, section: str, key: str = None, default: Any = None) -> Any:
        """获取参数值"""
        try:
            if key is None:
                return self._parameters.get(section, default)
            
            section_data = self._parameters.get(section, {})
            if isinstance(section_data, dict):
                return section_data.get(key, default)
            else:
                return default
        except Exception:
            return default
    
    def set_parameter(self, section: str, key: str, value: Any) -> bool:
        """设置参数值"""
        try:
            if section not in self._parameters:
                self._parameters[section] = {}
            
            if not isinstance(self._parameters[section], dict):
                self._parameters[section] = {}
            
            self._parameters[section][key] = value
            
            # 发射信号
            self.parameters_changed.emit(section, self._parameters[section])
            
            return True
        except Exception as e:
            print(f"Failed to set parameter {section}.{key}: {e}")
            return False
    
    def get_particle_parameter(self, module: str, particle_id: str, shape: str = None, param: str = None) -> Any:
        """
        获取粒子参数
        
        Args:
            module: 模块名 ('fitting', 'gisaxs_predict', 'classification')
            particle_id: 粒子ID ('particle_1', 'particle_2', 'particle_3')
            shape: 形状名 ('sphere', 'cylinder'), 如果为None则返回粒子信息
            param: 参数名, 如果为None则返回整个形状参数
        """
        try:
            particles = self._parameters.get(module, {}).get('particles', {})
            particle = particles.get(particle_id, {})
            
            if shape is None:
                return particle
            
            shape_params = particle.get('parameters', {}).get(shape.lower(), {})
            
            if param is None:
                return shape_params
            
            return shape_params.get(param)
        except Exception:
            return None
    
    def set_particle_parameter(self, module: str, particle_id: str, shape: str, param: str, value: Any) -> bool:
        """设置粒子参数"""
        try:
            # 确保路径存在
            if module not in self._parameters:
                self._parameters[module] = {'particles': {}}
            if 'particles' not in self._parameters[module]:
                self._parameters[module]['particles'] = {}
            if particle_id not in self._parameters[module]['particles']:
                self._parameters[module]['particles'][particle_id] = self._build_default_particle_entry()
            if 'parameters' not in self._parameters[module]['particles'][particle_id]:
                self._parameters[module]['particles'][particle_id]['parameters'] = {}
            if shape.lower() not in self._parameters[module]['particles'][particle_id]['parameters']:
                self._parameters[module]['particles'][particle_id]['parameters'][shape.lower()] = {}
            
            # 设置参数值
            self._parameters[module]['particles'][particle_id]['parameters'][shape.lower()][param] = value
            
            # 发射信号
            self.parameters_changed.emit(f"{module}.particles", self._parameters[module]['particles'])
            
            return True
        except Exception as e:
            print(f"Failed to set particle parameter {module}.{particle_id}.{shape}.{param}: {e}")
            return False
    
    def set_particle_shape(self, module: str, particle_id: str, shape: str) -> bool:
        """设置粒子形状"""
        try:
            # 确保路径存在
            if module not in self._parameters:
                self._parameters[module] = {'particles': {}}
            if 'particles' not in self._parameters[module]:
                self._parameters[module]['particles'] = {}
            if particle_id not in self._parameters[module]['particles']:
                self._parameters[module]['particles'][particle_id] = {
                    'shape': shape,
                    'enabled': shape != 'None',
                    'parameters': {}
                }
            
            # 设置形状和启用状态
            self._parameters[module]['particles'][particle_id]['shape'] = shape
            self._parameters[module]['particles'][particle_id]['enabled'] = shape != 'None'
            
            # 发射信号
            self.parameters_changed.emit(f"{module}.particles", self._parameters[module]['particles'])
            
            return True
        except Exception as e:
            print(f"Failed to set particle shape {module}.{particle_id}: {e}")
            return False
    
    def get_particle_shape(self, module: str, particle_id: str) -> str:
        """获取粒子形状"""
        try:
            return self._parameters.get(module, {}).get('particles', {}).get(particle_id, {}).get('shape', 'None')
        except Exception:
            return 'None'
    
    def get_particle_enabled(self, module: str, particle_id: str) -> bool:
        """获取粒子启用状态"""
        try:
            return self._parameters.get(module, {}).get('particles', {}).get(particle_id, {}).get('enabled', False)
        except Exception:
            return False
    
    def set_particle_enabled(self, module: str, particle_id: str, enabled: bool) -> bool:
        """设置粒子启用状态"""
        try:
            if module not in self._parameters:
                self._parameters[module] = {'particles': {}}
            if 'particles' not in self._parameters[module]:
                self._parameters[module]['particles'] = {}
            if particle_id not in self._parameters[module]['particles']:
                self._parameters[module]['particles'][particle_id] = self._build_default_particle_entry()
            
            self._parameters[module]['particles'][particle_id]['enabled'] = enabled
            return True
        except Exception as e:
            print(f"Failed to set particle enabled state: {e}")
            return False

    def ensure_particle_entry(self, module: str, particle_id: str, shape: str = 'None') -> Dict[str, Any]:
        """确保指定粒子存在，不存在时使用默认值创建。"""
        if module not in self._parameters:
            self._parameters[module] = {'particles': {}, 'global_parameters': {}}
        particles = self._parameters[module].setdefault('particles', {})
        if particle_id not in particles:
            particles[particle_id] = self._build_default_particle_entry(shape=shape)
            self.parameters_changed.emit(f"{module}.particles", particles)
            self.save_parameters()
        return particles[particle_id]

    def add_particle(self, module: str, particle_id: str, shape: str = 'Sphere', parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """新增粒子配置，可指定初始形状与参数。"""
        entry = self._build_default_particle_entry(shape=shape, parameters=parameters)
        if module not in self._parameters:
            self._parameters[module] = {'particles': {}, 'global_parameters': {}}
        particles = self._parameters[module].setdefault('particles', {})
        particles[particle_id] = entry
        self.parameters_changed.emit(f"{module}.particles", particles)
        self.save_parameters()
        return entry

    def remove_particle(self, module: str, particle_id: str) -> bool:
        """删除指定粒子配置。"""
        try:
            particles = self._parameters.get(module, {}).get('particles', {})
            if particle_id in particles:
                particles.pop(particle_id)
                self.parameters_changed.emit(f"{module}.particles", particles)
                self.save_parameters()
                return True
            return False
        except Exception as e:
            print(f"Failed to remove particle {module}.{particle_id}: {e}")
            return False
    
    def get_all_particles(self, module: str) -> Dict[str, Any]:
        """获取模块的所有粒子配置"""
        return self._parameters.get(module, {}).get('particles', {})
    
    def get_global_parameter(self, module: str, param: str, default: Any = None) -> Any:
        """
        获取全局参数
        
        Args:
            module: 模块名 ('fitting', 'gisaxs_predict', 'classification')
            param: 参数名 ('sigma_res', 'k_value')
            default: 默认值
        """
        try:
            return self._parameters.get(module, {}).get('global_parameters', {}).get(param, default)
        except Exception:
            return default
    
    def set_global_parameter(self, module: str, param: str, value: Any) -> bool:
        """
        设置全局参数
        
        Args:
            module: 模块名 ('fitting', 'gisaxs_predict', 'classification')
            param: 参数名 ('sigma_res', 'k_value')
            value: 参数值
        """
        try:
            # 确保路径存在
            if module not in self._parameters:
                self._parameters[module] = {'global_parameters': {}, 'particles': {}}
            if 'global_parameters' not in self._parameters[module]:
                self._parameters[module]['global_parameters'] = {}
            
            # 设置参数值
            self._parameters[module]['global_parameters'][param] = value
            
            # 发射信号
            self.parameters_changed.emit(f"{module}.global_parameters", self._parameters[module]['global_parameters'])
            
            return True
        except Exception as e:
            print(f"Failed to set global parameter {module}.{param}: {e}")
            return False
    
    def get_all_global_parameters(self, module: str) -> Dict[str, Any]:
        """获取模块的所有全局参数"""
        return self._parameters.get(module, {}).get('global_parameters', {})
    
    def reset_to_defaults(self):
        """重置为默认参数"""
        self._create_default_parameters()
        return self.save_parameters()