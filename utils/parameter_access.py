"""
参数访问工具模块 - 提供便捷的全局参数访问接口
可以在软件任何地方调用，获取整个软件的参数
"""

from typing import Dict, Any, Optional, Union, List
from core.global_params import global_params, get_param, set_param, get_all_params, get_physics_params


class ParameterAccessor:
    """参数访问器 - 提供更丰富的参数访问方法"""
    
    @staticmethod
    def get_software_parameters() -> Dict[str, Any]:
        """获取整个软件的所有参数
        
        Returns:
            包含所有模块参数的字典
        """
        return get_all_params()
    
    @staticmethod
    def get_physics_calculation_parameters() -> Dict[str, Any]:
        """获取物理计算所需的所有参数
        
        Returns:
            物理计算参数字典，包含beam, detector, sample, system
        """
        return get_physics_params()
    
    @staticmethod
    def get_beam_parameters() -> Dict[str, Any]:
        """获取光束参数"""
        return global_params.get_module_parameters('beam')
    
    @staticmethod
    def get_detector_parameters() -> Dict[str, Any]:
        """获取探测器参数"""
        return global_params.get_module_parameters('detector')
    
    @staticmethod
    def get_sample_parameters() -> Dict[str, Any]:
        """获取样品参数"""
        return global_params.get_module_parameters('sample')
    
    @staticmethod
    def get_preprocessing_parameters() -> Dict[str, Any]:
        """获取预处理参数"""
        return global_params.get_module_parameters('preprocessing')
    
    @staticmethod
    def get_trainset_parameters() -> Dict[str, Any]:
        """获取训练集参数"""
        return global_params.get_module_parameters('trainset')
    
    @staticmethod
    def get_fitting_parameters() -> Dict[str, Any]:
        """获取拟合参数"""
        return global_params.get_module_parameters('fitting')
    
    @staticmethod
    def get_classification_parameters() -> Dict[str, Any]:
        """获取分类参数"""
        return global_params.get_module_parameters('classification')
    
    @staticmethod
    def get_gisaxs_predict_parameters() -> Dict[str, Any]:
        """获取GISAXS预测参数"""
        return global_params.get_module_parameters('gisaxs_predict')
    
    @staticmethod
    def get_system_parameters() -> Dict[str, Any]:
        """获取系统参数"""
        return global_params.get_module_parameters('system')
    
    @staticmethod
    def get_parameter_by_path(path: str, default=None) -> Any:
        """通过路径获取参数
        
        Args:
            path: 参数路径，格式为 'module.param' 或 'module.nested.param'
            default: 默认值
            
        Returns:
            参数值
            
        Examples:
            get_parameter_by_path('beam.wavelength')
            get_parameter_by_path('preprocessing.focus_region.qr_min')
        """
        try:
            parts = path.split('.', 1)
            if len(parts) != 2:
                return default
            
            module, param_path = parts
            return get_param(module, param_path, default)
        except Exception:
            return default
    
    @staticmethod
    def set_parameter_by_path(path: str, value: Any) -> bool:
        """通过路径设置参数
        
        Args:
            path: 参数路径
            value: 参数值
            
        Returns:
            bool: 是否设置成功
        """
        try:
            parts = path.split('.', 1)
            if len(parts) != 2:
                return False
            
            module, param_path = parts
            set_param(module, param_path, value)
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_multiple_parameters(paths: List[str]) -> Dict[str, Any]:
        """批量获取多个参数
        
        Args:
            paths: 参数路径列表
            
        Returns:
            参数字典，键为路径，值为参数值
        """
        result = {}
        for path in paths:
            result[path] = ParameterAccessor.get_parameter_by_path(path)
        return result
    
    @staticmethod
    def get_scattering_geometry() -> Dict[str, Any]:
        """获取散射几何参数
        
        Returns:
            散射几何相关的所有参数
        """
        return {
            'wavelength': get_param('beam', 'wavelength'),
            'grazing_angle': get_param('beam', 'grazing_angle'),
            'detector_distance': get_param('detector', 'distance'),
            'beam_center_x': get_param('detector', 'beam_center_x'),
            'beam_center_y': get_param('detector', 'beam_center_y'),
            'pixel_size_x': get_param('detector', 'pixel_size_x'),
            'pixel_size_y': get_param('detector', 'pixel_size_y'),
            'nbins_x': get_param('detector', 'nbins_x'),
            'nbins_y': get_param('detector', 'nbins_y')
        }
    
    @staticmethod
    def get_sample_structure() -> Dict[str, Any]:
        """获取样品结构参数
        
        Returns:
            样品结构相关的所有参数
        """
        return {
            'particle_shape': get_param('sample', 'particle_shape'),
            'particle_size': get_param('sample', 'particle_size'),
            'size_distribution': get_param('sample', 'size_distribution'),
            'material': get_param('sample', 'material'),
            'substrate': get_param('sample', 'substrate'),
            'thickness': get_param('sample', 'thickness'),
            'roughness': get_param('sample', 'roughness'),
            'density': get_param('sample', 'density'),
            'orientation': get_param('sample', 'orientation')
        }
    
    @staticmethod
    def get_calculation_settings() -> Dict[str, Any]:
        """获取计算设置参数
        
        Returns:
            计算设置相关的所有参数
        """
        return {
            'calculation_method': get_param('system', 'calculation_method'),
            'approximation': get_param('system', 'approximation'),
            'substrate_layers': get_param('system', 'substrate_layers'),
            'max_iterations': get_param('system', 'max_iterations'),
            'convergence_threshold': get_param('system', 'convergence_threshold'),
            'parallel_processing': get_param('system', 'parallel_processing'),
            'num_threads': get_param('system', 'num_threads')
        }
    
    @staticmethod
    def export_parameters_for_physics(output_format='dict') -> Union[Dict, str]:
        """导出物理计算专用的参数格式
        
        Args:
            output_format: 输出格式 ('dict', 'json', 'flat')
            
        Returns:
            格式化的参数数据
        """
        physics_params = get_physics_params()
        
        if output_format == 'json':
            import json
            return json.dumps(physics_params, indent=2)
        elif output_format == 'flat':
            # 扁平化参数结构，方便传递给计算函数
            flat_params = {}
            for module, params in physics_params.items():
                for key, value in params.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            flat_params[f"{module}_{key}_{subkey}"] = subvalue
                    else:
                        flat_params[f"{module}_{key}"] = value
            return flat_params
        else:
            return physics_params
    
    @staticmethod
    def validate_physics_parameters() -> Dict[str, Any]:
        """验证物理参数的完整性和合理性
        
        Returns:
            验证结果字典
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'missing_parameters': []
        }
        
        # 检查必需的物理参数
        required_params = [
            ('beam', 'wavelength'),
            ('beam', 'grazing_angle'),
            ('detector', 'distance'),
            ('detector', 'pixel_size_x'),
            ('detector', 'pixel_size_y'),
            ('sample', 'particle_size'),
            ('sample', 'material')
        ]
        
        for module, param in required_params:
            value = get_param(module, param)
            if value is None:
                validation_result['missing_parameters'].append(f"{module}.{param}")
                validation_result['is_valid'] = False
        
        # 检查参数范围
        wavelength = get_param('beam', 'wavelength')
        if wavelength and (wavelength <= 0 or wavelength > 10):
            validation_result['warnings'].append("波长值异常")
        
        grazing_angle = get_param('beam', 'grazing_angle')
        if grazing_angle and (grazing_angle <= 0 or grazing_angle > 90):
            validation_result['errors'].append("掠射角度超出合理范围")
            validation_result['is_valid'] = False
        
        particle_size = get_param('sample', 'particle_size')
        if particle_size and particle_size <= 0:
            validation_result['errors'].append("粒子尺寸必须为正数")
            validation_result['is_valid'] = False
        
        return validation_result
    
    @staticmethod
    def sync_parameters_from_controllers():
        """从所有控制器同步参数到全局参数管理器"""
        global_params.sync_all_from_controllers()
    
    @staticmethod
    def sync_parameters_to_controllers():
        """同步全局参数到所有控制器"""
        global_params.sync_all_to_controllers()


# 创建全局访问器实例
params = ParameterAccessor()


# 便捷函数 - 在物理计算中最常用的函数
def get_all_software_params() -> Dict[str, Any]:
    """获取整个软件的所有参数 - 最常用的函数"""
    return params.get_software_parameters()


def get_physics_params_for_calculation() -> Dict[str, Any]:
    """获取物理计算所需参数 - 专用于物理计算"""
    return params.get_physics_calculation_parameters()


def get_param_by_path(path: str, default=None) -> Any:
    """通过路径获取单个参数 - 便捷访问"""
    return params.get_parameter_by_path(path, default)


def get_scattering_setup() -> Dict[str, Any]:
    """获取散射设置 - 物理计算常用"""
    return params.get_scattering_geometry()


def get_sample_info() -> Dict[str, Any]:
    """获取样品信息 - 物理计算常用"""
    return params.get_sample_structure()


def get_calc_settings() -> Dict[str, Any]:
    """获取计算设置 - 物理计算常用"""
    return params.get_calculation_settings()


def export_params_for_physics(format_type='dict'):
    """导出物理计算参数 - 传递给计算函数时使用"""
    return params.export_parameters_for_physics(format_type)


def validate_params_for_physics():
    """验证物理参数完整性"""
    return params.validate_physics_parameters()


# 快速访问常用参数的便捷函数
def get_wavelength():
    """获取波长"""
    return get_param('beam', 'wavelength')


def get_grazing_angle():
    """获取掠射角"""
    return get_param('beam', 'grazing_angle')


def get_detector_distance():
    """获取探测器距离"""
    return get_param('detector', 'distance')


def get_particle_size():
    """获取粒子尺寸"""
    return get_param('sample', 'particle_size')


def get_material():
    """获取材料"""
    return get_param('sample', 'material')


def get_substrate():
    """获取基底"""
    return get_param('sample', 'substrate')


def load_params(file_path: str):
    """便捷函数：加载参数"""
    global_params.load_parameters(file_path)


def reset_to_initial():
    """便捷函数：重置到初始参数"""
    global_params.reset_to_initial_parameters()


def force_save_params():
    """便捷函数：强制保存参数"""
    global_params.force_save_parameters()


# 新增便捷函数别名
def get_param_by_path(path: str, default=None) -> Any:
    """便捷函数：通过路径获取参数（别名）"""
    return ParameterAccessor.get_parameter_by_path(path, default)


def set_param_by_path(path: str, value: Any) -> bool:
    """便捷函数：通过路径设置参数（别名）"""
    return ParameterAccessor.set_parameter_by_path(path, value)


def get_multiple_params(paths: List[str]) -> Dict[str, Any]:
    """便捷函数：批量获取多个参数"""
    return ParameterAccessor.get_multiple_parameters(paths)


def get_scattering_setup() -> Dict[str, Any]:
    """便捷函数：获取散射几何设置"""
    return ParameterAccessor.get_scattering_geometry()


# 使用示例（可以删除）
if __name__ == "__main__":
    # 使用示例
    print("=== 参数访问示例 ===")
    
    # 1. 获取所有软件参数
    all_params = get_all_software_params()
    print(f"所有参数模块: {list(all_params.keys())}")
    
    # 2. 获取物理计算参数
    physics_params = get_physics_params_for_calculation()
    print(f"物理计算模块: {list(physics_params.keys())}")
    
    # 3. 通过路径获取参数
    wavelength = get_param_by_path('beam.wavelength')
    print(f"波长: {wavelength}")
    
    # 4. 获取散射设置
    geometry = get_scattering_setup()
    print(f"散射几何: {geometry}")
    
    # 5. 快速访问
    print(f"快速获取波长: {get_wavelength()}")
    print(f"快速获取粒子尺寸: {get_particle_size()}")
    
    # 6. 验证参数
    validation = validate_params_for_physics()
    print(f"参数验证: {validation}")
