"""
样品参数控制器 - 管理GISAXS实验的样品参数
"""

import json
import os
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox


class SampleController(QObject):
    """样品参数控制器"""
    
    # 参数改变信号
    parameters_changed = pyqtSignal(str, dict)
    
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.parent = parent
        
        # 材料预设配置
        self.material_presets = self._load_material_presets()
        
        # 当前选择的粒子形状
        self.current_particle_shape = "Sphere"
        
        # 设置信号连接
        self._setup_connections()
    
    def _load_material_presets(self):
        """加载材料预设配置"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'materials.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载材料配置失败: {e}")
            return {}
    
    def _setup_connections(self):
        """设置信号连接"""
        # 粒子形状选择
        self.ui.particleShapeInitValue.currentTextChanged.connect(self._on_particle_shape_changed)
        
        # 球形粒子参数
        self._setup_sphere_connections()
        
        # 椭球粒子参数
        self._setup_ellipsoid_connections()
        
        # 基底参数
        self._setup_substrate_connections()
        
        # 干涉函数参数
        self._setup_interference_function_connections()
    
    def _setup_sphere_connections(self):
        """设置球形粒子参数连接"""
        # 材料预设
        self.ui.particleSphereCombox.currentTextChanged.connect(self._on_sphere_material_changed)
        
        # 参数变化
        sphere_inputs = [
            self.ui.particleSphereDeltaValue,
            self.ui.particleSphereBetaValue,
            self.ui.particleSphereRminValue,
            self.ui.particleSphereRmaxValue,
            self.ui.particleSphereSigmaMinRValue,
            self.ui.particleSphereSigmaMaxRValue,
            self.ui.particleSphereRbinsValue,
        ]
        
        for input_widget in sphere_inputs:
            if hasattr(input_widget, 'textChanged'):
                input_widget.textChanged.connect(self._emit_parameters_changed)
        
        # 下拉框变化
        sphere_combos = [
            self.ui.particleSphereDistTypeRCombox,
            self.ui.particleSpherePolydispersityCombox,
            self.ui.particleSphereDistTypeSigmaRCombox,
        ]
        
        for combo_widget in sphere_combos:
            if hasattr(combo_widget, 'currentTextChanged'):
                combo_widget.currentTextChanged.connect(self._emit_parameters_changed)
        
        # 显示按钮
        self.ui.particleSphereShowButton.clicked.connect(self._show_sphere_parameters)
    
    def _setup_ellipsoid_connections(self):
        """设置椭球粒子参数连接"""
        # 材料预设
        self.ui.particleEllipsoidMaterialsPresetCombox.currentTextChanged.connect(
            self._on_ellipsoid_material_changed
        )
        
        # 显示按钮
        self.ui.particleEllipsoidShowButton.clicked.connect(self._show_ellipsoid_parameters)
    
    def _setup_substrate_connections(self):
        """设置基底参数连接"""
        # 基底预设
        self.ui.substratePresetLabelCombox.currentTextChanged.connect(self._on_substrate_material_changed)
        
        # 粗糙度类型
        self.ui.substrateRoughnessLabelCombox.currentTextChanged.connect(self._on_roughness_type_changed)
        
        # 参数变化
        substrate_inputs = [
            self.ui.substrateDeltaValue,
            self.ui.substrateBetaValue,
            self.ui.substrateRoughnessSigmaValue,
            self.ui.substrateRoughnessHurstValue,
            self.ui.substrateRoughnessEpsilonValue,
        ]
        
        for input_widget in substrate_inputs:
            if hasattr(input_widget, 'textChanged'):
                input_widget.textChanged.connect(self._emit_parameters_changed)
    
    def _setup_interference_function_connections(self):
        """设置干涉函数参数连接"""
        # 干涉函数类型
        self.ui.InterferenceFunctionTypeComBox.currentTextChanged.connect(
            self._on_interference_function_type_changed
        )
        
        # 1D准晶参数
        interference_inputs = [
            self.ui.InterferenceFunction1DParacrystalDMinValue,
            self.ui.InterferenceFunction1DParacrystalDMaxValue,
            self.ui.InterferenceFunction1DParacrystalSigmaMinDValue,
            self.ui.InterferenceFunction1DParacrystalSigmaMaxDValue,
        ]
        
        for input_widget in interference_inputs:
            if hasattr(input_widget, 'textChanged'):
                input_widget.textChanged.connect(self._emit_parameters_changed)
        
        # 显示按钮
        self.ui.InterferenceFunction1DParacrystalShowButton.clicked.connect(
            self._show_interference_function_parameters
        )
    
    def initialize(self):
        """初始化样品参数"""
        self._populate_material_presets()
        self._update_particle_shape_visibility()
        self._emit_parameters_changed()
    
    def _populate_material_presets(self):
        """填充材料预设下拉框"""
        if self.material_presets:
            # 球形粒子材料
            self.ui.particleSphereCombox.clear()
            for material_name in self.material_presets.keys():
                self.ui.particleSphereCombox.addItem(material_name)
            self.ui.particleSphereCombox.addItem("User-defined")
            
            # 椭球粒子材料
            self.ui.particleEllipsoidMaterialsPresetCombox.clear()
            for material_name in self.material_presets.keys():
                self.ui.particleEllipsoidMaterialsPresetCombox.addItem(material_name)
            self.ui.particleEllipsoidMaterialsPresetCombox.addItem("User-defined")
            
            # 基底材料
            self.ui.substratePresetLabelCombox.clear()
            for material_name in self.material_presets.keys():
                self.ui.substratePresetLabelCombox.addItem(material_name)
            self.ui.substratePresetLabelCombox.addItem("User-defined")
    
    def _on_particle_shape_changed(self, shape):
        """粒子形状改变处理"""
        self.current_particle_shape = shape
        self._update_particle_shape_visibility()
        self._emit_parameters_changed()
    
    def _update_particle_shape_visibility(self):
        """更新粒子形状相关界面的可见性"""
        shape = self.current_particle_shape
        
        if shape == "Sphere":
            self.ui.sampleParametersParticleStackedWidget.setCurrentIndex(0)
        elif shape == "Ellipsoid":
            self.ui.sampleParametersParticleStackedWidget.setCurrentIndex(1)
        # 可以添加更多形状的处理
    
    def _on_sphere_material_changed(self, material_name):
        """球形粒子材料改变处理"""
        if material_name in self.material_presets:
            material_config = self.material_presets[material_name]
            self._apply_sphere_material_config(material_config)
        self._emit_parameters_changed()
    
    def _apply_sphere_material_config(self, config):
        """应用球形粒子材料配置"""
        if 'delta' in config:
            self.ui.particleSphereDeltaValue.setText(str(config['delta']))
        if 'beta' in config:
            self.ui.particleSphereBetaValue.setText(str(config['beta']))
    
    def _on_ellipsoid_material_changed(self, material_name):
        """椭球粒子材料改变处理"""
        if material_name in self.material_presets:
            material_config = self.material_presets[material_name]
            self._apply_ellipsoid_material_config(material_config)
        self._emit_parameters_changed()
    
    def _apply_ellipsoid_material_config(self, config):
        """应用椭球粒子材料配置"""
        if 'delta' in config:
            self.ui.particleEllipsoidDeltaValue.setText(str(config['delta']))
        if 'beta' in config:
            self.ui.particleEllipsoidBetaValue.setText(str(config['beta']))
    
    def _on_substrate_material_changed(self, material_name):
        """基底材料改变处理"""
        if material_name in self.material_presets:
            material_config = self.material_presets[material_name]
            self._apply_substrate_material_config(material_config)
        self._emit_parameters_changed()
    
    def _apply_substrate_material_config(self, config):
        """应用基底材料配置"""
        if 'delta' in config:
            self.ui.substrateDeltaValue.setText(str(config['delta']))
        if 'beta' in config:
            self.ui.substrateBetaValue.setText(str(config['beta']))
    
    def _on_roughness_type_changed(self, roughness_type):
        """粗糙度类型改变处理"""
        # 根据粗糙度类型启用/禁用相关参数
        enable_roughness = roughness_type != "None"
        
        self.ui.substrateRoughnessSigmaValue.setEnabled(enable_roughness)
        self.ui.substrateRoughnessHurstValue.setEnabled(enable_roughness)
        self.ui.substrateRoughnessEpsilonValue.setEnabled(enable_roughness)
        
        self._emit_parameters_changed()
    
    def _on_interference_function_type_changed(self, function_type):
        """干涉函数类型改变处理"""
        if function_type == "2D Paracrystal":
            self.ui.InterferenceFunctionStackedWidgets.setCurrentIndex(0)
        elif function_type == "None":
            self.ui.InterferenceFunctionStackedWidgets.setCurrentIndex(1)
        
        self._emit_parameters_changed()
    
    def get_parameters(self):
        """获取当前样品参数"""
        try:
            parameters = {
                'particle_shape': self.ui.particleShapeInitValue.currentText(),
                'sphere': self._get_sphere_parameters(),
                'ellipsoid': self._get_ellipsoid_parameters(),
                'substrate': self._get_substrate_parameters(),
                'interference_function': self._get_interference_function_parameters(),
            }
            return parameters
        except Exception as e:
            print(f"获取样品参数错误: {e}")
            return {}
    
    def _get_sphere_parameters(self):
        """获取球形粒子参数"""
        try:
            return {
                'material': self.ui.particleSphereCombox.currentText(),
                'delta': float(self.ui.particleSphereDeltaValue.text()),
                'beta': float(self.ui.particleSphereBetaValue.text()),
                'r_min': float(self.ui.particleSphereRminValue.text()),
                'r_max': float(self.ui.particleSphereRmaxValue.text()),
                'sigma_min_r': float(self.ui.particleSphereSigmaMinRValue.text()),
                'sigma_max_r': float(self.ui.particleSphereSigmaMaxRValue.text()),
                'r_bins': int(self.ui.particleSphereRbinsValue.text()),
                'dist_type_r': self.ui.particleSphereDistTypeRCombox.currentText(),
                'polydispersity': self.ui.particleSpherePolydispersityCombox.currentText(),
                'dist_type_sigma_r': self.ui.particleSphereDistTypeSigmaRCombox.currentText(),
            }
        except (ValueError, AttributeError):
            return {}
    
    def _get_ellipsoid_parameters(self):
        """获取椭球粒子参数"""
        try:
            return {
                'material': self.ui.particleEllipsoidMaterialsPresetCombox.currentText(),
                'delta': float(self.ui.particleEllipsoidDeltaValue.text()),
                'beta': float(self.ui.particleEllipsoidBetaValue.text()),
                # 添加更多椭球参数
            }
        except (ValueError, AttributeError):
            return {}
    
    def _get_substrate_parameters(self):
        """获取基底参数"""
        try:
            return {
                'material': self.ui.substratePresetLabelCombox.currentText(),
                'delta': float(self.ui.substrateDeltaValue.text()),
                'beta': float(self.ui.substrateBetaValue.text()),
                'roughness_type': self.ui.substrateRoughnessLabelCombox.currentText(),
                'roughness_sigma': float(self.ui.substrateRoughnessSigmaValue.text()),
                'roughness_hurst': float(self.ui.substrateRoughnessHurstValue.text()),
                'roughness_epsilon': float(self.ui.substrateRoughnessEpsilonValue.text()),
            }
        except (ValueError, AttributeError):
            return {}
    
    def _get_interference_function_parameters(self):
        """获取干涉函数参数"""
        try:
            return {
                'type': self.ui.InterferenceFunctionTypeComBox.currentText(),
                'd_min': float(self.ui.InterferenceFunction1DParacrystalDMinValue.text()),
                'd_max': float(self.ui.InterferenceFunction1DParacrystalDMaxValue.text()),
                'sigma_min_d': float(self.ui.InterferenceFunction1DParacrystalSigmaMinDValue.text()),
                'sigma_max_d': float(self.ui.InterferenceFunction1DParacrystalSigmaMaxDValue.text()),
                'dist_type_d': self.ui.InterferenceFunction1DParacrystalDistTypeDCombox.currentText(),
                'dist_type_sigma_d': self.ui.InterferenceFunction1DParacrystalDistTypeSigmaDCombox.currentText(),
            }
        except (ValueError, AttributeError):
            return {}
    
    def set_parameters(self, parameters):
        """设置样品参数"""
        if 'particle_shape' in parameters:
            shape_index = self.ui.particleShapeInitValue.findText(parameters['particle_shape'])
            if shape_index >= 0:
                self.ui.particleShapeInitValue.setCurrentIndex(shape_index)
        
        if 'sphere' in parameters:
            self._set_sphere_parameters(parameters['sphere'])
        
        if 'ellipsoid' in parameters:
            self._set_ellipsoid_parameters(parameters['ellipsoid'])
        
        if 'substrate' in parameters:
            self._set_substrate_parameters(parameters['substrate'])
        
        if 'interference_function' in parameters:
            self._set_interference_function_parameters(parameters['interference_function'])
    
    def _set_sphere_parameters(self, sphere_params):
        """设置球形粒子参数"""
        if 'delta' in sphere_params:
            self.ui.particleSphereDeltaValue.setText(str(sphere_params['delta']))
        if 'beta' in sphere_params:
            self.ui.particleSphereBetaValue.setText(str(sphere_params['beta']))
        # 设置更多参数...
    
    def _set_ellipsoid_parameters(self, ellipsoid_params):
        """设置椭球粒子参数"""
        if 'delta' in ellipsoid_params:
            self.ui.particleEllipsoidDeltaValue.setText(str(ellipsoid_params['delta']))
        if 'beta' in ellipsoid_params:
            self.ui.particleEllipsoidBetaValue.setText(str(ellipsoid_params['beta']))
        # 设置更多参数...
    
    def _set_substrate_parameters(self, substrate_params):
        """设置基底参数"""
        if 'delta' in substrate_params:
            self.ui.substrateDeltaValue.setText(str(substrate_params['delta']))
        if 'beta' in substrate_params:
            self.ui.substrateBetaValue.setText(str(substrate_params['beta']))
        # 设置更多参数...
    
    def _set_interference_function_parameters(self, if_params):
        """设置干涉函数参数"""
        if 'd_min' in if_params:
            self.ui.InterferenceFunction1DParacrystalDMinValue.setText(str(if_params['d_min']))
        # 设置更多参数...
    
    def validate_parameters(self):
        """验证样品参数"""
        try:
            params = self.get_parameters()
            
            # 验证粒子参数
            if params.get('particle_shape') == 'Sphere':
                sphere_params = params.get('sphere', {})
                r_min = sphere_params.get('r_min', 0)
                r_max = sphere_params.get('r_max', 0)
                if r_min >= r_max or r_min < 0:
                    return False, "球形粒子半径范围无效"
            
            # 验证基底参数
            substrate_params = params.get('substrate', {})
            delta = substrate_params.get('delta', 0)
            beta = substrate_params.get('beta', 0)
            if delta <= 0 or beta <= 0:
                return False, "基底光学常数无效"
            
            return True, "样品参数有效"
            
        except Exception as e:
            return False, f"参数验证错误: {str(e)}"
    
    def reset_to_defaults(self):
        """重置为默认参数"""
        # 重置为第一个选项
        self.ui.particleShapeInitValue.setCurrentIndex(0)
        self.ui.particleSphereCombox.setCurrentIndex(0)
        self.ui.particleEllipsoidMaterialsPresetCombox.setCurrentIndex(0)
        self.ui.substratePresetLabelCombox.setCurrentIndex(0)
        self.ui.InterferenceFunctionTypeComBox.setCurrentIndex(0)
    
    def _emit_parameters_changed(self):
        """发出参数改变信号"""
        parameters = self.get_parameters()
        self.parameters_changed.emit("样品参数", parameters)
    
    def _show_sphere_parameters(self):
        """显示球形粒子参数详情"""
        params = self._get_sphere_parameters()
        message = f"""球形粒子参数:
材料: {params.get('material', 'N/A')}
δ: {params.get('delta', 'N/A')}
β: {params.get('beta', 'N/A')}
半径范围: {params.get('r_min', 'N/A')} - {params.get('r_max', 'N/A')} nm
σ_R 范围: {params.get('sigma_min_r', 'N/A')} - {params.get('sigma_max_r', 'N/A')}
R bins: {params.get('r_bins', 'N/A')}"""
        
        # 获取主窗口作为父窗口
        main_window = self.parent.parent if hasattr(self.parent, 'parent') else None
        QMessageBox.information(main_window, "球形粒子参数", message)
    
    def _show_ellipsoid_parameters(self):
        """显示椭球粒子参数详情"""
        params = self._get_ellipsoid_parameters()
        message = f"""椭球粒子参数:
材料: {params.get('material', 'N/A')}
δ: {params.get('delta', 'N/A')}
β: {params.get('beta', 'N/A')}"""
        
        # 获取主窗口作为父窗口
        main_window = self.parent.parent if hasattr(self.parent, 'parent') else None
        QMessageBox.information(main_window, "椭球粒子参数", message)
    
    def _show_interference_function_parameters(self):
        """显示干涉函数参数详情"""
        params = self._get_interference_function_parameters()
        message = f"""干涉函数参数:
类型: {params.get('type', 'N/A')}
D 范围: {params.get('d_min', 'N/A')} - {params.get('d_max', 'N/A')} nm
σ_D 范围: {params.get('sigma_min_d', 'N/A')} - {params.get('sigma_max_d', 'N/A')}"""
        
        # 获取主窗口作为父窗口
        main_window = self.parent.parent if hasattr(self.parent, 'parent') else None
        QMessageBox.information(main_window, "干涉函数参数", message)
