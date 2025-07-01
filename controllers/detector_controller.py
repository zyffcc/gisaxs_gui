"""
探测器参数控制器 - 管理GISAXS实验的探测器参数
"""

import json
import os
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox


class DetectorController(QObject):
    """探测器参数控制器"""
    
    # 参数改变信号
    parameters_changed = pyqtSignal(str, dict)
    
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.parent = parent
        
        # 探测器预设配置
        self.detector_presets = self._load_detector_presets()
        
        # 默认参数
        self.default_parameters = {
            'preset': 'Pilatus 2M',
            'distance': 2000,  # mm
            'nbins_x': 1475,
            'nbins_y': 1475,
            'pixel_size_x': 172,  # μm
            'pixel_size_y': 172,  # μm
            'beam_center_x': 50,  # bin
            'beam_center_y': 50,  # bin
        }
        
        # 设置信号连接
        self._setup_connections()
    
    def _load_detector_presets(self):
        """加载探测器预设配置"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'detectors.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载探测器配置失败: {e}")
            return {}
    
    def _setup_connections(self):
        """设置信号连接"""
        # 探测器预设选择
        self.ui.detectorPresetCombox.currentTextChanged.connect(self._on_preset_changed)
        
        # 距离参数
        self.ui.distanceValue.textChanged.connect(self._on_distance_changed)
        
        # X轴参数
        self.ui.NbinsXValue.textChanged.connect(self._on_parameter_changed)
        self.ui.pixelSizeXValue.textChanged.connect(self._on_parameter_changed)
        self.ui.beamCenterXValue.textChanged.connect(self._on_parameter_changed)
        
        # Y轴参数
        self.ui.NbinsYValue.textChanged.connect(self._on_parameter_changed)
        self.ui.pixelSizeYValue.textChanged.connect(self._on_parameter_changed)
        self.ui.beamCenterYValue.textChanged.connect(self._on_parameter_changed)
    
    def initialize(self):
        """初始化探测器参数"""
        self._populate_detector_presets()
        self.set_parameters(self.default_parameters)
    
    def _populate_detector_presets(self):
        """填充探测器预设下拉框"""
        self.ui.detectorPresetCombox.clear()
        if self.detector_presets:
            for preset_name in self.detector_presets.keys():
                self.ui.detectorPresetCombox.addItem(preset_name)
        self.ui.detectorPresetCombox.addItem("User-defined")
    
    def _on_preset_changed(self, preset_name):
        """探测器预设改变处理"""
        if preset_name in self.detector_presets:
            preset_config = self.detector_presets[preset_name]
            self._apply_preset_config(preset_config)
        self._emit_parameters_changed()
    
    def _apply_preset_config(self, config):
        """应用预设配置"""
        if 'nbins_x' in config:
            self.ui.NbinsXValue.setText(str(config['nbins_x']))
        if 'nbins_y' in config:
            self.ui.NbinsYValue.setText(str(config['nbins_y']))
        if 'pixel_size_x' in config:
            self.ui.pixelSizeXValue.setText(str(config['pixel_size_x']))
        if 'pixel_size_y' in config:
            self.ui.pixelSizeYValue.setText(str(config['pixel_size_y']))
        if 'beam_center_x' in config:
            self.ui.beamCenterXValue.setText(str(config['beam_center_x']))
        if 'beam_center_y' in config:
            self.ui.beamCenterYValue.setText(str(config['beam_center_y']))
    
    def _on_distance_changed(self):
        """距离参数改变处理"""
        self._emit_parameters_changed()
    
    def _on_parameter_changed(self):
        """通用参数改变处理"""
        self._emit_parameters_changed()
    
    def get_parameters(self):
        """获取当前探测器参数"""
        try:
            parameters = {
                'preset': self.ui.detectorPresetCombox.currentText(),
                'distance': float(self.ui.distanceValue.text()),
                'nbins_x': int(self.ui.NbinsXValue.text()),
                'nbins_y': int(self.ui.NbinsYValue.text()),
                'pixel_size_x': float(self.ui.pixelSizeXValue.text()),
                'pixel_size_y': float(self.ui.pixelSizeYValue.text()),
                'beam_center_x': float(self.ui.beamCenterXValue.text()),
                'beam_center_y': float(self.ui.beamCenterYValue.text()),
            }
            return parameters
        except (ValueError, AttributeError):
            return self.default_parameters.copy()
    
    def set_parameters(self, parameters):
        """设置探测器参数"""
        if 'preset' in parameters:
            index = self.ui.detectorPresetCombox.findText(parameters['preset'])
            if index >= 0:
                self.ui.detectorPresetCombox.setCurrentIndex(index)
        
        if 'distance' in parameters:
            self.ui.distanceValue.setText(str(parameters['distance']))
        
        if 'nbins_x' in parameters:
            self.ui.NbinsXValue.setText(str(parameters['nbins_x']))
        if 'nbins_y' in parameters:
            self.ui.NbinsYValue.setText(str(parameters['nbins_y']))
        
        if 'pixel_size_x' in parameters:
            self.ui.pixelSizeXValue.setText(str(parameters['pixel_size_x']))
        if 'pixel_size_y' in parameters:
            self.ui.pixelSizeYValue.setText(str(parameters['pixel_size_y']))
        
        if 'beam_center_x' in parameters:
            self.ui.beamCenterXValue.setText(str(parameters['beam_center_x']))
        if 'beam_center_y' in parameters:
            self.ui.beamCenterYValue.setText(str(parameters['beam_center_y']))
        
        self._emit_parameters_changed()
    
    def validate_parameters(self):
        """验证探测器参数"""
        try:
            params = self.get_parameters()
            
            # 验证距离
            distance = params.get('distance', 0)
            if distance <= 0 or distance > 10000:
                return False, "探测器距离必须在0-10000 mm范围内"
            
            # 验证像素数量
            nbins_x = params.get('nbins_x', 0)
            nbins_y = params.get('nbins_y', 0)
            if nbins_x <= 0 or nbins_y <= 0:
                return False, "像素数量必须大于0"
            
            # 验证像素尺寸
            pixel_size_x = params.get('pixel_size_x', 0)
            pixel_size_y = params.get('pixel_size_y', 0)
            if pixel_size_x <= 0 or pixel_size_y <= 0:
                return False, "像素尺寸必须大于0"
            
            # 验证光束中心
            beam_center_x = params.get('beam_center_x', 0)
            beam_center_y = params.get('beam_center_y', 0)
            if not (0 <= beam_center_x <= nbins_x) or not (0 <= beam_center_y <= nbins_y):
                return False, "光束中心必须在探测器范围内"
            
            return True, "探测器参数有效"
            
        except Exception as e:
            return False, f"参数验证错误: {str(e)}"
    
    def reset_to_defaults(self):
        """重置为默认参数"""
        self.set_parameters(self.default_parameters)
    
    def _emit_parameters_changed(self):
        """发出参数改变信号"""
        parameters = self.get_parameters()
        self.parameters_changed.emit("探测器参数", parameters)
    
    def get_detector_geometry(self):
        """获取探测器几何信息"""
        try:
            params = self.get_parameters()
            
            # 计算探测器物理尺寸
            detector_width = params['nbins_x'] * params['pixel_size_x'] / 1000  # mm
            detector_height = params['nbins_y'] * params['pixel_size_y'] / 1000  # mm
            
            # 计算q空间范围
            distance = params['distance']
            max_q_x = 2 * np.pi * np.arctan(detector_width / 2 / distance)
            max_q_y = 2 * np.pi * np.arctan(detector_height / 2 / distance)
            
            return {
                'detector_width_mm': detector_width,
                'detector_height_mm': detector_height,
                'max_q_x': max_q_x,
                'max_q_y': max_q_y,
                'distance_mm': distance
            }
            
        except Exception:
            return None
    
    def calculate_pixel_to_q_conversion(self, wavelength_nm):
        """计算像素到q空间的转换因子"""
        try:
            params = self.get_parameters()
            geometry = self.get_detector_geometry()
            
            if geometry is None:
                return None
            
            # 计算转换因子
            pixel_size_x_mm = params['pixel_size_x'] / 1000
            pixel_size_y_mm = params['pixel_size_y'] / 1000
            distance_mm = params['distance']
            
            # q = 4π sin(θ) / λ, where θ = arctan(r/distance)
            q_per_pixel_x = 4 * np.pi / (wavelength_nm * 1e-9) * \
                           pixel_size_x_mm / (distance_mm * 1000)
            q_per_pixel_y = 4 * np.pi / (wavelength_nm * 1e-9) * \
                           pixel_size_y_mm / (distance_mm * 1000)
            
            return {
                'q_per_pixel_x': q_per_pixel_x,
                'q_per_pixel_y': q_per_pixel_y,
                'wavelength_nm': wavelength_nm
            }
            
        except Exception:
            return None
