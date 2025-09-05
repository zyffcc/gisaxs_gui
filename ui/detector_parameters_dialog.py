from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
                             QLabel, QDoubleSpinBox, QCheckBox, QPushButton, 
                             QGroupBox, QGridLayout)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont
import json
import os
from core.global_params import global_params


class DetectorParametersDialog(QDialog):
    """探测器参数对话框"""
    
    # 定义信号
    parameters_changed = pyqtSignal(dict)  # 参数改变时发出信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detector Parameters")
        self.setModal(True)
        self.resize(400, 350)
        
        # 参数字典
        self.parameters = {}
        
        # 初始化UI
        self._init_ui()
        
        # 加载当前参数值
        self._load_parameters()
        
        # 连接信号
        self._connect_signals()
        
    def _init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建主要参数组
        main_group = QGroupBox("Detector Parameters")
        main_layout = QFormLayout(main_group)
        
        # Distance (mm) - 步长0.1
        self.distance_spinbox = QDoubleSpinBox()
        self.distance_spinbox.setRange(100.0, 10000.0)
        self.distance_spinbox.setDecimals(1)
        self.distance_spinbox.setSingleStep(0.1)
        self.distance_spinbox.setValue(2000.0)
        main_layout.addRow("Distance (mm):", self.distance_spinbox)
        
        # Grazing Angle (deg) - 步长0.01
        self.angle_spinbox = QDoubleSpinBox()
        self.angle_spinbox.setRange(0.01, 10.0)
        self.angle_spinbox.setDecimals(3)
        self.angle_spinbox.setSingleStep(0.01)
        self.angle_spinbox.setValue(0.4)
        main_layout.addRow("Grazing Angle (deg):", self.angle_spinbox)
        
        # Wavelength (nm) - 步长0.001
        self.wavelength_spinbox = QDoubleSpinBox()
        self.wavelength_spinbox.setRange(0.001, 1.0)
        self.wavelength_spinbox.setDecimals(4)
        self.wavelength_spinbox.setSingleStep(0.001)
        self.wavelength_spinbox.setValue(0.015)
        main_layout.addRow("Wavelength (nm):", self.wavelength_spinbox)
        
        # Beam Center X (pixels) - 步长0.01，显示两位小数
        self.beam_center_x_spinbox = QDoubleSpinBox()
        self.beam_center_x_spinbox.setRange(0.0, 2000.0)
        self.beam_center_x_spinbox.setDecimals(2)
        self.beam_center_x_spinbox.setSingleStep(0.01)
        self.beam_center_x_spinbox.setValue(50.0)
        main_layout.addRow("Beam Center X (px):", self.beam_center_x_spinbox)
        
        # Beam Center Y (pixels) - 步长0.01，显示两位小数
        self.beam_center_y_spinbox = QDoubleSpinBox()
        self.beam_center_y_spinbox.setRange(0.0, 2000.0)
        self.beam_center_y_spinbox.setDecimals(2)
        self.beam_center_y_spinbox.setSingleStep(0.01)
        self.beam_center_y_spinbox.setValue(50.0)
        main_layout.addRow("Beam Center Y (px):", self.beam_center_y_spinbox)
        
        # Pixel Size X (μm) - 步长0.1
        self.pixel_size_x_spinbox = QDoubleSpinBox()
        self.pixel_size_x_spinbox.setRange(1.0, 1000.0)
        self.pixel_size_x_spinbox.setDecimals(1)
        self.pixel_size_x_spinbox.setSingleStep(0.1)
        self.pixel_size_x_spinbox.setValue(172.0)
        main_layout.addRow("Pixel Size X (μm):", self.pixel_size_x_spinbox)
        
        # Pixel Size Y (μm) - 步长0.1
        self.pixel_size_y_spinbox = QDoubleSpinBox()
        self.pixel_size_y_spinbox.setRange(1.0, 1000.0)
        self.pixel_size_y_spinbox.setDecimals(1)
        self.pixel_size_y_spinbox.setSingleStep(0.1)
        self.pixel_size_y_spinbox.setValue(172.0)
        main_layout.addRow("Pixel Size Y (μm):", self.pixel_size_y_spinbox)
        
        layout.addWidget(main_group)
        
        # 显示选项组
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        # Show in q axis 复选框
        self.show_q_axis_checkbox = QCheckBox("Show in q axis")
        # 不在这里设置默认值，稍后从配置中读取
        display_layout.addWidget(self.show_q_axis_checkbox)
        
        layout.addWidget(display_group)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        button_layout.addStretch()
        
        # 应用按钮
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._apply_parameters)
        button_layout.addWidget(self.apply_button)
        
        # 确定按钮
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self._ok_clicked)
        button_layout.addWidget(self.ok_button)
        
        # 取消按钮
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
    def _connect_signals(self):
        """连接信号"""
        # 当参数变化时自动保存
        self.distance_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.angle_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.wavelength_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.beam_center_x_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.beam_center_y_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.pixel_size_x_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.pixel_size_y_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.show_q_axis_checkbox.toggled.connect(self._on_parameter_changed)
        
    def _disconnect_signals(self):
        """断开信号连接"""
        try:
            self.distance_spinbox.valueChanged.disconnect()
            self.angle_spinbox.valueChanged.disconnect()
            self.wavelength_spinbox.valueChanged.disconnect()
            self.beam_center_x_spinbox.valueChanged.disconnect()
            self.beam_center_y_spinbox.valueChanged.disconnect()
            self.pixel_size_x_spinbox.valueChanged.disconnect()
            self.pixel_size_y_spinbox.valueChanged.disconnect()
            self.show_q_axis_checkbox.toggled.disconnect()
        except Exception:
            pass  # 忽略断开连接的错误
        
    def _load_parameters(self):
        """从配置文件加载参数"""
        try:
            # 临时断开信号连接以避免在加载时触发自动保存
            self._disconnect_signals()
            
            # 设置探测器参数 - 从Fitting模块专用参数读取
            beam_x = global_params.get_parameter('fitting', 'detector.beam_center_x', 50.0)
            beam_y = global_params.get_parameter('fitting', 'detector.beam_center_y', 50.0)
            
            self.distance_spinbox.setValue(global_params.get_parameter('fitting', 'detector.distance', 2000.0))
            self.beam_center_x_spinbox.setValue(beam_x)
            self.beam_center_y_spinbox.setValue(beam_y)
            self.pixel_size_x_spinbox.setValue(global_params.get_parameter('fitting', 'detector.pixel_size_x', 172.0))
            self.pixel_size_y_spinbox.setValue(global_params.get_parameter('fitting', 'detector.pixel_size_y', 172.0))
            
            # 设置束流参数
            self.angle_spinbox.setValue(global_params.get_parameter('beam', 'grazing_angle', 0.4))
            self.wavelength_spinbox.setValue(global_params.get_parameter('beam', 'wavelength', 0.015))
            
            # 设置显示选项 - 从fitting.detector中读取
            self.show_q_axis_checkbox.setChecked(global_params.get_parameter('fitting', 'detector.show_q_axis', False))
            
            # 重新连接信号
            self._connect_signals()
            
        except Exception as e:
            print(f"加载探测器参数失败: {e}")
            # 确保重新连接信号
            self._connect_signals()
            
    def _on_parameter_changed(self):
        """参数改变时的处理"""
        # 自动保存到global_params
        self._save_parameters()
        
        # 发出参数改变信号
        params = self._get_current_parameters()
        self.parameters_changed.emit(params)
        
    def _get_current_parameters(self):
        """获取当前参数值"""
        return {
            'distance': self.distance_spinbox.value(),
            'grazing_angle': self.angle_spinbox.value(),
            'wavelength': self.wavelength_spinbox.value(),
            'beam_center_x': self.beam_center_x_spinbox.value(),
            'beam_center_y': self.beam_center_y_spinbox.value(),
            'pixel_size_x': self.pixel_size_x_spinbox.value(),
            'pixel_size_y': self.pixel_size_y_spinbox.value(),
            'show_q_axis': self.show_q_axis_checkbox.isChecked()
        }
        
    def _save_parameters(self):
        """保存参数到配置文件"""
        try:
            params = self._get_current_parameters()
            
            print(f"Saving detector parameters:")
            print(f"  Beam center: ({params['beam_center_x']}, {params['beam_center_y']})")
            print(f"  Distance: {params['distance']}")
            print(f"  Show Q axis: {params['show_q_axis']}")
            
            # 保存到Fitting模块专用参数
            global_params.set_parameter('fitting', 'detector.distance', params['distance'])
            global_params.set_parameter('fitting', 'detector.beam_center_x', params['beam_center_x'])
            global_params.set_parameter('fitting', 'detector.beam_center_y', params['beam_center_y'])
            global_params.set_parameter('fitting', 'detector.pixel_size_x', params['pixel_size_x'])
            global_params.set_parameter('fitting', 'detector.pixel_size_y', params['pixel_size_y'])
            global_params.set_parameter('fitting', 'detector.show_q_axis', params['show_q_axis'])
            
            # 更新beam参数
            global_params.set_parameter('beam', 'grazing_angle', params['grazing_angle'])
            global_params.set_parameter('beam', 'wavelength', params['wavelength'])
                
            # 触发保存
            global_params.save_user_parameters()
            
            print("✓ Detector parameters saved successfully")
            
        except Exception as e:
            print(f"保存探测器参数失败: {e}")
            
    def _apply_parameters(self):
        """应用参数"""
        self._save_parameters()
        params = self._get_current_parameters()
        self.parameters_changed.emit(params)
        
    def _ok_clicked(self):
        """确定按钮点击"""
        self._apply_parameters()
        self.accept()
        
    def get_parameters(self):
        """获取参数（供外部调用）"""
        return self._get_current_parameters()
